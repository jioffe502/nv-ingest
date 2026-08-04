# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone VectorDB microservice backed by LanceDB.

Provides three endpoints:

- ``POST /internal/vectordb/write`` -- append embedding rows from ingest workers
- ``POST /v1/query``               -- embed query text and search the index
- ``GET  /v1/health``              -- liveness probe

Run with a remote NIM embed endpoint::

    python -m nemo_retriever.service.vectordb_app \\
        --lancedb-uri /data/vectordb \\
        --embed-endpoint http://nemo-retriever-nim-embed-0...:8000/v1/embeddings \\
        --port 7671

Run with in-pod Hugging Face query embedding (requires ``[local]`` extras + GPU)::

    python -m nemo_retriever.service.vectordb_app \\
        --lancedb-uri /data/vectordb \\
        --local-embed \\
        --local-embed-backend hf \\
        --embed-model nvidia/llama-nemotron-embed-vl-1b-v2 \\
        --port 7671
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Literal, Union, cast

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import lancedb
from nemo_retriever.common.remote_auth import resolve_remote_api_key
from nemo_retriever.common.vdb.lancedb_capabilities import (
    LanceRetrievalMode,
    LanceTableCapabilities,
    inspect_lancedb_table_object,
)
from nemo_retriever.ingest.index_mode import resolve_ingest_index_mode
from nemo_retriever.query.evidence import build_evidence_result
from nemo_retriever.service.query_schema import (
    EvidenceQueryResponse,
    EvidenceResult,
    QueryRequest,
    QueryResponse,
    QueryResult,
)

logger = logging.getLogger(__name__)

ServiceIndexMode = Literal["auto", "dense", "hybrid"]
_OPTIMIZE_WRITE_THRESHOLD = 20
_OPTIMIZE_ROW_THRESHOLD = 100_000


def _validate_service_index_mode(index_mode: str) -> ServiceIndexMode:
    normalized = index_mode.strip().lower()
    if normalized not in {"auto", "dense", "hybrid"}:
        raise ValueError(f"index_mode must be one of auto, dense, hybrid; got {index_mode!r}.")
    return cast(ServiceIndexMode, normalized)


# ── Request / response models ────────────────────────────────────────


class WriteRequest(BaseModel):
    rows: list[dict[str, Any]]


class WriteResponse(BaseModel):
    written: int
    total_rows: int


# ── Embedding helpers ────────────────────────────────────────────────


def _tensor_to_embedding_rows(tensor: Any) -> list[list[float]]:
    """Convert a local embedder tensor output to JSON-serializable vectors."""
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "tolist"):
        rows = tensor.tolist()
        if rows and isinstance(rows[0], (int, float)):
            return [rows]
        return rows
    return list(tensor)


def _embed_queries_remote(
    texts: list[str],
    *,
    embed_model: str,
    embed_endpoint: str,
    embed_api_key: str,
    embed_model_provider_prefix: str | None = None,
) -> list[list[float]]:
    from nemo_retriever.models.nim.util import infer_microservice

    return infer_microservice(
        texts,
        model_name=embed_model,
        model_provider_prefix=embed_model_provider_prefix,
        embedding_endpoint=embed_endpoint,
        nvidia_api_key=embed_api_key or None,
        input_type="query",
        grpc=False,
    )


def _strategies_for_retrieval_mode(mode: LanceRetrievalMode | str) -> list[str]:
    if mode == "hybrid":
        return ["hybrid"]
    return ["dense"]


# ── VectorDB state ───────────────────────────────────────────────────


class VectorDBState:
    """Thread-safe wrapper around a LanceDB connection."""

    def __init__(
        self,
        lancedb_uri: str,
        table_name: str,
        embed_endpoint: str,
        embed_model: str,
        embed_api_key: str,
        *,
        embed_model_provider_prefix: str | None = None,
        local_embed: bool = False,
        local_embed_backend: str = "hf",
        hf_cache_dir: str | None = None,
        device: str | None = None,
        gpu_memory_utilization: float = 0.45,
        index_mode: ServiceIndexMode = "auto",
    ) -> None:
        self.lancedb_uri = lancedb_uri
        self.table_name = table_name
        self.embed_endpoint = embed_endpoint
        self.embed_model = embed_model
        self.embed_model_provider_prefix = embed_model_provider_prefix
        self.embed_api_key = embed_api_key
        self.local_embed = local_embed
        self.local_embed_backend = local_embed_backend
        self.hf_cache_dir = hf_cache_dir
        self.device = device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.index_mode = _validate_service_index_mode(index_mode)
        self._write_lock = threading.Lock()
        self._embed_lock = threading.Lock()
        self._local_embedder: Any | None = None
        self._db = lancedb.connect(uri=lancedb_uri)
        self._table_exists = False
        self._effective_index_mode: Literal["dense", "hybrid"] = (
            "hybrid" if self.index_mode == "auto" else self.index_mode
        )
        self._writes_since_optimize = 0
        self._rows_since_optimize = 0
        self._last_optimization: dict[str, Any] = {
            "status": "never",
            "completed_at": None,
            "error": None,
        }
        if self.table_name in self._db.list_tables().tables:
            table = self._db.open_table(table_name)
            self._table_exists = True
            caps = inspect_lancedb_table_object(table)
            if caps.retrieval_mode not in {"dense", "hybrid"}:
                raise ValueError(
                    f"LanceDB table {table_name!r} at {lancedb_uri!r} must have a vector column; "
                    f"detected mode {caps.retrieval_mode!r}."
                )
            self._effective_index_mode = cast(
                Literal["dense", "hybrid"],
                resolve_ingest_index_mode(
                    self.index_mode,
                    overwrite=False,
                    existing_mode=cast(Literal["dense", "hybrid"], caps.retrieval_mode),
                ),
            )
            if self._effective_index_mode == "hybrid" and not caps.has_fts:
                self._create_and_wait_for_fts(table)
            logger.info("Opened existing LanceDB table '%s' at %s", table_name, lancedb_uri)
        else:
            logger.info("LanceDB table '%s' does not exist yet at %s", table_name, lancedb_uri)

    @property
    def embed_mode(self) -> str:
        if self.embed_endpoint:
            return "remote"
        if self.local_embed:
            return "local"
        return "none"

    @property
    def table_exists(self) -> bool:
        return self._table_exists

    def _table_capabilities(self):
        if not self._table_exists:
            return None
        table = self._db.open_table(self.table_name)
        return inspect_lancedb_table_object(table)

    @staticmethod
    def _is_fts_index(index: Any) -> bool:
        name = index.name.lower()
        return "text" in name or "fts" in name or str(getattr(index, "index_type", "")).lower() == "fts"

    @staticmethod
    def _create_and_wait_for_fts(table: Any) -> None:
        table.create_fts_index("text", replace=True)
        fts_names = [index.name for index in table.list_indices() if VectorDBState._is_fts_index(index)]
        if fts_names:
            table.wait_for_index(fts_names, timeout=timedelta(seconds=600))

    @staticmethod
    def _fts_unindexed_rows(table: Any) -> int | None:
        values: list[int] = []
        for index in table.list_indices():
            if not VectorDBState._is_fts_index(index):
                continue
            try:
                stats = table.index_stats(index.name)
            except Exception:
                logger.debug("Unable to read LanceDB index stats for %s", index.name, exc_info=True)
                continue
            value = (
                stats.get("num_unindexed_rows")
                if isinstance(stats, dict)
                else getattr(stats, "num_unindexed_rows", None)
            )
            if value is not None:
                values.append(int(value))
        return sum(values) if values else None

    def _optimize_if_due(self, table: Any) -> None:
        if (
            self._writes_since_optimize < _OPTIMIZE_WRITE_THRESHOLD
            and self._rows_since_optimize < _OPTIMIZE_ROW_THRESHOLD
        ):
            return
        try:
            table.optimize()
        except Exception as exc:
            self._last_optimization = {
                "status": "error",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
            logger.exception("LanceDB optimization failed for table '%s'", self.table_name)
            return
        self._writes_since_optimize = 0
        self._rows_since_optimize = 0
        self._last_optimization = {
            "status": "ok",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
        }
        logger.info("Optimized LanceDB table '%s' after incremental writes", self.table_name)

    def index_health(self) -> dict[str, Any]:
        health = {
            "configured_index_mode": self.index_mode,
            "effective_index_mode": None,
            "fts_present": False,
            "fts_unindexed_rows": None,
            "writes_since_optimize": self._writes_since_optimize,
            "rows_since_optimize": self._rows_since_optimize,
            "last_optimization": dict(self._last_optimization),
        }
        try:
            caps = self._table_capabilities()
            if caps is None:
                return health
            unindexed_rows = self._fts_unindexed_rows(self._db.open_table(self.table_name)) if caps.has_fts else None
        except Exception:
            health["effective_index_mode"] = "unknown"
            logger.warning(
                "Failed to inspect index health for table '%s' at %s; reporting unavailable index state",
                self.table_name,
                self.lancedb_uri,
                exc_info=True,
            )
            return health
        health.update(
            effective_index_mode=caps.retrieval_mode,
            fts_present=caps.has_fts,
            fts_unindexed_rows=unindexed_rows,
        )
        return health

    def resolve_effective_retrieval_mode(self, caps: LanceTableCapabilities | None = None) -> LanceRetrievalMode:
        """Resolve retrieval mode from table capabilities (auto).

        Optional ``caps`` avoids re-opening the table when the caller already has it.
        """
        if not self._table_exists:
            return self._effective_index_mode

        if caps is None:
            caps = self._table_capabilities()
            if caps is None:
                raise ValueError(
                    f"Unable to inspect LanceDB table {self.table_name!r} at {self.lancedb_uri!r}: "
                    "capabilities could not be determined."
                )

        mode: LanceRetrievalMode = caps.retrieval_mode
        if mode == "unknown":
            raise ValueError(
                f"LanceDB table {self.table_name!r} at {self.lancedb_uri!r} is not queryable: "
                "no vector column or FTS index was detected."
            )
        if mode == "sparse":
            raise ValueError(
                f"LanceDB table {self.table_name!r} at {self.lancedb_uri!r} has an FTS index but no vector "
                "column; sparse-only retrieval is not supported by the VectorDB service."
            )
        return mode

    def write_rows(self, rows: list[dict[str, Any]]) -> int:
        """Append rows to the LanceDB table (creates table on first write)."""
        if not rows:
            return 0

        from nemo_retriever.common.vdb.lancedb_schema import (
            create_or_append_lancedb_table,
            infer_vector_dim,
            lancedb_schema,
        )

        with self._write_lock:
            # Decide create-vs-append on raw on-disk presence via ``list_tables``
            # rather than interpreting ``open_table`` errors, so a transient I/O
            # failure cannot be misread as "table absent" and route execution
            # into the destructive ``overwrite=True`` create path.
            if not self._table_exists and self.table_name not in self._db.list_tables().tables:
                dim = infer_vector_dim(rows)
                if dim == 0:
                    logger.warning("Cannot infer vector dimension from rows; skipping write")
                    return 0
                schema = lancedb_schema(vector_dim=dim)
                table = create_or_append_lancedb_table(
                    self._db,
                    self.table_name,
                    rows,
                    schema,
                    overwrite=True,
                )
                self._table_exists = True
                if self._effective_index_mode == "hybrid":
                    self._create_and_wait_for_fts(table)
                self._writes_since_optimize = 0
                self._rows_since_optimize = 0
                logger.info(
                    "Created LanceDB table '%s' with %d rows (dim=%d)",
                    self.table_name,
                    len(rows),
                    dim,
                )
            else:
                table = self._db.open_table(self.table_name)
                table.add(rows)
                self._table_exists = True
                self._writes_since_optimize += 1
                self._rows_since_optimize += len(rows)
                if self._effective_index_mode == "hybrid":
                    self._optimize_if_due(table)
                logger.info("Appended %d rows to table '%s'", len(rows), self.table_name)

        return len(rows)

    def total_rows(self) -> int:
        if not self._table_exists:
            return 0
        try:
            table = self._db.open_table(self.table_name)
            return table.count_rows()
        except Exception:
            logger.warning(
                "Failed to count rows in LanceDB table '%s' at %s; reporting 0 to health",
                self.table_name,
                self.lancedb_uri,
                exc_info=True,
            )
            return 0

    def search(
        self,
        vectors: list[list[float]],
        query_texts: list[str],
        top_k: int,
    ) -> tuple[list[list[dict[str, Any]]], list[str]]:
        """Search the LanceDB table with precomputed query vectors."""
        if not self._table_exists:
            return [[] for _ in vectors], _strategies_for_retrieval_mode("dense")

        caps = self._table_capabilities()
        mode = self.resolve_effective_retrieval_mode(caps)
        strategies = _strategies_for_retrieval_mode(mode)

        from nemo_retriever.common.vdb.hybrid_fusion import DEFAULT_HYBRID_FUSION_POLICY
        from nemo_retriever.common.vdb.lancedb import LanceDB
        from nemo_retriever.common.vdb.records import normalize_retrieval_results

        hybrid = mode == "hybrid"
        vdb = LanceDB(uri=self.lancedb_uri, table_name=self.table_name, overwrite=False, hybrid=hybrid)
        retrieval_kwargs: dict[str, Any] = {"top_k": top_k, "hybrid": hybrid}
        if hybrid:
            retrieval_kwargs["query_texts"] = query_texts
            retrieval_kwargs["hybrid_fusion"] = DEFAULT_HYBRID_FUSION_POLICY

        if caps is not None and caps.vector_column and caps.vector_column != "vector":
            retrieval_kwargs["vector_column_name"] = caps.vector_column

        raw_results = vdb.retrieval(vectors, **retrieval_kwargs)
        return normalize_retrieval_results(raw_results), strategies

    def _get_local_embedder(self) -> Any:
        if self._local_embedder is None:
            from nemo_retriever.models import create_local_embedder

            self._local_embedder = create_local_embedder(
                self.embed_model,
                backend=self.local_embed_backend,
                device=self.device,
                hf_cache_dir=self.hf_cache_dir,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
            logger.info(
                "Loaded local query embedder model=%s backend=%s",
                self.embed_model,
                self.local_embed_backend,
            )
        return self._local_embedder

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed query texts via remote NIM or in-pod Hugging Face."""
        if self.embed_endpoint:
            return _embed_queries_remote(
                texts,
                embed_model=self.embed_model,
                embed_model_provider_prefix=self.embed_model_provider_prefix,
                embed_endpoint=self.embed_endpoint,
                embed_api_key=self.embed_api_key,
            )
        if self.local_embed:
            with self._embed_lock:
                embedder = self._get_local_embedder()
                tensor = embedder.embed_queries(texts)
            return _tensor_to_embedding_rows(tensor)
        raise RuntimeError("No embedding backend configured (remote endpoint or --local-embed).")


# ── FastAPI app ──────────────────────────────────────────────────────

_state: VectorDBState | None = None
_query_semaphore: asyncio.Semaphore | None = None

MAX_CONCURRENT_QUERIES = 4


def create_vectordb_app(
    lancedb_uri: str = "/data/vectordb",
    table_name: str = "nemo_retriever",
    embed_endpoint: str = "",
    embed_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2",
    embed_model_provider_prefix: str | None = None,
    embed_api_key: str = "",
    *,
    local_embed: bool = False,
    local_embed_backend: str = "hf",
    hf_cache_dir: str | None = None,
    device: str | None = None,
    gpu_memory_utilization: float = 0.45,
    index_mode: ServiceIndexMode = "auto",
) -> FastAPI:
    """Build the VectorDB FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        global _state, _query_semaphore
        _state = VectorDBState(
            lancedb_uri=lancedb_uri,
            table_name=table_name,
            embed_endpoint=embed_endpoint,
            embed_model=embed_model,
            embed_model_provider_prefix=embed_model_provider_prefix,
            embed_api_key=embed_api_key,
            local_embed=local_embed,
            local_embed_backend=local_embed_backend,
            hf_cache_dir=hf_cache_dir,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            index_mode=index_mode,
        )
        _query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
        logger.info(
            "VectorDB service started: uri=%s table=%s embed_mode=%s max_concurrent_queries=%d",
            lancedb_uri,
            table_name,
            _state.embed_mode,
            MAX_CONCURRENT_QUERIES,
        )
        if _state.embed_mode == "none":
            logger.error(
                "VectorDB started without an embedding backend; /v1/query will "
                "return HTTP 501 until --embed-endpoint or --local-embed is "
                "configured."
            )
        yield
        _state = None
        _query_semaphore = None
        logger.info("VectorDB service stopped")

    app = FastAPI(
        title="NeMo Retriever VectorDB",
        description="LanceDB-backed vector storage and retrieval",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/v1/health", tags=["system"])
    async def health() -> dict[str, Any]:
        rows = _state.total_rows() if _state else 0
        index_health = (
            _state.index_health()
            if _state
            else {
                "configured_index_mode": index_mode,
                "effective_index_mode": None,
                "fts_present": False,
                "fts_unindexed_rows": None,
                "writes_since_optimize": 0,
                "rows_since_optimize": 0,
                "last_optimization": {"status": "never", "completed_at": None, "error": None},
            }
        )
        return {
            "status": "ok",
            "table": table_name,
            "total_rows": rows,
            "table_exists": _state.table_exists if _state else False,
            "embed_mode": _state.embed_mode if _state else "none",
            "effective_retrieval_mode": index_health["effective_index_mode"],
            **index_health,
        }

    @app.post("/internal/vectordb/write", response_model=WriteResponse, tags=["internal"])
    async def write(req: WriteRequest) -> WriteResponse:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        written = await asyncio.to_thread(_state.write_rows, req.rows)
        return WriteResponse(written=written, total_rows=_state.total_rows())

    @app.post("/v1/query", response_model=Union[QueryResponse, EvidenceQueryResponse], tags=["query"])
    async def query(req: QueryRequest) -> QueryResponse | EvidenceQueryResponse:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")

        if _state.embed_mode == "none":
            raise HTTPException(
                501,
                "No embedding backend configured. Set --embed-endpoint for a remote "
                "NIM or --local-embed for in-pod Hugging Face query embedding.",
            )

        if not _state.table_exists:
            raise HTTPException(
                status_code=422,
                detail="No data has been ingested yet. Ingest documents first, then query.",
            )

        queries = req.query if isinstance(req.query, list) else [req.query]
        if not queries:
            if req.format == "evidence":
                return EvidenceQueryResponse(results=[])
            return QueryResponse(results=[])

        try:
            async with _query_semaphore:
                vectors = await asyncio.to_thread(_state.embed_queries, queries)
                hits_per_query, strategies = await asyncio.to_thread(
                    _state.search,
                    vectors,
                    queries,
                    req.top_k,
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if req.format == "evidence":
            return EvidenceQueryResponse(
                results=[EvidenceResult(**build_evidence_result(hits, strategies)) for hits in hits_per_query]
            )

        return QueryResponse(results=[QueryResult(hits=hits) for hits in hits_per_query])

    return app


# ── CLI entry point ──────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="NeMo Retriever VectorDB service")
    parser.add_argument("--lancedb-uri", default="/data/vectordb", help="LanceDB directory")
    parser.add_argument("--table-name", default="nemo_retriever", help="LanceDB table name")
    parser.add_argument(
        "--index-mode",
        default="auto",
        choices=("auto", "dense", "hybrid"),
        help="Fresh-table index mode; auto creates hybrid and preserves existing table capabilities.",
    )
    parser.add_argument("--embed-endpoint", default="", help="Remote NIM/OpenAI-compatible embed URL")
    parser.add_argument("--embed-model", default="nvidia/llama-nemotron-embed-vl-1b-v2")
    parser.add_argument("--embed-model-provider-prefix", default="", help="Optional LiteLLM provider prefix")
    parser.add_argument(
        "--embed-api-key",
        default="",
        help="Remote embedding API key (defaults to NVIDIA_API_KEY, then NGC_API_KEY).",
    )
    parser.add_argument(
        "--local-embed",
        action="store_true",
        help="Load Hugging Face embedder in-pod for /v1/query (requires [local] extras + GPU).",
    )
    parser.add_argument(
        "--local-embed-backend",
        default="hf",
        choices=("hf", "vllm"),
        help="Backend for --local-embed (default: hf).",
    )
    parser.add_argument("--hf-cache-dir", default="", help="Hugging Face model cache directory")
    parser.add_argument("--device", default="", help="Torch device for --local-embed (e.g. cuda:0)")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.45,
        help="vLLM GPU memory fraction when --local-embed-backend=vllm.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7671)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    if args.embed_endpoint and args.local_embed:
        parser.error("Use either --embed-endpoint or --local-embed, not both.")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app = create_vectordb_app(
        lancedb_uri=args.lancedb_uri,
        table_name=args.table_name,
        embed_endpoint=args.embed_endpoint,
        embed_model=args.embed_model,
        embed_model_provider_prefix=args.embed_model_provider_prefix or None,
        embed_api_key=resolve_remote_api_key(args.embed_api_key) or "",
        local_embed=args.local_embed,
        local_embed_backend=args.local_embed_backend,
        hf_cache_dir=args.hf_cache_dir or None,
        device=args.device or None,
        gpu_memory_utilization=args.gpu_memory_utilization,
        index_mode=args.index_mode,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
