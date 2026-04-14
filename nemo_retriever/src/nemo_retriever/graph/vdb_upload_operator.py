# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator for streaming VDB uploads during pipeline execution.

Wraps existing client VDB classes (``nv_ingest_client.util.vdb``) so that
any backend implementing the client :class:`VDB` ABC can be used as a
pipeline sink.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Sequence

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator

logger = logging.getLogger(__name__)


def _canonical_to_nvingest(rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Convert canonical VDB records to NV-Ingest pipeline format.

    The client Milvus implementation expects ``list[list[dict]]`` where each
    inner dict has ``document_type`` and a nested ``metadata`` dict with keys
    ``embedding``, ``content``, ``content_metadata``, ``source_metadata``.
    """
    elements: List[Dict[str, Any]] = []
    for row in rows:
        meta_str = row.get("metadata", "{}")
        source_str = row.get("source", "{}")
        try:
            content_metadata = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
        except (json.JSONDecodeError, TypeError):
            content_metadata = {}
        try:
            source_metadata = json.loads(source_str) if isinstance(source_str, str) else source_str
        except (json.JSONDecodeError, TypeError):
            source_metadata = {}

        elements.append(
            {
                "document_type": "text",
                "metadata": {
                    "content": row.get("text", ""),
                    "embedding": row.get("vector"),
                    "content_metadata": content_metadata,
                    "source_metadata": source_metadata,
                },
            }
        )
    return [elements] if elements else []


class VDBUploadOperator(AbstractOperator, CPUOperator):
    """Write pipeline embeddings to a vector store as data flows through the graph.

    Wraps a client VDB instance (LanceDB, Milvus, etc.) from
    ``nv_ingest_client.util.vdb``.  ``preprocess`` extracts canonical
    records from the DataFrame (backend-agnostic); ``process`` converts
    to the target format and writes (backend-specific).

    The DataFrame is passed through unchanged — this is a side-effect
    operator.

    **Concurrency**: This operator must run with ``concurrency=1`` in batch
    mode.  The single actor creates the table on its first write (respecting
    ``overwrite``) and appends on subsequent writes.  Index creation happens
    post-pipeline via the client VDB's ``write_to_index`` called from the
    driver.
    """

    def __init__(
        self,
        *,
        params: Any = None,
    ) -> None:
        super().__init__()
        from nemo_retriever.params.models import LanceDbParams, VdbUploadParams

        # Store as self.params so get_constructor_kwargs() can capture it
        # for deferred reconstruction on Ray workers.
        self.params = params

        if isinstance(params, VdbUploadParams):
            self._vdb_params = params
            self._lance_params = params.lancedb
        elif isinstance(params, LanceDbParams):
            self._vdb_params = None
            self._lance_params = params
        else:
            self._vdb_params = VdbUploadParams()
            self._lance_params = self._vdb_params.lancedb

        self._backend_name: str = getattr(self._vdb_params, "backend", "lancedb") if self._vdb_params else "lancedb"
        self._client_vdb: Any = None
        self._table: Any = None
        self._pending_records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Client VDB construction
    # ------------------------------------------------------------------

    def _create_client_vdb(self) -> Any:
        """Lazily construct the client VDB instance."""
        from nv_ingest_client.util.vdb import get_vdb_op_cls

        if self._backend_name == "lancedb":
            LanceDB = get_vdb_op_cls("lancedb")
            return LanceDB(
                uri=self._lance_params.lancedb_uri,
                table_name=self._lance_params.table_name,
                overwrite=self._lance_params.overwrite,
                index_type=self._lance_params.index_type,
                metric=self._lance_params.metric,
                num_partitions=self._lance_params.num_partitions,
                num_sub_vectors=self._lance_params.num_sub_vectors,
                hybrid=self._lance_params.hybrid,
                fts_language=self._lance_params.fts_language,
            )

        kwargs = getattr(self._vdb_params, "client_vdb_kwargs", {}) or {}
        return get_vdb_op_cls(self._backend_name)(**kwargs)

    # ------------------------------------------------------------------
    # Operator lifecycle
    # ------------------------------------------------------------------

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        """Extract canonical VDB records from the DataFrame (backend-agnostic)."""
        import pandas as pd

        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        if not isinstance(data, pd.DataFrame) or data.empty:
            self._pending_records = []
            return data

        self._pending_records = build_vdb_records(
            data,
            embedding_column=self._lance_params.embedding_column,
            embedding_key=self._lance_params.embedding_key,
            include_text=self._lance_params.include_text,
            text_column=self._lance_params.text_column,
        )
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        """Write pending records to the backend (backend-specific)."""
        if not self._pending_records:
            return data

        if self._client_vdb is None:
            self._client_vdb = self._create_client_vdb()

        if self._backend_name == "lancedb":
            self._write_lancedb_batch(self._pending_records)
        else:
            self._write_via_client(self._pending_records)

        logger.debug("VDBUploadOperator: wrote %d records", len(self._pending_records))
        self._pending_records = []
        return data

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    # ------------------------------------------------------------------
    # LanceDB streaming write path
    # ------------------------------------------------------------------

    def _write_lancedb_batch(self, records: List[Dict[str, Any]]) -> None:
        """Stream records to LanceDB via table.add().

        The client LanceDB class is used for config (uri, table_name,
        overwrite) and post-pipeline index creation, but its
        ``create_index()`` writes all records at once with no append
        support.  For streaming writes we call the lancedb library
        directly.
        """
        if self._table is None:
            import lancedb

            from nemo_retriever.vector_store.lancedb_utils import infer_vector_dim, lancedb_schema

            dim = infer_vector_dim(records)
            schema = lancedb_schema(vector_dim=dim)
            mode = "overwrite" if self._client_vdb.overwrite else "create"
            db = lancedb.connect(uri=self._client_vdb.uri)
            self._table = db.create_table(
                self._client_vdb.table_name,
                schema=schema,
                mode=mode,
            )

        self._table.add(records)

    # ------------------------------------------------------------------
    # Non-LanceDB write path (Milvus, OpenSearch, etc.)
    # ------------------------------------------------------------------

    def _write_via_client(self, records: List[Dict[str, Any]]) -> None:
        """Convert canonical records to NV-Ingest format and delegate to the client VDB.

        Client VDB classes split their config into connection params
        (for ``create_index``) and write params (for ``write_to_index``)
        via ``get_connection_params()`` / ``get_write_params()``.  We
        mirror the same dispatch that ``Milvus.run()`` uses.
        """
        nvingest_records = _canonical_to_nvingest(records)
        if not nvingest_records:
            return

        if self._table is None:
            collection_name, create_params = self._client_vdb.get_connection_params()
            self._client_vdb.create_index(collection_name=collection_name, **create_params)
            self._table = True  # sentinel: schema created

        _, write_params = self._client_vdb.get_write_params()
        self._client_vdb.write_to_index(nvingest_records, **write_params)
