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

    **Configuring the backend.** Callers can either:

    * pass ``params`` and let the operator construct the client VDB from
      ``VdbUploadParams.backend`` + ``client_vdb_kwargs``, or
    * pass a pre-constructed client VDB via ``vdb_op``
      (e.g. ``Milvus(...)`` / ``LanceDB(...)`` from
      ``nv_ingest_client.util.vdb``).  ``params`` still supplies the
      record-shaping config (embedding column, text column, etc.).

    LanceDB is the default backend. Custom ``vdb_op`` implementations should
    follow the existing ``nv_ingest_client.util.vdb.adt_vdb.VDB`` contract:
    ``create_index()``, ``write_to_index(records)``, ``retrieval()``, and
    ``run(records)``. Milvus is handled specially inside this wrapper only to
    avoid per-batch index waits; callers still pass the standard Milvus VDB
    instance via ``vdb_op=Milvus(...)``.

    **Concurrency**: This operator must run with ``concurrency=1`` in batch
    mode.  The single actor creates the backend destination on its first
    write (respecting backend-specific overwrite/recreate settings) and
    appends on subsequent writes.  Backend work that must run exactly once
    after all batches is handled by ``finalize()`` on the driver.
    """

    def __init__(
        self,
        *,
        params: Any = None,
        vdb_op: Any = None,
    ) -> None:
        super().__init__()
        from nemo_retriever.params.models import LanceDbParams, VdbUploadParams

        # Store as self.<name> so get_constructor_kwargs() captures both for
        # deferred reconstruction on Ray workers.
        self.params = params
        self.vdb_op = vdb_op

        if isinstance(params, VdbUploadParams):
            self._vdb_params = params
            self._lance_params = params.lancedb
        elif isinstance(params, LanceDbParams):
            self._vdb_params = None
            self._lance_params = params
        else:
            self._vdb_params = VdbUploadParams()
            self._lance_params = self._vdb_params.lancedb

        if vdb_op is not None:
            self._backend_name = type(vdb_op).__name__.lower()
        else:
            self._backend_name = getattr(self._vdb_params, "backend", "lancedb") if self._vdb_params else "lancedb"

        self._client_vdb: Any = vdb_op
        self._table: Any = None
        self._milvus_client: Any = None
        self._index_created: bool = False
        self._pending_records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Client VDB construction
    # ------------------------------------------------------------------

    def _create_client_vdb(self) -> Any:
        """Lazily construct the client VDB instance."""
        from nv_ingest_client.util.vdb import get_vdb_op_cls

        if self._backend_name == "lancedb":
            from nemo_retriever.vector_store.lancedb_utils import build_client_lancedb

            return build_client_lancedb(self._lance_params)

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

    def finalize(self) -> None:
        """Driver-side post-pipeline hook.

        ``AbstractOperator.run()`` fires the preprocess/process/postprocess
        lifecycle once per batch, so anything that must run exactly once at
        the end of ingestion (flush, wait-for-index, etc.) is driven from
        here by the driver after ``executor.ingest()`` returns.
        """
        if self._client_vdb is None and self._backend_name == "milvus":
            self._client_vdb = self._create_client_vdb()
        if self._client_vdb is None:
            return
        if self._is_milvus_vdb(self._client_vdb):
            self._finalize_milvus()

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
    # Non-LanceDB write path (Milvus, custom VDBs, etc.)
    # ------------------------------------------------------------------

    def _write_via_client(self, records: List[Dict[str, Any]]) -> None:
        """Convert canonical records and delegate to the configured client VDB."""
        nvingest_records = _canonical_to_nvingest(records)
        if not nvingest_records:
            return

        if self._is_milvus_vdb(self._client_vdb):
            self._write_milvus_streaming(nvingest_records)
            return

        if not self._index_created:
            self._client_vdb.create_index()
            self._index_created = True

        self._client_vdb.write_to_index(nvingest_records)

    # ------------------------------------------------------------------
    # Milvus streaming write path
    # ------------------------------------------------------------------

    @staticmethod
    def _milvus_import_error() -> ImportError:
        return ImportError(
            "Milvus VDB upload requires pymilvus. Install it with "
            "`pip install 'nv-ingest-client[milvus]'`."
        )

    @staticmethod
    def _is_milvus_vdb(vdb_op: Any) -> bool:
        try:
            from nv_ingest_client.util.vdb.milvus import Milvus
        except ImportError:
            return False
        return isinstance(vdb_op, Milvus)

    @staticmethod
    def _milvus_client_kwargs(create_params: Dict[str, Any]) -> Dict[str, Any]:
        client_kwargs = {"uri": create_params["milvus_uri"]}
        username = create_params.get("username")
        password = create_params.get("password")
        if username or password:
            client_kwargs["token"] = f"{username or ''}:{password or ''}"
        alias = create_params.get("alias")
        if alias is not None:
            client_kwargs["alias"] = alias
        return client_kwargs

    def _write_milvus_streaming(self, nvingest_records: List[List[Dict[str, Any]]]) -> None:
        """Stream a batch to Milvus without per-batch wait_for_index."""
        (
            MilvusClient,
            cleanup_records,
            create_nvingest_collection,
            pandas_file_reader,
        ) = self._load_milvus_write_helpers()

        collection_name, create_params = self._client_vdb.get_connection_params()
        if not isinstance(collection_name, str):
            raise ValueError(
                "VDBUploadOperator's Milvus streaming path requires a string collection_name; "
                f"got {type(collection_name).__name__}."
            )
        if bool(create_params.get("sparse", False)):
            raise NotImplementedError(
                "Milvus sparse/hybrid ingestion via VDBUploadOperator's streaming path is not yet supported. "
                "Construct the Milvus client with sparse=False."
            )

        if self._milvus_client is None:
            create_nvingest_collection(collection_name=collection_name, **create_params)
            self._milvus_client = MilvusClient(**self._milvus_client_kwargs(create_params))
            self._index_created = True

        meta_dataframe = getattr(self._client_vdb, "meta_dataframe", None)
        if isinstance(meta_dataframe, str):
            meta_dataframe = pandas_file_reader(meta_dataframe)

        cleaned_records = cleanup_records(
            nvingest_records,
            enable_text=getattr(self._client_vdb, "enable_text", True),
            enable_charts=getattr(self._client_vdb, "enable_charts", True),
            enable_tables=getattr(self._client_vdb, "enable_tables", True),
            enable_images=getattr(self._client_vdb, "enable_images", True),
            enable_infographics=getattr(self._client_vdb, "enable_infographics", True),
            meta_dataframe=meta_dataframe,
            meta_source_field=getattr(self._client_vdb, "meta_source_field", None),
            meta_fields=getattr(self._client_vdb, "meta_fields", None),
        )
        if not cleaned_records:
            logger.warning("No records with embeddings to insert into Milvus.")
            return

        self._milvus_client.insert(collection_name=collection_name, data=cleaned_records)

    @staticmethod
    def _load_milvus_write_helpers() -> tuple[Any, Any, Any, Any]:
        try:
            from nv_ingest_client.util.vdb.milvus import (
                MilvusClient,
                cleanup_records,
                create_nvingest_collection,
                pandas_file_reader,
            )
        except ImportError as exc:
            raise VDBUploadOperator._milvus_import_error() from exc
        if MilvusClient is None:
            raise VDBUploadOperator._milvus_import_error()
        return MilvusClient, cleanup_records, create_nvingest_collection, pandas_file_reader

    @staticmethod
    def _load_milvus_finalize_helpers() -> tuple[Any, Any]:
        try:
            from nv_ingest_client.util.vdb.milvus import MilvusClient, wait_for_index
        except ImportError as exc:
            raise VDBUploadOperator._milvus_import_error() from exc
        if MilvusClient is None:
            raise VDBUploadOperator._milvus_import_error()
        return MilvusClient, wait_for_index

    def _finalize_milvus(self) -> None:
        """Flush Milvus and wait once for all collection indexes to catch up."""
        MilvusClient, wait_for_index = self._load_milvus_finalize_helpers()

        collection_name, create_params = self._client_vdb.get_connection_params()
        if not isinstance(collection_name, str):
            raise ValueError(
                "VDBUploadOperator.finalize() requires a string Milvus collection_name; "
                f"got {type(collection_name).__name__}."
            )

        client = MilvusClient(**self._milvus_client_kwargs(create_params))
        if hasattr(client, "has_collection") and not client.has_collection(collection_name):
            return

        client.flush(collection_name)
        row_count = int(client.get_collection_stats(collection_name=collection_name).get("row_count", 0))
        if row_count == 0:
            return

        index_names = client.list_indexes(collection_name)
        expected_rows = {index_name: row_count for index_name in index_names}
        if getattr(self._client_vdb, "no_wait_index", False) or not expected_rows:
            return

        wait_for_index(collection_name, expected_rows, client)
