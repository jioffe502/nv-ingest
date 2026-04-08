# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LanceDB implementation of the :class:`VectorStore` interface."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from nemo_retriever.params.models import LanceDbParams
from nemo_retriever.vector_store.lancedb_utils import infer_vector_dim, lancedb_schema
from nemo_retriever.vector_store.vdb import VectorStore

logger = logging.getLogger(__name__)


class LanceDBBackend(VectorStore):
    """LanceDB vector store backend.

    Lazily connects and creates the table on the first :meth:`write_rows`
    call so that the embedding dimension can be inferred from the data.
    """

    def __init__(self, params: LanceDbParams | None = None) -> None:
        self._params = params or LanceDbParams()
        self._db: Any = None
        self._table: Any = None

    def open_table(self) -> None:
        """Open an existing LanceDB table without creating it.

        Used by the driver to run post-pipeline finalization (e.g. index
        creation) after distributed workers have written all rows.
        """
        import lancedb

        self._db = lancedb.connect(uri=self._params.lancedb_uri)
        self._table = self._db.open_table(self._params.table_name)

    def create_table(self, *, dim: int, **kwargs: Any) -> None:
        import lancedb

        self._db = lancedb.connect(uri=self._params.lancedb_uri)
        schema = lancedb_schema(vector_dim=dim)
        mode = "overwrite" if self._params.overwrite else "create"
        self._table = self._db.create_table(
            self._params.table_name,
            schema=schema,
            mode=mode,
        )

    def write_rows(self, rows: Sequence[dict[str, Any]], **kwargs: Any) -> None:
        if not rows:
            return
        if self._table is None:
            self.create_table(dim=infer_vector_dim(list(rows)))
        self._table.add(list(rows))

    def create_index(self, **kwargs: Any) -> None:
        if self._table is None:
            return
        if not self._params.create_index:
            return

        from nemo_retriever.vector_store.lancedb_store import create_lancedb_index

        try:
            create_lancedb_index(self._table, cfg=self._params)
        except RuntimeError:
            logger.warning(
                "Index creation failed (likely too few rows for %d partitions); skipping.",
                self._params.num_partitions,
                exc_info=True,
            )
