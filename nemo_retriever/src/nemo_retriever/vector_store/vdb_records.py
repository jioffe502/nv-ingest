# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical VDB record builder.

Converts a pandas DataFrame (the graph pipeline's output format) into a list
of backend-neutral VDB record dicts.  Every VDB backend in ``nemo_retriever``
consumes this record format — it is the single source of truth for the
DataFrame → VDB record contract.

Canonical record schema (matches ``retriever.py`` query expectations)::

    vector       : list[float]   # embedding
    text         : str           # content
    metadata     : str           # JSON string (round-trips via json.loads)
    source       : str           # JSON string {"source_id": "..."}
    page_number  : int
    pdf_page     : str           # "basename_pagenum"
    pdf_basename : str
    filename     : str
    source_id    : str
    path         : str
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from nemo_retriever.vector_store.lancedb_utils import build_lancedb_row


def build_vdb_records(
    df: pd.DataFrame,
    *,
    embedding_column: str = "text_embeddings_1b_v2",
    embedding_key: str = "embedding",
    text_column: str = "text",
    include_text: bool = True,
) -> List[Dict[str, Any]]:
    """Convert a post-embed DataFrame into canonical VDB records.

    Rows without a valid embedding are silently skipped.
    """
    rows: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        row_out = build_lancedb_row(
            row,
            embedding_column=embedding_column,
            embedding_key=embedding_key,
            text_column=text_column,
            include_text=include_text,
        )
        if row_out is not None:
            rows.append(row_out)
    return rows
