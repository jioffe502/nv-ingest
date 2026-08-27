# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pandas as pd
import pytest

from nemo_retriever.common.vdb.lancedb_schema import build_lancedb_rows
from nemo_retriever.common.vdb.records import VdbUploadError


def test_build_lancedb_rows_rejects_partial_embedding_coverage() -> None:
    frame = pd.DataFrame(
        [
            {"text": "embedded", "metadata": {"embedding": [0.1, 0.2]}},
            {"text": "missing", "metadata": {}},
        ]
    )

    with pytest.raises(VdbUploadError, match="partial write.*missing embedding=1"):
        build_lancedb_rows(frame)


def test_build_lancedb_rows_preserves_embedding_split_identity() -> None:
    split_metadata = {
        "embedding": [0.1, 0.2],
        "embedding_parent_id": "parent",
        "embedding_chunk_id": "child",
        "embedding_chunk_index": 1,
        "embedding_chunk_count": 3,
        "embedding_chunk_start_token": 10,
        "embedding_chunk_end_token": 20,
    }
    frame = pd.DataFrame([{"text": "split child", "path": "doc.pdf", "page_number": 7, "metadata": split_metadata}])

    rows = build_lancedb_rows(frame)

    stored_metadata = json.loads(rows[0]["metadata"])
    assert {key: stored_metadata[key] for key in split_metadata if key != "embedding"} == {
        key: value for key, value in split_metadata.items() if key != "embedding"
    }
