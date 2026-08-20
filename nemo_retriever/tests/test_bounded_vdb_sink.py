# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

lancedb = pytest.importorskip("lancedb")

from nemo_retriever.common.vdb.sink import VdbSinkPolicy
from nemo_retriever.operators.vdb import IngestVdbOperator


def _embedded_frame(start: int, stop: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": f"chunk-{row_id}",
                "text_embeddings_1b_v2": {"embedding": [float(row_id), 1.0]},
                "source_id": f"/tmp/doc-{row_id}.pdf",
                "page_number": row_id,
                "metadata": {"content_metadata": {"type": "text", "id": f"row-{row_id}"}},
            }
            for row_id in range(start, stop)
        ]
    )


def _versions(uri: Path, table_name: str = "chunks") -> list[int]:
    table = lancedb.connect(str(uri)).open_table(table_name)
    return [int(version["version"]) for version in table.list_versions()]


def test_consume_batches_makes_one_lance_commit_for_many_input_batches(tmp_path: Path) -> None:
    operator = IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(tmp_path),
            "table_name": "chunks",
            "vector_dim": 2,
            "overwrite": True,
            "build_index": False,
        },
    )

    report = operator.consume_batches(
        iter([_embedded_frame(0, 2), _embedded_frame(2, 4), _embedded_frame(4, 6)]),
        operation_id="test-overwrite",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    table = lancedb.connect(str(tmp_path)).open_table("chunks")
    stored = table.to_arrow().sort_by("id").to_pylist()

    assert [row["id"] for row in stored] == [f"row-{row_id}" for row_id in range(6)]
    assert [row["text"] for row in stored] == [f"chunk-{row_id}" for row_id in range(6)]
    assert _versions(tmp_path) == [1]
    assert report.input_batches == 3
    assert report.rows_written == 6
    assert report.max_batch_bytes <= 1024


def test_projection_window_does_not_accumulate_a_small_corpus_before_first_emit(tmp_path: Path) -> None:
    operator = IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(tmp_path),
            "table_name": "chunks",
            "vector_dim": 2,
            "overwrite": True,
            "build_index": False,
        },
    )

    report = operator.consume_batches(
        iter([_embedded_frame(0, 600)]),
        operation_id="bounded-projection-window",
        policy=VdbSinkPolicy(max_batch_bytes=1 << 30, prefetch_batches=0),
    )

    assert report.rows_written == 600
    assert report.output_batches == 3
    assert report.max_pending_rows == 256
