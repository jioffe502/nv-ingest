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


def _operator(uri: Path, *, overwrite: bool) -> IngestVdbOperator:
    return IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(uri),
            "table_name": "chunks",
            "vector_dim": 2,
            "overwrite": overwrite,
            "build_index": False,
        },
    )


def _state(uri: Path) -> tuple[list[dict], list[int]]:
    table = lancedb.connect(str(uri)).open_table("chunks")
    rows = table.to_arrow().sort_by("id").to_pylist()
    versions = [int(version["version"]) for version in table.list_versions()]
    return rows, versions


def _fail_after_first_batch(start: int, stop: int):
    midpoint = start + (stop - start) // 2
    yield _embedded_frame(start, midpoint)
    raise RuntimeError("injected source failure after first batch")


def test_overwrite_midstream_failure_preserves_target_and_same_input_retry_succeeds(tmp_path: Path) -> None:
    policy = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)
    _operator(tmp_path, overwrite=True).consume_batches(
        iter([_embedded_frame(0, 2)]), operation_id="seed", policy=policy
    )
    before_rows, before_versions = _state(tmp_path)
    operator = _operator(tmp_path, overwrite=True)

    with pytest.raises(RuntimeError, match="injected source failure"):
        operator.consume_batches(
            _fail_after_first_batch(10, 14),
            operation_id="replace-10-14",
            policy=policy,
        )

    assert _state(tmp_path) == (before_rows, before_versions)

    report = operator.consume_batches(
        iter([_embedded_frame(10, 12), _embedded_frame(12, 14)]),
        operation_id="replace-10-14",
        policy=policy,
    )

    after_rows, after_versions = _state(tmp_path)
    assert [row["id"] for row in after_rows] == [f"row-{row_id}" for row_id in range(10, 14)]
    assert len(after_versions) == len(before_versions) + 1
    assert report.rows_written == 4


def test_append_midstream_failure_preserves_target_and_same_input_retry_does_not_duplicate(tmp_path: Path) -> None:
    policy = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)
    _operator(tmp_path, overwrite=True).consume_batches(
        iter([_embedded_frame(0, 2)]), operation_id="seed", policy=policy
    )
    before_rows, before_versions = _state(tmp_path)
    operator = _operator(tmp_path, overwrite=False)

    with pytest.raises(RuntimeError, match="injected source failure"):
        operator.consume_batches(
            _fail_after_first_batch(10, 14),
            operation_id="append-10-14",
            policy=policy,
        )

    assert _state(tmp_path) == (before_rows, before_versions)

    report = operator.consume_batches(
        iter([_embedded_frame(10, 12), _embedded_frame(12, 14)]),
        operation_id="append-10-14",
        policy=policy,
    )

    after_rows, after_versions = _state(tmp_path)
    stored_ids = [row["id"] for row in after_rows]
    assert stored_ids == ["row-0", "row-1", "row-10", "row-11", "row-12", "row-13"]
    assert len(stored_ids) == len(set(stored_ids))
    assert len(after_versions) == len(before_versions) + 1
    assert report.rows_written == 4
