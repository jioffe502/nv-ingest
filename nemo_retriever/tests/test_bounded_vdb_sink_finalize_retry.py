# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

lancedb = pytest.importorskip("lancedb")

from nemo_retriever.common.vdb.sink import (
    VdbOperationConflict,
    VdbSinkPolicy,
    VdbWriteNotFinalized,
)
from nemo_retriever.operators.vdb import IngestVdbOperator


def _frame(start: int, stop: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": f"chunk-{row_id}",
                "text_embeddings_1b_v2": {"embedding": [float(row_id), 1.0]},
                "source_id": f"/tmp/doc-{row_id}.pdf",
                "metadata": {"content_metadata": {"type": "text", "id": f"row-{row_id}"}},
            }
            for row_id in range(start, stop)
        ]
    )


def _operator(uri: Path, *, overwrite: bool, build_index: bool) -> IngestVdbOperator:
    return IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(uri),
            "table_name": "chunks",
            "vector_dim": 2,
            "overwrite": overwrite,
            "build_index": build_index,
            "index_type": "IVF_FLAT",
            "num_partitions": 2,
        },
    )


def test_append_retry_after_index_failure_resumes_without_readding_rows(tmp_path: Path, monkeypatch) -> None:
    policy = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)
    _operator(tmp_path, overwrite=True, build_index=False).consume_batches(
        iter([_frame(0, 2)]), operation_id="seed", policy=policy
    )
    operator = _operator(tmp_path, overwrite=False, build_index=True)
    original_write_to_index = operator._vdb.write_to_index

    def fail_index(*args, **kwargs):
        raise RuntimeError("injected vector index failure")

    monkeypatch.setattr(operator._vdb, "write_to_index", fail_index)
    with pytest.raises(RuntimeError, match="injected vector index failure"):
        operator.consume_batches(iter([_frame(10, 14)]), operation_id="append-and-index", policy=policy)

    table = lancedb.connect(str(tmp_path)).open_table("chunks")
    data_version = int(table.version)
    assert table.count_rows() == 6
    assert table.list_indices() == []

    monkeypatch.setattr(operator._vdb, "write_to_index", original_write_to_index)
    report = operator.consume_batches(
        iter([_frame(10, 12), _frame(12, 14)]),
        operation_id="append-and-index",
        policy=policy,
    )

    table = lancedb.connect(str(tmp_path)).open_table("chunks")
    ids = table.to_arrow().sort_by("id").column("id").to_pylist()
    assert ids == ["row-0", "row-1", "row-10", "row-11", "row-12", "row-13"]
    assert len(ids) == len(set(ids))
    assert report.data_version == data_version
    assert report.rows_written == 4
    assert table.list_indices()


def test_empty_retry_cannot_bypass_an_unfinished_operation(tmp_path: Path, monkeypatch) -> None:
    policy = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)
    operator = _operator(tmp_path, overwrite=True, build_index=True)

    def fail_index(*args, **kwargs):
        raise RuntimeError("injected vector index failure")

    monkeypatch.setattr(operator._vdb, "write_to_index", fail_index)
    with pytest.raises(RuntimeError, match="injected vector index failure"):
        operator.consume_batches(
            iter([_frame(0, 2)]),
            operation_id="unfinished-then-empty",
            policy=policy,
        )

    with pytest.raises(VdbOperationConflict, match="empty input cannot reconcile"):
        operator.consume_batches(
            iter(()),
            operation_id="unfinished-then-empty",
            policy=policy,
        )
    with pytest.raises(VdbWriteNotFinalized, match="not finalized"):
        operator._vdb.retrieval([[0.0, 1.0]], top_k=2)


@pytest.mark.parametrize("failed_state", ["data", "pending"])
def test_retry_finishes_marker_cleanup_after_success_became_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_state: str,
) -> None:
    policy = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)
    _operator(tmp_path, overwrite=True, build_index=False).consume_batches(
        iter([_frame(0, 2)]), operation_id="seed", policy=policy
    )
    operator = _operator(tmp_path, overwrite=False, build_index=False)
    tags_type = type(lancedb.connect(str(tmp_path)).open_table("chunks").tags)
    original_delete = tags_type.delete
    failed = False

    def fail_cleanup_once(self, tag: str) -> None:
        nonlocal failed
        if not failed and tag.startswith(f"nemo_sink_{failed_state}_"):
            failed = True
            raise RuntimeError(f"injected {failed_state} marker cleanup failure")
        original_delete(self, tag)

    monkeypatch.setattr(tags_type, "delete", fail_cleanup_once)
    with pytest.raises(RuntimeError, match="marker cleanup failure"):
        operator.consume_batches(
            iter([_frame(10, 12)]),
            operation_id=f"append-cleanup-{failed_state}",
            policy=policy,
        )

    with pytest.raises(VdbWriteNotFinalized, match="not finalized"):
        operator._vdb.retrieval([[10.0, 1.0]], top_k=4)

    monkeypatch.setattr(tags_type, "delete", original_delete)
    report = operator.consume_batches(
        iter([_frame(10, 12)]),
        operation_id=f"append-cleanup-{failed_state}",
        policy=policy,
    )

    table = lancedb.connect(str(tmp_path)).open_table("chunks")
    assert report.outcome == "already_succeeded"
    assert table.to_arrow().sort_by("id").column("id").to_pylist() == [
        "row-0",
        "row-1",
        "row-10",
        "row-11",
    ]
    assert not any(name.startswith(("nemo_sink_data_", "nemo_sink_pending_")) for name in table.tags.list())
    assert operator._vdb.retrieval([[10.0, 1.0]], top_k=4)
