# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.common.vdb.sink import (
    VdbOperationConflict,
    VdbSinkPolicy,
    VdbWriteNotFinalized,
)
from nemo_retriever.operators.vdb import IngestVdbOperator

_POLICY = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)


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


def _operator(uri: Path, *, overwrite: bool, build_index: bool = False) -> IngestVdbOperator:
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


def test_reader_rejects_data_commit_until_failed_finalization_is_resumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _operator(tmp_path, overwrite=True).consume_batches(
        iter([_frame(0, 2)]),
        operation_id="seed",
        policy=_POLICY,
    )
    operator = _operator(tmp_path, overwrite=False, build_index=True)
    original_write_to_index = operator._vdb.write_to_index
    fail_once = True

    def fail_finalization(*args, **kwargs):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("injected index failure")
        return original_write_to_index(*args, **kwargs)

    monkeypatch.setattr(operator._vdb, "write_to_index", fail_finalization)
    with pytest.raises(RuntimeError, match="injected index failure"):
        operator.consume_batches(
            iter([_frame(10, 14)]),
            operation_id="append-and-finalize",
            policy=_POLICY,
        )

    with pytest.raises(VdbWriteNotFinalized, match="not finalized"):
        operator._vdb.retrieval([[0.0, 1.0]], top_k=10)

    with pytest.raises(VdbOperationConflict, match="unfinished bounded-sink operation"):
        _operator(tmp_path, overwrite=False).consume_batches(
            iter([_frame(20, 22)]),
            operation_id="different-append",
            policy=_POLICY,
        )

    report = operator.consume_batches(
        iter([_frame(10, 14)]),
        operation_id="append-and-finalize",
        policy=_POLICY,
    )
    assert report.rows_written == 4
    assert operator._vdb.retrieval([[0.0, 1.0]], top_k=10)


def test_definite_precommit_failure_keeps_previous_success_readable(tmp_path: Path) -> None:
    operator = _operator(tmp_path, overwrite=True)
    operator.consume_batches(iter([_frame(0, 2)]), operation_id="seed", policy=_POLICY)

    def fail_before_second_batch():
        yield _frame(10, 12)
        raise RuntimeError("injected source failure")

    with pytest.raises(RuntimeError, match="injected source failure"):
        operator.consume_batches(fail_before_second_batch(), operation_id="failed-overwrite", policy=_POLICY)

    hits = operator._vdb.retrieval([[0.0, 1.0]], top_k=10)
    assert {hit["id"] for hit in hits[0]} == {"row-0", "row-1"}
