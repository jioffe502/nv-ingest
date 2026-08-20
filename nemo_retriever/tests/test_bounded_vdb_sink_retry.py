# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

lancedb = pytest.importorskip("lancedb")

from nemo_retriever.common.vdb.sink import (
    CommitOutcomeUnknown,
    VdbOperationConflict,
    VdbSinkPolicy,
)
from nemo_retriever.operators.vdb import IngestVdbOperator


def _frame(start: int, stop: int, *, text_prefix: str = "chunk") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": f"{text_prefix}-{row_id}",
                "text_embeddings_1b_v2": {"embedding": [float(row_id), 1.0]},
                "source_id": f"/tmp/doc-{row_id}.pdf",
                "page_number": row_id,
                "metadata": {"content_metadata": {"type": "text", "id": f"row-{row_id}"}},
            }
            for row_id in range(start, stop)
        ]
    )


def _operator(uri: Path, *, overwrite: bool, **overrides) -> IngestVdbOperator:
    kwargs = {
        "uri": str(uri),
        "table_name": "chunks",
        "vector_dim": 2,
        "overwrite": overwrite,
        "build_index": False,
    }
    kwargs.update(overrides)
    return IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs=kwargs,
    )


def _ids_and_versions(uri: Path) -> tuple[list[str], list[int]]:
    table = lancedb.connect(str(uri)).open_table("chunks")
    ids = table.to_arrow().sort_by("id").column("id").to_pylist()
    versions = [int(item["version"]) for item in table.list_versions()]
    return ids, versions


def test_acknowledged_append_retry_with_same_operation_id_is_a_verified_noop(tmp_path: Path) -> None:
    policy = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)
    _operator(tmp_path, overwrite=True).consume_batches(iter([_frame(0, 2)]), operation_id="seed", policy=policy)
    operator = _operator(tmp_path, overwrite=False)
    first = operator.consume_batches(iter([_frame(10, 14)]), operation_id="append-10-14", policy=policy)
    before_retry = _ids_and_versions(tmp_path)

    retry = operator.consume_batches(
        iter([_frame(10, 12), _frame(12, 14)]),
        operation_id="append-10-14",
        policy=policy,
    )

    assert _ids_and_versions(tmp_path) == before_retry
    assert retry.outcome == "already_succeeded"
    assert retry.rows_written == first.rows_written == 4


def test_reusing_operation_id_with_different_canonical_content_is_rejected(tmp_path: Path) -> None:
    policy = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)
    _operator(tmp_path, overwrite=True).consume_batches(iter([_frame(0, 2)]), operation_id="seed", policy=policy)
    operator = _operator(tmp_path, overwrite=False)
    operator.consume_batches(iter([_frame(10, 12)]), operation_id="append", policy=policy)
    before = _ids_and_versions(tmp_path)

    with pytest.raises(VdbOperationConflict, match="different canonical content"):
        operator.consume_batches(
            iter([_frame(10, 12, text_prefix="changed")]),
            operation_id="append",
            policy=policy,
        )

    assert _ids_and_versions(tmp_path) == before


def test_reusing_operation_id_with_different_sink_configuration_is_rejected(tmp_path: Path) -> None:
    policy = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)
    _operator(tmp_path, overwrite=True, metric="l2").consume_batches(
        iter([_frame(0, 4)]), operation_id="configured-write", policy=policy
    )
    before = _ids_and_versions(tmp_path)

    with pytest.raises(VdbOperationConflict, match="different write request"):
        _operator(tmp_path, overwrite=True, metric="cosine").consume_batches(
            iter([_frame(0, 4)]), operation_id="configured-write", policy=policy
        )

    assert _ids_and_versions(tmp_path) == before


def test_lost_data_marker_after_append_commit_fails_closed_on_retry(tmp_path: Path, monkeypatch) -> None:
    policy = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)
    _operator(tmp_path, overwrite=True).consume_batches(iter([_frame(0, 2)]), operation_id="seed", policy=policy)
    operator = _operator(tmp_path, overwrite=False)
    tags_type = type(lancedb.connect(str(tmp_path)).open_table("chunks").tags)
    original_create = tags_type.create

    def fail_data_marker(self, tag: str, version: int) -> None:
        if tag.startswith("nemo_sink_data_"):
            raise RuntimeError("injected lost acknowledgement after data commit")
        original_create(self, tag, version)

    monkeypatch.setattr(tags_type, "create", fail_data_marker)
    with pytest.raises(RuntimeError, match="injected lost acknowledgement"):
        operator.consume_batches(iter([_frame(10, 14)]), operation_id="append-unknown", policy=policy)
    after_commit = _ids_and_versions(tmp_path)
    monkeypatch.setattr(tags_type, "create", original_create)

    with pytest.raises(CommitOutcomeUnknown, match="refusing to replay append"):
        operator.consume_batches(iter([_frame(10, 14)]), operation_id="append-unknown", policy=policy)

    assert _ids_and_versions(tmp_path) == after_commit
