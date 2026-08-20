# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

lancedb = pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.common.vdb.sink import VdbSinkPolicy, VdbWriteNotFinalized
from nemo_retriever.operators.vdb import IngestVdbOperator

_POLICY = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)


def _embedded_frame(start: int, stop: int, *, vector_dim: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": f"chunk-{row_id}",
                "text_embeddings_1b_v2": {
                    "embedding": [float(row_id), *[float(offset) for offset in range(1, vector_dim)]]
                },
                "source_id": f"/tmp/doc-{row_id}.pdf",
                "page_number": row_id,
                "metadata": {"content_metadata": {"type": "text", "id": f"row-{row_id}"}},
            }
            for row_id in range(start, stop)
        ]
    )


def _operator(uri: Path, *, overwrite: bool, vector_dim: int | None) -> IngestVdbOperator:
    return IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(uri),
            "table_name": "chunks",
            "vector_dim": vector_dim,
            "overwrite": overwrite,
            "build_index": False,
        },
    )


def _table(uri: Path):
    return lancedb.connect(str(uri)).open_table("chunks")


def _vector_dim(uri: Path) -> int:
    schema = _table(uri).schema
    schema = schema() if callable(schema) else schema
    return int(schema.field("vector").type.list_size)


def test_new_append_lost_create_ack_retry_never_silently_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry may resume/no-op or fail closed, but must never append twice."""

    connection_type = type(lancedb.connect(str(tmp_path)))
    original_create_table = connection_type.create_table
    lost_ack = False

    def create_then_lose_ack(self, name, data=None, *args, **kwargs):
        nonlocal lost_ack
        table = original_create_table(self, name, data, *args, **kwargs)
        if name == "chunks" and not lost_ack:
            lost_ack = True
            raise RuntimeError("injected lost acknowledgement after create commit")
        return table

    monkeypatch.setattr(connection_type, "create_table", create_then_lose_ack)
    operator = _operator(tmp_path, overwrite=False, vector_dim=2)

    def batches():
        return iter(
            [
                _embedded_frame(10, 12, vector_dim=2),
                _embedded_frame(12, 14, vector_dim=2),
            ]
        )

    with pytest.raises(RuntimeError, match="lost acknowledgement"):
        operator.consume_batches(batches(), operation_id="new-append-10-14", policy=_POLICY)

    assert sorted(_table(tmp_path).to_arrow()["id"].to_pylist()) == [
        "row-10",
        "row-11",
        "row-12",
        "row-13",
    ]
    with pytest.raises(VdbWriteNotFinalized, match="not finalized"):
        operator._vdb.retrieval([[10.0, 1.0]], top_k=4)

    with pytest.raises(VdbWriteNotFinalized, match="unfinished create operation"):
        _operator(tmp_path, overwrite=False, vector_dim=2).consume_batches(
            iter([_embedded_frame(20, 22, vector_dim=2)]),
            operation_id="different-operation",
            policy=_POLICY,
        )
    assert sorted(_table(tmp_path).to_arrow()["id"].to_pylist()) == [
        "row-10",
        "row-11",
        "row-12",
        "row-13",
    ]

    retry = operator.consume_batches(
        batches(),
        operation_id="new-append-10-14",
        policy=_POLICY,
    )

    assert sorted(_table(tmp_path).to_arrow()["id"].to_pylist()) == [
        "row-10",
        "row-11",
        "row-12",
        "row-13",
    ]
    assert retry.outcome == "success"
    assert operator._vdb.retrieval([[10.0, 1.0]], top_k=4)


def test_lost_create_ack_recovery_hashes_the_vector_value_lancedb_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection_type = type(lancedb.connect(str(tmp_path)))
    original_create_table = connection_type.create_table
    lost_ack = False

    def create_then_lose_ack(self, name, data=None, *args, **kwargs):
        nonlocal lost_ack
        table = original_create_table(self, name, data, *args, **kwargs)
        if name == "chunks" and not lost_ack:
            lost_ack = True
            raise RuntimeError("injected lost acknowledgement after normalized create")
        return table

    monkeypatch.setattr(connection_type, "create_table", create_then_lose_ack)
    operator = IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(tmp_path),
            "table_name": "chunks",
            "vector_dim": 2,
            "overwrite": False,
            "build_index": False,
            "on_bad_vectors": "fill",
            "fill_value": 7.0,
        },
    )
    frame = _embedded_frame(0, 1, vector_dim=2)
    frame.at[0, "text_embeddings_1b_v2"] = {"embedding": [float("nan"), 1.0]}

    with pytest.raises(RuntimeError, match="lost acknowledgement"):
        operator.consume_batches(iter([frame]), operation_id="normalized-create", policy=_POLICY)

    retry = operator.consume_batches(iter([frame]), operation_id="normalized-create", policy=_POLICY)
    assert retry.outcome == "success"
    assert _table(tmp_path).to_arrow()["vector"].to_pylist() == [[7.0, 7.0]]
    assert operator._vdb.retrieval([[7.0, 7.0]], top_k=1)


def test_streamed_create_infers_vector_dim_from_first_canonical_row(tmp_path: Path) -> None:
    report = _operator(tmp_path, overwrite=False, vector_dim=None).consume_batches(
        iter(
            [
                _embedded_frame(0, 2, vector_dim=3),
                _embedded_frame(2, 4, vector_dim=3),
            ]
        ),
        operation_id="infer-create-dim",
        policy=_POLICY,
    )

    assert report.rows_written == 4
    assert _table(tmp_path).count_rows() == 4
    assert _vector_dim(tmp_path) == 3


def test_streamed_overwrite_infers_new_vector_dim_instead_of_reusing_old_schema(tmp_path: Path) -> None:
    _operator(tmp_path, overwrite=True, vector_dim=2).consume_batches(
        iter([_embedded_frame(0, 2, vector_dim=2)]),
        operation_id="seed-dim-2",
        policy=_POLICY,
    )

    report = _operator(tmp_path, overwrite=True, vector_dim=None).consume_batches(
        iter(
            [
                _embedded_frame(10, 12, vector_dim=3),
                _embedded_frame(12, 14, vector_dim=3),
            ]
        ),
        operation_id="infer-overwrite-dim",
        policy=_POLICY,
    )

    assert report.rows_written == 4
    assert _table(tmp_path).count_rows() == 4
    assert _vector_dim(tmp_path) == 3


def test_inferred_dimension_applies_legacy_drop_policy_to_later_rows(tmp_path: Path) -> None:
    report = _operator(tmp_path, overwrite=True, vector_dim=None).consume_batches(
        iter(
            [
                _embedded_frame(0, 1, vector_dim=3),
                _embedded_frame(10, 11, vector_dim=2),
            ]
        ),
        operation_id="infer-then-drop",
        policy=_POLICY,
    )

    assert report.rows_written == 1
    assert _table(tmp_path).to_arrow()["id"].to_pylist() == ["row-0"]
    assert _vector_dim(tmp_path) == 3


def test_new_operation_appends_to_table_created_by_bounded_sink(tmp_path: Path) -> None:
    operator = _operator(tmp_path, overwrite=False, vector_dim=2)
    operator.consume_batches(
        iter([_embedded_frame(0, 2, vector_dim=2)]),
        operation_id="create-first",
        policy=_POLICY,
    )
    report = operator.consume_batches(
        iter([_embedded_frame(2, 4, vector_dim=2)]),
        operation_id="append-second",
        policy=_POLICY,
    )

    assert report.rows_written == 2
    assert sorted(_table(tmp_path).to_arrow()["id"].to_pylist()) == [
        "row-0",
        "row-1",
        "row-2",
        "row-3",
    ]
    assert operator._vdb.retrieval([[0.0, 1.0]], top_k=4)
