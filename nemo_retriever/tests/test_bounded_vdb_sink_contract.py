# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

lancedb = pytest.importorskip("lancedb")

from nemo_retriever.common.vdb.sink import (
    OversizedVdbRowError,
    VdbSinkPolicy,
    VdbWriteNotFinalized,
    _schemas_have_same_fields,
)
from nemo_retriever.operators.vdb import IngestVdbOperator


def _frame(rows: list[tuple[str, str, list[float]]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": text,
                "text_embeddings_1b_v2": {"embedding": vector},
                "source_id": f"/tmp/{row_id}.pdf",
                "page_number": 1,
                "metadata": {"content_metadata": {"type": "text", "id": row_id}},
            }
            for row_id, text, vector in rows
        ]
    )


def _operator(uri: Path, **overrides) -> IngestVdbOperator:
    kwargs = {
        "uri": str(uri),
        "table_name": "chunks",
        "vector_dim": 2,
        "overwrite": True,
        "build_index": False,
    }
    kwargs.update(overrides)
    return IngestVdbOperator(vdb_op="lancedb", vdb_kwargs=kwargs)


def test_zero_batches_are_a_noop(tmp_path: Path) -> None:
    report = _operator(tmp_path).consume_batches(
        iter(()),
        operation_id="empty",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    assert report.outcome == "empty_noop"
    assert report.rows_written == 0
    assert set(report.timings) == {
        "prepare",
        "write",
        "time_to_data_commit",
        "validate",
        "vector_index",
        "fts_index",
        "index",
        "optimize",
        "total",
    }
    assert "chunks" not in lancedb.connect(str(tmp_path)).list_tables().tables


def test_one_row_larger_than_budget_fails_before_arrow_allocation_or_table_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_retriever.common.vdb import sink as sink_module

    operator = _operator(tmp_path)
    data = _frame([("huge", "x" * 4096, [1.0, 0.0])])

    def fail_if_arrow_batch_is_allocated(*args, **kwargs):
        raise AssertionError("an over-budget canonical row reached Arrow allocation")

    monkeypatch.setattr(sink_module, "_record_batch", fail_if_arrow_batch_is_allocated)

    with pytest.raises(OversizedVdbRowError, match="exceeding max_batch_bytes=256"):
        operator.consume_batches(
            iter([data]),
            operation_id="oversized",
            policy=VdbSinkPolicy(max_batch_bytes=256, prefetch_batches=1),
        )

    assert "chunks" not in lancedb.connect(str(tmp_path)).list_tables().tables


def test_null_vector_batches_never_allocate_an_over_budget_parent_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_retriever.common.vdb import sink as sink_module

    budget = 1024
    original_record_batch = sink_module._record_batch
    allocated_bytes: list[int] = []

    def observed_record_batch(*args, **kwargs):
        batch = original_record_batch(*args, **kwargs)
        allocated_bytes.append(int(batch.get_total_buffer_size()))
        return batch

    monkeypatch.setattr(sink_module, "_record_batch", observed_record_batch)
    operator = _operator(
        tmp_path,
        vector_dim=128,
        validate_vector_length=False,
        on_bad_vectors="null",
    )
    data = _frame([(f"row-{index}", f"text-{index}", [1.0]) for index in range(8)])

    report = operator.consume_batches(
        iter([data]),
        operation_id="null-vectors-under-budget",
        policy=VdbSinkPolicy(max_batch_bytes=budget, prefetch_batches=1),
    )

    assert allocated_bytes
    assert max(allocated_bytes) <= budget
    assert report.rows_written == 8


def test_invalid_partition_does_not_hide_valid_rows_in_another_partition(tmp_path: Path) -> None:
    invalid = pd.DataFrame([{"text": "missing vector", "source_id": "/tmp/bad.pdf"}])
    valid = _frame([("valid", "searchable", [1.0, 0.0])])

    report = _operator(tmp_path).consume_batches(
        iter([invalid, valid]),
        operation_id="partition-independent",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    table = lancedb.connect(str(tmp_path)).open_table("chunks")
    assert table.count_rows() == 1
    assert table.to_arrow().column("id").to_pylist() == ["valid"]
    assert report.input_batches == 2
    assert report.rows_written == 1


def test_sidecar_metadata_is_projected_before_the_arrow_boundary(tmp_path: Path) -> None:
    sidecar = pd.DataFrame([{"source": "doc.pdf", "department": "finance"}])
    operator = IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(tmp_path),
            "table_name": "chunks",
            "vector_dim": 2,
            "build_index": False,
            "meta_dataframe": sidecar,
            "meta_source_field": "source",
            "meta_fields": ["department"],
            "meta_join_key": "source_name",
        },
    )

    operator.consume_batches(
        iter([_frame([("doc", "quarterly report", [1.0, 0.0])])]),
        operation_id="sidecar",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    stored_metadata = lancedb.connect(str(tmp_path)).open_table("chunks").to_arrow().column("metadata")[0].as_py()
    assert stored_metadata == '{"type":"text","id":"doc","page_number":1,"department":"finance"}'


def test_vector_and_fts_indices_are_built_once_after_every_batch(tmp_path: Path, monkeypatch) -> None:
    operator = _operator(tmp_path, build_index=True, hybrid=True, index_type="IVF_FLAT", num_partitions=2)
    observed_counts: list[int] = []
    original_write_to_index = operator._vdb.write_to_index

    def observed_write_to_index(*args, table=None, **kwargs):
        observed_counts.append(int(table.count_rows()))
        return original_write_to_index(*args, table=table, **kwargs)

    monkeypatch.setattr(operator._vdb, "write_to_index", observed_write_to_index)
    batches = [
        _frame([("a", "alpha", [1.0, 0.0]), ("b", "beta", [0.9, 0.1])]),
        _frame([("c", "gamma", [0.0, 1.0]), ("d", "delta", [0.1, 0.9])]),
    ]

    report = operator.consume_batches(
        iter(batches),
        operation_id="indexed",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    table = lancedb.connect(str(tmp_path)).open_table("chunks")
    index_names = [index.name.lower() for index in table.list_indices()]
    assert observed_counts == [4]
    assert any("vector" in name for name in index_names)
    assert any("text" in name or "fts" in name for name in index_names)
    assert report.rows_written == 4


def test_success_requires_configured_index_coverage(tmp_path: Path, monkeypatch) -> None:
    operator = _operator(tmp_path, build_index=True, index_type="IVF_FLAT", num_partitions=2)
    monkeypatch.setattr(operator._vdb, "write_to_index", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="missing index columns"):
        operator.consume_batches(
            iter([_frame([("a", "alpha", [1.0, 0.0]), ("b", "beta", [0.0, 1.0])])]),
            operation_id="missing-index",
            policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
        )

    with pytest.raises(VdbWriteNotFinalized, match="not finalized"):
        operator._vdb.retrieval([[1.0, 0.0]], top_k=2)


def test_single_row_ivf_skip_remains_a_successful_single_batch(tmp_path: Path) -> None:
    operator = _operator(tmp_path, build_index=True, index_type="IVF_FLAT", num_partitions=2)

    report = operator.consume_batches(
        iter([_frame([("only", "one row", [1.0, 0.0])])]),
        operation_id="single-row",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    table = lancedb.connect(str(tmp_path)).open_table("chunks")
    assert report.outcome == "success"
    assert table.count_rows() == 1
    assert table.list_indices() == []
    assert operator._vdb.retrieval([[1.0, 0.0]], top_k=1)[0][0]["id"] == "only"


def test_schema_validation_ignores_only_internal_sink_identity() -> None:
    expected = pa.schema([pa.field("id", pa.string())], metadata={b"product": b"v1"})
    with_identity = expected.with_metadata(
        {
            b"product": b"v1",
            b"nemo_retriever.sink_create_operation_sha256": b"operation",
            b"nemo_retriever.sink_create_request_sha256": b"request",
        }
    )
    changed_product_metadata = with_identity.with_metadata(
        {
            **(with_identity.metadata or {}),
            b"product": b"v2",
        }
    )

    assert _schemas_have_same_fields(with_identity, expected)
    assert not _schemas_have_same_fields(changed_product_metadata, expected)


def test_append_preserves_a_compatible_schema_superset(tmp_path: Path) -> None:
    legacy_schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("id", pa.string()),
            pa.field("legacy_extra", pa.string()),
        ],
        metadata={
            b"retrieval_mode": b"dense",
            b"nemo_retriever.retrieval_mode": b"dense",
        },
    )
    db = lancedb.connect(str(tmp_path))
    db.create_table(
        "chunks",
        data=[
            {
                "vector": [1.0, 0.0],
                "text": "seed",
                "metadata": '{"type":"text","id":"seed"}',
                "source": '{"source_id":"/tmp/seed.pdf"}',
                "id": "seed",
                "legacy_extra": "preserved",
            }
        ],
        schema=legacy_schema,
    )

    report = _operator(tmp_path, overwrite=False).consume_batches(
        iter([_frame([("new", "new", [0.0, 1.0])])]),
        operation_id="append-schema-superset",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    table = db.open_table("chunks")
    rows = sorted(table.to_arrow().to_pylist(), key=lambda row: row["id"])
    assert report.outcome == "success"
    assert report.rows_written == 1
    assert table.schema == legacy_schema
    assert [(row["id"], row["legacy_extra"]) for row in rows] == [
        ("new", None),
        ("seed", "preserved"),
    ]
    assert _operator(tmp_path, overwrite=False)._vdb.retrieval([[0.0, 1.0]], top_k=1)[0][0]["id"] == "new"


def test_success_revalidates_index_coverage_after_optimize(tmp_path: Path, monkeypatch) -> None:
    from lancedb.table import LanceTable

    original_optimize = LanceTable.optimize

    def optimize_then_drop_indices(self, *args, **kwargs) -> None:
        original_optimize(self, *args, **kwargs)
        self.checkout_latest()
        for index in list(self.list_indices()):
            self.drop_index(index.name)

    monkeypatch.setattr(LanceTable, "optimize", optimize_then_drop_indices)
    operator = _operator(tmp_path, build_index=True, index_type="IVF_FLAT", num_partitions=2)

    with pytest.raises(RuntimeError, match="missing index columns"):
        operator.consume_batches(
            iter([_frame([("a", "alpha", [1.0, 0.0]), ("b", "beta", [0.0, 1.0])])]),
            operation_id="post-optimize-index-validation",
            policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1, optimize=True),
        )

    with pytest.raises(VdbWriteNotFinalized, match="not finalized"):
        operator._vdb.retrieval([[1.0, 0.0]], top_k=1)


def test_success_revalidates_row_count_after_optimize(tmp_path: Path, monkeypatch) -> None:
    from lancedb.table import LanceTable

    original_optimize = LanceTable.optimize

    def optimize_then_delete_a_row(self, *args, **kwargs) -> None:
        original_optimize(self, *args, **kwargs)
        self.delete("id = 'b'")

    monkeypatch.setattr(LanceTable, "optimize", optimize_then_delete_a_row)
    operator = _operator(tmp_path)

    with pytest.raises(RuntimeError, match="row-count validation failed after finalization"):
        operator.consume_batches(
            iter([_frame([("a", "alpha", [1.0, 0.0]), ("b", "beta", [0.0, 1.0])])]),
            operation_id="post-optimize-row-validation",
            policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1, optimize=True),
        )

    with pytest.raises(VdbWriteNotFinalized, match="not finalized"):
        operator._vdb.retrieval([[1.0, 0.0]], top_k=1)
