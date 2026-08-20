# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for projection memory before the bounded VDB sink."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

lancedb = pytest.importorskip("lancedb")

from nemo_retriever.common.vdb.sink import VdbSinkPolicy
from nemo_retriever.operators.vdb import IngestVdbOperator

_INPUT_ROWS = 128
_MAX_BATCH_BYTES = 4096
_MAX_LEGACY_PROJECTION_ROWS = 2


def _large_wide_frame() -> pd.DataFrame:
    """Return one Ray-sized pandas block with a much narrower stored projection."""

    return pd.DataFrame(
        [
            {
                "text": f"chunk-{row_id}-" + ("x" * 3072),
                "text_embeddings_1b_v2": {"embedding": [float(row_id), 1.0]},
                "source_id": f"/tmp/doc-{row_id}.pdf",
                "page_number": row_id,
                "metadata": {"content_metadata": {"type": "text", "id": f"row-{row_id}"}},
                # This graph-result-only payload must not be copied into one
                # corpus-sized list of Python records during VDB projection.
                "wide_result_only": "y" * (32 << 10),
            }
            for row_id in range(_INPUT_ROWS)
        ]
    )


def _observe_record_batches(
    data: Any,
    *,
    row_counts: list[int],
    retained_bytes: list[int],
) -> Iterable[pa.RecordBatch]:
    assert not isinstance(data, (list, tuple, pd.DataFrame, pa.Table))
    for batch in data:
        assert isinstance(batch, pa.RecordBatch)
        row_counts.append(batch.num_rows)
        retained_bytes.append(int(batch.get_total_buffer_size()))
        yield batch


def test_large_input_block_is_projected_incrementally_before_byte_bounding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The byte-bounded sink must not first build Python records for the whole block."""

    from nemo_retriever.common.vdb import records as records_module

    frame = _large_wide_frame()
    assert int(frame.memory_usage(index=True, deep=True).sum()) > 4 << 20

    original_project = records_module.to_client_vdb_records
    legacy_projection_rows: list[int] = []

    def bounded_legacy_projection(rows: Any) -> list[list[dict[str, Any]]]:
        row_count = len(rows)
        legacy_projection_rows.append(row_count)
        assert row_count <= _MAX_LEGACY_PROJECTION_ROWS, (
            "whole-block Python VDB projection is unbounded: "
            f"received {row_count} rows under max_batch_bytes={_MAX_BATCH_BYTES}"
        )
        return original_project(rows)

    monkeypatch.setattr(records_module, "to_client_vdb_records", bounded_legacy_projection)

    connection_type = type(lancedb.connect(str(tmp_path)))
    table_type = lancedb.table.LanceTable
    original_create_table = connection_type.create_table
    original_add = table_type.add
    record_batch_rows: list[int] = []
    record_batch_bytes: list[int] = []

    def observed_create_table(self, name, data=None, *args, **kwargs):
        if name == "chunks" and data is not None:
            data = _observe_record_batches(
                data,
                row_counts=record_batch_rows,
                retained_bytes=record_batch_bytes,
            )
        return original_create_table(self, name, data, *args, **kwargs)

    def observed_add(self, data, *args, **kwargs):
        if self.name == "chunks":
            data = _observe_record_batches(
                data,
                row_counts=record_batch_rows,
                retained_bytes=record_batch_bytes,
            )
        return original_add(self, data, *args, **kwargs)

    monkeypatch.setattr(connection_type, "create_table", observed_create_table)
    monkeypatch.setattr(table_type, "add", observed_add)

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
        iter([frame]),
        operation_id="bounded-projection",
        policy=VdbSinkPolicy(max_batch_bytes=_MAX_BATCH_BYTES, prefetch_batches=0),
    )

    # A replacement Arrow-native projector may bypass the legacy helper. If it
    # still uses that helper, every invocation must be a bounded slice.
    assert not legacy_projection_rows or max(legacy_projection_rows) <= _MAX_LEGACY_PROJECTION_ROWS
    assert len(record_batch_rows) > 1
    assert sum(record_batch_rows) == _INPUT_ROWS
    assert max(record_batch_bytes) <= _MAX_BATCH_BYTES
    assert report.rows_written == _INPUT_ROWS
    assert report.max_batch_bytes == max(record_batch_bytes)

    table = lancedb.connect(str(tmp_path)).open_table("chunks")
    assert table.count_rows() == _INPUT_ROWS
    assert table.schema.names == ["vector", "text", "metadata", "source", "id"]
    assert "wide_result_only" not in table.schema.names
