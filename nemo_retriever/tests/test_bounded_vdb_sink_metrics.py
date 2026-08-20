# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.common.vdb.sink import VdbSinkPolicy
from nemo_retriever.operators.vdb import IngestVdbOperator


def _frame(start: int, stop: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": f"chunk-{row_id}-" + ("x" * 256),
                "text_embeddings_1b_v2": {"embedding": [float(row_id), 1.0]},
                "source_id": f"/tmp/doc-{row_id}.pdf",
                "page_number": row_id,
                "metadata": {"content_metadata": {"type": "text", "id": f"row-{row_id}"}},
                "result_only": "y" * 1024,
            }
            for row_id in range(start, stop)
        ]
    )


def test_write_report_exposes_bounds_rates_inventory_and_index_phases(tmp_path: Path) -> None:
    operator = IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(tmp_path),
            "table_name": "chunks",
            "vector_dim": 2,
            "overwrite": True,
            "build_index": True,
            "hybrid": True,
            "index_type": "IVF_FLAT",
            "num_partitions": 2,
        },
    )
    policy = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=2)

    report = operator.consume_batches(
        iter([_frame(0, 2), _frame(2, 4)]),
        operation_id="observable-write",
        policy=policy,
    )

    assert report.configured_max_batch_bytes == 1024
    assert report.configured_prefetch_batches == 2
    assert report.input_batches == 2
    assert report.input_rows == 4
    assert report.input_bytes > report.logical_bytes
    assert report.max_input_batch_bytes > report.max_batch_bytes
    assert report.output_batches >= 2
    assert 0 < report.max_batch_bytes <= report.configured_max_batch_bytes
    assert 0 < report.max_pending_rows <= 256
    assert report.versions_before == 0
    assert report.versions_after >= 1
    assert report.fragments_before == 0
    assert report.fragments_after == 1
    assert report.data_files_before == 0
    assert report.data_files_after == 1
    assert report.write_rows_per_second > 0
    assert report.write_bytes_per_second > 0
    assert report.timings["write"] > 0
    assert report.timings["vector_index"] >= 0
    assert report.timings["fts_index"] >= 0
    assert report.timings["time_to_data_commit"] > 0
    assert report.timings["total"] >= report.timings["time_to_data_commit"]
