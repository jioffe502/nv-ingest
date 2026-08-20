# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_retriever.common.params import VdbSinkParams, VdbUploadParams
from nemo_retriever.graph.ingestor_runtime import build_post_extract_graph


def test_vdb_sink_params_reach_the_terminal_operator_without_entering_backend_kwargs(tmp_path) -> None:
    params = VdbUploadParams(
        vdb_kwargs={
            "uri": str(tmp_path),
            "table_name": "chunks",
            "vector_dim": 2,
            "build_index": False,
        },
        sink=VdbSinkParams(
            max_batch_bytes=64 << 20,
            prefetch_batches=2,
            optimize=True,
            operation_id="qualification-run",
        ),
    )

    graph = build_post_extract_graph(vdb_upload_params=params, stage_order=())
    operator = graph.roots[0].operator

    assert operator.sink_policy.max_batch_bytes == 64 << 20
    assert operator.sink_policy.prefetch_batches == 2
    assert operator.sink_policy.optimize is True
    assert operator.operation_id == "qualification-run"
    assert "sink" not in operator._vdb_kwargs
    assert "operation_id" not in operator._vdb_kwargs


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_batch_bytes", 0), ("prefetch_batches", -1), ("operation_id", "   ")],
)
def test_vdb_sink_params_reject_invalid_bounds_and_identity(field: str, value) -> None:
    with pytest.raises(ValueError):
        VdbSinkParams(**{field: value})
