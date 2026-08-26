# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned-Ray integration coverage for the terminal LanceDB sink boundary."""

from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

ray = pytest.importorskip("ray", minversion="2.56.1")
lancedb = pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.common.ray_resource_hueristics import ClusterResources, Resources
from nemo_retriever.common.vdb.sink import VdbSinkPolicy
from nemo_retriever.graph.executor import RayDataExecutor
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.vdb import IngestVdbOperator

_GRAPH_SCHEMA = pa.schema(
    [
        pa.field("block_id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field(
            "text_embeddings_1b_v2",
            pa.struct([pa.field("embedding", pa.list_(pa.float32()))]),
        ),
        pa.field("source_id", pa.string()),
        pa.field("page_number", pa.int64()),
        pa.field(
            "metadata",
            pa.struct(
                [
                    pa.field(
                        "content_metadata",
                        pa.struct(
                            [
                                pa.field("type", pa.string()),
                                pa.field("id", pa.string()),
                            ]
                        ),
                    )
                ]
            ),
        ),
        # Deliberately not a stored VDB column. The sink must project it away
        # while the public pandas result still retains it.
        pa.field("wide_result_only", pa.string()),
    ]
)


def _source_table(block_id: int) -> pa.Table:
    row_ids = range(block_id * 2, block_id * 2 + 2)
    return pa.Table.from_pylist(
        [
            {
                "block_id": block_id,
                "text": f"chunk-{row_id}",
                "text_embeddings_1b_v2": {"embedding": [float(row_id), 1.0]},
                "source_id": f"/tmp/doc-{row_id}.pdf",
                "page_number": row_id,
                "metadata": {
                    "content_metadata": {
                        "type": "text",
                        "id": f"row-{row_id}",
                    }
                },
                "wide_result_only": f"not-stored-{row_id}",
            }
            for row_id in row_ids
        ],
        schema=_GRAPH_SCHEMA,
    )


def _observe_record_batch_stream(
    data: Any,
    *,
    gate: Any,
    pulled_rows: list[int],
) -> Iterable[pa.RecordBatch]:
    """Wrap Lance input without replacing the real LanceDB write."""

    assert not isinstance(data, (list, tuple, pd.DataFrame, pa.Table)), (
        "the terminal sink must pass LanceDB a lazy RecordBatch source, " f"not a materialized {type(data).__name__}"
    )

    released = False
    for batch in data:
        assert isinstance(batch, pa.RecordBatch)
        pulled_rows.append(batch.num_rows)
        if not released:
            ray.get(gate.release.remote())
            released = True
        yield batch


@pytest.mark.integration
def test_terminal_vdb_sink_streams_ray_blocks_and_preserves_full_result(tmp_path, monkeypatch) -> None:
    """One Lance lifecycle must consume multiple Ray blocks without a global barrier."""

    assert ray.__version__ == "2.56.1", "this test qualifies Ray's pinned Datasink/iterator behavior"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False)

    try:
        # Keep worker code local so Ray cloudpickles it by value. Test modules
        # are not importable as ``tests.*`` in isolated Ray worker processes.
        class _FirstSinkPullGate:
            """Let the first arriving block lead; hold all later blocks."""

            def __init__(self) -> None:
                self._first_pull = threading.Event()
                self._completed_blocks: list[int] = []
                self._leader: int | None = None
                self._lock = threading.Lock()

            def arrive(self, block_id: int) -> bool:
                with self._lock:
                    if self._leader is None:
                        self._leader = block_id
                        return True
                    return False

            def wait_for_first_pull(self, timeout_s: float) -> None:
                if not self._first_pull.wait(timeout_s):
                    raise AssertionError(
                        "later Ray blocks were forced through a global barrier before "
                        "the terminal Lance iterator could pull its first batch"
                    )

            def release(self) -> None:
                self._first_pull.set()

            def record_completed_block(self, block_id: int) -> None:
                self._completed_blocks.append(block_id)

            def completed_blocks(self) -> list[int]:
                return list(self._completed_blocks)

        def _await_first_sink_pull(batch: pa.Table, gate: Any) -> pa.Table:
            """Make a total-corpus materialization barrier deterministic."""

            block_id = int(batch.column("block_id")[0].as_py())
            is_leader = ray.get(gate.arrive.remote(block_id))
            if not is_leader:
                ray.get(gate.wait_for_first_pull.remote(10.0))
            ray.get(gate.record_completed_block.remote(block_id))
            return batch

        gate_class = ray.remote(max_concurrency=8)(_FirstSinkPullGate)
        gate = gate_class.options(num_cpus=0).remote()

        source = ray.data.from_arrow([_source_table(i) for i in range(3)], override_num_blocks=3)
        assert source.num_blocks() == 3
        dataset = source.map_batches(
            _await_first_sink_pull,
            batch_format="pyarrow",
            batch_size=None,
            concurrency=3,
            fn_kwargs={"gate": gate},
        )

        original_repartition = ray.data.Dataset.repartition

        def reject_global_repartition(self, *args, **kwargs):
            requested_blocks = kwargs.get("num_blocks", args[0] if args else None)
            if requested_blocks == 1:
                raise AssertionError("terminal VDB sink must not call repartition(num_blocks=1)")
            return original_repartition(self, *args, **kwargs)

        monkeypatch.setattr(ray.data.Dataset, "repartition", reject_global_repartition)

        def reject_materialize(self, *args, **kwargs):
            raise AssertionError("post-sink continuation must not materialize the corpus before the first sink pull")

        monkeypatch.setattr(ray.data.Dataset, "materialize", reject_materialize)

        preflight_calls = 0
        initial_cluster = ClusterResources(
            total_resources=Resources(cpu_count=4, gpu_count=0),
            available_resources=Resources(cpu_count=2, gpu_count=0),
        )
        transient_cluster = ClusterResources(
            total_resources=Resources(cpu_count=4, gpu_count=0),
            available_resources=Resources(cpu_count=1, gpu_count=0),
        )

        def gather_changing_cluster_resources(_ray):
            nonlocal preflight_calls
            cluster = initial_cluster if preflight_calls == 0 else transient_cluster
            preflight_calls += 1
            return cluster

        monkeypatch.setattr(
            "nemo_retriever.graph.executor.gather_cluster_resources",
            gather_changing_cluster_resources,
        )

        # Observe the public Lance iterator boundary while retaining the real
        # LanceDB implementation and on-disk table mutation.
        connection_type = type(lancedb.connect(str(tmp_path)))
        table_type = lancedb.table.LanceTable
        original_create_table = connection_type.create_table
        original_add = table_type.add
        pulled_rows: list[int] = []

        def observed_create_table(self, name, data=None, *args, **kwargs):
            if name == "chunks" and data is not None:
                data = _observe_record_batch_stream(data, gate=gate, pulled_rows=pulled_rows)
            return original_create_table(self, name, data, *args, **kwargs)

        def observed_add(self, data, *args, **kwargs):
            if self.name == "chunks":
                data = _observe_record_batch_stream(data, gate=gate, pulled_rows=pulled_rows)
            return original_add(self, data, *args, **kwargs)

        monkeypatch.setattr(connection_type, "create_table", observed_create_table)
        monkeypatch.setattr(table_type, "add", observed_add)

        class _AddHeterogeneousResultColumn(AbstractOperator):
            """Model the ragged object columns emitted by extraction stages."""

            PRESERVE_PANDAS_OUTPUT = True

            def preprocess(self, data, **kwargs):
                return data

            def process(self, data, **kwargs):
                result = data.copy()
                result["tables"] = pd.Series(
                    [
                        np.array([{"text": f"table-{page_number}"}], dtype=object)
                        for page_number in result["page_number"]
                    ],
                    index=result.index,
                    dtype=object,
                )
                return result

            def postprocess(self, data, **kwargs):
                return data

        class _RequireFinalizedSink(AbstractOperator):
            """Stand in for the real post-VDB webhook ordering boundary."""

            def __init__(self, *, uri: str, expected_rows: int) -> None:
                super().__init__(uri=uri, expected_rows=expected_rows)

            def preprocess(self, data, **kwargs):
                return data

            def process(self, data, **kwargs):
                table = lancedb.connect(self.uri).open_table("chunks")
                assert table.count_rows() == self.expected_rows
                assert all(isinstance(value, np.ndarray) for value in data["tables"])
                return data

            def postprocess(self, data, **kwargs):
                return data

        graph = (
            Graph()
            >> _AddHeterogeneousResultColumn()
            >> IngestVdbOperator(
                vdb_op="lancedb",
                vdb_kwargs={
                    "uri": str(tmp_path),
                    "table_name": "chunks",
                    "vector_dim": 2,
                    "overwrite": True,
                    "build_index": False,
                },
            )
            >> _RequireFinalizedSink(uri=str(tmp_path), expected_rows=6)
        )

        executor = RayDataExecutor(graph)
        result = executor.ingest(
            dataset,
            vdb_sink_policy=VdbSinkPolicy(max_batch_bytes=256, prefetch_batches=0),
        )

        # The first sink pull had to happen before the non-leading blocks could
        # finish, and every block ran once (no second execution for the return).
        completed = ray.get(gate.completed_blocks.remote())
        assert Counter(completed) == Counter({0: 1, 1: 1, 2: 1})
        assert preflight_calls == 1
        assert len(pulled_rows) >= 2
        assert sum(pulled_rows) == 6

        expected_ids = list(range(6))
        result = result.sort_values("page_number", ignore_index=True)
        assert result.columns.tolist() == [*_GRAPH_SCHEMA.names, "tables"]
        assert result["block_id"].tolist() == [0, 0, 1, 1, 2, 2]
        assert result["text"].tolist() == [f"chunk-{row_id}" for row_id in expected_ids]
        assert result["source_id"].tolist() == [f"/tmp/doc-{row_id}.pdf" for row_id in expected_ids]
        assert result["page_number"].tolist() == expected_ids
        assert result["wide_result_only"].tolist() == [f"not-stored-{row_id}" for row_id in expected_ids]
        assert [list(value["embedding"]) for value in result["text_embeddings_1b_v2"]] == [
            [float(row_id), 1.0] for row_id in expected_ids
        ]
        assert [value["content_metadata"]["id"] for value in result["metadata"]] == [
            f"row-{row_id}" for row_id in expected_ids
        ]
        assert [value.tolist() for value in result["tables"]] == [
            [{"text": f"table-{row_id}"}] for row_id in expected_ids
        ]

        stored = lancedb.connect(str(tmp_path)).open_table("chunks").to_arrow().sort_by("id")
        assert stored.column_names == ["vector", "text", "metadata", "source", "id"]
        assert stored.column("id").to_pylist() == [f"row-{row_id}" for row_id in expected_ids]
        assert "wide_result_only" not in stored.column_names
        expected_result_bytes = int(result.memory_usage(index=True, deep=True).sum())
        assert executor.last_vdb_write_report.terminal_result_bytes == expected_result_bytes
        assert executor.last_vdb_write_report.terminal_result_bytes > 0

        with pytest.raises(TypeError, match="vdb_sink_policy must be a VdbSinkPolicy"):
            executor.ingest(dataset, vdb_sink_policy=object())
        assert executor.last_vdb_write_report is None
    finally:
        ray.shutdown()


@pytest.mark.integration
def test_final_producer_emits_canonical_stream_and_returns_write_receipt(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=2, num_gpus=0, include_dashboard=False, log_to_driver=False)

    try:
        class _WideFinalProducer(AbstractOperator):
            PRESERVE_PANDAS_OUTPUT = True

            def preprocess(self, data, **kwargs):
                return data

            def process(self, data, **kwargs):
                result = data.copy()
                result["producer_only_payload"] = ["x" * (32 << 10) for _ in range(len(result))]
                return result

            def postprocess(self, data, **kwargs):
                return data

        graph = (
            Graph()
            >> _WideFinalProducer()
            >> IngestVdbOperator(
                vdb_op="lancedb",
                vdb_kwargs={
                    "uri": str(tmp_path),
                    "table_name": "chunks",
                    "vector_dim": 2,
                    "overwrite": True,
                    "build_index": False,
                },
            )
        )
        dataset = ray.data.from_arrow([_source_table(0), _source_table(1)], override_num_blocks=2)
        executor = RayDataExecutor(graph, batch_size=None)
        receipt = executor.ingest(
            dataset,
            vdb_result_mode="write_receipt",
            vdb_transport_mode="canonical_stream",
            vdb_phase_telemetry=True,
            vdb_sink_policy=VdbSinkPolicy(max_batch_bytes=4096, prefetch_batches=0),
        )

        assert receipt.to_dict(orient="records") == [
            {
                "operation_id": executor.last_vdb_operation_id,
                "outcome": "success",
                "source_rows": 4,
                "rows_written": 4,
                "rejected_rows": 0,
                "upstream_error_count": 0,
                "upstream_error_fields": {},
                "rejection_reasons": {},
                "canonical_digest": executor.last_vdb_write_report.canonical_digest,
                "data_version": executor.last_vdb_write_report.data_version,
                "final_version": executor.last_vdb_write_report.final_version,
            }
        ]
        report = executor.last_vdb_write_report
        assert report.rows_written == report.input_rows == 4
        assert report.input_batches == 2
        assert report.input_unique_buffers > 0
        assert report.terminal_result_bytes == int(receipt.memory_usage(index=True, deep=True).sum())
        stored = lancedb.connect(str(tmp_path)).open_table("chunks").to_arrow().sort_by("id")
        assert stored.column_names == ["vector", "text", "metadata", "source", "id"]
        assert stored.column("id").to_pylist() == ["row-0", "row-1", "row-2", "row-3"]

        compatibility_uri = tmp_path / "compatibility"
        compatibility_graph = (
            Graph()
            >> _WideFinalProducer()
            >> IngestVdbOperator(
                vdb_op="lancedb",
                vdb_kwargs={
                    "uri": str(compatibility_uri),
                    "table_name": "chunks",
                    "vector_dim": 2,
                    "overwrite": True,
                    "build_index": False,
                },
            )
        )
        compatibility = RayDataExecutor(compatibility_graph, batch_size=None)
        compatibility_receipt = compatibility.ingest(
            dataset,
            vdb_result_mode="write_receipt",
            vdb_phase_telemetry=True,
            vdb_sink_policy=VdbSinkPolicy(max_batch_bytes=4096, prefetch_batches=0),
        )
        compatibility_stored = (
            lancedb.connect(str(compatibility_uri)).open_table("chunks").to_arrow().sort_by("id")
        )
        assert int(compatibility_receipt.at[0, "rows_written"]) == 4
        assert compatibility_stored.to_pylist() == stored.to_pylist()
        assert compatibility.last_vdb_write_report.canonical_digest == report.canonical_digest
        prefix = "NEMO_RETRIEVER_VDB_PULL "
        pull_reports = [
            json.loads(line.removeprefix(prefix))
            for line in capsys.readouterr().out.splitlines()
            if line.startswith(prefix)
        ]
        block_reports = [item for item in pull_reports if item["event"] == "block"]
        assert len(block_reports) == 4
        assert len([item for item in pull_reports if item["event"] == "end_of_stream"]) == 2
        assert [item["block_index"] for item in block_reports] == [0, 1, 0, 1]
        assert all(item["rows"] == 2 for item in block_reports)
        assert all(item["ray_wait_seconds"] >= 0 for item in pull_reports)
    finally:
        ray.shutdown()
