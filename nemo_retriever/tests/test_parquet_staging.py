# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet staging for batch ingest: worker sink, shard commits, resume, load, and CLI wiring."""

from __future__ import annotations

import importlib
import json
import os
from typing import Any

import lancedb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import nemo_retriever.ingest.execution as ingest_execution
import nemo_retriever.ingest.plan as ingest_plan
from nemo_retriever.common.vdb.arrow import cached_vector_schema
from nemo_retriever.common.vdb.lancedb import LanceDB, _create_lancedb_result
from nemo_retriever.common.vdb.records import VdbUploadError, _client_record_from_graph_row
from nemo_retriever.ingest import staging, staging_sink
from nemo_retriever.ingest.staging import StagingError, load_stage, stage_documents
from nemo_retriever.ingest.staging_sink import StagedParquetDatasink, StageTarget
from nemo_retriever.operators.vdb import STAGE_PARQUET_VDB_KWARG, IngestVdbOperator

_DIM = 4
_MODEL = "nvidia/test-embed"


def _rows(path: str, count: int = 3, *, explicit_slot: int | None = None, vectors: list | None = None) -> list[dict]:
    rows = []
    for index in range(count):
        metadata: dict[str, Any] = {"source_path": path}
        if explicit_slot is not None:
            metadata["content_metadata"] = {"id": f"slot-{explicit_slot}-row-{index}"}
        vector = vectors[index] if vectors else np.random.default_rng(index).standard_normal(_DIM).tolist()
        rows.append(
            {
                "text": f"{os.path.basename(path)} lexical row {index}",
                "text_embeddings_1b_v2": {"embedding": vector},
                "path": path,
                "page_number": index + 1,
                "metadata": metadata,
            }
        )
    return rows


def _sink(target: StageTarget, **vdb_kwargs: Any) -> StagedParquetDatasink:
    return StagedParquetDatasink(target, LanceDB(vector_dim=_DIM, embedding_model_name=_MODEL, **vdb_kwargs))


def _run_shard(*, explicit_ids: bool = False, fail_on: str | None = None):
    """Stand in for the Ray graph: one write task per shard."""

    def run(documents: list[str], target: StageTarget) -> list[dict]:
        if target.shard_id == fail_on:
            raise RuntimeError("simulated shard failure")
        # Explicit IDs win, so every file in a shard repeats the same IDs.
        rows = [row for path in documents for row in _rows(path, explicit_slot=0 if explicit_ids else None)]
        return [_sink(target).write([pd.DataFrame(rows)], None)]

    return run


def _inputs(tmp_path, count: int = 5) -> list[str]:
    paths = []
    for index in range(count):
        path = tmp_path / "inputs" / f"doc-{index}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"document {index}")
        paths.append(str(path))
    return paths


def _stage(tmp_path, documents: list[str], **run_kwargs: Any) -> dict:
    return stage_documents(
        documents,
        stage_dir=str(tmp_path / "stage"),
        shard_files=2,
        settings={"model": _MODEL},
        run_shard=_run_shard(**run_kwargs),
    )


def _lancedb_kwargs(tmp_path) -> dict:
    return {"uri": str(tmp_path / "lance"), "table_name": "chunks", "vector_dim": _DIM, "hybrid": True}


def test_staged_rows_match_the_in_driver_rows_and_cached_schema(tmp_path) -> None:
    rows = _rows("/docs/a.pdf")
    target = StageTarget(str(tmp_path), "000000", "a")

    result = _sink(target).write([pd.DataFrame(rows)], None)

    staged = pq.read_table(os.path.join(target.attempt_dir, result["part"]["name"]))
    expected = [_create_lancedb_result(_client_record_from_graph_row(row), expected_dim=_DIM)[0] for row in rows]
    # Parquet names the fixed-list child "element"; every other field and the model metadata match.
    assert staged.schema.remove(0).equals(cached_vector_schema(_DIM, _MODEL).remove(0), check_metadata=True)
    assert staged.drop_columns(["vector"]).to_pylist() == [
        {key: row[key] for key in ("id", "text", "source", "metadata")} for row in expected
    ]
    assert np.array_equal(
        np.asarray(staged.column("vector").to_pylist(), dtype=np.float32),
        np.asarray([row["vector"] for row in expected], dtype=np.float32),
    )


@pytest.mark.parametrize(("policy", "staged"), [("drop", 1), ("fill", 3)])
def test_bad_vectors_follow_on_bad_vectors(tmp_path, policy: str, staged: int) -> None:
    vectors = [[1.0, 0.0, 0.0, 0.0], [1.0, float("inf"), 0.0, 0.0], [1.0, None, 0.0, 0.0]]
    frame = pd.DataFrame(_rows("/docs/a.pdf", vectors=vectors))

    result = _sink(StageTarget(str(tmp_path), "000000", "a"), on_bad_vectors=policy).write([frame], None)

    assert result["part"]["rows"] == staged
    with pytest.raises(ValueError, match="finite values"):
        _sink(StageTarget(str(tmp_path), "000000", "b"), on_bad_vectors="error").write([frame], None)


def test_reader_batches_retain_at_most_one_bounded_row_group(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(staging_sink, "_ROW_GROUP_BYTES", 1 << 20)
    rows = [{**row, "text": f"{row['text']} " + "filler " * 20_000} for row in _rows("/docs/a.pdf", count=30)]
    target = StageTarget(str(tmp_path), "000000", "a")
    path = os.path.join(target.attempt_dir, _sink(target).write([pd.DataFrame(rows)], None)["part"]["name"])

    batches = list(staging._staged_reader([path]))

    assert pq.read_metadata(path).num_row_groups > 1
    assert sum(batch.num_rows for batch in batches) == 30
    assert max(batch.get_total_buffer_size() for batch in batches) <= 2 << 20


def test_stage_commits_shards_resumes_after_a_crash_and_skips_them_on_rerun(tmp_path, monkeypatch) -> None:
    documents = _inputs(tmp_path)
    write_json = staging._write_json

    def crash_on_second_commit(path: str, value: Any) -> None:
        if path.endswith(os.path.join("000001", "commit.json")):
            raise RuntimeError("simulated crash after the shard directory was published")
        write_json(path, value)

    monkeypatch.setattr(staging, "_write_json", crash_on_second_commit)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _stage(tmp_path, documents)
    monkeypatch.setattr(staging, "_write_json", write_json)
    (tmp_path / "stage" / "shards" / "stray.txt").write_text("not a shard")

    resumed = _stage(tmp_path, documents)
    rerun = stage_documents(
        documents, stage_dir=str(tmp_path / "stage"), shard_files=2, settings={"model": _MODEL}, run_shard=None
    )

    assert (resumed["shards"], resumed["staged_shards"], resumed["rows"]) == (3, 2, 15)
    assert (rerun["staged_shards"], rerun["rows"]) == (0, 15)
    assert sorted(os.listdir(tmp_path / "stage" / "shards")) == ["000000", "000001", "000002"]


def test_stage_fails_closed(tmp_path) -> None:
    documents = _inputs(tmp_path, 4)
    with pytest.raises(StagingError, match="share row IDs"):
        _stage(tmp_path / "duplicates", documents, explicit_ids=True)
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "notes.txt").write_text("x")
    with pytest.raises(StagingError, match="not empty"):
        stage_documents(documents, stage_dir=str(tmp_path / "used"), shard_files=2, settings={}, run_shard=None)

    _stage(tmp_path, documents)
    with open(documents[0], "a") as handle:
        handle.write("changed")
    with pytest.raises(StagingError, match="different inputs or settings"):
        _stage(tmp_path, documents)
    with staging._locked(str(tmp_path / "stage")):
        with pytest.raises(StagingError, match="Another process"):
            load_stage(str(tmp_path / "stage"), lancedb_kwargs=_lancedb_kwargs(tmp_path))


def test_missing_embeddings_refuse_the_shard(tmp_path) -> None:
    documents = _inputs(tmp_path, 1)

    def run(shard_documents: list[str], target: StageTarget) -> list[dict]:
        rows = _rows(shard_documents[0])
        rows[1].pop("text_embeddings_1b_v2")
        return [_sink(target).write([pd.DataFrame(rows)], None)]

    with pytest.raises(VdbUploadError, match="missing embedding=1"):
        stage_documents(documents, stage_dir=str(tmp_path / "stage"), shard_files=2, settings={}, run_shard=run)
    assert not (tmp_path / "stage" / "shards" / "000000").exists()


def test_load_replaces_the_table_once_and_reloads_a_removed_table(tmp_path) -> None:
    _stage(tmp_path, _inputs(tmp_path))

    loaded = load_stage(str(tmp_path / "stage"), lancedb_kwargs=_lancedb_kwargs(tmp_path))
    rerun = load_stage(str(tmp_path / "stage"), lancedb_kwargs=_lancedb_kwargs(tmp_path))
    lancedb.connect(str(tmp_path / "lance")).drop_table("chunks")
    reloaded = load_stage(str(tmp_path / "stage"), lancedb_kwargs=_lancedb_kwargs(tmp_path))

    table = lancedb.connect(str(tmp_path / "lance")).open_table("chunks")
    assert (loaded["rows"], loaded["skipped"], rerun["skipped"], reloaded["skipped"]) == (15, False, True, False)
    assert table.count_rows() == 15
    assert len(table.search("lexical", query_type="fts").limit(20).to_list()) == 15


def test_load_refuses_an_unfinished_or_damaged_stage(tmp_path) -> None:
    documents = _inputs(tmp_path)
    with pytest.raises(RuntimeError, match="simulated shard failure"):
        _stage(tmp_path, documents, fail_on="000001")
    with pytest.raises(StagingError, match="2 of 3 shards are not staged yet"):
        load_stage(str(tmp_path / "stage"), lancedb_kwargs=_lancedb_kwargs(tmp_path))

    _stage(tmp_path, documents)
    commit = json.loads((tmp_path / "stage" / "shards" / "000002" / "commit.json").read_text())
    os.unlink(tmp_path / "stage" / "shards" / "000002" / commit["parts"][0]["name"])
    with pytest.raises(StagingError, match="Staged parts are missing"):
        load_stage(str(tmp_path / "stage"), lancedb_kwargs=_lancedb_kwargs(tmp_path))
    assert not (tmp_path / "lance").exists()


def _request(tmp_path, *, run_mode: str = "batch", overwrite: bool = True, **staging_options: Any):
    document = tmp_path / "doc.txt"
    document.write_text("hello")
    return ingest_plan.IngestPlanRequest(
        source=ingest_plan.IngestSourceOptions(documents=[str(document)]),
        runtime=ingest_plan.IngestRuntimeOptions(run_mode=run_mode),
        storage=ingest_plan.IngestStorageOptions(lancedb_uri=str(tmp_path / "lance"), overwrite=overwrite),
        staging=ingest_plan.IngestStagingOptions(**staging_options),
    )


def test_plan_validates_staging_options(tmp_path) -> None:
    stage_dir = str(tmp_path / "stage")

    plan = ingest_plan.resolve_ingest_plan(_request(tmp_path, stage_dir=stage_dir))

    assert plan.staging == ingest_plan.IngestStagingOptions(stage_dir=stage_dir, shard_files=1000)
    for request, message in [
        (_request(tmp_path, shard_files=5), "requires --stage-dir"),
        (_request(tmp_path, run_mode="inprocess", stage_dir=stage_dir), "retriever ingest batch"),
        (_request(tmp_path, overwrite=False, stage_dir=stage_dir), "append"),
        (_request(tmp_path, stage_dir="s3://bucket/stage"), "local filesystem"),
    ]:
        with pytest.raises(ValueError, match=message):
            ingest_plan.resolve_ingest_plan(request)


def test_cli_stage_options_are_batch_only_and_shown_in_dry_run(tmp_path) -> None:
    cli = importlib.import_module("nemo_retriever.cli.main").app
    document = tmp_path / "doc.txt"
    document.write_text("hello")
    stage_dir = str(tmp_path / "stage")

    local = CliRunner().invoke(cli, ["ingest", "local", str(document), "--stage-dir", stage_dir])
    dry_run = CliRunner().invoke(cli, ["ingest", "batch", str(document), "--stage-dir", stage_dir, "--dry-run"])

    assert local.exit_code == 1 and "--stage-dir" in local.output
    assert json.loads(dry_run.output)["staging"] == {"stage_dir": stage_dir, "shard_files": 1000}


def test_staged_execution_passes_each_shard_its_target(tmp_path, monkeypatch) -> None:
    plan = ingest_plan.resolve_ingest_plan(_request(tmp_path, stage_dir=str(tmp_path / "stage")))
    shard_plans = []

    class FakeIngestor:
        def __init__(self, shard_plan: Any) -> None:
            self.shard_plan = shard_plan

        def _remote_stage_diagnostics(self) -> dict:
            return {}

        def ingest(self) -> list:
            shard_plans.append(self.shard_plan)
            vdb_kwargs = dict(self.shard_plan.vdb_params.vdb_kwargs)
            target = vdb_kwargs.pop(STAGE_PARQUET_VDB_KWARG)
            rows = [row for path in self.shard_plan.documents for row in _rows(path, vectors=[[0.5] * 2048] * 3)]
            return [StagedParquetDatasink(target, LanceDB(**vdb_kwargs)).write([pd.DataFrame(rows)], None)]

    monkeypatch.setattr(ingest_execution, "build_ingest_pipeline", FakeIngestor)

    summary = ingest_execution.execute_ingest_plan(plan).to_summary_dict()

    [shard_plan] = shard_plans
    assert shard_plan.staging is None and shard_plan.documents == plan.documents
    assert STAGE_PARQUET_VDB_KWARG not in plan.vdb_params.vdb_kwargs
    assert (summary["n_rows"], summary["result_n_rows"]) == (3, 3)


def test_operator_never_falls_back_to_a_direct_upload(tmp_path) -> None:
    target = StageTarget(str(tmp_path), "000000", "a")
    operator = IngestVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": str(tmp_path), STAGE_PARQUET_VDB_KWARG: target})
    sparse = IngestVdbOperator(vdb_op="lancedb", vdb_kwargs={"sparse": True, STAGE_PARQUET_VDB_KWARG: target})

    with pytest.raises(RuntimeError, match="terminal VDB upload"):
        operator.process(pd.DataFrame(_rows("/docs/a.pdf")))
    with pytest.raises(ValueError, match="dense vectors"):
        sparse.staging_datasink()
