# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resumable Parquet staging for ``retriever ingest batch --stage-dir``.

Sorted input files are grouped into shards. Each shard runs as one Ray
execution that ends in :class:`StagedParquetDatasink`, and commits when its
attempt directory is renamed to ``shards/<id>`` and ``commit.json`` is written
there. A rerun keeps committed shards, deletes everything else under
``shards/``, and stages the rest. :func:`load_stage` then loads every part
through one ``LanceDB.ingest_arrow`` call, which validates the row count and
index coverage before it returns.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from filelock import FileLock, Timeout

from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb.records import VdbConversionTally
from nemo_retriever.ingest.staging_sink import SHARDS_DIRNAME, StageTarget, fsync_path

logger = logging.getLogger(__name__)

_STAGE_FILE = "stage.json"
_COMMIT_FILE = "commit.json"
_LOAD_FILE = "load.json"
_LOCK_FILE = ".lock"


class StagingError(RuntimeError):
    """A staging directory, shard, or load failed a durability check."""


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _write_json(path: str, value: Any) -> None:
    """Replace ``path`` atomically and durably."""
    temp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    fsync_path(os.path.dirname(path))


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@contextlib.contextmanager
def _locked(stage_dir: str) -> Iterator[None]:
    try:
        with FileLock(os.path.join(stage_dir, _LOCK_FILE), timeout=0):
            yield
    except Timeout as exc:
        raise StagingError(f"Another process is using staging directory {stage_dir!r}") from exc


def _plan_shards(documents: Sequence[str], shard_files: int) -> tuple[list[list[str]], str]:
    """Group input files by sorted real path; return the shards and an input fingerprint."""
    paths: dict[str, str] = {}
    for document in documents:
        path = os.path.realpath(document)
        if not os.path.isfile(path):
            raise StagingError(f"Parquet staging requires local input files; {document!r} is not a file")
        if path in paths:
            raise StagingError(f"Input file {path!r} appears more than once")
        paths[path] = str(document)
    ordered = sorted(paths)
    stats = [os.stat(path) for path in ordered]
    fingerprint = _sha256_json([[path, stat.st_size, stat.st_mtime_ns] for path, stat in zip(ordered, stats)])
    shards = [
        [paths[path] for path in ordered[start : start + shard_files]] for start in range(0, len(ordered), shard_files)
    ]
    return shards, fingerprint


def _bind_stage(stage_dir: str, header: Mapping[str, Any]) -> None:
    """Create ``stage.json``, or require the existing one to match this run's inputs and settings."""
    path = os.path.join(stage_dir, _STAGE_FILE)
    if os.path.exists(path):
        if _read_json(path) != dict(header):
            raise StagingError(
                f"Staging directory {stage_dir!r} was created for different inputs or settings. "
                "Rerun with the original inputs and options, or use a new staging directory."
            )
        return
    if any(name != _LOCK_FILE for name in os.listdir(stage_dir)):
        raise StagingError(f"Staging directory {stage_dir!r} is not empty and has no {_STAGE_FILE}")
    _write_json(path, dict(header))


def _remove_uncommitted(stage_dir: str) -> set[str]:
    """Delete every shard directory or stray file without ``commit.json``; return committed shard IDs."""
    root = os.path.join(stage_dir, SHARDS_DIRNAME)
    committed = set()
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        path = os.path.join(root, name)
        if os.path.isfile(os.path.join(path, _COMMIT_FILE)):
            committed.add(name)
        elif os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)
    return committed


def _duplicate_ids(paths: Sequence[str]) -> list[str]:
    counts = pa.chunked_array(pq.read_table(path, columns=["id"]).column("id") for path in paths).value_counts()
    return counts.filter(pc.greater(counts.field("counts"), 1)).field("values").to_pylist()


def _commit_shard(target: StageTarget, documents: Sequence[str], results: Sequence[Mapping[str, Any]]) -> int:
    """Check one attempt's write results, publish its directory, and write ``commit.json``; return its rows."""
    tally = VdbConversionTally()
    for result in results:
        tally.merge(result["tally"])
    stage_errors = sum(result["stage_errors"] for result in results)
    if stage_errors:
        raise StagingError(f"Shard {target.shard_id} has {stage_errors} row(s) with stage errors")
    if tally.missing_embeddings:
        tally.raise_for_failures()
    parts = [result["part"] for result in results if result["part"]]
    paths = [os.path.join(target.attempt_dir, part["name"]) for part in parts]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise StagingError(f"Shard {target.shard_id} is missing published parts: {missing[:5]}")
    duplicates = _duplicate_ids(paths) if paths else []
    if duplicates:
        raise StagingError(f"Shard {target.shard_id} has rows that share row IDs: {duplicates[:5]}")

    shard_dir = os.path.join(target.stage_dir, SHARDS_DIRNAME, target.shard_id)
    if os.path.isdir(target.attempt_dir):
        os.rename(target.attempt_dir, shard_dir)
    else:
        os.makedirs(shard_dir)
    fsync_path(os.path.dirname(shard_dir))
    rows = sum(part["rows"] for part in parts)
    _write_json(
        os.path.join(shard_dir, _COMMIT_FILE),
        {"inputs": list(documents), "parts": parts, "rows": rows, "tally": dataclasses.asdict(tally)},
    )
    return rows


def stage_documents(
    documents: Sequence[str],
    *,
    stage_dir: str,
    shard_files: int,
    settings: Mapping[str, Any],
    run_shard: Callable[[list[str], StageTarget], Sequence[Mapping[str, Any]]],
    stage_error_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Stage every uncommitted shard of ``documents`` under ``stage_dir``.

    ``settings`` holds the resolved options that change staged rows; a rerun
    must match them and the inputs. ``run_shard(documents, target)`` runs the
    ingest graph for one shard with ``target`` as its staging target and
    returns the datasink's per-task results.
    """
    stage_dir = os.path.abspath(stage_dir)
    shards, inputs_sha256 = _plan_shards(documents, shard_files)
    header = {
        "version": 1,
        "shards": len(shards),
        "shard_files": shard_files,
        "inputs_sha256": inputs_sha256,
        "settings_sha256": _sha256_json(settings),
    }
    os.makedirs(stage_dir, exist_ok=True)
    with _locked(stage_dir):
        _bind_stage(stage_dir, header)
        committed = _remove_uncommitted(stage_dir)
        for index, shard_documents in enumerate(shards):
            shard_id = f"{index:06d}"
            if shard_id in committed:
                continue
            target = StageTarget(
                stage_dir, shard_id, uuid.uuid4().hex[:12], stage_error_columns=tuple(stage_error_columns)
            )
            logger.info("Staging shard %d of %d (%d files)", index + 1, len(shards), len(shard_documents))
            rows = _commit_shard(target, shard_documents, run_shard(shard_documents, target))
            logger.info("Committed shard %d of %d with %d rows", index + 1, len(shards), rows)

        tally = VdbConversionTally()
        rows = 0
        for index in range(len(shards)):
            commit = _read_json(os.path.join(stage_dir, SHARDS_DIRNAME, f"{index:06d}", _COMMIT_FILE))
            tally.merge(commit["tally"])
            rows += commit["rows"]
        tally.raise_for_failures()
    return {"stage_dir": stage_dir, "shards": len(shards), "staged_shards": len(shards) - len(committed), "rows": rows}


def _staged_reader(paths: Sequence[str]) -> pa.RecordBatchReader:
    """Read parts one row group at a time; each batch retains at most its decoded row group."""
    with pq.ParquetFile(paths[0]) as first:
        schema = first.schema_arrow

    def batches() -> Iterator[pa.RecordBatch]:
        for path in paths:
            with pq.ParquetFile(path) as parquet:
                for index in range(parquet.num_row_groups):
                    yield from parquet.read_row_group(index).to_batches()

    return pa.RecordBatchReader.from_batches(schema, batches())


def _table_rows(vdb: LanceDB) -> int | None:
    import lancedb

    db = lancedb.connect(vdb.uri)
    if vdb.table_name not in db.list_tables().tables:
        return None
    return int(db.open_table(vdb.table_name).count_rows())


def load_stage(stage_dir: str, *, lancedb_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Replace the LanceDB table with every staged row through one ``ingest_arrow`` call.

    The load is skipped when ``load.json`` records this table and the table
    still holds the staged row count. A failed load reruns from the Parquet.
    """
    stage_dir = os.path.abspath(stage_dir)
    with _locked(stage_dir):
        header = _read_json(os.path.join(stage_dir, _STAGE_FILE))
        shard_dirs = [os.path.join(stage_dir, SHARDS_DIRNAME, f"{index:06d}") for index in range(header["shards"])]
        commits = [_read_json(os.path.join(path, _COMMIT_FILE)) for path in shard_dirs if os.path.isdir(path)]
        if len(commits) != header["shards"]:
            raise StagingError(f"{header['shards'] - len(commits)} of {header['shards']} shards are not staged yet")
        rows = sum(commit["rows"] for commit in commits)
        if not rows:
            raise StagingError("Staging produced no rows to load")
        paths = [
            os.path.join(path, part["name"]) for path, commit in zip(shard_dirs, commits) for part in commit["parts"]
        ]
        vdb = LanceDB(**{**lancedb_kwargs, "overwrite": True})
        load_path = os.path.join(stage_dir, _LOAD_FILE)
        record = {"uri": os.path.abspath(vdb.uri), "table_name": vdb.table_name, "rows": rows}
        if os.path.exists(load_path) and _read_json(load_path) == record and _table_rows(vdb) == rows:
            return {**record, "skipped": True}
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise StagingError(f"Staged parts are missing: {missing[:5]}")
        vdb.ingest_arrow(_staged_reader(paths), expected_rows=rows)
        _write_json(load_path, record)
    return {**record, "skipped": False}
