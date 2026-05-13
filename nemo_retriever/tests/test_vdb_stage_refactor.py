# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import pytest

from nemo_retriever.vdb import lancedb_bulk
from nemo_retriever.vdb.lancedb_bulk import LanceDBConfig, handle_lancedb, write_text_embeddings_dir_to_lancedb


def test_local_stage6_reexports_moved_vdb_stage_app() -> None:
    from nemo_retriever.local.stages import stage6_vdb_upload
    from nemo_retriever.vdb import stage as vdb_stage

    assert stage6_vdb_upload.app is vdb_stage.app


def test_text_embeddings_dir_upload_uses_canonical_lancedb(monkeypatch, tmp_path) -> None:
    init_calls: list[dict[str, Any]] = []
    run_calls: list[Any] = []

    class FakeLanceDB:
        def __init__(self, **kwargs: Any) -> None:
            init_calls.append(kwargs)

        def run(self, records: Any) -> Any:
            run_calls.append(records)
            return records

    monkeypatch.setattr(lancedb_bulk, "LanceDB", FakeLanceDB)

    row = {
        "document_type": "text",
        "metadata": {
            "embedding": [0.1, 0.2],
            "content": "hello",
            "content_metadata": {"page_number": 1},
            "source_metadata": {"source_name": "doc.pdf"},
        },
    }
    (tmp_path / "doc.text_embeddings.json").write_text(json.dumps({"df_records": [row]}), encoding="utf-8")

    cfg = LanceDBConfig(
        uri="db",
        table_name="docs",
        overwrite=False,
        create_index=False,
        index_type="IVF_FLAT",
        metric="cosine",
        num_partitions=7,
        num_sub_vectors=8,
        hybrid=True,
        fts_language="English",
    )

    info = write_text_embeddings_dir_to_lancedb(tmp_path, cfg=cfg)

    assert init_calls == [
        {
            "uri": "db",
            "table_name": "docs",
            "overwrite": False,
            "create_index": False,
            "index_type": "IVF_FLAT",
            "metric": "cosine",
            "num_partitions": 7,
            "num_sub_vectors": 8,
            "hybrid": True,
            "fts_language": "English",
        }
    ]
    assert run_calls == [[[row]]]
    assert info["n_files"] == 1
    assert info["processed"] == 1
    assert info["skipped"] == 0


def test_handle_lancedb_honors_append_mode(monkeypatch) -> None:
    write_calls: list[LanceDBConfig] = []
    index_calls: list[LanceDBConfig] = []

    class FakeDb:
        @staticmethod
        def open_table(_table_name: str) -> object:
            return object()

    monkeypatch.setattr(lancedb_bulk.lancedb, "connect", lambda uri: FakeDb())
    monkeypatch.setattr(lancedb_bulk, "_build_lancedb_rows_from_df", lambda rows: [{"vector": [0.1, 0.2]}])
    monkeypatch.setattr(lancedb_bulk, "_write_rows_to_lancedb", lambda rows, *, cfg: write_calls.append(cfg))
    monkeypatch.setattr(lancedb_bulk, "create_lancedb_index", lambda table, *, cfg: index_calls.append(cfg))

    info = handle_lancedb([{"text": "hello"}], uri="db", table_name="docs", mode="append")

    assert info == {"rows_written": 1}
    assert write_calls[0].overwrite is False
    assert index_calls[0].overwrite is False


def test_handle_lancedb_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be 'overwrite' or 'append'"):
        handle_lancedb([], uri="db", table_name="docs", mode="merge")
