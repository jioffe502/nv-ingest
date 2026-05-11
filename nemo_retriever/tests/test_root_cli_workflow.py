# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
from typing import Any

from typer.testing import CliRunner

import nemo_retriever.adapters.cli.sdk_workflow as sdk_workflow


RUNNER = CliRunner()
cli_main = importlib.import_module("nemo_retriever.adapters.cli.main")


class _FakeIngestor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.vdb_upload_params = None

    def files(self, documents: list[str]) -> "_FakeIngestor":
        self.calls.append(("files", documents))
        return self

    def extract(self) -> "_FakeIngestor":
        self.calls.append(("extract", None))
        return self

    def embed(self) -> "_FakeIngestor":
        self.calls.append(("embed", None))
        return self

    def vdb_upload(self, params: Any) -> "_FakeIngestor":
        self.calls.append(("vdb_upload", params))
        self.vdb_upload_params = params
        return self

    def ingest(self) -> list[dict[str, str]]:
        self.calls.append(("ingest", None))
        return [{"status": "ok"}]


def test_root_ingest_runs_default_sdk_chain(monkeypatch) -> None:
    fake_ingestor = _FakeIngestor()
    create_calls: list[dict[str, Any]] = []

    def fake_create_ingestor(**kwargs: Any) -> _FakeIngestor:
        create_calls.append(kwargs)
        return fake_ingestor

    monkeypatch.setattr(sdk_workflow, "create_ingestor", fake_create_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", "data/multimodal_test.pdf"])

    assert result.exit_code == 0
    assert create_calls == [{"run_mode": "inprocess"}]
    assert [name for name, _value in fake_ingestor.calls] == [
        "files",
        "extract",
        "embed",
        "vdb_upload",
        "ingest",
    ]
    assert fake_ingestor.calls[0] == ("files", ["data/multimodal_test.pdf"])
    assert fake_ingestor.vdb_upload_params.vdb_op == "lancedb"
    assert fake_ingestor.vdb_upload_params.vdb_kwargs == {"uri": "lancedb", "table_name": "nv-ingest"}
    assert "Ingested 1 document(s) into LanceDB lancedb/nv-ingest." in result.output


def test_root_ingest_passes_vdb_options_and_run_mode(monkeypatch, tmp_path) -> None:
    fake_ingestor = _FakeIngestor()
    create_calls: list[dict[str, Any]] = []
    first_document = tmp_path / "a.pdf"
    globbed_document = tmp_path / "b" / "c.pdf"
    first_document.write_bytes(b"%PDF-1.4\n")
    globbed_document.parent.mkdir()
    globbed_document.write_bytes(b"%PDF-1.4\n")

    def fake_create_ingestor(**kwargs: Any) -> _FakeIngestor:
        create_calls.append(kwargs)
        return fake_ingestor

    monkeypatch.setattr(sdk_workflow, "create_ingestor", fake_create_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(first_document),
            str(globbed_document.parent),
            "--run-mode",
            "batch",
            "--lancedb-uri",
            "/tmp/lancedb",
            "--table-name",
            "docs",
        ],
    )

    assert result.exit_code == 0
    assert create_calls == [{"run_mode": "batch"}]
    assert fake_ingestor.calls[0] == ("files", [str(first_document), str(globbed_document)])
    assert fake_ingestor.vdb_upload_params.vdb_kwargs == {"uri": "/tmp/lancedb", "table_name": "docs"}
    assert "Ingested 2 document(s) into LanceDB /tmp/lancedb/docs." in result.output


def test_root_query_passes_query_options_and_prints_json(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []
    query_calls: list[str] = []
    hits = [
        {"text": "passage", "page_number": 1, "_distance": 0.2},
        {"text": "other", "page_number": 2, "_distance": 0.4},
    ]

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str) -> list[dict[str, Any]]:
            query_calls.append(query)
            return hits

    monkeypatch.setattr(sdk_workflow, "Retriever", FakeRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "Which animal is responsible for typos?",
            "--top-k",
            "3",
            "--lancedb-uri",
            "/tmp/lancedb",
            "--table-name",
            "docs",
        ],
    )

    assert result.exit_code == 0
    assert retriever_calls == [{"top_k": 3, "vdb_kwargs": {"uri": "/tmp/lancedb", "table_name": "docs"}}]
    assert query_calls == ["Which animal is responsible for typos?"]
    assert json.loads(result.output) == hits
    assert result.output == json.dumps(hits, indent=2, sort_keys=True, default=str) + "\n"
