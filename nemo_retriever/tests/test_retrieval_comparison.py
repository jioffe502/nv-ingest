# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import nemo_retriever.harness.retrieval_comparison as retrieval_comparison
import pytest
from nemo_retriever.harness.retrieval_comparison import (
    LoadedVidoreMetadata,
    _audit_qrel,
    _indexed_doc_id,
    _query_metrics,
    build_comparison_artifacts,
    compare_retrieval,
)
from nemo_retriever.tools.recall.beir import BeirDataset


def _result(query_id: str, mode: str, docs: list[str]) -> dict:
    return {
        "query_id": query_id,
        "query": f"query {query_id}",
        "mode": mode,
        "latency_ms": {"dense": 10.0, "sparse": 5.0, "hybrid": 12.0}[mode],
        "hits": [{"pdf_page": doc_id, "text": f"hit {doc_id}", "_relevance_score": 1.0} for doc_id in docs],
    }


def test_indexed_doc_id_maps_raw_lancedb_metadata_and_source() -> None:
    row = {
        "metadata": json.dumps({"page_number": 12, "type": "text"}),
        "source": json.dumps({"source_id": "/data/annual_report.pdf", "source_name": "annual_report.pdf"}),
    }

    assert _indexed_doc_id(row, doc_id_field="pdf_page") == "annual_report_12"


def test_empty_text_on_an_indexed_page_is_not_reported_as_a_missing_page() -> None:
    audit = _audit_qrel(
        query_row={"answer": "42%"},
        corpus_row={"text": "The result was 42%."},
        indexed_texts=[""],
    )

    assert audit["status"] == "needs_review"


def test_per_query_ndcg_preserves_graded_qrels() -> None:
    metrics = _query_metrics(
        [{"pdf_page": "secondary"}, {"pdf_page": "primary"}],
        {"primary": 2, "secondary": 1},
        doc_id_field="pdf_page",
    )

    expected = (1.0 + 2.0 / math.log2(3)) / (2.0 + 1.0 / math.log2(3))
    assert metrics["ndcg_10"] == pytest.approx(expected)


def test_tiny_comparison_covers_rescues_regressions_and_extraction_audit(tmp_path) -> None:
    query_ids = ["rare", "paraphrase", "missing", "anchors", "regression"]
    dataset = BeirDataset(
        dataset_name="tiny",
        query_ids=query_ids,
        queries=[
            "Find identifier ZX-9001",
            "Which team lowered its energy use?",
            "What is the absent answer?",
            "What percentage was reported?",
            "Explain the semantic policy",
        ],
        qrels={query_id: {f"doc_{index}": 1} for index, query_id in enumerate(query_ids, start=1)},
    )
    metadata = LoadedVidoreMetadata(
        query_rows={
            "rare": {"query_id": "rare", "answer": "ZX-9001", "query_format": "keyword"},
            "paraphrase": {"query_id": "paraphrase", "answer": "Facilities", "query_format": "question"},
            "missing": {"query_id": "missing", "answer": "missing", "query_format": "question"},
            "anchors": {"query_id": "anchors", "answer": "42%", "query_format": "question"},
            "regression": {"query_id": "regression", "answer": "semantic", "query_format": "instruction"},
        },
        corpus_rows={
            "doc_1": {"text": "The rare identifier is ZX-9001."},
            "doc_2": {"text": "Facilities reduced electricity consumption."},
            "doc_3": {"text": "missing reference page"},
            "doc_4": {"text": "The result was 42%."},
            "doc_5": {"text": "A semantic retrieval policy."},
        },
    )
    indexed_text = {
        "doc_1": ["Identifier ZX-9001 appears here."],
        "doc_2": ["Facilities reduced electricity consumption."],
        "doc_4": ["unrelated extraction without the answer"],
        "doc_5": ["semantic retrieval policy"],
    }
    rankings = {
        "dense": [[], ["doc_2"], [], ["doc_4"], ["doc_5"]],
        "sparse": [["doc_1"], [], [], ["doc_4"], []],
        "hybrid": [["doc_1"], ["doc_2"], [], ["doc_4"], []],
    }
    mode_results = {
        mode: [_result(query_id, mode, docs) for query_id, docs in zip(query_ids, mode_rankings)]
        for mode, mode_rankings in rankings.items()
    }

    summary = build_comparison_artifacts(
        output_dir=tmp_path,
        dataset=dataset,
        metadata=metadata,
        indexed_text=indexed_text,
        mode_results=mode_results,
        doc_id_field="pdf_page",
        provenance={"dataset": "tiny", "lancedb_version": "test"},
    )

    assert summary["query_count"] == 5
    assert summary["case_counts"] == {
        "all_modes_miss": 1,
        "extraction_concern": 1,
        "hybrid_regression": 1,
        "hybrid_rescue": 1,
        "other": 1,
    }
    audit = json.loads((tmp_path / "extraction_audit.json").read_text())
    assert audit["classified_positive_qrels"] == 5
    assert audit["status_counts"]["missing_indexed_page"] == 1
    assert audit["status_counts"]["needs_review"] == 1
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "query_cases.jsonl").exists()
    assert (tmp_path / "report.md").exists()
    for mode in rankings:
        assert (tmp_path / mode / "query_results.jsonl").exists()
        assert (tmp_path / mode / "beir_metrics.json").exists()
        assert (tmp_path / mode / "beir_run.trec").exists()


def test_compare_retrieval_replays_hybrid_even_when_source_run_was_hybrid(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps({"status": "complete"}))
    (run_dir / "results.json").write_text(json.dumps({"success": True, "dry_run": False, "mode": "local"}))
    (run_dir / "resolved_benchmark.json").write_text(
        json.dumps({"evaluation": {"loader": "vidore_hf", "dataset_name": "tiny", "split": "test"}})
    )
    (run_dir / "query_plan.json").write_text(
        json.dumps(
            {
                "lancedb_uri": str(tmp_path / "lancedb"),
                "table_name": "tiny",
                "retrieval_mode": "hybrid",
                "rerank": False,
            }
        )
    )
    table = object()
    monkeypatch.setattr(
        retrieval_comparison.lancedb,
        "connect",
        lambda _uri: SimpleNamespace(open_table=lambda _table_name: table),
    )
    monkeypatch.setattr(
        retrieval_comparison,
        "inspect_lancedb_table_object",
        lambda _table: SimpleNamespace(has_vector=True, has_fts=True, retrieval_mode="hybrid"),
    )
    dataset = BeirDataset(dataset_name="tiny", query_ids=["q1"], queries=["query"], qrels={"q1": {"d1": 1}})
    monkeypatch.setattr(retrieval_comparison, "load_beir_dataset", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(
        retrieval_comparison,
        "_load_vidore_metadata",
        lambda *_args, **_kwargs: LoadedVidoreMetadata(query_rows={}, corpus_rows={}),
    )
    monkeypatch.setattr(retrieval_comparison, "_load_indexed_text", lambda *_args, **_kwargs: {})
    resolved_plan = object()
    monkeypatch.setattr(retrieval_comparison, "_resolved_query_plan", lambda _payload, _mode: resolved_plan)
    calls = []
    replayed = [{"query_id": "q1", "query": "query", "mode": "hybrid", "latency_ms": 1.0, "hits": []}]

    def replay(_dataset, plan, mode):
        calls.append((plan, mode))
        return replayed

    monkeypatch.setattr(retrieval_comparison, "_run_mode_queries", replay)
    monkeypatch.setattr(
        retrieval_comparison,
        "_read_query_results",
        lambda *_args, **_kwargs: pytest.fail("source hybrid results must not be reused"),
    )
    monkeypatch.setattr(
        retrieval_comparison,
        "build_comparison_artifacts",
        lambda **kwargs: {"mode_results": kwargs["mode_results"]},
    )

    result = compare_retrieval(run_dir, modes=("hybrid",), output_dir=tmp_path / "comparison")

    assert calls == [(resolved_plan, "hybrid")]
    assert result["mode_results"]["hybrid"] == replayed
