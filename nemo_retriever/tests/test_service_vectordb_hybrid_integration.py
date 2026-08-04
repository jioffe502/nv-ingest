# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from nemo_retriever.service import vectordb_app as vectordb_module
from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app

import lancedb

_DIM = 4
_ROW = {
    "vector": [1.0, 0.0, 0.0, 0.0],
    "pdf_page": "10k_2023_12",
    "filename": "10k_2023.pdf",
    "pdf_basename": "10k_2023.pdf",
    "page_number": 12,
    "source": "10k_2023.pdf",
    "source_id": "10k_2023.pdf",
    "path": "/data/10k_2023.pdf",
    "text": "Revenue grew 12% year over year.",
    "metadata": json.dumps({"page_number": 12, "type": "text"}),
    "stored_image_uri": "",
    "content_type": "text",
    "bbox_xyxy_norm": "",
}


def _state(tmp_path, *, index_mode="auto") -> VectorDBState:
    return VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
        index_mode=index_mode,
    )


def _prebuild_fts_index(uri: str, table_name: str) -> None:
    """Simulate an ingestion pipeline that wrote a table with a BM25/FTS index.

    This also supports tests that seed the pre-migration storage shape.
    """
    table = lancedb.connect(uri).open_table(table_name)
    table.create_fts_index("text", replace=True)
    for stub in table.list_indices():
        if "text" in stub.name.lower() or "fts" in stub.name.lower():
            table.wait_for_index([stub.name], timeout=timedelta(seconds=600))


@pytest.mark.integration
def test_fresh_auto_write_builds_fts(tmp_path) -> None:
    state = _state(tmp_path)
    assert state.write_rows([_ROW]) == 1

    caps = state._table_capabilities()
    assert caps is not None
    assert caps.has_vector
    assert caps.has_fts
    assert state.resolve_effective_retrieval_mode() == "hybrid"


@pytest.mark.integration
def test_append_keeps_hybrid_fts_and_unindexed_tail_searchable(tmp_path) -> None:
    state = _state(tmp_path)
    assert state.write_rows([_ROW]) == 1

    appended = dict(_ROW)
    appended["vector"] = [0.0, 1.0, 0.0, 0.0]
    appended["text"] = "Zephyr quarterly guidance mentions unicorn synergy."
    assert state.write_rows([appended]) == 1

    table = state._db.open_table("nemo_retriever")
    assert table.count_rows() == 2
    caps = state._table_capabilities()
    assert caps.has_fts
    hits, strategies = state.search([[0.0, 1.0, 0.0, 0.0]], ["unicorn synergy"], top_k=5)
    assert strategies == ["hybrid"]
    assert any("Zephyr quarterly guidance" in hit["text"] for hit in hits[0])


@pytest.mark.integration
def test_incremental_write_threshold_runs_optimize_and_updates_health(tmp_path, monkeypatch) -> None:
    state = _state(tmp_path)
    state.write_rows([_ROW])
    table = state._db.open_table("nemo_retriever")
    monkeypatch.setattr(vectordb_module, "_OPTIMIZE_WRITE_THRESHOLD", 1)

    with patch.object(type(table), "optimize", autospec=True) as optimize:
        state.write_rows([{**_ROW, "text": "second row"}])

    optimize.assert_called_once()
    health = state.index_health()
    assert health["last_optimization"]["status"] == "ok"
    assert health["writes_since_optimize"] == 0
    assert health["rows_since_optimize"] == 0


@pytest.mark.integration
def test_auto_resolves_hybrid_when_fts_prebuilt(tmp_path) -> None:
    # Ingestion built the table with both a vector column and an FTS index.
    seed = _state(tmp_path, index_mode="dense")
    seed.write_rows([_ROW])
    _prebuild_fts_index(str(tmp_path), "nemo_retriever")

    state = _state(tmp_path)
    caps = state._table_capabilities()
    assert caps.has_vector
    assert caps.has_fts
    assert state.resolve_effective_retrieval_mode() == "hybrid"


@pytest.mark.integration
def test_auto_resolves_dense_when_no_fts(tmp_path) -> None:
    seed = _state(tmp_path, index_mode="dense")
    seed.write_rows([_ROW])

    state = _state(tmp_path)
    caps = state._table_capabilities()
    assert caps.has_vector
    assert not caps.has_fts
    assert state.resolve_effective_retrieval_mode() == "dense"


@pytest.mark.integration
def test_explicit_hybrid_upgrades_existing_dense_table(tmp_path) -> None:
    seed = _state(tmp_path, index_mode="dense")
    seed.write_rows([_ROW])

    upgraded = _state(tmp_path, index_mode="hybrid")
    caps = upgraded._table_capabilities()
    assert caps.has_vector
    assert caps.has_fts
    assert upgraded.resolve_effective_retrieval_mode() == "hybrid"


@pytest.mark.integration
def test_explicit_dense_rejects_existing_hybrid_table(tmp_path) -> None:
    seed = _state(tmp_path)
    seed.write_rows([_ROW])

    with pytest.raises(ValueError, match="Cannot append"):
        _state(tmp_path, index_mode="dense")


@pytest.mark.integration
def test_query_auto_selects_hybrid_for_fresh_default_table(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0, 0.0, 0.0]]):
        with TestClient(app) as client:
            write = client.post("/internal/vectordb/write", json={"rows": [_ROW]})
            assert write.status_code == 200, write.text

            resp = client.post(
                "/v1/query",
                json={"query": "revenue", "top_k": 5, "format": "evidence"},
            )

    assert resp.status_code == 200, resp.text
    coverage = resp.json()["results"][0]["coverage"]
    assert coverage["strategies_used"] == ["hybrid"]


@pytest.mark.integration
def test_query_auto_selects_dense_when_no_fts(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        index_mode="dense",
    )

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0, 0.0, 0.0]]):
        with TestClient(app) as client:
            write = client.post("/internal/vectordb/write", json={"rows": [_ROW]})
            assert write.status_code == 200, write.text

            resp = client.post(
                "/v1/query",
                json={"query": "revenue", "top_k": 5, "format": "evidence"},
            )

    assert resp.status_code == 200, resp.text
    coverage = resp.json()["results"][0]["coverage"]
    assert coverage["strategies_used"] == ["dense"]
