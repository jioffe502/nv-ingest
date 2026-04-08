# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the canonical VDB record contract, VectorStore ABC, and LanceDB backend.

These tests validate:
  - build_vdb_records produces the correct canonical record format
  - build_vdb_records_from_dicts (transitional list[dict] path) matches
  - VectorStore ABC enforces the interface contract
  - LanceDBBackend round-trips data correctly
  - handle_lancedb now uses the canonical builder (regression)
"""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _make_sample_dataframe() -> pd.DataFrame:
    """Build a minimal DataFrame matching the graph pipeline's post-embed output."""
    embedding = [0.1, 0.2, 0.3, 0.4]
    metadata = {
        "embedding": embedding,
        "source_path": "/data/test.pdf",
        "content_metadata": {"hierarchy": {"page": 0}},
    }
    return pd.DataFrame(
        [
            {
                "metadata": metadata,
                "text_embeddings_1b_v2": {"embedding": embedding, "info_msg": None},
                "text": "Hello world",
                "path": "/data/test.pdf",
                "page_number": 0,
                "page_elements_v3_num_detections": 5,
                "page_elements_v3_counts_by_label": {"text": 3, "table": 2},
            }
        ]
    )


# ---------------------------------------------------------------------------
# Canonical record builder tests
# ---------------------------------------------------------------------------


class TestBuildVdbRecords:
    def test_produces_all_required_fields(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        df = _make_sample_dataframe()
        rows = build_vdb_records(df)

        assert len(rows) == 1
        row = rows[0]
        for field in (
            "vector",
            "text",
            "metadata",
            "source",
            "page_number",
            "pdf_page",
            "pdf_basename",
            "source_id",
            "path",
            "filename",
        ):
            assert field in row, f"Missing required field: {field}"

    def test_metadata_is_valid_json(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        df = _make_sample_dataframe()
        rows = build_vdb_records(df)

        meta = json.loads(rows[0]["metadata"])
        assert isinstance(meta, dict)
        assert "page_number" in meta

    def test_metadata_includes_detection_counts(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        df = _make_sample_dataframe()
        rows = build_vdb_records(df)

        meta = json.loads(rows[0]["metadata"])
        assert meta["page_elements_v3_num_detections"] == 5
        assert meta["page_elements_v3_counts_by_label"] == {"text": 3, "table": 2}

    def test_source_is_json_object(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        df = _make_sample_dataframe()
        rows = build_vdb_records(df)

        source = json.loads(rows[0]["source"])
        assert isinstance(source, dict)
        assert "source_id" in source

    def test_does_not_mutate_input(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        df = _make_sample_dataframe()
        original_meta = copy.deepcopy(df.iloc[0]["metadata"])

        build_vdb_records(df)

        assert df.iloc[0]["metadata"] == original_meta

    def test_vector_is_embedding(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        df = _make_sample_dataframe()
        rows = build_vdb_records(df)

        assert rows[0]["vector"] == [0.1, 0.2, 0.3, 0.4]

    def test_skips_rows_without_embedding(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        df = pd.DataFrame(
            [
                {
                    "metadata": {"source_path": "/data/test.pdf"},
                    "text": "No embedding here",
                    "path": "/data/test.pdf",
                    "page_number": 0,
                }
            ]
        )
        rows = build_vdb_records(df)
        assert len(rows) == 0

    def test_empty_dataframe(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        df = pd.DataFrame()
        rows = build_vdb_records(df)
        assert rows == []

    def test_text_content(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        df = _make_sample_dataframe()
        rows = build_vdb_records(df)
        assert rows[0]["text"] == "Hello world"

    def test_include_text_false(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        df = _make_sample_dataframe()
        rows = build_vdb_records(df, include_text=False)
        assert rows[0]["text"] == ""


# ---------------------------------------------------------------------------
# Transitional list[dict] builder tests
# ---------------------------------------------------------------------------


class TestBuildVdbRecordsFromDicts:
    def test_matches_dataframe_path(self):
        """list[dict] path should produce identical output to DataFrame path."""
        from nemo_retriever.vector_store.vdb_records import build_vdb_records, build_vdb_records_from_dicts

        df = _make_sample_dataframe()
        records = df.to_dict("records")

        from_df = build_vdb_records(df)
        from_dicts = build_vdb_records_from_dicts(records)

        assert len(from_df) == len(from_dicts)
        assert from_df[0]["vector"] == from_dicts[0]["vector"]
        assert from_df[0]["text"] == from_dicts[0]["text"]
        assert from_df[0]["path"] == from_dicts[0]["path"]
        assert from_df[0]["page_number"] == from_dicts[0]["page_number"]

    def test_empty_list(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records_from_dicts

        assert build_vdb_records_from_dicts([]) == []


# ---------------------------------------------------------------------------
# VectorStore ABC tests
# ---------------------------------------------------------------------------


class TestVectorStoreABC:
    def test_cannot_instantiate_directly(self):
        from nemo_retriever.vector_store.vdb import VectorStore

        with pytest.raises(TypeError):
            VectorStore()

    def test_subclass_must_implement_all_methods(self):
        from nemo_retriever.vector_store.vdb import VectorStore

        class Incomplete(VectorStore):
            def create_table(self, *, dim, **kwargs):
                pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_lancedb_backend_is_valid_subclass(self):
        from nemo_retriever.vector_store.lancedb_backend import LanceDBBackend
        from nemo_retriever.vector_store.vdb import VectorStore

        assert issubclass(LanceDBBackend, VectorStore)


# ---------------------------------------------------------------------------
# LanceDB backend tests
# ---------------------------------------------------------------------------


class TestLanceDBBackend:
    def test_write_rows_creates_table_lazily(self, tmp_path):
        from nemo_retriever.params.models import LanceDbParams
        from nemo_retriever.vector_store.lancedb_backend import LanceDBBackend
        import lancedb

        params = LanceDbParams(lancedb_uri=str(tmp_path / "test_db"), table_name="test_table", create_index=False)
        backend = LanceDBBackend(params)

        rows = [
            {
                "vector": [0.1, 0.2, 0.3],
                "text": "hello",
                "metadata": "{}",
                "source": "{}",
                "page_number": 0,
                "pdf_page": "",
                "pdf_basename": "",
                "filename": "",
                "source_id": "",
                "path": "",
            }
        ]
        backend.write_rows(rows)

        db = lancedb.connect(str(tmp_path / "test_db"))
        table = db.open_table("test_table")
        assert table.count_rows() == 1

    def test_multiple_writes_accumulate(self, tmp_path):
        from nemo_retriever.params.models import LanceDbParams
        from nemo_retriever.vector_store.lancedb_backend import LanceDBBackend
        import lancedb

        params = LanceDbParams(lancedb_uri=str(tmp_path / "test_db"), table_name="test_table", create_index=False)
        backend = LanceDBBackend(params)

        row_template = {
            "vector": [0.1, 0.2, 0.3],
            "text": "hello",
            "metadata": "{}",
            "source": "{}",
            "page_number": 0,
            "pdf_page": "",
            "pdf_basename": "",
            "filename": "",
            "source_id": "",
            "path": "",
        }

        backend.write_rows([row_template])
        backend.write_rows([row_template, row_template])

        db = lancedb.connect(str(tmp_path / "test_db"))
        table = db.open_table("test_table")
        assert table.count_rows() == 3

    def test_empty_writes_are_noop(self, tmp_path):
        from nemo_retriever.params.models import LanceDbParams
        from nemo_retriever.vector_store.lancedb_backend import LanceDBBackend

        params = LanceDbParams(lancedb_uri=str(tmp_path / "test_db"), table_name="test_table", create_index=False)
        backend = LanceDBBackend(params)

        backend.write_rows([])
        assert backend._table is None

    def test_create_index_noop_when_no_writes(self, tmp_path):
        from nemo_retriever.params.models import LanceDbParams
        from nemo_retriever.vector_store.lancedb_backend import LanceDBBackend

        params = LanceDbParams(lancedb_uri=str(tmp_path / "test_db"), table_name="test_table")
        backend = LanceDBBackend(params)
        backend.create_index()  # should not raise


# ---------------------------------------------------------------------------
# Regression: handle_lancedb now uses canonical builder
# ---------------------------------------------------------------------------


class TestHandleLancedbRegression:
    def test_handle_lancedb_writes_valid_json_metadata(self, tmp_path):
        """After refactoring, handle_lancedb should produce valid JSON metadata."""
        from nemo_retriever.vector_store.lancedb_store import handle_lancedb
        import lancedb

        df = _make_sample_dataframe()
        rows = df.to_dict("records")

        uri = str(tmp_path / "test_db")
        handle_lancedb(rows, uri, "test_table", mode="overwrite")

        db = lancedb.connect(uri)
        table = db.open_table("test_table")
        result = table.to_pandas()

        assert len(result) == 1
        meta_str = result.iloc[0]["metadata"]
        meta = json.loads(meta_str)
        assert isinstance(meta, dict)
        assert "page_number" in meta

    def test_handle_lancedb_accepts_dataframe(self, tmp_path):
        """handle_lancedb should now accept a DataFrame directly."""
        from nemo_retriever.vector_store.lancedb_store import handle_lancedb
        import lancedb

        df = _make_sample_dataframe()

        uri = str(tmp_path / "test_db")
        handle_lancedb(df, uri, "test_table", mode="overwrite")

        db = lancedb.connect(uri)
        table = db.open_table("test_table")
        assert table.count_rows() == 1

    def test_handle_lancedb_round_trip_preserves_text(self, tmp_path):
        """Text content should survive the write→read round-trip."""
        from nemo_retriever.vector_store.lancedb_store import handle_lancedb
        import lancedb

        df = _make_sample_dataframe()
        rows = df.to_dict("records")

        uri = str(tmp_path / "test_db")
        handle_lancedb(rows, uri, "test_table", mode="overwrite")

        db = lancedb.connect(uri)
        table = db.open_table("test_table")
        result = table.to_pandas()

        assert result.iloc[0]["text"] == "Hello world"

    def test_handle_lancedb_round_trip_preserves_path(self, tmp_path):
        from nemo_retriever.vector_store.lancedb_store import handle_lancedb
        import lancedb

        df = _make_sample_dataframe()
        rows = df.to_dict("records")

        uri = str(tmp_path / "test_db")
        handle_lancedb(rows, uri, "test_table", mode="overwrite")

        db = lancedb.connect(uri)
        table = db.open_table("test_table")
        result = table.to_pandas()

        assert result.iloc[0]["path"] == "/data/test.pdf"
        assert result.iloc[0]["page_number"] == 0
