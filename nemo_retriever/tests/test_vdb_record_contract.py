# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the canonical VDB record contract and LanceDB store helpers.

These tests validate:
  - build_vdb_records produces the correct canonical record format
  - build_vdb_records_from_dicts (transitional list[dict] path) matches
  - _ensure_dict handles Arrow serialization robustness
"""

from __future__ import annotations

import copy
import json

import pandas as pd


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
# _ensure_dict — Arrow serialization robustness
# ---------------------------------------------------------------------------


class TestEnsureDict:
    """Tests for the _ensure_dict helper that handles Arrow-serialized dict columns."""

    def test_dict_passthrough(self):
        from nemo_retriever.vector_store.lancedb_utils import _ensure_dict

        d = {"a": 1}
        assert _ensure_dict(d) is d

    def test_json_string_parsed(self):
        from nemo_retriever.vector_store.lancedb_utils import _ensure_dict

        assert _ensure_dict('{"a": 1}') == {"a": 1}

    def test_none_returns_none(self):
        from nemo_retriever.vector_store.lancedb_utils import _ensure_dict

        assert _ensure_dict(None) is None

    def test_non_dict_json_returns_none(self):
        from nemo_retriever.vector_store.lancedb_utils import _ensure_dict

        assert _ensure_dict("[1, 2, 3]") is None

    def test_malformed_json_returns_none(self):
        from nemo_retriever.vector_store.lancedb_utils import _ensure_dict

        assert _ensure_dict("not json{") is None

    def test_integer_returns_none(self):
        from nemo_retriever.vector_store.lancedb_utils import _ensure_dict

        assert _ensure_dict(42) is None


class TestBuildVdbRecordsArrowCompat:
    """build_vdb_records handles string-encoded dict columns from Arrow."""

    def test_string_metadata_with_embedding(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        embedding = [0.1, 0.2, 0.3, 0.4]
        metadata = json.dumps(
            {
                "embedding": embedding,
                "source_path": "/data/test.pdf",
                "content_metadata": {"hierarchy": {"page": 0}},
            }
        )
        df = pd.DataFrame(
            [
                {
                    "metadata": metadata,
                    "text_embeddings_1b_v2": json.dumps({"embedding": embedding}),
                    "text": "hello world",
                    "path": "/data/test.pdf",
                    "page_number": 0,
                }
            ]
        )
        rows = build_vdb_records(df)
        assert len(rows) == 1
        assert rows[0]["vector"] == embedding

    def test_string_embed_column_only(self):
        from nemo_retriever.vector_store.vdb_records import build_vdb_records

        embedding = [1.0, 2.0]
        df = pd.DataFrame(
            [
                {
                    "metadata": "{}",
                    "text_embeddings_1b_v2": json.dumps({"embedding": embedding}),
                    "text": "test",
                    "path": "/x.pdf",
                    "page_number": 0,
                }
            ]
        )
        rows = build_vdb_records(df)
        assert len(rows) == 1
        assert rows[0]["vector"] == embedding
