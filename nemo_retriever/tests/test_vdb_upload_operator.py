# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VDBUploadOperator — the graph-based VDB write path."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_retriever.graph.vdb_upload_operator import VDBUploadOperator
from nemo_retriever.params.models import LanceDbParams, VdbUploadParams


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_embedded_df(n: int = 3, dim: int = 4) -> pd.DataFrame:
    """Build a minimal post-embed DataFrame with *n* rows."""
    rows = []
    for i in range(n):
        embedding = [float(i + j) for j in range(dim)]
        metadata = {
            "embedding": embedding,
            "source_path": f"/data/doc_{i}.pdf",
            "content_metadata": {"hierarchy": {"page": i}},
        }
        rows.append(
            {
                "metadata": metadata,
                "text_embeddings_1b_v2": {"embedding": embedding, "info_msg": None},
                "text": f"content of page {i}",
                "path": f"/data/doc_{i}.pdf",
                "page_number": i,
                "document_type": "text",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def lance_params(tmp_path):
    return LanceDbParams(lancedb_uri=str(tmp_path / "test_lancedb"), table_name="test_table")


@pytest.fixture()
def vdb_params(lance_params):
    return VdbUploadParams(lancedb=lance_params)


# ---------------------------------------------------------------------------
# LanceDB write path tests
# ---------------------------------------------------------------------------


class TestVDBUploadOperator:
    def test_writes_records_to_lancedb(self, vdb_params, lance_params):
        """Operator writes canonical VDB records during process()."""
        df = _make_embedded_df(3)
        op = VDBUploadOperator(params=vdb_params)
        result = op.run(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

        import lancedb

        db = lancedb.connect(lance_params.lancedb_uri)
        table = db.open_table(lance_params.table_name)
        assert table.count_rows() == 3

    def test_multiple_batches_accumulate(self, vdb_params, lance_params):
        """Multiple process() calls append rows, not overwrite."""
        op = VDBUploadOperator(params=vdb_params)
        op.run(_make_embedded_df(2))
        op.run(_make_embedded_df(3))

        import lancedb

        db = lancedb.connect(lance_params.lancedb_uri)
        table = db.open_table(lance_params.table_name)
        assert table.count_rows() == 5

    def test_empty_dataframe_is_noop(self, vdb_params):
        """Empty DataFrame doesn't create a table or crash."""
        op = VDBUploadOperator(params=vdb_params)
        result = op.run(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert op._client_vdb is None

    def test_no_embeddings_is_noop(self, vdb_params):
        """DataFrame without embedding columns produces no VDB records."""
        df = pd.DataFrame({"text": ["hello"], "path": ["/test.txt"]})
        op = VDBUploadOperator(params=vdb_params)
        result = op.run(df)
        assert len(result) == 1
        assert op._client_vdb is None

    def test_accepts_lance_params_directly(self, lance_params):
        """Operator accepts LanceDbParams in addition to VdbUploadParams."""
        df = _make_embedded_df(1)
        op = VDBUploadOperator(params=lance_params)
        op.run(df)

        import lancedb

        db = lancedb.connect(lance_params.lancedb_uri)
        table = db.open_table(lance_params.table_name)
        assert table.count_rows() == 1

    def test_default_params(self):
        """Operator works with no params (uses defaults)."""
        op = VDBUploadOperator()
        assert op._lance_params is not None

    def test_preprocess_extracts_records(self, vdb_params):
        """preprocess populates _pending_records from the DataFrame."""
        df = _make_embedded_df(3)
        op = VDBUploadOperator(params=vdb_params)
        result = op.preprocess(df)

        assert result is df
        assert len(op._pending_records) == 3
        assert all("vector" in r for r in op._pending_records)


# ---------------------------------------------------------------------------
# Arrow serialization compat
# ---------------------------------------------------------------------------


class TestArrowSerializationCompat:
    """Regression tests for Arrow-serialized dict columns (Ray Data pipeline)."""

    def _make_string_encoded_df(self, n: int = 3, dim: int = 4) -> pd.DataFrame:
        """Build a DataFrame where dict columns are JSON strings, simulating Arrow round-trip."""
        rows = []
        for i in range(n):
            embedding = [float(i + j) for j in range(dim)]
            metadata = json.dumps(
                {
                    "embedding": embedding,
                    "source_path": f"/data/doc_{i}.pdf",
                    "content_metadata": {"hierarchy": {"page": i}},
                }
            )
            embed_payload = json.dumps({"embedding": embedding, "info_msg": None})
            rows.append(
                {
                    "metadata": metadata,
                    "text_embeddings_1b_v2": embed_payload,
                    "text": f"content of page {i}",
                    "path": f"/data/doc_{i}.pdf",
                    "page_number": i,
                    "document_type": "text",
                }
            )
        return pd.DataFrame(rows)

    def test_writes_records_with_string_metadata(self, vdb_params, lance_params):
        """Operator handles Arrow-serialized string columns (not dicts)."""
        df = self._make_string_encoded_df(3)
        op = VDBUploadOperator(params=vdb_params)
        result = op.run(df)

        assert len(result) == 3

        import lancedb

        db = lancedb.connect(lance_params.lancedb_uri)
        table = db.open_table(lance_params.table_name)
        assert table.count_rows() == 3

    def test_writes_records_with_string_embed_column_only(self, vdb_params, lance_params):
        """Embedding extracted from string-encoded embedding column."""
        embedding = [1.0, 2.0, 3.0, 4.0]
        df = pd.DataFrame(
            [
                {
                    "metadata": {"source_path": "/test.pdf"},
                    "text_embeddings_1b_v2": json.dumps({"embedding": embedding}),
                    "text": "hello",
                    "path": "/test.pdf",
                    "page_number": 0,
                }
            ]
        )
        op = VDBUploadOperator(params=vdb_params)
        op.run(df)

        import lancedb

        db = lancedb.connect(lance_params.lancedb_uri)
        table = db.open_table(lance_params.table_name)
        assert table.count_rows() == 1


# ---------------------------------------------------------------------------
# Milvus write path (mocked client)
# ---------------------------------------------------------------------------


class TestVDBUploadMilvus:
    """VDBUploadOperator wrapping a mocked Milvus client VDB."""

    def _make_milvus_params(self):
        return VdbUploadParams(backend="milvus", client_vdb_kwargs={"collection_name": "test"})

    def _make_mock_client(self):
        mock_client = MagicMock()
        mock_client.collection_name = "test"
        mock_client.get_connection_params.return_value = ("test", {"milvus_uri": "http://localhost:19530"})
        mock_client.get_write_params.return_value = ("test", {"collection_name": "test"})
        return mock_client

    @patch("nemo_retriever.graph.vdb_upload_operator.get_vdb_op_cls", create=True)
    def test_delegates_to_client_write_to_index(self, mock_get_cls):
        mock_client = self._make_mock_client()
        mock_get_cls.return_value = lambda **kwargs: mock_client

        params = self._make_milvus_params()
        op = VDBUploadOperator(params=params)
        df = _make_embedded_df(3)

        with patch("nv_ingest_client.util.vdb.get_vdb_op_cls", return_value=lambda **kw: mock_client):
            op.run(df)

        mock_client.create_index.assert_called_once()
        mock_client.write_to_index.assert_called_once()
        call_args = mock_client.write_to_index.call_args[0][0]
        assert isinstance(call_args, list)
        assert isinstance(call_args[0], list)
        assert call_args[0][0]["document_type"] == "text"
        assert "embedding" in call_args[0][0]["metadata"]

    @patch("nemo_retriever.graph.vdb_upload_operator.get_vdb_op_cls", create=True)
    def test_multiple_batches_call_write_per_batch(self, mock_get_cls):
        mock_client = self._make_mock_client()
        mock_get_cls.return_value = lambda **kwargs: mock_client

        params = self._make_milvus_params()
        op = VDBUploadOperator(params=params)

        with patch("nv_ingest_client.util.vdb.get_vdb_op_cls", return_value=lambda **kw: mock_client):
            op.run(_make_embedded_df(2))
            op.run(_make_embedded_df(3))

        mock_client.create_index.assert_called_once()
        assert mock_client.write_to_index.call_count == 2


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


class TestVDBFinalization:
    def test_lancedb_table_accessible_after_writes(self, vdb_params, lance_params):
        """After operator writes, the table can be opened from disk."""
        df = _make_embedded_df(5)
        op = VDBUploadOperator(params=vdb_params)
        op.run(df)

        import lancedb

        db = lancedb.connect(lance_params.lancedb_uri)
        table = db.open_table(lance_params.table_name)
        assert table.count_rows() == 5
