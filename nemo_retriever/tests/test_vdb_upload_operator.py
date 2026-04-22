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
from nv_ingest_client.util.vdb.adt_vdb import VDB


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


class CustomVDB(VDB):
    """Minimal custom backend proving the operator uses the legacy VDB ADT."""

    def __init__(self):
        super().__init__()
        self.create_index_calls = []
        self.write_to_index_calls = []

    def create_index(self, **kwargs):
        self.create_index_calls.append(kwargs)

    def write_to_index(self, records: list, **kwargs):
        self.write_to_index_calls.append((records, kwargs))

    def retrieval(self, queries: list, **kwargs):
        return []

    def run(self, records):
        self.create_index()
        self.write_to_index(records)


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
    """VDBUploadOperator wrapping a Milvus client VDB."""

    def _make_milvus_vdb(self):
        from nv_ingest_client.util.vdb import get_vdb_op_cls

        Milvus = get_vdb_op_cls("milvus")
        return Milvus(
            milvus_uri="http://localhost:19530",
            collection_name="test",
            dense_dim=4,
            recreate=True,
            gpu_index=False,
            sparse=False,
        )

    def test_streams_directly_to_milvus_client(self):
        milvus_vdb = self._make_milvus_vdb()
        op = VDBUploadOperator(vdb_op=milvus_vdb)
        fake_client = MagicMock()
        fake_client_cls = MagicMock(return_value=fake_client)
        create_collection = MagicMock()
        cleanup_records = MagicMock(
            return_value=[
                {"text": "content 0", "vector": [0.0, 1.0, 2.0, 3.0]},
                {"text": "content 1", "vector": [1.0, 2.0, 3.0, 4.0]},
                {"text": "content 2", "vector": [2.0, 3.0, 4.0, 5.0]},
            ]
        )

        with patch.object(
            VDBUploadOperator,
            "_load_milvus_write_helpers",
            return_value=(fake_client_cls, cleanup_records, create_collection, MagicMock()),
        ):
            op.run(_make_embedded_df(3))

        create_collection.assert_called_once()
        fake_client_cls.assert_called_once_with(uri="http://localhost:19530")
        cleanup_records.assert_called_once()
        fake_client.insert.assert_called_once()
        assert fake_client.insert.call_args.kwargs["collection_name"] == "test"
        assert len(fake_client.insert.call_args.kwargs["data"]) == 3

    def test_multiple_batches_reuse_milvus_client(self):
        milvus_vdb = self._make_milvus_vdb()
        op = VDBUploadOperator(vdb_op=milvus_vdb)
        fake_client = MagicMock()
        fake_client_cls = MagicMock(return_value=fake_client)
        create_collection = MagicMock()
        cleanup_records = MagicMock(return_value=[{"text": "content", "vector": [0.0, 1.0, 2.0, 3.0]}])

        with patch.object(
            VDBUploadOperator,
            "_load_milvus_write_helpers",
            return_value=(fake_client_cls, cleanup_records, create_collection, MagicMock()),
        ):
            op.run(_make_embedded_df(2))
            op.run(_make_embedded_df(3))

        create_collection.assert_called_once()
        fake_client_cls.assert_called_once_with(uri="http://localhost:19530")
        assert fake_client.insert.call_count == 2

    def test_milvus_sparse_streaming_is_explicitly_unsupported(self):
        milvus_vdb = self._make_milvus_vdb()
        milvus_vdb.sparse = True
        op = VDBUploadOperator(vdb_op=milvus_vdb)

        with patch.object(
            VDBUploadOperator,
            "_load_milvus_write_helpers",
            return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()),
        ):
            with pytest.raises(NotImplementedError, match="sparse/hybrid"):
                op.run(_make_embedded_df(1))


# ---------------------------------------------------------------------------
# Pre-constructed VDB injection
# ---------------------------------------------------------------------------


class TestPreConstructedVDB:
    """Operator accepts a pre-built client VDB instance (lead's review ask)."""

    def test_accepts_client_lancedb_instance(self, lance_params):
        """Passing a client LanceDB object skips internal construction."""
        from nv_ingest_client.util.vdb import get_vdb_op_cls

        LanceDB = get_vdb_op_cls("lancedb")
        client_vdb = LanceDB(
            uri=lance_params.lancedb_uri,
            table_name=lance_params.table_name,
            overwrite=True,
        )

        op = VDBUploadOperator(vdb_op=client_vdb)
        assert op._client_vdb is client_vdb
        assert op._backend_name == "lancedb"

        op.run(_make_embedded_df(2))

        import lancedb

        db = lancedb.connect(lance_params.lancedb_uri)
        table = db.open_table(lance_params.table_name)
        assert table.count_rows() == 2

    def test_accepts_arbitrary_vdb_instance(self):
        """A non-LanceDB custom instance routes through ADT methods only."""
        mock_vdb = CustomVDB()

        op = VDBUploadOperator(vdb_op=mock_vdb)
        assert op._backend_name == "customvdb"
        assert op._client_vdb is mock_vdb

        op.run(_make_embedded_df(2))

        assert mock_vdb.create_index_calls == [{}]
        assert len(mock_vdb.write_to_index_calls) == 1
        written_records, write_kwargs = mock_vdb.write_to_index_calls[0]
        assert write_kwargs == {}
        assert isinstance(written_records, list)
        assert isinstance(written_records[0], list)
        assert written_records[0][0]["document_type"] == "text"
        assert not hasattr(mock_vdb, "get_connection_params")
        assert not hasattr(mock_vdb, "get_write_params")

    def test_constructor_kwargs_round_trip(self):
        """get_constructor_kwargs captures both params and vdb_op for Ray reconstruction."""
        mock_vdb = CustomVDB()

        op = VDBUploadOperator(params=VdbUploadParams(), vdb_op=mock_vdb)
        kwargs = op.get_constructor_kwargs()
        assert "vdb_op" in kwargs
        assert kwargs["vdb_op"] is mock_vdb
        assert "params" in kwargs


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

    def test_lancedb_finalize_is_noop(self, lance_params):
        from nv_ingest_client.util.vdb import get_vdb_op_cls

        LanceDB = get_vdb_op_cls("lancedb")
        op = VDBUploadOperator(vdb_op=LanceDB(uri=lance_params.lancedb_uri, table_name=lance_params.table_name))

        with patch.object(op, "_finalize_milvus") as finalize_milvus:
            op.finalize()

        finalize_milvus.assert_not_called()

    def test_milvus_finalize_waits_once(self):
        from nv_ingest_client.util.vdb import get_vdb_op_cls

        Milvus = get_vdb_op_cls("milvus")
        milvus_vdb = Milvus(
            milvus_uri="http://localhost:19530",
            collection_name="test",
            dense_dim=4,
            recreate=True,
            gpu_index=False,
            sparse=False,
        )
        op = VDBUploadOperator(vdb_op=milvus_vdb)
        fake_client = MagicMock()
        fake_client.has_collection.return_value = True
        fake_client.get_collection_stats.return_value = {"row_count": 7}
        fake_client.list_indexes.return_value = ["dense_index", "sparse_index"]
        fake_client_cls = MagicMock(return_value=fake_client)
        wait_for_index = MagicMock()

        with patch.object(
            VDBUploadOperator,
            "_load_milvus_finalize_helpers",
            return_value=(fake_client_cls, wait_for_index),
        ):
            op.finalize()

        fake_client_cls.assert_called_once_with(uri="http://localhost:19530")
        fake_client.flush.assert_called_once_with("test")
        wait_for_index.assert_called_once_with(
            "test",
            {"dense_index": 7, "sparse_index": 7},
            fake_client,
        )

    def test_milvus_finalize_constructs_params_backend(self):
        params = VdbUploadParams(backend="milvus", client_vdb_kwargs={"collection_name": "test"})
        op = VDBUploadOperator(params=params)
        fake_vdb = MagicMock()
        fake_cls = MagicMock(return_value=fake_vdb)

        with (
            patch("nv_ingest_client.util.vdb.get_vdb_op_cls", return_value=fake_cls),
            patch.object(VDBUploadOperator, "_is_milvus_vdb", return_value=True),
            patch.object(op, "_finalize_milvus") as finalize_milvus,
        ):
            op.finalize()

        fake_cls.assert_called_once_with(collection_name="test")
        assert op._client_vdb is fake_vdb
        finalize_milvus.assert_called_once_with()
