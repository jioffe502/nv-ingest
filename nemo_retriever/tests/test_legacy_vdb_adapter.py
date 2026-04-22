import json

import pandas as pd

from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.params import EmbedParams, LanceDbParams, VdbUploadParams
from nemo_retriever.vector_store.legacy_vdb_adapter import (
    dataframe_to_legacy_vdb_records,
    execution_result_to_legacy_vdb_records,
)


class FakeVDB:
    def __init__(self) -> None:
        self.calls = []

    def run(self, records):
        self.calls.append(records)
        return {"records": len(records[0])}


def test_dataframe_to_legacy_vdb_records_builds_required_shape_and_skips_missing_embeddings() -> None:
    df = pd.DataFrame(
        [
            {
                "text": "alpha",
                "path": "/tmp/a.pdf",
                "page_number": 3,
                "metadata": {
                    "content_metadata": {"section": "intro"},
                    "source_metadata": {"source_name": "a.pdf"},
                },
                "text_embeddings_1b_v2": {"embedding": [0.1, 0.2]},
            },
            {
                "text": "skip",
                "metadata": {"content_metadata": {}, "source_metadata": {}},
                "text_embeddings_1b_v2": {"embedding": []},
            },
            {
                "text": "also skip",
                "metadata": {"content_metadata": {}, "source_metadata": {}},
            },
        ]
    )

    records = dataframe_to_legacy_vdb_records(df)

    assert len(records) == 1
    assert len(records[0]) == 1
    record = records[0][0]
    assert record["document_type"] == "text"
    assert record["metadata"]["content"] == "alpha"
    assert record["metadata"]["embedding"] == [0.1, 0.2]
    assert record["metadata"]["content_metadata"]["section"] == "intro"
    assert record["metadata"]["content_metadata"]["page_number"] == 3
    assert record["metadata"]["source_metadata"]["source_name"] == "a.pdf"
    assert record["metadata"]["source_metadata"]["source_id"] == "/tmp/a.pdf"


def test_dataframe_to_legacy_vdb_records_accepts_json_metadata_and_embedding_payloads() -> None:
    df = pd.DataFrame(
        [
            {
                "text": "table text",
                "page_number": "7",
                "_content_type": "table_caption",
                "_bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4],
                "metadata": json.dumps(
                    {
                        "content_metadata": {"custom": "value"},
                        "source_metadata": {"source_name": "doc.pdf"},
                    }
                ),
                "text_embeddings_1b_v2": json.dumps({"embedding": [1.0, 2.0, 3.0]}),
            }
        ]
    )

    records = dataframe_to_legacy_vdb_records(df)

    record = records[0][0]
    assert record["document_type"] == "structured"
    assert record["metadata"]["content"] == "table text"
    assert record["metadata"]["embedding"] == [1.0, 2.0, 3.0]
    assert record["metadata"]["content_metadata"]["page_number"] == 7
    assert record["metadata"]["content_metadata"]["subtype"] == "table"
    assert record["metadata"]["table_metadata"]["table_content"] == "table text"
    assert record["metadata"]["table_metadata"]["table_location"] == [0.1, 0.2, 0.3, 0.4]
    assert record["metadata"]["source_metadata"]["source_name"] == "doc.pdf"


def test_execution_result_to_legacy_vdb_records_aggregates_ray_style_batches() -> None:
    class FakeDataset:
        def iter_batches(self, *, batch_format):
            assert batch_format == "pandas"
            yield pd.DataFrame(
                [
                    {
                        "text": "one",
                        "metadata": {"embedding": [1.0], "content_metadata": {}, "source_metadata": {}},
                    }
                ]
            )
            yield pd.DataFrame(
                [
                    {
                        "text": "two",
                        "metadata": {"embedding": [2.0], "content_metadata": {}, "source_metadata": {}},
                    }
                ]
            )

    records = execution_result_to_legacy_vdb_records(FakeDataset())

    assert [record["metadata"]["content"] for record in records[0]] == ["one", "two"]


def test_graph_ingestor_vdb_upload_invokes_legacy_vdb_op(monkeypatch) -> None:
    result_df = pd.DataFrame(
        [
            {
                "text": "embedded text",
                "metadata": {
                    "content_metadata": {"type": "text"},
                    "source_metadata": {"source_name": "doc.pdf"},
                },
                "custom_embeddings": {"embedding": [0.5, 0.25]},
            }
        ]
    )

    class FakeExecutor:
        def __init__(self, graph, **kwargs):
            self.graph = graph

        def ingest(self, data):
            return result_df

    monkeypatch.setattr("nemo_retriever.graph_ingestor.build_graph", lambda **kwargs: object())
    monkeypatch.setattr("nemo_retriever.graph_ingestor.InprocessExecutor", FakeExecutor)

    fake_vdb = FakeVDB()
    ingestor = GraphIngestor(run_mode="inprocess", documents=["/tmp/input.pdf"])
    result = (
        ingestor.embed(EmbedParams(output_column="custom_embeddings"))
        .vdb_upload(vdb_op=fake_vdb)
        .ingest()
    )

    assert result is result_df
    assert ingestor._vdb_op is fake_vdb
    assert len(fake_vdb.calls) == 1
    record = fake_vdb.calls[0][0][0]
    assert record["document_type"] == "text"
    assert record["metadata"]["content"] == "embedded text"
    assert record["metadata"]["embedding"] == [0.5, 0.25]
    assert record["metadata"]["content_metadata"] == {"type": "text"}
    assert record["metadata"]["source_metadata"] == {"source_name": "doc.pdf", "source_id": "doc.pdf"}


def test_graph_ingestor_default_vdb_upload_uses_native_lancedb_writer(monkeypatch) -> None:
    result_df = pd.DataFrame(
        [
            {
                "text": "retriever native text",
                "path": "/tmp/native.pdf",
                "page_number": 4,
                "metadata": {
                    "embedding": [0.1, 0.2],
                    "content_metadata": {"type": "text"},
                    "source_metadata": {"source_name": "native.pdf"},
                },
            }
        ]
    )

    class FakeExecutor:
        def __init__(self, graph, **kwargs):
            self.graph = graph

        def ingest(self, data):
            return result_df

    calls = []

    def fake_handle_lancedb(rows, uri, table_name, *, hybrid=False, mode="overwrite"):
        calls.append((rows, uri, table_name, hybrid, mode))

    monkeypatch.setattr("nemo_retriever.graph_ingestor.build_graph", lambda **kwargs: object())
    monkeypatch.setattr("nemo_retriever.graph_ingestor.InprocessExecutor", FakeExecutor)
    monkeypatch.setattr(
        "nemo_retriever.vector_store.lancedb_store.handle_lancedb",
        fake_handle_lancedb,
    )

    params = VdbUploadParams(
        lancedb=LanceDbParams(
            lancedb_uri="/tmp/native-lancedb",
            table_name="native_table",
            overwrite=False,
            create_index=False,
            hybrid=True,
        )
    )
    ingestor = GraphIngestor(run_mode="inprocess", documents=["/tmp/input.pdf"])
    result = ingestor.embed().vdb_upload(params).ingest()

    assert result is result_df
    assert ingestor._vdb_op is None
    assert len(calls) == 1
    rows, uri, table_name, hybrid, mode = calls[0]
    assert rows == result_df.to_dict(orient="records")
    assert uri == "/tmp/native-lancedb"
    assert table_name == "native_table"
    assert hybrid is True
    assert mode == "overwrite"
    assert ingestor._vdb_upload_result == {
        "backend": "lancedb",
        "uri": "/tmp/native-lancedb",
        "table_name": "native_table",
        "rows": 1,
    }


def test_graph_ingestor_default_batch_mode_vdb_upload_invokes_legacy_vdb_op(monkeypatch) -> None:
    result_df = pd.DataFrame(
        [
            {
                "text": "batch text",
                "metadata": {
                    "content_metadata": {"type": "text"},
                    "source_metadata": {"source_name": "batch.pdf"},
                },
                "custom_embeddings": {"embedding": [0.75, 0.125]},
            }
        ]
    )

    class FakeCluster:
        def available_gpu_count(self):
            return 0

    class FakeExecutor:
        def __init__(self, graph, **kwargs):
            self.graph = graph

        def ingest(self, data):
            return result_df

    monkeypatch.setattr("nemo_retriever.graph_ingestor.build_graph", lambda **kwargs: object())
    monkeypatch.setattr("nemo_retriever.graph_ingestor.gather_cluster_resources", lambda ray: FakeCluster())
    monkeypatch.setattr("nemo_retriever.graph_ingestor.batch_tuning_to_node_overrides", lambda *args, **kwargs: {})
    monkeypatch.setattr("nemo_retriever.graph_ingestor.RayDataExecutor", FakeExecutor)
    monkeypatch.setattr("ray.is_initialized", lambda: True)

    fake_vdb = FakeVDB()
    result = (
        GraphIngestor(documents=["/tmp/input.pdf"])
        .embed(EmbedParams(output_column="custom_embeddings"))
        .vdb_upload(vdb_op=fake_vdb)
        .ingest()
    )

    assert result is result_df
    assert len(fake_vdb.calls) == 1
    record = fake_vdb.calls[0][0][0]
    assert record["metadata"]["content"] == "batch text"
    assert record["metadata"]["embedding"] == [0.75, 0.125]
