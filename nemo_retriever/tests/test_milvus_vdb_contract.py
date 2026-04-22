import os
import uuid

import pandas as pd
import pytest
from pymilvus import MilvusClient

from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.params import EmbedParams
from nv_ingest_client.util.vdb.milvus import Milvus


@pytest.mark.integration
def test_graph_ingestor_legacy_milvus_vdb_op_writes_records(monkeypatch) -> None:
    milvus_uri = os.environ.get("NEMO_RETRIEVER_MILVUS_URI")
    if not milvus_uri:
        pytest.skip("Set NEMO_RETRIEVER_MILVUS_URI to run live Milvus VDB contract test.")

    collection_name = f"codex_vdb_contract_{uuid.uuid4().hex[:12]}"
    text = f"milvus contract smoke {uuid.uuid4().hex}"
    result_df = pd.DataFrame(
        [
            {
                "text": text,
                "path": "/tmp/contract.pdf",
                "page_number": 1,
                "metadata": {
                    "content_metadata": {"type": "text"},
                    "source_metadata": {"source_name": "contract.pdf"},
                },
                "contract_embeddings": {"embedding": [0.125, 0.875]},
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

    client = MilvusClient(uri=milvus_uri)
    try:
        vdb = Milvus(
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            sparse=False,
            recreate=True,
            gpu_index=False,
            gpu_search=False,
            dense_dim=2,
            stream=True,
            threshold=10,
            no_wait_index=True,
        )

        (
            GraphIngestor(run_mode="inprocess", documents=["/tmp/contract.pdf"])
            .embed(EmbedParams(output_column="contract_embeddings"))
            .vdb_upload(vdb_op=vdb)
            .ingest()
        )

        client.flush(collection_name)
        client.load_collection(collection_name)
        rows = client.query(
            collection_name=collection_name,
            filter=f'text == "{text}"',
            output_fields=["text", "source", "content_metadata"],
        )

        assert len(rows) == 1
        assert rows[0]["text"] == text
        assert rows[0]["source"]["source_name"] == "contract.pdf"
        assert rows[0]["content_metadata"]["page_number"] == 1
        assert rows[0]["content_metadata"]["type"] == "text"
    finally:
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
