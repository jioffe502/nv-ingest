# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: VDBUploadOperator writing to a real Milvus instance.

Requires a running Milvus at the URI below. Run manually:

    python tests/integration_test_milvus_vdb.py

This is NOT part of the pytest suite — it requires external infrastructure.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

MILVUS_URI = os.environ.get("NEMO_RETRIEVER_MILVUS_URI", "http://172.20.0.4:19530")
COLLECTION_NAME = "nemo_retriever_integration_test"
EMBED_DIM = 128


def _make_embedded_df(n: int = 10, dim: int = EMBED_DIM) -> pd.DataFrame:
    """Build a minimal post-embed DataFrame."""
    rows = []
    for i in range(n):
        embedding = [float(i * dim + j) / (n * dim) for j in range(dim)]
        metadata = {
            "embedding": embedding,
            "source_path": f"/data/doc_{i}.pdf",
            "content_metadata": {"hierarchy": {"page": i}},
        }
        rows.append(
            {
                "metadata": metadata,
                "text_embeddings_1b_v2": {"embedding": embedding, "info_msg": None},
                "text": f"This is the content of page {i} in the test document.",
                "path": f"/data/doc_{i}.pdf",
                "page_number": i,
                "document_type": "text",
            }
        )
    return pd.DataFrame(rows)


def main():
    from nemo_retriever.graph.vdb_upload_operator import VDBUploadOperator
    from nemo_retriever.params.models import VdbUploadParams

    print(f"=== Milvus Integration Test ===")
    print(f"Milvus URI: {MILVUS_URI}")
    print(f"Collection: {COLLECTION_NAME}")
    print()

    # --- Step 1: Create operator with Milvus backend ---
    params = VdbUploadParams(
        backend="milvus",
        client_vdb_kwargs={
            "milvus_uri": MILVUS_URI,
            "collection_name": COLLECTION_NAME,
            "dense_dim": EMBED_DIM,
            "recreate": True,
            "gpu_index": False,
            "stream": True,
            "sparse": False,
        },
    )
    op = VDBUploadOperator(params=params)
    print(f"[OK] VDBUploadOperator created with backend='milvus'")

    # --- Step 2: Run two batches through the operator ---
    df_batch1 = _make_embedded_df(5)
    df_batch2 = _make_embedded_df(5)

    result1 = op.run(df_batch1)
    print(f"[OK] Batch 1: {len(result1)} rows passed through, records written")

    result2 = op.run(df_batch2)
    print(f"[OK] Batch 2: {len(result2)} rows passed through, records written")

    # --- Step 3: Finalize indexing and verify data landed in Milvus ---
    op.finalize()
    print("[OK] Finalized Milvus collection")

    from pymilvus import MilvusClient

    client = MilvusClient(uri=MILVUS_URI)

    if not client.has_collection(COLLECTION_NAME):
        print(f"[FAIL] Collection {COLLECTION_NAME} not found!")
        return 1

    stats = client.get_collection_stats(COLLECTION_NAME)
    row_count = stats.get("row_count", 0)
    print(f"[OK] Collection exists with {row_count} rows")

    # Query to verify data is searchable
    query_vector = _make_embedded_df(1).iloc[0]["metadata"]["embedding"]
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=3,
        output_fields=["text"],
    )
    print(f"[OK] Search returned {len(results[0])} hits")
    for i, hit in enumerate(results[0]):
        text = hit.get("entity", {}).get("text", "")[:60]
        print(f"     Hit {i}: distance={hit['distance']:.4f} text='{text}...'")

    # --- Step 4: Cleanup ---
    client.drop_collection(COLLECTION_NAME)
    print(f"[OK] Cleaned up collection {COLLECTION_NAME}")

    print()
    print("=== PASS: Milvus integration test completed successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
