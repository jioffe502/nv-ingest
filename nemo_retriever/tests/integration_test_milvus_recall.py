# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Milvus recall test: ingest jp20 → embed queries → search → compute recall.

Requires:
  - Running Milvus instance
  - GPU for extraction + embedding
  - jp20 dataset at /datasets/nv-ingest/jp20
  - Query CSV at data/jp20_query_gt.csv

Usage:
    python tests/integration_test_milvus_recall.py [--milvus-uri URI]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

MILVUS_URI = "http://172.20.0.4:19530"
COLLECTION_NAME = "jp20_recall_test"
DATASET_DIR = "/datasets/nv-ingest/jp20"
QUERY_CSV = "/raid/jioffe/NeMo-Retriever/data/jp20_query_gt.csv"
EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
EMBED_DIM = 2048
TOP_K = 10


def ingest_jp20_to_milvus():
    """Run the full graph pipeline with Milvus as the VDB backend."""
    from nemo_retriever import create_ingestor
    from nemo_retriever.params import EmbedParams, ExtractParams, VdbUploadParams
    from nemo_retriever.params.models import LanceDbParams

    print("--- Step 1: Ingesting jp20 into Milvus ---")
    t0 = time.perf_counter()

    ingestor = (
        create_ingestor(run_mode="batch")
        .files(DATASET_DIR + "/*.pdf")
        .extract(extract_text=True, extract_tables=True, extract_charts=True)
        .embed(
            EmbedParams(
                model_name=EMBED_MODEL,
                inference_batch_size=32,
            )
        )
        .vdb_upload(
            VdbUploadParams(
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
        )
    )
    ingestor.ingest()

    elapsed = time.perf_counter() - t0
    print(f"[OK] Ingestion complete in {elapsed:.1f}s")
    return elapsed


def embed_queries():
    """Embed jp20 queries using the local model."""
    import pandas as pd

    from nemo_retriever.model import create_local_embedder

    print("--- Step 2: Embedding queries ---")

    df = pd.read_csv(QUERY_CSV)
    queries = df["query"].astype(str).tolist()
    gold_keys = []
    for _, row in df.iterrows():
        if "pdf_page" in df.columns:
            gold_keys.append(str(row["pdf_page"]))
        else:
            gold_keys.append(f"{row['pdf']}_{row['page']}")

    print(f"  {len(queries)} queries, embedding with {EMBED_MODEL}...")
    embedder = create_local_embedder(EMBED_MODEL, device="cuda")
    vecs = embedder.embed(["query: " + q for q in queries], batch_size=32)
    query_embeddings = vecs.detach().to("cpu").tolist()
    print(f"[OK] Embedded {len(query_embeddings)} queries")

    return queries, gold_keys, query_embeddings


def search_milvus(query_embeddings):
    """Search Milvus collection with pre-computed query embeddings."""
    from pymilvus import MilvusClient

    print("--- Step 3: Searching Milvus ---")

    client = MilvusClient(uri=MILVUS_URI)
    t0 = time.perf_counter()

    all_hits = []
    # Batch queries to avoid overwhelming Milvus
    batch_size = 20
    for i in range(0, len(query_embeddings), batch_size):
        batch = query_embeddings[i : i + batch_size]
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=batch,
            limit=TOP_K,
            output_fields=["text", "source", "content_metadata"],
        )
        all_hits.extend(results)

    elapsed = time.perf_counter() - t0
    print(f"[OK] Searched {len(query_embeddings)} queries in {elapsed:.1f}s ({len(query_embeddings)/elapsed:.1f} QPS)")

    return all_hits


def hits_to_keys(all_hits):
    """Extract pdf_page keys from Milvus search results."""
    import json

    retrieved_keys = []
    for hits in all_hits:
        keys = []
        for h in hits:
            entity = h.get("entity", {})
            # Milvus stores content_metadata and source as dicts (not JSON strings)
            content_meta = entity.get("content_metadata", {})
            source = entity.get("source", {})

            if isinstance(content_meta, str):
                try:
                    content_meta = json.loads(content_meta)
                except (json.JSONDecodeError, TypeError):
                    content_meta = {}
            if isinstance(source, str):
                try:
                    source = json.loads(source)
                except (json.JSONDecodeError, TypeError):
                    source = {}

            source_id = source.get("source_id", "")
            page_number = content_meta.get("page_number")
            if page_number is None:
                hierarchy = content_meta.get("hierarchy", {})
                if isinstance(hierarchy, dict):
                    page_number = hierarchy.get("page")

            if source_id and page_number is not None:
                filename = Path(str(source_id)).stem
                keys.append(f"{filename}_{page_number}")
        retrieved_keys.append(keys)

    return retrieved_keys


def compute_recall(gold_keys, retrieved_keys, ks=(1, 5, 10)):
    """Compute recall@k metrics."""
    print("--- Step 4: Computing recall ---")

    metrics = {}
    for k in ks:
        hits = 0
        for gold, retrieved in zip(gold_keys, retrieved_keys):
            parts = str(gold).rsplit("_", 1)
            if len(parts) == 2:
                specific = f"{parts[0]}_{parts[1]}"
                whole_doc = f"{parts[0]}_-1"
                top = retrieved[:k]
                if specific in top or whole_doc in top:
                    hits += 1
            elif gold in retrieved[:k]:
                hits += 1
        recall = hits / max(1, len(gold_keys))
        metrics[f"recall@{k}"] = recall
        print(f"  recall@{k}: {recall:.4f}")

    return metrics


def cleanup():
    from pymilvus import MilvusClient

    client = MilvusClient(uri=MILVUS_URI)
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"[OK] Cleaned up collection {COLLECTION_NAME}")


def main():
    print("=" * 60)
    print("Milvus End-to-End Recall Test (jp20)")
    print(f"  Milvus: {MILVUS_URI}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Queries: {QUERY_CSV}")
    print("=" * 60)
    print()

    try:
        ingest_secs = ingest_jp20_to_milvus()
        queries, gold_keys, query_embeddings = embed_queries()
        all_hits = search_milvus(query_embeddings)
        retrieved_keys = hits_to_keys(all_hits)
        metrics = compute_recall(gold_keys, retrieved_keys)

        print()
        print("=" * 60)
        print("RESULTS")
        print(f"  Ingestion: {ingest_secs:.1f}s")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()

        # Compare against LanceDB baseline
        baseline = {"recall@1": 0.6435, "recall@5": 0.8783, "recall@10": 0.9304}
        print("vs LanceDB baseline:")
        all_match = True
        for k, bl in baseline.items():
            mv = metrics.get(k, 0)
            delta = mv - bl
            status = "MATCH" if abs(delta) < 0.01 else ("BETTER" if delta > 0 else "WORSE")
            print(f"  {k}: Milvus={mv:.4f} LanceDB={bl:.4f} ({status})")
            if status == "WORSE":
                all_match = False

        print()
        if all_match:
            print("=== PASS: Milvus recall matches LanceDB baseline ===")
        else:
            print("=== NOTE: Milvus recall differs from LanceDB baseline ===")

        return 0

    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        cleanup()


if __name__ == "__main__":
    if "--milvus-uri" in sys.argv:
        idx = sys.argv.index("--milvus-uri")
        MILVUS_URI = sys.argv[idx + 1]
    sys.exit(main())
