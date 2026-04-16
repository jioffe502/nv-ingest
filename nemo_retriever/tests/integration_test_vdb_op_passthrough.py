# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: pass pre-constructed client VDB instances into the graph.

Exercises the lead-review ask for PR 1847 — that `GraphIngestor.vdb_upload`
accept a pre-built `nv_ingest_client.util.vdb` instance (`LanceDB(...)` or
`Milvus(...)`) and wire it directly into `VDBUploadOperator` without the
operator reconstructing it from `VdbUploadParams`.

Runs a small bo20 ingest against each backend and verifies the target table /
collection was populated.

Requires:
  - Running Milvus at ``MILVUS_URI``
  - GPU for extraction + embedding
  - bo20 dataset at ``DATASET_DIR``

Usage:
    python tests/integration_test_vdb_op_passthrough.py [--milvus-uri URI]
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

MILVUS_URI = "http://172.20.0.4:19530"
MILVUS_COLLECTION = "vdb_op_passthrough_bo20"
DATASET_DIR = "/datasets/nv-ingest/bo20"
EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
EMBED_DIM = 2048


def _run_ingest(vdb_op):
    from nemo_retriever import create_ingestor
    from nemo_retriever.params import EmbedParams

    t0 = time.perf_counter()
    (
        create_ingestor(run_mode="batch")
        .files(DATASET_DIR + "/*.pdf")
        .extract(extract_text=True, extract_tables=True, extract_charts=True)
        .embed(EmbedParams(model_name=EMBED_MODEL, inference_batch_size=32))
        .vdb_upload(vdb_op=vdb_op)
        .ingest()
    )
    return time.perf_counter() - t0


def test_lancedb_passthrough():
    from nv_ingest_client.util.vdb import get_vdb_op_cls

    LanceDB = get_vdb_op_cls("lancedb")
    tmp_dir = Path(tempfile.mkdtemp(prefix="vdb_op_passthrough_lance_"))
    uri = str(tmp_dir / "lancedb")
    table_name = "bo20_lance_passthrough"

    print("--- LanceDB pre-constructed instance ---")
    print(f"  uri: {uri}")
    print(f"  table: {table_name}")

    client = LanceDB(uri=uri, table_name=table_name, overwrite=True)
    assert type(client).__name__ == "LanceDB", f"expected LanceDB, got {type(client).__name__}"

    elapsed = _run_ingest(client)

    import lancedb

    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    n = table.count_rows()
    print(f"[OK] LanceDB passthrough ingested {n} rows in {elapsed:.1f}s")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return n > 0


def test_milvus_passthrough():
    from nv_ingest_client.util.vdb import get_vdb_op_cls
    from pymilvus import MilvusClient

    Milvus = get_vdb_op_cls("milvus")
    print("--- Milvus pre-constructed instance ---")
    print(f"  uri: {MILVUS_URI}")
    print(f"  collection: {MILVUS_COLLECTION}")

    client = Milvus(
        milvus_uri=MILVUS_URI,
        collection_name=MILVUS_COLLECTION,
        dense_dim=EMBED_DIM,
        recreate=True,
        gpu_index=False,
        stream=True,
        sparse=False,
    )
    assert type(client).__name__ == "Milvus", f"expected Milvus, got {type(client).__name__}"

    elapsed = _run_ingest(client)

    mc = MilvusClient(uri=MILVUS_URI)
    mc.load_collection(MILVUS_COLLECTION)
    stats = mc.get_collection_stats(collection_name=MILVUS_COLLECTION)
    n = int(stats.get("row_count", 0))
    print(f"[OK] Milvus passthrough ingested {n} rows in {elapsed:.1f}s")

    mc.drop_collection(MILVUS_COLLECTION)
    return n > 0


def main():
    print("=" * 60)
    print("VDB-op passthrough integration test (bo20)")
    print(f"  Dataset: {DATASET_DIR}")
    print("=" * 60)

    results = {}
    try:
        results["lancedb"] = test_lancedb_passthrough()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] LanceDB passthrough: {exc}")
        traceback.print_exc()
        results["lancedb"] = False

    try:
        results["milvus"] = test_milvus_passthrough()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Milvus passthrough: {exc}")
        traceback.print_exc()
        results["milvus"] = False

    print()
    print("=" * 60)
    print("RESULTS")
    for backend, ok in results.items():
        print(f"  {backend}: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    if "--milvus-uri" in sys.argv:
        idx = sys.argv.index("--milvus-uri")
        MILVUS_URI = sys.argv[idx + 1]
    sys.exit(main())
