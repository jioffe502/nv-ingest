# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import timedelta

from nv_ingest_client.util.vdb.lancedb import LanceDB
from nemo_retriever.params.models import LanceDbParams
from nemo_retriever.vector_store.vdb_records import build_vdb_records, build_vdb_records_from_dicts
import pandas as pd

logger = logging.getLogger(__name__)


def _read_text_embeddings_json_df(path: Path) -> pd.DataFrame:
    """
    Read a `*.text_embeddings.json` file emitted by `nemo_retriever.text_embed.stage`.

    Expected wrapper shape:
      {
        ...,
        "df_records": [ { "document_type": ..., "metadata": {...}, ... }, ... ],
        ...
      }
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        raise ValueError(f"Failed reading JSON {path}: {e}") from e

    if isinstance(obj, dict):
        recs = obj.get("df_records")
        if isinstance(recs, list):
            return pd.DataFrame([r for r in recs if isinstance(r, dict)])
        # Fall back to a single record.
        return pd.DataFrame([obj])

    if isinstance(obj, list):
        return pd.DataFrame([r for r in obj if isinstance(r, dict)])

    return pd.DataFrame([])


def _iter_text_embeddings_json_files(input_dir: Path, *, recursive: bool) -> List[Path]:
    """
    Return sorted list of `*.text_embeddings.json` files.

    The stage5 default naming is: `<input>.text_embeddings.json` (where `<input>` is
    typically a stage4 output filename).
    """
    if recursive:
        files = list(input_dir.rglob("*.text_embeddings.json"))
    else:
        files = list(input_dir.glob("*.text_embeddings.json"))
    return sorted([p for p in files if p.is_file()])


def create_lancedb_index(table: Any, *, cfg: LanceDbParams, text_column: str = "text") -> None:
    """Create vector (IVF_HNSW_SQ) and optionally FTS indices on a LanceDB table."""
    try:
        table.create_index(
            index_type=cfg.index_type,
            metric=cfg.metric,
            num_partitions=int(cfg.num_partitions),
            num_sub_vectors=int(cfg.num_sub_vectors),
            vector_column_name="vector",
        )
    except TypeError:
        table.create_index(vector_column_name="vector")

    if cfg.hybrid:
        try:
            table.create_fts_index(text_column, replace=True, language=cfg.fts_language)
        except Exception:
            logger.warning(
                "FTS index creation failed on column %r; continuing with vector-only search.",
                text_column,
                exc_info=True,
            )

    for index_stub in table.list_indices():
        table.wait_for_index([index_stub.name], timeout=timedelta(seconds=600))


def write_embeddings_to_lancedb(df_with_embeddings: pd.DataFrame, *, cfg: LanceDbParams) -> None:
    """
    Write embeddings found in *df_with_embeddings* to LanceDB.

    This is used programmatically by ``nemo_retriever.text_embed.stage``.
    """
    import lancedb

    from nemo_retriever.vector_store.lancedb_utils import infer_vector_dim, lancedb_schema

    records = build_vdb_records(df_with_embeddings)
    if not records:
        return
    dim = infer_vector_dim(records)
    schema = lancedb_schema(vector_dim=dim)
    mode = "overwrite" if cfg.overwrite else "create"
    db = lancedb.connect(uri=cfg.lancedb_uri)
    table = db.create_table(cfg.table_name, data=records, schema=schema, mode=mode)
    if cfg.create_index:
        create_lancedb_index(table, cfg=cfg)


def write_text_embeddings_dir_to_lancedb(
    input_dir: Path,
    *,
    cfg: LanceDbParams,
    recursive: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Read `*.text_embeddings.json` files from `input_dir` and upload their embeddings to LanceDB.
    """
    input_dir = Path(input_dir)
    files = _iter_text_embeddings_json_files(input_dir, recursive=bool(recursive))
    if limit is not None:
        files = files[: int(limit)]

    processed = 0
    skipped = 0
    failed = 0

    lancedb_client = LanceDB(uri=cfg.lancedb_uri, table_name=cfg.table_name, overwrite=cfg.overwrite)

    results = []

    for p in files:
        df = _read_text_embeddings_json_df(p)
        rows = df.to_dict(orient="records")
        results.append(rows)

    if not results:
        logger.warning("No *.text_embeddings.json files found in %s; nothing to write.", input_dir)
        return {
            "input_dir": str(input_dir),
            "n_files": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "lancedb": {"uri": cfg.lancedb_uri, "table_name": cfg.table_name, "overwrite": cfg.overwrite},
        }

    lancedb_client.run(results)

    return {
        "input_dir": str(input_dir),
        "n_files": len(files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "lancedb": {"uri": cfg.lancedb_uri, "table_name": cfg.table_name, "overwrite": cfg.overwrite},
    }

