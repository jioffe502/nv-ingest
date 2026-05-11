# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any, Sequence

from nemo_retriever.ingestor import create_ingestor
from nemo_retriever.params import VdbUploadParams
from nemo_retriever.retriever import Retriever
from nemo_retriever.utils.input_files import expand_input_file_patterns


def ingest_documents(
    documents: Sequence[str],
    *,
    run_mode: str = "inprocess",
    lancedb_uri: str = "lancedb",
    table_name: str = "nv-ingest",
) -> dict[str, Any]:
    """Run the minimal SDK ingestion chain used by the root CLI."""
    document_list = expand_input_file_patterns(str(document) for document in documents)
    params = VdbUploadParams(vdb_kwargs={"uri": lancedb_uri, "table_name": table_name})

    result = create_ingestor(run_mode=run_mode).files(document_list).extract().embed().vdb_upload(params).ingest()
    return {
        "documents": document_list,
        "lancedb_uri": lancedb_uri,
        "result": result,
        "table_name": table_name,
    }


def query_documents(
    query: str,
    *,
    top_k: int = 10,
    lancedb_uri: str = "lancedb",
    table_name: str = "nv-ingest",
) -> list[dict[str, Any]]:
    """Run the minimal SDK query path used by the root CLI."""
    retriever = Retriever(top_k=top_k, vdb_kwargs={"uri": lancedb_uri, "table_name": table_name})
    return retriever.query(query)


def hits_to_json(hits: Sequence[dict[str, Any]]) -> str:
    return json.dumps(list(hits), indent=2, sort_keys=True, default=str)
