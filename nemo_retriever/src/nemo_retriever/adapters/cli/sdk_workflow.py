# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from nemo_retriever.ingestor import create_ingestor
from nemo_retriever.params import VdbUploadParams
from nemo_retriever.retriever import Retriever
from nemo_retriever.utils.input_files import expand_input_file_patterns, resolve_input_files


# The ingest command accepts bare dataset directories; expand those to PDFs
# before passing file/glob inputs through the shared input normalizer.
def _expand_ingest_documents(documents: Sequence[str]) -> list[str]:
    inputs: list[str] = []
    for document in documents:
        raw_document = str(document)
        path = Path(raw_document).expanduser()
        if path.is_dir():
            directory_files = resolve_input_files(path, "pdf")
            if not directory_files:
                raise FileNotFoundError(f"No PDF files found under directory: {path}")
            inputs.extend(str(file) for file in directory_files)
        else:
            inputs.append(raw_document)

    return expand_input_file_patterns(inputs)


def ingest_documents(
    documents: Sequence[str],
    *,
    run_mode: str = "inprocess",
    lancedb_uri: str = "lancedb",
    table_name: str = "nv-ingest",
) -> dict[str, Any]:
    """Run the minimal SDK ingestion chain used by the root CLI."""
    document_list = _expand_ingest_documents(documents)
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
