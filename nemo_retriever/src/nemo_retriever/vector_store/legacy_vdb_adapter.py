# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters for invoking legacy NV-Ingest VDB operators from Retriever rows."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, List


_MISSING = object()
_STRUCTURED_CONTENT_TYPES = {"chart", "infographic", "table"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if value is _MISSING:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    return isinstance(value, float) and math.isnan(value)


def _maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return _MISSING
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def _as_dict(value: Any) -> dict[str, Any]:
    parsed = _maybe_parse_json(value)
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return {}


def _as_list(value: Any) -> list[Any] | None:
    parsed = _maybe_parse_json(value)
    if isinstance(parsed, list):
        return parsed if parsed else None
    if isinstance(parsed, tuple):
        return list(parsed) if parsed else None
    if isinstance(parsed, (str, bytes, bytearray, Mapping)) or _is_missing(parsed):
        return None
    tolist = getattr(parsed, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list) and converted:
            return converted
    return None


def _row_get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, _MISSING)
    if _is_missing(value):
        return default
    return value


def _safe_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _content_type_base(value: Any) -> str | None:
    if _is_missing(value):
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized.endswith("_caption"):
        normalized = normalized[: -len("_caption")]
    return normalized or None


def _merge_dicts(*values: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        merged.update(_as_dict(value))
    return merged


def _extract_embedding(row: Mapping[str, Any], metadata: Mapping[str, Any], embedding_column: str, embedding_key: str):
    embedding = _as_list(metadata.get("embedding"))
    if embedding is not None:
        return embedding

    payload = _maybe_parse_json(_row_get(row, embedding_column))
    if isinstance(payload, Mapping):
        return _as_list(payload.get(embedding_key))
    return _as_list(payload)


def _extract_content(row: Mapping[str, Any], metadata: Mapping[str, Any], text_column: str) -> str:
    content = _row_get(row, text_column)
    if _is_missing(content):
        content = metadata.get("content")
    if _is_missing(content):
        content = _row_get(row, "content", "")
    return "" if _is_missing(content) else str(content)


def _build_content_metadata(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    content_metadata = _merge_dicts(_row_get(row, "content_metadata"), metadata.get("content_metadata"))

    page_number = _safe_int(_row_get(row, "page_number"))
    if page_number is not None:
        content_metadata.setdefault("page_number", page_number)

    content_type = _content_type_base(
        _row_get(row, "_content_type")
        or metadata.get("_content_type")
        or metadata.get("content_type")
        or content_metadata.get("subtype")
        or content_metadata.get("type")
    )
    if content_type:
        content_metadata.setdefault("type", content_type)
        if content_type in _STRUCTURED_CONTENT_TYPES or content_type in {"image", "page_image", "audio"}:
            content_metadata.setdefault("subtype", content_type)

    bbox = _row_get(row, "_bbox_xyxy_norm")
    if bbox is not None:
        content_metadata.setdefault("location", bbox)
    return content_metadata


def _build_source_metadata(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    source_metadata = _merge_dicts(_row_get(row, "source_metadata"), metadata.get("source_metadata"))
    source_path = (
        _row_get(row, "path")
        or _row_get(row, "source_path")
        or metadata.get("source_path")
        or source_metadata.get("source_id")
        or source_metadata.get("source_name")
    )
    if source_path:
        source_value = str(source_path)
        source_metadata.setdefault("source_id", source_value)
        source_metadata.setdefault("source_name", source_value)
    else:
        source_metadata.setdefault("source_name", "unknown")
        source_metadata.setdefault("source_id", source_metadata["source_name"])
    return source_metadata


def _infer_document_type(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    content_metadata: Mapping[str, Any],
) -> str:
    explicit = _row_get(row, "document_type") or metadata.get("document_type")
    if explicit:
        return str(explicit)

    content_type = _content_type_base(
        _row_get(row, "_content_type")
        or metadata.get("_content_type")
        or metadata.get("content_type")
        or content_metadata.get("subtype")
        or content_metadata.get("type")
    )
    if content_type in _STRUCTURED_CONTENT_TYPES:
        return "structured"
    if content_type in {"image", "page_image"}:
        return "image"
    if content_type == "audio":
        return "audio"
    return "text"


def _build_legacy_record(
    row: Mapping[str, Any],
    *,
    embedding_column: str,
    embedding_key: str,
    text_column: str,
) -> dict[str, Any] | None:
    metadata = _as_dict(_row_get(row, "metadata"))
    embedding = _extract_embedding(row, metadata, embedding_column, embedding_key)
    if embedding is None:
        return None

    content = _extract_content(row, metadata, text_column)
    content_metadata = _build_content_metadata(row, metadata)
    source_metadata = _build_source_metadata(row, metadata)
    document_type = _infer_document_type(row, metadata, content_metadata)

    record_metadata = dict(metadata)
    record_metadata["content"] = content
    record_metadata["embedding"] = embedding
    record_metadata["content_metadata"] = content_metadata
    record_metadata["source_metadata"] = source_metadata

    if document_type == "structured":
        table_metadata = _as_dict(record_metadata.get("table_metadata"))
        table_metadata.setdefault("table_content", content)
        table_metadata.setdefault("table_location", content_metadata.get("location"))
        table_metadata.setdefault("table_location_max_dimensions", content_metadata.get("max_dimensions"))
        record_metadata["table_metadata"] = table_metadata
    elif document_type == "image":
        image_metadata = _as_dict(record_metadata.get("image_metadata"))
        image_metadata.setdefault("caption", content)
        image_metadata.setdefault("text", content)
        image_metadata.setdefault("image_location", content_metadata.get("location"))
        image_metadata.setdefault("image_location_max_dimensions", content_metadata.get("max_dimensions"))
        record_metadata["image_metadata"] = image_metadata
    elif document_type == "audio":
        audio_metadata = _as_dict(record_metadata.get("audio_metadata"))
        audio_metadata.setdefault("audio_transcript", content)
        record_metadata["audio_metadata"] = audio_metadata

    return {"document_type": document_type, "metadata": record_metadata}


def dataframe_to_legacy_vdb_records(
    df: Any,
    *,
    embedding_column: str = "text_embeddings_1b_v2",
    embedding_key: str = "embedding",
    text_column: str = "text",
) -> list[list[dict[str, Any]]]:
    """Convert a Retriever pandas DataFrame into legacy NV-Ingest VDB records."""
    if df is None:
        return [[]]

    if hasattr(df, "to_dict") and hasattr(df, "columns"):
        rows = df.to_dict(orient="records")
    elif isinstance(df, list):
        rows = df
    else:
        raise TypeError(f"Expected a pandas DataFrame or list of dict rows, got {type(df).__name__}.")

    records: List[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        record = _build_legacy_record(
            row,
            embedding_column=embedding_column,
            embedding_key=embedding_key,
            text_column=text_column,
        )
        if record is not None:
            records.append(record)
    return [records]


def execution_result_to_legacy_vdb_records(
    result: Any,
    *,
    embedding_column: str = "text_embeddings_1b_v2",
    embedding_key: str = "embedding",
    text_column: str = "text",
) -> list[list[dict[str, Any]]]:
    """Convert a GraphIngestor execution result into one legacy VDB record batch."""
    if hasattr(result, "to_dict") and hasattr(result, "columns"):
        return dataframe_to_legacy_vdb_records(
            result,
            embedding_column=embedding_column,
            embedding_key=embedding_key,
            text_column=text_column,
        )

    records: List[dict[str, Any]] = []
    iter_batches = getattr(result, "iter_batches", None)
    if callable(iter_batches):
        for batch_df in iter_batches(batch_format="pandas"):
            records.extend(
                dataframe_to_legacy_vdb_records(
                    batch_df,
                    embedding_column=embedding_column,
                    embedding_key=embedding_key,
                    text_column=text_column,
                )[0]
            )
        return [records]

    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        return dataframe_to_legacy_vdb_records(
            to_pandas(),
            embedding_column=embedding_column,
            embedding_key=embedding_key,
            text_column=text_column,
        )

    raise TypeError(f"Cannot convert {type(result).__name__} to legacy VDB records.")
