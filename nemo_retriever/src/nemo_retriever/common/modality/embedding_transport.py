# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned row contract between content reshape and embedding.

The terminal canonical-VDB pipeline does not need the page-sized extraction
payloads after content has been reshaped into embedding rows. This module owns
the deliberately small set of fields that may cross that Ray block boundary.
Adding a new field is therefore an explicit compatibility decision instead of
an accidental consequence of copying a whole page row.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nemo_retriever.common.io.image_handle import EMBEDDING_IMAGE_HANDLE_FIELD
from nemo_retriever.common.stage_errors import iter_stage_errors_from_value

EMBEDDING_TRANSPORT_VERSION = 1
EMBEDDING_TRANSPORT_VERSION_FIELD = "_embedding_transport_version"
EMBEDDING_TRANSPORT_CONTENT_COUNTS_FIELD = "_embedding_transport_content_counts"
EMBEDDING_TRANSPORT_PAGE_IMAGE_URI_FIELD = "_embedding_transport_page_image_uri"
EMBEDDING_TRANSPORT_DIAGNOSTICS_FIELD = "_embedding_transport_diagnostics"

_COUNTED_CONTENT_TYPES = ("table", "chart", "infographic")

# These fields are consumed by the embedding actor or by canonical VDB record
# conversion. Rich extraction payloads are intentionally absent. Metadata is
# retained as a unit because it is part of the public retrieval contract.
EMBEDDING_TRANSPORT_FIELDS = (
    "text",
    "content",
    "metadata",
    "path",
    "source_id",
    "source",
    "filename",
    "page_number",
    "_page_number",
    "document_type",
    "_embed_modality",
    "_image_b64",
    EMBEDDING_IMAGE_HANDLE_FIELD,
    "_content_type",
    "content_type",
    "_stored_image_uri",
    "stored_image_uri",
    "_bbox_xyxy_norm",
    "bbox_xyxy_norm",
    "page_elements_v3_num_detections",
    "page_elements_v3_counts_by_label",
    "text_embeddings_1b_v2",
    "text_embeddings_1b_v2_dim",
    "text_embeddings_1b_v2_has_embedding",
    "embedding_v1_num_detections",
    "embedding_v1_counts_by_label",
    EMBEDDING_TRANSPORT_VERSION_FIELD,
    EMBEDDING_TRANSPORT_CONTENT_COUNTS_FIELD,
    EMBEDDING_TRANSPORT_PAGE_IMAGE_URI_FIELD,
    EMBEDDING_TRANSPORT_DIAGNOSTICS_FIELD,
)
_EMBEDDING_TRANSPORT_FIELD_SET = frozenset(EMBEDDING_TRANSPORT_FIELDS)


def _is_content_collection(value: Any) -> bool:
    return isinstance(value, list) or (isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == object)


def _content_counts(source: dict[str, Any]) -> dict[str, int]:
    existing = source.get(EMBEDDING_TRANSPORT_CONTENT_COUNTS_FIELD)
    counts: dict[str, int] = {}
    if isinstance(existing, dict):
        counts = {
            key: int(value)
            for key, value in existing.items()
            if isinstance(key, str)
            and key in _COUNTED_CONTENT_TYPES
            and isinstance(value, (int, np.integer))
            and int(value) >= 0
        }
    for content_type in _COUNTED_CONTENT_TYPES:
        value = source.get(content_type)
        if _is_content_collection(value):
            counts[content_type] = len(value)
    return counts


def _page_image_uri(source: dict[str, Any]) -> str | None:
    existing = source.get(EMBEDDING_TRANSPORT_PAGE_IMAGE_URI_FIELD)
    if isinstance(existing, str) and existing.strip():
        return existing
    page_image = source.get("page_image")
    if isinstance(page_image, dict):
        uri = page_image.get("stored_image_uri")
        if isinstance(uri, str) and uri.strip():
            return uri
    return None


def _omitted_diagnostics(source: dict[str, Any]) -> list[dict[str, Any]]:
    existing = source.get(EMBEDDING_TRANSPORT_DIAGNOSTICS_FIELD)
    diagnostics = list(existing) if isinstance(existing, list) else []
    diagnostics.extend(
        error
        for field, value in source.items()
        if field not in _EMBEDDING_TRANSPORT_FIELD_SET
        for error in iter_stage_errors_from_value(value, path=field)
    )
    return diagnostics


def _normalize_bbox_column(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.get("_bbox_xyxy_norm", pd.Series([None] * len(frame.index), index=frame.index))
    normalized = []
    for value in values:
        if hasattr(value, "to_numpy") and callable(value.to_numpy):
            value = value.to_numpy()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, tuple):
            value = list(value)
        normalized.append(value)
    frame["_bbox_xyxy_norm"] = pd.Series(normalized, index=frame.index, dtype=object)
    return frame


def project_embedding_transport(batch_df: Any) -> Any:
    """Project reshaped rows onto the versioned embedding transport contract.

    Unknown columns are never forwarded. Structured errors found only in an
    omitted payload are consolidated so the terminal VDB admission policy can
    still fail closed without retaining the payload itself.
    """

    if not isinstance(batch_df, pd.DataFrame):
        return batch_df
    if batch_df.empty:
        projected = batch_df.loc[:, [field for field in EMBEDDING_TRANSPORT_FIELDS if field in batch_df.columns]].copy()
        projected[EMBEDDING_TRANSPORT_VERSION_FIELD] = pd.Series(index=batch_df.index, dtype="int64")
        for field in (
            EMBEDDING_TRANSPORT_CONTENT_COUNTS_FIELD,
            EMBEDDING_TRANSPORT_PAGE_IMAGE_URI_FIELD,
            EMBEDDING_TRANSPORT_DIAGNOSTICS_FIELD,
        ):
            projected[field] = pd.Series(index=batch_df.index, dtype=object)
        return _normalize_bbox_column(projected)

    projected_rows: list[dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        source = row.to_dict()
        existing_version = source.get(EMBEDDING_TRANSPORT_VERSION_FIELD)
        if existing_version not in (None, EMBEDDING_TRANSPORT_VERSION):
            raise ValueError(f"Unsupported embedding transport version: {existing_version!r}")

        projected = {field: source[field] for field in EMBEDDING_TRANSPORT_FIELDS if field in source}
        projected[EMBEDDING_TRANSPORT_VERSION_FIELD] = EMBEDDING_TRANSPORT_VERSION
        projected[EMBEDDING_TRANSPORT_CONTENT_COUNTS_FIELD] = _content_counts(source)
        projected[EMBEDDING_TRANSPORT_PAGE_IMAGE_URI_FIELD] = _page_image_uri(source)
        projected[EMBEDDING_TRANSPORT_DIAGNOSTICS_FIELD] = _omitted_diagnostics(source)
        projected_rows.append(projected)

    return _normalize_bbox_column(pd.DataFrame(projected_rows, index=batch_df.index))


def embedding_transport_visible_bytes(value: Any) -> int:
    """Estimate recursively visible payload bytes without copying strings."""

    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, pd.DataFrame):
        return sum(
            embedding_transport_visible_bytes(cell)
            for column in value.columns
            for cell in value[column].tolist()
        )
    if isinstance(value, dict):
        return sum(
            embedding_transport_visible_bytes(key) + embedding_transport_visible_bytes(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return sum(embedding_transport_visible_bytes(child) for child in value)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return sum(embedding_transport_visible_bytes(child) for child in value.flat)
        return int(value.nbytes)
    return 0
