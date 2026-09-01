# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical metadata contract for lossless embedding-input split children."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

EMBEDDING_SPLIT_METADATA_KEYS = (
    "embedding_parent_id",
    "embedding_chunk_id",
    "embedding_chunk_index",
    "embedding_chunk_count",
    "embedding_chunk_start_token",
    "embedding_chunk_end_token",
)


@dataclass(frozen=True)
class EmbeddingSplitProvenance:
    """Identity and token range carried by one lossless split child."""

    parent_id: str
    chunk_id: str
    chunk_index: int
    chunk_count: int
    start_token: int
    end_token: int

    def as_metadata(self, *, content: str) -> dict[str, Any]:
        """Return canonical child metadata, including its exact embedding text."""
        return {
            "content": content,
            "embedding_parent_id": self.parent_id,
            "embedding_chunk_id": self.chunk_id,
            "embedding_chunk_index": self.chunk_index,
            "embedding_chunk_count": self.chunk_count,
            "embedding_chunk_start_token": self.start_token,
            "embedding_chunk_end_token": self.end_token,
        }


@dataclass(frozen=True)
class SelectedEmbeddingText:
    """The source column and exact text selected for one embedding row."""

    column: str
    content: str
    is_split_child: bool = False


def select_embedding_text(row: Mapping[str, Any], *, text_column: str = "text") -> SelectedEmbeddingText | None:
    """Select embedding text using the one canonical fallback order.

    Split provenance wins because its ``content`` field is the exact reversible
    child text, including whitespace-only children.
    """
    split_content = embedding_split_content(row.get("metadata"))
    if split_content is not None:
        for column in dict.fromkeys((text_column, "text", "content", "chunk", "page_text")):
            value = row.get(column)
            if isinstance(value, str) and value == split_content:
                return SelectedEmbeddingText(column=column, content=split_content, is_split_child=True)
        return SelectedEmbeddingText(column=text_column, content=split_content, is_split_child=True)

    for column in dict.fromkeys((text_column, "text", "content", "chunk", "page_text")):
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return SelectedEmbeddingText(column=column, content=value)
    return None


def embedding_text_input(
    row: Mapping[str, Any],
    *,
    text_column: str = "text",
    default_modality: str = "text",
) -> str | None:
    """Return the exact runtime text, preserving split children byte-for-byte."""
    selected = select_embedding_text(row, text_column=text_column)
    if selected is None:
        return None
    requested_modality = embedding_runtime_modality(row, default_modality=default_modality)
    if selected.is_split_child or requested_modality == "text_image":
        return selected.content
    return selected.content.strip()


def embedding_runtime_modality(row: Mapping[str, Any], *, default_modality: str = "text") -> str:
    """Resolve the requested runtime modality, inheriting the default only for nulls."""
    value = row.get("_embed_modality")
    is_missing = value is None or (pd.api.types.is_scalar(value) and bool(pd.isna(value)))
    return str(default_modality if is_missing else value)


def requires_text_admission(row: Mapping[str, Any], *, default_modality: str = "text") -> bool:
    """Return whether the row will reach a text embedder and needs admission."""
    modality = embedding_runtime_modality(row, default_modality=default_modality)
    if modality == "image":
        return False
    image = row.get("_image_b64")
    return modality != "text_image" or not (isinstance(image, str) and image.strip())


def embedding_split_content(metadata: Any) -> str | None:
    """Return the exact child text when *metadata* identifies a split child."""
    if not isinstance(metadata, Mapping):
        return None
    chunk_id = metadata.get("embedding_chunk_id")
    content = metadata.get("content")
    if isinstance(chunk_id, str) and chunk_id.strip() and isinstance(content, str):
        return content
    return None


def embedding_split_index(metadata: Any) -> int | None:
    """Return the canonical child index when *metadata* identifies a split child."""
    if not isinstance(metadata, Mapping):
        return None
    chunk_id = metadata.get("embedding_chunk_id")
    chunk_index = metadata.get("embedding_chunk_index")
    if not isinstance(chunk_id, str) or not chunk_id.strip() or isinstance(chunk_index, bool):
        return None
    try:
        return int(chunk_index)
    except (TypeError, ValueError):
        return None


def embedding_split_id(metadata: Any) -> str | None:
    """Return the canonical child ID when present."""
    if not isinstance(metadata, Mapping):
        return None
    chunk_id = metadata.get("embedding_chunk_id")
    return chunk_id if isinstance(chunk_id, str) and chunk_id.strip() else None


__all__ = [
    "EMBEDDING_SPLIT_METADATA_KEYS",
    "EmbeddingSplitProvenance",
    "SelectedEmbeddingText",
    "embedding_runtime_modality",
    "embedding_text_input",
    "embedding_split_content",
    "embedding_split_id",
    "embedding_split_index",
    "requires_text_admission",
    "select_embedding_text",
]
