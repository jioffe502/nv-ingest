# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic admission and overflow splitting for text embedding rows."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nemo_retriever.common.modality.txt.tokenizer_provider import (
    ChunkTokenizer,
    load_chunk_tokenizer,
)
from nemo_retriever.common.schemas.embedding import (
    EmbeddingSplitProvenance,
    SelectedEmbeddingText,
    requires_text_admission,
    select_embedding_text,
)
from nemo_retriever.models import resolve_embed_model
from nemo_retriever.models.embed_model_spec import resolve_embed_model_spec
from nemo_retriever.models.hf_model_registry import HF_MODEL_REVISIONS


def _deep_copy_row(row: pd.Series) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value) if isinstance(value, (dict, list)) else value for key, value in row.to_dict().items()
    }


def _stable_json(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): normalize(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(nested) for nested in item]
        if hasattr(item, "tolist") and callable(item.tolist):
            return normalize(item.tolist())
        if hasattr(item, "item") and callable(item.item):
            return normalize(item.item())
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item

    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if pd.api.types.is_scalar(value) and pd.isna(value):
            continue
        return value
    return None


def _parent_id(row: dict[str, Any], text: str) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    content_metadata = metadata.get("content_metadata") if isinstance(metadata.get("content_metadata"), dict) else {}
    identity = {
        "source": _first_present(
            metadata.get("source_path"),
            row.get("path"),
            row.get("source_id"),
            row.get("source"),
        ),
        "page": _first_present(row.get("_page_number"), row.get("page_number"), metadata.get("page_number")),
        "content_type": _first_present(
            row.get("_content_type"),
            row.get("content_type"),
            content_metadata.get("type"),
        ),
        "element_id": _first_present(
            row.get("element_id"),
            row.get("id"),
            metadata.get("id"),
            content_metadata.get("id"),
        ),
        "bbox": _first_present(
            row.get("_bbox_xyxy_norm"),
            row.get("bbox_xyxy_norm"),
            content_metadata.get("bbox_xyxy_norm"),
        ),
        "source_chunk_index": metadata.get("chunk_index"),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(_stable_json(identity).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingPreparationResult:
    """Prepared frame plus events created by this admission invocation."""

    frame: pd.DataFrame
    input_rows: int
    split_child_positions: frozenset[int]
    split_parent_positions: frozenset[int]


@dataclass(frozen=True)
class EmbeddingInputPolicy:
    """Prepare text rows so every formatted prompt fits the model input limit."""

    tokenizer: ChunkTokenizer
    max_tokens: int
    prefix: str
    prefix_if_missing: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("Embedding input max_tokens must be positive")

    def _format(self, text: str) -> str:
        if self.prefix_if_missing and text.lower().startswith(self.prefix.lower()):
            return text
        return f"{self.prefix}{text}"

    def _formatted_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(self._format(text), add_special_tokens=True))

    def _formatted_token_counts(self, texts: list[str]) -> list[int]:
        encode_batch = getattr(self.tokenizer, "encode_batch", None)
        if callable(encode_batch):
            return [
                len(token_ids)
                for token_ids in encode_batch(
                    [self._format(text) for text in texts],
                    add_special_tokens=True,
                )
            ]
        return [self._formatted_token_count(text) for text in texts]

    def _largest_fitting_end(self, token_ids: list[int], start: int) -> int:
        low = start + 1
        high = min(len(token_ids), start + self.max_tokens)
        best = start
        while low <= high:
            candidate = (low + high) // 2
            text = self.tokenizer.decode(token_ids[start:candidate], skip_special_tokens=False)
            if text and self._formatted_token_count(text) <= self.max_tokens:
                best = candidate
                low = candidate + 1
            else:
                high = candidate - 1
        if best == start:
            raise ValueError(
                "Embedding input cannot fit one content token after applying the model prefix and special tokens"
            )
        return best

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: list[tuple[str, int, int]] = []
        start = 0
        while start < len(token_ids):
            end = self._largest_fitting_end(token_ids, start)
            chunk = self.tokenizer.decode(token_ids[start:end], skip_special_tokens=False)
            if self.tokenizer.encode(chunk, add_special_tokens=False) != token_ids[start:end]:
                raise ValueError(
                    "Embedding input cannot be split without changing its token sequence; "
                    "use the exact reversible tokenizer for this embedding model."
                )
            chunks.append((chunk, start, end))
            start = end
        if "".join(chunk for chunk, _, _ in chunks) != text:
            raise ValueError(
                "Embedding input cannot be split without changing its source text; "
                "use the exact reversible tokenizer for this embedding model."
            )
        return chunks

    def _expand_row(
        self,
        row: pd.Series,
        selected: SelectedEmbeddingText | None,
        parent_tokens: int | None,
    ) -> list[dict[str, Any]]:
        row_copy = _deep_copy_row(row)
        if selected is None or parent_tokens is None or parent_tokens <= self.max_tokens:
            return [row_copy]

        selected_column, text = selected.column, selected.content
        chunks = self._split(text)
        parent_id = _parent_id(row_copy, text)
        expanded: list[dict[str, Any]] = []
        for chunk_index, (chunk, start, end) in enumerate(chunks):
            child = copy.deepcopy(row_copy)
            child[selected_column] = chunk
            if selected_column == "text" and "content" in child:
                child["content"] = chunk
            metadata = child.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                child["metadata"] = metadata
            chunk_id = hashlib.sha256(f"{parent_id}\0{start}\0{end}".encode("utf-8")).hexdigest()
            provenance = EmbeddingSplitProvenance(
                parent_id=parent_id,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
                start_token=start,
                end_token=end,
            )
            metadata.update(provenance.as_metadata(content=chunk))
            expanded.append(child)
        return expanded

    def prepare_with_summary(
        self,
        frame: pd.DataFrame,
        *,
        text_column: str = "text",
        default_modality: str = "text",
    ) -> EmbeddingPreparationResult:
        """Prepare rows and report only split events created by this invocation."""
        input_rows = len(frame.index)

        def unchanged() -> EmbeddingPreparationResult:
            return EmbeddingPreparationResult(
                frame=frame.copy(),
                input_rows=input_rows,
                split_child_positions=frozenset(),
                split_parent_positions=frozenset(),
            )

        if frame.empty:
            return unchanged()

        text_admission = [
            requires_text_admission(row, default_modality=default_modality) for _, row in frame.iterrows()
        ]
        if not any(text_admission):
            return unchanged()

        rows = [row for _, row in frame.iterrows()]
        selected_inputs: list[SelectedEmbeddingText | None] = []
        text_positions: list[int] = []
        texts: list[str] = []
        for position, row in enumerate(rows):
            selected = (
                select_embedding_text(row, text_column=text_column)
                if requires_text_admission(row, default_modality=default_modality)
                else None
            )
            selected_inputs.append(selected)
            if selected is not None:
                text_positions.append(position)
                texts.append(selected.content)

        parent_token_counts: list[int | None] = [None] * len(rows)
        for position, token_count in zip(text_positions, self._formatted_token_counts(texts)):
            parent_token_counts[position] = token_count

        if not any(token_count is not None and token_count > self.max_tokens for token_count in parent_token_counts):
            return unchanged()

        prepared: list[dict[str, Any]] = []
        split_child_positions: set[int] = set()
        split_parent_positions: set[int] = set()
        for row, selected, parent_tokens in zip(rows, selected_inputs, parent_token_counts):
            first_output_position = len(prepared)
            expanded = self._expand_row(row, selected, parent_tokens)
            prepared.extend(expanded)
            if parent_tokens is not None and parent_tokens > self.max_tokens:
                split_parent_positions.add(first_output_position)
                split_child_positions.update(range(first_output_position, first_output_position + len(expanded)))

        return EmbeddingPreparationResult(
            frame=pd.DataFrame(prepared).reset_index(drop=True),
            input_rows=input_rows,
            split_child_positions=frozenset(split_child_positions),
            split_parent_positions=frozenset(split_parent_positions),
        )

    def prepare(
        self,
        frame: pd.DataFrame,
        *,
        text_column: str = "text",
        default_modality: str = "text",
    ) -> pd.DataFrame:
        """Return rows ready for embedding, expanding only overlength text inputs."""
        return self.prepare_with_summary(
            frame,
            text_column=text_column,
            default_modality=default_modality,
        ).frame


def resolve_embedding_input_policy(
    model_id: str,
    *,
    configured_max_tokens: int,
    input_type: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    prefix_if_missing: bool = False,
) -> EmbeddingInputPolicy:
    """Resolve a model-pinned input policy shared by local and remote adapters."""
    if configured_max_tokens <= 0:
        raise ValueError("Configured embedding max length must be positive")
    spec = resolve_embed_model_spec(model_id, revision=revision, hf_cache_dir=cache_dir)
    if spec.max_input_tokens is None:
        raise ValueError(
            f"Embedding model {spec.model_id!r} does not declare a supported input limit; "
            "exact pre-embedding admission cannot be enforced."
        )
    effective_max = min(configured_max_tokens, spec.max_input_tokens)
    is_query = str(input_type).strip().lower() == "query"
    prefix = spec.query_prefix if is_query else spec.document_prefix
    prefix_declared = spec.query_prefix_declared if is_query else spec.document_prefix_declared
    if not prefix_declared:
        prompt_type = "query" if is_query else "passage"
        raise ValueError(
            f"Embedding model {spec.model_id!r} does not declare a {prompt_type} prompt; "
            "exact pre-embedding admission cannot be enforced."
        )
    tokenizer = load_chunk_tokenizer(
        spec.model_id,
        cache_dir=cache_dir,
        revision=spec.revision,
    )
    return EmbeddingInputPolicy(
        tokenizer=tokenizer,
        max_tokens=effective_max,
        prefix=prefix,
        prefix_if_missing=prefix_if_missing,
    )


def resolve_known_embedding_input_policy(
    *,
    model_name: str | None,
    configured_max_tokens: int,
    input_type: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    prefix_if_missing: bool = False,
) -> EmbeddingInputPolicy:
    """Resolve input protection for a pinned or local model, failing closed otherwise."""
    model_id = resolve_embed_model(model_name)
    if model_id not in HF_MODEL_REVISIONS and not Path(model_id).expanduser().is_dir() and revision is None:
        raise ValueError(
            f"Embedding model {model_id!r} is not revision-pinned, so the embedding stage cannot enforce "
            "its tokenizer, prefix, and input limit. Use a registered model, a local checkpoint, or set an "
            "immutable embed_model_revision."
        )
    return resolve_embedding_input_policy(
        model_id,
        configured_max_tokens=configured_max_tokens,
        input_type=input_type,
        revision=revision,
        cache_dir=cache_dir,
        prefix_if_missing=prefix_if_missing,
    )


def configure_embedding_input_policy(kwargs: dict[str, Any]) -> EmbeddingInputPolicy:
    """Resolve and install the shared input policy for an embedding actor."""
    input_type = str(kwargs.get("input_type", "passage"))
    configured_max_tokens = (
        int(kwargs.get("query_max_length", 128))
        if input_type.strip().lower() == "query"
        else int(kwargs.get("max_length", 8192))
    )
    policy = resolve_known_embedding_input_policy(
        model_name=kwargs.get("embed_model_name") or kwargs.get("model_name"),
        configured_max_tokens=configured_max_tokens,
        input_type=input_type,
        revision=kwargs.get("embed_model_revision"),
        cache_dir=kwargs.get("hf_cache_dir"),
        prefix_if_missing=bool(kwargs.get("_embedding_prefix_if_missing", False)),
    )
    kwargs["embedding_input_policy"] = policy
    return policy


def ensure_embedding_input_policy_for_batch(kwargs: dict[str, Any], frame: Any) -> EmbeddingInputPolicy | None:
    """Lazily install text admission only when a batch will use the text route."""
    existing = kwargs.get("embedding_input_policy")
    if existing is not None:
        return existing
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    default_modality = str(kwargs.get("embed_modality", "text"))
    needs_text_policy = any(
        requires_text_admission(row, default_modality=default_modality) for _, row in frame.iterrows()
    )
    return configure_embedding_input_policy(kwargs) if needs_text_policy else None


__all__ = [
    "EmbeddingInputPolicy",
    "EmbeddingPreparationResult",
    "configure_embedding_input_policy",
    "ensure_embedding_input_policy_for_batch",
    "resolve_embedding_input_policy",
    "resolve_known_embedding_input_policy",
]
