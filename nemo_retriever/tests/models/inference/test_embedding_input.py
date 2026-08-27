# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import re
from typing import Sequence
from unittest.mock import Mock, patch

import httpx
import numpy as np
import pandas as pd
import pytest
from nemo_retriever.common.params import EmbedParams, ModelRuntimeParams
from nemo_retriever.common.vdb.records import to_client_vdb_records
from nemo_retriever.models.inference.embedding_input import (
    EmbeddingInputPolicy,
    resolve_embedding_input_policy,
)
from nemo_retriever.models.inference.runtime import embed_text_main_text_embed


class _RejectOneInputEmbedder:
    def embed(self, texts: Sequence[str], *, batch_size: int):
        if any("rejected input" in text for text in texts):
            raise ValueError("input exceeds the model limit")
        return [[float(len(text))] for text in texts]


class _AlwaysFailsEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str], *, batch_size: int):
        self.calls.append(list(texts))
        raise RuntimeError("embedding service unavailable")


class _WhitespaceTokenizer:
    def __init__(self) -> None:
        self._tokens: dict[str, int] = {}
        self._ids: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for token in re.findall(r"\S+\s*|\s+", text):
            if token not in self._tokens:
                token_id = len(self._tokens) + 1
                self._tokens[token] = token_id
                self._ids[token_id] = token
            ids.append(self._tokens[token])
        return ([-1] + ids) if add_special_tokens else ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return "".join(self._ids[token_id] for token_id in token_ids if token_id != -1)


class _NormalizingWhitespaceTokenizer(_WhitespaceTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return super().encode(" ".join(text.split()), add_special_tokens=add_special_tokens)


class _BatchWhitespaceTokenizer(_WhitespaceTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        self.batch_calls += 1
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]


class _NeverCalledTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        raise AssertionError("non-text batches must not be tokenized")

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        raise AssertionError("non-text batches must not be decoded")


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = [ord(character) for character in text]
        return ([-1] + ids) if add_special_tokens else ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return "".join(chr(token_id) for token_id in token_ids if token_id != -1)


class _LiteralSpecialTokenizer:
    _literal = "<SPECIAL>"
    _literal_id = -2

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        while text:
            if text.startswith(self._literal):
                ids.append(self._literal_id)
                text = text[len(self._literal) :]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ([-1] + ids) if add_special_tokens else ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        parts = []
        for token_id in token_ids:
            if token_id == self._literal_id:
                if not skip_special_tokens:
                    parts.append(self._literal)
            elif token_id != -1:
                parts.append(chr(token_id))
        return "".join(parts)


def test_systemic_local_failure_is_not_replayed_as_row_rejections(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="nemo_retriever.models.inference.runtime")
    embedder = _AlwaysFailsEmbedder()

    result = embed_text_main_text_embed(
        pd.DataFrame({"text": ["one", "two"]}),
        model=embedder,
    )

    assert embedder.calls == [["passage: one", "passage: two"]]
    assert result["embedding_v1_counts_by_label"].tolist() == [
        {"unembedded": 1, "failed": 1},
        {"unembedded": 1, "failed": 1},
    ]
    assert "failed=2 embedded=0 unembedded=2" in caplog.text


def test_oversized_middle_row_is_split_below_the_formatted_model_limit(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="nemo_retriever.models.inference.runtime")
    source = pd.DataFrame(
        {
            "text": ["before", "one two three four five", "after"],
            "path": ["doc.pdf"] * 3,
            "page_number": [2, 7, 9],
            "_bbox_xyxy_norm": [None, np.array([0.1, 0.2, 0.8, 0.9]), None],
            "metadata": [
                {"chunk_index": 0},
                {"chunk_index": 4, "element": {"type": "table"}},
                {"chunk_index": 8},
            ],
        }
    )
    policy = EmbeddingInputPolicy(
        tokenizer=_WhitespaceTokenizer(),
        max_tokens=4,
        prefix="passage: ",
    )

    result = embed_text_main_text_embed(
        source,
        model=_RejectOneInputEmbedder(),
        inference_batch_size=3,
        embedding_input_policy=policy,
    )

    assert result["text"].tolist() == [
        "before",
        "one two ",
        "three four ",
        "five",
        "after",
    ]
    assert result["page_number"].tolist() == [2, 7, 7, 7, 9]
    assert result["text_embeddings_1b_v2_has_embedding"].tolist() == [True] * 5

    split_metadata = result.loc[1:3, "metadata"].tolist()
    assert [metadata["chunk_index"] for metadata in split_metadata] == [4, 4, 4]
    assert [metadata["embedding_chunk_index"] for metadata in split_metadata] == [
        0,
        1,
        2,
    ]
    assert [metadata["embedding_chunk_count"] for metadata in split_metadata] == [
        3,
        3,
        3,
    ]
    assert [
        (metadata["embedding_chunk_start_token"], metadata["embedding_chunk_end_token"]) for metadata in split_metadata
    ] == [(0, 2), (2, 4), (4, 5)]
    assert [metadata["element"] for metadata in split_metadata] == [{"type": "table"}] * 3
    assert len({metadata["embedding_chunk_id"] for metadata in split_metadata}) == 3
    assert len({metadata["embedding_parent_id"] for metadata in split_metadata}) == 1
    assert [bbox.tolist() for bbox in result.loc[1:3, "_bbox_xyxy_norm"]] == [[0.1, 0.2, 0.8, 0.9]] * 3

    assert result["embedding_v1_counts_by_label"].tolist() == [
        {"embedded": 1},
        {"embedded": 1, "overlength": 1, "split": 1, "split_child": 1},
        {"embedded": 1, "split_child": 1},
        {"embedded": 1, "split_child": 1},
        {"embedded": 1},
    ]
    assert (
        "Embedding summary: input_rows=3 output_rows=5 overlength=1 split=1 truncated=0 "
        "failed=0 embedded=5 unembedded=0"
    ) in caplog.text
    assert any(record.levelno == logging.WARNING and "overlength=1" in record.message for record in caplog.records)


def test_policy_measures_and_preserves_leading_and_trailing_whitespace() -> None:
    policy = EmbeddingInputPolicy(tokenizer=_CharacterTokenizer(), max_tokens=5, prefix="p")

    result = policy.prepare(pd.DataFrame({"text": [" ab "]}))

    assert result["text"].tolist() == [" ab", " "]
    assert "".join(result["text"]) == " ab "
    assert [
        (metadata["embedding_chunk_start_token"], metadata["embedding_chunk_end_token"])
        for metadata in result["metadata"]
    ] == [(0, 3), (3, 4)]


def test_policy_fails_closed_when_tokenizer_decode_changes_source_text() -> None:
    policy = EmbeddingInputPolicy(tokenizer=_NormalizingWhitespaceTokenizer(), max_tokens=4, prefix="passage: ")

    with pytest.raises(ValueError, match="without changing its (token sequence|source text)"):
        policy.prepare(pd.DataFrame({"text": ["one  two three four five"]}))


def test_policy_batches_admission_and_preserves_non_overlength_rows_exactly() -> None:
    tokenizer = _BatchWhitespaceTokenizer()
    policy = EmbeddingInputPolicy(tokenizer=tokenizer, max_tokens=20, prefix="passage: ")
    source = pd.DataFrame(
        {
            "text": ["short input", "another short input"],
            "page_number": pd.Series([1, 2], dtype="int64"),
            "score": pd.Series([0.25, 0.5], dtype="float64"),
            "metadata": [{"chunk_index": 0}, {"chunk_index": 1}],
        }
    )

    result = policy.prepare(source)

    pd.testing.assert_frame_equal(result, source, check_exact=True)
    assert tokenizer.batch_calls == 1


def test_policy_preserves_non_text_batches_without_tokenizing() -> None:
    policy = EmbeddingInputPolicy(tokenizer=_NeverCalledTokenizer(), max_tokens=8192, prefix="passage: ")
    source = pd.DataFrame(
        {
            "text": ["image placeholder", "another placeholder"],
            "_embed_modality": ["image", "text_image"],
            "metadata": [{"page_number": 1}, {"page_number": 2}],
        }
    )

    result = policy.prepare(source)

    pd.testing.assert_frame_equal(result, source, check_exact=True)


def test_policy_preserves_mixed_short_batches_exactly() -> None:
    tokenizer = _BatchWhitespaceTokenizer()
    policy = EmbeddingInputPolicy(tokenizer=tokenizer, max_tokens=20, prefix="passage: ")
    source = pd.DataFrame(
        {
            "text": ["short text", "image placeholder"],
            "_embed_modality": ["text", "image"],
            "metadata": [{"page_number": 1}, {"page_number": 2}],
        }
    )

    result = policy.prepare(source)

    pd.testing.assert_frame_equal(result, source, check_exact=True)
    assert tokenizer.batch_calls == 1


def test_policy_preserves_literal_special_token_text() -> None:
    text = "abc<SPECIAL>def"
    policy = EmbeddingInputPolicy(tokenizer=_LiteralSpecialTokenizer(), max_tokens=5, prefix="p")

    result = policy.prepare(pd.DataFrame({"text": [text]}))

    assert "".join(result["text"]) == text
    assert "<SPECIAL>" in "".join(result["text"])


def test_policy_resolver_caps_runtime_length_at_checkpoint_support(monkeypatch, tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"llama_bidirec","architectures":["LlamaBidirectionalModel"],'
        '"hidden_size":2048,"pooling":"avg"}',
        encoding="utf-8",
    )
    (tmp_path / "config_sentence_transformers.json").write_text(
        '{"prompts":{"query":"query: ","document":"document: "}}',
        encoding="utf-8",
    )
    (tmp_path / "sentence_bert_config.json").write_text('{"max_seq_length":8192}', encoding="utf-8")
    tokenizer = _WhitespaceTokenizer()
    monkeypatch.setattr(
        "nemo_retriever.models.inference.embedding_input.load_chunk_tokenizer",
        lambda *args, **kwargs: tokenizer,
    )

    policy = resolve_embedding_input_policy(
        str(tmp_path),
        configured_max_tokens=131_072,
        input_type="passage",
    )

    assert policy.max_tokens == 8192
    assert policy.prefix == "document: "
    assert policy.tokenizer is tokenizer


def test_unpinned_model_fails_closed_before_embedding() -> None:
    from nemo_retriever.models.inference.embedding_input import (
        resolve_known_embedding_input_policy,
    )

    with pytest.raises(ValueError, match="is not revision-pinned"):
        resolve_known_embedding_input_policy(
            model_name="custom/unpinned-model",
            configured_max_tokens=8192,
            input_type="passage",
        )


def test_remote_actor_passes_the_resolved_input_policy_to_embedding(
    monkeypatch,
) -> None:
    from nemo_retriever.operators.embed import cpu_operator

    policy = object()
    embed = Mock(spec=cpu_operator.embed_text_main_text_embed, return_value=pd.DataFrame({"text": ["prepared"]}))

    def configure(kwargs):
        kwargs["embedding_input_policy"] = policy
        return policy

    monkeypatch.setattr(cpu_operator, "configure_embedding_input_policy", configure)
    monkeypatch.setattr(cpu_operator, "embed_text_main_text_embed", embed)
    monkeypatch.setattr(cpu_operator, "probe_endpoint", Mock(spec=cpu_operator.probe_endpoint))

    actor = cpu_operator._BatchEmbedCPUActor(
        params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            embed_invoke_url="http://embedding.test/v1",
        )
    )
    actor.process(pd.DataFrame({"text": ["source"]}))

    assert embed.call_args.kwargs["embedding_input_policy"] is policy


def test_query_actor_resolves_the_shared_policy_at_query_max_length(
    monkeypatch,
) -> None:
    from nemo_retriever.models.inference import embedding_input

    policy = object()
    resolver = Mock(spec=embedding_input.resolve_known_embedding_input_policy, return_value=policy)
    monkeypatch.setattr(embedding_input, "resolve_known_embedding_input_policy", resolver)

    kwargs = {
        "model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "input_type": "query",
        "query_max_length": 257,
        "max_length": 8192,
    }
    result = embedding_input.configure_embedding_input_policy(kwargs)

    assert result is policy
    assert kwargs["embedding_input_policy"] is policy
    assert resolver.call_args.kwargs["configured_max_tokens"] == 257
    assert resolver.call_args.kwargs["input_type"] == "query"


def test_local_actor_does_not_raise_vllm_above_checkpoint_support(monkeypatch) -> None:
    from nemo_retriever.models import create_local_embedder as create_local_embedder_factory
    from nemo_retriever.operators.embed import gpu_operator

    policy = EmbeddingInputPolicy(tokenizer=_WhitespaceTokenizer(), max_tokens=8192, prefix="passage: ")
    create_local_embedder = Mock(spec=create_local_embedder_factory, return_value=object())
    monkeypatch.setattr(gpu_operator, "configure_embedding_input_policy", lambda kwargs: policy)
    monkeypatch.setattr("nemo_retriever.models.create_local_embedder", create_local_embedder)
    monkeypatch.setattr("nemo_retriever.models.warmup_registry.get_warmed_model", lambda name: None)

    gpu_operator._BatchEmbedActor(
        params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            runtime=ModelRuntimeParams(max_length=131_072),
        )
    )

    assert create_local_embedder.call_args.kwargs["max_length"] == 8192


def test_remote_request_failure_is_not_replayed_or_persisted_with_response_body() -> None:
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts = json.loads(request.content)["input"]
        calls.append(prompts)
        return httpx.Response(400, text="echoed document: rejected input")

    client_factory = httpx.Client
    source = pd.DataFrame({"text": ["valid before", "rejected input", "valid after"]})
    with patch(
        "httpx.Client",
        side_effect=lambda **kwargs: client_factory(transport=httpx.MockTransport(handler)),
    ):
        result = embed_text_main_text_embed(
            source,
            embedding_endpoint="http://embedding.test/v1",
            model_name="test/model",
            inference_batch_size=3,
        )

    assert result["text_embeddings_1b_v2_has_embedding"].tolist() == [False, False, False]
    assert "echoed document" not in result.loc[1, "text_embeddings_1b_v2"]["error"]
    assert result.loc[1, "text_embeddings_1b_v2"]["error"] == (
        "RuntimeError: embedding batch failed; inspect embed-stage logs for the cause"
    )
    assert calls == [["valid before", "rejected input", "valid after"]]


def test_client_side_policy_disables_remote_silent_truncation() -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    policy = EmbeddingInputPolicy(tokenizer=_WhitespaceTokenizer(), max_tokens=4, prefix="passage: ")
    with patch("httpx.Client", return_value=client):
        result = embed_text_main_text_embed(
            pd.DataFrame({"text": ["short input"]}),
            embedding_endpoint="http://embedding.test/v1",
            model_name="test/model",
            embedding_input_policy=policy,
        )

    assert result["text_embeddings_1b_v2_has_embedding"].tolist() == [True]
    assert payloads[0]["truncate"] == "NONE"


def test_split_identity_is_stable_across_batch_composition() -> None:
    policy = EmbeddingInputPolicy(tokenizer=_WhitespaceTokenizer(), max_tokens=4, prefix="passage: ")
    oversized = {
        "text": "one two three four five",
        "path": "doc.pdf",
        "page_number": 7,
        "metadata": {"chunk_index": 4},
    }
    alone = policy.prepare(pd.DataFrame([oversized]))
    with_neighbors = policy.prepare(pd.DataFrame([{"text": "before"}, oversized, {"text": "after"}]))

    assert [metadata["embedding_chunk_id"] for metadata in alone["metadata"]] == [
        metadata["embedding_chunk_id"] for metadata in with_neighbors.loc[1:3, "metadata"]
    ]


def test_local_and_remote_adapters_apply_the_same_split_policy() -> None:
    source = pd.DataFrame({"text": ["before", "one two three four five", "after"]})
    policy = EmbeddingInputPolicy(tokenizer=_WhitespaceTokenizer(), max_tokens=4, prefix="passage: ")
    local = embed_text_main_text_embed(
        source,
        model=_RejectOneInputEmbedder(),
        inference_batch_size=3,
        embedding_input_policy=policy,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        prompts = json.loads(request.content)["input"]
        return httpx.Response(
            200,
            json={"data": [{"index": index, "embedding": [float(len(text))]} for index, text in enumerate(prompts)]},
        )

    client_factory = httpx.Client
    with patch(
        "httpx.Client",
        side_effect=lambda **kwargs: client_factory(transport=httpx.MockTransport(handler)),
    ):
        remote = embed_text_main_text_embed(
            source,
            embedding_endpoint="http://embedding.test/v1",
            model_name="test/model",
            inference_batch_size=3,
            embedding_input_policy=policy,
        )

    assert (
        local["text"].tolist()
        == remote["text"].tolist()
        == [
            "before",
            "one two ",
            "three four ",
            "five",
            "after",
        ]
    )
    assert not any(column.startswith("_embedding_input_") for column in local.columns)
    assert not any(column.startswith("_embedding_input_") for column in remote.columns)
    assert local["text_embeddings_1b_v2_has_embedding"].tolist() == [True] * 5
    assert remote["text_embeddings_1b_v2_has_embedding"].tolist() == [True] * 5


def test_split_rows_reach_vdb_conversion_with_complete_coverage() -> None:
    source = pd.DataFrame(
        {
            "text": ["before", "one two three four five", "after"],
            "path": ["doc.pdf"] * 3,
            "page_number": [1, 2, 3],
            "metadata": [{}, {}, {}],
        }
    )
    policy = EmbeddingInputPolicy(tokenizer=_WhitespaceTokenizer(), max_tokens=4, prefix="passage: ")

    embedded = embed_text_main_text_embed(
        source,
        model=_RejectOneInputEmbedder(),
        inference_batch_size=3,
        embedding_input_policy=policy,
    )
    records = to_client_vdb_records(embedded)

    assert len(records) == 1
    assert len(records[0]) == 5
    assert [record["metadata"]["content_metadata"]["page_number"] for record in records[0]] == [1, 2, 2, 2, 3]
    assert all(record["metadata"]["embedding"] for record in records[0])
