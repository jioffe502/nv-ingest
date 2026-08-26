# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PROTOTYPE: state model for lossless image-handle transport.

Question: can one content-addressed page-image handle own exact pixel bytes
across a Ray boundary, fail closed on missing or corrupt data, and remain alive
until the terminal VDB receipt makes cleanup safe?

This module is intentionally throwaway. If the model is accepted, absorb the
contract into production types and delete this prototype.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field, replace
from enum import StrEnum


class Phase(StrEnum):
    INLINE = "inline"
    STORED = "stored"
    PUBLISHED = "published"
    REHYDRATED = "rehydrated"
    EMBEDDED = "embedded"
    COMMITTED = "committed"
    RELEASED = "released"
    FAILED = "failed"


@dataclass(frozen=True)
class ImageHandle:
    version: int
    uri: str
    sha256: str
    byte_length: int
    media_type: str


@dataclass(frozen=True)
class PrototypeState:
    phase: Phase
    inline_b64: str | None
    handle: ImageHandle | None = None
    blobs: dict[str, bytes] = field(default_factory=dict)
    rehydrated_b64: str | None = None
    embedding_digest: str | None = None
    receipt_committed: bool = False
    failure: str | None = None


def initial_state() -> PrototypeState:
    raw = b"lossless-page-image-prototype\x00\x01\x02"
    return PrototypeState(phase=Phase.INLINE, inline_b64=base64.b64encode(raw).decode("ascii"))


def _fail(state: PrototypeState, message: str) -> PrototypeState:
    return replace(state, phase=Phase.FAILED, failure=message, rehydrated_b64=None)


def _decoded_inline(state: PrototypeState) -> bytes:
    if state.inline_b64 is None:
        raise ValueError("inline image is absent")
    return base64.b64decode(state.inline_b64, validate=True)


def persist(state: PrototypeState) -> PrototypeState:
    if state.phase not in {Phase.INLINE, Phase.STORED}:
        return _fail(state, f"persist is illegal in phase {state.phase}")
    if state.phase == Phase.STORED:
        return state

    try:
        raw = _decoded_inline(state)
    except Exception as exc:  # pragma: no cover - interactive prototype
        return _fail(state, f"invalid inline base64: {exc}")
    digest = hashlib.sha256(raw).hexdigest()
    uri = f"memory://embedding-images/{digest}.png"
    handle = ImageHandle(
        version=1,
        uri=uri,
        sha256=digest,
        byte_length=len(raw),
        media_type="image/png",
    )
    blobs = dict(state.blobs)
    blobs.setdefault(uri, raw)
    return replace(state, phase=Phase.STORED, inline_b64=None, handle=handle, blobs=blobs, failure=None)


def publish(state: PrototypeState) -> PrototypeState:
    if state.phase != Phase.STORED or state.handle is None:
        return _fail(state, "publish requires a stored handle")
    return replace(state, phase=Phase.PUBLISHED, failure=None)


def rehydrate(state: PrototypeState) -> PrototypeState:
    if state.phase != Phase.PUBLISHED or state.handle is None:
        return _fail(state, "rehydrate requires a published handle")
    raw = state.blobs.get(state.handle.uri)
    if raw is None:
        return _fail(state, "image object is missing; embedding must not run")
    if len(raw) != state.handle.byte_length:
        return _fail(state, "image byte length changed; embedding must not run")
    if hashlib.sha256(raw).hexdigest() != state.handle.sha256:
        return _fail(state, "image digest changed; embedding must not run")
    return replace(
        state,
        phase=Phase.REHYDRATED,
        rehydrated_b64=base64.b64encode(raw).decode("ascii"),
        failure=None,
    )


def embed(state: PrototypeState) -> PrototypeState:
    if state.phase != Phase.REHYDRATED or state.rehydrated_b64 is None:
        return _fail(state, "embed requires a verified rehydrated image")
    raw = base64.b64decode(state.rehydrated_b64, validate=True)
    return replace(
        state,
        phase=Phase.EMBEDDED,
        rehydrated_b64=None,
        embedding_digest=hashlib.sha256(b"embedding:" + raw).hexdigest(),
        failure=None,
    )


def commit_receipt(state: PrototypeState) -> PrototypeState:
    if state.phase != Phase.EMBEDDED:
        return _fail(state, "receipt requires a successful embedding")
    return replace(state, phase=Phase.COMMITTED, receipt_committed=True, failure=None)


def release(state: PrototypeState) -> PrototypeState:
    if state.phase != Phase.COMMITTED or not state.receipt_committed or state.handle is None:
        return _fail(state, "release requires a committed terminal receipt")
    blobs = dict(state.blobs)
    blobs.pop(state.handle.uri, None)
    return replace(state, phase=Phase.RELEASED, blobs=blobs, failure=None)


def corrupt_blob(state: PrototypeState) -> PrototypeState:
    if state.handle is None or state.handle.uri not in state.blobs:
        return _fail(state, "there is no stored blob to corrupt")
    blobs = dict(state.blobs)
    blobs[state.handle.uri] = blobs[state.handle.uri] + b"corrupt"
    return replace(state, blobs=blobs)


def delete_blob(state: PrototypeState) -> PrototypeState:
    if state.handle is None:
        return _fail(state, "there is no handle whose blob can be deleted")
    blobs = dict(state.blobs)
    blobs.pop(state.handle.uri, None)
    return replace(state, blobs=blobs)
