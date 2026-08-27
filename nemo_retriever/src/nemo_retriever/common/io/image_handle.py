# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless image references for the reshape-to-embedding boundary.

An embedding image handle is deliberately self-verifying.  The producer
publishes a URI together with the exact byte length and SHA-256 digest; the
embedding worker verifies both before it exposes bytes to a visual model.
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
from typing import Any, Iterable, Mapping, Sequence

import fsspec

EMBEDDING_IMAGE_HANDLE_FIELD = "_embedding_image_handle"
IMAGE_HANDLE_CONTAINER_FIELD = "image_handle"
IMAGE_HANDLE_VERSION = 1
IMAGE_HANDLE_AUDIT_ENV = "NEMO_RETRIEVER_IMAGE_HANDLE_AUDIT"

_SUPPORTED_MEDIA_TYPES = frozenset({"image/png", "image/jpeg"})


class ImageHandleError(ValueError):
    """Raised when an embedding image reference cannot be trusted."""


def image_transport_audit_enabled() -> bool:
    """Return whether qualification-grade handle accounting should be retained."""

    return os.environ.get(IMAGE_HANDLE_AUDIT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def build_image_handle(
    raw: bytes,
    *,
    uri: str,
    media_type: str,
    crop_bbox_xyxy_norm: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build a versioned handle for bytes already persisted at *uri*."""

    if not isinstance(raw, bytes):
        raise TypeError("raw image payload must be bytes")
    handle: dict[str, Any] = {
        "version": IMAGE_HANDLE_VERSION,
        "uri": str(uri),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_length": len(raw),
        "media_type": str(media_type),
        "crop_bbox_xyxy_norm": None,
    }
    if crop_bbox_xyxy_norm is not None:
        handle["crop_bbox_xyxy_norm"] = _normalize_crop_bbox(crop_bbox_xyxy_norm)
    return validate_image_handle(handle)


def handle_with_crop(handle: Mapping[str, Any], bbox_xyxy_norm: Sequence[float] | None) -> dict[str, Any]:
    """Return a validated copy of *handle* selecting a normalized crop."""

    out = validate_image_handle(handle)
    out["crop_bbox_xyxy_norm"] = None if bbox_xyxy_norm is None else _normalize_crop_bbox(bbox_xyxy_norm)
    return out


def image_handle_from_container(value: Any) -> dict[str, Any] | None:
    """Return a copied handle from a nested image container, if present."""

    if not isinstance(value, Mapping):
        return None
    handle = value.get(IMAGE_HANDLE_CONTAINER_FIELD)
    return validate_image_handle(handle) if isinstance(handle, Mapping) else None


def image_transport_stats(
    *,
    row_count: int,
    inline_values: Iterable[Any] = (),
    handle_values: Iterable[Any] = (),
) -> dict[str, int]:
    """Return logical and unique byte accounting for one transport boundary."""

    inline_payloads = [value.strip() for value in inline_values if isinstance(value, str) and value.strip()]
    handles = [value for value in handle_values if isinstance(value, Mapping)]
    valid_identities = [
        (value.get("uri"), value.get("sha256"), value.get("byte_length"))
        for value in handles
        if isinstance(value.get("uri"), str)
        and isinstance(value.get("sha256"), str)
        and isinstance(value.get("byte_length"), int)
        and not isinstance(value.get("byte_length"), bool)
        and value.get("byte_length", -1) >= 0
    ]
    unique_identities = set(valid_identities)
    return {
        "rows": int(row_count),
        "inline_rows": len(inline_payloads),
        "inline_base64_chars": sum(len(value) for value in inline_payloads),
        "handle_rows": len(handles),
        "logical_handle_bytes": sum(int(identity[2]) for identity in valid_identities),
        "unique_handles": len(unique_identities),
        "unique_handle_bytes": sum(int(identity[2]) for identity in unique_identities),
    }


def validate_image_handle(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize an untrusted image-handle mapping."""

    if not isinstance(value, Mapping):
        raise ImageHandleError("Embedding image handle must be a mapping")
    if value.get("version") != IMAGE_HANDLE_VERSION:
        raise ImageHandleError(f"Unsupported embedding image handle version: {value.get('version')!r}")

    uri = value.get("uri")
    if not isinstance(uri, str) or not uri.strip():
        raise ImageHandleError("Embedding image handle URI is missing")

    digest = value.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ImageHandleError("Embedding image handle SHA-256 is invalid")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ImageHandleError("Embedding image handle SHA-256 is invalid") from exc

    byte_length = value.get("byte_length")
    if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length < 0:
        raise ImageHandleError("Embedding image handle byte length is invalid")

    media_type = value.get("media_type")
    if media_type not in _SUPPORTED_MEDIA_TYPES:
        raise ImageHandleError(f"Unsupported embedding image media type: {media_type!r}")

    crop = value.get("crop_bbox_xyxy_norm")
    return {
        "version": IMAGE_HANDLE_VERSION,
        "uri": uri.strip(),
        "sha256": digest.lower(),
        "byte_length": byte_length,
        "media_type": media_type,
        "crop_bbox_xyxy_norm": None if crop is None else _normalize_crop_bbox(crop),
    }


def load_verified_image_b64(value: Mapping[str, Any]) -> str:
    """Read, verify, and optionally crop an image handle for one embedding call."""

    handle = validate_image_handle(value)
    try:
        with fsspec.open(handle["uri"], mode="rb") as stream:
            raw = stream.read()
    except Exception as exc:
        raise ImageHandleError(f"Failed to read embedding image handle {handle['uri']!r}: {exc}") from exc

    if len(raw) != handle["byte_length"]:
        raise ImageHandleError(
            f"Embedding image byte length changed for {handle['uri']!r}: "
            f"expected {handle['byte_length']}, got {len(raw)}"
        )
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != handle["sha256"]:
        raise ImageHandleError(f"Embedding image digest changed for {handle['uri']!r}")

    image_b64 = base64.b64encode(raw).decode("ascii")
    crop = handle["crop_bbox_xyxy_norm"]
    if crop is None:
        return image_b64

    # Import lazily so text-only deployments do not acquire the image/OCR
    # dependency chain merely by importing the embedding module.
    from nemo_retriever.common.modality.ocr.shared import _crop_b64_image_by_norm_bbox

    cropped_b64, _ = _crop_b64_image_by_norm_bbox(image_b64, bbox_xyxy_norm=crop)
    if not cropped_b64:
        raise ImageHandleError(f"Embedding image crop failed for {handle['uri']!r}")
    return cropped_b64


def _normalize_crop_bbox(value: Sequence[float]) -> list[float]:
    if isinstance(value, (str, bytes)) or len(value) != 4:
        raise ImageHandleError("Embedding image crop bbox must contain four coordinates")
    try:
        bbox = [float(coordinate) for coordinate in value]
    except (TypeError, ValueError) as exc:
        raise ImageHandleError("Embedding image crop bbox is invalid") from exc
    if not all(math.isfinite(coordinate) for coordinate in bbox):
        raise ImageHandleError("Embedding image crop bbox is invalid")
    if not (0.0 <= bbox[0] < bbox[2] <= 1.0 and 0.0 <= bbox[1] < bbox[3] <= 1.0):
        raise ImageHandleError("Embedding image crop bbox must be normalized xyxy coordinates")
    return bbox
