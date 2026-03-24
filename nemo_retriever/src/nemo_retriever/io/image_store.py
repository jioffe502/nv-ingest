# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Store extracted images to disk or cloud storage via fsspec."""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from typing import Any, Dict, Optional, Sequence

import pandas as pd
from PIL import Image
from upath import UPath

logger = logging.getLogger(__name__)

# Known limitation: _safe_stem derives the output subdirectory from the
# filename alone (e.g. "report.pdf" → "report/").  Two source files with
# the same basename but different parent directories will write to the same
# subdirectory and may overwrite each other.  This matches the legacy
# nv-ingest store behaviour.  A future PR should incorporate a short hash
# of the full source path to eliminate collisions.


def _safe_stem(name: str) -> str:
    """Derive a filesystem-safe stem from a source path."""
    s = str(name or "").strip() or "document"
    s = os.path.splitext(os.path.basename(s))[0] or "document"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:160] if len(s) > 160 else s


def _decode_and_write(dest: UPath, image_b64: str) -> None:
    """Decode a base64 image and write raw bytes to *dest* via UPath."""
    raw = base64.b64decode(image_b64)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        f.write(raw)


def _crop_and_write(
    dest: UPath,
    page_image: Image.Image,
    bbox_xyxy_norm: Sequence[float],
    image_format: str = "png",
) -> bool:
    """Crop a region from an already-decoded page image and write to *dest*.

    Returns ``True`` on success, ``False`` on skip/failure.
    """
    try:
        w, h = page_image.size
        if w <= 1 or h <= 1:
            return False

        x1n, y1n, x2n, y2n = (float(v) for v in bbox_xyxy_norm)
        x1 = int(min(max(x1n * w, 0), w))
        y1 = int(min(max(y1n * h, 0), h))
        x2 = int(min(max(x2n * w, 0), w))
        y2 = int(min(max(y2n * h, 0), h))
        if x2 <= x1 or y2 <= y1:
            return False

        crop = page_image.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format=image_format.upper())

        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            f.write(buf.getvalue())
        return True
    except Exception as exc:
        logger.warning("Failed to crop and write %s: %s", dest, exc)
        return False


def _decode_page_image(page_image_b64: str) -> Image.Image | None:
    """Decode a base64-encoded page image into a PIL Image (once per row)."""
    try:
        raw = base64.b64decode(page_image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to decode page image: %s", exc)
        return None


def _build_uri_info(
    dest: UPath,
    storage_root: UPath,
    public_base_url: Optional[str],
) -> Dict[str, Optional[str]]:
    """Build a dict with ``stored_image_uri`` and optionally ``stored_image_url``."""
    relative_key = dest.relative_to(storage_root).as_posix()
    info: Dict[str, Optional[str]] = {"stored_image_uri": dest.as_uri()}
    if public_base_url:
        info["stored_image_url"] = f"{public_base_url.rstrip('/')}/{relative_key}"
    return info


def store_extracted_images(
    df: pd.DataFrame,
    *,
    storage_uri: str = "stored_images",
    storage_options: dict[str, Any] | None = None,
    public_base_url: str | None = None,
    store_page_images: bool = True,
    store_tables: bool = True,
    store_charts: bool = True,
    store_infographics: bool = True,
    store_images: bool = True,
    image_format: str = "png",
) -> pd.DataFrame:
    """Pipeline task: store extracted images to disk or cloud storage.

    For each row in the DataFrame:

    * Writes the full page image (from ``page_image["image_b64"]``) when
      *store_page_images* is ``True``.
    * Crops and writes sub-page images for tables / charts / infographics
      using ``bbox_xyxy_norm`` from the page image.
    * Writes natural sub-page images from the ``images`` column.
    * Updates the DataFrame in-place with ``stored_image_uri`` (and an
      optional ``stored_image_url`` when *public_base_url* is set).

    Parameters
    ----------
    df : pd.DataFrame
        Primitives DataFrame produced by the extraction pipeline.
    storage_uri : str
        Base URI for storage.  Local path (``"./output"``) or
        fsspec-compatible URI (``"s3://bucket/prefix"``).
    storage_options : dict | None
        Extra options forwarded to fsspec / UPath (auth keys, endpoint, etc.).
    public_base_url : str | None
        When set, each stored item also receives a ``stored_image_url``
        built as ``{public_base_url}/{relative_key}``.
    store_page_images : bool
        Save full page images.
    store_tables : bool
        Save table crops.
    store_charts : bool
        Save chart crops.
    store_infographics : bool
        Save infographic crops.
    store_images : bool
        Save natural sub-page images from the ``images`` column.
    image_format : str
        Output image format (default ``"png"``).

    Returns
    -------
    pd.DataFrame
        The (mutated) input DataFrame with storage URIs added.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    storage_root = UPath(storage_uri, **(storage_options or {}))
    ext = image_format.lower()

    col_flags: dict[str, str] = {}
    if store_tables:
        col_flags["tables"] = "table"
    if store_charts:
        col_flags["charts"] = "chart"
    if store_infographics:
        col_flags["infographics"] = "infographic"

    for idx, row in df.iterrows():
        try:
            source_path = row.get("path") or ""
            stem = _safe_stem(source_path)
            page_num = row.get("page_number", 1)

            page_image = row.get("page_image")
            page_image_b64: str | None = None
            if isinstance(page_image, dict):
                page_image_b64 = page_image.get("image_b64")

            # Decode the page image once for this row; reused by all crops.
            page_pil: Image.Image | None = None
            needs_crops = bool(col_flags) or store_images
            if needs_crops and isinstance(page_image_b64, str) and page_image_b64:
                page_pil = _decode_page_image(page_image_b64)

            # -- Full page image --
            if store_page_images and isinstance(page_image_b64, str) and page_image_b64:
                dest = storage_root / stem / f"page_{page_num}.{ext}"
                _decode_and_write(dest, page_image_b64)
                uri_info = _build_uri_info(dest, storage_root, public_base_url)
                page_image.update(uri_info)
                df.at[idx, "page_image"] = page_image

            # -- Structured content (tables / charts / infographics) --
            for col_name, type_label in col_flags.items():
                content_list = row.get(col_name)
                if not isinstance(content_list, list):
                    continue
                for item_idx, item in enumerate(content_list):
                    if not isinstance(item, dict):
                        continue
                    dest = storage_root / stem / f"page_{page_num}_{type_label}_{item_idx}.{ext}"
                    item_b64 = item.get("image_b64")
                    if isinstance(item_b64, str) and item_b64:
                        _decode_and_write(dest, item_b64)
                        item.update(_build_uri_info(dest, storage_root, public_base_url))
                    elif page_pil is not None:
                        bbox = item.get("bbox_xyxy_norm")
                        if bbox and len(bbox) == 4:
                            if _crop_and_write(dest, page_pil, bbox, image_format=ext):
                                item.update(_build_uri_info(dest, storage_root, public_base_url))
                df.at[idx, col_name] = content_list

            # -- Natural sub-page images --
            if store_images:
                images_list = row.get("images")
                if isinstance(images_list, list):
                    for img_idx, img_item in enumerate(images_list):
                        if not isinstance(img_item, dict):
                            continue
                        dest = storage_root / stem / f"page_{page_num}_image_{img_idx}.{ext}"
                        img_b64 = img_item.get("image_b64")
                        if isinstance(img_b64, str) and img_b64:
                            _decode_and_write(dest, img_b64)
                            img_item.update(_build_uri_info(dest, storage_root, public_base_url))
                        elif page_pil is not None:
                            bbox = img_item.get("bbox_xyxy_norm")
                            if bbox and len(bbox) == 4:
                                if _crop_and_write(dest, page_pil, bbox, image_format=ext):
                                    img_item.update(_build_uri_info(dest, storage_root, public_base_url))
                    df.at[idx, "images"] = images_list

        except Exception as exc:
            logger.exception("Failed to store images for row %s: %s", idx, exc)

    return df
