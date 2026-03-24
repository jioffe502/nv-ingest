# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_retriever.io.image_store."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pandas as pd
import pytest

from nemo_retriever.io.image_store import _safe_stem, store_extracted_images
from nemo_retriever.params import StoreParams


def _make_tiny_png_b64(width: int = 4, height: int = 4, color=(255, 0, 0)) -> str:
    """Create a minimal PNG image encoded as base64."""
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# _safe_stem
# ---------------------------------------------------------------------------


class TestSafeStem:
    def test_normal_path(self):
        assert _safe_stem("/data/docs/report.pdf") == "report"

    def test_special_characters(self):
        assert _safe_stem("my file (copy).pdf") == "my_file_copy_"

    def test_empty_string(self):
        assert _safe_stem("") == "document"

    def test_none_value(self):
        assert _safe_stem(None) == "document"

    def test_long_name_truncated(self):
        long_name = "a" * 200 + ".pdf"
        result = _safe_stem(long_name)
        assert len(result) <= 160

    def test_slashes_only(self):
        assert _safe_stem("///") == "document"


# ---------------------------------------------------------------------------
# store_extracted_images — page images
# ---------------------------------------------------------------------------


class TestStorePageImages:
    def test_writes_file_and_updates_uri(self, tmp_path: Path):
        b64 = _make_tiny_png_b64()
        df = pd.DataFrame(
            [
                {
                    "path": "/docs/test.pdf",
                    "page_number": 1,
                    "page_image": {"image_b64": b64, "encoding": "png"},
                    "tables": [],
                    "charts": [],
                    "infographics": [],
                    "images": [],
                    "metadata": {},
                }
            ]
        )

        result = store_extracted_images(df, storage_uri=str(tmp_path))
        expected_file = tmp_path / "test" / "page_1.png"
        assert expected_file.exists()
        assert expected_file.stat().st_size > 0

        page_img = result.iloc[0]["page_image"]
        assert "stored_image_uri" in page_img
        assert page_img["stored_image_uri"].startswith("file://")

    def test_skips_when_no_page_image(self, tmp_path: Path):
        df = pd.DataFrame(
            [
                {
                    "path": "/docs/test.pdf",
                    "page_number": 1,
                    "page_image": None,
                    "tables": [],
                    "charts": [],
                    "infographics": [],
                    "images": [],
                    "metadata": {},
                }
            ]
        )
        result = store_extracted_images(df, storage_uri=str(tmp_path))
        assert not any(tmp_path.rglob("*.png"))
        assert result.iloc[0]["page_image"] is None

    def test_disabled_flag(self, tmp_path: Path):
        b64 = _make_tiny_png_b64()
        df = pd.DataFrame(
            [
                {
                    "path": "/docs/test.pdf",
                    "page_number": 1,
                    "page_image": {"image_b64": b64},
                    "tables": [],
                    "charts": [],
                    "infographics": [],
                    "images": [],
                    "metadata": {},
                }
            ]
        )
        store_extracted_images(df, storage_uri=str(tmp_path), store_page_images=False)
        assert not any(tmp_path.rglob("page_*.png"))


# ---------------------------------------------------------------------------
# store_extracted_images — structured content crops
# ---------------------------------------------------------------------------


class TestStoreStructuredContent:
    def test_crop_table_from_page_image(self, tmp_path: Path):
        b64 = _make_tiny_png_b64(width=100, height=100)
        df = pd.DataFrame(
            [
                {
                    "path": "/docs/report.pdf",
                    "page_number": 2,
                    "page_image": {"image_b64": b64, "encoding": "png"},
                    "tables": [
                        {"text": "col1|col2", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.9]},
                    ],
                    "charts": [],
                    "infographics": [],
                    "images": [],
                    "metadata": {},
                }
            ]
        )

        result = store_extracted_images(df, storage_uri=str(tmp_path))
        expected_file = tmp_path / "report" / "page_2_table_0.png"
        assert expected_file.exists()

        table_item = result.iloc[0]["tables"][0]
        assert "stored_image_uri" in table_item

    def test_direct_image_b64_preferred(self, tmp_path: Path):
        page_b64 = _make_tiny_png_b64(color=(255, 0, 0))
        item_b64 = _make_tiny_png_b64(color=(0, 255, 0))
        df = pd.DataFrame(
            [
                {
                    "path": "/docs/test.pdf",
                    "page_number": 1,
                    "page_image": {"image_b64": page_b64},
                    "tables": [
                        {"text": "data", "image_b64": item_b64, "bbox_xyxy_norm": [0, 0, 1, 1]},
                    ],
                    "charts": [],
                    "infographics": [],
                    "images": [],
                    "metadata": {},
                }
            ]
        )
        result = store_extracted_images(df, storage_uri=str(tmp_path))
        table_item = result.iloc[0]["tables"][0]
        assert "stored_image_uri" in table_item

        # Verify the file was written from item_b64, not cropped from page
        written_bytes = (tmp_path / "test" / "page_1_table_0.png").read_bytes()
        assert written_bytes == base64.b64decode(item_b64)

    def test_selective_flags_skip_tables(self, tmp_path: Path):
        b64 = _make_tiny_png_b64(width=100, height=100)
        df = pd.DataFrame(
            [
                {
                    "path": "/docs/test.pdf",
                    "page_number": 1,
                    "page_image": {"image_b64": b64},
                    "tables": [{"text": "t", "bbox_xyxy_norm": [0.1, 0.1, 0.5, 0.5]}],
                    "charts": [{"text": "c", "bbox_xyxy_norm": [0.1, 0.1, 0.5, 0.5]}],
                    "infographics": [],
                    "images": [],
                    "metadata": {},
                }
            ]
        )
        store_extracted_images(df, storage_uri=str(tmp_path), store_tables=False)
        files = list(tmp_path.rglob("*.png"))
        names = [f.name for f in files]
        assert not any("table" in n for n in names)
        assert any("chart" in n for n in names)


# ---------------------------------------------------------------------------
# store_extracted_images — natural sub-page images
# ---------------------------------------------------------------------------


class TestStoreNaturalImages:
    def test_writes_from_image_b64(self, tmp_path: Path):
        img_b64 = _make_tiny_png_b64(color=(0, 0, 255))
        df = pd.DataFrame(
            [
                {
                    "path": "/docs/test.pdf",
                    "page_number": 1,
                    "page_image": None,
                    "tables": [],
                    "charts": [],
                    "infographics": [],
                    "images": [{"image_b64": img_b64, "bbox_xyxy_norm": [0, 0, 1, 1]}],
                    "metadata": {},
                }
            ]
        )
        result = store_extracted_images(df, storage_uri=str(tmp_path))
        expected_file = tmp_path / "test" / "page_1_image_0.png"
        assert expected_file.exists()
        assert result.iloc[0]["images"][0].get("stored_image_uri") is not None


# ---------------------------------------------------------------------------
# store_extracted_images — edge cases
# ---------------------------------------------------------------------------


class TestStoreEdgeCases:
    def test_empty_dataframe(self, tmp_path: Path):
        df = pd.DataFrame()
        result = store_extracted_images(df, storage_uri=str(tmp_path))
        assert result.empty

    def test_public_base_url(self, tmp_path: Path):
        b64 = _make_tiny_png_b64()
        df = pd.DataFrame(
            [
                {
                    "path": "/docs/test.pdf",
                    "page_number": 1,
                    "page_image": {"image_b64": b64},
                    "tables": [],
                    "charts": [],
                    "infographics": [],
                    "images": [],
                    "metadata": {},
                }
            ]
        )
        result = store_extracted_images(
            df,
            storage_uri=str(tmp_path),
            public_base_url="https://cdn.example.com/assets",
        )
        page_img = result.iloc[0]["page_image"]
        assert page_img["stored_image_url"] == "https://cdn.example.com/assets/test/page_1.png"

    def test_multiple_pages(self, tmp_path: Path):
        b64 = _make_tiny_png_b64()
        df = pd.DataFrame(
            [
                {
                    "path": "/docs/test.pdf",
                    "page_number": i,
                    "page_image": {"image_b64": b64},
                    "tables": [],
                    "charts": [],
                    "infographics": [],
                    "images": [],
                    "metadata": {},
                }
                for i in range(1, 4)
            ]
        )
        store_extracted_images(df, storage_uri=str(tmp_path))
        for i in range(1, 4):
            assert (tmp_path / "test" / f"page_{i}.png").exists()


# ---------------------------------------------------------------------------
# StoreParams model
# ---------------------------------------------------------------------------


class TestStoreParams:
    def test_defaults(self):
        p = StoreParams()
        assert p.storage_uri == "stored_images"
        assert p.store_page_images is True
        assert p.store_tables is True
        assert p.image_format == "png"

    def test_overrides(self):
        p = StoreParams(storage_uri="s3://bucket/prefix", store_tables=False, image_format="jpeg")
        assert p.storage_uri == "s3://bucket/prefix"
        assert p.store_tables is False
        assert p.image_format == "jpeg"
