# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for multimodal embedding helpers and explode_content_to_rows.
"""

from __future__ import annotations

import base64
import hashlib
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from nemo_retriever.common.io.image_handle import (
    EMBEDDING_IMAGE_HANDLE_FIELD,
    ImageHandleError,
    image_transport_stats,
)

# ---------------------------------------------------------------------------
# Pure helpers from main_text_embed (no transitive-import issues)
# ---------------------------------------------------------------------------
from nemo_retriever.models.inference.main_text_embed import (
    TextEmbeddingConfig,
    _format_image_input_string,
    _format_text_image_pair_input_string,
    _image_from_row,
    _multimodal_callable_runner,
    create_text_embeddings_for_df,
)

# ---------------------------------------------------------------------------
# Stub heavy internal modules so the content-transform helpers can be imported
# in lightweight CI (only pytest, pandas, pydantic, pyyaml).
#
# Older ingest modules can pull in ray, torch, nemotron_*, nemo_retriever.common.api,
# etc. And inprocess.py itself imports model/local (torch, nemotron_*),
# page_elements, ocr, and pdf.extract — each with their own heavy transitive
# deps.
#
# Rather than chasing every third-party leaf dependency, we pre-populate
# sys.modules for the heavy *internal* nemo_retriever sub-packages with MagicMock.
# This cuts off the entire transitive tree at the root.
# ---------------------------------------------------------------------------
_HEAVY_INTERNAL = [
    # -- sibling ingest modes (prevents batch.py from loading) ------------------
    "nemo_retriever.ingest_modes.batch",
    # -- model / ML packages (torch, nemotron_*, transformers) ---------------
    "nemo_retriever.models.local",
    "nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder",
    "nemo_retriever.models.local.nemotron_page_elements_v3",
    "nemo_retriever.models.local.nemotron_ocr_v1",
    "nemo_retriever.models.local.nemotron_table_structure_v1",
    # -- detection / OCR (nemotron_page_elements_v3, PIL, requests) ----------
    "nemo_retriever.page_elements",
    "nemo_retriever.operators.extract.page_elements.page_elements",
    "nemo_retriever.ocr",
    "nemo_retriever.operators.extract.ocr.ocr",
    # -- table (nemo_retriever.common.api → cv2) ----------------------------------------
    "nemo_retriever.table",
    "nemo_retriever.operators.extract.table.table_detection",
    "nemo_retriever.table.stage",
    # -- PDF (pypdfium2 and heavy extraction dependencies) -------------------
    "nemo_retriever.pdf",
    "nemo_retriever.operators.extract.pdf.extract",
    "nemo_retriever.operators.extract.pdf.split",
]
# Track which modules we injected (vs. ones already loaded) so we can
# remove only our stubs after the import, preventing leaks into other
# test files that need the real modules.
_injected: list[str] = []
for _mod_name in _HEAVY_INTERNAL:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()
        _injected.append(_mod_name)

from nemo_retriever.common.modality.content_transforms import (  # noqa: E402
    collapse_content_to_page_rows,
    explode_content_to_rows,
)

# Clean up injected mocks so they don't poison imports in other test files.
for _mod_name in _injected:
    sys.modules.pop(_mod_name, None)
del _injected


# ===================================================================
# Pure helpers
# ===================================================================


class TestImageFromRow:
    def test_returns_b64_when_present(self):
        row = pd.Series({"_image_b64": "abc123"})
        assert _image_from_row(row) == "abc123"

    @pytest.mark.parametrize("value", [None, "", "   ", 42])
    def test_returns_none_for_missing_empty_whitespace(self, value):
        data = {"_image_b64": value} if value is not None else {}
        row = pd.Series(data)
        assert _image_from_row(row) is None

    def test_rehydrates_and_crops_verified_handle(self, tmp_path: Path):
        buf = io.BytesIO()
        Image.new("RGB", (100, 80), color=(255, 0, 0)).save(buf, format="PNG")
        raw = buf.getvalue()
        image_path = tmp_path / "page.png"
        image_path.write_bytes(raw)
        row = pd.Series(
            {
                EMBEDDING_IMAGE_HANDLE_FIELD: {
                    "version": 1,
                    "uri": image_path.as_uri(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "byte_length": len(raw),
                    "media_type": "image/png",
                    "crop_bbox_xyxy_norm": [0.25, 0.25, 0.75, 0.75],
                }
            }
        )

        rehydrated = _image_from_row(row)

        assert rehydrated is not None
        with Image.open(io.BytesIO(base64.b64decode(rehydrated))) as image:
            assert image.size == (50, 40)

    def test_corrupt_handle_fails_closed_before_embedding(self, tmp_path: Path):
        image_path = tmp_path / "page.png"
        image_path.write_bytes(b"corrupt")
        row = pd.Series(
            {
                EMBEDDING_IMAGE_HANDLE_FIELD: {
                    "version": 1,
                    "uri": image_path.as_uri(),
                    "sha256": hashlib.sha256(b"original").hexdigest(),
                    "byte_length": len(b"original"),
                    "media_type": "image/png",
                    "crop_bbox_xyxy_norm": None,
                }
            }
        )

        with pytest.raises(ImageHandleError, match="byte length"):
            _image_from_row(row)

    def test_missing_handle_object_fails_closed_before_embedding(self, tmp_path: Path):
        row = pd.Series(
            {
                EMBEDDING_IMAGE_HANDLE_FIELD: {
                    "version": 1,
                    "uri": (tmp_path / "missing.png").as_uri(),
                    "sha256": "0" * 64,
                    "byte_length": 1,
                    "media_type": "image/png",
                    "crop_bbox_xyxy_norm": None,
                }
            }
        )

        with pytest.raises(ImageHandleError, match="Failed to read"):
            _image_from_row(row)


def test_image_transport_stats_separate_logical_references_from_unique_objects():
    shared = {"uri": "s3://bucket/page.png", "sha256": "a" * 64, "byte_length": 100}
    other = {"uri": "s3://bucket/other.png", "sha256": "b" * 64, "byte_length": 40}

    stats = image_transport_stats(
        row_count=4,
        inline_values=["abcd", None, "  ", "xy"],
        handle_values=[shared, dict(shared), other, {"uri": []}],
    )

    assert stats == {
        "rows": 4,
        "inline_rows": 2,
        "inline_base64_chars": 6,
        "handle_rows": 4,
        "logical_handle_bytes": 240,
        "unique_handles": 2,
        "unique_handle_bytes": 140,
    }


class TestFormatInputStrings:
    def test_format_image_input_string(self):
        result = _format_image_input_string("AAAA")
        assert result == "data:image/png;base64,AAAA"

    def test_format_image_input_string_custom_mime(self):
        result = _format_image_input_string("BBBB", mime="image/jpeg")
        assert result == "data:image/jpeg;base64,BBBB"

    def test_format_text_image_pair_input_string(self):
        result = _format_text_image_pair_input_string("hello world", "CCCC")
        assert result == "hello world\ndata:image/png;base64,CCCC"


# ===================================================================
# _multimodal_callable_runner
# ===================================================================


class TestMultimodalCallableRunner:
    def test_image_mode(self):
        """Image-only mode calls embedder.embed_images() and returns embeddings."""
        embedder = MagicMock()
        embedder.embed_images.return_value = [[0.1, 0.2], [0.3, 0.4]]

        df = pd.DataFrame(
            {
                "text": ["page one", "page two"],
                "_image_b64": ["img1_b64", "img2_b64"],
            }
        )

        result = _multimodal_callable_runner(
            df,
            embedder=embedder,
            batch_size=64,
            embed_modality="image",
        )

        embedder.embed_images.assert_called_once()
        assert result["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
        assert len(result["info_msgs"]) == 2

    def test_text_image_fallback(self):
        """text_image mode: rows with images use embed_text_image(), rows without fall back to embed()."""
        embedder = MagicMock()
        # Row 0 has image -> embed_text_image
        # Row 1 has no image -> embed (text-only fallback)
        embedder.embed_text_image.return_value = [[1.0, 2.0]]
        embedder.embed.return_value = [[3.0, 4.0]]

        df = pd.DataFrame(
            {
                "text": ["with image", "text only"],
                "_image_b64": ["imgB64", ""],
            }
        )

        result = _multimodal_callable_runner(
            df,
            embedder=embedder,
            batch_size=64,
            embed_modality="text_image",
        )

        embedder.embed_text_image.assert_called_once()
        embedder.embed.assert_called_once()
        # Order must be preserved: row 0 (multimodal), row 1 (text fallback)
        assert result["embeddings"] == [[1.0, 2.0], [3.0, 4.0]]
        assert len(result["info_msgs"]) == 2

    @patch("nemo_retriever.models.inference.main_text_embed.load_verified_image_b64", return_value="verified_b64")
    def test_verified_handle_is_loaded_once_audited_and_discarded_after_local_embedding(self, mock_load, caplog):
        embedder = MagicMock()
        embedder.embed_text_image.return_value = [[1.0, 2.0]]
        handle = {
            "version": 1,
            "uri": "file:///shared/page.png",
            "sha256": "a" * 64,
            "byte_length": 123,
            "media_type": "image/png",
            "crop_bbox_xyxy_norm": None,
        }

        with caplog.at_level("INFO", logger="nemo_retriever.models.inference.main_text_embed"):
            result, _ = create_text_embeddings_for_df(
                pd.DataFrame({"text": ["with image"], EMBEDDING_IMAGE_HANDLE_FIELD: [handle]}),
                task_config={"multimodal_embedder": embedder, "endpoint_url": None},
                transform_config=TextEmbeddingConfig(embed_modality="text_image"),
            )

        assert mock_load.call_count == 1
        assert "_verified_image_b64" not in result.columns
        assert "inline_rows=0" in caplog.text
        assert "handle_rows=1" in caplog.text
        assert "verified_handle_rows=1" in caplog.text
        assert "logical_handle_bytes=123" in caplog.text
        assert "unique_handles=1" in caplog.text
        assert "unique_handle_bytes=123" in caplog.text

    @patch(
        "nemo_retriever.models.inference.main_text_embed.load_verified_image_b64",
        side_effect=ImageHandleError("digest changed"),
    )
    def test_handle_verification_failure_is_audited_and_fails_closed(self, mock_load, caplog):
        handle = {
            "version": 1,
            "uri": "file:///shared/page.png",
            "sha256": "a" * 64,
            "byte_length": 123,
            "media_type": "image/png",
            "crop_bbox_xyxy_norm": None,
        }

        with caplog.at_level("ERROR", logger="nemo_retriever.models.inference.main_text_embed"):
            with pytest.raises(ImageHandleError, match="digest changed"):
                create_text_embeddings_for_df(
                    pd.DataFrame({"text": ["with image"], EMBEDDING_IMAGE_HANDLE_FIELD: [handle]}),
                    task_config={"multimodal_embedder": MagicMock(), "endpoint_url": None},
                    transform_config=TextEmbeddingConfig(embed_modality="text_image"),
                )

        assert mock_load.call_count == 1
        assert "Embedding image transport verification failed" in caplog.text
        assert "'handle_rows': 1" in caplog.text


# ===================================================================
# explode_content_to_rows
# ===================================================================


class TestExplodeContentToRows:
    def test_text_mode_tags_modality(self):
        """Default text mode tags every row with _embed_modality='text' and no _image_b64."""
        df = pd.DataFrame(
            {
                "text": ["Hello world"],
                "table": [[{"text": "cell data"}]],
            }
        )

        result = explode_content_to_rows(df)

        assert "_embed_modality" in result.columns
        assert list(result["_embed_modality"]) == ["text", "text"]
        assert "_image_b64" not in result.columns

    def test_arrow_backed_structured_arrays_expand_into_element_rows(self):
        """Ray Arrow-backed list cells expand like their Python-list equivalents."""
        df = pd.DataFrame(
            {
                "text": ["page text"],
                "table": [np.array([{"text": "table text"}], dtype=object)],
                "chart": [np.array([{"text": "chart text"}], dtype=object)],
            }
        )

        result = explode_content_to_rows(df)

        assert result["text"].tolist() == ["page text", "table text", "chart text"]
        assert result["_content_type"].tolist() == ["text", "table", "chart"]
        assert result.iloc[0]["table"] is not result.iloc[1]["table"]

    @pytest.mark.parametrize("value", [np.array(1, dtype=object), np.ones((2, 2))])
    def test_non_collection_arrays_are_not_expanded(self, value):
        result = explode_content_to_rows(pd.DataFrame({"text": ["page text"], "table": [value]}))

        assert result["text"].tolist() == ["page text"]

    @patch("nemo_retriever.common.modality.content_transforms._crop_b64_image_by_norm_bbox")
    def test_text_image_carries_image(self, mock_crop):
        """text_image mode copies page image to _image_b64, crops for structured content."""
        mock_crop.return_value = ("cropped_b64", None)

        df = pd.DataFrame(
            {
                "text": ["some page text"],
                "page_image": [{"image_b64": "full_page_b64"}],
                "table": [[{"text": "table cell", "bbox_xyxy_norm": [0.1, 0.2, 0.9, 0.8]}]],
            }
        )

        result = explode_content_to_rows(df, modality="text_image")

        assert "_image_b64" in result.columns
        images = list(result["_image_b64"])
        modalities = list(result["_embed_modality"])

        # Row 0: page text row gets full page image
        assert images[0] == "full_page_b64"
        assert modalities[0] == "text_image"

        # Row 1: structured content row gets cropped image
        assert images[1] == "cropped_b64"
        assert modalities[1] == "text_image"

        mock_crop.assert_called_once_with(
            "full_page_b64",
            bbox_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
        )


# ===================================================================
# collapse_content_to_page_rows
# ===================================================================


class TestCollapseContentToPageRows:
    def test_text_concatenation(self):
        """Page text + table + chart text are concatenated into one string per page."""
        df = pd.DataFrame(
            {
                "text": ["Hello world"],
                "table": [[{"text": "table data"}]],
                "chart": [[{"text": "chart data"}]],
                "infographic": [[]],
            }
        )

        result = collapse_content_to_page_rows(df)

        assert len(result) == 1
        assert result["text"].iloc[0] == "Hello world\n\ntable data\n\nchart data"
        assert result["_embed_modality"].iloc[0] == "text"

    def test_arrow_backed_structured_arrays_are_collapsed_into_page_text(self):
        """Ray Arrow-backed list cells contribute their text to the page row."""
        df = pd.DataFrame(
            {
                "text": ["page text"],
                "table": [np.array([{"text": "table text"}], dtype=object)],
                "chart": [np.array([{"text": "chart text"}], dtype=object)],
            }
        )

        result = collapse_content_to_page_rows(df)

        assert result["text"].tolist() == ["page text\n\ntable text\n\nchart text"]

    def test_full_page_image_used(self):
        """In image modalities, _image_b64 is the full page image (no cropping)."""
        df = pd.DataFrame(
            {
                "text": ["some text"],
                "page_image": [{"image_b64": "full_page_b64"}],
                "table": [[{"text": "table cell", "bbox_xyxy_norm": [0.1, 0.2, 0.9, 0.8]}]],
            }
        )

        result = collapse_content_to_page_rows(df, modality="text_image")

        assert len(result) == 1
        assert result["_image_b64"].iloc[0] == "full_page_b64"
        assert result["_embed_modality"].iloc[0] == "text_image"

    def test_multiple_pages_produce_one_row_each(self):
        """Each page produces exactly one row in the output."""
        df = pd.DataFrame(
            {
                "text": ["page 1 text", "page 2 text"],
                "table": [[{"text": "t1"}], [{"text": "t2"}]],
            }
        )

        result = collapse_content_to_page_rows(df)

        assert len(result) == 2
        assert "t1" in result["text"].iloc[0]
        assert "t2" in result["text"].iloc[1]

    def test_empty_content_handled(self):
        """Pages with no text and no structured content produce an empty string."""
        df = pd.DataFrame(
            {
                "text": ["", None],
                "table": [[], None],
            }
        )

        result = collapse_content_to_page_rows(df)

        assert len(result) == 2
        assert result["text"].iloc[0] == ""
        assert result["text"].iloc[1] == ""

    def test_image_modality_without_page_image_column(self):
        """When page_image column is missing, _image_b64 is set to None."""
        df = pd.DataFrame(
            {
                "text": ["some text"],
                "table": [[{"text": "data"}]],
            }
        )

        result = collapse_content_to_page_rows(df, modality="image")

        assert len(result) == 1
        assert result["_image_b64"].iloc[0] is None

    def test_empty_dataframe_passthrough(self):
        """Empty DataFrame is returned as-is."""
        df = pd.DataFrame()
        result = collapse_content_to_page_rows(df)
        assert result.empty

    def test_non_dataframe_passthrough(self):
        """Non-DataFrame input is returned as-is."""
        result = collapse_content_to_page_rows(None)
        assert result is None
