# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-level tests for StoreOperator."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from nemo_retriever.graph import InprocessExecutor, StoreOperator, UDFOperator
from nemo_retriever.params import StoreParams


def _make_tiny_png_b64(width: int = 4, height: int = 4, color=(255, 0, 0)) -> str:
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_embedded_df(b64: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path": "/docs/test.pdf",
                "page_number": 1,
                "text": "Sample page text",
                "_content_type": "text",
                "_bbox_xyxy_norm": None,
                "_image_b64": b64,
                "page_image": {"image_b64": b64, "stored_image_uri": "file:///old/page.png"},
                "table": [
                    {
                        "text": "col1|col2",
                        "image_b64": b64,
                        "stored_image_uri": "file:///old/table.png",
                    }
                ],
            }
        ]
    )


class TestStoreOperatorInGraph:
    def test_store_operator_writes_row_image_and_sets_top_level_uri(self, tmp_path: Path):
        b64 = _make_tiny_png_b64()
        df = _make_embedded_df(b64)

        graph = UDFOperator(lambda x: x, name="Identity") >> StoreOperator(
            params=StoreParams(storage_uri=str(tmp_path))
        )
        result = InprocessExecutor(graph, show_progress=False).ingest(df)

        files = list(tmp_path.rglob("*.png"))
        assert len(files) == 1
        assert files[0].read_bytes() == base64.b64decode(b64)

        stored_uri = result.iloc[0]["_stored_image_uri"]
        assert stored_uri.startswith("file://")
        assert Path(urlparse(stored_uri).path).exists()

    def test_store_operator_preserves_inline_and_nested_payloads(self, tmp_path: Path):
        b64 = _make_tiny_png_b64()
        df = _make_embedded_df(b64)

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        assert result.iloc[0]["_image_b64"] == b64
        assert result.iloc[0]["page_image"]["image_b64"] == b64
        assert result.iloc[0]["page_image"]["stored_image_uri"] == "file:///old/page.png"
        assert result.iloc[0]["table"][0]["image_b64"] == b64
        assert result.iloc[0]["table"][0]["stored_image_uri"] == "file:///old/table.png"

    def test_store_operator_skips_rows_without_image_b64(self, tmp_path: Path):
        df = _make_embedded_df(None)

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        assert not list(tmp_path.rglob("*"))
        assert "_stored_image_uri" not in result.columns

    def test_store_operator_uses_page_image_when_row_image_column_is_absent(self, tmp_path: Path):
        b64 = _make_tiny_png_b64()
        df = _make_embedded_df(b64).drop(columns=["_image_b64"])

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        files = list(tmp_path.rglob("*.png"))
        assert len(files) == 1
        assert files[0].read_bytes() == base64.b64decode(b64)
        assert result.iloc[0]["_stored_image_uri"].startswith("file://")

    def test_store_operator_forwards_storage_options(self, monkeypatch):
        b64 = _make_tiny_png_b64()
        df = _make_embedded_df(b64)
        calls: list[tuple[str, str, dict]] = []

        class _Writer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def write(self, data: bytes) -> int:
                self.data = data
                return len(data)

        def _fake_open(path: str, mode: str = "rb", **kwargs):
            calls.append((path, mode, kwargs))
            return _Writer()

        monkeypatch.setattr("nemo_retriever.graph.store_operator.fsspec.open", _fake_open)

        params = StoreParams(
            storage_uri="s3://bucket/prefix",
            storage_options={"key": "YOUR_KEY", "secret": "YOUR_SECRET"},
        )
        result = StoreOperator(params=params).process(df)

        assert len(calls) == 1
        assert calls[0][0].startswith("s3://bucket/prefix/")
        assert calls[0][1] == "wb"
        assert calls[0][2] == {"key": "YOUR_KEY", "secret": "YOUR_SECRET"}
        assert result.iloc[0]["_stored_image_uri"].startswith("s3://bucket/prefix/")

    def test_embedding_preserves_image_b64_for_post_embed_store(self, monkeypatch):
        from nemo_retriever.text_embed import runtime

        b64 = _make_tiny_png_b64()
        df = _make_embedded_df(b64)
        df["_embed_modality"] = "text_image"

        def _fake_embed_group(group_df: pd.DataFrame, **kwargs):
            out = group_df.copy()
            out["metadata"] = [{"embedding": [0.1, 0.2]} for _ in range(len(out.index))]
            return out

        monkeypatch.setattr(runtime, "_embed_group", _fake_embed_group)

        result = runtime.embed_text_main_text_embed(df, model=object(), embed_modality="text_image")

        assert result.iloc[0]["_image_b64"] == b64
        assert "_embed_modality" not in result.columns
