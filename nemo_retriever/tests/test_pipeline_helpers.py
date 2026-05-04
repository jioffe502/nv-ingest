# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import typer

import nemo_retriever.pipeline as pipeline_pkg
import nemo_retriever.pipeline.__main__ as pipeline_main
from nemo_retriever.params import EmbedParams, ExtractParams, TextChunkParams
from nemo_retriever.pipeline.__main__ import (
    _build_ingestor,
    _build_embed_params,
    _build_runtime_observability_fields,
    _collect_results,
    _count_input_units,
    _count_uploadable_vdb_records,
    _parse_ray_data_operator_timings,
    _parse_vdb_kwargs_json,
    _resolve_file_patterns,
)


def test_pipeline_package_exports_cli_app_and_run() -> None:
    from nemo_retriever.pipeline.__main__ import app, run

    assert pipeline_pkg.app is app
    assert pipeline_pkg.run is run
    assert set(pipeline_pkg.__all__) == {"app", "run"}


@pytest.mark.parametrize(
    ("input_type", "files", "expected_globs"),
    [
        ("pdf", ["nested/doc.pdf"], ["*.pdf"]),
        ("doc", ["deck.pptx", "report.docx"], ["*.docx", "*.pptx"]),
        ("image", ["plot.png"], ["*.png"]),
        ("video", ["clip.mp4"], ["*.mp4"]),
    ],
)
def test_resolve_file_patterns_recurses_directory_inputs(
    tmp_path: Path,
    input_type: str,
    files: list[str],
    expected_globs: list[str],
) -> None:
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    assert _resolve_file_patterns(tmp_path, input_type) == [str(tmp_path / "**" / glob) for glob in expected_globs]


class TestBuildIngestor:
    def test_store_is_attached_after_embed(self, monkeypatch, tmp_path: Path) -> None:
        calls: list[str] = []
        captured: dict[str, Any] = {}

        class _FakeIngestor:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            def files(self, file_patterns):
                calls.append("files")
                captured["file_patterns"] = file_patterns
                return self

            def extract(self, params):
                calls.append("extract")
                return self

            def embed(self, params):
                calls.append("embed")
                captured["embed_params"] = params
                return self

            def store(self, params):
                calls.append("store")
                captured["store_params"] = params
                return self

        monkeypatch.setattr(pipeline_main, "GraphIngestor", _FakeIngestor)

        _build_ingestor(
            run_mode="inprocess",
            ray_address=None,
            file_patterns=[str(tmp_path / "doc.pdf")],
            input_type="pdf",
            extract_params=ExtractParams(method="ocr"),
            embed_params=EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2"),
            text_chunk_params=TextChunkParams(),
            enable_text_chunk=False,
            enable_dedup=False,
            enable_caption=False,
            dedup_iou_threshold=0.8,
            caption_invoke_url=None,
            caption_remote_api_key=None,
            caption_model_name="nvidia/llama-nemotron-rerank-vl-1b-v2",
            caption_device=None,
            caption_context_text_max_chars=0,
            caption_gpu_memory_utilization=0.5,
            caption_gpus_per_actor=None,
            caption_temperature=1.0,
            caption_top_p=None,
            caption_max_tokens=1024,
            store_images_uri=str(tmp_path / "stored"),
            segment_audio=False,
            audio_split_type="time",
            audio_split_interval=30,
            video_extract_audio=True,
            video_extract_frames=True,
            video_frame_fps=0.5,
            video_frame_dedup=True,
            video_frame_text_dedup=True,
            video_frame_text_dedup_max_dropped_frames=2,
            video_av_fuse=True,
        )

        assert calls == ["files", "extract", "embed", "store"]
        assert captured["store_params"].storage_uri.endswith("/stored")
        assert captured["init"]["node_overrides"] is None

    def test_store_tuning_flags_create_store_node_overrides(self, monkeypatch, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        class _FakeIngestor:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            def files(self, file_patterns):
                return self

            def extract(self, params):
                return self

            def embed(self, params):
                return self

            def store(self, params):
                captured["store_params"] = params
                return self

        monkeypatch.setattr(pipeline_main, "GraphIngestor", _FakeIngestor)

        _build_ingestor(
            run_mode="batch",
            ray_address=None,
            file_patterns=[str(tmp_path / "doc.pdf")],
            input_type="pdf",
            extract_params=ExtractParams(method="ocr"),
            embed_params=EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2"),
            text_chunk_params=TextChunkParams(),
            enable_text_chunk=False,
            enable_dedup=False,
            enable_caption=False,
            dedup_iou_threshold=0.8,
            caption_invoke_url=None,
            caption_remote_api_key=None,
            caption_model_name="nvidia/llama-nemotron-rerank-vl-1b-v2",
            caption_device=None,
            caption_context_text_max_chars=0,
            caption_gpu_memory_utilization=0.5,
            caption_gpus_per_actor=None,
            caption_temperature=1.0,
            caption_top_p=None,
            caption_max_tokens=1024,
            store_images_uri=str(tmp_path / "stored"),
            store_actors=4,
            store_cpus_per_actor=0.5,
            segment_audio=False,
            audio_split_type="time",
            audio_split_interval=30,
        )

        assert captured["init"]["node_overrides"] == {"StoreOperator": {"concurrency": 4, "num_cpus": 0.5}}
        assert captured["store_params"].storage_uri.endswith("/stored")

    def test_default_store_tuning_adds_no_store_node_override(self, monkeypatch, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        class _FakeIngestor:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            def files(self, file_patterns):
                return self

            def extract(self, params):
                return self

            def embed(self, params):
                return self

            def store(self, params):
                return self

        monkeypatch.setattr(pipeline_main, "GraphIngestor", _FakeIngestor)

        _build_ingestor(
            run_mode="batch",
            ray_address=None,
            file_patterns=[str(tmp_path / "doc.pdf")],
            input_type="pdf",
            extract_params=ExtractParams(method="ocr"),
            embed_params=EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2"),
            text_chunk_params=TextChunkParams(),
            enable_text_chunk=False,
            enable_dedup=False,
            enable_caption=False,
            dedup_iou_threshold=0.8,
            caption_invoke_url=None,
            caption_remote_api_key=None,
            caption_model_name="nvidia/llama-nemotron-rerank-vl-1b-v2",
            caption_device=None,
            caption_context_text_max_chars=0,
            caption_gpu_memory_utilization=0.5,
            caption_gpus_per_actor=None,
            caption_temperature=1.0,
            caption_top_p=None,
            caption_max_tokens=1024,
            store_images_uri=str(tmp_path / "stored"),
            segment_audio=False,
            audio_split_type="time",
            audio_split_interval=30,
        )

        assert captured["init"]["node_overrides"] is None


def test_resolve_file_patterns_returns_existing_file_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"x")

    assert _resolve_file_patterns(path, "audio") == [str(path)]


def test_resolve_file_patterns_rejects_missing_or_empty_inputs(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter, match="Path does not exist"):
        _resolve_file_patterns(tmp_path / "missing", "pdf")

    (tmp_path / "sidecar.json").write_text("{}", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="No files found"):
        _resolve_file_patterns(tmp_path, "pdf")


def test_parse_vdb_kwargs_json_keeps_backend_kwargs_opaque() -> None:
    assert _parse_vdb_kwargs_json(None) == {}
    assert _parse_vdb_kwargs_json('{"collection_name": "docs", "uri": "http://localhost:19530"}') == {
        "collection_name": "docs",
        "uri": "http://localhost:19530",
    }


def test_parse_vdb_kwargs_json_rejects_non_object_json() -> None:
    with pytest.raises(typer.BadParameter, match="JSON object"):
        _parse_vdb_kwargs_json('["not", "an", "object"]')


def test_build_embed_params_forwards_remote_and_modality_flags() -> None:
    params = _build_embed_params(
        embed_model_name="nvidia/test-embed",
        embed_invoke_url="http://embed.example/v1",
        embed_remote_api_key="nvapi-secret",
        embed_modality="text_image",
        text_elements_modality="text",
        structured_elements_modality="image",
        embed_granularity="element",
        embed_actors=2,
        embed_batch_size=16,
        embed_cpus_per_actor=1.5,
        embed_gpus_per_actor=0.5,
    )

    assert isinstance(params, EmbedParams)
    assert params.model_name == "nvidia/test-embed"
    assert params.embed_invoke_url == "http://embed.example/v1"
    assert params.api_key == "nvapi-secret"
    assert params.embed_modality == "text_image"
    assert params.text_elements_modality == "text"
    assert params.structured_elements_modality == "image"
    assert params.embed_granularity == "element"
    assert params.inference_batch_size == 16
    assert params.batch_tuning.embed_workers == 2
    assert params.batch_tuning.embed_batch_size == 16
    assert params.batch_tuning.embed_cpus_per_actor == 1.5
    assert params.batch_tuning.gpu_embed == 0.0


class TestCollectResults:
    """Ingest returns a DataFrame (``ingestor.ingest()`` → ``ds.to_pandas()``); _collect_results consumes it."""

    def test_batch_mode_accepts_ingest_dataframe(self):
        rows = [
            {"source_id": "a", "text": "hello"},
            {"source_id": "a", "text": "world"},
            {"source_id": "b", "text": "!"},
        ]
        # Same shape as the graph executor return after ``Dataset.to_pandas()``.
        result_df = pd.DataFrame(rows)

        records, df, download_time, num_units = _collect_results("batch", result_df)

        assert records == rows
        assert df is result_df
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["source_id", "text"]
        assert len(df) == 3
        # ``source_id`` has two distinct values → that is the unit count.
        assert num_units == 2
        assert download_time >= 0.0

    def test_batch_mode_handles_empty_result(self):
        result_df = pd.DataFrame()
        records, df, download_time, num_units = _collect_results("batch", result_df)
        assert records == []
        assert df.empty
        # Empty DataFrame has no columns → falls through to len(df.index) == 0.
        assert num_units == 0
        assert download_time >= 0.0

    def test_inprocess_mode_accepts_dataframe_directly(self):
        rows = [
            {"source_id": "a", "text": "x"},
            {"source_id": "b", "text": "y"},
        ]
        df_in = pd.DataFrame(rows)

        records, df_out, download_time, num_units = _collect_results("inprocess", df_in)

        # The DataFrame is passed through unchanged (same object).
        assert df_out is df_in
        assert records == rows
        # inprocess mode never incurs Ray download time.
        assert download_time == 0.0
        assert num_units == 2


def test_collect_results_accepts_inprocess_dataframe() -> None:
    df_in = pd.DataFrame([{"source_path": "/a.pdf"}, {"source_path": "/b.pdf"}])

    records, df_out, download_time, num_units = _collect_results("inprocess", df_in)

    assert df_out is df_in
    assert records == [{"source_path": "/a.pdf"}, {"source_path": "/b.pdf"}]
    assert download_time == 0.0
    assert num_units == 2


def test_count_uploadable_vdb_records_filters_rows_without_embedding_or_text() -> None:
    rows = [
        {
            "text": "keep",
            "text_embeddings_1b_v2": {"embedding": [0.1, 0.2]},
            "source_id": "/tmp/doc-a.pdf",
            "page_number": 1,
        },
        {
            "text": "drop missing embedding",
            "source_id": "/tmp/doc-a.pdf",
            "page_number": 2,
        },
        {
            "text_embeddings_1b_v2": {"embedding": [0.3, 0.4]},
            "source_id": "/tmp/doc-a.pdf",
            "page_number": 3,
        },
    ]

    assert _count_uploadable_vdb_records(rows) == 1
    assert _count_uploadable_vdb_records([]) == 0


def test_ray_data_stats_are_written_and_parsed_for_store_operator(tmp_path: Path) -> None:
    stats_text = """Operator 0 FromItems: 1 tasks executed, 10 blocks produced in 0.50s
* Operator throughput:
\t* Total input num rows: 0 rows
\t* Total output num rows: 10 rows
\t* Ray Data throughput: 20 rows/s

Operator 1 MapBatches(_BatchEmbedActor): 2 tasks executed, 5 blocks produced in 2.00s
* UDF time: 200ms min, 1.00s max, 600ms mean, 1.20s total
* Operator throughput:
\t* Total input num rows: 10 rows
\t* Total output num rows: 10 rows
\t* Ray Data throughput: 5 rows/s

Operator 2 MapBatches(StoreOperator): 4 tasks executed, 4 blocks produced in 2.50s
* Remote wall time: 100ms min, 1.00s max, 500ms mean, 2.00s total
* Remote cpu time: 20ms min, 200ms max, 100ms mean, 400ms total
* UDF time: 90ms min, 900ms max, 450ms mean, 1.80s total
* Operator throughput:
\t* Total input num rows: 10 rows
\t* Total output num rows: 10 rows
\t* Ray Data throughput: 4 rows/s
"""

    class _FakeIngestor:
        ray_data_stats = stats_text
        ray_data_stats_error = None

    fields = _build_runtime_observability_fields(
        ingestor=_FakeIngestor(),
        run_mode="batch",
        runtime_metrics_dir=tmp_path,
        runtime_metrics_prefix="stats-run",
        num_rows=10,
        store_actors=4,
        store_cpus_per_actor=0.5,
    )

    raw_stats_path = tmp_path / "stats-run.ray-data.stats.txt"
    assert raw_stats_path.read_text(encoding="utf-8") == stats_text.rstrip() + "\n"
    assert fields["ray_data_stats_available"] is True
    assert fields["ray_data_stats_file"] == str(raw_stats_path.resolve())
    assert len(fields["ray_data_operator_timings"]) == 3

    store_metrics = fields["store_operator"]
    assert store_metrics["requested_actors"] == 4
    assert store_metrics["effective_actors"] == 4
    assert store_metrics["requested_cpus_per_actor"] == 0.5
    assert store_metrics["effective_cpus_per_actor"] == 0.5
    assert store_metrics["stats_detected"] is True
    assert store_metrics["elapsed_secs"] == 2.5
    assert store_metrics["udf_total_secs"] == 1.8
    assert store_metrics["rows_per_sec"] == 4.0
    assert store_metrics["tail_after_previous_operator_secs"] == 0.5
    assert store_metrics["previous_operator"] == "MapBatches(_BatchEmbedActor)"


def test_ray_data_stats_helpers_tolerate_missing_stats(tmp_path: Path) -> None:
    class _StatsUnavailable:
        @property
        def ray_data_stats(self):
            raise RuntimeError("stats unavailable")

    fields = _build_runtime_observability_fields(
        ingestor=_StatsUnavailable(),
        run_mode="batch",
        runtime_metrics_dir=tmp_path,
        runtime_metrics_prefix="missing",
        num_rows=8,
        store_actors=0,
        store_cpus_per_actor=0.0,
    )

    assert not (tmp_path / "missing.ray-data.stats.txt").exists()
    assert fields["ray_data_stats_available"] is False
    assert fields["ray_data_operator_timings"] == []
    assert fields["ray_data_stats_error"] == "RuntimeError: stats unavailable"
    assert fields["store_operator"] == {
        "requested_actors": 0,
        "effective_actors": 1,
        "requested_cpus_per_actor": 0.0,
        "effective_cpus_per_actor": 1.0,
        "cpus_per_actor_is_resource_reservation": True,
        "stats_detected": False,
    }


def test_parse_ray_data_operator_timings_handles_empty_stats() -> None:
    assert _parse_ray_data_operator_timings(None) == []
    assert _parse_ray_data_operator_timings("") == []


def test_count_input_units_prefers_source_id_then_source_path() -> None:
    assert _count_input_units(pd.DataFrame({"source_id": ["a", "a", "b"]})) == 2
    assert _count_input_units(pd.DataFrame({"source_path": ["/a", "/b", "/b"]})) == 2
    assert _count_input_units(pd.DataFrame({"text": ["x", "y", "z"]})) == 3
