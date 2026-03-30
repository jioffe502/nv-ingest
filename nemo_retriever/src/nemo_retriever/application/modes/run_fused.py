# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Fused-mode orchestration with structured run report artifacts."""

from pathlib import Path
import time

from pydantic import Field

from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import VdbUploadParams
from nemo_retriever.utils.detection_summary import (
    print_detection_summary,
    print_pages_per_second,
)
from nemo_retriever.utils.input_files import resolve_input_files

from .executor import run_mode_ingest
from .reports import (
    RunArtifactConfig,
    RunArtifacts,
    RunEvaluationConfig,
    RunMetrics,
    RunReport,
    persist_run_report_artifacts,
)
from .shared import (
    ModePipelineConfigModel,
    count_lancedb_rows,
    ensure_lancedb_table,
    estimate_processed_pages,
    evaluate_lancedb_metrics,
    persist_detection_summary_artifact,
    print_evaluation_metrics,
    resolve_lancedb_target,
    resolve_mode_file_patterns,
    resolve_input_pages,
    shutdown_ray_safely,
)


class FusedPipelineConfig(ModePipelineConfigModel):
    input_path: str
    input_type: str = "pdf"
    file_patterns: list[str] = Field(default_factory=list)
    create_params: IngestorCreateParams = Field(default_factory=IngestorCreateParams)
    execute_params: IngestExecuteParams = Field(default_factory=IngestExecuteParams)
    extract_params: ExtractParams = Field(default_factory=ExtractParams)
    embed_params: EmbedParams = Field(default_factory=EmbedParams)
    vdb_upload_params: VdbUploadParams = Field(default_factory=VdbUploadParams)
    evaluation: RunEvaluationConfig = Field(default_factory=RunEvaluationConfig)
    artifacts: RunArtifactConfig = Field(default_factory=RunArtifactConfig)


def run_fused_pipeline(cfg: FusedPipelineConfig) -> RunReport:
    try:
        from nemo_retriever.ingestor import create_ingestor

        if cfg.input_type != "pdf":
            raise ValueError("Fused mode currently supports only pdf input_type.")

        input_path = Path(cfg.input_path).expanduser().resolve()
        input_files = resolve_input_files(input_path, cfg.input_type)
        file_patterns = resolve_mode_file_patterns(
            input_path=cfg.input_path,
            input_type=cfg.input_type,
            file_patterns=cfg.file_patterns,
        )
        lancedb_uri, lancedb_table = resolve_lancedb_target(
            artifacts_lancedb_uri=cfg.artifacts.lancedb_uri,
            artifacts_lancedb_table=cfg.artifacts.lancedb_table,
            vdb_lancedb_uri=cfg.vdb_upload_params.lancedb.lancedb_uri,
            vdb_lancedb_table=cfg.vdb_upload_params.lancedb.table_name,
        )

        ensure_lancedb_table(lancedb_uri, lancedb_table)
        ingestor = create_ingestor(run_mode="fused", params=cfg.create_params)
        ingestor = (
            ingestor.files(file_patterns)
            .extract(cfg.extract_params)
            .embed(cfg.embed_params)
            .vdb_upload(cfg.vdb_upload_params)
        )

        ingest_start = time.perf_counter()
        ingestor.ingest(params=cfg.execute_params)
        ingest_secs = time.perf_counter() - ingest_start

        from nemo_retriever.utils.detection_summary import collect_detection_summary_from_lancedb

        detection_payload = collect_detection_summary_from_lancedb(lancedb_uri, lancedb_table)
        detection_summary_file = persist_detection_summary_artifact(
            detection_summary_file=cfg.artifacts.detection_summary_file,
            detection_payload=detection_payload,
        )

        evaluation_summary, evaluation_secs = evaluate_lancedb_metrics(
            cfg.evaluation,
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
            embed_model_name=str(cfg.embed_params.model_name or "nemo_retriever_v1"),
            embed_invoke_url=cfg.embed_params.embed_invoke_url,
            embed_api_key=cfg.embed_params.api_key,
            hybrid=bool(cfg.vdb_upload_params.lancedb.hybrid),
        )
        total_secs = ingest_secs + evaluation_secs

        report = RunReport(
            run_mode="fused",
            input_path=str(input_path),
            input_type=cfg.input_type,
            evaluation_mode=cfg.evaluation.evaluation_mode,
            metrics=RunMetrics(
                input_files=len(input_files) or None,
                input_pages=resolve_input_pages(cfg.input_type, input_files),
                processed_pages=(
                    detection_payload.get("pages_seen")
                    if isinstance(detection_payload, dict) and detection_payload.get("pages_seen") is not None
                    else estimate_processed_pages(lancedb_uri, lancedb_table)
                ),
                rows_processed=count_lancedb_rows(lancedb_uri, lancedb_table),
                ingest_secs=ingest_secs,
                evaluation_secs=evaluation_secs,
                total_secs=total_secs,
            ),
            evaluation=evaluation_summary,
            detection_summary=detection_payload,
            artifacts=RunArtifacts(
                detection_summary_file=detection_summary_file,
                lancedb_uri=lancedb_uri,
                lancedb_table=lancedb_table,
            ),
            extras={"file_patterns": file_patterns},
        )
        return persist_run_report_artifacts(
            report,
            runtime_metrics_dir=cfg.execute_params.runtime_metrics_dir,
            prefix=cfg.execute_params.runtime_metrics_prefix,
        )
    finally:
        shutdown_ray_safely()


def render_fused_run_report(report: RunReport) -> None:
    if report.detection_summary is not None:
        print_detection_summary(report.detection_summary)
    if report.metrics.ingest_secs is not None:
        print_pages_per_second(report.metrics.processed_pages, report.metrics.ingest_secs)
    print_evaluation_metrics(label=report.evaluation.label, metrics=report.evaluation.metrics)


def run_fused(
    *,
    create_params: IngestorCreateParams | None = None,
    ingest_params: IngestExecuteParams | None = None,
) -> object:
    return run_mode_ingest(run_mode="fused", create_params=create_params, ingest_params=ingest_params)
