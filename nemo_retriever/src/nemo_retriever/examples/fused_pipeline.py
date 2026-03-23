# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fused ingestion pipeline with optional recall evaluation.
Run with: uv run python -m nemo_retriever.examples.fused_pipeline <input-dir>
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import typer
from nemo_retriever.application.modes.reports import RunArtifactConfig, RunEvaluationConfig
from nemo_retriever.application.modes.run_fused import (
    FusedPipelineConfig,
    render_fused_run_report,
    run_fused_pipeline,
)
from nemo_retriever.application.modes.shared import (
    DEFAULT_LANCEDB_TABLE as LANCEDB_TABLE,
    DEFAULT_LANCEDB_URI as LANCEDB_URI,
    configure_cli_logging,
    restore_cli_logging,
)
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import VdbUploadParams

app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        help="File or directory containing PDFs to ingest.",
        path_type=Path,
    ),
    ray_address: Optional[str] = typer.Option(
        None,
        "--ray-address",
        help="URL or address of a running Ray cluster (e.g. 'auto' or 'ray://host:10001'). Omit for in-process Ray.",
    ),
    start_ray: bool = typer.Option(
        False,
        "--start-ray",
        help=(
            "Start a Ray head node (ray start --head) and connect to it. "
            "Dashboard at http://127.0.0.1:8265. Ignores --ray-address."
        ),
    ),
    query_csv: Path = typer.Option(
        "bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help=(
            "Path to query CSV for recall evaluation. Default: bo767_query_gt.csv "
            "(current directory). Recall is skipped if the file does not exist."
        ),
    ),
    no_recall_details: bool = typer.Option(
        False,
        "--no-recall-details",
        help=(
            "Do not print per-query retrieval details (query, gold, hits). "
            "Only the missed-gold summary and recall metrics are printed."
        ),
    ),
    pdf_extract_tasks: int = typer.Option(
        12,
        "--pdf-extract-tasks",
        min=1,
        help="Number of CPU tasks for PDF extraction stage.",
    ),
    pdf_extract_cpus_per_task: float = typer.Option(
        2.0,
        "--pdf-extract-cpus-per-task",
        min=0.1,
        help="CPUs reserved per PDF extraction task.",
    ),
    pdf_extract_batch_size: int = typer.Option(
        4,
        "--pdf-extract-batch-size",
        min=1,
        help="Batch size for PDF extraction stage.",
    ),
    pdf_split_batch_size: int = typer.Option(
        1,
        "--pdf-split-batch-size",
        min=1,
        help="Batch size for PDF split stage.",
    ),
    fused_workers: int = typer.Option(
        1,
        "--fused-workers",
        min=1,
        help="Actor count for fused model stage.",
    ),
    fused_batch_size: int = typer.Option(
        64,
        "--fused-batch-size",
        min=1,
        help="Ray Data batch size for fused model stage.",
    ),
    fused_cpus_per_actor: float = typer.Option(
        1.0,
        "--fused-cpus-per-actor",
        min=0.1,
        help="CPUs reserved per fused actor.",
    ),
    fused_gpus_per_actor: float = typer.Option(
        1.0,
        "--fused-gpus-per-actor",
        min=0.0,
        help="GPUs reserved per fused actor.",
    ),
    runtime_metrics_dir: Optional[Path] = typer.Option(
        None,
        "--runtime-metrics-dir",
        path_type=Path,
        file_okay=False,
        dir_okay=True,
        help="Optional directory where Ray runtime metrics are written per run.",
    ),
    runtime_metrics_prefix: Optional[str] = typer.Option(
        None,
        "--runtime-metrics-prefix",
        help="Optional filename prefix for per-run metrics artifacts.",
    ),
    lancedb_uri: str = typer.Option(
        LANCEDB_URI,
        "--lancedb-uri",
        help="LanceDB URI/path for this run.",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        path_type=Path,
        dir_okay=False,
        help="Optional file to collect all pipeline + Ray driver logs for this run.",
    ),
    ray_log_to_driver: bool = typer.Option(
        True,
        "--ray-log-to-driver/--no-ray-log-to-driver",
        help="Forward Ray worker logs to the driver (recommended with --log-file).",
    ),
    detection_summary_file: Optional[Path] = typer.Option(
        None,
        "--detection-summary-file",
        path_type=Path,
        dir_okay=False,
        help="Optional JSON file path to write end-of-run detection counts summary.",
    ),
    embed_granularity: str = typer.Option(
        "element",
        "--embed-granularity",
        help="Embedding granularity: 'element' (one row per table/chart/text) or 'page' (one row per page).",
    ),
) -> None:
    log_handle, original_stdout, original_stderr = configure_cli_logging(log_file)
    try:
        os.environ["RAY_LOG_TO_DRIVER"] = "1" if ray_log_to_driver else "0"
        lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())

        if start_ray:
            subprocess.run(["ray", "start", "--head"], check=True, env=os.environ)
            ray_address = "auto"

        input_path = Path(input_path)
        if input_path.is_file():
            file_patterns = [str(input_path)]
        elif input_path.is_dir():
            file_patterns = [str(input_path / "*.pdf")]
        else:
            raise typer.BadParameter(f"Path does not exist: {input_path}")

        report = run_fused_pipeline(
            FusedPipelineConfig(
                input_path=str(input_path),
                input_type="pdf",
                file_patterns=file_patterns,
                create_params=IngestorCreateParams(ray_address=ray_address, ray_log_to_driver=ray_log_to_driver),
                execute_params=IngestExecuteParams(
                    runtime_metrics_dir=str(runtime_metrics_dir) if runtime_metrics_dir is not None else None,
                    runtime_metrics_prefix=runtime_metrics_prefix,
                ),
                extract_params=ExtractParams(
                    extract_text=True,
                    extract_tables=True,
                    extract_charts=True,
                    extract_infographics=False,
                    batch_tuning={
                        "pdf_extract_workers": int(pdf_extract_tasks),
                        "pdf_extract_num_cpus": float(pdf_extract_cpus_per_task),
                        "pdf_split_batch_size": int(pdf_split_batch_size),
                        "pdf_extract_batch_size": int(pdf_extract_batch_size),
                    },
                ),
                embed_params=EmbedParams(
                    model_name="nemo_retriever_v1",
                    embed_granularity=embed_granularity,
                    fused_tuning={
                        "fused_workers": int(fused_workers),
                        "fused_batch_size": int(fused_batch_size),
                        "fused_cpus_per_actor": float(fused_cpus_per_actor),
                        "fused_gpus_per_actor": float(fused_gpus_per_actor),
                    },
                ),
                vdb_upload_params=VdbUploadParams(
                    lancedb={
                        "lancedb_uri": lancedb_uri,
                        "table_name": LANCEDB_TABLE,
                        "overwrite": True,
                        "create_index": True,
                    }
                ),
                evaluation=RunEvaluationConfig(
                    evaluation_mode="recall",
                    query_csv=str(query_csv),
                ),
                artifacts=RunArtifactConfig(
                    lancedb_uri=lancedb_uri,
                    lancedb_table=LANCEDB_TABLE,
                    detection_summary_file=str(detection_summary_file) if detection_summary_file is not None else None,
                    log_file=str(log_file) if log_file is not None else None,
                ),
            )
        )
        render_fused_run_report(report)
    finally:
        restore_cli_logging(log_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    app()
