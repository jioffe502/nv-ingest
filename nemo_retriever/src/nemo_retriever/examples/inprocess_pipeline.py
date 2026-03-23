# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
In-process ingestion pipeline (no Ray) with optional recall evaluation.
Run with: uv run python -m nemo_retriever.examples.inprocess_pipeline <input-dir>
"""

from pathlib import Path
from typing import Optional

import typer
from nemo_retriever.application.modes.reports import RunArtifactConfig, RunEvaluationConfig
from nemo_retriever.application.modes.run_inprocess import (
    InProcessPipelineConfig,
    render_inprocess_run_report,
    run_inprocess_pipeline,
)
from nemo_retriever.application.modes.shared import (
    DEFAULT_LANCEDB_TABLE as LANCEDB_TABLE,
    DEFAULT_LANCEDB_URI as LANCEDB_URI,
)
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.params import VdbUploadParams

app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        help="File or directory containing PDFs, .txt, .html, or .doc/.pptx files to ingest.",
        path_type=Path,
    ),
    input_type: str = typer.Option(
        "pdf",
        "--input-type",
        help="Input format: 'pdf', 'txt', 'html', 'doc', or 'image'. Use 'txt' for .txt, 'html' for .html (markitdown -> chunks), 'doc' for .docx/.pptx (converted to PDF via LibreOffice), 'image' for standalone image files (PNG, JPEG, BMP, TIFF, SVG).",  # noqa: E501
    ),
    query_csv: Path = typer.Option(
        "bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help="Path to query CSV for recall evaluation. Default: bo767_query_gt.csv (current directory). Recall is skipped if the file does not exist.",  # noqa: E501
    ),
    no_recall_details: bool = typer.Option(
        False,
        "--no-recall-details",
        help="Do not print per-query retrieval details (query, gold, hits). Only the missed-gold summary and recall metrics are printed.",  # noqa: E501
    ),
    max_workers: int = typer.Option(
        16,
        "--max-workers",
        help="Maximum number of parallel ingest workers.",
    ),
    gpu_devices: Optional[str] = typer.Option(
        None,
        "--gpu-devices",
        help="Comma-separated GPU device IDs (e.g. --gpu-devices 0,1,2). Mutually exclusive with --num-gpus.",
    ),
    num_gpus: Optional[int] = typer.Option(
        None,
        "--num-gpus",
        help="Number of GPUs to use, starting from device 0 (e.g. --num-gpus 2 → GPUs 0,1). Mutually exclusive with --gpu-devices.",  # noqa: E501
    ),
    page_elements_invoke_url: Optional[str] = typer.Option(
        None,
        "--page-elements-invoke-url",
        help="Optional remote endpoint URL for page-elements model inference.",
    ),
    ocr_invoke_url: Optional[str] = typer.Option(
        None,
        "--ocr-invoke-url",
        help="Optional remote endpoint URL for OCR model inference.",
    ),
    embed_invoke_url: Optional[str] = typer.Option(
        None,
        "--embed-invoke-url",
        help="Optional remote endpoint URL for embedding model inference.",
    ),
    embed_model_name: str = typer.Option(
        "nvidia/llama-nemotron-embed-1b-v2",
        "--embed-model-name",
        help="Embedding model name passed to .embed().",
    ),
    method: str = typer.Option(
        "pdfium",
        "--method",
        help="PDF text extraction method: 'pdfium' (native only), 'pdfium_hybrid' (native + OCR for scanned), 'ocr' (OCR all pages), or 'nemotron_parse' (Nemotron Parse only).",  # noqa: E501
    ),
    embed_modality: str = typer.Option(
        "text",
        "--embed-modality",
        help="Default embedding modality for all element types: 'text', 'image', or 'text_image'.",
    ),
    text_elements_modality: Optional[str] = typer.Option(
        None,
        "--text-elements-modality",
        help="Embedding modality override for page-text rows. Falls back to --embed-modality.",
    ),
    structured_elements_modality: Optional[str] = typer.Option(
        None,
        "--structured-elements-modality",
        help="Embedding modality override for table/chart/infographic rows. Falls back to --embed-modality.",
    ),
    use_table_structure: bool = typer.Option(
        False,
        "--use-table-structure",
        help="Enable the combined table-structure + OCR stage for tables (requires extract_tables).",
    ),
    table_output_format: Optional[str] = typer.Option(
        None,
        "--table-output-format",
        help=(
            "Table output format: 'pseudo_markdown' (OCR-only) or 'markdown' "
            "(table-structure + OCR). Defaults to 'markdown' when table-structure "
            "is enabled, 'pseudo_markdown' otherwise."
        ),
    ),
    table_structure_invoke_url: Optional[str] = typer.Option(
        None,
        "--table-structure-invoke-url",
        help=(
            "Optional remote endpoint URL for table-structure model inference "
            "(used when --table-output-format=markdown)."
        ),
    ),
    embed_granularity: str = typer.Option(
        "element",
        "--embed-granularity",
        help="Embedding granularity: 'element' (one row per table/chart/text) or 'page' (one row per page).",
    ),
    use_graphic_elements: bool = typer.Option(
        False,
        "--use-graphic-elements",
        help="Enable the combined graphic-elements + OCR stage for charts (requires extract_charts).",
    ),
    graphic_elements_invoke_url: Optional[str] = typer.Option(
        None,
        "--graphic-elements-invoke-url",
        help="Optional remote endpoint URL for graphic-elements model inference.",
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid/--no-hybrid",
        help="Enable LanceDB hybrid mode (dense + FTS text).",
    ),
    text_chunk: bool = typer.Option(
        False,
        "--text-chunk",
        help=(
            "Re-chunk extracted page text by token count before embedding. "
            "Uses --text-chunk-max-tokens and --text-chunk-overlap-tokens (defaults: 1024, 150)."
        ),
    ),
    text_chunk_max_tokens: Optional[int] = typer.Option(
        None,
        "--text-chunk-max-tokens",
        help="Max tokens per text chunk (default: 1024). Implies --text-chunk.",
    ),
    text_chunk_overlap_tokens: Optional[int] = typer.Option(
        None,
        "--text-chunk-overlap-tokens",
        help="Token overlap between consecutive text chunks (default: 150). Implies --text-chunk.",
    ),
) -> None:
    if gpu_devices is not None and num_gpus is not None:
        raise typer.BadParameter("--gpu-devices and --num-gpus are mutually exclusive.")
    if gpu_devices is not None:
        gpu_device_list = [d.strip() for d in gpu_devices.split(",") if d.strip()]
    elif num_gpus is not None:
        gpu_device_list = [str(i) for i in range(num_gpus)] if num_gpus > 0 else ["0"]
    else:
        gpu_device_list = ["0"]

    input_path = Path(input_path)
    if input_path.is_file():
        file_patterns = [str(input_path)]
    elif input_path.is_dir():
        ext_map = {
            "txt": ["*.txt"],
            "html": ["*.html"],
            "doc": ["*.docx", "*.pptx"],
            "image": ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif", "*.svg"],
        }
        exts = ext_map.get(input_type, ["*.pdf"])
        file_patterns = [str(input_path / e) for e in exts]
    else:
        raise typer.BadParameter(f"Path does not exist: {input_path}")

    enable_text_chunk = text_chunk or text_chunk_max_tokens is not None or text_chunk_overlap_tokens is not None
    extract_params = ExtractParams(
        method=method,
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=False,
        use_graphic_elements=use_graphic_elements,
        graphic_elements_invoke_url=graphic_elements_invoke_url,
        use_table_structure=use_table_structure,
        table_output_format=table_output_format,
        table_structure_invoke_url=table_structure_invoke_url,
        page_elements_invoke_url=page_elements_invoke_url,
        ocr_invoke_url=ocr_invoke_url,
    )
    chunk_params = TextChunkParams(
        max_tokens=text_chunk_max_tokens or 1024,
        overlap_tokens=text_chunk_overlap_tokens if text_chunk_overlap_tokens is not None else 150,
    )

    report = run_inprocess_pipeline(
        InProcessPipelineConfig(
            input_path=str(input_path),
            input_type=input_type,
            file_patterns=file_patterns,
            execute_params=IngestExecuteParams(
                parallel=True,
                max_workers=max_workers,
                gpu_devices=gpu_device_list,
                show_progress=True,
            ),
            extract_params=extract_params,
            embed_params=EmbedParams(
                model_name=str(embed_model_name),
                embed_invoke_url=embed_invoke_url,
                embed_modality=embed_modality,
                text_elements_modality=text_elements_modality,
                structured_elements_modality=structured_elements_modality,
                embed_granularity=embed_granularity,
            ),
            text_chunk_params=chunk_params,
            enable_text_chunk=enable_text_chunk,
            vdb_upload_params=VdbUploadParams(
                lancedb={
                    "lancedb_uri": LANCEDB_URI,
                    "table_name": LANCEDB_TABLE,
                    "overwrite": True,
                    "create_index": True,
                    "hybrid": hybrid,
                }
            ),
            evaluation=RunEvaluationConfig(
                evaluation_mode="recall",
                query_csv=str(query_csv),
            ),
            artifacts=RunArtifactConfig(
                lancedb_uri=LANCEDB_URI,
                lancedb_table=LANCEDB_TABLE,
            ),
        )
    )
    render_inprocess_run_report(report, include_ingest_summary=False)


if __name__ == "__main__":
    app()
