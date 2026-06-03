# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Any

from nemo_retriever.adapters.cli.ingest_execution import execute_ingest_plan, ingest_pipeline_stages_from_plan
from nemo_retriever.adapters.cli.ingest_plan import ResolvedIngestPlan
from nemo_retriever.ingest_manifest import format_branch_summary
from nemo_retriever.params import DedupParams, StoreParams

_DRY_RUN_SECRET_FIELD_PATTERNS = ("api_key", "password", "secret", "credential", "bearer")


def _build_dedup_params(
    *,
    dedup: bool,
    dedup_iou_threshold: float | None,
) -> DedupParams | None:
    if not dedup:
        if dedup_iou_threshold is not None:
            raise ValueError("Dedup options require --dedup: dedup_iou_threshold.")
        return None
    dedup_kwargs = {}
    if dedup_iou_threshold is not None:
        dedup_kwargs["iou_threshold"] = dedup_iou_threshold
    return DedupParams(**dedup_kwargs)


def params_to_dry_run_dict(params: Any | None) -> dict[str, Any] | None:
    if params is None:
        return None
    if hasattr(params, "model_dump"):
        data = params.model_dump(mode="json")
    elif isinstance(params, dict):
        data = dict(params)
    else:
        return {"value": str(params)}
    return _strip_secret_values(data)


def _is_dry_run_secret_field(key: Any) -> bool:
    normalized_key = str(key).lower().replace("-", "_")
    return normalized_key.endswith("token") or any(
        pattern in normalized_key for pattern in _DRY_RUN_SECRET_FIELD_PATTERNS
    )


def _strip_secret_values(value: Any) -> Any:
    """Redact secrets from dry-run reporting only."""
    if hasattr(value, "model_dump"):
        return _strip_secret_values(value.model_dump(mode="json"))
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            if _is_dry_run_secret_field(key):
                out[key] = "<redacted>" if nested else nested
            else:
                out[key] = _strip_secret_values(nested)
        return out
    if isinstance(value, list):
        return [_strip_secret_values(item) for item in value]
    return value


def ingest_plan_to_dry_run_data(plan: ResolvedIngestPlan) -> dict[str, Any]:
    return {
        "dry_run": True,
        "profile": plan.profile,
        "documents": list(plan.documents),
        "branches": [
            {
                "family": branch.family,
                "extraction_mode": branch.extraction_mode,
                "count": len(branch.input_paths),
                "input_paths": list(branch.input_paths),
            }
            for branch in plan.branches
        ],
        "branch_summary": format_branch_summary(plan.branches),
        "create_ingestor": dict(plan.create_kwargs),
        "extract": params_to_dry_run_dict(plan.extract_params),
        "text": params_to_dry_run_dict(plan.text_params),
        "html": params_to_dry_run_dict(plan.html_params),
        "audio": params_to_dry_run_dict(plan.audio_chunk_params),
        "asr": params_to_dry_run_dict(plan.asr_params),
        "video_frames": params_to_dry_run_dict(plan.video_frame_params),
        "video_frame_text_dedup": params_to_dry_run_dict(plan.video_text_dedup_params),
        "audio_visual_fuse": params_to_dry_run_dict(plan.av_fuse_params),
        "split_config": params_to_dry_run_dict(plan.split_config),
        "caption": params_to_dry_run_dict(plan.caption_params),
        "embed": params_to_dry_run_dict(plan.embed_params),
        "vdb_upload": params_to_dry_run_dict(plan.vdb_params),
    }


def run_ingest_workflow(
    plan: ResolvedIngestPlan,
    *,
    dry_run: bool = False,
    dedup: bool = False,
    dedup_iou_threshold: float | None = None,
    store_images_uri: str | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Apply root ingest workflow policy to an already-resolved plan."""
    dedup_params = _build_dedup_params(dedup=dedup, dedup_iou_threshold=dedup_iou_threshold)
    store_params = StoreParams(storage_uri=store_images_uri) if store_images_uri is not None else None
    stages = replace(
        ingest_pipeline_stages_from_plan(plan),
        dedup_params=dedup_params,
        store_params=store_params,
    )
    if dry_run:
        dry_run_data = ingest_plan_to_dry_run_data(plan)
        dry_run_data["dedup"] = params_to_dry_run_dict(dedup_params)
        dry_run_data["store"] = params_to_dry_run_dict(store_params)
        return dry_run_data

    return execute_ingest_plan(
        plan,
        overwrite=overwrite,
        stages=stages,
    ).to_summary_dict()
