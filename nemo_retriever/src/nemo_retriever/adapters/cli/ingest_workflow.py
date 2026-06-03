# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_retriever.adapters.cli.ingest_execution import execute_ingest_plan
from nemo_retriever.adapters.cli.ingest_plan import ResolvedIngestPlan, params_to_dry_run_dict
from nemo_retriever.params import DedupParams, StoreParams


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
    if dry_run:
        dry_run_data = plan.dry_run_data()
        dry_run_data["dedup"] = params_to_dry_run_dict(dedup_params)
        dry_run_data["store"] = params_to_dry_run_dict(store_params)
        return dry_run_data

    return execute_ingest_plan(
        plan,
        overwrite=overwrite,
        dedup_params=dedup_params,
        store_params=store_params,
    ).to_summary_dict()
