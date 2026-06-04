# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from nemo_retriever.ingest.plan import ResolvedIngestPlan
from nemo_retriever.ingest_manifest import format_branch_summary
from nemo_retriever.ingestor import Ingestor, create_ingestor
from nemo_retriever.params import CaptionParams, DedupParams, StoreParams, VdbUploadParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestExecutionResult:
    """Structured result from executing a resolved root ingest plan."""

    plan: ResolvedIngestPlan
    result: object
    n_rows: int | None
    initial_n_rows: int | None
    lancedb_uri: str
    table_name: str
    metadata: dict[str, Any]

    @property
    def documents(self) -> list[str]:
        return list(self.plan.documents)

    @property
    def lancedb_target(self) -> str:
        return f"{self.lancedb_uri}/{self.table_name}"

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "lancedb_uri": self.lancedb_uri,
            "result": self.result,
            "table_name": self.table_name,
            "n_rows": self.n_rows,
        }


@dataclass(frozen=True)
class IngestPipelineStages:
    """Concrete SDK stages resolved from an ingest plan before execution."""

    create_kwargs: dict[str, Any]
    split_config: dict[str, Any] | None
    dedup_params: DedupParams | None
    caption_params: CaptionParams | None
    store_params: StoreParams | None
    vdb_params: VdbUploadParams | None


def ingest_pipeline_stages_from_plan(plan: ResolvedIngestPlan) -> IngestPipelineStages:
    return IngestPipelineStages(
        create_kwargs=dict(plan.create_kwargs),
        split_config=plan.split_config,
        dedup_params=plan.dedup_params,
        caption_params=plan.caption_params,
        store_params=plan.store_params,
        vdb_params=plan.vdb_params,
    )


def build_ingest_pipeline(
    plan: ResolvedIngestPlan,
    *,
    stages: IngestPipelineStages | None = None,
) -> Ingestor:
    """Build the SDK ingest chain from a resolved plan without executing it.

    This is the shared implementation used by root ``retriever ingest`` and
    development callers that add extra stages around the same manifest-routed
    extract/embed path.
    """

    stages = stages or ingest_pipeline_stages_from_plan(plan)
    extract_kwargs = plan.extract_call_kwargs()
    if stages.split_config is not None:
        extract_kwargs["split_config"] = stages.split_config

    ingestor = create_ingestor(**stages.create_kwargs).files(plan.documents)
    ingestor = ingestor.extract(plan.extract_params, **extract_kwargs)
    if stages.dedup_params is not None:
        ingestor = ingestor.dedup(stages.dedup_params)

    if stages.caption_params is not None:
        ingestor = ingestor.caption(stages.caption_params)

    ingestor = ingestor.embed(plan.embed_params) if plan.embed_params is not None else ingestor.embed()
    if stages.store_params is not None:
        ingestor = ingestor.store(stages.store_params)

    if stages.vdb_params is not None:
        ingestor = ingestor.vdb_upload(stages.vdb_params)
    return ingestor


def execute_ingest_plan(
    plan: ResolvedIngestPlan,
    *,
    verify_rows: bool = True,
    raise_on_empty: bool = True,
    stages: IngestPipelineStages | None = None,
) -> IngestExecutionResult:
    """Execute a resolved ingest plan and return structured execution data.

    Root ``retriever ingest`` uses row verification as its public success bar.
    Development wrappers can disable verification while preserving the exact
    shared plan/build/ingest path and layering their own reporting afterward.
    """

    effective_stages = stages or ingest_pipeline_stages_from_plan(plan)
    lancedb_stage = _resolve_lancedb_stage(plan, effective_stages)
    if verify_rows and lancedb_stage is None:
        raise ValueError("Row verification requires an effective VDB upload stage; pass verify_rows=False to skip it.")
    lancedb_uri, table_name, overwrite = lancedb_stage or (plan.lancedb_uri, plan.table_name, True)

    initial_n_rows = None
    if verify_rows and not overwrite:
        initial_n_rows = _count_lancedb_rows(lancedb_uri, table_name)

    result = build_ingest_pipeline(plan, stages=effective_stages).ingest()

    n_rows = _count_lancedb_rows(lancedb_uri, table_name) if verify_rows else None
    if verify_rows and raise_on_empty:
        _raise_for_empty_ingest(
            documents=plan.documents,
            lancedb_uri=lancedb_uri,
            table_name=table_name,
            n_rows=n_rows,
            initial_n_rows=initial_n_rows,
        )

    return IngestExecutionResult(
        plan=plan,
        result=result,
        n_rows=n_rows,
        initial_n_rows=initial_n_rows,
        lancedb_uri=lancedb_uri,
        table_name=table_name,
        metadata={
            "lancedb_target": f"{lancedb_uri}/{table_name}",
            "profile": plan.profile,
            "branch_summary": format_branch_summary(plan.branches),
        },
    )


def _resolve_lancedb_stage(
    plan: ResolvedIngestPlan,
    stages: IngestPipelineStages,
) -> tuple[str, str, bool] | None:
    if stages.vdb_params is None:
        return None
    vdb_kwargs = dict(stages.vdb_params.vdb_kwargs)
    lancedb_uri = str(vdb_kwargs.get("uri") or vdb_kwargs.get("lancedb_uri") or plan.lancedb_uri)
    table_name = str(vdb_kwargs.get("table_name") or vdb_kwargs.get("lancedb_table") or plan.table_name)
    overwrite = bool(vdb_kwargs.get("overwrite", True))
    return lancedb_uri, table_name, overwrite


def _raise_for_empty_ingest(
    *,
    documents: Sequence[str],
    lancedb_uri: str,
    table_name: str,
    n_rows: int | None,
    initial_n_rows: int | None,
) -> None:
    target = f"{lancedb_uri}/{table_name}"
    if n_rows is None:
        raise RuntimeError(
            f"retriever ingest could not verify rows in LanceDB {target} for {len(documents)} input file(s). "
            "This usually means the LanceDB table was not created or could not be read after ingestion; check "
            "the captured stage logs above, and verify NVIDIA_API_KEY/NGC_API_KEY or the configured local/remote "
            "endpoints."
        )
    if n_rows > 0 and (initial_n_rows is None or n_rows > initial_n_rows):
        return

    if initial_n_rows is not None:
        raise RuntimeError(
            f"retriever ingest did not add rows to LanceDB {target}; row count stayed at {n_rows} "
            f"for {len(documents)} input file(s). This usually means extraction or embedding failed before "
            "any rows were written; check the captured stage logs above, and verify NVIDIA_API_KEY/NGC_API_KEY "
            "or the configured local/remote endpoints."
        )

    raise RuntimeError(
        f"retriever ingest produced 0 rows in LanceDB {target} for {len(documents)} input file(s). "
        "This usually means extraction or embedding failed before any rows were written; check the captured "
        "stage logs above, and verify NVIDIA_API_KEY/NGC_API_KEY or the configured local/remote endpoints."
    )


def _count_lancedb_rows(lancedb_uri: str, table_name: str) -> int | None:
    """Return the actual row count in ``<lancedb_uri>/<table_name>`` or ``None``.

    The low-level reader is best-effort so callers can decide whether an
    unknown count is acceptable. Root ingest treats an unknown final count as a
    failure because agents need proof that rows landed.
    """
    try:
        import lancedb  # local import — keeps the CLI startup snappy

        return int(lancedb.connect(lancedb_uri).open_table(table_name).count_rows())
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        logger.debug("could not count rows in %s/%s: %s", lancedb_uri, table_name, exc)
        return None
