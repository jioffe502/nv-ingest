# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import logging

from nemo_retriever.ingest_manifest import format_branch_summary
from nemo_retriever.ingestor import Ingestor, create_ingestor
from nemo_retriever.params import DedupParams, StoreParams
from nemo_retriever.adapters.cli.ingest_plan import ResolvedIngestPlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestExecutionResult:
    """Structured result from executing a resolved root ingest plan."""

    plan: ResolvedIngestPlan
    result: object
    n_rows: int | None
    initial_n_rows: int | None
    metadata: dict[str, Any]

    @property
    def documents(self) -> list[str]:
        return list(self.plan.documents)

    @property
    def lancedb_uri(self) -> str:
        return self.plan.lancedb_uri

    @property
    def table_name(self) -> str:
        return self.plan.table_name

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


_USE_PLAN_PARAMS = object()


def build_ingest_pipeline(
    plan: ResolvedIngestPlan,
    *,
    create_kwargs: dict[str, Any] | None = None,
    split_config: Any = _USE_PLAN_PARAMS,
    dedup_params: DedupParams | None = None,
    caption_params: Any = _USE_PLAN_PARAMS,
    store_params: StoreParams | None = None,
    vdb_params: Any = _USE_PLAN_PARAMS,
) -> Ingestor:
    """Build the SDK ingest chain from a resolved plan without executing it.

    This is the shared implementation used by root ``retriever ingest`` and
    development callers that add extra stages around the same manifest-routed
    extract/embed path.
    """

    extract_kwargs = plan.extract_call_kwargs()
    resolved_split_config = plan.split_config if split_config is _USE_PLAN_PARAMS else split_config
    if resolved_split_config is not None:
        extract_kwargs["split_config"] = resolved_split_config

    resolved_create_kwargs = dict(plan.create_kwargs)
    if create_kwargs:
        resolved_create_kwargs.update(create_kwargs)
    ingestor = create_ingestor(**resolved_create_kwargs).files(plan.documents)
    ingestor = ingestor.extract(plan.extract_params, **extract_kwargs)
    if dedup_params is not None:
        ingestor = ingestor.dedup(dedup_params)

    resolved_caption_params = plan.caption_params if caption_params is _USE_PLAN_PARAMS else caption_params
    if resolved_caption_params is not None:
        ingestor = ingestor.caption(resolved_caption_params)

    ingestor = ingestor.embed(plan.embed_params) if plan.embed_params is not None else ingestor.embed()
    if store_params is not None:
        ingestor = ingestor.store(store_params)

    resolved_vdb_params = plan.vdb_params if vdb_params is _USE_PLAN_PARAMS else vdb_params
    if resolved_vdb_params is not None:
        ingestor = ingestor.vdb_upload(resolved_vdb_params)
    return ingestor


def execute_ingest_plan(
    plan: ResolvedIngestPlan,
    *,
    overwrite: bool = True,
    verify_rows: bool = True,
    raise_on_empty: bool = True,
    create_kwargs: dict[str, Any] | None = None,
    split_config: Any = _USE_PLAN_PARAMS,
    dedup_params: DedupParams | None = None,
    caption_params: Any = _USE_PLAN_PARAMS,
    store_params: StoreParams | None = None,
    vdb_params: Any = _USE_PLAN_PARAMS,
) -> IngestExecutionResult:
    """Execute a resolved ingest plan and return structured execution data.

    Root ``retriever ingest`` uses row verification as its public success bar.
    Development wrappers can disable verification while preserving the exact
    shared plan/build/ingest path and layering their own reporting afterward.
    """

    initial_n_rows = None
    if verify_rows and not overwrite:
        initial_n_rows = _count_lancedb_rows(plan.lancedb_uri, plan.table_name)

    result = build_ingest_pipeline(
        plan,
        create_kwargs=create_kwargs,
        split_config=split_config,
        dedup_params=dedup_params,
        caption_params=caption_params,
        store_params=store_params,
        vdb_params=vdb_params,
    ).ingest()

    n_rows = _count_lancedb_rows(plan.lancedb_uri, plan.table_name) if verify_rows else None
    if verify_rows and raise_on_empty:
        _raise_for_empty_ingest(
            documents=plan.documents,
            lancedb_uri=plan.lancedb_uri,
            table_name=plan.table_name,
            n_rows=n_rows,
            initial_n_rows=initial_n_rows,
        )

    return IngestExecutionResult(
        plan=plan,
        result=result,
        n_rows=n_rows,
        initial_n_rows=initial_n_rows,
        metadata={
            "lancedb_target": f"{plan.lancedb_uri}/{plan.table_name}",
            "profile": plan.profile,
            "branch_summary": format_branch_summary(plan.branches),
        },
    )


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
