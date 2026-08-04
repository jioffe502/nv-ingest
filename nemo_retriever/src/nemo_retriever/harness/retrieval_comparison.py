# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay dense, sparse, and hybrid retrieval against one completed ViDoRe index."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import lancedb
from nemo_retriever.common.vdb.hybrid_fusion import DEFAULT_HYBRID_FUSION_POLICY
from nemo_retriever.common.vdb.lancedb_capabilities import inspect_lancedb_table_object
from nemo_retriever.harness.json_io import write_json
from nemo_retriever.query.workflow import ResolvedQueryPlan
from nemo_retriever.tools.recall.beir import (
    BeirDataset,
    build_beir_run_from_hits,
    compute_beir_metrics,
    load_beir_dataset,
)

SUPPORTED_COMPARISON_MODES = ("dense", "sparse", "hybrid")
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_ANCHOR_RE = re.compile(r"(?:[$€£¥]\s?\d|\d(?:[.,]\d+)?\s?%|\b[A-Z]{2,}[A-Z0-9_-]*\d[A-Z0-9_-]*\b|\b\d{2,}\b)")
_LABEL_FIELD_HINTS = (
    "format",
    "type",
    "evidence",
    "source",
    "origin",
    "generator",
    "generation",
    "domain",
    "language",
    "synthetic",
    "human",
    "hop",
)
_REFERENCE_FIELD_HINTS = (
    "answer",
    "reference",
    "text",
    "markdown",
    "caption",
    "content",
    "ocr",
    "title",
    "entity",
)


@dataclass(frozen=True)
class LoadedVidoreMetadata:
    query_rows: dict[str, dict[str, Any]]
    corpus_rows: dict[str, dict[str, Any]]
    qrel_rows: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    corpus_image_indices: dict[str, int] = field(default_factory=dict)
    corpus_dataset: Any | None = field(default=None, repr=False, compare=False)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Required run artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _safe_scalar(value: Any) -> Any | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(item, (str, int, float, bool)) for item in value):
        return list(value)
    return None


def _tokens(value: Any) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(str(value or "")) if len(token) > 1}


def _flatten_reference_text(row: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key, value in row.items():
        normalized = key.casefold()
        if not any(hint in normalized for hint in _REFERENCE_FIELD_HINTS):
            continue
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value if isinstance(item, (str, int, float)))
    return "\n".join(values)


def _existing_labels(row: Mapping[str, Any]) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    for key, value in row.items():
        normalized = key.casefold()
        if key in {"query", "query_id", "corpus_id", "doc_id", "image"}:
            continue
        if not any(hint in normalized for hint in _LABEL_FIELD_HINTS):
            continue
        safe_value = _safe_scalar(value)
        if safe_value not in (None, "", []):
            labels[key] = safe_value
    return labels


def _doc_id_from_corpus_row(row: Mapping[str, Any], *, doc_id_field: str) -> str | None:
    doc_id = str(row.get("doc_id") or "").strip()
    if doc_id_field == "source_id":
        value = row.get("corpus_id")
        return str(value) if value is not None else None
    if doc_id_field == "path":
        return f"{doc_id}.pdf" if doc_id else None
    if doc_id_field == "pdf_basename":
        return doc_id or None
    if doc_id_field == "pdf_page":
        try:
            page = int(row.get("page_number_in_doc")) + 1
        except (TypeError, ValueError):
            return None
        return f"{doc_id}_{page}" if doc_id else None
    return None


def _load_vidore_metadata(dataset_name: str, split: str, *, doc_id_field: str) -> LoadedVidoreMetadata:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ValueError("ViDoRe comparison requires the datasets package") from exc

    repository = f"vidore/{dataset_name}"

    def load_partition(name: str):
        try:
            return load_dataset(repository, name, split=split)
        except Exception:
            return load_dataset(repository, data_dir=name, split=split)

    query_rows: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(load_partition("queries")):
        payload = dict(row)
        query_rows[str(payload.get("query_id", index))] = payload

    corpus_partition = load_partition("corpus")
    if "image" in corpus_partition.column_names:
        from datasets import Image as DatasetImage  # type: ignore

        corpus_partition = corpus_partition.cast_column("image", DatasetImage(decode=False))

    corpus_rows: dict[str, dict[str, Any]] = {}
    corpus_id_to_doc_id: dict[str, str] = {}
    corpus_image_indices: dict[str, int] = {}
    for index, row in enumerate(corpus_partition):
        payload = dict(row)
        payload.pop("image", None)
        doc_id = _doc_id_from_corpus_row(payload, doc_id_field=doc_id_field)
        if doc_id:
            corpus_rows[doc_id] = payload
            corpus_id_to_doc_id[str(payload.get("corpus_id"))] = doc_id
            corpus_image_indices[doc_id] = index

    qrel_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_partition("qrels"):
        payload = dict(row)
        query_id = str(payload.get("query_id"))
        doc_id = corpus_id_to_doc_id.get(str(payload.get("corpus_id")))
        if doc_id:
            qrel_rows[(query_id, doc_id)] = payload
    return LoadedVidoreMetadata(
        query_rows=query_rows,
        corpus_rows=corpus_rows,
        qrel_rows=qrel_rows,
        corpus_image_indices=corpus_image_indices,
        corpus_dataset=corpus_partition,
    )


def _parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _indexed_doc_id(row: Mapping[str, Any], *, doc_id_field: str) -> str | None:
    metadata = _parse_mapping(row.get("metadata"))
    source_metadata = _parse_mapping(row.get("source"))
    source = (
        row.get("pdf_basename")
        or metadata.get("pdf_basename")
        or source_metadata.get("source_name")
        or source_metadata.get("source_id")
        or row.get("source_id")
    )
    if doc_id_field == "pdf_page":
        direct = row.get("pdf_page") or metadata.get("pdf_page")
        if direct:
            return str(direct)
        page = row.get("page_number", metadata.get("page_number"))
        if source and page is not None:
            basename = Path(str(source)).name.removesuffix(".pdf")
            return f"{basename}_{int(page)}"
    if doc_id_field == "pdf_basename":
        return Path(str(source)).name.removesuffix(".pdf") if source else None
    value = row.get(doc_id_field, metadata.get(doc_id_field))
    return str(value) if value not in (None, "") else None


def _load_indexed_text(table: Any, *, doc_id_field: str) -> dict[str, list[str]]:
    mapped: defaultdict[str, list[str]] = defaultdict(list)
    for row in table.to_arrow().to_pylist():
        doc_id = _indexed_doc_id(row, doc_id_field=doc_id_field)
        text = str(row.get("text") or "").strip()
        if doc_id:
            # An indexed page with empty text is an extraction concern, not a missing page.
            mapped[doc_id].append(text)
    return dict(mapped)


def _named_gold_entity(query: str, gold_rows: Iterable[Mapping[str, Any]]) -> bool:
    normalized_query = " ".join(query.casefold().split())
    query_tokens = _tokens(query)
    for row in gold_rows:
        for key, value in row.items():
            if not any(hint in key.casefold() for hint in ("title", "entity", "doc_id", "name")):
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            candidate = " ".join(value.casefold().replace("_", " ").split())
            candidate_tokens = _tokens(candidate)
            if candidate in normalized_query or (len(candidate_tokens) >= 2 and candidate_tokens <= query_tokens):
                return True
    return False


def _lexical_overlap(query: str, reference: str) -> float | None:
    query_tokens = _tokens(query)
    reference_tokens = _tokens(reference)
    if not query_tokens or not reference_tokens:
        return None
    return len(query_tokens & reference_tokens) / len(query_tokens)


def _terciles(values: Mapping[str, float | None]) -> dict[str, str]:
    observed = sorted(value for value in values.values() if value is not None)
    if not observed:
        return {key: "unavailable" for key in values}
    low = observed[(len(observed) - 1) // 3]
    high = observed[(2 * (len(observed) - 1)) // 3]
    output: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            output[key] = "unavailable"
        elif value <= low:
            output[key] = "low"
        elif value <= high:
            output[key] = "middle"
        else:
            output[key] = "high"
    return output


def _best_relevant_rank(hits: Sequence[Mapping[str, Any]], relevant: set[str], *, doc_id_field: str) -> int | None:
    seen: set[str] = set()
    rank = 0
    for hit in hits:
        value = hit.get(doc_id_field)
        if value is None and doc_id_field == "pdf_page":
            value = _indexed_doc_id(hit, doc_id_field=doc_id_field)
        if value is None or str(value) in seen:
            continue
        seen.add(str(value))
        rank += 1
        if str(value) in relevant:
            return rank
    return None


def _query_metrics(
    hits: Sequence[Mapping[str, Any]], qrels: Mapping[str, int], *, doc_id_field: str
) -> dict[str, float | int | None]:
    ranked: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        value = hit.get(doc_id_field)
        if value is None:
            value = _indexed_doc_id(hit, doc_id_field=doc_id_field)
        if value is None or str(value) in seen:
            continue
        seen.add(str(value))
        ranked.append(str(value))
    relevant = {doc_id for doc_id, score in qrels.items() if int(score) > 0}
    rel_count = max(1, len(relevant))

    def recall(k: int) -> float:
        return len(set(ranked[:k]) & relevant) / rel_count

    dcg = sum(float(qrels.get(doc_id, 0)) / math.log2(rank + 1) for rank, doc_id in enumerate(ranked[:10], start=1))
    ideal_scores = sorted((int(score) for score in qrels.values() if int(score) > 0), reverse=True)[:10]
    ideal = sum(float(score) / math.log2(rank + 1) for rank, score in enumerate(ideal_scores, start=1))
    return {
        "recall_5": recall(5),
        "recall_10": recall(10),
        "ndcg_10": dcg / ideal if ideal else 0.0,
        "best_relevant_rank": _best_relevant_rank(hits, relevant, doc_id_field=doc_id_field),
    }


def _audit_qrel(
    *,
    query_row: Mapping[str, Any],
    corpus_row: Mapping[str, Any],
    indexed_texts: Sequence[str],
) -> dict[str, Any]:
    if not indexed_texts:
        return {"status": "missing_indexed_page", "answer_token_recall": None, "reference_token_recall": None}

    indexed = "\n".join(indexed_texts)
    answer_text = " ".join(
        str(value)
        for key, value in query_row.items()
        if "answer" in key.casefold() and isinstance(value, (str, int, float))
    )
    reference_text = _flatten_reference_text(corpus_row)
    answer_tokens = _tokens(answer_text)
    reference_tokens = _tokens(reference_text)
    indexed_tokens = _tokens(indexed)
    answer_recall = len(answer_tokens & indexed_tokens) / len(answer_tokens) if answer_tokens else None
    reference_recall = len(reference_tokens & indexed_tokens) / len(reference_tokens) if reference_tokens else None
    anchors = _ANCHOR_RE.findall(f"{answer_text} {reference_text}")
    normalized_indexed = indexed.casefold()
    anchors_covered = bool(anchors) and all(anchor.casefold() in normalized_indexed for anchor in anchors)

    if answer_recall is None and reference_recall is None and not anchors:
        status = "not_machine_verifiable"
    elif (
        (answer_recall is not None and answer_recall >= 0.6)
        or (reference_recall is not None and reference_recall >= 0.35)
        or anchors_covered
    ):
        status = "covered"
    else:
        status = "needs_review"
    return {
        "status": status,
        "answer_token_recall": answer_recall,
        "reference_token_recall": reference_recall,
        "anchors": anchors[:20],
        "anchors_covered": anchors_covered,
        "indexed_text_excerpts": [text[:500] for text in indexed_texts[:3]],
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _paired_ci(values: Sequence[float], *, seed: int) -> list[float] | None:
    """Return a deterministic paired normal-approximation confidence interval."""
    del seed  # Kept in the interface so each reported comparison has a stable identity.
    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return [mean, mean]
    margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - margin, mean + margin]


def _mode_summary(cases: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    latencies = [float(case["modes"][mode]["latency_ms"]) for case in cases]
    return {
        "query_count": len(cases),
        "recall_5": statistics.fmean(float(case["modes"][mode]["recall_5"]) for case in cases),
        "recall_10": statistics.fmean(float(case["modes"][mode]["recall_10"]) for case in cases),
        "ndcg_10": statistics.fmean(float(case["modes"][mode]["ndcg_10"]) for case in cases),
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
    }


def _comparable_rank(case: Mapping[str, Any], mode: str) -> int:
    """Use one rank beyond the replay depth for a missing relevant result."""
    rank = case["modes"][mode]["best_relevant_rank"]
    if rank is not None:
        return int(rank)
    replay_depth = max(len(payload["top_doc_ids"]) for payload in case["modes"].values())
    return replay_depth + 1


def _paired_delta_summary(cases: Sequence[Mapping[str, Any]], *, baseline: str, seed_prefix: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in ("recall_5", "recall_10", "ndcg_10"):
        deltas = [float(case["modes"]["hybrid"][metric]) - float(case["modes"][baseline][metric]) for case in cases]
        metrics[metric] = {
            "mean": statistics.fmean(deltas),
            "ci95": _paired_ci(deltas, seed=sum(ord(char) for char in f"{seed_prefix}:{baseline}:{metric}")),
        }
    rank_deltas = [float(_comparable_rank(case, baseline) - _comparable_rank(case, "hybrid")) for case in cases]
    metrics["best_relevant_rank"] = {
        "mean_improvement": statistics.fmean(rank_deltas) if rank_deltas else None,
        "paired_sample_count": len(rank_deltas),
        "ci95": _paired_ci(rank_deltas, seed=sum(ord(char) for char in f"{seed_prefix}:{baseline}:rank")),
    }
    return metrics


def _slice_memberships(case: Mapping[str, Any]) -> list[tuple[str, str]]:
    memberships: list[tuple[str, str]] = []
    for key, value in case.get("labels", {}).items():
        values = value if isinstance(value, list) else [value]
        memberships.extend((str(key), str(item)) for item in values)
    for key, value in case.get("derived_slices", {}).items():
        memberships.append((str(key), str(value)))
    return memberships


def _slice_summaries(cases: Sequence[Mapping[str, Any]], modes: Sequence[str]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        for membership in _slice_memberships(case):
            groups[membership].append(case)
    summaries: list[dict[str, Any]] = []
    for (slice_field, value), members in sorted(groups.items()):
        if len(members) < 30:
            continue
        payload: dict[str, Any] = {
            "field": slice_field,
            "value": value,
            "sample_count": len(members),
            "modes": {},
        }
        for mode in modes:
            payload["modes"][mode] = _mode_summary(members, mode)
        if "hybrid" in modes:
            payload["paired_deltas"] = {}
            for baseline in ("dense", "sparse"):
                if baseline not in modes:
                    continue
                payload["paired_deltas"][f"hybrid_vs_{baseline}"] = _paired_delta_summary(
                    members,
                    baseline=baseline,
                    seed_prefix=f"{slice_field}:{value}",
                )
        summaries.append(payload)
    return summaries


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _read_query_results(path: Path, *, mode: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"Required source query results are missing: {path}")
    results: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                results.append({**payload, "mode": mode})
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read source query results {path}: {exc}") from exc
    return results


def _write_trec(path: Path, query_results: Sequence[Mapping[str, Any]], *, doc_id_field: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in query_results:
            for rank, hit in enumerate(result["hits"], start=1):
                doc_id = hit.get(doc_id_field) or _indexed_doc_id(hit, doc_id_field=doc_id_field) or "unknown"
                score = float(hit.get("_relevance_score", 1.0 / rank))
                handle.write(f"{result['query_id']} Q0 {doc_id} {rank} {score:.6f} nrl-{result['mode']}\n")


def _write_report(path: Path, summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Retrieval comparison",
        "",
        "All modes replay the same extracted LanceDB rows. Extraction statuses are deterministic heuristics; "
        "`needs_review` is not a confirmed extraction failure.",
        "Best-relevant-rank deltas assign misses one rank beyond the replay depth so rescues and regressions remain paired.",
        "",
        "## Overall",
        "",
        "| Mode | Queries | Recall@5 | Recall@10 | nDCG@10 | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in summary["modes"].items():
        lines.append(
            f"| {mode} | {metrics['query_count']} | {metrics['recall_5']:.4f} | {metrics['recall_10']:.4f} | "
            f"{metrics['ndcg_10']:.4f} | {metrics['latency_p50_ms']:.2f} | {metrics['latency_p95_ms']:.2f} |"
        )
    lines.append("")
    for comparison, metrics in summary.get("paired_deltas", {}).items():
        recall = metrics["recall_10"]
        ndcg = metrics["ndcg_10"]
        rank = metrics["best_relevant_rank"]
        lines.append(
            f"- {comparison}: paired ΔR@10 {recall['mean']:.4f} (95% CI {recall['ci95']}), "
            f"ΔnDCG@10 {ndcg['mean']:.4f} (95% CI {ndcg['ci95']}), "
            f"best-rank improvement {rank['mean_improvement']} (95% CI {rank['ci95']}, n={rank['paired_sample_count']})"
        )
    lines.extend(["", "## Slices with at least 30 queries", ""])
    for slice_result in summary.get("slices", []):
        lines.append(f"### {slice_result['field']} = {slice_result['value']} (n={slice_result['sample_count']})")
        lines.append("")
        for mode, metrics in slice_result["modes"].items():
            lines.append(
                f"- {mode}: R@5 {metrics['recall_5']:.4f}, R@10 {metrics['recall_10']:.4f}, "
                f"nDCG@10 {metrics['ndcg_10']:.4f}, p50/p95 {metrics['latency_p50_ms']:.2f}/{metrics['latency_p95_ms']:.2f} ms"
            )
        for comparison, metrics in slice_result.get("paired_deltas", {}).items():
            recall = metrics["recall_10"]
            ndcg = metrics["ndcg_10"]
            rank = metrics["best_relevant_rank"]
            lines.append(
                f"- {comparison}: paired ΔR@10 {recall['mean']:.4f} (95% CI {recall['ci95']}), "
                f"ΔnDCG@10 {ndcg['mean']:.4f} (95% CI {ndcg['ci95']}), "
                f"best-rank improvement {rank['mean_improvement']} (95% CI {rank['ci95']}, n={rank['paired_sample_count']})"
            )
        lines.append("")
    lines.extend(["## Representative cases", ""])
    categories = Counter(case["comparison_category"] for case in cases)
    for category, count in sorted(categories.items()):
        lines.append(f"- {category}: {count}")
    lines.extend(
        [
            "",
            "See `query_cases.jsonl` for ranked hits and indexed-text excerpts, and `evidence_crops/` for available annotated page evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _bbox_values(value: Any) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if isinstance(value, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(key in value for key in keys):
            boxes.append(tuple(float(value[key]) for key in keys))
        for nested in value.values():
            boxes.extend(_bbox_values(nested))
    elif isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            boxes.append(tuple(float(item) for item in value))
        else:
            for nested in value:
                boxes.extend(_bbox_values(nested))
    return boxes


def _evidence_case_severity(case: Mapping[str, Any]) -> float:
    category = case["comparison_category"]
    dense_delta = float(case.get("paired_deltas", {}).get("hybrid_vs_dense", {}).get("recall_10", 0.0))
    if category == "hybrid_rescue":
        return dense_delta
    if category == "hybrid_regression":
        return -dense_delta
    status_weight = {"missing_indexed_page": 3.0, "needs_review": 2.0, "not_machine_verifiable": 1.0}
    return max((status_weight.get(audit["status"], 0.0) for audit in case.get("extraction_audit", [])), default=0.0)


def _evidence_doc_id(case: Mapping[str, Any]) -> str | None:
    status_weight = {"missing_indexed_page": 3, "needs_review": 2, "not_machine_verifiable": 1, "covered": 0}
    audits = sorted(
        case.get("extraction_audit", []),
        key=lambda audit: (
            bool(audit.get("vidore_qrel", {}).get("bounding_boxes")),
            status_weight.get(audit.get("status"), 0),
        ),
        reverse=True,
    )
    if audits:
        return str(audits[0]["doc_id"])
    return next(iter(case.get("relevant_doc_ids", [])), None)


def _write_evidence_crops(
    output_dir: Path,
    cases: Sequence[Mapping[str, Any]],
    metadata: LoadedVidoreMetadata,
) -> dict[str, list[str]]:
    crops: defaultdict[str, list[str]] = defaultdict(list)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return {}
    for category in ("hybrid_rescue", "hybrid_regression", "all_modes_miss", "extraction_concern"):
        candidates = sorted(
            (case for case in cases if case["comparison_category"] == category),
            key=_evidence_case_severity,
            reverse=True,
        )[:5]
        for case in candidates:
            doc_id = _evidence_doc_id(case)
            image_index = metadata.corpus_image_indices.get(str(doc_id))
            image_value = (
                metadata.corpus_dataset[image_index].get("image")
                if metadata.corpus_dataset is not None and image_index is not None
                else None
            )
            if isinstance(image_value, dict) and image_value.get("bytes"):
                image = Image.open(BytesIO(image_value["bytes"]))
            elif isinstance(image_value, dict) and image_value.get("path"):
                image = Image.open(image_value["path"])
            else:
                image = image_value
            if image is None or not hasattr(image, "copy"):
                continue
            rendered = image.copy().convert("RGB")
            draw = ImageDraw.Draw(rendered)
            boxes = _bbox_values(metadata.qrel_rows.get((str(case["query_id"]), str(doc_id)), {}))
            pixel_boxes: list[tuple[float, float, float, float]] = []
            for x1, y1, x2, y2 in boxes:
                if max(x1, y1, x2, y2) <= 1.0:
                    x1, x2 = x1 * rendered.width, x2 * rendered.width
                    y1, y2 = y1 * rendered.height, y2 * rendered.height
                pixel_boxes.append((x1, y1, x2, y2))
                draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
            if pixel_boxes:
                padding = max(16, int(0.02 * max(rendered.width, rendered.height)))
                rendered = rendered.crop(
                    (
                        max(0, min(box[0] for box in pixel_boxes) - padding),
                        max(0, min(box[1] for box in pixel_boxes) - padding),
                        min(rendered.width, max(box[2] for box in pixel_boxes) + padding),
                        min(rendered.height, max(box[3] for box in pixel_boxes) + padding),
                    )
                )
            target = output_dir / "evidence_crops" / category / f"{case['query_id']}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            rendered.save(target)
            crops[category].append(str(target.relative_to(output_dir)))
    return dict(crops)


def build_comparison_artifacts(
    *,
    output_dir: Path,
    dataset: BeirDataset,
    metadata: LoadedVidoreMetadata,
    indexed_text: Mapping[str, Sequence[str]],
    mode_results: Mapping[str, Sequence[Mapping[str, Any]]],
    doc_id_field: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Analyze complete per-mode results and write the comparison artifact contract."""
    modes = tuple(mode_results)
    for mode, results in mode_results.items():
        observed = {str(result["query_id"]) for result in results}
        missing = set(dataset.query_ids) - observed
        if missing:
            raise ValueError(f"Mode {mode!r} is missing results for {len(missing)} queries")

    overlap_values: dict[str, float | None] = {}
    for query_id, query in zip(dataset.query_ids, dataset.queries):
        relevant = dataset.qrels[query_id]
        reference = " ".join(_flatten_reference_text(metadata.corpus_rows.get(doc_id, {})) for doc_id in relevant)
        overlap_values[query_id] = _lexical_overlap(query, reference)
    overlap_terciles = _terciles(overlap_values)

    results_by_mode = {
        mode: {str(result["query_id"]): result for result in results} for mode, results in mode_results.items()
    }
    extraction_cases: list[dict[str, Any]] = []
    extraction_by_query: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    cases: list[dict[str, Any]] = []
    for query_id, query in zip(dataset.query_ids, dataset.queries):
        relevant = set(dataset.qrels[query_id])
        query_qrels = dataset.qrels[query_id]
        query_row = metadata.query_rows.get(query_id, {})
        gold_rows = [metadata.corpus_rows.get(doc_id, {}) for doc_id in relevant]
        labels = _existing_labels(query_row)
        derived = {
            "names_relevant_document_or_entity": _named_gold_entity(query, gold_rows),
            "lexical_overlap_tercile": overlap_terciles[query_id],
            "has_numeric_currency_percentage_or_identifier_anchor": bool(_ANCHOR_RE.search(query)),
            "relevance_scope": "single_page" if len(relevant) == 1 else "multi_page",
        }
        mode_payload: dict[str, Any] = {}
        for mode in modes:
            result = results_by_mode[mode][query_id]
            metrics = _query_metrics(result["hits"], query_qrels, doc_id_field=doc_id_field)
            mode_payload[mode] = {
                "latency_ms": float(result["latency_ms"]),
                **metrics,
                "top_doc_ids": [
                    str(hit.get(doc_id_field) or _indexed_doc_id(hit, doc_id_field=doc_id_field) or "")
                    for hit in result["hits"][:10]
                ],
            }
        for doc_id in sorted(relevant):
            audit = {
                "query_id": query_id,
                "doc_id": doc_id,
                "vidore_qrel": {
                    key: value
                    for key, value in metadata.qrel_rows.get((query_id, doc_id), {}).items()
                    if key in {"score", "content_type", "bounding_boxes"}
                },
                **_audit_qrel(
                    query_row=query_row,
                    corpus_row=metadata.corpus_rows.get(doc_id, {}),
                    indexed_texts=indexed_text.get(doc_id, ()),
                ),
            }
            extraction_cases.append(audit)
            extraction_by_query[query_id].append(audit)

        dense_hit = mode_payload.get("dense", {}).get("recall_10", 0.0) > 0
        hybrid_hit = mode_payload.get("hybrid", {}).get("recall_10", 0.0) > 0
        selected_hits = [payload.get("recall_10", 0.0) > 0 for payload in mode_payload.values()]
        if "dense" in modes and "hybrid" in modes and hybrid_hit and not dense_hit:
            category = "hybrid_rescue"
        elif "dense" in modes and "hybrid" in modes and dense_hit and not hybrid_hit:
            category = "hybrid_regression"
        elif not any(selected_hits):
            category = "all_modes_miss"
        elif any(item["status"] in {"missing_indexed_page", "needs_review"} for item in extraction_by_query[query_id]):
            category = "extraction_concern"
        else:
            category = "other"
        paired_deltas: dict[str, dict[str, float | int | None]] = {}
        if "hybrid" in modes:
            for baseline in ("dense", "sparse"):
                if baseline not in modes:
                    continue
                paired_deltas[f"hybrid_vs_{baseline}"] = {
                    metric: float(mode_payload["hybrid"][metric]) - float(mode_payload[baseline][metric])
                    for metric in ("recall_5", "recall_10", "ndcg_10")
                }
                replay_depth = max(len(payload["top_doc_ids"]) for payload in mode_payload.values())
                hybrid_rank = mode_payload["hybrid"]["best_relevant_rank"] or replay_depth + 1
                baseline_rank = mode_payload[baseline]["best_relevant_rank"] or replay_depth + 1
                paired_deltas[f"hybrid_vs_{baseline}"]["best_relevant_rank_improvement"] = int(baseline_rank) - int(
                    hybrid_rank
                )
        cases.append(
            {
                "query_id": query_id,
                "query": query,
                "labels": labels,
                "derived_slices": derived,
                "relevant_doc_ids": sorted(relevant),
                "relevant_page_count": len(relevant),
                "modes": mode_payload,
                "paired_deltas": paired_deltas,
                "comparison_category": category,
                "extraction_audit": extraction_by_query[query_id],
            }
        )

    audit_counts = Counter(item["status"] for item in extraction_cases)
    extraction_audit = {
        "total_positive_qrels": len(extraction_cases),
        "classified_positive_qrels": len(extraction_cases),
        "status_counts": dict(sorted(audit_counts.items())),
        "method": {
            "covered": "answer token recall >= 0.60, reference-page token recall >= 0.35, or all anchors present",
            "warning": "needs_review is a weak-overlap heuristic and is never a confirmed extraction failure",
        },
        "cases": extraction_cases,
    }
    summary = {
        **dict(provenance),
        "query_count": len(cases),
        "positive_qrel_count": len(extraction_cases),
        "modes": {mode: _mode_summary(cases, mode) for mode in modes},
        "paired_deltas": {
            f"hybrid_vs_{baseline}": _paired_delta_summary(cases, baseline=baseline, seed_prefix="overall")
            for baseline in ("dense", "sparse")
            if "hybrid" in modes and baseline in modes
        },
        "slices": _slice_summaries(cases, modes),
        "case_counts": dict(sorted(Counter(case["comparison_category"] for case in cases).items())),
        "best_relevant_rank_method": "A miss is assigned replay_depth + 1 for paired deltas.",
    }
    crops = _write_evidence_crops(output_dir, cases, metadata)
    summary["evidence_crops"] = crops
    write_json(output_dir / "summary.json", summary)
    _write_jsonl(output_dir / "query_cases.jsonl", cases)
    write_json(output_dir / "extraction_audit.json", extraction_audit)
    _write_report(output_dir / "report.md", summary, cases)

    for mode, results in mode_results.items():
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(mode_dir / "query_results.jsonl", results)
        run = build_beir_run_from_hits(
            dataset.query_ids,
            [results_by_mode[mode][query_id]["hits"] for query_id in dataset.query_ids],
            doc_id_field=doc_id_field,
        )
        metrics = compute_beir_metrics(dataset.qrels, run, ks=(1, 3, 5, 10))
        write_json(mode_dir / "beir_metrics.json", metrics)
        _write_trec(mode_dir / "beir_run.trec", results, doc_id_field=doc_id_field)
    return summary


def _resolved_query_plan(payload: Mapping[str, Any], mode: str) -> ResolvedQueryPlan:
    return ResolvedQueryPlan(
        top_k=int(payload.get("top_k", 10)),
        candidate_k=payload.get("candidate_k"),
        page_dedup=bool(payload.get("page_dedup", False)),
        content_types=payload.get("content_types"),
        lancedb_uri=str(payload["lancedb_uri"]),
        table_name=str(payload["table_name"]),
        retrieval_mode=mode,
        embed_kwargs=dict(payload.get("embed_kwargs") or {}),
        rerank=bool(payload.get("rerank", False)),
        rerank_kwargs=dict(payload.get("rerank_kwargs") or {}),
    )


def _run_mode_queries(dataset: BeirDataset, plan: ResolvedQueryPlan, mode: str) -> list[dict[str, Any]]:
    retriever = plan.create_retriever()
    query_kwargs = plan.query_kwargs()
    results: list[dict[str, Any]] = []
    for query_id, query in zip(dataset.query_ids, dataset.queries):
        started = time.perf_counter()
        hits = [dict(hit) for hit in retriever.query(query, **query_kwargs)]
        results.append(
            {
                "query_id": query_id,
                "query": query,
                "mode": mode,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "hits": hits,
            }
        )
    return results


def compare_retrieval(
    run_dir: Path,
    *,
    modes: Sequence[str] = SUPPORTED_COMPARISON_MODES,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate a completed ViDoRe run, replay its index, and write diagnostics."""
    run_dir = run_dir.expanduser().resolve()
    normalized_modes = tuple(dict.fromkeys(str(mode).strip().lower() for mode in modes))
    invalid_modes = sorted(set(normalized_modes) - set(SUPPORTED_COMPARISON_MODES))
    if invalid_modes or not normalized_modes:
        raise ValueError(f"modes must be a non-empty subset of {', '.join(SUPPORTED_COMPARISON_MODES)}")

    status = _read_json(run_dir / "status.json")
    results = _read_json(run_dir / "results.json")
    if status.get("status") != "complete" or not results.get("success") or results.get("dry_run"):
        raise ValueError(f"RUN_DIR must contain a completed, successful, non-dry-run harness run: {run_dir}")
    if results.get("mode") not in {"local", "batch"}:
        raise ValueError("compare-retrieval requires a completed local or batch run")

    resolved = _read_json(run_dir / "resolved_benchmark.json")
    evaluation = dict(resolved.get("evaluation") or {})
    if evaluation.get("loader") != "vidore_hf":
        raise ValueError("compare-retrieval currently requires a ViDoRe Hugging Face evaluation run")
    query_plan_payload = _read_json(run_dir / "query_plan.json")
    if query_plan_payload.get("rerank"):
        raise ValueError("compare-retrieval does not run model reranking; use an unrereanked source run")

    table = lancedb.connect(str(query_plan_payload["lancedb_uri"])).open_table(str(query_plan_payload["table_name"]))
    caps = inspect_lancedb_table_object(table)
    if not (caps.has_vector and caps.has_fts):
        raise ValueError("RUN_DIR must contain a hybrid-capable LanceDB table with vector and FTS capabilities")

    target = (output_dir or run_dir / "retrieval_comparison").expanduser().resolve()
    if target.exists():
        raise ValueError(f"Refusing to overwrite existing comparison directory: {target}")

    dataset_name = str(evaluation.get("dataset_name") or "")
    split = str(evaluation.get("split") or "test")
    doc_id_field = str(evaluation.get("doc_id_field") or "pdf_page")
    dataset = load_beir_dataset(
        "vidore_hf",
        dataset_name=dataset_name,
        split=split,
        query_language=evaluation.get("query_language"),
        doc_id_field=doc_id_field,
    )
    metadata = _load_vidore_metadata(dataset_name, split, doc_id_field=doc_id_field)
    indexed_text = _load_indexed_text(table, doc_id_field=doc_id_field)
    mode_results: dict[str, Sequence[Mapping[str, Any]]] = {}
    for mode in normalized_modes:
        mode_results[mode] = _run_mode_queries(dataset, _resolved_query_plan(query_plan_payload, mode), mode)
    target.mkdir(parents=True, exist_ok=False)
    lancedb_version = getattr(lancedb, "__version__", "unknown")
    return build_comparison_artifacts(
        output_dir=target,
        dataset=dataset,
        metadata=metadata,
        indexed_text=indexed_text,
        mode_results=mode_results,
        doc_id_field=doc_id_field,
        provenance={
            "source_run_dir": str(run_dir),
            "dataset": dataset_name,
            "split": split,
            "doc_id_field": doc_id_field,
            "resolved_index_mode": caps.retrieval_mode,
            "lancedb_version": lancedb_version,
            "fusion": str(DEFAULT_HYBRID_FUSION_POLICY),
        },
    )
