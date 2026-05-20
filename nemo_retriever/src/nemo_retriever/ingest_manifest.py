# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest planning for input-aware retriever ingest extraction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from nemo_retriever.utils.input_files import _is_explicit_glob_path, input_type_for_path


_BRANCH_ORDER = ("pdf", "image", "txt", "html", "audio", "video")
_INPUT_TYPE_TO_BRANCH_FAMILY = {
    "pdf": "pdf",
    "doc": "pdf",
    "image": "image",
    "txt": "txt",
    "html": "html",
    "audio": "audio",
    "video": "video",
}
_BRANCH_FAMILY_TO_EXTRACTION_MODE = {
    "pdf": "pdf",
    "image": "image",
    "txt": "text",
    "html": "html",
    "audio": "audio",
    # Video is represented by dedicated video params in build_graph(); the
    # extraction mode stays "auto" for compatibility with the existing video
    # pipeline construction path.
    "video": "auto",
}


@dataclass(frozen=True)
class ManifestEntry:
    """One concrete or optional ingest input in a manifest."""

    path: str
    input_type: str | None
    is_explicit_glob: bool = False


@dataclass(frozen=True)
class InputManifest:
    """Classified input files for planner-backed ingest."""

    entries: tuple[ManifestEntry, ...]
    unsupported_files: tuple[str, ...]

    @property
    def files_by_family(self) -> dict[str, tuple[str, ...]]:
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for entry in self.entries:
            if entry.input_type is None:
                continue
            family = _INPUT_TYPE_TO_BRANCH_FAMILY[entry.input_type]
            grouped[family].append(entry.path)
        return {family: tuple(paths) for family, paths in grouped.items()}

    @property
    def optional_globs(self) -> tuple[str, ...]:
        return tuple(entry.path for entry in self.entries if entry.is_explicit_glob)


@dataclass(frozen=True)
class ExtractionBranchPlan:
    """A single typed extraction branch to execute before common stages."""

    family: str
    extraction_mode: str
    input_paths: tuple[str, ...]


def build_input_manifest(input_paths: Iterable[str]) -> InputManifest:
    """Classify concrete input paths without loading modality dependencies."""

    entries: list[ManifestEntry] = []
    unsupported: list[str] = []
    for path in input_paths:
        is_glob = _is_explicit_glob_path(path)
        input_type = None if is_glob else input_type_for_path(path)
        entries.append(ManifestEntry(path=path, input_type=input_type, is_explicit_glob=is_glob))
        if input_type is None and not is_glob:
            unsupported.append(path)
    return InputManifest(entries=tuple(entries), unsupported_files=tuple(unsupported))


def plan_extraction_branches(manifest: InputManifest) -> tuple[ExtractionBranchPlan, ...]:
    """Emit deterministic extraction branches for a validated manifest."""

    if manifest.unsupported_files:
        examples = ", ".join(manifest.unsupported_files[:3])
        raise ValueError(f"Unsupported input file type(s) for default GraphIngestor.extract(): {examples}")

    files_by_family = manifest.files_by_family
    if not files_by_family:
        # Empty optional globs should preserve the old empty-input behavior
        # without inventing modality branches that require extra dependencies.
        return (
            ExtractionBranchPlan(
                family="pdf",
                extraction_mode="pdf",
                input_paths=manifest.optional_globs,
            ),
        )

    branches: list[ExtractionBranchPlan] = []
    for family in _BRANCH_ORDER:
        paths = files_by_family.get(family)
        if not paths:
            continue
        branches.append(
            ExtractionBranchPlan(
                family=family,
                extraction_mode=_BRANCH_FAMILY_TO_EXTRACTION_MODE[family],
                input_paths=paths,
            )
        )
    return tuple(branches)
