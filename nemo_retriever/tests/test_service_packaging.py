# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import tomllib
from packaging.requirements import Requirement

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _extra_requirements(extra: str) -> list[Requirement]:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["optional-dependencies"][extra]
    return [Requirement(dep) for dep in deps]


def test_service_extra_includes_litellm_for_answer_generation() -> None:
    requirements = _extra_requirements("service")

    litellm = next((req for req in requirements if req.name == "litellm"), None)

    assert litellm is not None
    assert any(str(spec).startswith(">=") and "1.95.0rc3" in str(spec) for spec in litellm.specifier)


def test_core_dependencies_include_tokenizer_stack() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [Requirement(dep) for dep in pyproject["project"]["dependencies"]]
    names = {req.name for req in requirements}

    assert "tokenizers" in names
    assert "huggingface-hub" in names


def test_service_images_precache_all_embedding_input_admission_assets() -> None:
    dockerfile = (PROJECT_ROOT.parent / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.count("resolve_embedding_input_policy(DEFAULT_TOKENIZER_MODEL_ID") == 2
    assert dockerfile.count("configured_max_tokens=8192, input_type='passage'") == 2
