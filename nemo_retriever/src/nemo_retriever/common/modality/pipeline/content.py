# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared content row transforms for ingestion pipelines.

Canonical implementations live in
:mod:`nemo_retriever.common.modality.content_transforms`. This module re-exports
them so older pipeline imports keep a single Arrow-safe behavior.
"""

from nemo_retriever.common.modality.content_transforms import (
    _CONTENT_COLUMNS,
    collapse_content_to_page_rows,
    explode_content_to_rows,
)

__all__ = [
    "_CONTENT_COLUMNS",
    "collapse_content_to_page_rows",
    "explode_content_to_rows",
]
