# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for vector store backends.

Backends receive rows in the canonical VDB record format produced by
:func:`nemo_retriever.vector_store.vdb_records.build_vdb_records`.
Write-path only; retrieval support will be added with the second backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class VectorStore(ABC):
    """Abstract base for vector store backends."""

    @abstractmethod
    def create_table(self, *, dim: int, **kwargs: Any) -> None:
        """Create or reset the storage table / index."""

    @abstractmethod
    def write_rows(self, rows: Sequence[dict[str, Any]], **kwargs: Any) -> None:
        """Write a batch of canonical VDB records."""

    @abstractmethod
    def create_index(self, **kwargs: Any) -> None:
        """Build search indices after all writes complete."""
