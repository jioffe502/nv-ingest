# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pyarrow as pa
from nemo_retriever.common.vdb.sink import _record_canonical_batch, _StreamStats

_SCHEMA = pa.schema(
    [
        pa.field("vector", pa.list_(pa.float32(), 2)),
        pa.field("text", pa.string()),
        pa.field("metadata", pa.string()),
        pa.field("source", pa.string()),
        pa.field("id", pa.string()),
        pa.field("page_number", pa.int32()),
    ]
)


def _batch(rows: list[dict[str, object]]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist(rows, schema=_SCHEMA)


def _digest(*batches: pa.RecordBatch) -> str:
    stats = _StreamStats()
    for batch in batches:
        _record_canonical_batch(stats, batch)
    return stats.digest


def test_canonical_digest_is_independent_of_batching_and_partition_order() -> None:
    rows = [
        {
            "vector": [1.0, 2.0],
            "text": "alpha",
            "metadata": "{}",
            "source": "a",
            "id": "a",
            "page_number": 1,
        },
        {
            "vector": [3.0, 4.0],
            "text": "beta",
            "metadata": None,
            "source": "b",
            "id": "b",
            "page_number": None,
        },
        {
            "vector": [5.0, 6.0],
            "text": "gamma",
            "metadata": "{}",
            "source": "c",
            "id": "c",
            "page_number": 3,
        },
    ]
    whole = _batch(rows)

    assert _digest(whole) == _digest(whole.slice(0, 1), whole.slice(1, 2))
    assert _digest(_batch([rows[2]]), _batch(rows[:2])) == _digest(whole)


def test_canonical_digest_uses_arrow_normalized_values() -> None:
    # These Python floats differ but normalize to the same float32 value in
    # the canonical Arrow schema. Retry identity follows stored content.
    base = {"text": "alpha", "metadata": "{}", "source": "a", "id": "a", "page_number": 1}
    rounded = _batch([{**base, "vector": [1.00000001, 2.0]}])
    exact = _batch([{**base, "vector": [1.0, 2.0]}])
    changed = _batch([{**base, "vector": [1.25, 2.0]}])

    assert _digest(rounded) == _digest(exact)
    assert _digest(changed) != _digest(exact)


def test_canonical_digest_does_not_reconstruct_or_serialize_rows() -> None:
    batch = _batch(
        [
            {
                "vector": [1.0, 2.0],
                "text": "alpha",
                "metadata": "{}",
                "source": "a",
                "id": "a",
                "page_number": 1,
            },
            {
                "vector": [3.0, 4.0],
                "text": "beta",
                "metadata": "{}",
                "source": "b",
                "id": "b",
                "page_number": 2,
            },
        ]
    )

    class ExistingCanonicalBatch:
        schema = batch.schema
        num_rows = batch.num_rows
        columns = batch.columns

        def slice(self, *args, **kwargs):
            raise AssertionError("digest must not reconstruct or serialize individual rows")

    # The digest hot path already has canonical Arrow columns. It should scan
    # those columns in place, without slicing out and rebuilding every row.
    assert _digest(ExistingCanonicalBatch())
