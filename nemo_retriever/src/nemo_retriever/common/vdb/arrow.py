# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Columnar input checks shared by cached-vector VDB adapters."""

from collections.abc import Iterator

import pyarrow as pa
import pyarrow.compute as pc

from nemo_retriever.common.vdb.adt_vdb import VDBInvalidRequest


def cached_vector_dimension(schema: pa.Schema) -> int:
    """Validate the cached row schema without materializing any rows."""
    if len(set(schema.names)) != len(schema.names):
        raise VDBInvalidRequest("Cached Arrow columns must have unique names")
    for name in ("vector", "id", "text", "source", "metadata"):
        if name not in schema.names:
            raise VDBInvalidRequest(f"Cached Arrow input is missing column {name!r}")
        data_type = schema.field(name).type
        if name == "vector":
            if not (
                pa.types.is_fixed_size_list(data_type)
                and pa.types.is_float32(data_type.value_type)
                and data_type.list_size > 0
            ):
                raise VDBInvalidRequest("Cached Arrow vectors must be fixed-size lists of float32 values")
        elif not (pa.types.is_string(data_type) or pa.types.is_large_string(data_type)):
            raise VDBInvalidRequest(f"Cached Arrow column {name!r} must contain strings")
    return int(schema.field("vector").type.list_size)


def checked_cached_batches(
    reader: pa.RecordBatchReader,
    *,
    max_batch_bytes: int,
    expected_rows: int | None,
) -> Iterator[pa.RecordBatch]:
    """Check a bounded reader using Arrow kernels, without Python vector lists.

    The producer must bound its retained input buffers. A slice of a large
    table can retain the entire parent allocation, so check retained bytes,
    rather than only the logical size of each slice. An invalid later batch
    raises while the native writer is still consuming its data transaction.
    """
    rows = 0
    for batch in reader:
        if not batch.schema.equals(reader.schema, check_metadata=True):
            raise VDBInvalidRequest("Cached Arrow batch schema changed during loading")
        if not batch.num_rows:
            continue
        retained_bytes = int(batch.get_total_buffer_size())
        if retained_bytes > max_batch_bytes:
            raise VDBInvalidRequest(
                f"Cached Arrow batch retains {retained_bytes} bytes, exceeding stream_batch_bytes={max_batch_bytes}; "
                "read smaller source batches instead of slicing a materialized table"
            )
        vector = batch.column(batch.schema.get_field_index("vector"))
        values = vector.flatten()
        if vector.null_count or values.null_count or not pc.all(pc.is_finite(values)).as_py():
            raise VDBInvalidRequest("Cached Arrow vectors must contain finite, non-null float32 values")
        rows += batch.num_rows
        if expected_rows is not None and rows > expected_rows:
            raise VDBInvalidRequest(f"Cached Arrow row count exceeds expected_rows={expected_rows}")
        yield batch
    if expected_rows is not None and rows != expected_rows:
        raise VDBInvalidRequest(f"Cached Arrow row count differs: expected {expected_rows}, got {rows}")
