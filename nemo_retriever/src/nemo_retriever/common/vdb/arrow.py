# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public cached-vector schema and columnar checks shared by VDB adapters."""

from collections.abc import Iterator
from typing import Final

import pyarrow as pa
import pyarrow.compute as pc

from nemo_retriever.common.vdb.adt_vdb import VDBInvalidRequest


EMBEDDING_MODEL_METADATA_KEY: Final[bytes] = b"nemo_retriever.embedding_model_name"
EMBEDDING_MODEL_REVISION_METADATA_KEY: Final[bytes] = b"nemo_retriever.embedding_model_revision"


def cached_vector_schema(
    dim: int,
    embedding_model_name: str | None = None,
    embedding_model_revision: str | None = None,
) -> pa.Schema:
    """Return the canonical schema for staging cached vectors for ``ingest_arrow``.

    ``dim`` must be a positive integer. Model identity is optional UTF-8 schema
    metadata under the public bytes constants above. Producers can add typed
    columns and user metadata; adapters retain their existing validation rules.
    """
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise VDBInvalidRequest("Cached Arrow vector dimension must be a positive integer")
    metadata = {}
    if embedding_model_name:
        metadata[EMBEDDING_MODEL_METADATA_KEY] = embedding_model_name.encode("utf-8")
    if embedding_model_revision:
        metadata[EMBEDDING_MODEL_REVISION_METADATA_KEY] = embedding_model_revision.encode("utf-8")
    return pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("metadata", pa.string()),
        ],
        metadata=metadata or None,
    )


def cached_vector_dimension(schema: pa.Schema) -> int:
    """Validate the cached row schema without materializing any rows."""
    if len(set(schema.names)) != len(schema.names):
        raise VDBInvalidRequest("Cached Arrow columns must have unique names")
    if "vector" not in schema.names:
        raise VDBInvalidRequest("Cached Arrow input is missing column 'vector'")
    vector_type = schema.field("vector").type
    if not pa.types.is_fixed_size_list(vector_type) or vector_type.list_size <= 0:
        raise VDBInvalidRequest("Cached Arrow vectors must be fixed-size lists of float32 values")
    contract = cached_vector_schema(vector_type.list_size)
    for field in contract:
        name = field.name
        if name not in schema.names:
            raise VDBInvalidRequest(f"Cached Arrow input is missing column {name!r}")
        data_type = schema.field(name).type
        if name == "vector":
            if data_type.value_type != field.type.value_type:
                raise VDBInvalidRequest("Cached Arrow vectors must be fixed-size lists of float32 values")
        elif data_type != field.type and not pa.types.is_large_string(data_type):
            raise VDBInvalidRequest(f"Cached Arrow column {name!r} must contain strings")
    return int(vector_type.list_size)


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
