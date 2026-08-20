# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

lancedb = pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.common.vdb.sink import VdbSinkPolicy
from nemo_retriever.operators.vdb import IngestVdbOperator


def _heterogeneous_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": "alpha financial table",
                "text_embeddings_1b_v2": {"embedding": [1.0, 0.0]},
                "source_id": "/tmp/report-a.pdf",
                "page_number": 2,
                "_content_type": "table",
                "_bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
                "metadata": {"content_metadata": {"type": "table", "id": "table-a"}},
                "wide_result_only": {"nested": ["not", "stored"]},
            },
            {
                "text": "beta narrative",
                "text_embeddings_1b_v2": {"embedding": [0.0, 1.0]},
                "source_id": "/tmp/report-b.pdf",
                "page_number": 7,
                "_content_type": "text",
                "metadata": {"content_metadata": {"type": "text", "id": "text-b"}},
                "wide_result_only": {"nested": ["also", "not", "stored"]},
            },
            {
                "text": "",
                "text_embeddings_1b_v2": {"embedding": [0.8, 0.2]},
                "source_id": "/tmp/report-c.pdf",
                "page_number": 1,
                "_stored_image_uri": "s3://bucket/report-c/page-1.png",
                "_content_type": "image",
                "metadata": {"content_metadata": {"type": "image", "id": "image-c"}},
                "wide_result_only": {"binary": "omitted"},
            },
        ]
    )


def _operator(uri: Path, sidecar: pd.DataFrame) -> IngestVdbOperator:
    return IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(uri),
            "table_name": "chunks",
            "vector_dim": 2,
            "overwrite": True,
            "build_index": True,
            "hybrid": True,
            "index_type": "IVF_FLAT",
            "num_partitions": 2,
            "meta_dataframe": sidecar,
            "meta_source_field": "source",
            "meta_fields": ["department", "retained_optional"],
            "meta_join_key": "source_name",
        },
    )


def _product_metadata(schema) -> dict[bytes, bytes]:
    return {key: value for key, value in (schema.metadata or {}).items() if not key.startswith(b"nemo_retriever.sink_")}


def test_bounded_sink_matches_legacy_stored_rows_indices_and_retrieval(tmp_path: Path) -> None:
    sidecar = pd.DataFrame(
        [
            {"source": "report-a.pdf", "department": "finance", "retained_optional": None},
            {"source": "report-b.pdf", "department": "legal", "retained_optional": "yes"},
            {"source": "report-c.pdf", "department": "design", "retained_optional": 3},
        ]
    )
    frame = _heterogeneous_frame()
    legacy = _operator(tmp_path / "legacy", sidecar)
    bounded = _operator(tmp_path / "bounded", sidecar)

    legacy.process(frame)
    bounded.consume_batches(
        iter([frame.iloc[:1], frame.iloc[1:]]),
        operation_id="heterogeneous-parity",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    legacy_table = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks")
    bounded_table = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks")
    assert legacy_table.schema.remove_metadata() == bounded_table.schema.remove_metadata()
    assert _product_metadata(legacy_table.schema) == _product_metadata(bounded_table.schema)
    assert set(bounded_table.schema.metadata or {}) - set(legacy_table.schema.metadata or {}) == {
        b"nemo_retriever.sink_create_operation_sha256",
        b"nemo_retriever.sink_create_request_sha256",
    }
    assert legacy_table.to_arrow().sort_by("id").to_pylist() == bounded_table.to_arrow().sort_by("id").to_pylist()

    legacy_indices = {tuple(index.columns) for index in legacy_table.list_indices()}
    bounded_indices = {tuple(index.columns) for index in bounded_table.list_indices()}
    assert legacy_indices == bounded_indices == {("vector",), ("text",)}

    vectors = [[1.0, 0.0], [0.0, 1.0]]
    query_texts = ["alpha", "beta"]
    legacy_hits = legacy._vdb.retrieval(vectors, query_texts=query_texts, hybrid=True, top_k=3)
    bounded_hits = bounded._vdb.retrieval(vectors, query_texts=query_texts, hybrid=True, top_k=3)
    assert [[hit["id"] for hit in hits] for hits in legacy_hits] == [
        [hit["id"] for hit in hits] for hits in bounded_hits
    ]


def _bad_vector_frame(vector: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": "searchable",
                "text_embeddings_1b_v2": {"embedding": vector},
                "source_id": "/tmp/bad-vector.pdf",
                "page_number": 1,
                "metadata": {"content_metadata": {"type": "text", "id": "bad-vector"}},
            }
        ]
    )


def _plain_operator(uri: Path, *, vector_dim: int | None = 2, **overrides) -> IngestVdbOperator:
    kwargs = {
        "uri": str(uri),
        "table_name": "chunks",
        "vector_dim": vector_dim,
        "overwrite": True,
        "build_index": False,
    }
    kwargs.update(overrides)
    return IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs=kwargs,
    )


def test_all_filtered_rows_create_the_same_empty_table_as_legacy(tmp_path: Path) -> None:
    frame = _bad_vector_frame([1.0])
    legacy = _plain_operator(tmp_path / "legacy")
    bounded = _plain_operator(tmp_path / "bounded")

    legacy.process(frame)
    report = bounded.consume_batches(
        iter([frame]),
        operation_id="all-filtered-create",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    legacy_table = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks")
    bounded_table = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks")
    assert legacy_table.count_rows() == bounded_table.count_rows() == 0
    assert legacy_table.schema.remove_metadata() == bounded_table.schema.remove_metadata()
    assert report.outcome == "success"
    assert report.rows_written == 0


def test_all_filtered_rows_overwrite_the_same_existing_table_as_legacy(tmp_path: Path) -> None:
    seed = _heterogeneous_frame().iloc[:1]
    bad_vector = _bad_vector_frame([1.0])
    legacy = _plain_operator(tmp_path / "legacy")
    bounded = _plain_operator(tmp_path / "bounded")
    legacy.process(seed)
    bounded.consume_batches(
        iter([seed]),
        operation_id="seed",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    legacy.process(bad_vector)
    report = bounded.consume_batches(
        iter([bad_vector]),
        operation_id="all-filtered-overwrite",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    assert lancedb.connect(str(tmp_path / "legacy")).open_table("chunks").count_rows() == 0
    assert lancedb.connect(str(tmp_path / "bounded")).open_table("chunks").count_rows() == 0
    assert report.outcome == "success"
    assert report.rows_written == 0


def test_all_filtered_rows_preserve_legacy_dimension_inference_error(tmp_path: Path) -> None:
    frame = _bad_vector_frame([])
    legacy = _plain_operator(tmp_path / "legacy", vector_dim=None)
    bounded = _plain_operator(tmp_path / "bounded", vector_dim=None)
    error = "Cannot infer LanceDB vector_dim because no non-empty embedding was produced"

    with pytest.raises(ValueError, match=error):
        legacy.process(frame)
    with pytest.raises(ValueError, match=error):
        bounded.consume_batches(
            iter([frame]),
            operation_id="all-filtered-infer",
            policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
        )


@pytest.mark.parametrize(
    ("bad_vector_policy", "expected_vectors"),
    [
        pytest.param("drop", [], id="drop"),
        pytest.param("fill", [[-3.5, -3.5]], id="fill"),
        pytest.param("null", [None], id="null"),
    ],
)
def test_validation_disabled_preserves_lancedb_bad_vector_policy(
    tmp_path: Path,
    bad_vector_policy: str,
    expected_vectors: list[list[float] | None],
) -> None:
    frame = _bad_vector_frame([1.0])
    common = {
        "validate_vector_length": False,
        "on_bad_vectors": bad_vector_policy,
        "fill_value": -3.5,
    }
    legacy = _plain_operator(tmp_path / "legacy", **common)
    bounded = _plain_operator(tmp_path / "bounded", **common)

    legacy.process(frame)
    report = bounded.consume_batches(
        iter([frame]),
        operation_id=f"validation-disabled-{bad_vector_policy}",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    legacy_vectors = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks").to_arrow()["vector"].to_pylist()
    bounded_vectors = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks").to_arrow()["vector"].to_pylist()
    assert legacy_vectors == bounded_vectors == expected_vectors
    assert report.rows_written == len(expected_vectors)


@pytest.mark.parametrize("bad_vector_policy", ["drop", "fill", "null", "error"])
def test_nullable_vector_elements_are_not_misclassified_as_bad_vectors(
    tmp_path: Path,
    bad_vector_policy: str,
) -> None:
    frame = _bad_vector_frame([1.0, None])
    common = {
        "validate_vector_length": False,
        "on_bad_vectors": bad_vector_policy,
        "fill_value": -3.5,
    }
    legacy = _plain_operator(tmp_path / "legacy", **common)
    bounded = _plain_operator(tmp_path / "bounded", **common)

    legacy.process(frame)
    report = bounded.consume_batches(
        iter([frame]),
        operation_id=f"nullable-child-{bad_vector_policy}",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    legacy_vectors = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks").to_arrow()["vector"].to_pylist()
    bounded_vectors = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks").to_arrow()["vector"].to_pylist()
    assert legacy_vectors == bounded_vectors == [[1.0, None]]
    assert report.rows_written == 1


def test_numeric_vector_values_coercible_by_lancedb_have_stored_value_parity(tmp_path: Path) -> None:
    frame = _bad_vector_frame(["1.25", "-2.5"])
    common = {"validate_vector_length": False, "on_bad_vectors": "drop"}
    legacy = _plain_operator(tmp_path / "legacy", **common)
    bounded = _plain_operator(tmp_path / "bounded", **common)

    legacy.process(frame)
    report = bounded.consume_batches(
        iter([frame]),
        operation_id="coercible-vector-values",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    legacy_vectors = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks").to_arrow()["vector"].to_pylist()
    bounded_vectors = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks").to_arrow()["vector"].to_pylist()
    assert legacy_vectors == bounded_vectors == [[1.25, -2.5]]
    assert report.rows_written == 1


def test_strict_bad_vector_error_matches_lancedb_contract(tmp_path: Path) -> None:
    frame = _bad_vector_frame([1.0])
    common = {"validate_vector_length": False, "on_bad_vectors": "error"}
    legacy = _plain_operator(tmp_path / "legacy", **common)
    bounded = _plain_operator(tmp_path / "bounded", **common)
    error = "Vector column 'vector' has variable length vectors"

    with pytest.raises(RuntimeError, match=error):
        legacy.process(frame)
    with pytest.raises(RuntimeError, match=error):
        bounded.consume_batches(
            iter([frame]),
            operation_id="strict-bad-vector",
            policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
        )

    assert "chunks" not in lancedb.connect(str(tmp_path / "bounded")).list_tables().tables


def test_nan_drop_policy_matches_lancedb_without_leaving_an_unfinished_write(tmp_path: Path) -> None:
    frame = pd.concat(
        [
            _bad_vector_frame([1.0, 0.0]),
            _bad_vector_frame([float("nan"), 1.0]),
        ],
        ignore_index=True,
    )
    frame.at[0, "metadata"] = {"content_metadata": {"type": "text", "id": "good-vector"}}
    frame.at[1, "metadata"] = {"content_metadata": {"type": "text", "id": "nan-vector"}}
    common = {"on_bad_vectors": "drop"}
    legacy = _plain_operator(tmp_path / "legacy", **common)
    bounded = _plain_operator(tmp_path / "bounded", **common)

    legacy.process(frame)
    report = bounded.consume_batches(
        iter([frame]),
        operation_id="nan-drop",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    legacy_table = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks")
    bounded_table = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks")
    assert legacy_table.to_arrow().to_pylist() == bounded_table.to_arrow().to_pylist()
    assert bounded._vdb.retrieval([[1.0, 0.0]], top_k=1)[0][0]["id"] == "good-vector"
    assert report.rows_written == 1


def test_dimension_inference_scans_past_an_empty_vector_like_legacy(tmp_path: Path) -> None:
    frame = pd.concat(
        [
            _bad_vector_frame([]),
            _bad_vector_frame([1.0, 0.0, 2.0]),
        ],
        ignore_index=True,
    )
    frame.at[0, "metadata"] = {"content_metadata": {"type": "text", "id": "empty-vector"}}
    frame.at[1, "metadata"] = {"content_metadata": {"type": "text", "id": "good-vector"}}
    legacy = _plain_operator(tmp_path / "legacy", vector_dim=None)
    bounded = _plain_operator(tmp_path / "bounded", vector_dim=None)

    legacy.process(frame)
    report = bounded.consume_batches(
        iter([frame]),
        operation_id="infer-after-empty",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    legacy_table = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks")
    bounded_table = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks")
    assert bounded_table.schema.field("vector").type.list_size == 3
    assert legacy_table.to_arrow().to_pylist() == bounded_table.to_arrow().to_pylist()
    assert report.rows_written == 1


def _client_record(row_id: str, vector) -> dict:
    return {
        "document_type": "text",
        "metadata": {
            "embedding": vector,
            "content": row_id,
            "content_metadata": {"type": "text", "id": row_id},
            "source_metadata": {"source_id": f"/tmp/{row_id}.pdf"},
        },
    }


@pytest.mark.parametrize("bad_vector_policy", ["drop", "fill", "null", "error"])
def test_validation_disabled_accepts_exact_length_numpy_vectors_like_legacy(
    tmp_path: Path,
    bad_vector_policy: str,
) -> None:
    records = [[_client_record("numpy-vector", np.array([1.0, 2.0], dtype=np.float32))]]
    common = {
        "validate_vector_length": False,
        "on_bad_vectors": bad_vector_policy,
        "fill_value": -3.5,
    }
    legacy = _plain_operator(tmp_path / "legacy", **common)
    bounded = _plain_operator(tmp_path / "bounded", **common)

    legacy.process(records)
    report = bounded.consume_batches(
        iter([records]),
        operation_id=f"numpy-vector-{bad_vector_policy}",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    legacy_vectors = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks").to_arrow()["vector"].to_pylist()
    bounded_vectors = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks").to_arrow()["vector"].to_pylist()
    assert legacy_vectors == bounded_vectors == [[1.0, 2.0]]
    assert report.rows_written == 1


def test_dimension_inference_replays_a_matching_tuple_prefix_like_legacy(tmp_path: Path) -> None:
    records = [[_client_record("tuple-prefix", (1.0, 0.0, 2.0)), _client_record("list", [0.0, 1.0, 2.0])]]
    legacy = _plain_operator(tmp_path / "legacy", vector_dim=None)
    bounded = _plain_operator(tmp_path / "bounded", vector_dim=None)

    legacy.process(records)
    report = bounded.consume_batches(
        iter([records]),
        operation_id="infer-after-tuple",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    legacy_rows = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks").to_arrow().sort_by("id").to_pylist()
    bounded_rows = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks").to_arrow().sort_by("id").to_pylist()
    assert legacy_rows == bounded_rows
    assert [row["id"] for row in bounded_rows] == ["list", "tuple-prefix"]
    assert report.rows_written == 2


def test_dimension_inference_replays_prefix_through_fill_policy_like_legacy(tmp_path: Path) -> None:
    records = [
        [
            _client_record("empty-prefix", []),
            _client_record("tuple-prefix", (1.0, 0.0, 2.0)),
            _client_record("list", [0.0, 1.0, 2.0]),
        ]
    ]
    common = {
        "validate_vector_length": False,
        "on_bad_vectors": "fill",
        "fill_value": -3.5,
    }
    legacy = _plain_operator(tmp_path / "legacy", vector_dim=None, **common)
    bounded = _plain_operator(tmp_path / "bounded", vector_dim=None, **common)

    legacy.process(records)
    report = bounded.consume_batches(
        iter([records]),
        operation_id="infer-prefix-fill",
        policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
    )

    legacy_rows = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks").to_arrow().sort_by("id").to_pylist()
    bounded_rows = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks").to_arrow().sort_by("id").to_pylist()
    assert legacy_rows == bounded_rows
    assert {row["id"]: row["vector"] for row in bounded_rows} == {
        "empty-prefix": [-3.5, -3.5, -3.5],
        "list": [0.0, 1.0, 2.0],
        "tuple-prefix": [1.0, 0.0, 2.0],
    }
    assert report.rows_written == 3


def test_dimension_inference_replays_prefix_through_error_policy_like_legacy(tmp_path: Path) -> None:
    records = [[_client_record("empty-prefix", []), _client_record("list", [0.0, 1.0, 2.0])]]
    common = {"validate_vector_length": False, "on_bad_vectors": "error"}
    legacy = _plain_operator(tmp_path / "legacy", vector_dim=None, **common)
    bounded = _plain_operator(tmp_path / "bounded", vector_dim=None, **common)
    error = "Vector column 'vector' has variable length vectors"

    with pytest.raises(RuntimeError, match=error):
        legacy.process(records)
    with pytest.raises(RuntimeError, match=error):
        bounded.consume_batches(
            iter([records]),
            operation_id="infer-prefix-error",
            policy=VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1),
        )
