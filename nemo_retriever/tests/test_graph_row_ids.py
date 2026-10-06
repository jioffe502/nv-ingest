# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic row IDs for graph-ingested records."""

from __future__ import annotations

import pandas as pd
import pytest

from nemo_retriever.common.modality.content_transforms import explode_content_to_rows
from nemo_retriever.common.schemas.embedding import embedding_split_metadata
from nemo_retriever.common.vdb.lancedb import _create_lancedb_result
from nemo_retriever.common.vdb.records import _client_record_from_graph_row, graph_row_id

_COLUMNS = ("table", "chart", "infographic")


def _embedded(row: dict) -> dict:
    return {**row, "text_embeddings_1b_v2": {"embedding": [0.1, 0.2]}}


def _pdf_page(page: int = 3) -> dict:
    return {
        "path": "/docs/report.pdf",
        "page_number": page,
        "text": "Quarterly revenue grew.",
        "metadata": {"source_path": "/docs/report.pdf"},
        "table": [
            {"text": "a | b", "bbox_xyxy_norm": [0.1, 0.1, 0.5, 0.5], "caption": "Revenue table"},
            {"text": "a | b", "bbox_xyxy_norm": [0.1, 0.6, 0.5, 0.9]},
        ],
        "chart": [{"text": "bars", "bbox_xyxy_norm": [0.5, 0.1, 0.9, 0.5]}],
        "infographic": [{"text": "ocr words", "caption": "ocr words", "bbox_xyxy_norm": [0.2, 0.2, 0.6, 0.6]}],
    }


def _audio_row(**metadata: object) -> dict:
    return _embedded(
        {
            "path": "/tmp/retriever_audio_chunk_x/call_chunk_0001.mp3",
            "page_number": 1,
            "text": "hello from audio",
            "_content_type": "audio",
            "metadata": {"source_path": "/media/call.wav", "_content_type": "audio", "segment_start_seconds": 30.0}
            | metadata,
        }
    )


def test_row_id_derivation_is_pinned() -> None:
    """IDs are stored, so changing their derivation must be deliberate."""
    row = _embedded(
        {
            "path": "/docs/report.pdf",
            "page_number": 3,
            "text": "Quarterly revenue grew.",
            "_content_type": "table",
            "_bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
            "metadata": {"source_path": "/docs/report.pdf", "chunk_index": 1, "chunk_count": 2},
        }
    )

    assert graph_row_id(row) == "44fee34faab07301e0f8f8b049b9f86bae270c904787e415bda834bf56b462f8"


def test_records_carry_the_derived_id_unless_an_explicit_id_is_set() -> None:
    derived = _embedded({"text": "chunk", "path": "/docs/a.pdf", "page_number": 1})
    explicit = {**derived, "metadata": {"content_metadata": {"id": "row-1"}}}
    blank = {**derived, "metadata": {"id": "", "content_metadata": {"id": " "}}}

    stored = [
        _create_lancedb_result(_client_record_from_graph_row(row), expected_dim=2)[0]["id"]
        for row in (derived, explicit, blank)
    ]

    assert stored == [graph_row_id(derived), "row-1", graph_row_id(blank)]


def test_page_elements_get_unique_ids_that_ignore_batch_composition() -> None:
    alone = explode_content_to_rows(pd.DataFrame([_pdf_page()]), content_columns=_COLUMNS)
    batched = explode_content_to_rows(pd.DataFrame([_pdf_page(4), _pdf_page()]), content_columns=_COLUMNS)

    alone_ids = [graph_row_id(_embedded(row)) for row in alone.to_dict(orient="records")]
    batched_ids = [graph_row_id(_embedded(row)) for row in batched.to_dict(orient="records") if row["page_number"] == 3]

    # Page text, two tables, a table caption, a chart, and an infographic with a caption.
    assert len(set(alone_ids)) == len(alone_ids) == 7
    assert sorted(alone_ids) == sorted(batched_ids)


def test_media_rows_keep_their_id_when_a_mixed_batch_relabels_them() -> None:
    audio = _audio_row()
    reshaped = explode_content_to_rows(
        pd.DataFrame([_pdf_page(), {k: v for k, v in audio.items() if k != "text_embeddings_1b_v2"}]),
        content_columns=_COLUMNS,
    ).to_dict(orient="records")
    relabelled = next(row for row in reshaped if row["metadata"].get("_content_type") == "audio")

    assert relabelled["_content_type"] == "text"
    assert graph_row_id(_embedded(relabelled)) == graph_row_id(audio)
    assert graph_row_id(_audio_row()) == graph_row_id({**_audio_row(), "path": "/tmp/other_chunk.mp3"})


@pytest.mark.parametrize(
    "variants",
    [
        pytest.param([_audio_row(segment_index=index) for index in range(3)], id="audio-segments"),
        pytest.param(
            [
                _embedded(
                    {
                        "text": "whole parent",
                        "path": "/d/a.pdf",
                        "metadata": embedding_split_metadata(
                            content="alpha",
                            parent_id="parent",
                            chunk_id=f"child-{index}",
                            chunk_index=index,
                            chunk_count=2,
                            start_token=100 * index,
                            end_token=100 * index + 100,
                        ),
                    }
                )
                for index in range(2)
            ],
            id="embedding-splits",
        ),
        pytest.param(
            [
                _embedded({"path": "/d/a.pdf", "page_number": 2, "text": "same", "_content_type": kind})
                for kind in ("table", "chart", "infographic")
            ],
            id="full-page-fallback-kinds",
        ),
    ],
)
def test_rows_with_identical_text_get_distinct_ids(variants: list[dict]) -> None:
    assert len({graph_row_id(row) for row in variants}) == len(variants)


def test_ray_block_round_trip_keeps_ids() -> None:
    """Rows that share a Ray block gain null metadata keys and float page numbers."""
    pytest.importorskip("ray")
    from ray.data.block import BlockAccessor

    from nemo_retriever.graph.executor import arrow_table_to_pandas

    rows = [
        _audio_row(),
        _embedded({"path": "/docs/a.pdf", "page_number": None, "text": "no page", "metadata": {"has_text": True}}),
    ]
    round_tripped = arrow_table_to_pandas(BlockAccessor.batch_to_block(pd.DataFrame(rows))).to_dict(orient="records")

    assert round_tripped[0]["metadata"] != rows[0]["metadata"]
    assert [graph_row_id(row) for row in round_tripped] == [graph_row_id(row) for row in rows]
