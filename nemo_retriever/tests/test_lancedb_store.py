# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from nemo_retriever.vector_store.lancedb_store import _build_lancedb_rows_from_df


def test_build_lancedb_rows_from_df_serializes_metadata_as_json_with_detection_counts() -> None:
    rows = [
        {
            "metadata": {
                "embedding": [0.1, 0.2],
                "has_text": True,
                "source_path": "/docs/sample.pdf",
            },
            "path": "/docs/sample.pdf",
            "page_number": 3,
            "text": "sample text",
            "page_elements_v3_num_detections": 5,
            "page_elements_v3_counts_by_label": {"text": 3, "table": 2},
            "table": [{}, {}],
            "chart": [{}],
            "infographic": [],
        }
    ]

    out = _build_lancedb_rows_from_df(rows)
    assert len(out) == 1

    payload = out[0]
    metadata = json.loads(payload["metadata"])
    assert payload["source"] == "/docs/sample.pdf"
    assert metadata["has_text"] is True
    assert metadata["source_path"] == "/docs/sample.pdf"
    assert metadata["page_elements_v3_num_detections"] == 5
    assert metadata["page_elements_v3_counts_by_label"] == {"text": 3, "table": 2}
    assert metadata["ocr_table_detections"] == 2
    assert metadata["ocr_chart_detections"] == 1
    assert metadata["ocr_infographic_detections"] == 0
