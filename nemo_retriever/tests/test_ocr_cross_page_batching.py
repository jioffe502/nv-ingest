# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for OCR crop batching across page rows."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from nemo_retriever.common.params import BatchTuningParams, ExtractParams, RemoteRetryParams
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.operators.extract.ocr.cpu_ocr import OCRCPUActor
from nemo_retriever.operators.extract.ocr.gpu_ocr import OCRActor
from nemo_retriever.operators.extract.ocr.ocr import OCRActor as OCRArchetype
from nemo_retriever.models.nim.nim import NIMClient
from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractCPUActor


def _page_png_b64(crop_id: int) -> str:
    image = Image.new("RGB", (16, 16), color=(crop_id, crop_id, crop_id))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _detection(detection_id: str, label_name: str, bbox: list[float]) -> dict[str, Any]:
    return {
        "detection_id": detection_id,
        "label_name": label_name,
        "bbox_xyxy_norm": bbox,
    }


def _page(page_id: str, crop_id: int, detections: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "page_id": page_id,
        "metadata": {},
        "page_image": {"image_b64": _page_png_b64(crop_id)},
        "page_elements_v3": {"detections": detections},
    }


def _ocr_prediction(crop_id: int) -> list[dict[str, Any]]:
    return [
        {
            "left": 0.0,
            "right": 1.0,
            "upper": 0.0,
            "lower": 1.0,
            "text": f"crop-{crop_id}",
        }
    ]


class _RecordingListModel:
    """Record the list-input contract used by the persistent local model."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, crops: Any, *, merge_level: str) -> list[list[dict[str, Any]]]:
        assert isinstance(crops, list)
        crop_ids = [int(crop[0, 0, 0]) for crop in crops]
        self.calls.append(
            {
                "crop_count": len(crops),
                "crop_ids": crop_ids,
                "merge_level": merge_level,
            }
        )
        return [_ocr_prediction(crop_id) for crop_id in crop_ids]


class _FallbackIsolationModel:
    """Force batched failure, then fail one crop during per-item isolation."""

    def __init__(self, failing_crop_id: int) -> None:
        self.failing_crop_id = failing_crop_id
        self.calls: list[dict[str, Any]] = []

    def invoke(self, crops: Any, *, merge_level: str) -> Any:
        is_batch = isinstance(crops, list)
        crop_arrays = crops if is_batch else [crops]
        crop_ids = [int(crop[0, 0, 0]) for crop in crop_arrays]
        self.calls.append(
            {
                "input_kind": "batch" if is_batch else "single",
                "crop_ids": crop_ids,
                "merge_level": merge_level,
            }
        )
        if is_batch:
            raise RuntimeError("force per-crop fallback")
        if crop_ids[0] == self.failing_crop_id:
            raise ValueError(f"bad crop {self.failing_crop_id}")
        return _ocr_prediction(crop_ids[0])


class _WrongCountFallbackModel:
    """Return too few batched results, then succeed during per-item fallback."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, crops: Any, *, merge_level: str) -> Any:
        is_batch = isinstance(crops, list)
        crop_arrays = crops if is_batch else [crops]
        crop_ids = [int(crop[0, 0, 0]) for crop in crop_arrays]
        self.calls.append(
            {
                "input_kind": "batch" if is_batch else "single",
                "crop_ids": crop_ids,
                "merge_level": merge_level,
            }
        )
        if is_batch:
            return [_ocr_prediction(crop_ids[0])]
        return _ocr_prediction(crop_ids[0])


class _RecordingNIMClient:
    """Record the cross-page list submitted to the remote OCR client."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke_image_inference_batches(self, **kwargs: Any) -> list[dict[str, Any]]:
        crop_ids = []
        for encoded in kwargs["image_b64_list"]:
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                crop_ids.append(image.convert("RGB").getpixel((0, 0))[0])
        self.calls.append(
            {
                "crop_ids": crop_ids,
                "max_batch_size": kwargs["max_batch_size"],
            }
        )
        return [{"text_detections": _ocr_prediction(crop_id)} for crop_id in crop_ids]


class _FallbackIsolationNIMClient(_RecordingNIMClient):
    """Reject the combined call, then fail only one page during fallback."""

    def __init__(self, failing_crop_id: int) -> None:
        super().__init__()
        self.failing_crop_id = failing_crop_id

    def invoke_image_inference_batches(self, **kwargs: Any) -> list[dict[str, Any]]:
        response = super().invoke_image_inference_batches(**kwargs)
        crop_ids = self.calls[-1]["crop_ids"]
        if len(crop_ids) > 1:
            raise RuntimeError("force per-page fallback")
        if crop_ids[0] == self.failing_crop_id:
            raise ValueError(f"bad crop {self.failing_crop_id}")
        return response


def _local_actor(
    model: Any,
    *,
    inference_batch_size: int,
    extract_text: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
) -> OCRActor:
    actor = object.__new__(OCRActor)
    actor._graph_init_kwargs = {}
    actor.ocr_kwargs = {
        "extract_text": extract_text,
        "extract_tables": extract_tables,
        "extract_charts": extract_charts,
        "extract_infographics": extract_infographics,
        "use_table_structure": False,
        "request_timeout_s": 120.0,
        "inference_batch_size": inference_batch_size,
    }
    actor._remote_retry = RemoteRetryParams()
    actor._model = model
    actor._nim_client = None
    return actor


def _remote_actor(
    client: Any,
    *,
    inference_batch_size: int,
    extract_charts: bool = True,
) -> OCRCPUActor:
    actor = object.__new__(OCRCPUActor)
    actor._graph_init_kwargs = {}
    actor.ocr_kwargs = {
        "invoke_url": "http://ocr.example/v1/ocr",
        "extract_text": False,
        "extract_tables": False,
        "extract_charts": extract_charts,
        "extract_infographics": False,
        "use_table_structure": False,
        "request_timeout_s": 120.0,
        "inference_batch_size": inference_batch_size,
    }
    actor._remote_retry = RemoteRetryParams(remote_max_retries=1, remote_max_429_retries=1)
    actor._model = None
    actor._nim_client = client
    return actor


def test_local_actor_batches_compatible_crops_across_page_rows() -> None:
    model = _RecordingListModel()
    actor = _local_actor(model, inference_batch_size=2, extract_charts=True)
    chart_bboxes = [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]
    batch = pd.DataFrame(
        [
            _page("page-A", 11, [_detection("chart-A", "chart", chart_bboxes[0])]),
            _page("page-B", 22, [_detection("chart-B", "chart", chart_bboxes[1])]),
        ]
    )

    result = actor(batch)

    assert {
        "model_calls": model.calls,
        "rows": [{"page_id": row.page_id, "chart": row.chart} for row in result.itertuples(index=False)],
    } == {
        "model_calls": [
            {
                "crop_count": 2,
                "crop_ids": [11, 22],
                "merge_level": "paragraph",
            }
        ],
        "rows": [
            {
                "page_id": "page-A",
                "chart": [{"bbox_xyxy_norm": chart_bboxes[0], "text": "crop-11"}],
            },
            {
                "page_id": "page-B",
                "chart": [{"bbox_xyxy_norm": chart_bboxes[1], "text": "crop-22"}],
            },
        ],
    }


def test_remote_actor_batches_crops_across_page_rows() -> None:
    client = _RecordingNIMClient()
    actor = _remote_actor(client, inference_batch_size=1)
    bbox = [0.0, 0.0, 1.0, 1.0]
    batch = pd.DataFrame(
        [
            _page("page-A", 11, [_detection("chart-A", "chart", bbox)]),
            _page("page-B", 22, [_detection("chart-B", "chart", bbox)]),
        ]
    )

    result = actor(batch)

    assert client.calls == [{"crop_ids": [11, 22], "max_batch_size": 1}]
    assert [row.page_id for row in result.itertuples(index=False)] == ["page-A", "page-B"]
    assert [row.chart[0]["text"] for row in result.itertuples(index=False)] == ["crop-11", "crop-22"]
    assert [row.ocr["error"] for row in result.itertuples(index=False)] == [None, None]


def test_remote_actor_falls_back_per_page_and_isolates_errors() -> None:
    client = _FallbackIsolationNIMClient(failing_crop_id=22)
    actor = _remote_actor(client, inference_batch_size=2)
    bbox = [0.0, 0.0, 1.0, 1.0]
    batch = pd.DataFrame(
        [
            _page("page-A", 11, [_detection("chart-A", "chart", bbox)]),
            _page("page-B", 22, [_detection("chart-B", "chart", bbox)]),
            _page("page-C", 33, [_detection("chart-C", "chart", bbox)]),
        ]
    )

    result = actor(batch)

    assert [call["crop_ids"] for call in client.calls] == [[11, 22, 33], [11], [22], [33]]
    assert [len(chart) for chart in result["chart"]] == [1, 0, 1]
    assert [metadata["error"] is None for metadata in result["ocr"]] == [True, False, True]
    assert result.at[1, "ocr"]["error"]["type"] == "ValueError"


@pytest.mark.parametrize(
    ("inference_batch_size", "ocr_inference_batch_size", "expected"),
    [(3, None, 3), (3, 5, 5)],
)
def test_pdf_graph_forwards_effective_remote_ocr_batch_size(
    inference_batch_size: int,
    ocr_inference_batch_size: int | None,
    expected: int,
) -> None:
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
            ocr_invoke_url="http://ocr.example/v1/ocr",
            inference_batch_size=inference_batch_size,
            batch_tuning=BatchTuningParams(ocr_inference_batch_size=ocr_inference_batch_size),
        ),
    )
    node = graph.roots[0]
    while node.operator_class is not OCRArchetype:
        node = node.children[0]

    assert node.operator_kwargs["inference_batch_size"] == expected


@pytest.mark.parametrize("ocr_batch_size,expected", [(None, 3), (5, 5)])
def test_multi_type_forwards_ocr_batch_and_retry_settings(monkeypatch, ocr_batch_size, expected):
    operator = MultiTypeExtractCPUActor(
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=False,
            use_table_structure=False,
            inference_batch_size=3,
            batch_tuning=BatchTuningParams(ocr_inference_batch_size=ocr_batch_size, page_elements_batch_size=7),
            remote_retry=RemoteRetryParams(remote_max_pool_workers=2, remote_max_retries=3, remote_max_429_retries=4),
        )
    )
    calls = []

    def instantiate(cls, **kwargs):
        calls.append((cls, kwargs))
        return type("PassThrough", (), {"run": lambda self, data: data})()

    monkeypatch.setattr(operator, "_instantiate_resolved", instantiate)
    operator._run_detection_pipeline(pd.DataFrame())
    assert calls[-1][0] is OCRArchetype
    assert calls[-1][1]["inference_batch_size"] == expected
    for key, value in {"remote_max_pool_workers": 2, "remote_max_retries": 3, "remote_max_429_retries": 4}.items():
        assert calls[-1][1][key] == value


def test_remote_actor_bounds_pending_crops_and_preserves_empty_pages():
    client = _RecordingNIMClient()
    actor = _remote_actor(client, inference_batch_size=2)
    actor._remote_retry = RemoteRetryParams(remote_max_pool_workers=2)
    bbox = [0.0, 0.0, 1.0, 1.0]
    pages = [_page(str(i), i, [_detection(str(i), "chart", bbox)] * 3) for i in range(1, 5)]
    pages.insert(2, _page("empty", 99, []))
    result = actor(pd.DataFrame(pages))
    assert [len(call["crop_ids"]) for call in client.calls] == [4, 4, 4]
    assert list(result.page_id) == ["1", "2", "empty", "3", "4"]
    assert [len(items) for items in result.chart] == [3, 3, 0, 3, 3]
    assert all(item["error"] is None for item in result.ocr)
    assert [call["crop_ids"] for call in client.calls] == [[1, 1, 1, 2], [2, 2, 3, 3], [3, 4, 4, 4]]


def test_remote_wrong_count_falls_back_without_losing_or_duplicating_regions():
    class WrongCountClient(_RecordingNIMClient):
        def invoke_image_inference_batches(self, **kwargs):
            response = super().invoke_image_inference_batches(**kwargs)
            return response[:1] if len(response) > 2 else response

    client = WrongCountClient()
    actor = _remote_actor(client, inference_batch_size=2)
    actor.ocr_kwargs.update(extract_tables=True, extract_infographics=True)
    bbox = [0.0, 0.0, 1.0, 1.0]
    pages = pd.DataFrame(
        [
            _page("a", 11, [_detection("a1", "table", bbox), _detection("a2", "chart", bbox)]),
            _page("b", 22, [_detection("b1", "infographic", bbox)]),
        ]
    )
    result = actor(pages)
    assert [call["crop_ids"] for call in client.calls] == [[11, 11, 22], [11, 11], [22]]
    assert [len(items) for items in result.table] == [1, 0]
    assert [len(items) for items in result.chart] == [1, 0]
    assert [len(items) for items in result.infographic] == [0, 1]
    assert all(item["error"] is None for item in result.ocr)


@pytest.fixture
def remote_ocr_http_server():
    """Exercise the actual HTTP client, including auth, ordering, and pool bounds."""
    state = {"active": 0, "peak": 0, "calls": [], "fail": False}
    lock = threading.Lock()
    both_started = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ids = []
            for item in payload["input"]:
                with Image.open(io.BytesIO(base64.b64decode(item["url"].split(",", 1)[1]))) as image:
                    ids.append(image.getpixel((0, 0))[0])
            with lock:
                state["calls"].append((ids, self.headers.get("Authorization") == "Bearer test-only"))
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                if state["active"] == 2:
                    both_started.set()
            both_started.wait(timeout=5)
            time.sleep(0.08 if ids[0] == 11 else 0.01)
            failed = state["fail"] and ids[0] == 22
            body = json.dumps({"data": [{"text_detections": _ocr_prediction(i)} for i in ids]}).encode()
            with lock:
                state["active"] -= 1
            self.send_response(400 if failed else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/ocr", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_remote_http_overlap_auth_chunking_and_output_order(remote_ocr_http_server, batch_size):
    url, state = remote_ocr_http_server
    client = NIMClient(max_pool_workers=2)
    try:
        actor = _remote_actor(client, inference_batch_size=batch_size)
        actor.ocr_kwargs.update(invoke_url=url, api_key="test-only")
        actor._remote_retry = RemoteRetryParams(remote_max_pool_workers=2, remote_max_retries=1)
        bbox = [0.0, 0.0, 1.0, 1.0]
        pages = [_page(str(i), i, [_detection(str(i), "chart", bbox)]) for i in (11, 22, 33, 44)]
        result = actor(pd.DataFrame(pages))
    finally:
        client.shutdown()
    assert state["peak"] == 2
    assert len(state["calls"]) == 4 // batch_size
    assert all(len(ids) == batch_size and authorized for ids, authorized in state["calls"])
    assert [items[0]["text"] for items in result.chart] == [f"crop-{i}" for i in (11, 22, 33, 44)]
    assert all(item["error"] is None for item in result.ocr)


def test_remote_http_failure_drains_running_requests_before_return(remote_ocr_http_server):
    url, state = remote_ocr_http_server
    state["fail"] = True
    client = NIMClient(max_pool_workers=2)
    try:
        with pytest.raises(Exception, match="HTTP 400"):
            client.invoke_image_inference_batches(
                invoke_url=url,
                image_b64_list=[_page_png_b64(11), _page_png_b64(22)],
                max_batch_size=1,
                max_retries=1,
            )
        assert state["active"] == 0
    finally:
        client.shutdown()


def test_local_actor_separates_merge_levels_chunks_and_preserves_detection_order() -> None:
    model = _RecordingListModel()
    actor = _local_actor(
        model,
        inference_batch_size=2,
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )
    table_a = [0.0, 0.0, 0.4, 0.4]
    chart_a = [0.5, 0.0, 1.0, 0.4]
    title_a = [0.0, 0.5, 1.0, 0.7]
    table_b1 = [0.0, 0.0, 0.3, 0.3]
    table_b2 = [0.35, 0.0, 0.65, 0.3]
    infographic_b = [0.0, 0.5, 1.0, 1.0]
    page_a = _page(
        "page-A",
        11,
        [
            _detection("table-A", "table", table_a),
            _detection("chart-A", "chart", chart_a),
            _detection("title-A", "title", title_a),
        ],
    )
    page_a["metadata"] = {"needs_ocr_for_text": True}
    page_a["text"] = "native-A"
    page_b = _page(
        "page-B",
        22,
        [
            _detection("table-B1", "table", table_b1),
            _detection("table-B2", "table", table_b2),
            _detection("infographic-B", "infographic", infographic_b),
        ],
    )
    page_b["text"] = "native-B"
    empty_page = _page("page-empty", 33, [])
    empty_page["text"] = "native-empty"
    batch = pd.DataFrame(
        [
            page_a,
            page_b,
            empty_page,
            {
                "page_id": "page-malformed",
                "metadata": {"error": {"stage": "page_render", "message": "missing raster"}},
                "page_image": None,
                "page_elements_v3": {"detections": [_detection("chart-malformed", "chart", [0.0, 0.0, 1.0, 1.0])]},
                "text": "native-malformed",
            },
        ]
    )

    result = actor(batch)

    assert {
        "model_calls": model.calls,
        "page_A": {
            "table_bboxes": [entry["bbox_xyxy_norm"] for entry in result.at[0, "table"]],
            "chart": result.at[0, "chart"],
            "text": result.at[0, "text"],
            "metadata": {
                "num_detections": result.at[0, "ocr"]["num_detections"],
                "counts_by_label": result.at[0, "ocr"]["counts_by_label"],
            },
        },
        "page_B": {
            "table_bboxes": [entry["bbox_xyxy_norm"] for entry in result.at[1, "table"]],
            "infographic": result.at[1, "infographic"],
            "text": result.at[1, "text"],
            "metadata": {
                "num_detections": result.at[1, "ocr"]["num_detections"],
                "counts_by_label": result.at[1, "ocr"]["counts_by_label"],
            },
        },
        "empty_page": {
            "table": result.at[2, "table"],
            "chart": result.at[2, "chart"],
            "infographic": result.at[2, "infographic"],
            "text": result.at[2, "text"],
            "num_detections": result.at[2, "ocr"]["num_detections"],
        },
        "malformed_page": {
            "table": result.at[3, "table"],
            "chart": result.at[3, "chart"],
            "infographic": result.at[3, "infographic"],
            "text": result.at[3, "text"],
            "error": result.at[3, "ocr"]["error"],
            "num_detections": result.at[3, "ocr"]["num_detections"],
        },
    } == {
        "model_calls": [
            {"crop_count": 2, "crop_ids": [11, 22], "merge_level": "word"},
            {"crop_count": 1, "crop_ids": [22], "merge_level": "word"},
            {"crop_count": 2, "crop_ids": [11, 11], "merge_level": "paragraph"},
            {"crop_count": 1, "crop_ids": [22], "merge_level": "paragraph"},
        ],
        "page_A": {
            "table_bboxes": [table_a],
            "chart": [{"bbox_xyxy_norm": chart_a, "text": "crop-11"}],
            "text": "crop-11",
            "metadata": {
                "num_detections": 3,
                "counts_by_label": {"table": 1, "chart": 1, "text": 1},
            },
        },
        "page_B": {
            "table_bboxes": [table_b1, table_b2],
            "infographic": [{"bbox_xyxy_norm": infographic_b, "text": "crop-22"}],
            "text": "native-B",
            "metadata": {
                "num_detections": 3,
                "counts_by_label": {"table": 2, "infographic": 1},
            },
        },
        "empty_page": {
            "table": [],
            "chart": [],
            "infographic": [],
            "text": "native-empty",
            "num_detections": 0,
        },
        "malformed_page": {
            "table": [],
            "chart": [],
            "infographic": [],
            "text": "native-malformed",
            "error": {"stage": "page_render", "message": "missing raster"},
            "num_detections": 0,
        },
    }


def test_local_actor_isolates_one_failed_crop_to_its_source_row() -> None:
    model = _FallbackIsolationModel(failing_crop_id=22)
    actor = _local_actor(model, inference_batch_size=3, extract_charts=True)
    bbox = [0.0, 0.0, 1.0, 1.0]
    batch = pd.DataFrame(
        [
            _page("page-A", 11, [_detection("chart-A", "chart", bbox)]),
            _page("page-B", 22, [_detection("chart-B", "chart", bbox)]),
            _page("page-C", 33, [_detection("chart-C", "chart", bbox)]),
        ]
    )

    result = actor(batch)

    assert {
        "model_calls": model.calls,
        "charts": result["chart"].tolist(),
        "errors": [meta["error"] for meta in result["ocr"]],
    } == {
        "model_calls": [
            {
                "input_kind": "batch",
                "crop_ids": [11, 22, 33],
                "merge_level": "paragraph",
            },
            {
                "input_kind": "single",
                "crop_ids": [11],
                "merge_level": "paragraph",
            },
            {
                "input_kind": "single",
                "crop_ids": [22],
                "merge_level": "paragraph",
            },
            {
                "input_kind": "single",
                "crop_ids": [33],
                "merge_level": "paragraph",
            },
        ],
        "charts": [
            [{"bbox_xyxy_norm": bbox, "text": "crop-11"}],
            [],
            [{"bbox_xyxy_norm": bbox, "text": "crop-33"}],
        ],
        "errors": [
            None,
            {
                "stage": "ocr_page_elements",
                "type": "ValueError",
                "message": "bad crop 22",
                "traceback": result.at[1, "ocr"]["error"]["traceback"],
            },
            None,
        ],
    }


def test_local_actor_falls_back_when_batch_result_count_is_wrong() -> None:
    model = _WrongCountFallbackModel()
    actor = _local_actor(model, inference_batch_size=2, extract_charts=True)
    bbox = [0.0, 0.0, 1.0, 1.0]
    batch = pd.DataFrame(
        [
            _page("page-A", 11, [_detection("chart-A", "chart", bbox)]),
            _page("page-B", 22, [_detection("chart-B", "chart", bbox)]),
        ]
    )

    result = actor(batch)

    assert {
        "model_calls": model.calls,
        "charts": result["chart"].tolist(),
        "errors": [meta["error"] for meta in result["ocr"]],
    } == {
        "model_calls": [
            {
                "input_kind": "batch",
                "crop_ids": [11, 22],
                "merge_level": "paragraph",
            },
            {
                "input_kind": "single",
                "crop_ids": [11],
                "merge_level": "paragraph",
            },
            {
                "input_kind": "single",
                "crop_ids": [22],
                "merge_level": "paragraph",
            },
        ],
        "charts": [
            [{"bbox_xyxy_norm": bbox, "text": "crop-11"}],
            [{"bbox_xyxy_norm": bbox, "text": "crop-22"}],
        ],
        "errors": [None, None],
    }
