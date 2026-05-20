from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from nemo_retriever.graph import Graph
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.ingest_manifest import build_input_manifest, plan_extraction_branches


class _TagOperator(AbstractOperator):
    def __init__(self, *, tag: str) -> None:
        super().__init__(tag=tag)
        self.tag = tag

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return pd.DataFrame(
            {
                "path": list(data["path"]),
                f"{self.tag}_value": [self.tag] * len(data),
            }
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class _PostOperator(AbstractOperator):
    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data.assign(post_extract=True)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def _graph_with(operator: AbstractOperator) -> Graph:
    return Graph() >> operator


def test_manifest_planner_pdf_doc_share_dedicated_pdf_branch(tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    pptx = tmp_path / "deck.pptx"
    pdf.write_bytes(b"pdf")
    pptx.write_bytes(b"pptx")

    branches = plan_extraction_branches(build_input_manifest([str(pdf), str(pptx)]))

    assert [(branch.family, branch.extraction_mode, branch.input_paths) for branch in branches] == [
        ("pdf", "pdf", (str(pdf), str(pptx))),
    ]


def test_manifest_planner_mixed_inputs_use_stable_family_order(tmp_path) -> None:
    text = tmp_path / "notes.txt"
    image = tmp_path / "scan.png"
    pdf = tmp_path / "manual.pdf"
    text.write_text("notes", encoding="utf-8")
    image.write_bytes(b"png")
    pdf.write_bytes(b"pdf")

    branches = plan_extraction_branches(build_input_manifest([str(text), str(image), str(pdf)]))

    assert [branch.family for branch in branches] == ["pdf", "image", "txt"]


def test_manifest_planner_rejects_unsupported_concrete_extensions(tmp_path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"unknown")

    with pytest.raises(ValueError, match="payload.bin"):
        plan_extraction_branches(build_input_manifest([str(payload)]))


def test_manifest_planner_empty_glob_does_not_invent_modal_branches(tmp_path) -> None:
    branches = plan_extraction_branches(build_input_manifest([str(tmp_path / "*.wav")]))

    assert [(branch.family, branch.input_paths) for branch in branches] == [("pdf", (str(tmp_path / "*.wav"),))]


def test_explicit_extraction_mode_bypasses_manifest_planning(tmp_path) -> None:
    image = tmp_path / "scan.png"
    image.write_bytes(b"png")
    ingestor = GraphIngestor(run_mode="inprocess").files([str(image)]).extract(extraction_mode="auto")

    assert ingestor._plan_default_extraction_branches() is None
    assert ingestor._resolve_effective_extraction_inputs().extraction_mode == "auto"


def test_inprocess_branch_execution_unions_schemas_and_runs_post_once(monkeypatch, tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    image = tmp_path / "scan.png"
    text = tmp_path / "notes.txt"
    pdf.write_bytes(b"pdf")
    image.write_bytes(b"png")
    text.write_text("notes", encoding="utf-8")
    extraction_calls: list[dict[str, Any]] = []
    post_calls: list[dict[str, Any]] = []

    def fake_build_graph(**kwargs: Any) -> Graph:
        extraction_calls.append(kwargs)
        return _graph_with(_TagOperator(tag=kwargs["extraction_mode"]))

    def fake_post_graph(**kwargs: Any) -> Graph:
        post_calls.append(kwargs)
        return _graph_with(_PostOperator())

    monkeypatch.setattr("nemo_retriever.graph_ingestor.build_graph", fake_build_graph)
    monkeypatch.setattr("nemo_retriever.graph_ingestor.build_post_extract_graph", fake_post_graph)

    result = (
        GraphIngestor(run_mode="inprocess", show_progress=False)
        .files([str(text), str(image), str(pdf)])
        .extract()
        .embed()
        .ingest()
    )

    assert [call["extraction_mode"] for call in extraction_calls] == ["pdf", "image", "text"]
    assert all(call.get("embed_params") is None for call in extraction_calls)
    assert len(post_calls) == 1
    assert post_calls[0]["embed_params"] is not None
    assert set(result.columns) == {"path", "pdf_value", "image_value", "text_value", "post_extract"}
    assert result["post_extract"].tolist() == [True, True, True]


class _FakeDataset:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.unioned: list[_FakeDataset] = []
        self.normalized_columns: tuple[str, ...] | None = None

    def schema(self) -> Any:
        return SimpleNamespace(names=self.columns)

    def map_batches(self, *_args: Any, **kwargs: Any) -> "_FakeDataset":
        self.normalized_columns = kwargs["fn_kwargs"]["columns"]
        return self

    def union(self, other: "_FakeDataset") -> "_FakeDataset":
        self.unioned.append(other)
        return self


class _LazySchemaDataset:
    def __init__(self) -> None:
        self.map_batches_called = False

    def schema(self, *, fetch_if_missing: bool = True) -> None:
        assert fetch_if_missing is False
        return None

    def map_batches(self, *_args: Any, **_kwargs: Any) -> "_LazySchemaDataset":
        self.map_batches_called = True
        return self


def test_ray_schema_normalization_does_not_trigger_lazy_schema_fetch() -> None:
    datasets = [_LazySchemaDataset(), _LazySchemaDataset()]

    normalized = GraphIngestor._normalize_ray_branch_datasets(datasets)

    assert normalized == datasets
    assert all(not dataset.map_batches_called for dataset in datasets)


def test_batch_branch_execution_uses_dataset_union(monkeypatch, tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    image = tmp_path / "scan.png"
    pdf.write_bytes(b"pdf")
    image.write_bytes(b"png")
    datasets = [_FakeDataset(["path", "pdf_value"]), _FakeDataset(["path", "image_value"])]
    executor_calls: list[dict[str, Any]] = []

    class FakeCluster:
        def available_gpu_count(self) -> int:
            return 0

        def total_cpu_count(self) -> int:
            return 64

    class FakeExecutor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def ingest(self, data: Any, *, materialize: bool = True, **kwargs: Any) -> Any:
            executor_calls.append({"data": data, "materialize": materialize})
            if materialize:
                return pd.DataFrame({"done": [True]})
            return datasets.pop(0)

    monkeypatch.setattr(GraphIngestor, "_ensure_batch_runtime", lambda self: (None, FakeCluster()))
    monkeypatch.setattr("nemo_retriever.graph_ingestor.RayDataExecutor", FakeExecutor)
    monkeypatch.setattr("nemo_retriever.graph_ingestor.build_graph", lambda **_kwargs: Graph())
    monkeypatch.setattr("nemo_retriever.graph_ingestor.build_post_extract_graph", lambda **_kwargs: Graph())

    result = GraphIngestor(run_mode="batch").files([str(pdf), str(image)]).extract().ingest()

    assert len(executor_calls) == 3
    assert executor_calls[0]["materialize"] is False
    assert executor_calls[1]["materialize"] is False
    combined = executor_calls[2]["data"]
    assert isinstance(combined, _FakeDataset)
    assert len(combined.unioned) == 1
    assert combined.normalized_columns == ("path", "pdf_value", "image_value")
    assert result["done"].tolist() == [True]
