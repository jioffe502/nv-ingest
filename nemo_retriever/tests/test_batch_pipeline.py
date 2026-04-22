import sys
import json
from types import ModuleType, SimpleNamespace

from typer.testing import CliRunner

import nemo_retriever.examples.graph_pipeline as batch_pipeline
import nemo_retriever.model as model_module
import nemo_retriever.recall.beir as beir_module
import nemo_retriever.utils.detection_summary as detection_summary_module
from nemo_retriever.utils.input_files import resolve_input_patterns

RUNNER = CliRunner()


class _FakeDataset:
    def materialize(self):
        return self

    def take_all(self):
        return []

    def groupby(self, _key):
        class _FakeGrouped:
            @staticmethod
            def count():
                class _FakeCounted:
                    @staticmethod
                    def count():
                        return 1

                return _FakeCounted()

        return _FakeGrouped()


class _FakeRowsDataset:
    def __init__(self, rows):
        self._rows = rows

    def materialize(self):
        return self

    def take_all(self):
        return list(self._rows)


class _FakeErrorRows:
    def materialize(self):
        return self

    def count(self) -> int:
        return 0


class _FakeIngestor:
    def __init__(self) -> None:
        self.extract_params = None
        self.audio_extract_params = None
        self.audio_asr_params = None
        self.embed_params = None
        self.file_patterns = None

    def files(self, file_patterns):
        self.file_patterns = file_patterns
        return self

    def extract(self, params):
        self.extract_params = params
        return self

    def extract_image_files(self, params):
        self.extract_params = params
        return self

    def extract_audio(self, params=None, asr_params=None):
        self.audio_extract_params = params
        self.audio_asr_params = asr_params
        return self

    def extract_txt(self, params):
        return self

    def extract_html(self, params):
        return self

    def split(self, params):
        return self

    def embed(self, params):
        self.embed_params = params
        return self

    def ingest(self, params=None):
        return _FakeDataset()

    def get_error_rows(self, dataset=None):
        return _FakeErrorRows()


class _FakeLegacyVDB:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.records = None
        self.__class__.instances.append(self)

    def run(self, records):
        self.records = records
        return {"ok": True}


def test_resolve_input_file_patterns_recurses_for_directory_inputs(tmp_path) -> None:
    dataset_dir = tmp_path / "earnings_consulting"
    dataset_dir.mkdir()

    pdf_patterns = resolve_input_patterns(dataset_dir, "pdf")
    txt_patterns = resolve_input_patterns(dataset_dir, "txt")
    doc_patterns = resolve_input_patterns(dataset_dir, "doc")

    assert pdf_patterns == [str(dataset_dir / "**" / "*.pdf")]
    assert txt_patterns == [str(dataset_dir / "**" / "*.txt")]
    assert doc_patterns == [str(dataset_dir / "**" / "*.docx"), str(dataset_dir / "**" / "*.pptx")]


def test_graph_pipeline_resolves_nested_pdf_directories(tmp_path) -> None:
    dataset_dir = tmp_path / "earnings_consulting"
    nested_dir = dataset_dir / "amazon_earnings_call"
    nested_dir.mkdir(parents=True)
    (nested_dir / "sample.pdf").write_text("placeholder", encoding="utf-8")

    patterns = batch_pipeline._resolve_file_patterns(dataset_dir, "pdf")

    assert patterns == [str(dataset_dir / "**" / "*.pdf")]


def test_batch_pipeline_accepts_multimodal_embed_and_page_image_flags(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "sample.pdf").write_text("placeholder", encoding="utf-8")
    missing_query_csv = tmp_path / "missing.csv"

    fake_ingestor = _FakeIngestor()
    monkeypatch.setattr(batch_pipeline, "GraphIngestor", lambda *args, **kwargs: fake_ingestor)
    monkeypatch.setattr(batch_pipeline, "_ensure_lancedb_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_pipeline, "handle_lancedb", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(shutdown=lambda: None))

    class _FakeTable:
        def count_rows(self) -> int:
            return 1

    class _FakeDb:
        def open_table(self, _name):
            return _FakeTable()

    monkeypatch.setitem(sys.modules, "lancedb", SimpleNamespace(connect=lambda _uri: _FakeDb()))
    monkeypatch.setattr(model_module, "resolve_embed_model", lambda _name: "fake-embed-model")

    result = RUNNER.invoke(
        batch_pipeline.app,
        [
            str(dataset_dir),
            "--query-csv",
            str(missing_query_csv),
            "--embed-modality",
            "text_image",
            "--embed-granularity",
            "page",
            "--extract-infographics",
            "--no-extract-page-as-image",
        ],
    )

    assert result.exit_code == 0
    assert isinstance(fake_ingestor.file_patterns, list)
    assert fake_ingestor.extract_params.extract_infographics is True
    assert fake_ingestor.extract_params.extract_page_as_image is False
    assert fake_ingestor.embed_params.embed_modality == "text_image"
    assert fake_ingestor.embed_params.embed_granularity == "page"


def test_batch_pipeline_routes_audio_input_to_audio_ingestor(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "sample.mp3").write_text("placeholder", encoding="utf-8")
    missing_query_csv = tmp_path / "missing.csv"

    fake_ingestor = _FakeIngestor()
    monkeypatch.setattr(batch_pipeline, "GraphIngestor", lambda *args, **kwargs: fake_ingestor)
    monkeypatch.setattr(batch_pipeline, "_ensure_lancedb_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_pipeline, "handle_lancedb", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(shutdown=lambda: None))
    monkeypatch.setattr(
        batch_pipeline, "asr_params_from_env", lambda: SimpleNamespace(model_copy=lambda update: update)
    )

    class _FakeTable:
        def count_rows(self) -> int:
            return 1

    class _FakeDb:
        def open_table(self, _name):
            return _FakeTable()

    monkeypatch.setitem(sys.modules, "lancedb", SimpleNamespace(connect=lambda _uri: _FakeDb()))
    monkeypatch.setattr(model_module, "resolve_embed_model", lambda _name: "fake-embed-model")

    result = RUNNER.invoke(
        batch_pipeline.app,
        [
            str(dataset_dir),
            "--input-type",
            "audio",
            "--query-csv",
            str(missing_query_csv),
            "--recall-match-mode",
            "audio_segment",
            "--audio-match-tolerance-secs",
            "3.0",
            "--segment-audio",
            "--audio-split-type",
            "time",
            "--audio-split-interval",
            "45",
        ],
    )

    assert result.exit_code == 0
    assert isinstance(fake_ingestor.file_patterns, list)
    assert fake_ingestor.audio_extract_params.split_type == "time"
    assert fake_ingestor.audio_extract_params.split_interval == 45
    assert fake_ingestor.audio_asr_params["segment_audio"] is True


def test_batch_pipeline_routes_beir_mode_to_evaluator(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "sample.pdf").write_text("placeholder", encoding="utf-8")

    fake_ingestor = _FakeIngestor()
    monkeypatch.setattr(batch_pipeline, "GraphIngestor", lambda *args, **kwargs: fake_ingestor)
    monkeypatch.setattr(batch_pipeline, "_ensure_lancedb_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_pipeline, "handle_lancedb", lambda *args, **kwargs: None)
    monkeypatch.setattr(detection_summary_module, "print_run_summary", lambda *args, **kwargs: None)

    class _FakeTable:
        def count_rows(self) -> int:
            return 1

    class _FakeDb:
        def open_table(self, _name):
            return _FakeTable()

    class _FakeLanceModule:
        @staticmethod
        def connect(_uri):
            return _FakeDb()

    monkeypatch.setitem(sys.modules, "lancedb", _FakeLanceModule())
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(shutdown=lambda: None))
    monkeypatch.setattr(model_module, "resolve_embed_model", lambda _name: "fake-embed-model")

    captured = {}

    def _fake_evaluate(cfg):
        captured["cfg"] = cfg
        return type("Dataset", (), {"query_ids": ["1", "2"]})(), [], {}, {"ndcg@10": 0.75, "recall@5": 0.6}

    monkeypatch.setattr(beir_module, "evaluate_lancedb_beir", _fake_evaluate)

    result = RUNNER.invoke(
        batch_pipeline.app,
        [
            str(dataset_dir),
            "--evaluation-mode",
            "beir",
            "--beir-loader",
            "vidore_hf",
            "--beir-dataset-name",
            "vidore_v3_computer_science",
            "--beir-k",
            "5",
            "--beir-k",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert captured["cfg"].loader == "vidore_hf"
    assert captured["cfg"].dataset_name == "vidore_v3_computer_science"
    assert tuple(captured["cfg"].ks) == (5, 10)


def test_graph_pipeline_routes_custom_vdb_op_through_legacy_adt(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "sample.pdf").write_text("placeholder", encoding="utf-8")
    runtime_dir = tmp_path / "runtime_metrics"

    rows = [
        {
            "text": "alpha",
            "text_embeddings_1b_v2": {"embedding": [0.1, 0.2]},
            "source_path": str(dataset_dir / "sample.pdf"),
            "page_number": 1,
            "_content_type": "text",
        }
    ]

    class _FakeRowsIngestor(_FakeIngestor):
        def ingest(self, params=None):
            return _FakeRowsDataset(rows)

    fake_ingestor = _FakeRowsIngestor()
    monkeypatch.setattr(batch_pipeline, "GraphIngestor", lambda *args, **kwargs: fake_ingestor)
    monkeypatch.setattr(
        batch_pipeline,
        "_ensure_lancedb_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LanceDB table should not be initialized")),
    )
    monkeypatch.setattr(
        batch_pipeline,
        "handle_lancedb",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LanceDB writer should not run")),
    )
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(shutdown=lambda: None))

    _FakeLegacyVDB.instances.clear()
    fake_module = ModuleType("fake_legacy_vdb_module")
    fake_module.FakeLegacyVDB = _FakeLegacyVDB
    monkeypatch.setitem(sys.modules, "fake_legacy_vdb_module", fake_module)

    result = RUNNER.invoke(
        batch_pipeline.app,
        [
            str(dataset_dir),
            "--vdb-op",
            "fake_legacy_vdb_module:FakeLegacyVDB",
            "--vdb-op-kwargs",
            '{"collection_name":"custom"}',
            "--runtime-metrics-dir",
            str(runtime_dir),
            "--runtime-metrics-prefix",
            "custom-vdb",
        ],
    )

    assert result.exit_code == 0, result.output
    vdb = _FakeLegacyVDB.instances[-1]
    assert vdb.kwargs == {"collection_name": "custom"}
    assert len(vdb.records) == 1
    assert len(vdb.records[0]) == 1
    assert vdb.records[0][0]["metadata"]["content"] == "alpha"

    payload = json.loads((runtime_dir / "custom-vdb.runtime.summary.json").read_text(encoding="utf-8"))
    assert payload["vdb_backend"] == "legacy_vdb"
    assert payload["vdb_records"] == 1
    assert payload["evaluation_mode"] == "skipped"


def test_batch_pipeline_accepts_harness_runtime_metric_flags(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "sample.pdf").write_text("placeholder", encoding="utf-8")
    missing_query_csv = tmp_path / "missing.csv"
    runtime_dir = tmp_path / "runtime_metrics"

    fake_ingestor = _FakeIngestor()
    monkeypatch.setattr(batch_pipeline, "GraphIngestor", lambda *args, **kwargs: fake_ingestor)
    monkeypatch.setattr(batch_pipeline, "_ensure_lancedb_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_pipeline, "handle_lancedb", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(shutdown=lambda: None))

    class _FakeTable:
        def count_rows(self) -> int:
            return 1

    class _FakeDb:
        def open_table(self, _name):
            return _FakeTable()

    monkeypatch.setitem(sys.modules, "lancedb", SimpleNamespace(connect=lambda _uri: _FakeDb()))
    monkeypatch.setattr(model_module, "resolve_embed_model", lambda _name: "fake-embed-model")

    result = RUNNER.invoke(
        batch_pipeline.app,
        [
            str(dataset_dir),
            "--query-csv",
            str(missing_query_csv),
            "--runtime-metrics-dir",
            str(runtime_dir),
            "--runtime-metrics-prefix",
            "sample-run",
            "--no-recall-details",
        ],
    )

    assert result.exit_code == 0
    summary_path = runtime_dir / "sample-run.runtime.summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["recall_details"] is False
    assert payload["evaluation_mode"] == "recall"
