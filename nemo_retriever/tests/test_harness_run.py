from pathlib import Path

from typer.testing import CliRunner

from nemo_retriever.harness.cli import app as harness_app
from nemo_retriever.harness.artifacts import create_run_artifact_dir
from nemo_retriever.harness.config import HarnessConfig
from nemo_retriever.harness import run as harness_run
from nemo_retriever.harness.run import _build_command, _evaluate_run_outcome, _normalize_recall_metric_key

RUNNER = CliRunner()


def test_evaluate_run_outcome_passes_when_process_succeeds_and_recall_present() -> None:
    rc, reason, success = _evaluate_run_outcome(0, True, {"recall@5": 0.9})
    assert rc == 0
    assert reason == ""
    assert success is True


def test_evaluate_run_outcome_fails_when_recall_required_and_missing() -> None:
    rc, reason, success = _evaluate_run_outcome(0, True, {})
    assert rc == 98
    assert reason == "missing_recall_metrics"
    assert success is False


def test_evaluate_run_outcome_uses_subprocess_error_code() -> None:
    rc, reason, success = _evaluate_run_outcome(2, True, {"recall@5": 0.9})
    assert rc == 2
    assert reason == "subprocess_exit_2"
    assert success is False


def test_create_run_artifact_dir_defaults_to_dataset_label(tmp_path: Path) -> None:
    out = create_run_artifact_dir("jp20", run_name=None, base_dir=str(tmp_path))
    assert out.name.startswith("jp20_")


def test_build_command_uses_hidden_detection_file_by_default(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("q,s,p\nx,y,1\n", encoding="utf-8")

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
        query_csv=str(query_csv),
        write_detection_file=False,
    )
    cmd, runtime_dir, detection_file, effective_query_csv = _build_command(cfg, tmp_path, run_id="r1")
    assert "--detection-summary-file" in cmd
    assert "--recall-match-mode" in cmd
    assert "pdf_page" in cmd
    assert detection_file.parent == runtime_dir
    assert detection_file.name == ".detection_summary.json"
    assert effective_query_csv == query_csv


def test_build_command_uses_top_level_detection_file_when_enabled(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("q,s,p\nx,y,1\n", encoding="utf-8")

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
        query_csv=str(query_csv),
        write_detection_file=True,
    )
    cmd, runtime_dir, detection_file, effective_query_csv = _build_command(cfg, tmp_path, run_id="r1")
    assert "--detection-summary-file" in cmd
    assert detection_file.parent == tmp_path
    assert detection_file.name == "detection_summary.json"
    assert effective_query_csv == query_csv


def test_build_command_applies_page_plus_one_adapter(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf,page\nq,doc_name.pdf,0\n", encoding="utf-8")

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="earnings",
        preset="single_gpu",
        query_csv=str(query_csv),
        recall_adapter="page_plus_one",
    )
    cmd, runtime_dir, _detection_file, effective_query_csv = _build_command(cfg, tmp_path, run_id="r1")

    assert effective_query_csv.parent == runtime_dir
    assert effective_query_csv.name == "query_adapter.page_plus_one.csv"
    assert "--query-csv" in cmd
    assert str(effective_query_csv) in cmd
    csv_contents = effective_query_csv.read_text(encoding="utf-8")
    assert "query,pdf_page" in csv_contents
    assert "q,doc_name_1" in csv_contents


def test_normalize_recall_metric_key_removes_duplicate_prefix() -> None:
    assert _normalize_recall_metric_key("recall@1") == "recall_1"
    assert _normalize_recall_metric_key("recall@10") == "recall_10"


def test_run_single_writes_tags_to_results_json(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")
    runtime_dir = tmp_path / "runtime_metrics"
    runtime_dir.mkdir()

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
        query_csv=str(query_csv),
    )

    monkeypatch.setattr(
        harness_run,
        "_build_command",
        lambda *_args, **_kwargs: (["python", "-V"], runtime_dir, runtime_dir / ".detection_summary.json", query_csv),
    )

    def _fake_run_subprocess(_cmd: list[str], metrics) -> int:
        metrics.files = 20
        metrics.pages = 100
        metrics.ingest_secs = 10.0
        metrics.pages_per_sec_ingest = 10.0
        metrics.recall_metrics = {"recall@1": 0.5, "recall@5": 0.8}
        return 0

    monkeypatch.setattr(harness_run, "_run_subprocess_with_tty", _fake_run_subprocess)
    monkeypatch.setattr(harness_run, "last_commit", lambda: "abc123")
    monkeypatch.setattr(harness_run, "now_timestr", lambda: "20260305_000000_UTC")

    captured: dict[str, dict] = {}

    def _fake_write_json(_path: Path, payload: dict) -> None:
        captured["payload"] = payload

    monkeypatch.setattr(harness_run, "write_json", _fake_write_json)

    harness_run._run_single(cfg, tmp_path, run_id="r1", tags=["nightly", "candidate"])
    assert captured["payload"]["tags"] == ["nightly", "candidate"]
    assert captured["payload"]["metrics"]["recall_1"] == 0.5
    assert captured["payload"]["metrics"]["recall_5"] == 0.8


def test_run_entry_session_artifact_dir_uses_run_name(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
    )
    monkeypatch.setattr(harness_run, "load_harness_config", lambda **_: cfg)

    def _fake_run_single(_cfg: HarnessConfig, _artifact_dir: Path, run_id: str, tags: list[str] | None = None) -> dict:
        assert tags == []
        return {
            "success": True,
            "return_code": 0,
            "failure_reason": None,
            "metrics": {"files": 0, "pages": 0},
        }

    monkeypatch.setattr(harness_run, "_run_single", _fake_run_single)

    result = harness_run._run_entry(
        run_name="jp20_single",
        config_file=None,
        session_dir=tmp_path,
        dataset="jp20",
        preset="single_gpu",
    )

    assert Path(result["artifact_dir"]).name == "jp20_single"


def test_run_entry_returns_tags(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
    )
    monkeypatch.setattr(harness_run, "load_harness_config", lambda **_: cfg)

    def _fake_run_single(_cfg: HarnessConfig, _artifact_dir: Path, run_id: str, tags: list[str] | None = None) -> dict:
        assert run_id == "jp20_single"
        assert tags == ["nightly", "candidate"]
        return {
            "success": True,
            "return_code": 0,
            "failure_reason": None,
            "metrics": {"files": 0, "pages": 0},
        }

    monkeypatch.setattr(harness_run, "_run_single", _fake_run_single)

    result = harness_run._run_entry(
        run_name="jp20_single",
        config_file=None,
        session_dir=tmp_path,
        dataset="jp20",
        preset="single_gpu",
        tags=["nightly", "candidate"],
    )

    assert result["tags"] == ["nightly", "candidate"]


def test_execute_runs_does_not_write_sweep_results_file(monkeypatch, tmp_path: Path) -> None:
    session_dir = tmp_path / "nightly_session"
    session_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(harness_run, "create_session_dir", lambda *_args, **_kwargs: session_dir)

    def _fake_run_entry(**_kwargs) -> dict:
        return {
            "run_name": "jp20_single",
            "dataset": "jp20",
            "preset": "single_gpu",
            "artifact_dir": str((session_dir / "jp20_single").resolve()),
            "success": True,
            "return_code": 0,
            "failure_reason": None,
            "metrics": {"files": 20, "pages": 3181},
        }

    monkeypatch.setattr(harness_run, "_run_entry", _fake_run_entry)

    harness_run.execute_runs(
        runs=[{"name": "jp20_single", "dataset": "jp20", "preset": "single_gpu"}],
        config_file=None,
        session_prefix="nightly",
        preset_override=None,
    )

    assert not (session_dir / "sweep_results.json").exists()


def test_cli_run_accepts_repeated_tags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_entry(**kwargs) -> dict:
        captured.update(kwargs)
        return {
            "run_name": "jp20",
            "dataset": "jp20",
            "preset": "single_gpu",
            "artifact_dir": "/tmp/jp20",
            "success": True,
            "return_code": 0,
            "failure_reason": None,
            "metrics": {"files": 20, "pages": 100},
            "tags": ["nightly", "candidate"],
        }

    monkeypatch.setattr(harness_run, "_run_entry", _fake_run_entry)

    result = RUNNER.invoke(
        harness_app,
        ["run", "--dataset", "jp20", "--tag", "nightly", "--tag", "candidate"],
    )

    assert result.exit_code == 0
    assert captured["tags"] == ["nightly", "candidate"]
