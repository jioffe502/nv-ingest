<!-- SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retriever Harness

The Retriever Harness runs registered ingest and retrieval benchmarks with
repeatable configuration and machine-readable results. Use `retriever ingest`
and `retriever query` for your own data; use `retriever harness` for benchmark
and evaluation work.

One harness supports two execution paths:

- **Library execution** runs Retriever directly in `local` or Ray-backed
  `batch` mode.
- **Service execution** tests an existing Retriever endpoint or a temporary
  Helm deployment.

Both paths use the same runfiles, metric gates, and artifact contract.

## Start Here

| Goal | Guide |
| --- | --- |
| Run one small benchmark | [Library execution](docs/library.md#run-one-benchmark) |
| Run a larger or multi-GPU benchmark | [Library execution](docs/library.md#run-in-batch) |
| Test an existing Retriever service | [Service execution](docs/service.md#test-an-existing-service) |
| Provision and test a service with Helm | [Service execution](docs/service.md#provision-a-service-with-helm) |
| Run the workstation suite every day | [Nightly launcher](../../ops/retriever-nightly/README.md) |
| Review known dataset facts and observed results | [Expected results](docs/expected-results.md) |

Run harness commands from the repository root:

```bash
uv run --project nemo_retriever retriever harness list --runsets
uv run --project nemo_retriever retriever harness show jp20_beir --json
```

Use `retriever harness <command> --help` for the complete option list.

Service-mode benchmarks wait for remote document completion without downloading
retained result payloads. Their `rows_processed` value is the sum of
`result_rows` from successful document-completion events; the independent
VectorDB coverage checks remain the authoritative end-to-end validation. As a
result, service `ingest_secs` measures remote ingestion completion rather than
ingestion plus client-side result materialization. Service timing baselines
recorded before this behavior changed are not directly comparable.

## Commands

| Command | Purpose |
| --- | --- |
| `list` | List registered benchmarks and optional runsets. |
| `show` | Show one resolved benchmark definition. |
| `run` | Run one registered benchmark. |
| `run-set` | Run a registered benchmark group using registry paths. |
| `run-files` | Run one or more portable runfiles as a session. |
| `run-helm` | Provision a service, run one session, collect failure logs, and tear it down. |
| `check-vidore-access` | Check remote ViDoRe queries, qrels, and corpora without downloading them. |
| `compare-retrieval` | Replay dense, sparse, and hybrid retrieval against one completed ViDoRe index without re-ingesting. |
| `post-slack` | Preview or post completed artifacts without rerunning a benchmark. |
| `diff` | Compare summary metrics from two completed runs. |

`run-files` is the normal portable entrypoint. It accepts one runfile for one
benchmark or several runfiles for a suite. Real children run sequentially in
fresh processes so each run releases its model and Ray resources.

## Configuration

The Python registry owns benchmark definitions. A runfile selects one registered
benchmark and may set its mode, narrow overrides, and explicit metric gates:

```json
{
  "schema_version": 1,
  "benchmark": "jp20_beir",
  "mode": "local",
  "require": ["files==20", "pages==1940"]
}
```

Runfiles may be JSON or YAML. They cannot define new benchmarks. For a
`run-files` session, set `--output-dir` and `--dry-run` on the command rather
than in individual runfiles.

Dataset locations are machine-specific. Copy
[`dataset_paths.example.yaml`](dataset_paths.example.yaml) outside the
repository, edit it, and pass the file with `--dataset-paths`:

```bash
cp nemo_retriever/harness/dataset_paths.example.yaml \
  /local/path/to/dataset_paths.yaml
```

Relative paths in that file resolve from the file's directory. Settings resolve
from lowest to highest precedence:

1. Benchmark registry defaults.
2. Runfile values.
3. Machine-local dataset paths.
4. CLI `--set` values.

Keep credentials, webhooks, and machine-local paths out of runfiles and source
control.

## Results and Artifacts

Use the process exit code and terminal JSON files—not console output—to decide
whether a run succeeded:

| File | Read it when |
| --- | --- |
| `status.json` | A run is active and you need its current phase. |
| `results.json` | One run is terminal. |
| `session_summary.json` | A `run-files` or `run-set` session is terminal. |

Terminal files contain `success`, `exit_code`, summary metrics, and relative
paths to detailed evidence. Follow those paths only when needed:

- `run.log` for full errors and lower-level output
- `resolved_benchmark.json`, `ingest_plan.json`, and `query_plan.json` for the
  effective configuration
- `environment.json` for the source revision, GPU inventory, workload-visible
  GPU count, and runtime context
- `beir_metrics.json`, `beir_run.trec`, and `query_results.jsonl` for evaluation
  details
- `service_logs/` for a failed managed Helm deployment

A multi-run session has this stable layout:

```text
<session-output-dir>/
  expanded_runs.json
  session_summary.json
  001_<runfile-name>/
  002_<runfile-name>/
```

Dry-runs resolve configuration and write planning artifacts, but they are not
execution evidence.

### Compare ViDoRe retrieval modes

Canonical BEIR benchmarks now ingest hybrid-capable tables. Explicitly named
`*_dense` benchmarks and the `vidore_v3_all_dense` runset retain the historical
dense configuration. Every run records both requested and resolved index mode
in `results.json`, so baseline tooling can keep them distinct.

Hybrid query execution uses the shared no-model-reranker weighted-RRF policy:
50 candidates per dense/FTS leg, dense weight 0.8, sparse weight 0.2, and RRF
`k=10`. Comparison summaries record this resolved fusion policy alongside the
LanceDB version.

After a completed local or batch ViDoRe run, replay all three retrieval modes
against the identical extracted rows:

```bash
uv run --project nemo_retriever retriever harness compare-retrieval \
  /path/to/completed-run \
  --modes dense,sparse,hybrid
```

The command refuses non-hybrid source indexes and existing output directories.
It writes `retrieval_comparison/summary.json`, `query_cases.jsonl`,
`extraction_audit.json`, `report.md`, per-mode query/BEIR/TREC artifacts, and
available annotated evidence crops. Extraction `needs_review` classifications
are weak-overlap diagnostics, never confirmed extraction failures.

## Gates and Exit Codes

Metric gates compare values in `results.json.summary_metrics`:

```bash
--require 'files==20'
--require 'recall_5>=0.85'
```

During a dry-run, static facts can be evaluated; execution metrics are recorded
as skipped. The harness has no implicit quality or performance threshold.
[`expected-results.md`](docs/expected-results.md) contains human-readable observations
that can inform explicit gates.

| Code | Meaning |
| ---: | --- |
| `0` | Success |
| `2` | Invalid benchmark, configuration, override, or gate |
| `3` | Missing dataset or input |
| `4` | Managed Helm lifecycle failure |
| `10` | Ingest failure |
| `11` | Query failure |
| `12` | Evaluation failure |
| `20` | Metric gate failure |
| `30` | Artifact write failure |
| `70` | Unexpected internal error |

## Report Completed Results

Execution and reporting are separate. Preview a Slack report without a webhook:

```bash
uv run --project nemo_retriever retriever harness post-slack \
  --preview \
  --title "Retriever benchmark results" \
  /path/to/session
```

To post, export `SLACK_WEBHOOK_URL` and remove `--preview`. Each invocation
creates a new message and never changes the completed run.

A nightly can display the current release beside matching observations. Keep
one release snapshot outside the repository:

```json
{
  "baselines": [
    {
      "name": "RC26.05 Perflab",
      "dataset": "bo767",
      "resolved_index_mode": "dense",
      "environment": {"gpu_sku": "NVIDIA H100 80GB HBM3", "gpu_count": 8},
      "metrics": {"ingest_secs": 4036.85, "pages_per_sec_ingest": 13.56}
    }
  ]
}
```

Pass it with `--reference-file` or set
`RETRIEVER_HARNESS_REFERENCE_FILE`. The report shows the two observations with
their GPU context; it does not assign a verdict, update the file, or maintain
history. Retrieval references match only runs with the same
`resolved_index_mode`; this prevents a dense baseline from being presented as
a hybrid comparison.

Compare two local runs without Slack:

```bash
uv run --project nemo_retriever retriever harness diff \
  /path/to/left/results.json \
  /path/to/right/results.json
```

## Automation Contract

For agents and other callers:

1. Discover benchmarks with `list --runsets --json`.
2. Inspect the target with `show <benchmark> --json`.
3. Supply machine-local dataset paths outside the repository.
4. Dry-run the exact command before expensive GPU work.
5. Use the exit code and terminal JSON file as the result.
6. Read detailed artifacts only when the terminal result points to them.

Do not parse progress bars, human CLI formatting, or raw stdout. Scheduling,
retry policy, secret distribution, and result history belong to the caller. The
[nightly launcher](../../ops/retriever-nightly/README.md) is one such caller.
