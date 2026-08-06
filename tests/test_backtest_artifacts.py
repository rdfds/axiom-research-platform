from pathlib import Path

from src.backtest_artifacts import (
    build_backtest_artifact_manifest,
    fingerprint_path,
    resolve_backtest_artifact_root,
    resolve_snapshot_cache_dir,
)
from src.backtest_costs import resolve_transaction_cost_model
from src.backtest_protocol import resolve_backtest_protocol


def test_resolve_artifact_directories_default_under_runs_root(tmp_path: Path):
    runs_root = tmp_path / "runs"

    artifact_root = resolve_backtest_artifact_root(runs_root=runs_root)
    snapshot_cache_dir = resolve_snapshot_cache_dir(runs_root=runs_root)

    assert artifact_root == runs_root / "_backtest_artifacts"
    assert snapshot_cache_dir == runs_root / "_backtest_artifacts" / "snapshot_cache"


def test_fingerprint_path_marks_tmp_paths_and_files(tmp_path: Path):
    payload = tmp_path / "example.json"
    payload.write_text("{\"ok\": true}\n")

    fingerprint = fingerprint_path(payload)

    assert fingerprint["exists"] is True
    assert fingerprint["kind"] == "file"
    assert fingerprint["sample_sha256"]


def test_build_backtest_artifact_manifest_captures_inputs_and_outputs(tmp_path: Path):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}")
    input_path = tmp_path / "input.parquet"
    input_path.write_text("data")
    output_path = tmp_path / "report.json"
    output_path.write_text("{}")

    manifest = build_backtest_artifact_manifest(
        suite="manual_replay_historical_benchmark",
        benchmark_key="capital_return_holdout",
        protocol=resolve_backtest_protocol(benchmark_key="capital_return_holdout"),
        cost_model=resolve_transaction_cost_model("manual_replay_event_equal_weight_v1"),
        lock_path=lock_path,
        runs_root=tmp_path / "runs",
        artifact_root=tmp_path / "artifacts",
        snapshot_cache_dir=tmp_path / "artifacts" / "snapshot_cache",
        resolved_paths={"outcomes_path": input_path},
        resolved_env={"AXIOM_METRIC_POLICY_PATH": str(input_path)},
        outputs={"historical_report": output_path},
    )

    assert manifest["protocol"]["key"] == "capital_return_holdout_v1"
    assert manifest["resolved_inputs"]["outcomes_path"]["exists"] is True
    assert manifest["outputs"]["historical_report"]["exists"] is True
