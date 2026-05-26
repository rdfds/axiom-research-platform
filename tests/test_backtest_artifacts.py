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


