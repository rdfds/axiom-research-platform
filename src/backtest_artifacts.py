from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .backtest_costs import TransactionCostModel
from .backtest_protocol import BacktestProtocol


def resolve_backtest_artifact_root(
    *,
    runs_root: str | Path,
    artifact_root: Optional[str | Path] = None,
) -> Path:
    if artifact_root:
        return Path(artifact_root)
    return Path(runs_root) / "_backtest_artifacts"


def resolve_snapshot_cache_dir(
    *,
    runs_root: str | Path,
    artifact_root: Optional[str | Path] = None,
    snapshot_cache_dir: Optional[str | Path] = None,
) -> Path:
    if snapshot_cache_dir:
        return Path(snapshot_cache_dir)
    return resolve_backtest_artifact_root(runs_root=runs_root, artifact_root=artifact_root) / "snapshot_cache"


def _sample_file_digest(path: Path, sample_bytes: int = 1_048_576) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
        h = hashlib.sha256()
        with path.open("rb") as handle:
            head = handle.read(sample_bytes)
            h.update(head)
            h.update(str(size).encode("utf-8"))
            if size > sample_bytes:
                handle.seek(max(0, size - sample_bytes))
                h.update(handle.read(sample_bytes))
        return h.hexdigest()
    except Exception:
        return None


def fingerprint_path(path_like: str | Path) -> Dict[str, Any]:
    path = Path(path_like)
    exists = path.exists()
    payload: Dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "is_tmp_path": str(path).startswith("/tmp/") or str(path).startswith("/private/tmp/"),
        "kind": "missing",
    }
    if not exists:
        return payload
    stat = path.stat()
    payload["size_bytes"] = int(stat.st_size)
    payload["modified_at"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    if path.is_dir():
        payload["kind"] = "directory"
        try:
            payload["child_count"] = sum(1 for _ in path.iterdir())
        except Exception:
            payload["child_count"] = None
        return payload
    payload["kind"] = "file"
    payload["sample_sha256"] = _sample_file_digest(path)
    return payload


def build_backtest_artifact_manifest(
    *,
    suite: str,
    benchmark_key: str,
    protocol: BacktestProtocol,
    cost_model: TransactionCostModel,
    lock_path: str | Path,
    runs_root: str | Path,
    artifact_root: str | Path,
    snapshot_cache_dir: str | Path,
    resolved_paths: Mapping[str, str | Path],
    resolved_env: Mapping[str, str],
    outputs: Mapping[str, str | Path],
) -> Dict[str, Any]:
    return {
        "suite": suite,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_key": benchmark_key,
        "protocol": protocol.to_dict(),
        "cost_model": cost_model.to_dict(),
        "lock_config": fingerprint_path(lock_path),
        "runs_root": fingerprint_path(runs_root),
        "artifact_root": fingerprint_path(artifact_root),
        "snapshot_cache_dir": fingerprint_path(snapshot_cache_dir),
        "resolved_inputs": {
            str(name): fingerprint_path(path)
            for name, path in dict(resolved_paths or {}).items()
        },
        "resolved_env_overrides": {
            str(name): fingerprint_path(path)
            for name, path in dict(resolved_env or {}).items()
        },
        "outputs": {
            str(name): fingerprint_path(path)
            for name, path in dict(outputs or {}).items()
        },
    }


__all__ = [
    "build_backtest_artifact_manifest",
    "fingerprint_path",
    "resolve_backtest_artifact_root",
    "resolve_snapshot_cache_dir",
]
