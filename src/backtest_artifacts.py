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


