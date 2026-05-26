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


