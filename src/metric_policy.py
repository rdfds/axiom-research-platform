from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metric_methodology import MetricMethodologyRegistry


ROOT = Path(__file__).resolve().parents[1]

_FALLBACK_SECTOR_FIELD_CANDIDATES = [
    "gics_sector",
    "sector",
    "Sector",
]

_FALLBACK_SUBSECTOR_FIELD_CANDIDATES = [
    "gics_sub_industry",
    "subsector",
    "Subsector",
    "industry",
    "Industry",
]

_FALLBACK_UNSUPPORTED_FINANCIAL_METRICS = {
    "capital_structure.gross_leverage",
    "capital_structure.net_leverage",
}


def _default_policy_path() -> Path:
    env = str(os.environ.get("AXIOM_METRIC_POLICY_PATH", "") or "").strip()
    if env:
        return Path(env)
    return ROOT / "configs" / "metric_policies" / "market_metric_policy_v1.json"


