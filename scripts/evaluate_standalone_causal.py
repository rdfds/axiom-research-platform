#!/usr/bin/env python
"""Evaluate standalone causal impact for a single company/action input.

Example:
  python -u scripts/evaluate_standalone_causal.py \
    --company-id 0000320193 \
    --as-of 2026-02-28 \
    --action-id capital_return.open_market_buyback \
    --param size_pct_market_cap=0.05 \
    --param funding_mix.cash=1 \
    --snapshot-root data/company_state_snapshots/final_run_2026-02-28 \
    --entity-identifier-path data/inputs_layer/entity_identifier.parquet
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal_impact_model import load_default_causal_impact_model
from src.recommendation_run import _resolve_snapshot, _snapshot_company_aliases


def _parse_param_values(items: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid --param format: {raw}. Expected key=value.")
        key, val = raw.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            raise ValueError(f"Invalid --param key: {raw}")
        if val.lower() in {"true", "false"}:
            parsed: Any = val.lower() == "true"
        else:
            try:
                parsed = float(val) if "." in val else int(val)
            except Exception:
                parsed = val

        cur = out
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = parsed
    return out


def _as_of_datetime(raw: str) -> datetime:
    s = str(raw).strip()
    if "T" not in s:
        s = s + "T00:00:00+00:00"
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


