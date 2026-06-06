#!/usr/bin/env python
"""
Run precedent-based matching for a single company + action.

Example:
  python -u scripts/50_run_precedent_pipeline.py \
    --company-id 001690 \
    --as-of 2024-12-31 \
    --action-id capital_return.open_market_buyback \
    --param size_pct_market_cap=0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.run import run_precedent


def _default_precedent_outcomes_path() :
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet")


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
            out[key] = val.lower() == "true"
            continue
        try:
            if "." in val:
                out[key] = float(val)
            else:
                out[key] = int(val)
            continue
        except Exception:
            pass
        out[key] = val
    return out


