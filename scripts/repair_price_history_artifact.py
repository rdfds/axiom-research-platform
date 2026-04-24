#!/usr/bin/env python3
"""Repair price-history-derived market metrics in a materialized artifact."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import pyarrow.parquet as pq


REPAIR_METRICS = [
    "market.volatility_30d",
    "market.volatility_90d",
    "market.drawdown_90d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--market-cache-path", help="Optional CRSP market cache parquet path")
    parser.add_argument("--out", required=True, help="Output repaired JSONL artifact")
    parser.add_argument("--summary-out", help="Optional summary JSON")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _node_support(node: Dict[str, Any] | None) -> str:
    if not node:
        return "unsupported"
    return str(node.get("support_mode") or "unsupported")


def _union_provenance(*nodes: Dict[str, Any] | None) -> list[Dict[str, Any]]:
    merged: list[Dict[str, Any]] = []
    seen = set()
    for node in nodes:
        for prov in (node or {}).get("provenance") or []:
            key = json.dumps(prov, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(copy.deepcopy(prov))
    return merged


def _base_repaired_node(node: Dict[str, Any], *, computed_at: str) -> Dict[str, Any]:
    repaired = copy.deepcopy(node)
    repaired["computed_at"] = computed_at
    repaired["missing_reason"] = None
    repaired["quality_flags"] = repaired.get("quality_flags") or None
    return repaired


def _needs_exact_price_history_repair(node: Dict[str, Any] | None) -> bool:
    if not node:
        return False
    if node.get("value") is None:
        return True
    fallback_used = str(node.get("fallback_used") or "")
    quality_flags = {str(flag) for flag in (node['quality_flags'] or [])}
    support_mode = str(node.get("support_mode") or "unsupported")
    component_breakdown = node.get("component_breakdown") or {}
    formula = str(component_breakdown.get("formula") or "")
    source_kind = str(component_breakdown.get("source_kind") or "")
    selected_series = component_breakdown.get("selected_price_series") or {}
    selected_source_kind = str(selected_series.get("source_kind") or "")

    if support_mode != "exact":
        return True
    if fallback_used == "monthly_price_history_proxy":
        return True
    if "monthly_price_history_proxy" in quality_flags:
        return True
    if "monthly_returns" in formula or "monthly_price_window" in formula:
        return True
    if source_kind and source_kind != "crsp_market_cache":
        return True
    if selected_source_kind and selected_source_kind != "crsp_market_cache":
        return True
    return False


def _infer_market_cache_path(artifact_path: Path) -> Path | None:
    for row in iter_rows(artifact_path):
        features = row.get("features") or {}
        for metric in ("market.total_return_3m_standardized", "market.total_return_12m_standardized"):
            node = features.get(metric) or {}
            for prov in node.get("provenance") or []:
                source = prov.get("source")
                if source and str(source).endswith(".parquet"):
                    return Path(str(source))
    return None


