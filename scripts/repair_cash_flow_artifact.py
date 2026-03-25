#!/usr/bin/env python3
"""Repair cash-flow-based metrics in a materialized company-state artifact."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    from repair_operating_history_artifact import (
        _companyfacts_priority_ttm,
        _load_companyfacts,
    )
except Exception:  # noqa: BLE001
    from scripts.repair_operating_history_artifact import (  # type: ignore
        _companyfacts_priority_ttm,
        _load_companyfacts,
    )


REPAIR_METRICS = [
    "market.fcf_yield",
    "operating.fcf_conversion",
]

OPERATING_CASH_FLOW_TTM_CONCEPTS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
CAPEX_TTM_CONCEPTS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PurchaseOfPropertyPlantAndEquipment",
    "PropertyPlantAndEquipmentAdditions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--companyfacts-root", required=True, help="SEC companyfacts folder")
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


def _node_value(node: Dict[str, Any] | None) -> float | None:
    if not node:
        return None
    value = node.get("value")
    if value is None:
        return None
    return float(value)


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


def _companyfacts_provenance(companyfacts_path: Path, *, as_of_time: str, computed_at: str) -> list[dict[str, Any]]:
    return [
        {
            "artifact_type": "SecCompanyFacts",
            "artifact_id": f"sec_companyfacts:{companyfacts_path.name}",
            "source": str(companyfacts_path),
            "published_at": as_of_time,
            "ingested_at": computed_at,
            "hash": None,
        }
    ]


def _base_repaired_node(node: Dict[str, Any], *, computed_at: str) -> Dict[str, Any]:
    repaired = copy.deepcopy(node)
    repaired["computed_at"] = computed_at
    repaired["missing_reason"] = None
    repaired["quality_flags"] = repaired.get("quality_flags") or None
    return repaired


def _repairable_fcf_inputs(
    *,
    companyfacts: Dict[str, Any] | None,
    as_of_date: str,
) -> tuple[float | None, float | None, float | None, Dict[str, Any] | None]:
    operating_cash_flow, operating_cash_flow_meta = _companyfacts_priority_ttm(
        companyfacts,
        OPERATING_CASH_FLOW_TTM_CONCEPTS,
        as_of_date=as_of_date,
    )
    capex_raw, capex_meta = _companyfacts_priority_ttm(
        companyfacts,
        CAPEX_TTM_CONCEPTS,
        as_of_date=as_of_date,
    )
    if operating_cash_flow is None or capex_raw is None:
        return None, operating_cash_flow, capex_raw, None
    capex = abs(float(capex_raw))
    return (
        float(operating_cash_flow) - capex,
        float(operating_cash_flow),
        capex,
        {
            "operating_cash_flow_ttm": float(operating_cash_flow),
            "capex_ttm": capex,
            "capex_raw_value": float(capex_raw),
            "operating_cash_flow_meta": operating_cash_flow_meta,
            "capex_meta": capex_meta,
            "formula": "operating_cash_flow_ttm - abs(capex_ttm)",
        },
    )


