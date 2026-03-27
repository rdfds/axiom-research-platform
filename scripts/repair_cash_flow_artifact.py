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


def repair_market_fcf_yield(
    *,
    features: Dict[str, Any],
    companyfacts: Dict[str, Any] | None,
    companyfacts_path: Path,
    computed_at: str,
    as_of_time: str,
) -> bool:
    target = features.get("market.fcf_yield")
    if not target or target.get("value") is not None:
        return False

    market_cap_node = features.get("market.market_cap_provider_direct")
    market_cap = _node_value(market_cap_node)
    fcf_value, operating_cash_flow, capex, fcf_breakdown = _repairable_fcf_inputs(
        companyfacts=companyfacts,
        as_of_date=as_of_time[:10],
    )
    if market_cap in (None, 0) or fcf_value is None:
        return False

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = fcf_value / market_cap
    repaired["fallback_used"] = "sec_companyfacts_free_cash_flow_ttm"
    repaired["support_mode"] = "exact" if _node_support(market_cap_node) == "exact" else "proxy_missing_component"
    repaired["provenance"] = _union_provenance(market_cap_node) + _companyfacts_provenance(
        companyfacts_path,
        as_of_time=as_of_time,
        computed_at=computed_at,
    )
    repaired["component_breakdown"] = {
        "market_cap": market_cap,
        "free_cash_flow_ttm": fcf_value,
        "operating_cash_flow_ttm": operating_cash_flow,
        "capex_ttm": capex,
        "formula": "free_cash_flow_ttm / market_cap_provider_direct",
        "cash_flow_source": "sec_companyfacts",
        "cash_flow_context": fcf_breakdown,
    }
    repaired["quality_flags"] = None
    features["market.fcf_yield"] = repaired
    return True


def repair_operating_fcf_conversion(
    *,
    features: Dict[str, Any],
    companyfacts: Dict[str, Any] | None,
    companyfacts_path: Path,
    computed_at: str,
    as_of_time: str,
) -> bool:
    target = features.get("operating.fcf_conversion")
    if not target or target.get("value") is not None:
        return False

    ebitda_node = features.get("operating.ebitda_ltm_provider_direct")
    normalized_node = features.get("operating.operating_earnings_normalized")
    denominator = _node_value(ebitda_node)
    denominator_source = "operating.ebitda_ltm_provider_direct"
    fallback_used = "sec_companyfacts_fcf_plus_provider_ebitda"
    denominator_node = ebitda_node
    if denominator in (None, 0):
        denominator = _node_value(normalized_node)
        denominator_source = "operating.operating_earnings_normalized"
        fallback_used = "sec_companyfacts_fcf_plus_normalized_operating_earnings"
        denominator_node = normalized_node

    fcf_value, operating_cash_flow, capex, fcf_breakdown = _repairable_fcf_inputs(
        companyfacts=companyfacts,
        as_of_date=as_of_time[:10],
    )
    if denominator in (None, 0) or fcf_value is None:
        return False

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = fcf_value / denominator
    repaired["fallback_used"] = fallback_used
    repaired["support_mode"] = (
        "exact"
        if _node_support(denominator_node) == "exact"
        else "proxy_missing_component"
    )
    repaired["provenance"] = _union_provenance(ebitda_node, normalized_node) + _companyfacts_provenance(
        companyfacts_path,
        as_of_time=as_of_time,
        computed_at=computed_at,
    )
    repaired["component_breakdown"] = {
        "free_cash_flow_ttm": fcf_value,
        "operating_cash_flow_ttm": operating_cash_flow,
        "capex_ttm": capex,
        "ebitda": denominator,
        "ebitda_source_metric": denominator_source,
        "formula": "free_cash_flow_ttm / ebitda",
        "cash_flow_source": "sec_companyfacts",
        "cash_flow_context": fcf_breakdown,
    }
    repaired["quality_flags"] = None
    features["operating.fcf_conversion"] = repaired
    return True


def build_summary(path: Path) -> Dict[str, Dict[str, int]]:
    counters: Dict[str, Counter[str]] = {metric: Counter() for metric in REPAIR_METRICS}
    for row in iter_rows(path):
        features = row.get("features") or {}
        for metric in REPAIR_METRICS:
            node = features.get(metric) or {}
            mode = str(node.get("support_mode") or "unsupported")
            if node.get("value") is None:
                mode = "unsupported"
            counters[metric][mode] += 1
    summary: Dict[str, Dict[str, int]] = {}
    for metric, counter in counters.items():
        summary[metric] = {
            "exact": counter["exact"],
            "proxy_missing_component": counter["proxy_missing_component"],
            "unsupported": counter["unsupported"],
        }
    return summary


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    companyfacts_root = Path(args.companyfacts_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    computed_at = _now_iso()

    with out_path.open("w") as out_handle:
        for row in iter_rows(artifact_path):
            entity_id = str(row.get("company_id") or "")
            companyfacts_path = companyfacts_root / f"CIK{entity_id}.json"
            companyfacts = _load_companyfacts(companyfacts_path)
            features = row.get("features") or {}
            repair_market_fcf_yield(
                features=features,
                companyfacts=companyfacts,
                companyfacts_path=companyfacts_path,
                computed_at=computed_at,
                as_of_time=str(row.get("as_of_time") or ""),
            )
            repair_operating_fcf_conversion(
                features=features,
                companyfacts=companyfacts,
                companyfacts_path=companyfacts_path,
                computed_at=computed_at,
                as_of_time=str(row.get("as_of_time") or ""),
            )
            out_handle.write(json.dumps(row) + "\n")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.write_text(json.dumps(build_summary(out_path), indent=2))

    print(f"Repaired cash-flow metrics -> {out_path}")


if __name__ == "__main__":
    main()
