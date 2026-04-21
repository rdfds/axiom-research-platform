#!/usr/bin/env python3
"""Extract high-value exact SEC companyfacts components into a snapshot artifact.

Current scope:
- `liquidity.restricted_cash_sec_exact`
- `liquidity.marketable_securities_sec_exact`
- `liquidity.revolver_undrawn_sec_exact`
- `capital_structure.lease_liabilities_sec_exact`

These are component metrics intended to feed the smart-normalized layer. We are
deliberately conservative: only companyfacts concepts that directly represent
remaining / unused borrowing capacity are promoted into the exact revolver path.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


MAX_FACT_AGE_DAYS = 550
DEFAULT_LOCAL_COMPANYFACTS_ROOT = Path(__file__).resolve().parents[1] / "data" / "sec" / "companyfacts"
LEASE_EXACT_MAX_AGE_DAYS = 220
LEASE_STALE_CARRY_FORWARD_MAX_AGE_DAYS = 420
LEASE_ROU_FRESH_MAX_AGE_DAYS = 220
LEASE_COMPONENT_ALIGNMENT_MAX_GAP_DAYS = 10

# Keep the restricted-cash exact set narrow by default, but allow a small
# fallback set of direct "restricted cash and cash equivalents" concepts.
# We still exclude the broader cash-flow reconciliation total
# `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents`, which is not
# itself a restricted-cash balance.
RESTRICTED_CASH_EXACT_CONCEPTS = {
    "RestrictedCash",
    "RestrictedCashCurrent",
}
RESTRICTED_CASH_NONCURRENT_EXACT_CONCEPTS = {
    "RestrictedCashNoncurrent",
    "RestrictedCashAndCashEquivalentsNoncurrent",
}
RESTRICTED_CASH_MIXED_FALLBACK_CONCEPTS = {
    "RestrictedCashAndCashEquivalents",
    "RestrictedCashAndCashEquivalentsAtCarryingValue",
    "RestrictedCashAndInvestmentsCurrent",
}
MARKETABLE_SECURITY_EXACT_CONCEPTS = {
    "ShortTermInvestments",
    "MarketableSecurities",
    "AvailableForSaleSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    "AvailableForSaleDebtSecuritiesCurrent",
    "MarketableSecuritiesCurrent",
}
MARKETABLE_SECURITY_ANY_CONCEPTS = {
    "ShortTermInvestments",
    "MarketableSecurities",
    "MarketableSecuritiesCurrent",
    "MarketableSecuritiesNoncurrent",
    "AvailableForSaleSecurities",
    "AvailableForSaleSecuritiesCurrent",
    "AvailableForSaleSecuritiesNoncurrent",
    "AvailableForSaleSecuritiesDebtSecurities",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    "AvailableForSaleDebtSecuritiesCurrent",
}
RESTRICTED_CASH_ANY_CONCEPTS = (
    RESTRICTED_CASH_EXACT_CONCEPTS
    | RESTRICTED_CASH_NONCURRENT_EXACT_CONCEPTS
    | {
        "RestrictedCashAndCashEquivalents",
        "RestrictedCashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "RestrictedCashAndInvestmentsCurrent",
    }
)
REVOLVER_UNDRAWN_EXACT_CONCEPTS = [
    "LineOfCreditFacilityRemainingBorrowingCapacity",
    "DebtInstrumentUnusedBorrowingCapacityAmount",
]
RESTRICTED_CASH_TOTAL_RECONCILIATION_CONCEPT = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
OPERATING_LEASE_CURRENT_EXACT_CONCEPTS = {
    "OperatingLeaseLiabilityCurrent",
    "LesseeOperatingLeaseLiabilityCurrent",
}
FINANCE_LEASE_CURRENT_EXACT_CONCEPTS = {
    "FinanceLeaseLiabilityCurrent",
    "LesseeFinanceLeaseLiabilityCurrent",
}
OPERATING_LEASE_NONCURRENT_EXACT_CONCEPTS = {
    "OperatingLeaseLiabilityNoncurrent",
    "LesseeOperatingLeaseLiabilityNoncurrent",
}
FINANCE_LEASE_NONCURRENT_EXACT_CONCEPTS = {
    "FinanceLeaseLiabilityNoncurrent",
    "LesseeFinanceLeaseLiabilityNoncurrent",
}
OPERATING_LEASE_TOTAL_EXACT_CONCEPTS = {
    "OperatingLeaseLiability",
    "LesseeOperatingLeaseLiability",
}
FINANCE_LEASE_TOTAL_EXACT_CONCEPTS = {
    "FinanceLeaseLiability",
    "LesseeFinanceLeaseLiability",
}
LEASE_AGGREGATE_TOTAL_EXACT_CONCEPTS = {
    "LeaseLiabilities",
}
OPERATING_LEASE_RIGHT_OF_USE_ASSET_CONCEPTS = {
    "OperatingLeaseRightOfUseAsset",
    "LesseeOperatingLeaseRightOfUseAsset",
}
FINANCE_LEASE_RIGHT_OF_USE_ASSET_CONCEPTS = {
    "FinanceLeaseRightOfUseAsset",
    "LesseeFinanceLeaseRightOfUseAsset",
}
OPERATING_LEASE_PAYMENTS_DUE_CONCEPTS = {
    "OperatingLeaseLiabilityPaymentsDue",
    "LesseeOperatingLeaseLiabilityPaymentsDue",
}
OPERATING_LEASE_CURRENT_DUE_CONCEPTS = {
    "OperatingLeaseLiabilityPaymentsDueNextTwelveMonths",
    "LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths",
}
OPERATING_LEASE_UNDISCOUNTED_EXCESS_CONCEPTS = {
    "OperatingLeaseLiabilityUndiscountedExcessAmount",
    "LesseeOperatingLeaseLiabilityUndiscountedExcessAmount",
}
FINANCE_LEASE_PAYMENTS_DUE_CONCEPTS = {
    "FinanceLeaseLiabilityPaymentsDue",
    "LesseeFinanceLeaseLiabilityPaymentsDue",
}
FINANCE_LEASE_CURRENT_DUE_CONCEPTS = {
    "FinanceLeaseLiabilityPaymentsDueNextTwelveMonths",
    "LesseeFinanceLeaseLiabilityPaymentsDueNextTwelveMonths",
}
FINANCE_LEASE_UNDISCOUNTED_EXCESS_CONCEPTS = {
    "FinanceLeaseLiabilityUndiscountedExcessAmount",
    "LesseeFinanceLeaseLiabilityUndiscountedExcessAmount",
}
LEASE_CURRENT_EXACT_CONCEPTS = OPERATING_LEASE_CURRENT_EXACT_CONCEPTS | FINANCE_LEASE_CURRENT_EXACT_CONCEPTS
LEASE_NONCURRENT_EXACT_CONCEPTS = OPERATING_LEASE_NONCURRENT_EXACT_CONCEPTS | FINANCE_LEASE_NONCURRENT_EXACT_CONCEPTS
LEASE_TOTAL_EXACT_CONCEPTS = (
    OPERATING_LEASE_TOTAL_EXACT_CONCEPTS | FINANCE_LEASE_TOTAL_EXACT_CONCEPTS | LEASE_AGGREGATE_TOTAL_EXACT_CONCEPTS
)
LEASE_RIGHT_OF_USE_ASSET_CONCEPTS = OPERATING_LEASE_RIGHT_OF_USE_ASSET_CONCEPTS | FINANCE_LEASE_RIGHT_OF_USE_ASSET_CONCEPTS
OPERATING_LEASE_ANY_CONCEPTS = (
    OPERATING_LEASE_CURRENT_EXACT_CONCEPTS
    | OPERATING_LEASE_NONCURRENT_EXACT_CONCEPTS
    | OPERATING_LEASE_TOTAL_EXACT_CONCEPTS
    | OPERATING_LEASE_RIGHT_OF_USE_ASSET_CONCEPTS
    | OPERATING_LEASE_PAYMENTS_DUE_CONCEPTS
    | OPERATING_LEASE_CURRENT_DUE_CONCEPTS
    | OPERATING_LEASE_UNDISCOUNTED_EXCESS_CONCEPTS
)
FINANCE_LEASE_ANY_CONCEPTS = (
    FINANCE_LEASE_CURRENT_EXACT_CONCEPTS
    | FINANCE_LEASE_NONCURRENT_EXACT_CONCEPTS
    | FINANCE_LEASE_TOTAL_EXACT_CONCEPTS
    | FINANCE_LEASE_RIGHT_OF_USE_ASSET_CONCEPTS
    | FINANCE_LEASE_PAYMENTS_DUE_CONCEPTS
    | FINANCE_LEASE_CURRENT_DUE_CONCEPTS
    | FINANCE_LEASE_UNDISCOUNTED_EXCESS_CONCEPTS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", required=True, help="Input snapshot JSONL")
    parser.add_argument(
        "--companyfacts-root",
        help="Local folder with SEC companyfacts JSON files. Defaults to the local canonical companyfacts root when present.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_snapshot_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _feature_template(
    *,
    metric_name: str,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    support_mode: str,
    value: Any,
    unit: str,
    missing_reason: str | None,
    component_breakdown: Dict[str, Any] | None,
    quality_flags: list[str] | None,
) -> Dict[str, Any]:
    return {
        "name": metric_name,
        "value": value,
        "unit": unit,
        "computed_at": computed_at,
        "as_of_time": as_of_time,
        "window": None,
        "confidence": 1.0 if value is not None else None,
        "provenance": [
            {
                "artifact_type": "SecCompanyFacts",
                "artifact_id": f"sec_companyfacts:{Path(provenance_source).name}",
                "source": provenance_source,
                "published_at": as_of_time,
                "ingested_at": computed_at,
                "hash": None,
            }
        ],
        "missing_reason": missing_reason,
        "fallback_used": None,
        "metric_policy_id": None,
        "market_owner": None,
        "primary_source_basis": "sec_companyfacts",
        "methodology_registry_id": None,
        "methodology_metric_id": None,
        "canonical_owner_id": None,
        "canonical_owner_name": None,
        "canonical_classification": None,
        "market_layer_status": None,
        "current_alignment_status": None,
        "primary_source_document_id": None,
        "recommended_metric_name": None,
        "input_source_registry_id": None,
        "input_source_owner_id": None,
        "input_source_owner_name": None,
        "input_source_classification": "sec_companyfacts",
        "input_source_formula_basis": None,
        "input_source_alignment_status": "aligned",
        "input_source_document_ids": None,
        "definition_requirement": None,
        "definition_requirement_reason": None,
        "methodology_execution_decision": None,
        "methodology_execution_reason": None,
        "input_layer_bucket": "reference",
        "input_layer_bucket_reason": "sec_companyfacts",
        "strict_market_defined": None,
        "archetype": None,
        "sector": None,
        "subsector": None,
        "override_level_applied": None,
        "support_mode": support_mode,
        "applicability_status": None,
        "component_breakdown": component_breakdown,
        "quality_flags": quality_flags,
        "view_type": None,
    }


def _computed_metric_template(
    *,
    metric_name: str,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    support_mode: str,
    value: Any,
    unit: str,
    missing_reason: str | None,
    component_breakdown: Dict[str, Any] | None,
    quality_flags: list[str] | None,
) -> Dict[str, Any]:
    node = _feature_template(
        metric_name=metric_name,
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        support_mode=support_mode,
        value=value,
        unit=unit,
        missing_reason=missing_reason,
        component_breakdown=component_breakdown,
        quality_flags=quality_flags,
    )
    provenance = list(node.get("provenance") or [])
    if provenance:
        provenance[0]["artifact_type"] = "ComputedMetric"
        provenance[0]["artifact_id"] = f"computed_metric:{metric_name}"
    node["provenance"] = provenance
    node["primary_source_basis"] = "computed_metric"
    node["input_source_classification"] = "computed_metric"
    node["input_layer_bucket_reason"] = "computed_from_reference_metrics"
    return node


def _load_companyfacts(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _candidate_units_map(companyfacts: dict, concept_name: str) -> dict | None:
    for taxonomy in ("us-gaap", "dei", "ifrs-full"):
        facts = (companyfacts.get("facts") or {}).get(taxonomy) or {}
        if concept_name in facts:
            return facts[concept_name].get("units") or {}
    return None


def _latest_fact_value(companyfacts: dict, concept_name: str, as_of_date: str) -> tuple[float | None, dict[str, Any] | None]:
    units_map = _candidate_units_map(companyfacts, concept_name)
    if not units_map:
        return None, None
    as_of_dt = datetime.fromisoformat(as_of_date).date()
    candidates = []
    for unit, entries in units_map.items():
        if unit.upper() != "USD":
            continue
        for entry in entries:
            end = entry.get("end")
            filed = entry.get("filed")
            value = entry.get("val")
            if end is None or value is None:
                continue
            if end > as_of_date:
                continue
            if filed is not None and filed > as_of_date:
                continue
            try:
                end_dt = datetime.fromisoformat(end).date()
            except ValueError:
                continue
            if (as_of_dt - end_dt).days > MAX_FACT_AGE_DAYS:
                continue
            candidates.append((end, filed or "", entry, unit))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, chosen, unit = candidates[-1]
    meta = {
        "concept": concept_name,
        "end": chosen.get("end"),
        "filed": chosen.get("filed"),
        "fy": chosen.get("fy"),
        "fp": chosen.get("fp"),
        "frame": chosen.get("frame"),
        "form": chosen.get("form"),
        "unit": unit,
    }
    return float(chosen["val"]), meta


def _parse_iso_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(str(text)[:10])
    except ValueError:
        return None


def _statement_fact_node_is_fresh_enough(
    node: dict[str, Any] | None,
    as_of_date: str,
    *,
    max_age_days: int = 450,
) -> bool:
    if not node or node.get("support_mode") != "exact":
        return False
    component_breakdown = node['component_breakdown'] or {}
    end_dt = _parse_iso_date(component_breakdown.get("end")) or _parse_iso_date(component_breakdown.get("effective_at"))
    as_of_dt = _parse_iso_date(as_of_date)
    if end_dt is None or as_of_dt is None:
        return False
    return (as_of_dt - end_dt).days <= max_age_days


def _repair_cash_sti_from_statement_cash(
    *,
    cash_sti_node: dict[str, Any],
    cash_eq_node: dict[str, Any],
    marketable_node: dict[str, Any],
    companyfacts_path: Path,
    as_of_date: str,
    as_of_time: str,
    computed_at: str,
) -> dict[str, Any] | None:
    cash_eq_value = cash_eq_node.get("value")
    marketable_value = marketable_node.get("value")
    marketable_absent = marketable_node.get("missing_reason") == "sec_concept_absent"
    if cash_sti_node.get("support_mode") == "exact":
        return None
    if not _statement_fact_node_is_fresh_enough(cash_eq_node, as_of_date):
        return None
    if cash_eq_value is None:
        return None
    if not (
        (marketable_node.get("support_mode") == "exact" and marketable_value is not None)
        or marketable_absent
    ):
        return None
    repaired_value = float(cash_eq_value + (marketable_value or 0.0))
    repaired_node = dict(cash_sti_node)
    repaired_node["value"] = repaired_value
    repaired_node["unit"] = "usd"
    repaired_node["computed_at"] = computed_at
    repaired_node["confidence"] = 1.0
    repaired_node["missing_reason"] = None
    repaired_node["support_mode"] = "exact"
    repaired_node["primary_source_basis"] = (
        "statement_direct_plus_sec_companyfacts"
        if marketable_node.get("support_mode") == "exact"
        else "statement_direct_plus_zero_short_term_investments_inference"
    )
    repaired_node["input_source_classification"] = repaired_node["primary_source_basis"]
    repaired_node["input_layer_bucket_reason"] = "statement_cash_plus_sec_marketable"
    repaired_node["quality_flags"] = (
        None
        if marketable_node.get("support_mode") == "exact"
        else ["short_term_investments_absent_in_companyfacts"]
    )
    repaired_node["component_breakdown"] = {
        "mode": (
            "cash_and_equivalents_plus_marketable_securities"
            if marketable_node.get("support_mode") == "exact"
            else "cash_and_equivalents_plus_inferred_zero_short_term_investments"
        ),
        "cash_and_equivalents_statement_direct": cash_eq_node.get("component_breakdown"),
        "marketable_securities_sec_exact": marketable_node.get("component_breakdown"),
        "formula": (
            "cash_and_equivalents_statement_direct + marketable_securities_sec_exact"
            if marketable_node.get("support_mode") == "exact"
            else "cash_and_equivalents_statement_direct + 0_inferred_short_term_investments"
        ),
    }
    repaired_node["provenance"] = list(cash_eq_node.get("provenance") or [])
    if marketable_node.get("support_mode") == "exact":
        repaired_node["provenance"] += list(marketable_node.get("provenance") or [])
    else:
        repaired_node["provenance"].append(
            {
                "artifact_type": "SecCompanyFacts",
                "artifact_id": f"sec_companyfacts:{companyfacts_path.name}",
                "source": str(companyfacts_path),
                "published_at": as_of_time,
                "ingested_at": computed_at,
                "hash": None,
            }
        )
    return repaired_node


def _extract_exact_candidates(
    companyfacts: dict,
    as_of_date: str,
    concept_names: set[str],
) -> list[dict[str, Any]]:
    facts = ((companyfacts.get("facts") or {}).get("us-gaap") or {})
    candidates: list[dict[str, Any]] = []
    for concept in sorted(concept_names):
        if concept not in facts:
            continue
        value, meta = _latest_fact_value(companyfacts, concept, as_of_date)
        if value is None or meta is None:
            continue
        end_dt = _parse_iso_date(meta.get("end"))
        filed_dt = _parse_iso_date(meta.get("filed")) or end_dt
        if end_dt is None:
            continue
        candidates.append(
            {
                "value": float(value),
                "meta": meta,
                "end_dt": end_dt,
                "filed_dt": filed_dt or end_dt,
            }
        )
    candidates.sort(key=lambda item: (item["end_dt"], item["filed_dt"]), reverse=True)
    return candidates


def _select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    return candidates[0] if candidates else None


def _extract_candidate_for_end(
    companyfacts: dict,
    concept_names: set[str],
    target_end_dt: date | None,
    *,
    as_of_date: str,
) -> dict[str, Any] | None:
    if target_end_dt is None:
        return None
    facts = ((companyfacts.get("facts") or {}).get("us-gaap") or {})
    matches: list[dict[str, Any]] = []
    target_end = target_end_dt.isoformat()
    for concept in sorted(concept_names):
        concept_facts = facts.get(concept) or {}
        units_map = concept_facts.get("units") or {}
        for unit, entries in units_map.items():
            if unit.upper() != "USD":
                continue
            for entry in entries:
                end = entry.get("end")
                filed = entry.get("filed")
                value = entry.get("val")
                if end != target_end or value is None:
                    continue
                if filed is not None and filed > as_of_date:
                    continue
                end_dt = _parse_iso_date(end)
                filed_dt = _parse_iso_date(filed) or end_dt
                if end_dt is None or filed_dt is None:
                    continue
                matches.append(
                    {
                        "value": float(value),
                        "meta": {
                            "concept": concept,
                            "end": end,
                            "filed": filed,
                            "fy": entry.get("fy"),
                            "fp": entry.get("fp"),
                            "frame": entry.get("frame"),
                            "form": entry.get("form"),
                            "unit": unit,
                        },
                        "end_dt": end_dt,
                        "filed_dt": filed_dt,
                    }
                )
    if not matches:
        return None
    matches.sort(key=lambda item: (item["end_dt"], item["filed_dt"]), reverse=True)
    return matches[0]


