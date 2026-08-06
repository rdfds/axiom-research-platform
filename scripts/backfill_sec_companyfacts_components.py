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
    component_breakdown = node.get("component_breakdown") or {}
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


def _select_candidate_for_end_date(
    candidates: list[dict[str, Any]],
    end_dt: date | None,
) -> dict[str, Any] | None:
    if end_dt is None:
        return None
    same_period = [candidate for candidate in candidates if candidate.get("end_dt") == end_dt]
    if not same_period:
        return None
    same_period.sort(key=lambda item: (item["filed_dt"], item["end_dt"]), reverse=True)
    return same_period[0]


def _select_aligned_candidate_pair(
    left_candidates: list[dict[str, Any]],
    right_candidates: list[dict[str, Any]],
    *,
    max_gap_days: int = LEASE_COMPONENT_ALIGNMENT_MAX_GAP_DAYS,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    best_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_key = None
    for left in left_candidates:
        for right in right_candidates:
            gap_days = abs((left["end_dt"] - right["end_dt"]).days)
            if gap_days > max_gap_days:
                continue
            pair_key = (
                max(left["end_dt"], right["end_dt"]),
                max(left["filed_dt"], right["filed_dt"]),
                -gap_days,
            )
            if best_key is None or pair_key > best_key:
                best_key = pair_key
                best_pair = (left, right)
    if best_pair is None:
        return None, None
    return best_pair


def _candidate_is_fresh(candidate: dict[str, Any] | None, as_of_date: str, *, max_age_days: int = LEASE_EXACT_MAX_AGE_DAYS) -> bool:
    if candidate is None:
        return False
    end_dt = candidate.get("end_dt")
    as_of_dt = _parse_iso_date(as_of_date)
    if end_dt is None or as_of_dt is None:
        return False
    return (as_of_dt - end_dt).days <= max_age_days


def _candidate_is_stale_corroborated_by_fresh_rou(
    candidate: dict[str, Any] | None,
    rou_asset: dict[str, Any] | None,
    as_of_date: str,
) -> bool:
    if candidate is None or rou_asset is None:
        return False
    candidate_end_dt = candidate.get("end_dt")
    rou_end_dt = rou_asset.get("end_dt")
    as_of_dt = _parse_iso_date(as_of_date)
    if candidate_end_dt is None or rou_end_dt is None or as_of_dt is None:
        return False
    candidate_age_days = (as_of_dt - candidate_end_dt).days
    rou_age_days = (as_of_dt - rou_end_dt).days
    if candidate_age_days <= LEASE_EXACT_MAX_AGE_DAYS:
        return False
    if candidate_age_days > LEASE_STALE_CARRY_FORWARD_MAX_AGE_DAYS:
        return False
    if rou_age_days < 0 or rou_age_days > LEASE_ROU_FRESH_MAX_AGE_DAYS:
        return False
    return True


def _candidate_is_stale_within_carry_forward_window(
    candidate: dict[str, Any] | None,
    as_of_date: str,
) -> bool:
    if candidate is None:
        return False
    candidate_end_dt = candidate.get("end_dt")
    as_of_dt = _parse_iso_date(as_of_date)
    if candidate_end_dt is None or as_of_dt is None:
        return False
    candidate_age_days = (as_of_dt - candidate_end_dt).days
    return LEASE_EXACT_MAX_AGE_DAYS < candidate_age_days <= LEASE_STALE_CARRY_FORWARD_MAX_AGE_DAYS


def _reference_contains_fresh_lease_fact(reference: Any, as_of_date: str) -> bool:
    as_of_dt = _parse_iso_date(as_of_date)
    if as_of_dt is None:
        return False

    if isinstance(reference, dict):
        end_value = reference.get("end")
        if isinstance(end_value, str):
            end_dt = _parse_iso_date(end_value)
            if end_dt is not None:
                age_days = (as_of_dt - end_dt).days
                if 0 <= age_days <= LEASE_EXACT_MAX_AGE_DAYS:
                    return True
        for nested in reference.values():
            if _reference_contains_fresh_lease_fact(nested, as_of_date):
                return True
        return False

    if isinstance(reference, list):
        return any(_reference_contains_fresh_lease_fact(item, as_of_date) for item in reference)

    return False


def _passes_lease_plausibility(
    value: float,
    *,
    rou_value: float | None,
    reference_value: float | None,
) -> bool:
    if value < 0:
        return False
    if reference_value is not None and reference_value > 0:
        if value < reference_value * 0.5 or value > reference_value * 2.0:
            return False
    if rou_value is not None and rou_value > 0:
        if value < rou_value * 0.1 or value > rou_value * 10.0:
            return False
    return True


def _values_are_lease_corroborative(left: float | None, right: float | None, *, tolerance: float = 0.10) -> bool:
    if left is None or right is None:
        return False
    left = float(left)
    right = float(right)
    if left <= 0 or right <= 0:
        return False
    return abs(left - right) / max(abs(left), abs(right)) <= tolerance


def _find_first_matching_concept(companyfacts: dict, matcher) -> str | None:
    facts = ((companyfacts.get("facts") or {}).get("us-gaap") or {})
    for concept_name in facts:
        if matcher(concept_name):
            return concept_name
    return None


def _extract_exact_concepts(
    companyfacts: dict,
    as_of_date: str,
    concept_names: set[str],
) -> list[tuple[float, dict[str, Any]]]:
    return [(candidate["value"], candidate["meta"]) for candidate in _extract_exact_candidates(companyfacts, as_of_date, concept_names)]


def _has_any_us_gaap_concepts(companyfacts: dict, concept_names: set[str] | list[str]) -> bool:
    facts = ((companyfacts.get("facts") or {}).get("us-gaap") or {})
    return any(concept in facts for concept in concept_names)


def _metric_value(node: dict[str, Any] | None) -> float | None:
    if not node:
        return None
    value = node.get("value")
    if value is None:
        return None
    return float(value)


def _metric_support(node: dict[str, Any] | None) -> str:
    if not node:
        return "unsupported"
    return str(node.get("support_mode") or "unsupported")


def _cash_sti_proxy_represents_cash_only(
    *,
    cash_sti_node: dict[str, Any],
    marketable_node: dict[str, Any],
) -> bool:
    component_breakdown = cash_sti_node.get("component_breakdown") or {}
    return (
        _metric_support(cash_sti_node) == "proxy_missing_component"
        and cash_sti_node.get("missing_reason") == "cash_or_sti_component_missing"
        and _metric_value(cash_sti_node) is not None
        and component_breakdown.get("mode") == "partial_cash_stack"
        and component_breakdown.get("cash") is not None
        and component_breakdown.get("short_term_investments") is None
        and _metric_support(marketable_node) == "unsupported"
        and marketable_node.get("missing_reason") == "sec_concept_absent"
    )


def _repair_restricted_cash_from_total_cash_reconciliation(
    *,
    restricted_node: dict[str, Any],
    cash_eq_node: dict[str, Any],
    cash_sti_node: dict[str, Any] | None,
    marketable_node: dict[str, Any] | None,
    companyfacts: dict | None,
    companyfacts_path: Path,
    as_of_date: str,
    as_of_time: str,
    computed_at: str,
) -> dict[str, Any] | None:
    if _metric_support(restricted_node) == "exact":
        return None
    if companyfacts is None:
        return None
    cash_basis_value = None
    cash_basis_breakdown = None
    formula_suffix = None
    primary_source_basis = None
    provenance = None
    input_layer_bucket_reason = None

    if _metric_support(cash_eq_node) == "exact":
        cash_eq_value = _metric_value(cash_eq_node)
        if cash_eq_value is None:
            return None
        cash_basis_value = float(cash_eq_value)
        cash_basis_breakdown = cash_eq_node.get("component_breakdown")
        formula_suffix = "cash_and_equivalents_statement_direct"
        primary_source_basis = "statement_direct_plus_sec_companyfacts"
        provenance = list(cash_eq_node.get("provenance") or [])
        input_layer_bucket_reason = "restricted_cash_from_total_cash_reconciliation"
    elif (
        cash_sti_node is not None
        and marketable_node is not None
        and _cash_sti_proxy_represents_cash_only(
            cash_sti_node=cash_sti_node,
            marketable_node=marketable_node,
        )
    ):
        cash_sti_value = _metric_value(cash_sti_node)
        if cash_sti_value is None:
            return None
        cash_basis_value = float(cash_sti_value)
        cash_basis_breakdown = cash_sti_node.get("component_breakdown")
        formula_suffix = "cash_and_short_term_investments_provider_direct_cash_only_proxy"
        primary_source_basis = "provider_direct_plus_sec_companyfacts"
        provenance = list(cash_sti_node.get("provenance") or [])
        input_layer_bucket_reason = "restricted_cash_from_grouped_cash_total_reconciliation"
    else:
        return None
    total_cash_restricted, total_meta = _latest_fact_value(
        companyfacts,
        RESTRICTED_CASH_TOTAL_RECONCILIATION_CONCEPT,
        as_of_date,
    )
    if total_cash_restricted is None or total_meta is None:
        return None
    derived_value = float(total_cash_restricted) - cash_basis_value
    if derived_value < -1.0:
        return None
    floored_small_negative = derived_value < 0.0
    if floored_small_negative:
        derived_value = 0.0
    node = dict(restricted_node)
    node["value"] = float(derived_value)
    node["unit"] = "usd"
    node["computed_at"] = computed_at
    node["confidence"] = 1.0
    node["missing_reason"] = None
    node["support_mode"] = "exact"
    node["primary_source_basis"] = primary_source_basis
    node["input_source_classification"] = node["primary_source_basis"]
    node["input_layer_bucket_reason"] = input_layer_bucket_reason
    node["quality_flags"] = (
        ["rounded_small_negative_reconciliation_gap"]
        if floored_small_negative
        else None
    )
    node["component_breakdown"] = {
        "mode": (
            "cash_plus_restricted_total_minus_cash_equivalents"
            if formula_suffix == "cash_and_equivalents_statement_direct"
            else "cash_plus_restricted_total_minus_grouped_cash_cash_only_proxy"
        ),
        "cash_cash_equivalents_restricted_cash_total": total_meta,
        formula_suffix: cash_basis_breakdown,
        "formula": (
            f"{RESTRICTED_CASH_TOTAL_RECONCILIATION_CONCEPT} - "
            f"{formula_suffix}"
        ),
    }
    node["provenance"] = provenance
    node["provenance"].append(
        {
            "artifact_type": "SecCompanyFacts",
            "artifact_id": f"sec_companyfacts:{companyfacts_path.name}",
            "source": str(companyfacts_path),
            "published_at": as_of_time,
            "ingested_at": computed_at,
            "hash": None,
        }
    )
    return node


def _build_combo_metric(
    *,
    metric_name: str,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    unit: str,
    numerator: float | None = None,
    denominator: float | None = None,
    extra_components: Dict[str, Any] | None = None,
    component_supports: Dict[str, str] | None = None,
    formula: str,
    allow_numerator_only: bool = False,
) -> Dict[str, Any]:
    components = dict(extra_components or {})
    components["formula"] = formula
    non_exact_components = sorted(
        component_name
        for component_name, support_mode in (component_supports or {}).items()
        if support_mode != "exact"
    )
    if numerator is None:
        return _computed_metric_template(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            support_mode="unsupported",
            value=None,
            unit=unit,
            missing_reason="component_unavailable",
            component_breakdown=components,
            quality_flags=["component_unavailable"],
        )
    if denominator is None:
        if not allow_numerator_only:
            return _computed_metric_template(
                metric_name=metric_name,
                as_of_time=as_of_time,
                computed_at=computed_at,
                provenance_source=provenance_source,
                support_mode="unsupported",
                value=None,
                unit=unit,
                missing_reason="component_unavailable",
                component_breakdown=components,
                quality_flags=["component_unavailable"],
            )
        support_mode = "exact" if not non_exact_components else "proxy_missing_component"
        quality_flags = None if not non_exact_components else [f"component_not_exact:{name}" for name in non_exact_components]
        return _computed_metric_template(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            support_mode=support_mode,
            value=float(numerator),
            unit=unit,
            missing_reason=None if support_mode == "exact" else "component_not_exact",
            component_breakdown=components,
            quality_flags=quality_flags,
        )
    if denominator <= 0:
        return _computed_metric_template(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            support_mode="unsupported",
            value=None,
            unit=unit,
            missing_reason="non_positive_denominator",
            component_breakdown=components,
            quality_flags=["non_positive_denominator"],
        )
    support_mode = "exact" if not non_exact_components else "proxy_missing_component"
    quality_flags = None if not non_exact_components else [f"component_not_exact:{name}" for name in non_exact_components]
    return _computed_metric_template(
        metric_name=metric_name,
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        support_mode=support_mode,
        value=float(numerator) / float(denominator),
        unit=unit,
        missing_reason=None if support_mode == "exact" else "component_not_exact",
        component_breakdown=components,
        quality_flags=quality_flags,
    )


def _extract_restricted_cash(companyfacts: dict, as_of_date: str) -> tuple[float | None, dict[str, Any] | None]:
    current_candidates = _extract_exact_candidates(companyfacts, as_of_date, RESTRICTED_CASH_EXACT_CONCEPTS)
    noncurrent_candidates = _extract_exact_candidates(companyfacts, as_of_date, RESTRICTED_CASH_NONCURRENT_EXACT_CONCEPTS)
    current_match = _select_best_candidate(current_candidates)
    noncurrent_match = _select_best_candidate(noncurrent_candidates)

    if current_match is not None and noncurrent_match is not None:
        return (
            float(current_match["value"] + noncurrent_match["value"]),
            {
                "mode": "sum_current_noncurrent",
                "current_components": [current_match["meta"]],
                "noncurrent_components": [noncurrent_match["meta"]],
            },
        )

    if current_match is not None:
        return float(current_match["value"]), current_match["meta"]

    if noncurrent_match is not None:
        return float(noncurrent_match["value"]), noncurrent_match["meta"]

    fallback_candidates = _extract_exact_candidates(companyfacts, as_of_date, RESTRICTED_CASH_MIXED_FALLBACK_CONCEPTS)
    fallback = _select_best_candidate(fallback_candidates)
    if fallback is not None:
        return (
            float(fallback["value"]),
            {
                "mode": "mixed_total_restricted_cash_fallback",
                "chosen": fallback["meta"],
                "alternatives": [candidate["meta"] for candidate in fallback_candidates[1:]],
            },
        )

    return None, None


def _extract_marketable_securities(companyfacts: dict, as_of_date: str) -> tuple[float | None, dict[str, Any] | None]:
    facts = ((companyfacts.get("facts") or {}).get("us-gaap") or {})
    for concept in MARKETABLE_SECURITY_EXACT_CONCEPTS:
        if concept in facts:
            value, meta = _latest_fact_value(companyfacts, concept, as_of_date)
            if value is not None:
                return value, meta
    return None, None


def _extract_revolver_undrawn(companyfacts: dict, as_of_date: str) -> tuple[float | None, dict[str, Any] | None]:
    facts = ((companyfacts.get("facts") or {}).get("us-gaap") or {})
    references: list[dict[str, Any]] = []
    for concept in REVOLVER_UNDRAWN_EXACT_CONCEPTS:
        if concept not in facts:
            continue
        value, meta = _latest_fact_value(companyfacts, concept, as_of_date)
        if value is None or meta is None:
            continue
        if concept == "LineOfCreditFacilityRemainingBorrowingCapacity":
            if references:
                return value, {
                    "mode": "preferred_remaining_borrowing_capacity",
                    "chosen": meta,
                    "alternatives": references,
                }
            return value, meta
        references.append(meta | {"value": float(value)})
    if references:
        chosen = references[0].copy()
        value = float(chosen.pop("value"))
        if len(references) > 1:
            return value, {
                "mode": "preferred_unused_borrowing_capacity",
                "chosen": chosen,
                "alternatives": references[1:],
            }
        return value, chosen
    return None, None


def _build_lease_class_candidate(
    *,
    companyfacts: dict,
    as_of_date: str,
    label: str,
    total_concepts: set[str],
    current_concepts: set[str],
    noncurrent_concepts: set[str],
    payments_due_concepts: set[str],
    current_due_concepts: set[str],
    undiscounted_excess_concepts: set[str],
    rou_asset_concepts: set[str],
) -> dict[str, Any]:
    total_candidates = _extract_exact_candidates(companyfacts, as_of_date, total_concepts)
    current_candidates = _extract_exact_candidates(companyfacts, as_of_date, current_concepts)
    noncurrent_candidates = _extract_exact_candidates(companyfacts, as_of_date, noncurrent_concepts)
    payments_due_candidates = _extract_exact_candidates(companyfacts, as_of_date, payments_due_concepts)
    current_due_candidates = _extract_exact_candidates(companyfacts, as_of_date, current_due_concepts)
    undiscounted_excess_candidates = _extract_exact_candidates(companyfacts, as_of_date, undiscounted_excess_concepts)
    rou_asset_candidates = _extract_exact_candidates(companyfacts, as_of_date, rou_asset_concepts)

    direct_total = _select_best_candidate(total_candidates)
    freshest_current_component = _select_best_candidate(current_candidates)
    freshest_noncurrent_component = _select_best_candidate(noncurrent_candidates)
    current_component, noncurrent_component = _select_aligned_candidate_pair(current_candidates, noncurrent_candidates)
    payments_due_component, undiscounted_excess_component = _select_aligned_candidate_pair(
        payments_due_candidates,
        undiscounted_excess_candidates,
    )
    rou_asset = _select_best_candidate(rou_asset_candidates)
    rou_value = None if rou_asset is None else float(rou_asset["value"])

    direct_total_value = None if direct_total is None else float(direct_total["value"])
    pair_total_value = (
        None
        if current_component is None or noncurrent_component is None
        else float(current_component["value"]) + float(noncurrent_component["value"])
    )
    derived_total_value = (
        None
        if payments_due_component is None or undiscounted_excess_component is None
        else float(payments_due_component["value"]) - float(undiscounted_excess_component["value"])
    )
    mixed_pair_total_value = (
        None
        if freshest_current_component is None or freshest_noncurrent_component is None
        else float(freshest_current_component["value"]) + float(freshest_noncurrent_component["value"])
    )
    fresh_direct_total_value = (
        direct_total_value
        if direct_total is not None and _candidate_is_fresh(direct_total, as_of_date)
        else None
    )
    fresh_pair_total_value = (
        pair_total_value
        if current_component is not None
        and noncurrent_component is not None
        and _candidate_is_fresh(current_component, as_of_date)
        and _candidate_is_fresh(noncurrent_component, as_of_date)
        else None
    )
    fresh_derived_total_value = (
        derived_total_value
        if payments_due_component is not None
        and undiscounted_excess_component is not None
        and _candidate_is_fresh(payments_due_component, as_of_date)
        and _candidate_is_fresh(undiscounted_excess_component, as_of_date)
        else None
    )
    pair_reference_value = (
        fresh_direct_total_value
        if fresh_direct_total_value is not None
        else fresh_derived_total_value
    )
    direct_reference_value = (
        fresh_pair_total_value
        if fresh_pair_total_value is not None
        else fresh_derived_total_value
    )
    derived_reference_value = (
        fresh_pair_total_value
        if fresh_pair_total_value is not None
        else fresh_direct_total_value
    )

    candidates: list[dict[str, Any]] = []

    def _candidate_meta_with_stale_support(meta: dict[str, Any]) -> dict[str, Any]:
        return {
            **meta,
            "support_override": "stale_liability_total_corroborated_by_fresh_rou_asset",
        }

    def _candidate_meta_with_mixed_support(meta: dict[str, Any], *, stale_component: str) -> dict[str, Any]:
        return {
            **meta,
            "support_override": "mixed_fresh_and_stale_components_corroborated_by_fresh_rou_asset",
            "stale_component": stale_component,
        }

    freshest_current_end_dt = None if freshest_current_component is None else freshest_current_component["end_dt"]
    stale_period_direct_total = _extract_candidate_for_end(
        companyfacts,
        total_concepts,
        freshest_current_end_dt,
        as_of_date=as_of_date,
    )
    stale_period_rou_asset = _extract_candidate_for_end(
        companyfacts,
        rou_asset_concepts,
        freshest_current_end_dt,
        as_of_date=as_of_date,
    )
    stale_period_current_due = _extract_candidate_for_end(
        companyfacts,
        current_due_concepts,
        freshest_current_end_dt,
        as_of_date=as_of_date,
    )

    if (
        current_component is not None
        and noncurrent_component is not None
        and _candidate_is_fresh(current_component, as_of_date)
        and _candidate_is_fresh(noncurrent_component, as_of_date)
        and _passes_lease_plausibility(
            float(pair_total_value),
            rou_value=rou_value,
            reference_value=pair_reference_value,
        )
    ):
        candidates.append(
            {
                "value": float(pair_total_value),
                "meta": {
                    "mode": f"{label}_sum_current_noncurrent",
                    "current_components": [current_component["meta"]],
                    "noncurrent_components": [noncurrent_component["meta"]],
                    "class": label,
                },
                "end_dt": max(current_component["end_dt"], noncurrent_component["end_dt"]),
                "filed_dt": max(current_component["filed_dt"], noncurrent_component["filed_dt"]),
                "priority": 3,
            }
        )
    elif (
        current_component is not None
        and noncurrent_component is not None
        and _candidate_is_stale_corroborated_by_fresh_rou(current_component, rou_asset, as_of_date)
        and _candidate_is_stale_corroborated_by_fresh_rou(noncurrent_component, rou_asset, as_of_date)
        and _passes_lease_plausibility(
            float(pair_total_value),
            rou_value=rou_value,
            reference_value=direct_total_value if direct_total_value is not None else derived_total_value,
        )
    ):
        candidates.append(
            {
                "value": float(pair_total_value),
                "meta": _candidate_meta_with_stale_support(
                    {
                        "mode": f"{label}_sum_current_noncurrent",
                        "current_components": [current_component["meta"]],
                        "noncurrent_components": [noncurrent_component["meta"]],
                        "class": label,
                    }
                ),
                "end_dt": max(current_component["end_dt"], noncurrent_component["end_dt"]),
                "filed_dt": max(current_component["filed_dt"], noncurrent_component["filed_dt"]),
                "priority": 0,
            }
        )
    elif (
        freshest_current_component is not None
        and freshest_noncurrent_component is not None
        and mixed_pair_total_value is not None
        and _candidate_is_stale_corroborated_by_fresh_rou(freshest_current_component, rou_asset, as_of_date)
        and _candidate_is_fresh(freshest_noncurrent_component, as_of_date)
        and _passes_lease_plausibility(
            float(mixed_pair_total_value),
            rou_value=rou_value,
            reference_value=pair_reference_value,
        )
    ):
        candidates.append(
            {
                "value": float(mixed_pair_total_value),
                "meta": _candidate_meta_with_mixed_support(
                    {
                        "mode": f"{label}_sum_current_noncurrent",
                        "current_components": [freshest_current_component["meta"]],
                        "noncurrent_components": [freshest_noncurrent_component["meta"]],
                        "class": label,
                    },
                    stale_component="current",
                ),
                "end_dt": max(freshest_current_component["end_dt"], freshest_noncurrent_component["end_dt"]),
                "filed_dt": max(freshest_current_component["filed_dt"], freshest_noncurrent_component["filed_dt"]),
                "priority": 1,
            }
        )

        if (
            stale_period_direct_total is not None
            and stale_period_rou_asset is not None
            and rou_value is not None
        ):
            stale_basis_delta = float(stale_period_direct_total["value"]) - float(stale_period_rou_asset["value"])
            rebased_current_value = rou_value + stale_basis_delta - float(freshest_noncurrent_component["value"])
            if stale_period_current_due is not None:
                rebased_current_value = min(rebased_current_value, float(stale_period_current_due["value"]))
            stale_current_value = float(freshest_current_component["value"])
            if (
                stale_current_value > 0
                and rebased_current_value > 0
                and abs(rebased_current_value - stale_current_value) / stale_current_value <= 0.10
            ):
                rebased_total_value = rebased_current_value + float(freshest_noncurrent_component["value"])
                if _passes_lease_plausibility(
                    float(rebased_total_value),
                    rou_value=rou_value,
                    reference_value=pair_reference_value,
                ):
                    rebased_meta = {
                        "mode": f"{label}_sum_current_noncurrent",
                        "current_components": [freshest_current_component["meta"]],
                        "noncurrent_components": [freshest_noncurrent_component["meta"]],
                        "class": label,
                        "support_override": "mixed_fresh_and_stale_components_rebased_by_rou_basis_delta",
                        "stale_component": "current",
                        "stale_basis_delta": stale_basis_delta,
                        "stale_basis_total_component": stale_period_direct_total["meta"],
                        "stale_basis_rou_component": stale_period_rou_asset["meta"],
                    }
                    if stale_period_current_due is not None:
                        rebased_meta["stale_current_due_cap_component"] = stale_period_current_due["meta"]
                    candidates.append(
                        {
                            "value": float(rebased_total_value),
                            "meta": rebased_meta,
                            "end_dt": max(
                                freshest_current_component["end_dt"],
                                freshest_noncurrent_component["end_dt"],
                                rou_asset["end_dt"],
                            ),
                            "filed_dt": max(
                                freshest_current_component["filed_dt"],
                                freshest_noncurrent_component["filed_dt"],
                                rou_asset["filed_dt"],
                            ),
                            "priority": 2,
                        }
                    )

    if (
        direct_total is not None
        and _candidate_is_fresh(direct_total, as_of_date)
        and _passes_lease_plausibility(
            float(direct_total_value),
            rou_value=rou_value,
            reference_value=direct_reference_value,
        )
    ):
        candidates.append(
            {
                "value": float(direct_total_value),
                "meta": {
                    "mode": f"{label}_direct_total",
                    "components": [direct_total["meta"]],
                    "class": label,
                },
                "end_dt": direct_total["end_dt"],
                "filed_dt": direct_total["filed_dt"],
                "priority": 2,
            }
        )
    elif (
        direct_total is not None
        and _candidate_is_stale_corroborated_by_fresh_rou(direct_total, rou_asset, as_of_date)
        and _passes_lease_plausibility(
            float(direct_total_value),
            rou_value=rou_value,
            reference_value=pair_total_value if pair_total_value is not None else derived_total_value,
        )
    ):
        candidates.append(
            {
                "value": float(direct_total_value),
                "meta": _candidate_meta_with_stale_support(
                    {
                        "mode": f"{label}_direct_total",
                        "components": [direct_total["meta"]],
                        "class": label,
                    }
                ),
                "end_dt": direct_total["end_dt"],
                "filed_dt": direct_total["filed_dt"],
                "priority": 0,
            }
        )

    if (
        payments_due_component is not None
        and undiscounted_excess_component is not None
        and derived_total_value is not None
        and _candidate_is_fresh(payments_due_component, as_of_date)
        and _candidate_is_fresh(undiscounted_excess_component, as_of_date)
        and _passes_lease_plausibility(
            float(derived_total_value),
            rou_value=rou_value,
            reference_value=derived_reference_value,
        )
    ):
        candidates.append(
            {
                "value": float(derived_total_value),
                "meta": {
                    "mode": f"{label}_payments_due_minus_undiscounted_excess",
                    "payments_due_component": payments_due_component["meta"],
                    "undiscounted_excess_component": undiscounted_excess_component["meta"],
                    "class": label,
                    "formula": "payments_due - undiscounted_excess_amount",
                },
                "end_dt": max(payments_due_component["end_dt"], undiscounted_excess_component["end_dt"]),
                "filed_dt": max(payments_due_component["filed_dt"], undiscounted_excess_component["filed_dt"]),
                "priority": 1,
            }
        )
    elif (
        payments_due_component is not None
        and undiscounted_excess_component is not None
        and derived_total_value is not None
        and _candidate_is_stale_corroborated_by_fresh_rou(payments_due_component, rou_asset, as_of_date)
        and _candidate_is_stale_corroborated_by_fresh_rou(undiscounted_excess_component, rou_asset, as_of_date)
        and _passes_lease_plausibility(
            float(derived_total_value),
            rou_value=rou_value,
            reference_value=pair_total_value if pair_total_value is not None else direct_total_value,
        )
    ):
        candidates.append(
            {
                "value": float(derived_total_value),
                "meta": _candidate_meta_with_stale_support(
                    {
                        "mode": f"{label}_payments_due_minus_undiscounted_excess",
                        "payments_due_component": payments_due_component["meta"],
                        "undiscounted_excess_component": undiscounted_excess_component["meta"],
                        "class": label,
                        "formula": "payments_due - undiscounted_excess_amount",
                    }
                ),
                "end_dt": max(payments_due_component["end_dt"], undiscounted_excess_component["end_dt"]),
                "filed_dt": max(payments_due_component["filed_dt"], undiscounted_excess_component["filed_dt"]),
                "priority": 0,
            }
        )

    has_any_fresh_class_support = any(
        _candidate_is_fresh(candidate, as_of_date)
        for candidate in (
            direct_total,
            current_component,
            noncurrent_component,
            payments_due_component,
            undiscounted_excess_component,
            rou_asset,
        )
    )
    stale_direct_is_eligible = _candidate_is_stale_within_carry_forward_window(direct_total, as_of_date)
    stale_pair_is_eligible = (
        current_component is not None
        and noncurrent_component is not None
        and _candidate_is_stale_within_carry_forward_window(current_component, as_of_date)
        and _candidate_is_stale_within_carry_forward_window(noncurrent_component, as_of_date)
    )
    stale_derived_is_eligible = (
        payments_due_component is not None
        and undiscounted_excess_component is not None
        and _candidate_is_stale_within_carry_forward_window(payments_due_component, as_of_date)
        and _candidate_is_stale_within_carry_forward_window(undiscounted_excess_component, as_of_date)
    )
    stale_direct_corroborated = stale_direct_is_eligible and (
        _values_are_lease_corroborative(direct_total_value, pair_total_value)
        or _values_are_lease_corroborative(direct_total_value, derived_total_value)
    )
    stale_pair_corroborated = stale_pair_is_eligible and (
        _values_are_lease_corroborative(pair_total_value, direct_total_value)
        or _values_are_lease_corroborative(pair_total_value, derived_total_value)
    )
    stale_derived_corroborated = stale_derived_is_eligible and (
        _values_are_lease_corroborative(derived_total_value, direct_total_value)
        or _values_are_lease_corroborative(derived_total_value, pair_total_value)
    )

    def _candidate_meta_with_stale_carry_forward(meta: dict[str, Any], *, corroborated_by: list[str]) -> dict[str, Any]:
        return {
            **meta,
            "support_override": "stale_internally_consistent_lease_carry_forward",
            "corroborated_by": corroborated_by,
        }

    if not has_any_fresh_class_support:
        if (
            stale_pair_is_eligible
            and stale_pair_corroborated
            and pair_total_value is not None
            and _passes_lease_plausibility(
                float(pair_total_value),
                rou_value=rou_value,
                reference_value=direct_total_value if stale_direct_is_eligible else derived_total_value,
            )
        ):
            corroborated_by = []
            if _values_are_lease_corroborative(pair_total_value, direct_total_value):
                corroborated_by.append("direct_total")
            if _values_are_lease_corroborative(pair_total_value, derived_total_value):
                corroborated_by.append("payments_due_minus_undiscounted_excess")
            candidates.append(
                {
                    "value": float(pair_total_value),
                    "meta": _candidate_meta_with_stale_carry_forward(
                        {
                            "mode": f"{label}_sum_current_noncurrent",
                            "current_components": [current_component["meta"]],
                            "noncurrent_components": [noncurrent_component["meta"]],
                            "class": label,
                        },
                        corroborated_by=corroborated_by,
                    ),
                    "end_dt": max(current_component["end_dt"], noncurrent_component["end_dt"]),
                    "filed_dt": max(current_component["filed_dt"], noncurrent_component["filed_dt"]),
                    "priority": -1,
                }
            )
        elif (
            stale_direct_is_eligible
            and stale_direct_corroborated
            and direct_total_value is not None
            and _passes_lease_plausibility(
                float(direct_total_value),
                rou_value=rou_value,
                reference_value=pair_total_value if stale_pair_is_eligible else derived_total_value,
            )
        ):
            corroborated_by = []
            if _values_are_lease_corroborative(direct_total_value, pair_total_value):
                corroborated_by.append("sum_current_noncurrent")
            if _values_are_lease_corroborative(direct_total_value, derived_total_value):
                corroborated_by.append("payments_due_minus_undiscounted_excess")
            candidates.append(
                {
                    "value": float(direct_total_value),
                    "meta": _candidate_meta_with_stale_carry_forward(
                        {
                            "mode": f"{label}_direct_total",
                            "components": [direct_total["meta"]],
                            "class": label,
                        },
                        corroborated_by=corroborated_by,
                    ),
                    "end_dt": direct_total["end_dt"],
                    "filed_dt": direct_total["filed_dt"],
                    "priority": -2,
                }
            )
        elif (
            stale_derived_is_eligible
            and stale_derived_corroborated
            and derived_total_value is not None
            and _passes_lease_plausibility(
                float(derived_total_value),
                rou_value=rou_value,
                reference_value=pair_total_value if stale_pair_is_eligible else direct_total_value,
            )
        ):
            corroborated_by = []
            if _values_are_lease_corroborative(derived_total_value, pair_total_value):
                corroborated_by.append("sum_current_noncurrent")
            if _values_are_lease_corroborative(derived_total_value, direct_total_value):
                corroborated_by.append("direct_total")
            candidates.append(
                {
                    "value": float(derived_total_value),
                    "meta": _candidate_meta_with_stale_carry_forward(
                        {
                            "mode": f"{label}_payments_due_minus_undiscounted_excess",
                            "payments_due_component": payments_due_component["meta"],
                            "undiscounted_excess_component": undiscounted_excess_component["meta"],
                            "class": label,
                            "formula": "payments_due - undiscounted_excess_amount",
                        },
                        corroborated_by=corroborated_by,
                    ),
                    "end_dt": max(payments_due_component["end_dt"], undiscounted_excess_component["end_dt"]),
                    "filed_dt": max(payments_due_component["filed_dt"], undiscounted_excess_component["filed_dt"]),
                    "priority": -3,
                }
            )

    candidates.sort(key=lambda item: (item["end_dt"], item["filed_dt"], item["priority"]), reverse=True)
    chosen = candidates[0] if candidates else None

    references: dict[str, Any] = {
        "class": label,
        "present": bool(
            total_candidates
            or current_candidates
            or noncurrent_candidates
            or payments_due_candidates
            or current_due_candidates
            or undiscounted_excess_candidates
            or rou_asset_candidates
        ),
    }
    if direct_total is not None:
        references["direct_total_reference"] = {
            "value": float(direct_total["value"]),
            "components": [direct_total["meta"]],
        }
    if freshest_current_component is not None or freshest_noncurrent_component is not None:
        references["partial_component_reference"] = {
            "value": mixed_pair_total_value,
            "current_components": [] if freshest_current_component is None else [freshest_current_component["meta"]],
            "noncurrent_components": [] if freshest_noncurrent_component is None else [freshest_noncurrent_component["meta"]],
        }
    if payments_due_component is not None and undiscounted_excess_component is not None:
        references["payments_due_reference"] = {
            "payments_due": payments_due_component["meta"],
            "undiscounted_excess": undiscounted_excess_component["meta"],
            "derived_total_value": derived_total_value,
        }
    if stale_period_current_due is not None:
        references["current_due_reference"] = {
            "value": float(stale_period_current_due["value"]),
            "components": [stale_period_current_due["meta"]],
        }
    if rou_asset is not None:
        references["right_of_use_asset_reference"] = {
            "value": rou_value,
            "components": [rou_asset["meta"]],
        }

    return {
        "candidate": chosen,
        "present": references["present"],
        "references": references,
    }


def _extract_lease_liabilities(companyfacts: dict, as_of_date: str) -> tuple[float | None, dict[str, Any] | None]:
    operating = _build_lease_class_candidate(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        label="operating",
        total_concepts=OPERATING_LEASE_TOTAL_EXACT_CONCEPTS,
        current_concepts=OPERATING_LEASE_CURRENT_EXACT_CONCEPTS,
        noncurrent_concepts=OPERATING_LEASE_NONCURRENT_EXACT_CONCEPTS,
        payments_due_concepts=OPERATING_LEASE_PAYMENTS_DUE_CONCEPTS,
        current_due_concepts=OPERATING_LEASE_CURRENT_DUE_CONCEPTS,
        undiscounted_excess_concepts=OPERATING_LEASE_UNDISCOUNTED_EXCESS_CONCEPTS,
        rou_asset_concepts=OPERATING_LEASE_RIGHT_OF_USE_ASSET_CONCEPTS,
    )
    finance = _build_lease_class_candidate(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        label="finance",
        total_concepts=FINANCE_LEASE_TOTAL_EXACT_CONCEPTS,
        current_concepts=FINANCE_LEASE_CURRENT_EXACT_CONCEPTS,
        noncurrent_concepts=FINANCE_LEASE_NONCURRENT_EXACT_CONCEPTS,
        payments_due_concepts=FINANCE_LEASE_PAYMENTS_DUE_CONCEPTS,
        current_due_concepts=FINANCE_LEASE_CURRENT_DUE_CONCEPTS,
        undiscounted_excess_concepts=FINANCE_LEASE_UNDISCOUNTED_EXCESS_CONCEPTS,
        rou_asset_concepts=FINANCE_LEASE_RIGHT_OF_USE_ASSET_CONCEPTS,
    )

    aggregate_total = _select_best_candidate(_extract_exact_candidates(companyfacts, as_of_date, LEASE_AGGREGATE_TOTAL_EXACT_CONCEPTS))
    aggregate_candidate = None
    if aggregate_total is not None and _candidate_is_fresh(aggregate_total, as_of_date):
        operating_reference = operating["candidate"]["value"] if operating["candidate"] is not None else None
        finance_reference = finance["candidate"]["value"] if finance["candidate"] is not None else None
        combined_reference = None
        if operating_reference is not None or finance_reference is not None:
            combined_reference = float(operating_reference or 0.0) + float(finance_reference or 0.0)
        if _passes_lease_plausibility(
            float(aggregate_total["value"]),
            rou_value=None,
            reference_value=combined_reference,
        ):
            aggregate_candidate = {
                "value": float(aggregate_total["value"]),
                "meta": {
                    "mode": "aggregate_total",
                    "components": [aggregate_total["meta"]],
                },
                "end_dt": aggregate_total["end_dt"],
                "filed_dt": aggregate_total["filed_dt"],
                "priority": 2,
            }

    candidates: list[dict[str, Any]] = []
    operating_candidate = operating["candidate"]
    finance_candidate = finance["candidate"]
    finance_has_fresh_support = _reference_contains_fresh_lease_fact(finance["references"], as_of_date)
    operating_has_fresh_support = _reference_contains_fresh_lease_fact(operating["references"], as_of_date)

    if operating_candidate is not None and finance_candidate is not None:
        gap_days = abs((operating_candidate["end_dt"] - finance_candidate["end_dt"]).days)
        if gap_days <= LEASE_COMPONENT_ALIGNMENT_MAX_GAP_DAYS:
            candidates.append(
                {
                    "value": float(operating_candidate["value"]) + float(finance_candidate["value"]),
                    "meta": {
                        "mode": "sum_operating_finance",
                        "operating_component": operating_candidate["meta"],
                        "finance_component": finance_candidate["meta"],
                        "operating_reference": operating["references"],
                        "finance_reference": finance["references"],
                        "alignment_gap_days": gap_days,
                    },
                    "end_dt": max(operating_candidate["end_dt"], finance_candidate["end_dt"]),
                    "filed_dt": max(operating_candidate["filed_dt"], finance_candidate["filed_dt"]),
                    "priority": 3,
                }
            )

    if operating_candidate is not None and not finance["present"]:
        candidates.append(
            {
                "value": float(operating_candidate["value"]),
                "meta": {
                    "mode": "operating_only_no_finance_concepts",
                    "operating_component": operating_candidate["meta"],
                    "operating_reference": operating["references"],
                },
                "end_dt": operating_candidate["end_dt"],
                "filed_dt": operating_candidate["filed_dt"],
                "priority": 1,
            }
        )
    elif operating_candidate is not None and finance_candidate is None and not finance_has_fresh_support:
        candidates.append(
            {
                "value": float(operating_candidate["value"]),
                "meta": {
                    "mode": "operating_only_no_fresh_finance_support",
                    "operating_component": operating_candidate["meta"],
                    "operating_reference": operating["references"],
                    "finance_reference": finance["references"],
                },
                "end_dt": operating_candidate["end_dt"],
                "filed_dt": operating_candidate["filed_dt"],
                "priority": 1,
            }
        )

    if finance_candidate is not None and not operating["present"]:
        candidates.append(
            {
                "value": float(finance_candidate["value"]),
                "meta": {
                    "mode": "finance_only_no_operating_concepts",
                    "finance_component": finance_candidate["meta"],
                    "finance_reference": finance["references"],
                },
                "end_dt": finance_candidate["end_dt"],
                "filed_dt": finance_candidate["filed_dt"],
                "priority": 1,
            }
        )
    elif finance_candidate is not None and operating_candidate is None and not operating_has_fresh_support:
        candidates.append(
            {
                "value": float(finance_candidate["value"]),
                "meta": {
                    "mode": "finance_only_no_fresh_operating_support",
                    "finance_component": finance_candidate["meta"],
                    "finance_reference": finance["references"],
                    "operating_reference": operating["references"],
                },
                "end_dt": finance_candidate["end_dt"],
                "filed_dt": finance_candidate["filed_dt"],
                "priority": 1,
            }
        )

    if aggregate_candidate is not None:
        aggregate_candidate["meta"]["operating_reference"] = operating["references"]
        aggregate_candidate["meta"]["finance_reference"] = finance["references"]
        candidates.append(aggregate_candidate)

    candidates.sort(key=lambda item: (item["end_dt"], item["filed_dt"], item["priority"]), reverse=True)
    if candidates:
        chosen = candidates[0]
        return float(chosen["value"]), chosen["meta"]

    return None, {
        "mode": "lease_total_unavailable",
        "operating_reference": operating["references"],
        "finance_reference": finance["references"],
        "aggregate_total_reference": None
        if aggregate_total is None
        else {
            "value": float(aggregate_total["value"]),
            "components": [aggregate_total["meta"]],
        },
    }


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.snapshot_path)
    companyfacts_root = Path(args.companyfacts_root) if args.companyfacts_root else (
        DEFAULT_LOCAL_COMPANYFACTS_ROOT if DEFAULT_LOCAL_COMPANYFACTS_ROOT.exists() else None
    )
    if companyfacts_root is None:
        raise SystemExit("companyfacts root is required")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    computed_at = _now_iso()
    counters: Counter[str] = Counter()

    with out_path.open("w") as out_handle:
        for row in iter_snapshot_rows(snapshot_path):
            entity_id = row["company_id"]
            as_of_time = row["as_of_time"]
            as_of_date = as_of_time[:10]
            companyfacts_path = companyfacts_root / f"CIK{entity_id}.json"
            companyfacts = _load_companyfacts(companyfacts_path)
            features = row.setdefault("features", {})

            metric_builds = []
            if companyfacts is None:
                metric_builds = [
                    ("liquidity.restricted_cash_sec_exact", None, None, "companyfacts_unavailable", "usd"),
                    ("liquidity.marketable_securities_sec_exact", None, None, "companyfacts_unavailable", "usd"),
                    ("liquidity.revolver_undrawn_sec_exact", None, None, "companyfacts_unavailable", "usd"),
                    ("capital_structure.lease_liabilities_sec_exact", None, None, "companyfacts_unavailable", "usd"),
                ]
            else:
                restricted_value, restricted_meta = _extract_restricted_cash(companyfacts, as_of_date)
                marketable_value, marketable_meta = _extract_marketable_securities(companyfacts, as_of_date)
                revolver_value, revolver_meta = _extract_revolver_undrawn(companyfacts, as_of_date)
                lease_value, lease_meta = _extract_lease_liabilities(companyfacts, as_of_date)
                restricted_missing_reason = (
                    "sec_concept_unavailable"
                    if _has_any_us_gaap_concepts(companyfacts, RESTRICTED_CASH_ANY_CONCEPTS)
                    else "sec_concept_absent"
                )
                marketable_missing_reason = (
                    "sec_concept_unavailable"
                    if _has_any_us_gaap_concepts(companyfacts, MARKETABLE_SECURITY_ANY_CONCEPTS)
                    else "sec_concept_absent"
                )
                revolver_missing_reason = (
                    "sec_concept_unavailable"
                    if _has_any_us_gaap_concepts(companyfacts, REVOLVER_UNDRAWN_EXACT_CONCEPTS)
                    else "sec_concept_absent"
                )
                lease_missing_reason = (
                    "sec_concept_unavailable"
                    if _has_any_us_gaap_concepts(
                        companyfacts,
                        OPERATING_LEASE_ANY_CONCEPTS
                        | FINANCE_LEASE_ANY_CONCEPTS
                        | LEASE_AGGREGATE_TOTAL_EXACT_CONCEPTS,
                    )
                    else "sec_concept_absent"
                )
                metric_builds = [
                    ("liquidity.restricted_cash_sec_exact", restricted_value, restricted_meta, restricted_missing_reason, "usd"),
                    ("liquidity.marketable_securities_sec_exact", marketable_value, marketable_meta, marketable_missing_reason, "usd"),
                    ("liquidity.revolver_undrawn_sec_exact", revolver_value, revolver_meta, revolver_missing_reason, "usd"),
                    ("capital_structure.lease_liabilities_sec_exact", lease_value, lease_meta, lease_missing_reason, "usd"),
                ]

            for metric_name, value, meta, missing_reason, unit in metric_builds:
                if value is None:
                    node = _feature_template(
                        metric_name=metric_name,
                        as_of_time=as_of_time,
                        computed_at=computed_at,
                        provenance_source=str(companyfacts_path),
                        support_mode="unsupported",
                        value=None,
                        unit=unit,
                        missing_reason=missing_reason,
                        component_breakdown=(
                            {"companyfacts_path": str(companyfacts_path)}
                            if meta is None
                            else {**meta, "companyfacts_path": str(companyfacts_path)}
                        ),
                        quality_flags=[missing_reason],
                    )
                else:
                    node = _feature_template(
                        metric_name=metric_name,
                        as_of_time=as_of_time,
                        computed_at=computed_at,
                        provenance_source=str(companyfacts_path),
                        support_mode="exact",
                        value=float(value),
                        unit=unit,
                        missing_reason=None,
                        component_breakdown=meta,
                        quality_flags=None,
                    )
                features[metric_name] = node
                counters[f"{metric_name}:{node['support_mode']}"] += 1

            restricted_node = features.get("liquidity.restricted_cash_sec_exact") or {}
            cash_eq_node = features.get("liquidity.cash_and_equivalents_statement_direct") or {}
            cash_sti_node = features.get("liquidity.cash_and_short_term_investments_provider_direct") or {}
            marketable_node = features.get("liquidity.marketable_securities_sec_exact") or {}
            repaired_restricted_node = _repair_restricted_cash_from_total_cash_reconciliation(
                restricted_node=restricted_node,
                cash_eq_node=cash_eq_node,
                cash_sti_node=cash_sti_node,
                marketable_node=marketable_node,
                companyfacts=companyfacts,
                companyfacts_path=companyfacts_path,
                as_of_date=as_of_date,
                as_of_time=as_of_time,
                computed_at=computed_at,
            )
            if repaired_restricted_node is not None:
                prior_mode = _metric_support(restricted_node)
                counters[f"liquidity.restricted_cash_sec_exact:{prior_mode}"] -= 1
                features["liquidity.restricted_cash_sec_exact"] = repaired_restricted_node
                counters[
                    f"liquidity.restricted_cash_sec_exact:{repaired_restricted_node['support_mode']}"
                ] += 1

            cash_sti_node = features.get("liquidity.cash_and_short_term_investments_provider_direct") or {}
            cash_eq_node = features.get("liquidity.cash_and_equivalents_statement_direct") or {}
            marketable_node = features.get("liquidity.marketable_securities_sec_exact") or {}
            repaired_node = _repair_cash_sti_from_statement_cash(
                cash_sti_node=cash_sti_node,
                cash_eq_node=cash_eq_node,
                marketable_node=marketable_node,
                companyfacts_path=companyfacts_path,
                as_of_date=as_of_date,
                as_of_time=as_of_time,
                computed_at=computed_at,
            )
            if repaired_node is not None:
                features["liquidity.cash_and_short_term_investments_provider_direct"] = repaired_node

            total_debt_value = _metric_value(features.get("capital_structure.total_debt_provider_direct"))
            total_debt_support = _metric_support(features.get("capital_structure.total_debt_provider_direct"))
            cash_sti_value = _metric_value(features.get("liquidity.cash_and_short_term_investments_provider_direct"))
            cash_sti_support = _metric_support(features.get("liquidity.cash_and_short_term_investments_provider_direct"))
            ebitda_value = _metric_value(features.get("operating.ebitda_ltm_provider_direct"))
            ebitda_support = _metric_support(features.get("operating.ebitda_ltm_provider_direct"))
            net_debt_value = None if total_debt_value is None or cash_sti_value is None else total_debt_value - cash_sti_value

            features["capital_structure.net_debt_standardized"] = _build_combo_metric(
                metric_name="capital_structure.net_debt_standardized",
                as_of_time=as_of_time,
                computed_at=computed_at,
                provenance_source=str(companyfacts_path),
                unit="usd",
                numerator=net_debt_value,
                denominator=None,
                extra_components={
                    "total_debt_provider_direct": total_debt_value,
                    "cash_and_short_term_investments_provider_direct": cash_sti_value,
                },
                component_supports={
                    "total_debt_provider_direct": total_debt_support,
                    "cash_and_short_term_investments_provider_direct": cash_sti_support,
                },
                formula="total_debt_provider_direct - cash_and_short_term_investments_provider_direct",
                allow_numerator_only=True,
            )
            features["capital_structure.net_leverage_standardized"] = _build_combo_metric(
                metric_name="capital_structure.net_leverage_standardized",
                as_of_time=as_of_time,
                computed_at=computed_at,
                provenance_source=str(companyfacts_path),
                unit="x",
                numerator=net_debt_value,
                denominator=ebitda_value,
                extra_components={
                    "net_debt_standardized": net_debt_value,
                    "ebitda_ltm_provider_direct": ebitda_value,
                },
                component_supports={
                    "net_debt_standardized": _metric_support(features.get("capital_structure.net_debt_standardized")),
                    "ebitda_ltm_provider_direct": ebitda_support,
                },
                formula="net_debt_standardized / ebitda_ltm_provider_direct",
            )

            out_handle.write(json.dumps(row) + "\n")

    if args.summary_out:
        summary = {}
        for metric_name in [
            "liquidity.restricted_cash_sec_exact",
            "liquidity.marketable_securities_sec_exact",
            "liquidity.revolver_undrawn_sec_exact",
            "capital_structure.lease_liabilities_sec_exact",
        ]:
            summary[metric_name] = {
                "exact": counters[f"{metric_name}:exact"],
                "unsupported": counters[f"{metric_name}:unsupported"],
            }
        Path(args.summary_out).write_text(json.dumps(summary, indent=2))

    print(f"Wrote SEC companyfacts components -> {out_path}")


if __name__ == "__main__":
    main()
