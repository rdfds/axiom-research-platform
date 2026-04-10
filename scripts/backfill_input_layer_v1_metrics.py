#!/usr/bin/env python3
"""Materialize the approved v1 input-layer metrics into a snapshot JSONL artifact.

This builder now prefers as-of-safe SEC companyfacts logic for the core company
metrics. The old provider sidecar remains available only as a legacy fallback
when no companyfacts root is supplied.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
import signal
from typing import Any, Dict, Iterable

import pandas as pd

try:
    from repair_statement_debt_override_artifact import _repair_total_debt_from_sec_filing, _sec_session
except Exception:  # noqa: BLE001
    try:
        from scripts.repair_statement_debt_override_artifact import _repair_total_debt_from_sec_filing, _sec_session
    except Exception:  # noqa: BLE001
        _repair_total_debt_from_sec_filing = None
        _sec_session = None

try:
    from backfill_market_macro_input_layer_v1 import (
        DEFAULT_LOCAL_CRSP_DAILY_ROOT,
        _build_market_cap_metric,
        _build_market_cap_metric_from_companyfacts,
        _build_price_metrics_from_crsp,
        _build_price_metrics,
        _load_crsp_daily_from_repo,
        _load_crsp_market_cache,
        _load_price_history,
        _permno_map as _market_permno_map,
    )
except Exception:  # noqa: BLE001
    try:
        from scripts.backfill_market_macro_input_layer_v1 import (
            DEFAULT_LOCAL_CRSP_DAILY_ROOT,
            _build_market_cap_metric,
            _build_market_cap_metric_from_companyfacts,
            _build_price_metrics_from_crsp,
            _build_price_metrics,
            _load_crsp_daily_from_repo,
            _load_crsp_market_cache,
            _load_price_history,
            _permno_map as _market_permno_map,
        )
    except Exception:  # noqa: BLE001
        DEFAULT_LOCAL_CRSP_DAILY_ROOT = None
        _build_market_cap_metric = None
        _build_market_cap_metric_from_companyfacts = None
        _build_price_metrics_from_crsp = None
        _build_price_metrics = None
        _load_crsp_daily_from_repo = None
        _load_crsp_market_cache = None
        _load_price_history = None
        _market_permno_map = None


MAX_SEC_FACT_AGE_DAYS = 550
DEBT_COMPONENT_ALIGNMENT_MAX_GAP_DAYS = 10
CASH_COMPONENT_ALIGNMENT_MAX_GAP_DAYS = 10
EXACT_BALANCE_SHEET_MAX_AGE_DAYS = 130
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_COMPANYFACTS_ROOT = REPO_ROOT / "data" / "sec" / "companyfacts"
DEFAULT_LOCAL_RAW_TIMESERIES_PATH = REPO_ROOT / "data" / "inputs_layer" / "raw_timeseries.parquet"

DIRECT_METRIC_SPECS = {
    "market.market_cap_provider_direct": {"unit": "usd"},
    "operating.revenue_ttm_provider_direct": {"unit": "usd"},
    "operating.revenue_ttm_lag_1y": {"unit": "usd"},
    "operating.ebitda_ltm_provider_direct": {"unit": "usd"},
    "earnings.net_income_ttm_provider_direct": {"unit": "usd"},
    "liquidity.cash_and_short_term_investments_provider_direct": {"unit": "usd"},
    "capital_structure.total_debt_provider_direct": {"unit": "usd"},
}

DERIVED_METRIC_SPECS = {
    "capital_structure.net_debt_standardized": {"unit": "usd"},
    "capital_structure.gross_leverage_standardized": {"unit": "x"},
    "capital_structure.net_leverage_standardized": {"unit": "x"},
    "operating.ebitda_margin_standardized": {"unit": "ratio"},
    "earnings.net_margin_standardized": {"unit": "ratio"},
}

ALL_OUTPUT_METRIC_SPECS = {
    **DIRECT_METRIC_SPECS,
    **DERIVED_METRIC_SPECS,
}

LEGACY_PROVIDER_SOURCE_COLUMNS = {
    "market.market_cap_provider_direct": "Company Market Cap",
    "operating.revenue_ttm_provider_direct": "Revenue",
    "operating.ebitda_ltm_provider_direct": "EBITDA",
    "earnings.net_income_ttm_provider_direct": "Net Income Incl Extra Before Distributions",
    "liquidity.cash_and_short_term_investments_provider_direct": "Cash and Short Term Investments",
    "capital_structure.total_debt_provider_direct": "Total Debt",
}

REVENUE_TTM_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueServicesNet",
    "SalesRevenueGoodsNet",
]
NET_INCOME_TTM_CONCEPTS = [
    "NetIncomeLoss",
    "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
]
OPERATING_INCOME_TTM_CONCEPTS = ["OperatingIncomeLoss"]
DEPRECIATION_TTM_CONCEPT_GROUPS = [
    ["DepreciationAmortizationAndAccretionNet"],
    ["DepreciationDepletionAndAmortization"],
    ["DepreciationAndAmortization"],
    ["OtherDepreciationAndAmortization"],
    ["Depreciation", "AmortizationOfAcquiredIntangibleAssets"],
    ["Depreciation", "FinanceLeaseRightOfUseAssetAmortization"],
    ["Depreciation"],
    ["Depreciation", "AmortizationOfIntangibleAssets"],
    ["Depreciation", "FiniteLivedIntangibleAssetsAmortizationExpense"],
    ["Depreciation", "CapitalizedComputerSoftwareAmortization"],
    ["Depreciation", "CapitalizedComputerSoftwareAmortization1"],
]
CASH_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "Cash",
]
COMBINED_CASH_STI_CONCEPTS = [
    "CashCashEquivalentsAndShortTermInvestments",
]
COMBINED_CASH_RESTRICTED_TOTAL_CONCEPTS = [
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
]
STI_CONCEPTS = [
    "ShortTermInvestments",
    "MarketableSecuritiesCurrent",
    "MarketableSecurities",
    "AvailableForSaleSecuritiesCurrent",
    "AvailableForSaleDebtSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtMaturitiesWithinOneYearFairValue",
    "AvailableForSaleSecuritiesDebtMaturitiesNextRollingTwelveMonthsFairValue",
]
RESTRICTED_CASH_TOTAL_CONCEPTS = [
    "RestrictedCashAndCashEquivalentsAtCarryingValue",
    "RestrictedCashAndCashEquivalents",
    "RestrictedCash",
    "RestrictedCashCurrent",
    "RestrictedCashAndInvestmentsCurrent",
    "RestrictedCashAndInvestments",
]
TOTAL_DEBT_COMBINED_CONCEPTS = [
    "DebtLongtermAndShorttermCombinedAmount",
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    "DebtAndCapitalLeaseObligations",
]
SHORT_TERM_BORROWINGS_CONCEPTS = [
    "ShortTermBorrowings",
    "CommercialPaper",
    "CommercialPaperCurrent",
    "NotesPayableCurrent",
    "LoansPayableCurrent",
    "LinesOfCreditCurrent",
    "ShortTermDebt",
    "SecuredDebt",
    "TransfersAccountedForAsSecuredBorrowingsAssociatedLiabilitiesCarryingAmount",
]
ADDITIVE_SHORT_TERM_BORROWINGS_CONCEPTS = {
    "SecuredDebt",
    "TransfersAccountedForAsSecuredBorrowingsAssociatedLiabilitiesCarryingAmount",
}
DEBT_CURRENT_CONCEPTS = [
    "DebtCurrent",
    "LongTermDebtCurrent",
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
    "ConvertibleDebtCurrent",
]
DEBT_NONCURRENT_CONCEPTS = [
    "LongTermDebtNoncurrent",
    "LongTermDebt",
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermLineOfCredit",
    "LongTermNotesPayable",
    "ConvertibleDebtNoncurrent",
    "ConvertibleDebt",
]
DEBT_BALANCE_CONCEPTS = set(TOTAL_DEBT_COMBINED_CONCEPTS) | set(SHORT_TERM_BORROWINGS_CONCEPTS) | set(DEBT_CURRENT_CONCEPTS) | set(DEBT_NONCURRENT_CONCEPTS)
NONCURRENT_TOTAL_ONLY_EXACT_CONCEPTS = {
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "LongTermLineOfCredit",
    "LongTermNotesPayable",
    "ConvertibleDebtNoncurrent",
    "ConvertibleDebt",
}
FINANCE_LEASE_CURRENT_CONCEPTS = [
    "FinanceLeaseLiabilityCurrent",
    "LesseeFinanceLeaseLiabilityCurrent",
]
FINANCE_LEASE_NONCURRENT_CONCEPTS = [
    "FinanceLeaseLiabilityNoncurrent",
    "LesseeFinanceLeaseLiabilityNoncurrent",
]
FINANCE_LEASE_TOTAL_CONCEPTS = [
    "FinanceLeaseLiability",
    "LesseeFinanceLeaseLiability",
]
FINANCE_LEASE_ANY_CONCEPTS = set(FINANCE_LEASE_CURRENT_CONCEPTS + FINANCE_LEASE_NONCURRENT_CONCEPTS + FINANCE_LEASE_TOTAL_CONCEPTS)
SHARES_OUT_CONCEPTS = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
]


class _CompanyProcessingTimeout(RuntimeError):
    """Raised when a single-company build exceeds the allowed timeout."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", required=True, help="Input snapshot JSONL path")
    parser.add_argument("--taxonomy-reference-path", required=True, help="Provider reference parquet")
    parser.add_argument("--entity-identifier-path", required=True, help="Entity identifier parquet with ticker rows")
    parser.add_argument(
        "--companyfacts-root",
        help="Optional SEC companyfacts folder. Defaults to the local canonical companyfacts root when present.",
    )
    parser.add_argument(
        "--raw-timeseries-path",
        help="Optional raw_timeseries parquet used to make market cap PIT-safe. Defaults to the local canonical file when present.",
    )
    parser.add_argument("--crsp-market-cache-path", help="Optional filtered CRSP daily market parquet cache")
    parser.add_argument(
        "--crsp-daily-root",
        help="Optional CRSP daily parquet folder. Defaults to the local canonical WRDS CRSP folder when present.",
    )
    parser.add_argument(
        "--allow-monthly-market-proxy",
        action="store_true",
        help="Allow monthly raw-timeseries price proxies when exact CRSP daily data is unavailable.",
    )
    parser.add_argument(
        "--sec-filing-cache-root",
        default="/tmp/sec_filing_debt_cache",
        help="Cache directory for SEC submissions and filing HTML used by the debt fallback",
    )
    parser.add_argument(
        "--enable-sec-filing-debt-repair",
        action="store_true",
        help="Opt in to slower SEC filing HTML debt repair when companyfacts debt concepts are insufficient.",
    )
    parser.add_argument(
        "--company-processing-timeout-seconds",
        type=float,
        default=30.0,
        help="Fail open on a single company if metric construction exceeds this timeout. Use 0 to disable.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ticker_to_entity_map(entity_identifier_path: Path) -> pd.DataFrame:
    ids = pd.read_parquet(entity_identifier_path)
    ids = ids[ids["identifier_type"].astype(str).str.lower() == "ticker"].copy()
    ids["ticker"] = ids["identifier_value"].astype(str).str.upper().str.strip()
    return ids[["entity_id", "ticker"]].drop_duplicates()


def _provider_reference_map(taxonomy_reference_path: Path, entity_identifier_path: Path) -> pd.DataFrame:
    ref = pd.read_parquet(taxonomy_reference_path).copy()
    ref["ticker"] = ref["Instrument"].astype(str).str.replace(r"\..*$", "", regex=True).str.upper().str.strip()
    tickers = _ticker_to_entity_map(entity_identifier_path)
    merged = ref.merge(tickers, on="ticker", how="inner")
    merged = merged.sort_values(["entity_id", "Instrument"]).drop_duplicates("entity_id", keep="first")
    return merged


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
    provenance_artifact_type: str = "ReferenceFact",
    primary_source_basis: str = "provider_direct",
    input_layer_bucket_reason: str = "provider_reference_sidecar",
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
                "artifact_type": provenance_artifact_type,
                "artifact_id": f"{primary_source_basis}:{Path(provenance_source).name}",
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
        "primary_source_basis": primary_source_basis,
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
        "input_source_classification": primary_source_basis,
        "input_source_formula_basis": None,
        "input_source_alignment_status": "aligned",
        "input_source_document_ids": None,
        "definition_requirement": None,
        "definition_requirement_reason": None,
        "methodology_execution_decision": None,
        "methodology_execution_reason": None,
        "input_layer_bucket": "reference",
        "input_layer_bucket_reason": input_layer_bucket_reason,
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


def _build_metric_from_value(
    *,
    metric_name: str,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    unit: str,
    value: float | None,
    support_mode: str,
    missing_reason: str | None,
    component_breakdown: Dict[str, Any] | None,
    quality_flags: list[str] | None,
    primary_source_basis: str,
    provenance_artifact_type: str,
    input_layer_bucket_reason: str,
) -> Dict[str, Any]:
    return _feature_template(
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
        provenance_artifact_type=provenance_artifact_type,
        primary_source_basis=primary_source_basis,
        input_layer_bucket_reason=input_layer_bucket_reason,
    )


def _support_rank(node: Dict[str, Any] | None) -> int:
    support_mode = (node or {}).get("support_mode")
    if support_mode == "exact":
        return 2
    if support_mode == "proxy_missing_component":
        return 1
    return 0


def _merge_quality_flags(*flag_lists: list[str] | None) -> list[str] | None:
    merged: list[str] = []
    for values in flag_lists:
        for value in values or []:
            value_text = str(value)
            if value_text and value_text not in merged:
                merged.append(value_text)
    return merged or None


def _annotate_selected_metric(
    node: Dict[str, Any],
    *,
    metric_name: str,
    selection_policy: str,
    comparison_node: Dict[str, Any] | None,
    extra_quality_flags: list[str] | None = None,
) -> Dict[str, Any]:
    selected = deepcopy(node)
    breakdown = deepcopy(selected.get("component_breakdown")) if isinstance(selected.get("component_breakdown"), dict) else (
        deepcopy(selected.get("component_breakdown"))
    )
    if isinstance(breakdown, dict):
        breakdown["selection_policy"] = selection_policy
        if comparison_node is not None:
            comparison_value = comparison_node.get("value")
            if comparison_value is not None and selected.get("value") is not None:
                try:
                    comparison_gap = float(selected["value"]) - float(comparison_value)
                except Exception:  # noqa: BLE001
                    comparison_gap = None
            else:
                comparison_gap = None
            breakdown["comparison_candidate"] = {
                "primary_source_basis": comparison_node.get("primary_source_basis"),
                "support_mode": comparison_node.get("support_mode"),
                "missing_reason": comparison_node.get("missing_reason"),
                "value": comparison_value,
                "value_gap_vs_selected": comparison_gap,
            }
    selected["component_breakdown"] = breakdown
    selected["quality_flags"] = _merge_quality_flags(selected.get("quality_flags"), extra_quality_flags)
    if metric_name == "market.market_cap_provider_direct" and selected.get("primary_source_basis") == "provider_direct":
        selected["quality_flags"] = _merge_quality_flags(
            selected.get("quality_flags"),
            ["market_cap_provider_fallback_used"],
        )
    return selected


def _select_preferred_direct_metric(
    *,
    metric_name: str,
    sec_or_market_node: Dict[str, Any] | None,
    provider_node: Dict[str, Any],
) -> Dict[str, Any]:
    preferred_rank = _support_rank(sec_or_market_node)
    provider_rank = _support_rank(provider_node)

    if metric_name == "market.market_cap_provider_direct":
        if preferred_rank >= 2:
            return _annotate_selected_metric(
                sec_or_market_node or provider_node,
                metric_name=metric_name,
                selection_policy="prefer_exact_pit_market_cap_when_available",
                comparison_node=provider_node,
                extra_quality_flags=["provider_direct_superseded_by_pit_market_cap"],
            )
        if provider_rank >= 1:
            return _annotate_selected_metric(
                provider_node,
                metric_name=metric_name,
                selection_policy="retain_provider_when_pit_market_cap_is_only_proxy",
                comparison_node=sec_or_market_node,
                extra_quality_flags=["provider_direct_retained_due_to_proxy_pit_market_cap"],
            )
        if preferred_rank >= 1:
            return _annotate_selected_metric(
                sec_or_market_node or provider_node,
                metric_name=metric_name,
                selection_policy="use_proxy_pit_market_cap_when_no_provider_fallback_exists",
                comparison_node=provider_node,
                extra_quality_flags=["proxy_pit_market_cap_used_due_to_missing_provider_fallback"],
            )
        return _annotate_selected_metric(
            provider_node,
            metric_name=metric_name,
            selection_policy="fallback_to_provider_when_pit_market_cap_unavailable",
            comparison_node=sec_or_market_node,
            extra_quality_flags=["provider_direct_retained_due_to_unavailable_pit_market_cap"],
        )

    if metric_name in {
        "operating.revenue_ttm_provider_direct",
        "operating.revenue_ttm_lag_1y",
        "earnings.net_income_ttm_provider_direct",
        "liquidity.cash_and_short_term_investments_provider_direct",
    }:
        if preferred_rank >= 1:
            return _annotate_selected_metric(
                sec_or_market_node or provider_node,
                metric_name=metric_name,
                selection_policy="prefer_sec_companyfacts_reconstruction",
                comparison_node=provider_node,
                extra_quality_flags=["provider_direct_superseded_by_sec_companyfacts"],
            )
        return _annotate_selected_metric(
            provider_node,
            metric_name=metric_name,
            selection_policy="fallback_to_provider_when_sec_unavailable",
            comparison_node=sec_or_market_node,
            extra_quality_flags=["provider_direct_retained_due_to_unavailable_sec_companyfacts"],
        )

    if metric_name == "operating.ebitda_ltm_provider_direct":
        if preferred_rank >= 2:
            return _annotate_selected_metric(
                sec_or_market_node or provider_node,
                metric_name=metric_name,
                selection_policy="prefer_exact_sec_ebitda_bridge",
                comparison_node=provider_node,
                extra_quality_flags=["provider_direct_superseded_by_exact_sec_bridge"],
            )
        if provider_rank >= 1:
            return _annotate_selected_metric(
                provider_node,
                metric_name=metric_name,
                selection_policy="retain_provider_when_sec_ebitda_is_partial_or_unavailable",
                comparison_node=sec_or_market_node,
                extra_quality_flags=["provider_direct_retained_due_to_partial_sec_ebitda_bridge"],
            )
        return _annotate_selected_metric(
            sec_or_market_node or provider_node,
            metric_name=metric_name,
            selection_policy="use_partial_sec_ebitda_when_no_provider_fallback_exists",
            comparison_node=provider_node,
        )

    if metric_name == "capital_structure.total_debt_provider_direct":
        if preferred_rank >= 2:
            return _annotate_selected_metric(
                sec_or_market_node or provider_node,
                metric_name=metric_name,
                selection_policy="prefer_exact_sec_debt_stack",
                comparison_node=provider_node,
                extra_quality_flags=["provider_direct_superseded_by_exact_sec_debt_stack"],
            )
        if provider_rank >= 1:
            return _annotate_selected_metric(
                provider_node,
                metric_name=metric_name,
                selection_policy="retain_provider_when_sec_debt_stack_is_partial_or_unavailable",
                comparison_node=sec_or_market_node,
                extra_quality_flags=["provider_direct_retained_due_to_partial_sec_debt_stack"],
            )
        return _annotate_selected_metric(
            sec_or_market_node or provider_node,
            metric_name=metric_name,
            selection_policy="use_partial_sec_debt_stack_when_no_provider_fallback_exists",
            comparison_node=provider_node,
        )

    return _annotate_selected_metric(
        sec_or_market_node or provider_node,
        metric_name=metric_name,
        selection_policy="default_selection_policy",
        comparison_node=provider_node if sec_or_market_node is not provider_node else None,
    )


def _resolve_local_optional_path(explicit_path: str | None, default_path: Path) -> Path | None:
    if explicit_path:
        return Path(explicit_path)
    if default_path.exists():
        return default_path
    return None


def _parse_iso_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(str(text)[:10])
    except ValueError:
        return None


def _load_companyfacts(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _candidate_units_map(companyfacts: dict, concept_name: str, taxonomy: str | None = None) -> dict | None:
    taxonomies = [taxonomy] if taxonomy else ["us-gaap", "dei", "ifrs-full"]
    for current_taxonomy in taxonomies:
        facts = (companyfacts.get("facts") or {}).get(current_taxonomy) or {}
        if concept_name in facts:
            return facts[concept_name].get("units") or {}
    return None


def _latest_instant_value(
    companyfacts: dict,
    concepts: list[tuple[str | None, str]] | list[str],
    *,
    as_of_date: str,
    unit_filter: str,
) -> tuple[float | None, dict[str, Any] | None]:
    as_of_dt = date.fromisoformat(as_of_date)
    normalized: list[tuple[str | None, str]] = []
    for concept in concepts:
        if isinstance(concept, tuple):
            normalized.append(concept)
        else:
            normalized.append((None, concept))

    best_candidate = None
    for priority, (taxonomy, concept_name) in enumerate(normalized):
        units_map = _candidate_units_map(companyfacts, concept_name, taxonomy)
        if not units_map:
            continue
        for unit, entries in units_map.items():
            if unit.lower() != unit_filter.lower():
                continue
            for entry in entries:
                end_text = entry.get("end")
                filed_text = entry.get("filed")
                value = entry.get("val")
                if end_text is None or value is None:
                    continue
                end_dt = _parse_iso_date(end_text)
                filed_dt = _parse_iso_date(filed_text)
                if end_dt is None or end_dt > as_of_dt:
                    continue
                if filed_dt is not None and filed_dt > as_of_dt:
                    continue
                if (as_of_dt - end_dt).days > MAX_SEC_FACT_AGE_DAYS:
                    continue
                candidate = (
                    end_dt,
                    filed_dt or end_dt,
                    -priority,
                    entry,
                    unit,
                    taxonomy or "us-gaap",
                    concept_name,
                )
                if best_candidate is None or candidate[:3] > best_candidate[:3]:
                    best_candidate = candidate
    if best_candidate is None:
        return None, None
    end_dt, filed_dt, _, chosen, unit, chosen_taxonomy, chosen_concept = best_candidate
    return float(chosen["val"]), {
        "concept": chosen_concept,
        "taxonomy": chosen_taxonomy,
        "end": end_dt.isoformat(),
        "filed": filed_dt.isoformat(),
        "fy": chosen.get("fy"),
        "fp": chosen.get("fp"),
        "frame": chosen.get("frame"),
        "form": chosen.get("form"),
        "unit": unit,
        "formula": "latest_instant_value_on_or_before_asof",
    }


def _instant_candidates(
    companyfacts: dict,
    concepts: list[tuple[str | None, str]] | list[str],
    *,
    as_of_date: str,
    unit_filter: str,
) -> list[dict[str, Any]]:
    as_of_dt = date.fromisoformat(as_of_date)
    normalized: list[tuple[str | None, str]] = []
    for concept in concepts:
        if isinstance(concept, tuple):
            normalized.append(concept)
        else:
            normalized.append((None, concept))

    candidates: list[dict[str, Any]] = []
    for priority, (taxonomy, concept_name) in enumerate(normalized):
        units_map = _candidate_units_map(companyfacts, concept_name, taxonomy)
        if not units_map:
            continue
        for unit, entries in units_map.items():
            if unit.lower() != unit_filter.lower():
                continue
            for entry in entries:
                end_text = entry.get("end")
                filed_text = entry.get("filed")
                value = entry.get("val")
                if end_text is None or value is None:
                    continue
                end_dt = _parse_iso_date(end_text)
                filed_dt = _parse_iso_date(filed_text)
                if end_dt is None or end_dt > as_of_dt:
                    continue
                if filed_dt is not None and filed_dt > as_of_dt:
                    continue
                if (as_of_dt - end_dt).days > MAX_SEC_FACT_AGE_DAYS:
                    continue
                candidates.append(
                    {
                        "value": float(value),
                        "meta": {
                            "concept": concept_name,
                            "taxonomy": taxonomy or "us-gaap",
                            "end": end_dt.isoformat(),
                            "filed": (filed_dt or end_dt).isoformat(),
                            "fy": entry.get("fy"),
                            "fp": entry.get("fp"),
                            "frame": entry.get("frame"),
                            "form": entry.get("form"),
                            "unit": unit,
                            "formula": "latest_instant_value_on_or_before_asof",
                        },
                        "end_dt": end_dt,
                        "filed_dt": filed_dt or end_dt,
                        "priority": priority,
                    }
                )
    candidates.sort(
        key=lambda item: (item["end_dt"], item["filed_dt"], -item["priority"]),
        reverse=True,
    )
    return candidates


def _select_aligned_instant_pair(
    left_candidates: list[dict[str, Any]],
    right_candidates: list[dict[str, Any]],
    *,
    max_gap_days: int = DEBT_COMPONENT_ALIGNMENT_MAX_GAP_DAYS,
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
                -left["priority"],
                -right["priority"],
            )
            if best_key is None or pair_key > best_key:
                best_key = pair_key
                best_pair = (left, right)
    if best_pair is None:
        return None, None
    return best_pair


def _select_aligned_instant_candidate(
    candidates: list[dict[str, Any]],
    *,
    target_end_dt: date,
    max_gap_days: int = DEBT_COMPONENT_ALIGNMENT_MAX_GAP_DAYS,
) -> dict[str, Any] | None:
    best_candidate = None
    best_key = None
    for candidate in candidates:
        gap_days = abs((candidate["end_dt"] - target_end_dt).days)
        if gap_days > max_gap_days:
            continue
        candidate_key = (
            -gap_days,
            candidate["end_dt"],
            candidate["filed_dt"],
            -candidate["priority"],
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_candidate = candidate
    return best_candidate


def _candidate_approximately_matches(candidate: dict[str, Any] | None, value: float | None) -> bool:
    if candidate is None or value is None:
        return False
    return _approx_equal(float(candidate["value"]), float(value))


def _candidate_has_concept(candidate: dict[str, Any] | None, concept_name: str) -> bool:
    if candidate is None:
        return False
    return (candidate.get("meta") or {}).get("concept") == concept_name


def _has_any_concepts(companyfacts: dict, concepts: set[str] | list[str]) -> bool:
    for concept_name in concepts:
        if _candidate_units_map(companyfacts, concept_name):
            return True
    return False


def _concept_includes_capital_lease(meta: dict[str, Any] | None) -> bool:
    concept = str((meta or {}).get("concept") or "")
    return "CapitalLease" in concept


def _exact_finance_lease_stack(companyfacts: dict, as_of_date: str) -> dict[str, tuple[float, dict[str, Any]]]:
    stack: dict[str, tuple[float, dict[str, Any]]] = {}
    current_value, current_meta = _latest_instant_value(
        companyfacts,
        FINANCE_LEASE_CURRENT_CONCEPTS,
        as_of_date=as_of_date,
        unit_filter="USD",
    )
    noncurrent_value, noncurrent_meta = _latest_instant_value(
        companyfacts,
        FINANCE_LEASE_NONCURRENT_CONCEPTS,
        as_of_date=as_of_date,
        unit_filter="USD",
    )
    total_value, total_meta = _latest_instant_value(
        companyfacts,
        FINANCE_LEASE_TOTAL_CONCEPTS,
        as_of_date=as_of_date,
        unit_filter="USD",
    )
    if current_value is not None and current_meta is not None:
        stack["current"] = (float(current_value), current_meta)
    if noncurrent_value is not None and noncurrent_meta is not None:
        stack["noncurrent"] = (float(noncurrent_value), noncurrent_meta)
    if total_value is not None and total_meta is not None:
        stack["total"] = (float(total_value), total_meta)
    return stack
    return None, None


def _collect_duration_entries(companyfacts: dict, concept_name: str, as_of_date: str) -> list[dict[str, Any]]:
    units_map = _candidate_units_map(companyfacts, concept_name)
    if not units_map:
        return []
    as_of_dt = date.fromisoformat(as_of_date)
    rows: list[dict[str, Any]] = []
    for unit, entries in units_map.items():
        if unit.upper() != "USD":
            continue
        for entry in entries:
            start_dt = _parse_iso_date(entry.get("start"))
            end_dt = _parse_iso_date(entry.get("end"))
            filed_dt = _parse_iso_date(entry.get("filed"))
            value = entry.get("val")
            if start_dt is None or end_dt is None or value is None:
                continue
            if end_dt > as_of_dt:
                continue
            if filed_dt is not None and filed_dt > as_of_dt:
                continue
            if (as_of_dt - end_dt).days > MAX_SEC_FACT_AGE_DAYS:
                continue
            duration_days = max(1, (end_dt - start_dt).days + 1)
            rows.append(
                {
                    "concept": concept_name,
                    "start": start_dt,
                    "end": end_dt,
                    "filed": filed_dt or end_dt,
                    "value": float(value),
                    "fy": entry.get("fy"),
                    "fp": entry.get("fp"),
                    "frame": entry.get("frame"),
                    "form": entry.get("form"),
                    "duration_days": duration_days,
                }
            )
    rows.sort(key=lambda item: (item["end"], item["filed"], item["duration_days"]))
    return rows


def _duration_entry_rank(entry: dict[str, Any]) -> tuple[date, date, int, int]:
    return (
        entry["end"],
        entry["filed"],
        entry["duration_days"],
        1 if not entry.get("frame") else 0,
    )


def _latest_entry_by_fp(entries: list[dict[str, Any]], fp: str) -> dict[str, Any] | None:
    candidates = [entry for entry in entries if str(entry.get("fp") or "").upper() == fp.upper()]
    if not candidates:
        return None
    candidates.sort(key=_duration_entry_rank)
    return candidates[-1]


def _compute_ttm_from_concept(companyfacts: dict, concept_name: str, as_of_date: str) -> tuple[float | None, dict[str, Any] | None]:
    entries = _collect_duration_entries(companyfacts, concept_name, as_of_date)
    if not entries:
        return None, None

    latest = max(entries, key=_duration_entry_rank)
    latest_fp = str(latest.get("fp") or "").upper()
    if latest_fp == "FY" or latest["duration_days"] >= 300:
        return latest["value"], {
            "concept": concept_name,
            "mode": "latest_fy",
            "end": latest["end"].isoformat(),
            "filed": latest["filed"].isoformat(),
            "fy": latest.get("fy"),
            "fp": latest.get("fp"),
            "frame": latest.get("frame"),
            "form": latest.get("form"),
            "value": latest["value"],
            "formula": "latest_fiscal_year_value",
        }

    if latest_fp not in {"Q1", "Q2", "Q3"}:
        return None, None

    current_fy = latest.get("fy")
    if current_fy is None:
        return None, None
    try:
        prior_fy = int(current_fy) - 1
    except Exception:  # noqa: BLE001
        return None, None

    annual = None
    prior_same = None
    comparative_candidates: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("fy") == prior_fy and str(entry.get("fp") or "").upper() == "FY":
            if annual is None or _duration_entry_rank(entry) > _duration_entry_rank(annual):
                annual = entry
        if str(entry.get("fp") or "").upper() != latest_fp:
            continue
        if entry["end"] >= latest["end"]:
            continue
        gap_days = (latest["end"] - entry["end"]).days
        if 330 <= gap_days <= 380:
            comparative_candidates.append(entry)
            continue
        if entry.get("fy") == prior_fy:
            comparative_candidates.append(entry)

    if comparative_candidates:
        comparative_candidates.sort(
            key=lambda entry: (
                1 if 330 <= (latest["end"] - entry["end"]).days <= 380 else 0,
                -abs((latest["end"] - entry["end"]).days - 365),
                *_duration_entry_rank(entry),
            )
        )
        prior_same = comparative_candidates[-1]

    if annual is None or prior_same is None:
        return None, None

    ttm_value = float(latest["value"] + annual["value"] - prior_same["value"])
    return ttm_value, {
        "concept": concept_name,
        "mode": "ytd_plus_prior_fy_minus_prior_ytd",
        "latest": {
            "end": latest["end"].isoformat(),
            "filed": latest["filed"].isoformat(),
            "fy": latest.get("fy"),
            "fp": latest.get("fp"),
            "frame": latest.get("frame"),
            "form": latest.get("form"),
            "value": latest["value"],
        },
        "prior_fy": {
            "end": annual["end"].isoformat(),
            "filed": annual["filed"].isoformat(),
            "fy": annual.get("fy"),
            "fp": annual.get("fp"),
            "frame": annual.get("frame"),
            "form": annual.get("form"),
            "value": annual["value"],
        },
        "prior_same_period": {
            "end": prior_same["end"].isoformat(),
            "filed": prior_same["filed"].isoformat(),
            "fy": prior_same.get("fy"),
            "fp": prior_same.get("fp"),
            "frame": prior_same.get("frame"),
            "form": prior_same.get("form"),
            "value": prior_same["value"],
        },
        "formula": "latest_ytd + prior_fy - prior_same_period_ytd",
    }


