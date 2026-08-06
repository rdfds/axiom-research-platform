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


def _one_year_prior_as_of_date(as_of_date: str) -> str:
    current = date.fromisoformat(as_of_date)
    try:
        prior = current.replace(year=current.year - 1)
    except ValueError:
        prior = current.replace(month=2, day=28, year=current.year - 1)
    return prior.isoformat()


def _ttm_meta_latest_record(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(meta, dict):
        return None
    latest = meta.get("latest")
    if isinstance(latest, dict):
        return latest
    components = meta.get("components")
    if isinstance(components, list) and components:
        best_component = None
        best_key = None
        for component in components:
            candidate_key = _ttm_meta_rank(component, concept_priority=0)
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_component = _ttm_meta_latest_record(component)
        if isinstance(best_component, dict):
            return best_component
    return meta


def _ttm_meta_is_ytd_bridge(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("mode") == "ytd_plus_prior_fy_minus_prior_ytd":
        return True
    components = meta.get("components")
    if isinstance(components, list) and components:
        return all(_ttm_meta_is_ytd_bridge(component) for component in components)
    return False


def _ttm_meta_rank(meta: dict[str, Any] | None, *, concept_priority: int) -> tuple[date, date, int, int, int]:
    if not isinstance(meta, dict):
        return (date.min, date.min, 0, 0, -concept_priority)
    latest = _ttm_meta_latest_record(meta) or meta
    end_dt = _parse_iso_date(latest.get("end")) or date.min
    filed_dt = _parse_iso_date(latest.get("filed")) or end_dt
    unframed = 1 if not latest.get("frame") else 0
    ytd_mode = 1 if _ttm_meta_is_ytd_bridge(meta) else 0
    return (end_dt, filed_dt, unframed, ytd_mode, -concept_priority)


def _ttm_meta_is_latest_fy_only(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("mode") == "latest_fy":
        return True
    components = meta.get("components")
    if isinstance(components, list) and components:
        return all(_ttm_meta_is_latest_fy_only(component) for component in components)
    return False


def _ttm_meta_has_stale_latest_fy_component(meta: dict[str, Any] | None, reference_end: date | None) -> bool:
    if not isinstance(meta, dict) or reference_end is None:
        return False
    if meta.get("mode") == "latest_fy":
        latest = _ttm_meta_latest_record(meta)
        end_dt = _parse_iso_date(latest.get("end")) if isinstance(latest, dict) else None
        value = latest.get("value") if isinstance(latest, dict) else None
        if value is not None and abs(float(value)) < 1e-9:
            return False
        return end_dt is not None and end_dt < reference_end
    components = meta.get("components")
    if isinstance(components, list) and components:
        return any(_ttm_meta_has_stale_latest_fy_component(component, reference_end) for component in components)
    return False


def _latest_ttm_from_priority(companyfacts: dict, concepts: list[str], as_of_date: str) -> tuple[float | None, dict[str, Any] | None]:
    best_value = None
    best_meta = None
    best_key = None
    for priority, concept_name in enumerate(concepts):
        value, meta = _compute_ttm_from_concept(companyfacts, concept_name, as_of_date)
        if value is None:
            continue
        candidate_key = _ttm_meta_rank(meta, concept_priority=priority)
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_value = value
            best_meta = meta
    return best_value, best_meta


def _latest_nonnegative_ttm_from_priority(companyfacts: dict, concepts: list[str], as_of_date: str) -> tuple[float | None, dict[str, Any] | None]:
    best_value = None
    best_meta = None
    best_key = None
    for priority, concept_name in enumerate(concepts):
        value, meta = _compute_ttm_from_concept(companyfacts, concept_name, as_of_date)
        if value is None:
            continue
        if value >= 0:
            candidate_key = _ttm_meta_rank(meta, concept_priority=priority)
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_value = value
                best_meta = meta
    return best_value, best_meta


def _select_depreciation_ttm_candidate(companyfacts: dict, as_of_date: str) -> tuple[float | None, dict[str, Any] | None, str, list[str] | None]:
    best_value = None
    best_meta = None
    best_support_mode = "unsupported"
    best_quality_flags: list[str] | None = None
    best_key = None

    for group_priority, concept_group in enumerate(DEPRECIATION_TTM_CONCEPT_GROUPS):
        candidate_value = None
        candidate_meta = None
        candidate_support_mode = "exact"
        candidate_quality_flags: list[str] = []

        if len(concept_group) == 1:
            candidate_value, candidate_meta = _latest_ttm_from_priority(companyfacts, concept_group, as_of_date)
            if candidate_value is None:
                continue
            if concept_group[0] == "Depreciation":
                candidate_support_mode = "proxy_missing_component"
                candidate_quality_flags.append("partial_depreciation_without_full_amortization")
        else:
            parts = []
            parts_meta = []
            for concept_name in concept_group:
                part_value, part_meta = _latest_ttm_from_priority(companyfacts, [concept_name], as_of_date)
                if part_value is None:
                    parts = []
                    break
                parts.append(part_value)
                parts_meta.append(part_meta)
            if not parts:
                continue
            candidate_value = float(sum(parts))
            candidate_meta = {
                "mode": "sum_concepts",
                "components": parts_meta,
                "formula": "sum_component_ttm_values",
            }

        candidate_key = _ttm_meta_rank(candidate_meta, concept_priority=0) + (
            1 if candidate_support_mode == "exact" else 0,
            len(concept_group),
            -group_priority,
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_value = candidate_value
            best_meta = candidate_meta
            best_support_mode = candidate_support_mode
            best_quality_flags = candidate_quality_flags or None

    return best_value, best_meta, best_support_mode, best_quality_flags


def _approx_equal(a: float | None, b: float | None, *, rel_tol: float = 1e-3, abs_tol: float = 1e6) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= max(abs_tol, rel_tol * max(abs(float(a)), abs(float(b)), 1.0))


def _build_sec_core_metric(metric_name: str, companyfacts: dict, as_of_date: str) -> tuple[float | None, str, str | None, dict[str, Any] | None, list[str] | None]:
    if metric_name == "market.market_cap_provider_direct":
        return None, "unsupported", "recomputed_in_market_layer", {"formula": "price_spot * shares_outstanding_asof"}, ["recomputed_in_market_layer"]

    if metric_name == "operating.revenue_ttm_provider_direct":
        value, meta = _latest_nonnegative_ttm_from_priority(companyfacts, REVENUE_TTM_CONCEPTS, as_of_date)
        if value is None:
            return None, "unsupported", "sec_nonnegative_ttm_unavailable", None, ["sec_nonnegative_ttm_unavailable"]
        if meta.get("mode") == "latest_fy" and meta.get("frame"):
            return value, "proxy_missing_component", "framed_latest_fy_value", meta, ["framed_latest_fy_value"]
        return value, "exact", None, meta, None

    if metric_name == "operating.revenue_ttm_lag_1y":
        lagged_as_of_date = _one_year_prior_as_of_date(as_of_date)
        value, meta = _latest_nonnegative_ttm_from_priority(companyfacts, REVENUE_TTM_CONCEPTS, lagged_as_of_date)
        if value is None:
            return None, "unsupported", "sec_prior_year_nonnegative_ttm_unavailable", None, [
                "sec_prior_year_nonnegative_ttm_unavailable"
            ]
        enriched_meta = dict(meta or {})
        enriched_meta["lagged_as_of_date"] = lagged_as_of_date
        enriched_meta["formula"] = "latest_nonnegative_ttm_asof_prior_year"
        if meta.get("mode") == "latest_fy" and meta.get("frame"):
            return value, "proxy_missing_component", "framed_latest_fy_value", enriched_meta, ["framed_latest_fy_value"]
        return value, "exact", None, enriched_meta, None

    if metric_name == "earnings.net_income_ttm_provider_direct":
        value, meta = _latest_ttm_from_priority(companyfacts, NET_INCOME_TTM_CONCEPTS, as_of_date)
        if value is None:
            return None, "unsupported", "sec_ttm_unavailable", None, ["sec_ttm_unavailable"]
        return value, "exact", None, meta, None

    if metric_name == "operating.ebitda_ltm_provider_direct":
        operating_income, operating_meta = _latest_ttm_from_priority(companyfacts, OPERATING_INCOME_TTM_CONCEPTS, as_of_date)
        if operating_income is None:
            return None, "unsupported", "sec_operating_income_ttm_unavailable", None, ["sec_operating_income_ttm_unavailable"]
        depreciation_value, depreciation_meta, depreciation_support_mode, depreciation_quality_flags = _select_depreciation_ttm_candidate(
            companyfacts,
            as_of_date,
        )
        if depreciation_value is None:
            return None, "unsupported", "sec_depreciation_ttm_unavailable", None, ["sec_depreciation_ttm_unavailable"]
        operating_latest_meta = _ttm_meta_latest_record(operating_meta)
        operating_end = _parse_iso_date(operating_latest_meta.get("end")) if isinstance(operating_latest_meta, dict) else None
        depreciation_latest_meta = _ttm_meta_latest_record(depreciation_meta)
        depreciation_end = _parse_iso_date(depreciation_latest_meta.get("end")) if isinstance(depreciation_latest_meta, dict) else None
        if depreciation_support_mode == "exact" and operating_end is not None:
            latest_fy_only_stale = (
                _ttm_meta_is_latest_fy_only(depreciation_meta)
                and depreciation_end is not None
                and depreciation_end < operating_end
            )
            mixed_stale_components = _ttm_meta_has_stale_latest_fy_component(depreciation_meta, operating_end)
            if latest_fy_only_stale or mixed_stale_components:
                depreciation_support_mode = "proxy_missing_component"
                depreciation_quality_flags = list(depreciation_quality_flags or [])
                depreciation_quality_flags.append("stale_depreciation_amortization_bridge")
        return float(operating_income + depreciation_value), depreciation_support_mode, None, {
            "mode": "operating_income_plus_depreciation_amortization",
            "operating_income": operating_meta,
            "depreciation_amortization": depreciation_meta,
            "formula": "operating_income_ttm + depreciation_amortization_ttm",
        }, depreciation_quality_flags or None

    if metric_name == "liquidity.cash_and_short_term_investments_provider_direct":
        as_of_dt = date.fromisoformat(as_of_date)
        combined_candidates = _instant_candidates(
            companyfacts,
            COMBINED_CASH_STI_CONCEPTS,
            as_of_date=as_of_date,
            unit_filter="USD",
        )
        latest_combined_candidate = combined_candidates[0] if combined_candidates else None
        combined_cash_restricted_candidates = _instant_candidates(
            companyfacts,
            COMBINED_CASH_RESTRICTED_TOTAL_CONCEPTS,
            as_of_date=as_of_date,
            unit_filter="USD",
        )
        latest_combined_cash_restricted_candidate = (
            combined_cash_restricted_candidates[0] if combined_cash_restricted_candidates else None
        )
        cash_candidates = _instant_candidates(
            companyfacts,
            CASH_CONCEPTS,
            as_of_date=as_of_date,
            unit_filter="USD",
        )
        sti_candidates = _instant_candidates(
            companyfacts,
            STI_CONCEPTS,
            as_of_date=as_of_date,
            unit_filter="USD",
        )
        restricted_cash_candidates = _instant_candidates(
            companyfacts,
            RESTRICTED_CASH_TOTAL_CONCEPTS,
            as_of_date=as_of_date,
            unit_filter="USD",
        )
        cash_value = cash_candidates[0]["value"] if cash_candidates else None
        cash_meta = cash_candidates[0]["meta"] if cash_candidates else None
        sti_value = sti_candidates[0]["value"] if sti_candidates else None
        sti_meta = sti_candidates[0]["meta"] if sti_candidates else None
        aligned_cash_candidate, aligned_sti_candidate = _select_aligned_instant_pair(
            cash_candidates,
            sti_candidates,
            max_gap_days=CASH_COMPONENT_ALIGNMENT_MAX_GAP_DAYS,
        )
        aligned_cash_value = aligned_cash_candidate["value"] if aligned_cash_candidate is not None else None
        aligned_cash_meta = aligned_cash_candidate["meta"] if aligned_cash_candidate is not None else None
        aligned_sti_value = aligned_sti_candidate["value"] if aligned_sti_candidate is not None else None
        aligned_sti_meta = aligned_sti_candidate["meta"] if aligned_sti_candidate is not None else None
        if latest_combined_candidate is not None:
            combined_age_days = (as_of_dt - latest_combined_candidate["end_dt"]).days
            freshest_separate_end = max(
                [candidate["end_dt"] for candidate in (cash_candidates[:1] + sti_candidates[:1])] or [latest_combined_candidate["end_dt"]]
            )
            freshness_gap_days = (freshest_separate_end - latest_combined_candidate["end_dt"]).days
            if (
                combined_age_days <= EXACT_BALANCE_SHEET_MAX_AGE_DAYS
                and freshness_gap_days <= CASH_COMPONENT_ALIGNMENT_MAX_GAP_DAYS
            ):
                return latest_combined_candidate["value"], "exact", None, latest_combined_candidate["meta"], None
        if latest_combined_cash_restricted_candidate is not None:
            combined_cash_restricted_age_days = (as_of_dt - latest_combined_cash_restricted_candidate["end_dt"]).days
            if combined_cash_restricted_age_days <= EXACT_BALANCE_SHEET_MAX_AGE_DAYS:
                aligned_restricted_candidate = _select_aligned_instant_candidate(
                    restricted_cash_candidates,
                    target_end_dt=latest_combined_cash_restricted_candidate["end_dt"],
                    max_gap_days=CASH_COMPONENT_ALIGNMENT_MAX_GAP_DAYS,
                )
                aligned_combined_sti_candidate = _select_aligned_instant_candidate(
                    sti_candidates,
                    target_end_dt=latest_combined_cash_restricted_candidate["end_dt"],
                    max_gap_days=CASH_COMPONENT_ALIGNMENT_MAX_GAP_DAYS,
                )
                restricted_value = aligned_restricted_candidate["value"] if aligned_restricted_candidate is not None else 0.0
                unrestricted_cash_value = latest_combined_cash_restricted_candidate["value"] - restricted_value
                if unrestricted_cash_value >= -1e-6:
                    combined_cash_restricted_meta = {
                        "mode": "combined_cash_restricted_less_restricted_plus_short_term_investments",
                        "cash_and_restricted_total": latest_combined_cash_restricted_candidate["meta"],
                        "restricted_cash_adjustment": (
                            aligned_restricted_candidate["meta"]
                            if aligned_restricted_candidate is not None
                            else {
                                "mode": "infer_zero_restricted_cash_due_to_absent_current_restricted_cash_concept",
                                "value": 0.0,
                            }
                        ),
                        "short_term_investments": (
                            aligned_combined_sti_candidate["meta"] if aligned_combined_sti_candidate is not None else None
                        ),
                        "formula": "cash_and_restricted_total - restricted_cash + short_term_investments",
                    }
                    return (
                        float(unrestricted_cash_value + (aligned_combined_sti_candidate["value"] if aligned_combined_sti_candidate is not None else 0.0)),
                        "exact",
                        None,
                        combined_cash_restricted_meta,
                        None,
                    )
        if cash_value is None and sti_value is None:
            return None, "unsupported", "sec_cash_components_unavailable", None, ["sec_cash_components_unavailable"]
        if aligned_cash_value is not None and aligned_sti_value is not None:
            pair_end_dt = max(aligned_cash_candidate["end_dt"], aligned_sti_candidate["end_dt"])
            pair_age_days = (as_of_dt - pair_end_dt).days
            if (
                pair_age_days <= EXACT_BALANCE_SHEET_MAX_AGE_DAYS
                and aligned_cash_candidate is cash_candidates[0]
                and aligned_sti_candidate is sti_candidates[0]
            ):
                return float(aligned_cash_value + aligned_sti_value), "exact", None, {
                    "mode": "cash_plus_short_term_investments",
                    "cash": aligned_cash_meta,
                    "short_term_investments": aligned_sti_meta,
                    "formula": "cash + short_term_investments",
                }, None
        if cash_value is not None and sti_value is not None:
            cash_end = _parse_iso_date((cash_meta or {}).get("end"))
            sti_end = _parse_iso_date((sti_meta or {}).get("end"))
            gap_days = abs((cash_end - sti_end).days) if cash_end is not None and sti_end is not None else None
            if gap_days is not None and gap_days > CASH_COMPONENT_ALIGNMENT_MAX_GAP_DAYS:
                if cash_end is not None and sti_end is not None and cash_end > sti_end:
                    return float(cash_value), "proxy_missing_component", "short_term_investment_component_stale", {
                        "mode": "partial_cash_stack",
                        "cash": cash_meta,
                        "short_term_investments": sti_meta,
                        "alignment_gap_days": gap_days,
                        "formula": "latest_cash_only_due_to_stale_short_term_investments",
                    }, ["short_term_investment_component_stale"]
                if cash_end is not None and sti_end is not None and sti_end > cash_end:
                    return float(sti_value), "proxy_missing_component", "cash_component_stale", {
                        "mode": "partial_short_term_investments_stack",
                        "cash": cash_meta,
                        "short_term_investments": sti_meta,
                        "alignment_gap_days": gap_days,
                        "formula": "latest_short_term_investments_only_due_to_stale_cash",
                    }, ["cash_component_stale"]
            return float(cash_value + sti_value), "proxy_missing_component", "cash_component_period_mismatch", {
                "mode": "cash_plus_short_term_investments_period_mismatch",
                "cash": cash_meta,
                "short_term_investments": sti_meta,
                "alignment_gap_days": gap_days,
                "formula": "latest_cash + latest_short_term_investments",
            }, ["cash_component_period_mismatch"]
        return float(cash_value or sti_value or 0.0), "proxy_missing_component", "cash_or_sti_component_missing", {
            "mode": "partial_cash_stack",
            "cash": cash_meta,
            "short_term_investments": sti_meta,
            "formula": "partial_cash_stack",
        }, ["cash_or_sti_component_missing"]

    if metric_name == "capital_structure.total_debt_provider_direct":
        combined_value, combined_meta = _latest_instant_value(companyfacts, TOTAL_DEBT_COMBINED_CONCEPTS, as_of_date=as_of_date, unit_filter="USD")
        short_term_borrowings_candidates = _instant_candidates(
            companyfacts,
            SHORT_TERM_BORROWINGS_CONCEPTS,
            as_of_date=as_of_date,
            unit_filter="USD",
        )
        current_candidates = _instant_candidates(
            companyfacts,
            DEBT_CURRENT_CONCEPTS,
            as_of_date=as_of_date,
            unit_filter="USD",
        )
        noncurrent_candidates = _instant_candidates(
            companyfacts,
            DEBT_NONCURRENT_CONCEPTS,
            as_of_date=as_of_date,
            unit_filter="USD",
        )
        short_term_borrowings_value = short_term_borrowings_candidates[0]["value"] if short_term_borrowings_candidates else None
        short_term_borrowings_meta = short_term_borrowings_candidates[0]["meta"] if short_term_borrowings_candidates else None
        current_value = current_candidates[0]["value"] if current_candidates else None
        current_meta = current_candidates[0]["meta"] if current_candidates else None
        noncurrent_value = noncurrent_candidates[0]["value"] if noncurrent_candidates else None
        noncurrent_meta = noncurrent_candidates[0]["meta"] if noncurrent_candidates else None
        aligned_current_candidate, aligned_noncurrent_candidate = _select_aligned_instant_pair(
            current_candidates,
            noncurrent_candidates,
        )
        aligned_current_value = aligned_current_candidate["value"] if aligned_current_candidate is not None else None
        aligned_current_meta = aligned_current_candidate["meta"] if aligned_current_candidate is not None else None
        aligned_noncurrent_value = aligned_noncurrent_candidate["value"] if aligned_noncurrent_candidate is not None else None
        aligned_noncurrent_meta = aligned_noncurrent_candidate["meta"] if aligned_noncurrent_candidate is not None else None
        aligned_pair_end_dt = max(aligned_current_candidate["end_dt"], aligned_noncurrent_candidate["end_dt"]) if aligned_current_candidate is not None and aligned_noncurrent_candidate is not None else None
        aligned_short_term_candidate = (
            _select_aligned_instant_candidate(
                short_term_borrowings_candidates,
                target_end_dt=aligned_pair_end_dt,
            )
            if aligned_pair_end_dt is not None
            else None
        )
        aligned_short_term_with_noncurrent_candidate, aligned_noncurrent_with_short_term_candidate = _select_aligned_instant_pair(
            short_term_borrowings_candidates,
            noncurrent_candidates,
        )
        aligned_short_term_borrowings_value = aligned_short_term_candidate["value"] if aligned_short_term_candidate is not None else None
        aligned_short_term_borrowings_meta = aligned_short_term_candidate["meta"] if aligned_short_term_candidate is not None else None
        aligned_short_term_to_noncurrent_value = (
            aligned_short_term_with_noncurrent_candidate["value"]
            if aligned_short_term_with_noncurrent_candidate is not None
            else None
        )
        aligned_short_term_to_noncurrent_meta = (
            aligned_short_term_with_noncurrent_candidate["meta"]
            if aligned_short_term_with_noncurrent_candidate is not None
            else None
        )
        aligned_noncurrent_with_short_term_value = (
            aligned_noncurrent_with_short_term_candidate["value"]
            if aligned_noncurrent_with_short_term_candidate is not None
            else None
        )
        aligned_noncurrent_with_short_term_meta = (
            aligned_noncurrent_with_short_term_candidate["meta"]
            if aligned_noncurrent_with_short_term_candidate is not None
            else None
        )
        finance_lease_stack = _exact_finance_lease_stack(companyfacts, as_of_date)
        noncurrent_total_only_candidate = next(
            (
                candidate
                for candidate in noncurrent_candidates
                if ((candidate.get("meta") or {}).get("concept") in NONCURRENT_TOTAL_ONLY_EXACT_CONCEPTS)
            ),
            None,
        )

        def _finance_adjustment_for_debt_stack() -> tuple[float | None, dict[str, Any] | None]:
            combined_overlap = _concept_includes_capital_lease(combined_meta)
            current_overlap = _concept_includes_capital_lease(current_meta)
            noncurrent_overlap = _concept_includes_capital_lease(noncurrent_meta)
            finance_lease_concepts_present = _has_any_concepts(companyfacts, FINANCE_LEASE_ANY_CONCEPTS)
            if combined_overlap:
                if "total" in finance_lease_stack:
                    value, meta = finance_lease_stack["total"]
                    return value, {
                        "mode": "subtract_finance_lease_total_from_combined_debt",
                        "finance_lease_total": meta,
                    }
                if not finance_lease_concepts_present:
                    return 0.0, {
                        "mode": "infer_zero_finance_lease_adjustment_due_to_absent_finance_lease_concepts",
                    }
                return None, None
            if current_overlap and noncurrent_overlap:
                if "total" in finance_lease_stack:
                    value, meta = finance_lease_stack["total"]
                    return value, {
                        "mode": "subtract_finance_lease_total_from_current_and_noncurrent_debt",
                        "finance_lease_total": meta,
                    }
                if "current" in finance_lease_stack and "noncurrent" in finance_lease_stack:
                    current_finance_value, current_finance_meta = finance_lease_stack["current"]
                    noncurrent_finance_value, noncurrent_finance_meta = finance_lease_stack["noncurrent"]
                    return current_finance_value + noncurrent_finance_value, {
                        "mode": "subtract_finance_lease_current_plus_noncurrent",
                        "finance_lease_current": current_finance_meta,
                        "finance_lease_noncurrent": noncurrent_finance_meta,
                    }
                if not finance_lease_concepts_present:
                    return 0.0, {
                        "mode": "infer_zero_finance_lease_adjustment_due_to_absent_finance_lease_concepts",
                    }
                return None, None
            if current_overlap:
                if "current" in finance_lease_stack:
                    value, meta = finance_lease_stack["current"]
                    return value, {
                        "mode": "subtract_finance_lease_current",
                        "finance_lease_current": meta,
                    }
                if not finance_lease_concepts_present:
                    return 0.0, {
                        "mode": "infer_zero_finance_lease_adjustment_due_to_absent_finance_lease_concepts",
                    }
                return None, None
            if noncurrent_overlap:
                if "noncurrent" in finance_lease_stack:
                    value, meta = finance_lease_stack["noncurrent"]
                    return value, {
                        "mode": "subtract_finance_lease_noncurrent",
                        "finance_lease_noncurrent": meta,
                    }
                if not finance_lease_concepts_present:
                    return 0.0, {
                        "mode": "infer_zero_finance_lease_adjustment_due_to_absent_finance_lease_concepts",
                    }
                return None, None
            return 0.0, None

        finance_adjustment_value, finance_adjustment_meta = _finance_adjustment_for_debt_stack()
        short_term_borrowings_duplicate_current = (
            aligned_short_term_candidate is not None
            and aligned_current_candidate is not None
            and aligned_short_term_candidate["end_dt"] == aligned_current_candidate["end_dt"]
            and _candidate_approximately_matches(aligned_short_term_candidate, aligned_current_value)
        )

        def _apply_finance_adjustment(
            base_value: float,
            *,
            support_mode: str,
            missing_reason: str | None,
            component_breakdown: dict[str, Any],
            quality_flags: list[str] | None,
        ) -> tuple[float, str, str | None, dict[str, Any], list[str] | None]:
            if finance_adjustment_value == 0.0 and finance_adjustment_meta is None:
                return base_value, support_mode, missing_reason, component_breakdown, quality_flags
            if finance_adjustment_value is None:
                flags = list(quality_flags or [])
                if "finance_lease_adjustment_unavailable" not in flags:
                    flags.append("finance_lease_adjustment_unavailable")
                breakdown = {
                    **component_breakdown,
                    "capital_lease_overlap_detected": True,
                    "formula_before_finance_adjustment": component_breakdown.get("formula"),
                }
                return base_value, "proxy_missing_component", "finance_lease_adjustment_unavailable", breakdown, flags
            adjusted_value = float(base_value - finance_adjustment_value)
            if adjusted_value < 0:
                flags = list(quality_flags or [])
                if "finance_lease_adjustment_exceeds_debt" not in flags:
                    flags.append("finance_lease_adjustment_exceeds_debt")
                breakdown = {
                    **component_breakdown,
                    "finance_lease_adjustment": finance_adjustment_meta,
                    "finance_lease_adjustment_value": finance_adjustment_value,
                    "formula_before_finance_adjustment": component_breakdown.get("formula"),
                }
                return base_value, "proxy_missing_component", "finance_lease_adjustment_exceeds_debt", breakdown, flags
            breakdown = {
                **component_breakdown,
                "finance_lease_adjustment": finance_adjustment_meta,
                "finance_lease_adjustment_value": finance_adjustment_value,
                "formula_before_finance_adjustment": component_breakdown.get("formula"),
                "formula": f"{component_breakdown.get('formula')} - finance_lease_liabilities_exact",
            }
            return adjusted_value, support_mode, missing_reason, breakdown, quality_flags

        if combined_value is not None:
            if (
                aligned_short_term_borrowings_value is not None
                and aligned_current_value is not None
                and aligned_noncurrent_value is not None
                and _approx_equal(combined_value, float(aligned_current_value + aligned_noncurrent_value))
                and not short_term_borrowings_duplicate_current
            ):
                total_value = float(combined_value + aligned_short_term_borrowings_value)
                total_value, support_mode, missing_reason, component_breakdown, quality_flags = _apply_finance_adjustment(
                    total_value,
                    support_mode="exact",
                    missing_reason=None,
                    component_breakdown={
                        "mode": "combined_debt_plus_short_term_borrowings",
                        "combined_debt": combined_meta,
                        "current": aligned_current_meta,
                        "noncurrent": aligned_noncurrent_meta,
                        "short_term_borrowings": aligned_short_term_borrowings_meta,
                        "formula": "combined_debt + short_term_borrowings",
                    },
                    quality_flags=None,
                )
                return total_value, support_mode, missing_reason, component_breakdown, quality_flags
            total_value, support_mode, missing_reason, component_breakdown, quality_flags = _apply_finance_adjustment(
                combined_value,
                support_mode="exact",
                missing_reason=None,
                component_breakdown={
                    "mode": "combined_debt",
                    "combined_debt": combined_meta,
                    "formula": "combined_debt",
                },
                quality_flags=None,
            )
            return total_value, support_mode, missing_reason, component_breakdown, quality_flags
        if (
            current_value is None
            and short_term_borrowings_value is None
            and noncurrent_total_only_candidate is not None
        ):
            noncurrent_total_meta = noncurrent_total_only_candidate["meta"]
            noncurrent_total_concept = noncurrent_total_meta.get("concept")
            if noncurrent_total_concept == "LongTermDebt":
                component_breakdown = {
                    "mode": "long_term_debt_total_only",
                    "long_term_debt_total": noncurrent_total_meta,
                    "formula": "exact_long_term_debt_total",
                }
            else:
                component_breakdown = {
                    "mode": "noncurrent_debt_total_only",
                    "noncurrent_debt_total": noncurrent_total_meta,
                    "formula": "exact_noncurrent_debt_total",
                }
            total_value, support_mode, missing_reason, component_breakdown, quality_flags = _apply_finance_adjustment(
                float(noncurrent_total_only_candidate["value"]),
                support_mode="exact",
                missing_reason=None,
                component_breakdown=component_breakdown,
                quality_flags=None,
            )
            return total_value, support_mode, missing_reason, component_breakdown, quality_flags
        if current_value is None and noncurrent_value is None:
            if short_term_borrowings_value is not None:
                return float(short_term_borrowings_value), "proxy_missing_component", "long_term_debt_components_missing", {
                    "mode": "short_term_borrowings_only",
                    "short_term_borrowings": short_term_borrowings_meta,
                    "formula": "short_term_borrowings_only",
                }, ["long_term_debt_components_missing"]
            if not _has_any_concepts(companyfacts, DEBT_BALANCE_CONCEPTS):
                return 0.0, "proxy_missing_component", "no_debt_balance_concepts_present", {
                    "mode": "no_debt_balance_concepts_present",
                    "formula": "infer_zero_debt_when_no_balance_sheet_debt_concepts_are_present",
                }, ["no_debt_balance_concepts_present"]
            return None, "unsupported", "sec_debt_components_unavailable", None, ["sec_debt_components_unavailable"]
        if aligned_current_value is not None and aligned_noncurrent_value is not None:
            if (
                short_term_borrowings_duplicate_current
                and noncurrent_total_only_candidate is not None
                and aligned_current_candidate is not None
                and noncurrent_total_only_candidate["end_dt"] == aligned_current_candidate["end_dt"]
            ):
                total_value, support_mode, missing_reason, component_breakdown, quality_flags = _apply_finance_adjustment(
                    float(noncurrent_total_only_candidate["value"]),
                    support_mode="exact",
                    missing_reason=None,
                    component_breakdown={
                        "mode": "long_term_debt_with_overlapping_short_term_borrowings",
                        "current": aligned_current_meta,
                        "noncurrent": aligned_noncurrent_meta,
                        "short_term_borrowings": aligned_short_term_borrowings_meta,
                        "long_term_debt_total": noncurrent_total_only_candidate["meta"],
                        "formula": "exact_long_term_debt_total_due_to_current_short_term_overlap",
                    },
                    quality_flags=None,
                )
                return total_value, support_mode, missing_reason, component_breakdown, quality_flags
            total_value = float(aligned_current_value + aligned_noncurrent_value)
            component_breakdown = {
                "mode": "current_plus_noncurrent_debt",
                "current": aligned_current_meta,
                "noncurrent": aligned_noncurrent_meta,
                "formula": "debt_current + debt_noncurrent",
            }
            if (
                aligned_short_term_borrowings_value is not None
                and (
                    (aligned_current_meta or {}).get("concept") == "LongTermDebtCurrent"
                    or (aligned_short_term_borrowings_meta or {}).get("concept") in ADDITIVE_SHORT_TERM_BORROWINGS_CONCEPTS
                )
                and not short_term_borrowings_duplicate_current
            ):
                total_value += float(aligned_short_term_borrowings_value)
                component_breakdown = {
                    "mode": "current_plus_noncurrent_debt_plus_short_term_borrowings",
                    "current": aligned_current_meta,
                    "noncurrent": aligned_noncurrent_meta,
                    "short_term_borrowings": aligned_short_term_borrowings_meta,
                    "formula": "debt_current + debt_noncurrent + short_term_borrowings",
                }
            total_value, support_mode, missing_reason, component_breakdown, quality_flags = _apply_finance_adjustment(
                total_value,
                support_mode="exact",
                missing_reason=None,
                component_breakdown=component_breakdown,
                quality_flags=None,
            )
            return total_value, support_mode, missing_reason, component_breakdown, quality_flags
        if (
            aligned_current_value is None
            and aligned_noncurrent_with_short_term_value is not None
            and aligned_short_term_to_noncurrent_value is not None
        ):
            total_value, support_mode, missing_reason, component_breakdown, quality_flags = _apply_finance_adjustment(
                float(aligned_noncurrent_with_short_term_value + aligned_short_term_to_noncurrent_value),
                support_mode="exact",
                missing_reason=None,
                component_breakdown={
                    "mode": "short_term_borrowings_plus_noncurrent_debt",
                    "noncurrent": aligned_noncurrent_with_short_term_meta,
                    "short_term_borrowings": aligned_short_term_to_noncurrent_meta,
                    "formula": "short_term_borrowings + debt_noncurrent",
                },
                quality_flags=None,
            )
            return total_value, support_mode, missing_reason, component_breakdown, quality_flags
        if current_value is not None and noncurrent_value is not None:
            partial_value = float(current_value + noncurrent_value)
            if short_term_borrowings_value is not None:
                partial_value += float(short_term_borrowings_value)
            partial_breakdown = {
                "mode": "partial_debt_stack_period_mismatch",
                "current": current_meta,
                "noncurrent": noncurrent_meta,
                "short_term_borrowings": short_term_borrowings_meta,
                "formula": "latest_current + latest_noncurrent + optional_short_term_borrowings",
            }
            partial_value, support_mode, missing_reason, component_breakdown, quality_flags = _apply_finance_adjustment(
                partial_value,
                support_mode="proxy_missing_component",
                missing_reason="debt_component_period_mismatch",
                component_breakdown=partial_breakdown,
                quality_flags=["debt_component_period_mismatch"],
            )
            return partial_value, support_mode, missing_reason, component_breakdown, quality_flags
        partial_value = float(current_value or noncurrent_value or 0.0)
        if short_term_borrowings_value is not None:
            partial_value += float(short_term_borrowings_value)
        partial_breakdown = {
            "mode": "partial_debt_stack",
            "current": current_meta,
            "noncurrent": noncurrent_meta,
            "short_term_borrowings": short_term_borrowings_meta,
            "formula": "partial_debt_stack_with_short_term_borrowings",
        }
        partial_value, support_mode, missing_reason, component_breakdown, quality_flags = _apply_finance_adjustment(
            partial_value,
            support_mode="proxy_missing_component",
            missing_reason="debt_component_missing",
            component_breakdown=partial_breakdown,
            quality_flags=["debt_component_missing"],
        )
        return partial_value, support_mode, missing_reason, component_breakdown, quality_flags

    return None, "unsupported", "unsupported_metric", None, ["unsupported_metric"]


def _build_legacy_provider_metric(
    metric_name: str,
    provider_row: pd.Series | None,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    unit: str,
) -> Dict[str, Any]:
    source_column = LEGACY_PROVIDER_SOURCE_COLUMNS.get(metric_name)
    if source_column is None:
        return _feature_template(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            support_mode="unsupported",
            value=None,
            unit=unit,
            missing_reason="provider_direct_field_not_defined_for_metric",
            component_breakdown={"source_column": None},
            quality_flags=["provider_direct_field_not_defined_for_metric"],
        )
    if provider_row is None:
        return _feature_template(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            support_mode="unsupported",
            value=None,
            unit=unit,
            missing_reason="provider_row_unavailable",
            component_breakdown={"source_column": source_column},
            quality_flags=["provider_row_unavailable"],
        )

    raw_value = provider_row.get(source_column)
    if pd.isna(raw_value):
        return _feature_template(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            support_mode="unsupported",
            value=None,
            unit=unit,
            missing_reason="provider_field_unavailable",
            component_breakdown={
                "source_column": source_column,
                "reference_instrument": provider_row.get("Instrument"),
            },
            quality_flags=["provider_field_unavailable"],
        )

    return _feature_template(
        metric_name=metric_name,
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        support_mode="exact",
        value=float(raw_value),
        unit=unit,
        missing_reason=None,
        component_breakdown={
            "provider_field": source_column,
            "reference_instrument": provider_row.get("Instrument"),
            "provider_company_name": provider_row.get("Company Common Name"),
            "formula": "legacy_provider_direct_field",
        },
        quality_flags=["legacy_provider_direct_fallback"],
    )


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
        return _feature_template(
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
            provenance_artifact_type="ComputedMetric",
            primary_source_basis="computed_metric",
            input_layer_bucket_reason="computed_from_reference_metrics",
        )

    if denominator is None:
        if not allow_numerator_only:
            return _feature_template(
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
                provenance_artifact_type="ComputedMetric",
                primary_source_basis="computed_metric",
                input_layer_bucket_reason="computed_from_reference_metrics",
            )
        support_mode = "exact" if not non_exact_components else "proxy_missing_component"
        quality_flags = None if not non_exact_components else [f"component_not_exact:{name}" for name in non_exact_components]
        return _feature_template(
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
            provenance_artifact_type="ComputedMetric",
            primary_source_basis="computed_metric",
            input_layer_bucket_reason="computed_from_reference_metrics",
        )

    if denominator <= 0:
        return _feature_template(
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
            provenance_artifact_type="ComputedMetric",
            primary_source_basis="computed_metric",
            input_layer_bucket_reason="computed_from_reference_metrics",
        )

    support_mode = "exact" if not non_exact_components else "proxy_missing_component"
    quality_flags = None if not non_exact_components else [f"component_not_exact:{name}" for name in non_exact_components]
    return _feature_template(
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
        provenance_artifact_type="ComputedMetric",
        primary_source_basis="computed_metric",
        input_layer_bucket_reason="computed_from_reference_metrics",
    )


def iter_snapshot_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _metric_value(features: Dict[str, Any], metric_name: str) -> float | None:
    node = features.get(metric_name) or {}
    value = node.get("value")
    return None if value is None else float(value)


def _metric_support(features: Dict[str, Any], metric_name: str) -> str:
    node = features.get(metric_name) or {}
    return node.get("support_mode") or "missing_metric"


@contextmanager
def _company_processing_guard(timeout_seconds: float | None):
    if (
        timeout_seconds is None
        or timeout_seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(signum, frame):  # noqa: ARG001
        raise _CompanyProcessingTimeout(f"company_processing_timeout_after_{timeout_seconds:g}s")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_fail_open_metric_set(
    *,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    error_type: str,
    error_message: str,
) -> Dict[str, Dict[str, Any]]:
    error_text = str(error_message).strip()[:240]
    missing_reason = "company_processing_timeout" if error_type == "company_processing_timeout" else "company_processing_failed"
    breakdown = {
        "error_type": error_type,
        "error_message": error_text,
    }
    quality_flags = ["company_processing_fail_open", error_type]
    return {
        metric_name: _build_metric_from_value(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            unit=spec["unit"],
            value=None,
            support_mode="unsupported",
            missing_reason=missing_reason,
            component_breakdown=breakdown,
            quality_flags=quality_flags,
            primary_source_basis="input_layer_fail_open",
            provenance_artifact_type="DerivedComputation",
            input_layer_bucket_reason="company_processing_fail_open",
        )
        for metric_name, spec in ALL_OUTPUT_METRIC_SPECS.items()
    }


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.snapshot_path)
    taxonomy_reference_path = Path(args.taxonomy_reference_path)
    entity_identifier_path = Path(args.entity_identifier_path)
    companyfacts_root = _resolve_local_optional_path(args.companyfacts_root, DEFAULT_LOCAL_COMPANYFACTS_ROOT)
    raw_timeseries_path = _resolve_local_optional_path(args.raw_timeseries_path, DEFAULT_LOCAL_RAW_TIMESERIES_PATH)
    crsp_market_cache_path = Path(args.crsp_market_cache_path) if args.crsp_market_cache_path else None
    crsp_daily_root = Path(args.crsp_daily_root) if args.crsp_daily_root else (
        DEFAULT_LOCAL_CRSP_DAILY_ROOT if DEFAULT_LOCAL_CRSP_DAILY_ROOT and Path(DEFAULT_LOCAL_CRSP_DAILY_ROOT).exists() else None
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot_rows = list(iter_snapshot_rows(snapshot_path))
    snapshot_company_ids = {str(row.get("company_id")) for row in snapshot_rows if row.get("company_id")}

    provider = _provider_reference_map(taxonomy_reference_path, entity_identifier_path)
    if snapshot_company_ids:
        provider = provider[provider["entity_id"].astype(str).isin(snapshot_company_ids)].copy()
    provider_by_entity = provider.set_index("entity_id").to_dict(orient="index")
    companyfacts_cache: dict[str, dict | None] = {}
    permno_by_entity: dict[str, str] = {}
    price_by_permno: dict[str, pd.DataFrame] = {}
    if (
        _market_permno_map is not None
        and (_build_market_cap_metric_from_companyfacts is not None or _build_market_cap_metric is not None)
    ):
        permnos = _market_permno_map(entity_identifier_path)
        if snapshot_company_ids:
            permnos = permnos[permnos["entity_id"].astype(str).isin(snapshot_company_ids)].copy()
        permno_by_entity = permnos.set_index("entity_id")["permno"].to_dict()
        as_of_times = [
            pd.Timestamp(row["as_of_time"]).tz_convert("UTC").normalize()
            for row in snapshot_rows
            if row.get("as_of_time")
        ]
        min_asof_date = min(as_of_times) if as_of_times else pd.Timestamp("1970-01-01", tz="UTC")
        max_asof_date = max(as_of_times) if as_of_times else pd.Timestamp("1970-01-01", tz="UTC")
        market_price_builder = None
        price_history = pd.DataFrame()
        exact_market_price_source = crsp_market_cache_path or crsp_daily_root
        if (
            crsp_market_cache_path is not None
            and _load_crsp_market_cache is not None
            and _build_price_metrics_from_crsp is not None
            and _build_market_cap_metric is not None
        ):
            price_history = _load_crsp_market_cache(crsp_market_cache_path, permnos["permno"].tolist())
            market_price_builder = _build_price_metrics_from_crsp
        elif (
            crsp_daily_root is not None
            and _load_crsp_daily_from_repo is not None
            and _build_price_metrics_from_crsp is not None
            and _build_market_cap_metric is not None
        ):
            price_history = _load_crsp_daily_from_repo(
                crsp_daily_root,
                permnos["permno"].tolist(),
                min_asof_date=min_asof_date,
                max_asof_date=max_asof_date,
            )
            market_price_builder = _build_price_metrics_from_crsp
        elif (
            args.allow_monthly_market_proxy
            and raw_timeseries_path is not None
            and _load_price_history is not None
            and _build_price_metrics is not None
        ):
            price_history = _load_price_history(raw_timeseries_path, permnos["permno"].tolist())
            market_price_builder = _build_price_metrics
        else:
            market_price_builder = _build_price_metrics_from_crsp or _build_price_metrics
        price_by_permno = {
            permno: frame.reset_index(drop=True)
            for permno, frame in price_history.groupby("permno")
        }
    else:
        market_price_builder = None
    sec_filing_cache_root = Path(args.sec_filing_cache_root)
    sec_filing_cache_root.mkdir(parents=True, exist_ok=True)
    sec_session = (
        _sec_session()
        if (args.enable_sec_filing_debt_repair and companyfacts_root is not None and _sec_session is not None)
        else None
    )
    computed_at = _now_iso()
    counters: Counter[str] = Counter()

    with out_path.open("w") as out_handle:
        for row in snapshot_rows:
            entity_id = row.get("company_id")
            provider_row = provider_by_entity.get(entity_id)
            features = row.setdefault("features", {})
            as_of_time = row.get("as_of_time")
            as_of_date = as_of_time[:10]
            companyfacts = None
            companyfacts_path = (companyfacts_root / f"CIK{entity_id}.json") if companyfacts_root is not None else None
            row_metrics: Dict[str, Dict[str, Any]] = {}
            sec_filing_repair_applied = False

            try:
                with _company_processing_guard(args.company_processing_timeout_seconds):
                    if companyfacts_root is not None:
                        companyfacts = companyfacts_cache.get(entity_id)
                        if entity_id not in companyfacts_cache:
                            companyfacts = _load_companyfacts(companyfacts_path)
                            companyfacts_cache[entity_id] = companyfacts

                    for metric_name, spec in DIRECT_METRIC_SPECS.items():
                        provider_node = _build_legacy_provider_metric(
                            metric_name=metric_name,
                            provider_row=provider_row,
                            as_of_time=as_of_time,
                            computed_at=computed_at,
                            provenance_source=str(taxonomy_reference_path),
                            unit=spec["unit"],
                        )
                        if metric_name == "market.market_cap_provider_direct":
                            market_cap_node = None
                            if (
                                market_price_builder is not None
                                and (
                                    exact_market_price_source is not None
                                    or companyfacts is not None
                                )
                            ):
                                permno = permno_by_entity.get(entity_id)
                                price_metrics = market_price_builder(
                                    permno=permno,
                                    price_history=price_by_permno.get(permno),
                                    as_of_time=as_of_time,
                                    computed_at=computed_at,
                                    provenance_source=str(
                                        crsp_market_cache_path
                                        or crsp_daily_root
                                        or raw_timeseries_path
                                        or "market_timeseries_unavailable"
                                    ),
                                )
                                if (
                                    market_price_builder is _build_price_metrics_from_crsp
                                    and exact_market_price_source is not None
                                    and _build_market_cap_metric is not None
                                ):
                                    market_cap_node = _build_market_cap_metric(
                                        price_history=price_by_permno.get(permno),
                                        price_node=price_metrics["market.price_spot"],
                                        issuer_shares_outstanding=None,
                                        issuer_shares_meta=None,
                                        as_of_time=as_of_time,
                                        computed_at=computed_at,
                                        provenance_source=str(exact_market_price_source),
                                    )
                                if (
                                    market_cap_node is None
                                    and companyfacts is not None
                                    and _build_market_cap_metric_from_companyfacts is not None
                                ):
                                    market_cap_node = _build_market_cap_metric_from_companyfacts(
                                        companyfacts=companyfacts,
                                        price_node=price_metrics["market.price_spot"],
                                        as_of_time=as_of_time,
                                        computed_at=computed_at,
                                        companyfacts_path=companyfacts_path,
                                    )
                            node = _select_preferred_direct_metric(
                                metric_name=metric_name,
                                sec_or_market_node=market_cap_node,
                                provider_node=provider_node,
                            )
                        elif companyfacts is not None:
                            value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
                                metric_name,
                                companyfacts,
                                as_of_date,
                            )
                            sec_node = _build_metric_from_value(
                                metric_name=metric_name,
                                as_of_time=as_of_time,
                                computed_at=computed_at,
                                provenance_source=str(companyfacts_path),
                                unit=spec["unit"],
                                value=value,
                                support_mode=support_mode,
                                missing_reason=missing_reason,
                                component_breakdown=component_breakdown,
                                quality_flags=quality_flags,
                                primary_source_basis="sec_companyfacts",
                                provenance_artifact_type="SecCompanyFacts",
                                input_layer_bucket_reason="sec_companyfacts_asof",
                            )
                            node = _select_preferred_direct_metric(
                                metric_name=metric_name,
                                sec_or_market_node=sec_node,
                                provider_node=provider_node,
                            )
                        else:
                            node = provider_node
                        row_metrics[metric_name] = node

                    if (
                        companyfacts is not None
                        and sec_session is not None
                        and _repair_total_debt_from_sec_filing is not None
                    ):
                        temp_row = dict(row)
                        temp_features = dict(features)
                        temp_features.update(row_metrics)
                        temp_row["features"] = temp_features
                        if _repair_total_debt_from_sec_filing(
                            row=temp_row,
                            computed_at=computed_at,
                            provenance_source=str(companyfacts_path),
                            session=sec_session,
                            cache_dir=sec_filing_cache_root,
                            companyfacts=companyfacts,
                        ):
                            row_metrics["capital_structure.total_debt_provider_direct"] = temp_row["features"]["capital_structure.total_debt_provider_direct"]
                            sec_filing_repair_applied = True

                    revenue = _metric_value(row_metrics, "operating.revenue_ttm_provider_direct")
                    ebitda = _metric_value(row_metrics, "operating.ebitda_ltm_provider_direct")
                    net_income = _metric_value(row_metrics, "earnings.net_income_ttm_provider_direct")
                    cash_sti = _metric_value(row_metrics, "liquidity.cash_and_short_term_investments_provider_direct")
                    total_debt = _metric_value(row_metrics, "capital_structure.total_debt_provider_direct")

                    revenue_support = _metric_support(row_metrics, "operating.revenue_ttm_provider_direct")
                    ebitda_support = _metric_support(row_metrics, "operating.ebitda_ltm_provider_direct")
                    net_income_support = _metric_support(row_metrics, "earnings.net_income_ttm_provider_direct")
                    cash_sti_support = _metric_support(row_metrics, "liquidity.cash_and_short_term_investments_provider_direct")
                    total_debt_support = _metric_support(row_metrics, "capital_structure.total_debt_provider_direct")

                    net_debt = None if total_debt is None or cash_sti is None else total_debt - cash_sti
                    combo_provenance = str(companyfacts_root) if companyfacts_root is not None else str(taxonomy_reference_path)

                    row_metrics["capital_structure.net_debt_standardized"] = _build_combo_metric(
                        metric_name="capital_structure.net_debt_standardized",
                        as_of_time=as_of_time,
                        computed_at=computed_at,
                        provenance_source=combo_provenance,
                        unit="usd",
                        numerator=net_debt,
                        denominator=None,
                        extra_components={
                            "total_debt_provider_direct": total_debt,
                            "cash_and_short_term_investments_provider_direct": cash_sti,
                        },
                        component_supports={
                            "total_debt_provider_direct": total_debt_support,
                            "cash_and_short_term_investments_provider_direct": cash_sti_support,
                        },
                        formula="total_debt_provider_direct - cash_and_short_term_investments_provider_direct",
                        allow_numerator_only=True,
                    )

                    row_metrics["capital_structure.gross_leverage_standardized"] = _build_combo_metric(
                        metric_name="capital_structure.gross_leverage_standardized",
                        as_of_time=as_of_time,
                        computed_at=computed_at,
                        provenance_source=combo_provenance,
                        unit="x",
                        numerator=total_debt,
                        denominator=ebitda,
                        extra_components={
                            "total_debt_provider_direct": total_debt,
                            "ebitda_ltm_provider_direct": ebitda,
                        },
                        component_supports={
                            "total_debt_provider_direct": total_debt_support,
                            "ebitda_ltm_provider_direct": ebitda_support,
                        },
                        formula="total_debt_provider_direct / ebitda_ltm_provider_direct",
                    )

                    row_metrics["capital_structure.net_leverage_standardized"] = _build_combo_metric(
                        metric_name="capital_structure.net_leverage_standardized",
                        as_of_time=as_of_time,
                        computed_at=computed_at,
                        provenance_source=combo_provenance,
                        unit="x",
                        numerator=net_debt,
                        denominator=ebitda,
                        extra_components={
                            "net_debt_standardized": net_debt,
                            "ebitda_ltm_provider_direct": ebitda,
                        },
                        component_supports={
                            "net_debt_standardized": row_metrics["capital_structure.net_debt_standardized"]["support_mode"],
                            "ebitda_ltm_provider_direct": ebitda_support,
                        },
                        formula="net_debt_standardized / ebitda_ltm_provider_direct",
                    )

                    row_metrics["operating.ebitda_margin_standardized"] = _build_combo_metric(
                        metric_name="operating.ebitda_margin_standardized",
                        as_of_time=as_of_time,
                        computed_at=computed_at,
                        provenance_source=combo_provenance,
                        unit="ratio",
                        numerator=ebitda,
                        denominator=revenue,
                        extra_components={
                            "ebitda_ltm_provider_direct": ebitda,
                            "revenue_ttm_provider_direct": revenue,
                        },
                        component_supports={
                            "ebitda_ltm_provider_direct": ebitda_support,
                            "revenue_ttm_provider_direct": revenue_support,
                        },
                        formula="ebitda_ltm_provider_direct / revenue_ttm_provider_direct",
                    )

                    row_metrics["earnings.net_margin_standardized"] = _build_combo_metric(
                        metric_name="earnings.net_margin_standardized",
                        as_of_time=as_of_time,
                        computed_at=computed_at,
                        provenance_source=combo_provenance,
                        unit="ratio",
                        numerator=net_income,
                        denominator=revenue,
                        extra_components={
                            "net_income_ttm_provider_direct": net_income,
                            "revenue_ttm_provider_direct": revenue,
                        },
                        component_supports={
                            "net_income_ttm_provider_direct": net_income_support,
                            "revenue_ttm_provider_direct": revenue_support,
                        },
                        formula="net_income_ttm_provider_direct / revenue_ttm_provider_direct",
                    )
            except Exception as exc:  # noqa: BLE001
                error_type = "company_processing_timeout" if isinstance(exc, _CompanyProcessingTimeout) else "company_processing_failed"
                counters[f"row:{error_type}"] += 1
                row_metrics = _build_fail_open_metric_set(
                    as_of_time=as_of_time,
                    computed_at=computed_at,
                    provenance_source=str(companyfacts_path or taxonomy_reference_path),
                    error_type=error_type,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                sec_filing_repair_applied = False

            features.update(row_metrics)
            if sec_filing_repair_applied:
                counters["capital_structure.total_debt_provider_direct:sec_filing_table_repair"] += 1
            for metric_name in ALL_OUTPUT_METRIC_SPECS:
                node = row_metrics[metric_name]
                counters[f"{metric_name}:{node['support_mode']}"] += 1

            out_handle.write(json.dumps(row) + "\n")

    print(f"Wrote input-layer v1 snapshots -> {out_path}")
    print(f"provider_rows={len(provider_by_entity)}")
    for metric_name in ALL_OUTPUT_METRIC_SPECS:
        exact = counters[f"{metric_name}:exact"]
        proxy = counters[f"{metric_name}:proxy_missing_component"]
        unsupported = counters[f"{metric_name}:unsupported"]
        print(f"{metric_name}: exact={exact} proxy={proxy} unsupported={unsupported}")
    if counters["row:company_processing_failed"] or counters["row:company_processing_timeout"]:
        print(
            "row_fail_open:"
            f" failed={counters['row:company_processing_failed']}"
            f" timeout={counters['row:company_processing_timeout']}"
        )


if __name__ == "__main__":
    main()
