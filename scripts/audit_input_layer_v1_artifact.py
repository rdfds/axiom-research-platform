#!/usr/bin/env python3
"""Audit the v1 input-layer artifact for impossible values and formula drift."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


MAX_SEC_FACT_AGE_DAYS = 550
LEASE_EXACT_MAX_AGE_DAYS = 220
LEASE_STALE_CARRY_FORWARD_MAX_AGE_DAYS = 420
DEBT_COMPONENT_ALIGNMENT_MAX_GAP_DAYS = 10
LEASE_COMPONENT_ALIGNMENT_MAX_GAP_DAYS = 10
CASH_COMPONENT_ALIGNMENT_MAX_GAP_DAYS = 10
BALANCE_SHEET_EXACT_MAX_AGE_DAYS = 130
GROUPED_CASH_STATEMENT_REPAIR_MAX_AGE_DAYS = 450
APPROVED_MACRO_METRICS = [
    "macro.fed_funds_effective",
    "macro.sofr",
    "macro.sofr_or_fed_funds",
    "macro.ust_2y_yield",
    "macro.ust_10y_yield",
    "macro.curve_2s10s",
    "macro.ig_oas",
    "macro.hy_oas",
    "macro.real_gdp_growth_yoy",
    "macro.cpi_yoy",
    "macro.unemployment_rate",
    "macro.retail_sales_yoy",
    "macro.wti_crude",
]
AUDITED_METRICS = {
    "market.market_cap_provider_direct",
    "operating.revenue_ttm_provider_direct",
    "operating.ebitda_ltm_provider_direct",
    "earnings.net_income_ttm_provider_direct",
    "liquidity.cash_and_short_term_investments_provider_direct",
    "capital_structure.total_debt_provider_direct",
    "capital_structure.net_debt_standardized",
    "capital_structure.gross_leverage_standardized",
    "capital_structure.net_leverage_standardized",
    "operating.ebitda_margin_standardized",
    "earnings.net_margin_standardized",
    "market.price_spot",
    "market.total_return_1m_standardized",
    "market.total_return_3m_standardized",
    "market.total_return_6m_standardized",
    "market.total_return_12m_standardized",
    "liquidity.cash_and_equivalents_statement_direct",
    "capital_structure.current_debt_statement_direct",
    "capital_structure.long_term_debt_statement_direct",
    "operating.ebit_statement_direct",
    "capital_structure.interest_expense_statement_direct",
    "liquidity.restricted_cash_sec_exact",
    "liquidity.marketable_securities_sec_exact",
    "liquidity.revolver_undrawn_sec_exact",
    "capital_structure.lease_liabilities_sec_exact",
    "capital_structure.debt_like_obligations_normalized",
    "capital_structure.net_pension_liability",
    "capital_structure.other_postretirement_benefit_liability",
    "capital_structure.combined_retirement_liability",
    "capital_structure.debt_like_obligations_including_pension",
    "capital_structure.debt_like_obligations_including_retirement",
    "liquidity.available_liquidity_normalized",
    "operating.operating_earnings_normalized",
    "capital_structure.net_debt_normalized",
    "capital_structure.net_debt_including_pension",
    "capital_structure.net_debt_including_retirement",
    "capital_structure.gross_leverage_normalized",
    "capital_structure.gross_leverage_including_pension",
    "capital_structure.gross_leverage_including_retirement",
    "capital_structure.net_leverage_normalized",
    "capital_structure.net_leverage_including_pension",
    "capital_structure.net_leverage_including_retirement",
    *APPROVED_MACRO_METRICS,
}
RESTRICTED_CASH_EXACT_CONCEPTS = {
    "RestrictedCash",
    "RestrictedCashCurrent",
    "RestrictedCashNoncurrent",
    "RestrictedCashAndCashEquivalentsNoncurrent",
}
RESTRICTED_CASH_MIXED_FALLBACK_CONCEPTS = {
    "RestrictedCashAndCashEquivalents",
    "RestrictedCashAndCashEquivalentsAtCarryingValue",
    "RestrictedCashAndInvestmentsCurrent",
}
RESTRICTED_CASH_TOTAL_RECONCILIATION_CONCEPT = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
MARKETABLE_SECURITY_EXACT_CONCEPTS = {
    "ShortTermInvestments",
    "MarketableSecurities",
    "AvailableForSaleSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    "AvailableForSaleDebtSecuritiesCurrent",
    "MarketableSecuritiesCurrent",
}
LEASE_APPROVED_STALE_SUPPORT_OVERRIDES = {
    "stale_internally_consistent_lease_carry_forward",
    "stale_liability_total_corroborated_by_fresh_rou_asset",
}
LEASE_APPROVED_MIXED_SUPPORT_OVERRIDES = {
    "mixed_fresh_and_stale_components_corroborated_by_fresh_rou_asset",
    "mixed_fresh_and_stale_components_rebased_by_rou_basis_delta",
}
GROUPED_CASH_APPROVED_REPAIR_MODES = {
    "cash_and_equivalents_plus_marketable_securities",
    "cash_and_equivalents_plus_inferred_zero_short_term_investments",
}


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--entity-identifier-path")
    parser.add_argument("--raw-timeseries-path")
    parser.add_argument("--crsp-market-cache-path")
    parser.add_argument("--crsp-daily-root")
    return parser.parse_args()


def _value(node: dict[str, Any] | None) -> float | None:
    if not node:
        return None
    value = node.get("value")
    return None if value is None else float(value)


def _support(node: dict[str, Any] | None) -> str:
    if not node:
        return "missing_metric"
    return node.get("support_mode") or "missing_metric"


def _approx_equal(a: float | None, b: float | None, tolerance: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is b
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= tolerance * scale


def _company_label(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("features") or {}
    provider_fields = [
        "market.market_cap_provider_direct",
        "liquidity.cash_and_short_term_investments_provider_direct",
        "capital_structure.total_debt_provider_direct",
    ]
    company_name = None
    reference_instrument = None
    for metric_name in provider_fields:
        breakdown = (features.get(metric_name) or {}).get("component_breakdown") or {}
        company_name = company_name or breakdown.get("provider_company_name")
        reference_instrument = reference_instrument or breakdown.get("reference_instrument")
    return {
        "company_id": row.get("company_id"),
        "company_name": company_name,
        "reference_instrument": reference_instrument,
    }


def _iter_component_ends(component_breakdown: Any) -> list[str]:
    ends: list[str] = []
    if isinstance(component_breakdown, dict):
        if "end" in component_breakdown and component_breakdown["end"]:
            ends.append(component_breakdown["end"])
        for value in component_breakdown.values():
            ends.extend(_iter_component_ends(value))
    elif isinstance(component_breakdown, list):
        for item in component_breakdown:
            ends.extend(_iter_component_ends(item))
    return ends


def _latest_end_age_days(as_of_date: date, component_breakdown: Any) -> int | None:
    ends = []
    for end_text in _iter_component_ends(component_breakdown):
        try:
            ends.append(date.fromisoformat(end_text))
        except ValueError:
            continue
    if not ends:
        return None
    latest_end = max(ends)
    return (as_of_date - latest_end).days


def _iter_component_concepts(component_breakdown: Any) -> list[str]:
    concepts: list[str] = []
    if isinstance(component_breakdown, dict):
        concept = component_breakdown.get("concept")
        if concept:
            concepts.append(str(concept))
        for value in component_breakdown.values():
            concepts.extend(_iter_component_concepts(value))
    elif isinstance(component_breakdown, list):
        for item in component_breakdown:
            concepts.extend(_iter_component_concepts(item))
    return concepts


def _restricted_cash_breakdown_is_semantically_valid(component_breakdown: Any) -> bool:
    if not isinstance(component_breakdown, dict):
        return False
    mode = component_breakdown.get("mode")
    concepts = _iter_component_concepts(component_breakdown)
    if not concepts:
        concept = component_breakdown.get("concept")
        return concept in RESTRICTED_CASH_EXACT_CONCEPTS

    if mode in {
        "cash_plus_restricted_total_minus_cash_equivalents",
        "cash_plus_restricted_total_minus_grouped_cash_cash_only_proxy",
    }:
        return (
            RESTRICTED_CASH_TOTAL_RECONCILIATION_CONCEPT in concepts
            and (
                "cash_and_equivalents_statement_direct" in component_breakdown
                or "cash_and_short_term_investments_provider_direct_cash_only_proxy" in component_breakdown
            )
        )

    if mode == "mixed_total_restricted_cash_fallback":
        return all(
            found in RESTRICTED_CASH_MIXED_FALLBACK_CONCEPTS
            for found in concepts
        )

    return all(found in RESTRICTED_CASH_EXACT_CONCEPTS for found in concepts)


