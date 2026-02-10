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


