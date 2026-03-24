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


