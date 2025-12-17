#!/usr/bin/env python3
"""Generate a repeatable validation pack for the canonical input-layer artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


APPROVED_METRICS = [
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
    "macro.sofr_or_fed_funds",
    "macro.ust_2y_yield",
    "macro.ust_10y_yield",
    "macro.curve_2s10s",
    "macro.ig_oas",
    "macro.hy_oas",
    "macro.cpi_yoy",
    "macro.unemployment_rate",
    "macro.retail_sales_yoy",
    "macro.wti_crude",
    "capital_structure.debt_like_obligations_normalized",
    "liquidity.available_liquidity_normalized",
    "operating.operating_earnings_normalized",
    "capital_structure.net_debt_normalized",
    "capital_structure.gross_leverage_normalized",
    "capital_structure.net_leverage_normalized",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def _value(node: dict[str, Any] | None) -> float | None:
    if not node:
        return None
    value = node.get("value")
    return None if value is None else float(value)


