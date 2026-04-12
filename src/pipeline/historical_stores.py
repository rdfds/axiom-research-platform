from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd


_EVENT_COLS = [
    "source_event_id",
    "company_id",
    "action_date",
    "action_id",
    "action_type",
    "action_subtype",
    "action_size",
]

_STATE_COLS = [
    "source_event_id",
    "base_leverage",
    "base_margin",
    "base_market_cap",
    "base_revenue_ttm",
    "base_revenue_ttm_lag_1y",
    "base_revenue_growth_yoy",
    "base_ebitda_ttm",
    "base_cash",
    "base_total_debt",
    "base_current_debt",
    "base_available_liquidity",
    "base_interest_expense",
    "base_fcf_yield",
    "base_volatility_30d",
    "base_volatility_90d",
    "base_drawdown_90d",
    "base_momentum_60d",
    "base_credit_spread_level",
    "base_equity_window_proxy",
    "base_credit_window_proxy",
    "base_roic",
    "base_fcf_margin",
    "base_sector",
    "sector",
    "gics_sector",
    "sector_name",
    "sic",
    "base_sic",
    "taxonomy.sector",
    "taxonomy.subsector",
    "retirement_regime",
    "state_vector_v1.size_log_revenue",
    "state_vector_v1.profitability",
    "state_vector_v1.growth",
    "state_vector_v1.gross_obligation_burden",
    "state_vector_v1.net_obligation_burden",
    "state_vector_v1.liquidity_flexibility",
    "state_vector_v1.interest_coverage",
    "state_vector_v1.valuation_multiple",
    "state_vector_v1.cash_generation",
    "state_vector_v1.market_stress",
    "state_vector_v1.market_access",
    "state_vector_v1.rates_level",
    "state_vector_v1.credit_spread",
    "macro_fed_funds_effective",
    "macro_real_gdp_growth_yoy",
]

_OUTCOME_COLS = [
    "source_event_id",
    "outcome_pe_6m",
    "outcome_pe_12m",
    "outcome_ev_ebitda_6m",
    "outcome_ev_ebitda_12m",
    "credit_spread_change_1m",
    "credit_spread_change_6m",
    "credit_spread_change_12m",
    "credit_spread_change_24m",
    "rating_migration_1m",
    "rating_migration_6m",
    "rating_migration_12m",
    "rating_migration_24m",
    "leverage_delta",
    "fcf_margin_delta",
]

_REGIME_COLS = [
    "source_event_id",
    "macro_hy_oas",
    "macro_vix",
]


def _ensure_source_event_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "source_event_id" in out.columns:
        out["source_event_id"] = out["source_event_id"].astype(str)
        return out
    company = out["company_id"].astype(str) if "company_id" in out.columns else pd.Series("", index=out.index, dtype=str)
    action_date = out["action_date"].astype(str) if "action_date" in out.columns else pd.Series("", index=out.index, dtype=str)
    out["source_event_id"] = (
        company
        + "::"
        + action_date
        + "::"
        + out.index.astype(str)
    )
    return out


def _select_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return pd.DataFrame()
    return df[keep].copy()


@dataclass(frozen=True)
class HistoricalEventStore:
    events: pd.DataFrame
    dataset_version: str

    def filter_by_action_subtype(self, action_keys: List[str]) -> pd.DataFrame:
        if self.events.empty:
            return self.events.copy()
        out = self.events.copy()
        mask = pd.Series(False, index=out.index)
        for col in ("action_subtype", "action_type", "action_id"):
            if col in out.columns:
                mask = mask | out[col].astype(str).isin(action_keys)
        return out.loc[mask].copy()


@dataclass(frozen=True)
class HistoricalCompanyStateSnapshotStore:
    snapshots: pd.DataFrame
    dataset_version: str


@dataclass(frozen=True)
class HistoricalOutcomeStore:
    outcomes: pd.DataFrame
    dataset_version: str


@dataclass(frozen=True)
class RegimeHistory:
    regimes: pd.DataFrame
    dataset_version: str


