#!/usr/bin/env python
"""
Build action outcome dataset for precedent matching.

Example:
  python -u scripts/51_build_action_outcomes.py \
    --action-types buyback,acquisition \
    --start-date 2000-01-01 \
    --out data/curated/action_outcomes.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.action_normalization import augment_action_outcomes_df
from src.pipeline.config import load_config
from src.pipeline.features import FeatureBuilder


DATA_DIR = Path(__file__).parent.parent / "data"

FMP_INCOME_ITEMS = [
    "Revenue",
    "EBITDA",
    "NetIncome",
    "EPS",
    "EPSDiluted",
    "SharesOut",
    "SharesOutDiluted",
]
FMP_BALANCE_ITEMS = [
    "Cash",
    "ShortTermInvestments",
    "DebtCurrent",
    "DebtLongTerm",
    "TotalAssets",
]
FMP_CASH_ITEMS = [
    "OperatingCashFlow",
    "Capex",
    "FreeCashFlow",
]


def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    try:
        old_val = float(old)
        new_val = float(new)
    except Exception:
        return None
    if old_val == 0 or pd.isna(old_val) or pd.isna(new_val):
        return None
    return (new_val - old_val) / abs(old_val)


def _pp_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    try:
        old_val = float(old)
        new_val = float(new)
    except Exception:
        return None
    if pd.isna(old_val) or pd.isna(new_val):
        return None
    return new_val - old_val


def _pick_date(row: pd.Series, date_field: str) -> Optional[pd.Timestamp]:
    if date_field != "auto":
        val = row.get(date_field)
        return pd.to_datetime(val, errors="coerce") if val is not None else None
    for field in ("announcement_date", "event_time", "effective_date", "action_date"):
        val = row.get(field)
        if val is not None and not pd.isna(val):
            return pd.to_datetime(val, errors="coerce")
    return None


def _months_from_quarters(quarters: int) -> int:
    return int(quarters) * 3


def _date_key(value: pd.Timestamp) -> str:
    return value.normalize().strftime("%Y-%m-%d")


def first_non_null(*values: Any) -> Optional[float]:
    for value in values:
        if value is None or pd.isna(value):
            continue
        return value
    return None


def first_positive_non_null(*values: Any) -> Optional[float]:
    for value in values:
        if value is None or pd.isna(value):
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if numeric > 0:
            return numeric
    return None


def _resolved_action_size(row: pd.Series) -> Optional[float]:
    action_type = str(row.get("action_type") or "").strip().lower()
    if action_type in {"split", "stock_split", "reverse_split"}:
        return first_positive_non_null(
            row.get("split_factor"),
            row.get("facpr"),
            row.get("ratio"),
            row.get("size"),
            row.get("amount"),
            row.get("divamt"),
            row.get("deal_value"),
            row.get("offering_amt_k"),
            row.get("principal_amt"),
            row.get("dealamount"),
        )
    return first_non_null(
        row.get("size"),
        row.get("amount"),
        row.get("ratio"),
        row.get("split_factor"),
        row.get("facpr"),
        row.get("divamt"),
        row.get("deal_value"),
        row.get("offering_amt_k"),
        row.get("principal_amt"),
        row.get("dealamount"),
    )


def _base_available_liquidity(metrics: Dict[str, Any]) -> Optional[float]:
    value = metrics.get("available_liquidity")
    if value is not None:
        return value
    return metrics.get("cash")


def _apply_richer_base_fields(record: Dict[str, Any], base: Dict[str, Any]) -> None:
    record["base_cash"] = base.get("cash")
    record["base_total_debt"] = base.get("total_debt", base.get("debt"))
    record["base_available_liquidity"] = _base_available_liquidity(base)


def _enrich_macro_columns_from_helper(
    df: pd.DataFrame,
    *,
    macro_series: Dict[str, str],
) -> pd.DataFrame:
    if df.empty or "action_date" not in df.columns or not macro_series:
        return df

    helper = FeatureBuilder()
    dates = pd.to_datetime(df["action_date"], errors="coerce")
    unique_dates = sorted({pd.Timestamp(d).normalize() for d in dates.dropna().tolist()})
    if not unique_dates:
        return df

    macro_cache: Dict[str, Dict[str, Any]] = {}
    for as_of in unique_dates:
        macro_cache[_date_key(as_of)] = helper.compute_macro_features(as_of, macro_series)

    enriched = df.copy()
    date_keys = dates.dt.normalize().dt.strftime("%Y-%m-%d")
    candidate_columns = sorted(
        {
            col
            for payload in macro_cache.values()
            for col in payload.keys()
            if str(col).startswith("macro_")
        }
    )
    for column in candidate_columns:
        mapped = date_keys.map(lambda key: (macro_cache.get(key) or {}).get(column))
        if column not in enriched.columns:
            enriched[column] = mapped
        else:
            enriched[column] = enriched[column].where(enriched[column].notna(), mapped)
    return enriched


def _load_actions(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing corporate actions dataset: {path}")
    return pd.read_parquet(path, columns=list(columns))


def _normalize_gvkey(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.extract(r"(\d+)")[0]
    return cleaned.str.zfill(6)


class FundamentalsProvider:
    def __init__(self, path: Path, preload: bool = False):
        self.path = path
        self._cache: Dict[str, pd.DataFrame] = {}
        self._full: Optional[pd.DataFrame] = None
        if preload and self.path.exists():
            df = pd.read_parquet(
                self.path,
                columns=[
                    "gvkey",
                    "datadate",
                    "revtq",
                    "oibdpq",
                    "niq",
                    "epspxq",
                    "cshoq",
                    "cheq",
                    "dlttq",
                    "dlcq",
                    "atq",
                    "oancfy",
                    "capxy",
                    "prccq",
                    "mkvaltq",
                    "sic",
                ],
            )
            if not df.empty:
                df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
                df = df.sort_values(["gvkey", "datadate"], ascending=[True, False])
            self._full = df

    def _load_company(self, gvkey: str) -> pd.DataFrame:
        if gvkey in self._cache:
            return self._cache[gvkey]
        if self._full is not None:
            df = self._full[self._full["gvkey"] == gvkey].copy()
            self._cache[gvkey] = df
            return df
        if not self.path.exists():
            self._cache[gvkey] = pd.DataFrame()
            return self._cache[gvkey]
        df = pd.read_parquet(
            self.path,
            columns=[
                "gvkey",
                "datadate",
                "revtq",
                "oibdpq",
                "niq",
                "epspxq",
                "cshoq",
                "cheq",
                "dlttq",
                "dlcq",
                "atq",
                "oancfy",
                "capxy",
                "prccq",
                "mkvaltq",
                "sic",
            ],
            filters=[("gvkey", "=", gvkey)],
        )
        if not df.empty:
            df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
            df = df.sort_values("datadate", ascending=False)
        self._cache[gvkey] = df
        return df

    def get_metrics(self, gvkey: str, as_of: pd.Timestamp) -> Dict[str, Any]:
        df = self._load_company(gvkey)
        if df.empty:
            return {}
        df = df[df["datadate"] <= as_of]
        if df.empty:
            return {}
        df = df.sort_values("datadate", ascending=False)

        def ttm_sum(col: str) -> Optional[float]:
            series = df[col].dropna().head(4)
            return float(series.sum()) if not series.empty else None

        def latest(col: str) -> Optional[float]:
            series = df[col].dropna().head(1)
            return float(series.iloc[0]) if not series.empty else None

        revenue_ttm = ttm_sum("revtq")
        ebitda_ttm = ttm_sum("oibdpq")
        net_income_ttm = ttm_sum("niq")
        eps_ttm = ttm_sum("epspxq")

        shares_out = latest("cshoq")
        cash = latest("cheq")
        debt_long = latest("dlttq")
        debt_short = latest("dlcq")
        total_assets = latest("atq")
        price = latest("prccq")
        market_cap = latest("mkvaltq")

        if market_cap is None and price is not None and shares_out is not None:
            market_cap = price * shares_out

        debt = None
        if debt_long is not None or debt_short is not None:
            debt = (debt_long or 0.0) + (debt_short or 0.0)

        net_debt = None
        if debt is not None:
            net_debt = debt - (cash or 0.0)

        ebitda_margin = None
        if revenue_ttm and ebitda_ttm is not None:
            ebitda_margin = ebitda_ttm / revenue_ttm if revenue_ttm != 0 else None

        leverage = None
        if net_debt is not None and ebitda_ttm:
            leverage = net_debt / ebitda_ttm if ebitda_ttm != 0 else None

        roic = None
        if net_income_ttm is not None and total_assets:
            roic = net_income_ttm / total_assets if total_assets != 0 else None

        fcf = None
        oancfy = latest("oancfy")
        capxy = latest("capxy")
        if oancfy is not None and capxy is not None:
            fcf = oancfy - capxy

        fcf_margin = None
        if fcf is not None and revenue_ttm:
            fcf_margin = fcf / revenue_ttm if revenue_ttm != 0 else None

        pe = None
        if price is not None and eps_ttm:
            if eps_ttm != 0:
                pe = price / eps_ttm

        ev_ebitda = None
        if market_cap is not None and net_debt is not None and ebitda_ttm:
            if ebitda_ttm != 0:
                ev_ebitda = (market_cap + net_debt) / ebitda_ttm

        sic = latest("sic")

        return {
            "revenue_ttm": revenue_ttm,
            "ebitda_ttm": ebitda_ttm,
            "net_income_ttm": net_income_ttm,
            "eps_ttm": eps_ttm,
            "shares_out": shares_out,
            "cash": cash,
            "debt": debt,
            "total_debt": debt,
            "net_debt": net_debt,
            "available_liquidity": cash,
            "total_assets": total_assets,
            "ebitda_margin": ebitda_margin,
            "leverage_net_debt_ebitda": leverage,
            "roic_proxy": roic,
            "fcf_margin": fcf_margin,
            "price": price,
            "market_cap": market_cap,
            "pe": pe,
            "ev_ebitda": ev_ebitda,
            "sic": sic,
            "fundamentals_source": "compustat",
        }


