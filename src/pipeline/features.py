from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..asof_store import AsOfWarehouse
from ..warehouse_pivots import fetch_financials_asof
from .types import CompanyStateSnapshot


DATA_DIR = Path(__file__).parent.parent.parent / "data"


class FeatureBuilder:
    def __init__(self, warehouse: Optional[AsOfWarehouse] = None):
        self.warehouse = warehouse or AsOfWarehouse()
        self.link_table = None
        link_path = DATA_DIR / "wrds" / "crsp" / "ccmxpf_lnkhist.parquet"
        if link_path.exists():
            self.link_table = pd.read_parquet(link_path)
            self.link_table["linkdt"] = pd.to_datetime(self.link_table["linkdt"], errors="coerce")
            self.link_table["linkenddt"] = pd.to_datetime(self.link_table["linkenddt"], errors="coerce")
            self.link_table["linkenddt"] = self.link_table["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
            self.link_table["gvkey"] = self.link_table["gvkey"].astype(str)

    def _map_gvkey_to_permno(self, gvkey: str, as_of_date: datetime) -> Optional[int]:
        if self.link_table is None:
            return None
        as_of = pd.to_datetime(as_of_date)
        gvkey_str = str(gvkey)
        links = self.link_table[self.link_table["gvkey"] == gvkey_str].copy()
        if links.empty:
            return None
        links = links[(links["linkdt"] <= as_of) & (links["linkenddt"] >= as_of)]
        if links.empty:
            return None
        links["rank"] = 0
        links.loc[links["linkprim"] != "P", "rank"] += 1
        links.loc[~links["linktype"].isin(["LC", "LU", "LD", "LN"]), "rank"] += 1
        links = links.sort_values(["rank", "linkdt"], ascending=[True, False])
        permno = links.iloc[0].get("lpermno")
        if pd.isna(permno):
            return None
        try:
            return int(permno)
        except Exception:
            return None

    def _latest_price(self, permno: int, as_of_date: datetime) -> Optional[float]:
        if permno is None:
            return None
        as_of = pd.to_datetime(as_of_date)
        df = self.warehouse.query(
            "warehouse_prices_daily",
            as_of=as_of,
            where=f"entity_id = '{permno}'",
        )
        if df.empty:
            df = self.warehouse.query(
                "warehouse_prices_daily_rdp",
                as_of=as_of,
                where=f"entity_id = '{permno}'",
            )
        if df.empty:
            return None
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
        df = df.sort_values("event_time")
        last = df.iloc[-1]
        price = last.get("adjusted_close")
        if price is None or pd.isna(price):
            price = last.get("close")
        return float(price) if price is not None and not pd.isna(price) else None

    @staticmethod
    def _ttm_sum(df: pd.DataFrame, col: str, n: int = 4) -> Optional[float]:
        if df.empty or col not in df.columns:
            return None
        series = df[col].dropna().head(n)
        if series.empty:
            return None
        return float(series.sum())

    @staticmethod
    def _latest(df: pd.DataFrame, col: str) -> Optional[float]:
        if df.empty or col not in df.columns:
            return None
        val = df[col].dropna().head(1)
        if val.empty:
            return None
        return float(val.iloc[0])

    def _macro_series_history(self, series_id: str, as_of: datetime) -> pd.DataFrame:
        if not series_id:
            return pd.DataFrame()
        df = self.warehouse.query(
            "warehouse_macro",
            as_of=as_of,
            where=f"entity_id = '{series_id}'",
        )
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["event_time"] = pd.to_datetime(df.get("event_time"), errors="coerce")
        df["value"] = pd.to_numeric(df.get("value"), errors="coerce")
        df = df[df["event_time"].notna() & df["value"].notna()].sort_values("event_time")
        return df

    @staticmethod
    def _latest_macro_value(history: pd.DataFrame) -> Optional[float]:
        if history.empty:
            return None
        try:
            return float(history.iloc[-1]["value"])
        except Exception:
            return None

    @staticmethod
    def _real_gdp_growth_yoy(history: pd.DataFrame) -> Optional[float]:
        if history.empty:
            return None
        ordered = history.sort_values("event_time")
        latest = ordered.iloc[-1]
        latest_value = pd.to_numeric(pd.Series([latest.get("value")]), errors="coerce").iloc[0]
        latest_time = pd.to_datetime(latest.get("event_time"), errors="coerce")
        if pd.isna(latest_value) or pd.isna(latest_time):
            return None
        prior_cutoff = latest_time - pd.DateOffset(years=1)
        prior = ordered[ordered["event_time"] <= prior_cutoff]
        if prior.empty:
            return None
        prior_value = pd.to_numeric(pd.Series([prior.iloc[-1].get("value")]), errors="coerce").iloc[0]
        if pd.isna(prior_value) or prior_value == 0:
            return None
        return float((latest_value / prior_value) - 1.0)

    def _compute_financial_metrics(self, company_id: str, as_of: datetime) -> Dict[str, Any]:
        fin = fetch_financials_asof(as_of, warehouse=self.warehouse, company_id=company_id)
        if fin.empty:
            return {}
        fin = fin.sort_values("event_time", ascending=False)

        revenue_ttm = self._ttm_sum(fin, "revtq")
        ebitda_ttm = self._ttm_sum(fin, "oibdpq")
        net_income_ttm = self._ttm_sum(fin, "niq")
        eps_ttm = self._ttm_sum(fin, "epspxq")

        shares_out = self._latest(fin, "cshoq")
        cash = self._latest(fin, "cheq")
        debt_long = self._latest(fin, "dlttq")
        debt_short = self._latest(fin, "dlcq")
        total_assets = self._latest(fin, "atq")

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
        if "oancfy" in fin.columns and "capxy" in fin.columns:
            oancfy = self._latest(fin, "oancfy")
            capxy = self._latest(fin, "capxy")
            if oancfy is not None and capxy is not None:
                fcf = oancfy - capxy

        fcf_margin = None
        if fcf is not None and revenue_ttm:
            fcf_margin = fcf / revenue_ttm if revenue_ttm != 0 else None

        return {
            "revenue_ttm": revenue_ttm,
            "ebitda_ttm": ebitda_ttm,
            "net_income_ttm": net_income_ttm,
            "eps_ttm": eps_ttm,
            "shares_out": shares_out,
            "cash": cash,
            "debt": debt,
            "net_debt": net_debt,
            "total_assets": total_assets,
            "ebitda_margin": ebitda_margin,
            "leverage_net_debt_ebitda": leverage,
            "roic_proxy": roic,
            "fcf_margin": fcf_margin,
        }

    def _compute_valuation_metrics(self, company_id: str, as_of: datetime) -> Dict[str, Any]:
        features = self._compute_financial_metrics(company_id, as_of)
        gvkey = company_id
        permno = self._map_gvkey_to_permno(gvkey, as_of) if gvkey else None
        price = self._latest_price(permno, as_of) if permno else None

        market_cap = None
        if price is not None and features.get("shares_out") is not None:
            market_cap = price * features["shares_out"]

        pe = None
        if price is not None and features.get("eps_ttm"):
            eps = features["eps_ttm"]
            if eps and eps != 0:
                pe = price / eps

        ev_ebitda = None
        if market_cap is not None and features.get("net_debt") is not None and features.get("ebitda_ttm"):
            ev = market_cap + features["net_debt"]
            ebitda = features["ebitda_ttm"]
            if ebitda and ebitda != 0:
                ev_ebitda = ev / ebitda

        return {
            "price": price,
            "market_cap": market_cap,
            "pe": pe,
            "ev_ebitda": ev_ebitda,
        }

    def compute_macro_features(self, as_of: datetime, series_map: Dict[str, str]) -> Dict[str, Any]:
        macro: Dict[str, Any] = {}
        history_cache: Dict[str, pd.DataFrame] = {}
        for name, series_id in series_map.items():
            if series_id not in history_cache:
                history_cache[series_id] = self._macro_series_history(series_id, as_of)
            macro[f"macro_{name}"] = self._latest_macro_value(history_cache[series_id])

        real_gdp_series_id = (
            series_map.get("real_gdp")
            or series_map.get("real_gdp_level")
            or series_map.get("real_gdp_growth_yoy")
        )
        if real_gdp_series_id:
            if real_gdp_series_id not in history_cache:
                history_cache[real_gdp_series_id] = self._macro_series_history(real_gdp_series_id, as_of)
            macro["macro_real_gdp_growth_yoy"] = self._real_gdp_growth_yoy(history_cache[real_gdp_series_id])
        return macro

    def build_company_state(
        self,
        company_id: str,
        as_of: datetime,
        macro_series: Optional[Dict[str, str]] = None,
    ) -> CompanyStateSnapshot:
        fin = self._compute_financial_metrics(company_id, as_of)
        val = self._compute_valuation_metrics(company_id, as_of)
        features = {**fin, **val}
        macro = self.compute_macro_features(as_of, macro_series or {})
        features.update(macro)
        return CompanyStateSnapshot(
            company_id=company_id,
            as_of_time=as_of,
            features=features,
            regime={},
            constraint_set=[],
            provenance={},
        )
