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


class FmpFundamentalsProvider:
    def __init__(self, path: str, price_path: Path, ciq_path: Path):
        self.path = path
        self.price_path = price_path
        self.ciq_path = ciq_path
        self._cache: Dict[str, pd.DataFrame] = {}
        self._con = None
        self._price_con = None
        self._ticker_to_cusip: Dict[str, str] = {}
        self._logged_first_query = False
        self._use_cache = "fmp_fundamentals_cache" in path or "*" not in path

    def _connect(self):
        if self._con is None:
            import duckdb

            self._con = duckdb.connect()

    def _connect_price(self):
        if self._price_con is None:
            import duckdb

            self._price_con = duckdb.connect()

    def _load_ciq_map(self):
        if self._ticker_to_cusip or not self.ciq_path.exists():
            return
        print("[build_action_outcomes] Loading CIQ ticker->CUSIP map for FMP price lookup...")
        df = pd.read_parquet(self.ciq_path, columns=["ticker", "cusip8"])
        df = df.dropna(subset=["ticker", "cusip8"]).copy()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["cusip8"] = df["cusip8"].astype(str).str.upper().str.strip()
        self._ticker_to_cusip = dict(zip(df["ticker"], df["cusip8"]))
        print(f"[build_action_outcomes] Loaded CIQ map: {len(self._ticker_to_cusip):,} tickers")

    @staticmethod
    def _ticker_variants(ticker: str) -> Iterable[str]:
        t = str(ticker).strip().upper()
        variants = [t]
        if "." in t:
            variants.append(t.replace(".", "-"))
        if "-" in t:
            variants.append(t.replace("-", "."))
        return list(dict.fromkeys([v for v in variants if v]))

    def _query(self, ticker: str, statement_type: str, items: Iterable[str]) -> pd.DataFrame:
        self._connect()
        if not self._logged_first_query:
            print("[build_action_outcomes] FMP fallback active; querying FMP financials...")
            self._logged_first_query = True
        if self._use_cache:
            query = f"""
                select
                    datadate,
                    fiscal_quarter,
                    revenue,
                    ebitda,
                    net_income,
                    eps,
                    eps_diluted,
                    shares_out,
                    shares_out_diluted,
                    cash,
                    short_term_investments,
                    debt_current,
                    debt_long_term,
                    total_assets,
                    operating_cash_flow,
                    capex,
                    free_cash_flow
                from read_parquet('{self.path}')
                where ticker = ?
            """
            return self._con.execute(query, [ticker]).df()
        items_sql = ",".join([f"'{item}'" for item in items])
        query = f"""
            select
                fiscal_period_end as datadate,
                fiscal_quarter,
                line_item,
                value
            from read_parquet('{self.path}', hive_partitioning=1)
            where source_system='fmp_financials'
              and statement_type='{statement_type}'
              and company_id = ?
              and line_item in ({items_sql})
        """
        return self._con.execute(query, [ticker]).df()

    def _load_company(self, ticker: str) -> pd.DataFrame:
        if ticker in self._cache:
            return self._cache[ticker]
        df = pd.DataFrame()
        for variant in self._ticker_variants(ticker):
            if self._use_cache:
                df = self._query(variant, "income", FMP_INCOME_ITEMS)
            else:
                income = self._query(variant, "income", FMP_INCOME_ITEMS)
                balance = self._query(variant, "balance_sheet", FMP_BALANCE_ITEMS)
                cash = self._query(variant, "cash_flow", FMP_CASH_ITEMS)
                df = pd.concat([income, balance, cash], ignore_index=True)
            if not df.empty:
                self._cache[ticker] = df
                self._cache[variant] = df
                return df
        self._cache[ticker] = df
        return df

    def _price_for(self, ticker: str, as_of: pd.Timestamp) -> Optional[float]:
        self._load_ciq_map()
        cusip = self._ticker_to_cusip.get(str(ticker).upper())
        if not cusip:
            return None
        self._connect_price()
        query = f"""
            select close
            from read_parquet('{self.price_path}')
            where cusip = ?
              and event_time <= ?
            order by event_time desc
            limit 1
        """
        rows = self._price_con.execute(query, [cusip, as_of]).fetchall()
        if not rows:
            return None
        return float(rows[0][0]) if rows[0][0] is not None else None

    def get_metrics(self, ticker: str, as_of: pd.Timestamp) -> Dict[str, Any]:
        df = self._load_company(ticker)
        if df.empty:
            return {}
        df = df.copy()
        df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
        df = df[df["datadate"].notna()]
        df = df[df["datadate"] <= as_of]
        if df.empty:
            return {}
        if "fiscal_quarter" in df.columns:
            df = df[df["fiscal_quarter"].isin([1, 2, 3, 4])]
        if df.empty:
            return {}
        df = df.sort_values("datadate", ascending=False)

        col_map = {
            "Revenue": "revenue",
            "EBITDA": "ebitda",
            "NetIncome": "net_income",
            "EPS": "eps",
            "EPSDiluted": "eps_diluted",
            "SharesOut": "shares_out",
            "SharesOutDiluted": "shares_out_diluted",
            "Cash": "cash",
            "ShortTermInvestments": "short_term_investments",
            "DebtCurrent": "debt_current",
            "DebtLongTerm": "debt_long_term",
            "TotalAssets": "total_assets",
            "OperatingCashFlow": "operating_cash_flow",
            "Capex": "capex",
            "FreeCashFlow": "free_cash_flow",
        }

        def ttm_sum(item: str) :
            if self._use_cache:
                col = col_map.get(item)
                series = df[col].dropna().head(4) if col and col in df.columns else pd.Series([], dtype=float)
            else:
                series = df[df["line_item"] == item]["value"].dropna().head(4)
            return float(series.sum()) if not series.empty else None

        def latest(item: str) -> Optional[float]:
            if self._use_cache:
                col = col_map.get(item)
                series = df[col].dropna().head(1) if col and col in df.columns else pd.Series([], dtype=float)
            else:
                series = df[df["line_item"] == item]["value"].dropna().head(1)
            return float(series.iloc[0]) if not series.empty else None

        revenue_ttm = ttm_sum("Revenue")
        ebitda_ttm = ttm_sum("EBITDA")
        net_income_ttm = ttm_sum("NetIncome")
        eps_ttm = ttm_sum("EPS") or ttm_sum("EPSDiluted")
        shares_out = latest("SharesOutDiluted") or latest("SharesOut")

        cash = latest("Cash")
        short_invest = latest("ShortTermInvestments")
        debt_short = latest("DebtCurrent")
        debt_long = latest("DebtLongTerm")
        total_assets = latest("TotalAssets")

        cash_total = None
        if cash is not None or short_invest is not None:
            cash_total = (cash or 0.0) + (short_invest or 0.0)

        debt = None
        if debt_short is not None or debt_long is not None:
            debt = (debt_short or 0.0) + (debt_long or 0.0)

        net_debt = None
        if debt is not None:
            net_debt = debt - (cash_total or 0.0)

        leverage = None
        if net_debt is not None and ebitda_ttm:
            leverage = net_debt / ebitda_ttm if ebitda_ttm != 0 else None

        roic = None
        if net_income_ttm is not None and total_assets:
            roic = net_income_ttm / total_assets if total_assets != 0 else None

        ebitda_margin = None
        if revenue_ttm and ebitda_ttm is not None:
            ebitda_margin = ebitda_ttm / revenue_ttm if revenue_ttm != 0 else None

        fcf = latest("FreeCashFlow")
        if fcf is None:
            ocf = latest("OperatingCashFlow")
            capex = latest("Capex")
            if ocf is not None and capex is not None:
                fcf = ocf - capex

        fcf_margin = None
        if fcf is not None and revenue_ttm:
            fcf_margin = fcf / revenue_ttm if revenue_ttm != 0 else None

        price = self._price_for(ticker, as_of)
        market_cap = None
        if price is not None and shares_out is not None:
            market_cap = price * shares_out

        pe = None
        if price is not None and eps_ttm:
            pe = price / eps_ttm if eps_ttm != 0 else None

        ev_ebitda = None
        if market_cap is not None and net_debt is not None and ebitda_ttm:
            ev_ebitda = (market_cap + net_debt) / ebitda_ttm if ebitda_ttm != 0 else None

        return {
            "revenue_ttm": revenue_ttm,
            "ebitda_ttm": ebitda_ttm,
            "net_income_ttm": net_income_ttm,
            "eps_ttm": eps_ttm,
            "shares_out": shares_out,
            "cash": cash_total,
            "debt": debt,
            "total_debt": debt,
            "net_debt": net_debt,
            "available_liquidity": cash_total,
            "total_assets": total_assets,
            "ebitda_margin": ebitda_margin,
            "leverage_net_debt_ebitda": leverage,
            "roic_proxy": roic,
            "fcf_margin": fcf_margin,
            "price": price,
            "market_cap": market_cap,
            "pe": pe,
            "ev_ebitda": ev_ebitda,
            "fundamentals_source": "fmp",
        }


def _load_link_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    link = pd.read_parquet(
        path,
        columns=["gvkey", "lpermno", "linkdt", "linkenddt", "linkprim", "linktype"],
    )
    link["linkdt"] = pd.to_datetime(link["linkdt"], errors="coerce")
    link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce")
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    link["lpermno"] = pd.to_numeric(link["lpermno"], errors="coerce").astype("Int64")
    link["gvkey"] = link["gvkey"].astype(str).str.zfill(6)
    return link


def _map_permno_to_gvkey(
    mna: pd.DataFrame,
    link: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    if mna.empty or link.empty:
        return pd.DataFrame()
    mna = mna.copy()
    mna["permno"] = pd.to_numeric(mna["acquiror_permno"], errors="coerce").astype("Int64")
    mna = mna[mna["permno"].notna()].copy()
    if mna.empty:
        return pd.DataFrame()

    permnos = mna["permno"].dropna().unique().tolist()
    link = link[link["lpermno"].isin(permnos)].copy()
    if link.empty:
        return pd.DataFrame()

    mna["action_date"] = pd.to_datetime(mna[date_col], errors="coerce")
    merged = mna.merge(link, left_on="permno", right_on="lpermno", how="left")
    merged = merged[
        (merged["linkdt"] <= merged["action_date"]) & (merged["linkenddt"] >= merged["action_date"])
    ]
    if merged.empty:
        return pd.DataFrame()

    merged["rank"] = 0
    merged.loc[merged["linkprim"] != "P", "rank"] += 1
    merged.loc[~merged["linktype"].isin(["LC", "LU", "LD", "LN"]), "rank"] += 1
    merged = merged.sort_values(["deal_id", "rank", "linkdt"], ascending=[True, True, False])
    merged = merged.drop_duplicates(subset=["deal_id"], keep="first")
    merged["company_id"] = merged["gvkey"].astype(str).str.zfill(6)
    return merged


def _map_gvkey_fallback(mna: pd.DataFrame, mapped_ids: Optional[Iterable[Any]]) -> pd.DataFrame:
    if mna.empty or "acquiror_gvkey" not in mna.columns:
        return pd.DataFrame()
    fallback = mna.copy()
    fallback["company_id"] = _normalize_gvkey(fallback["acquiror_gvkey"])
    fallback = fallback[fallback["company_id"].notna()].copy()
    if mapped_ids is not None:
        mapped_set = set(mapped_ids)
        if mapped_set:
            fallback = fallback[~fallback["deal_id"].isin(mapped_set)].copy()
    fallback["mapping_source"] = "ciq_gvkey"
    return fallback


