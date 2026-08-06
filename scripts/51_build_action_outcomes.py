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

        def ttm_sum(item: str) -> Optional[float]:
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


def build_action_outcomes(
    actions_path: Path,
    out_path: Path,
    action_types: Optional[Iterable[str]],
    start_date: Optional[str],
    end_date: Optional[str],
    date_field: str,
    limit: Optional[int],
    log_every: int,
    config_path: Optional[str],
    fundamentals_path: Optional[str],
    include_mna: bool,
    mna_path: Optional[str],
    mna_statuses: Optional[Iterable[str]],
    mna_require_universe: bool,
    mna_countries: Optional[Iterable[str]],
    mna_limit: Optional[int],
    mna_limit_total: Optional[int],
    fmp_fallback: bool,
    fmp_path: Optional[str],
    horizons_override: Optional[Iterable[int]] = None,
    processed_log_every: int = 0,
    preload_fundamentals: bool = False,
) -> None:
    config = load_config(config_path)
    baseline_cfg = config.get("baseline", {})
    lag_q = int(baseline_cfg.get("post_action_lag_quarters", 1))
    t1_q = int(baseline_cfg.get("t1_quarters", 4))
    t1_offset = _months_from_quarters(lag_q + t1_q)

    outcome_cfg = config.get("outcome", {})
    horizons = list(horizons_override) if horizons_override else outcome_cfg.get("horizons_months", [3, 6, 12])

    cols = [
        "company_id",
        "gvkey",
        "acquiror_gvkey",
        "target_gvkey",
        "permco",
        "permno",
        "action_type",
        "action_subtype",
        "action_date",
        "event_time",
        "announcement_date",
        "effective_date",
        "size",
        "amount",
        "ratio",
        "split_factor",
        "ticker",
        "mapping_source",
        "source_id",
        "source_dataset",
    ]
    actions = _load_actions(actions_path, cols)
    if "company_id" not in actions.columns:
        actions["company_id"] = pd.NA
    if actions["company_id"].isna().any():
        for fallback_col in ("gvkey", "acquiror_gvkey", "target_gvkey", "permco", "permno"):
            if fallback_col in actions.columns:
                fallback = actions[fallback_col]
                actions["company_id"] = actions["company_id"].where(actions["company_id"].notna(), fallback)
    actions = actions.dropna(subset=["company_id"]).copy()
    actions["company_id"] = actions["company_id"].astype(str).str.zfill(6)
    actions["source_dataset"] = "corp_actions"
    print(f"[build_action_outcomes] loaded actions: {len(actions):,}", flush=True)
    print(f"[build_action_outcomes] horizons (months): {horizons}", flush=True)

    if action_types:
        action_types = [a.strip() for a in action_types if a.strip()]
        if action_types:
            actions = actions[actions["action_type"].isin(action_types)]
            print(f"[build_action_outcomes] filtered action_types: {len(actions):,}", flush=True)

    if start_date:
        start = pd.to_datetime(start_date)
        actions = actions[actions["event_time"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        actions = actions[actions["event_time"] <= end]

    actions = actions.sort_values(["company_id", "event_time"]).reset_index(drop=True)
    if limit:
        actions = actions.head(limit)

    macro_series = config.get("macro_series", {})
    macro_helper = FeatureBuilder()
    macro_cache: Dict[str, Dict[str, Any]] = {}

    fundamentals_path = fundamentals_path or config.get(
        "fundamentals_path", DATA_DIR / "curated" / "fundamentals_master.parquet"
    )
    fundamentals = FundamentalsProvider(Path(fundamentals_path), preload=preload_fundamentals)
    print(f"[build_action_outcomes] preload_fundamentals={preload_fundamentals}", flush=True)
    fmp_provider = None
    if fmp_fallback:
        fmp_path = fmp_path or str(DATA_DIR / "warehouse" / "warehouse_financials" / "year=*" / "part_*.parquet")
        price_path = DATA_DIR / "warehouse" / "warehouse_prices.parquet"
        ciq_path = DATA_DIR / "wrds" / "ciq" / "ciq_identifiers_map.parquet"
        fmp_provider = FmpFundamentalsProvider(fmp_path, price_path, ciq_path)

    state_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def get_state(
        company_id: str,
        as_of: pd.Timestamp,
        ticker: Optional[str] = None,
        prefer_fmp: bool = False,
    ) -> Dict[str, Any]:
        key = (company_id, _date_key(as_of), str(ticker).upper() if ticker else "", "fmp" if prefer_fmp else "compustat")
        if key in state_cache:
            return state_cache[key]
        features: Dict[str, Any] = {}
        if not prefer_fmp:
            features = fundamentals.get_metrics(company_id, as_of)
        if (not features or (features.get("revenue_ttm") is None and features.get("pe") is None)) and fmp_provider and ticker:
            features = fmp_provider.get_metrics(ticker, as_of)
        macro_key = _date_key(as_of)
        if macro_key in macro_cache:
            macro = macro_cache[macro_key]
        else:
            macro = macro_helper.compute_macro_features(as_of, macro_series)
            macro_cache[macro_key] = macro
        if features:
            features.update(macro)
        state_cache[key] = features
        return features

    rows = []
    skipped = 0
    processed = 0

    def process_actions(df: pd.DataFrame, label: str) -> int:
        nonlocal skipped, rows, processed
        if df.empty:
            return 0
        kept = 0
        for idx, row in df.iterrows():
            processed += 1
            action_date = _pick_date(row, date_field)
            if action_date is None or pd.isna(action_date):
                skipped += 1
                continue
            company_id = str(row["company_id"])
            t0 = pd.to_datetime(action_date)
            ticker = row.get("ticker") or row.get("acquiror_ticker")
            base = get_state(company_id, t0, ticker=ticker)

            if not base or (base.get("revenue_ttm") is None and base.get("pe") is None):
                skipped += 1
                continue
            source = base.get("fundamentals_source", "compustat")

            t1 = t0 + pd.DateOffset(months=t1_offset)
            after = get_state(company_id, t1, ticker=ticker, prefer_fmp=source == "fmp")

            record: Dict[str, Any] = {
                "company_id": company_id,
                "action_type": row.get("action_type"),
                "action_subtype": row.get("action_subtype"),
                "action_date": t0,
                "action_size": _resolved_action_size(row),
                "source_dataset": row.get("source_dataset", label),
                "source_id": row.get("source_id"),
                "fundamentals_source": source,
                "mapping_source": row.get("mapping_source"),
                "base_market_cap": base.get("market_cap"),
                "base_leverage": base.get("leverage_net_debt_ebitda"),
                "base_margin": base.get("ebitda_margin"),
                "base_revenue_ttm": base.get("revenue_ttm"),
                "base_roic": base.get("roic_proxy"),
                "base_fcf_margin": base.get("fcf_margin"),
                "base_pe": base.get("pe"),
                "base_ev_ebitda": base.get("ev_ebitda"),
                "revenue_delta": _pct_change(after.get("revenue_ttm"), base.get("revenue_ttm")),
                "margin_delta": _pp_change(after.get("ebitda_margin"), base.get("ebitda_margin")),
                "leverage_delta": _pp_change(
                    after.get("leverage_net_debt_ebitda"),
                    base.get("leverage_net_debt_ebitda"),
                ),
                "eps_delta": _pct_change(after.get("eps_ttm"), base.get("eps_ttm")),
                "roic_delta": _pp_change(after.get("roic_proxy"), base.get("roic_proxy")),
                "fcf_margin_delta": _pp_change(after.get("fcf_margin"), base.get("fcf_margin")),
            }
            _apply_richer_base_fields(record, base)

            for key, value in base.items():
                if key.startswith("macro_"):
                    record[key] = value

            base_pe = base.get("pe")
            base_ev = base.get("ev_ebitda")
            for horizon in horizons:
                t_h = t0 + pd.DateOffset(months=int(horizon))
                future = get_state(company_id, t_h, ticker=ticker, prefer_fmp=source == "fmp")
                record[f"outcome_pe_{int(horizon)}m"] = _pct_change(future.get("pe"), base_pe)
                record[f"outcome_ev_ebitda_{int(horizon)}m"] = _pct_change(
                    future.get("ev_ebitda"), base_ev
                )

            rows.append(record)
            kept += 1
            if log_every and (len(rows) % log_every == 0):
                print(
                    f"[build_action_outcomes] {label} kept {len(rows):,} rows "
                    f"(skipped {skipped:,})"
                )
            if processed_log_every and (processed % processed_log_every == 0):
                print(
                    f"[build_action_outcomes] {label} processed {processed:,} rows "
                    f"(kept {len(rows):,}, skipped {skipped:,})"
                )
        return kept

    if actions.empty and not include_mna:
        raise RuntimeError("No actions found after filtering.")

    process_actions(actions, "corp_actions")

    if include_mna:
        mna_path = Path(mna_path or (DATA_DIR / "curated" / "mna_master.parquet"))
        if not mna_path.exists():
            raise FileNotFoundError(f"Missing M&A dataset: {mna_path}")
        link_path = DATA_DIR / "wrds" / "crsp" / "ccmxpf_lnkhist.parquet"
        link_table = _load_link_table(link_path)
        if link_table.empty:
            raise FileNotFoundError(f"Missing CRSP link table: {link_path}")

        status_list = [s.strip() for s in (mna_statuses or ["Completed"]) if s.strip()]
        country_list = [c.strip().lower() for c in (mna_countries or []) if c.strip()]
        if country_list:
            country_aliases = {
                "us": "united states",
                "usa": "united states",
                "united states of america": "united states",
                "u.s.": "united states",
            }
            country_list = [country_aliases.get(c, c) for c in country_list]

        # Load by year to keep memory controlled
        years = pd.read_parquet(mna_path, columns=["year"])["year"].dropna().unique().tolist()
        years = sorted(int(y) for y in years)
        mna_kept = 0
        for yr in years:
            if mna_limit_total and mna_kept >= mna_limit_total:
                break
            mna = pd.read_parquet(
                mna_path,
                columns=[
                    "deal_id",
                    "announce_date",
                    "event_date",
                    "completion_date",
                    "deal_status",
                    "deal_type",
                    "deal_value",
                    "acquiror_permno",
                    "acquiror_gvkey",
                    "acquiror_name",
                    "acquiror_ticker",
                    "acquiror_country",
                    "acquiror_in_universe",
                    "year",
                ],
                filters=[("year", "=", yr)],
            )
            if mna.empty:
                continue
            if log_every:
                print(f"[build_action_outcomes] mna_master year {yr} rows {len(mna):,}")
            if status_list:
                mna = mna[mna["deal_status"].isin(status_list)]
            if mna_require_universe and "acquiror_in_universe" in mna.columns:
                mna = mna[mna["acquiror_in_universe"] == True]
            if country_list and "acquiror_country" in mna.columns:
                mna["_country_norm"] = (
                    mna["acquiror_country"].astype("string").str.strip().str.lower()
                )
                mna.loc[mna["_country_norm"] == "u.s.", "_country_norm"] = "united states"
                mna = mna[mna["_country_norm"].isin(country_list)]

            date_col = "announce_date"
            mna["announce_date"] = pd.to_datetime(mna["announce_date"], errors="coerce")
            mna["event_date"] = pd.to_datetime(mna["event_date"], errors="coerce")
            mna["completion_date"] = pd.to_datetime(mna["completion_date"], errors="coerce")
            mna["action_date"] = mna["announce_date"].fillna(mna["event_date"]).fillna(mna["completion_date"])
            if start_date:
                mna = mna[mna["action_date"] >= pd.to_datetime(start_date)]
            if end_date:
                mna = mna[mna["action_date"] <= pd.to_datetime(end_date)]
            if mna.empty:
                continue

            mapped = _map_permno_to_gvkey(mna, link_table, date_col="action_date")
            if not mapped.empty:
                mapped = mapped.copy()
                mapped["mapping_source"] = "permno_link"

            fallback = _map_gvkey_fallback(mna, mapped["deal_id"] if not mapped.empty else None)
            combined = pd.concat([mapped, fallback], ignore_index=True, sort=False)
            if combined.empty:
                continue

            combined["action_type"] = "acquisition"
            combined["action_subtype"] = combined.get("deal_type")
            combined["size"] = combined.get("deal_value")
            combined["source_id"] = combined.get("deal_id")
            combined["source_dataset"] = "mna_master"

            combined = combined.rename(
                columns={
                    "action_date": "event_time",
                }
            )

            if mna_limit:
                combined = combined.head(mna_limit)
            if mna_limit_total:
                remaining = max(mna_limit_total - mna_kept, 0)
                combined = combined.head(remaining)

            mna_kept += process_actions(combined, "mna_master")


    df = pd.DataFrame(rows)
    df = augment_action_outcomes_df(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(
        f"[build_action_outcomes] wrote {len(df):,} rows to {out_path} "
        f"(skipped {skipped:,})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actions-path",
        default=str(DATA_DIR / "warehouse" / "warehouse_corp_actions.parquet"),
    )
    parser.add_argument(
        "--out",
        default=str(DATA_DIR / "curated" / "action_outcomes.parquet"),
    )
    parser.add_argument(
        "--action-types",
        default=None,
        help="Comma-separated action types to include. Default is all action types.",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date-field", default="auto", choices=["auto", "event_time", "announcement_date", "effective_date"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--fundamentals-path",
        default=str(DATA_DIR / "curated" / "fundamentals_master.parquet"),
        help="Parquet file with quarterly fundamentals (Compustat-derived).",
    )
    parser.add_argument("--include-mna", action="store_true", help="Include M&A actions from mna_master.parquet.")
    parser.add_argument(
        "--mna-path",
        default=str(DATA_DIR / "curated" / "mna_master.parquet"),
        help="Parquet file with M&A deal data.",
    )
    parser.add_argument("--mna-status", default="Completed", help="Comma-separated M&A deal statuses to include.")
    parser.add_argument("--mna-require-universe", dest="mna_require_universe", action="store_true")
    parser.add_argument("--mna-no-require-universe", dest="mna_require_universe", action="store_false")
    parser.set_defaults(mna_require_universe=True)
    parser.add_argument(
        "--mna-country",
        default=None,
        help="Comma-separated acquiror countries to include (case-insensitive).",
    )
    parser.add_argument("--mna-limit", type=int, default=None)
    parser.add_argument(
        "--mna-limit-total",
        type=int,
        default=None,
        help="Cap total M&A rows kept across all years.",
    )
    parser.add_argument("--fmp-fallback", action="store_true", help="Fallback to FMP fundamentals when Compustat missing.")
    parser.add_argument(
        "--fmp-path",
        default=str(DATA_DIR / "warehouse" / "warehouse_financials" / "year=*" / "part_*.parquet"),
        help="Path glob to FMP financials parquet files.",
    )
    parser.add_argument(
        "--horizons",
        default=None,
        help="Comma-separated outcome horizons in months (e.g., 6,12). Overrides config.",
    )
    parser.add_argument(
        "--processed-log-every",
        type=int,
        default=0,
        help="Log progress every N processed rows (kept + skipped).",
    )
    parser.add_argument(
        "--preload-fundamentals",
        action="store_true",
        help="Preload fundamentals_master into memory for faster lookups.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use DuckDB set-based pipeline for faster builds (ignores FMP fallback).",
    )
    parser.add_argument(
        "--skip-macro",
        action="store_true",
        help="Skip macro join in fast mode; use backfill script later.",
    )
    parser.add_argument("--duckdb-memory", default=None, help="DuckDB memory limit, e.g. 8GB.")
    parser.add_argument("--duckdb-threads", type=int, default=None, help="DuckDB threads (lower reduces memory).")
    parser.add_argument(
        "--duckdb-preserve-order",
        action="store_true",
        default=None,
        help="Set preserve_insertion_order=true (default false).",
    )
    args = parser.parse_args()

    action_types = args.action_types.split(",") if args.action_types else None
    mna_statuses = args.mna_status.split(",") if args.mna_status else None
    mna_countries = args.mna_country.split(",") if args.mna_country else None

    horizons_override = None
    if args.horizons:
        horizons_override = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    if args.fast:
        if args.include_mna or args.fmp_fallback:
            print("[build_action_outcomes_fast] fast mode ignores M&A + FMP fallback for now.", flush=True)
        build_action_outcomes_fast(
            actions_path=Path(args.actions_path),
            out_path=Path(args.out),
            action_types=action_types,
            start_date=args.start_date,
            end_date=args.end_date,
            date_field=args.date_field,
            config_path=args.config,
            fundamentals_path=args.fundamentals_path,
            horizons_override=horizons_override,
            include_macro=not args.skip_macro,
            duckdb_memory=args.duckdb_memory,
            duckdb_threads=args.duckdb_threads,
            duckdb_preserve_order=args.duckdb_preserve_order,
        )
    else:
        build_action_outcomes(
            actions_path=Path(args.actions_path),
            out_path=Path(args.out),
            action_types=action_types,
            start_date=args.start_date,
            end_date=args.end_date,
            date_field=args.date_field,
            limit=args.limit,
            log_every=args.log_every,
            config_path=args.config,
            fundamentals_path=args.fundamentals_path,
            include_mna=args.include_mna,
            mna_path=args.mna_path,
            mna_statuses=mna_statuses,
            mna_require_universe=args.mna_require_universe,
            mna_countries=mna_countries,
            mna_limit=args.mna_limit,
            mna_limit_total=args.mna_limit_total,
            fmp_fallback=args.fmp_fallback,
            fmp_path=args.fmp_path,
            horizons_override=horizons_override,
            processed_log_every=args.processed_log_every,
            preload_fundamentals=args.preload_fundamentals,
        )


def _sql_in_list(values: Iterable[str]) -> str:
    vals = [f"'{v}'" for v in values]
    return ",".join(vals) if vals else ""


def build_action_outcomes_fast(
    actions_path: Path,
    out_path: Path,
    action_types: Optional[Iterable[str]],
    start_date: Optional[str],
    end_date: Optional[str],
    date_field: str,
    config_path: Optional[str],
    fundamentals_path: Optional[str],
    horizons_override: Optional[Iterable[int]] = None,
    include_macro: bool = True,
    duckdb_memory: Optional[str] = None,
    duckdb_threads: Optional[int] = None,
    duckdb_preserve_order: Optional[bool] = None,
) -> None:
    config = load_config(config_path)
    baseline_cfg = config.get("baseline", {})
    lag_q = int(baseline_cfg.get("post_action_lag_quarters", 1))
    t1_q = int(baseline_cfg.get("t1_quarters", 4))
    t1_offset = _months_from_quarters(lag_q + t1_q)

    outcome_cfg = config.get("outcome", {})
    horizons = list(horizons_override) if horizons_override else outcome_cfg.get("horizons_months", [3, 6, 12])

    if not actions_path.exists():
        raise FileNotFoundError(f"Missing corporate actions dataset: {actions_path}")
    if not fundamentals_path or not Path(fundamentals_path).exists():
        raise FileNotFoundError(f"Missing fundamentals dataset: {fundamentals_path}")

    macro_series = config.get("macro_series", {})
    macro_ids = [macro_series[k] for k in macro_series.keys()] if include_macro else []

    import pyarrow.parquet as pq

    cols = pq.ParquetFile(actions_path.as_posix()).schema.names
    def has(col: str) -> bool:
        return col in cols

    date_candidates = [
        "announcement_date",
        "announce_date",
        "action_date",
        "event_time",
        "effective_date",
        "completion_date",
        "event_date",
    ]
    available_dates = [c for c in date_candidates if has(c)]
    if date_field != "auto":
        date_expr = date_field
    else:
        if not available_dates:
            raise RuntimeError("No date columns found for auto date selection.")
        date_expr = "COALESCE(" + ",".join(available_dates) + ")"

    def coalesce_expr(candidates: Iterable[str], cast_text: bool = False) -> str:
        cols_local = [c for c in candidates if has(c)]
        if not cols_local:
            return "NULL"
        if cast_text:
            cols_local = [f"CAST({c} AS VARCHAR)" for c in cols_local]
        return "COALESCE(" + ",".join(cols_local) + ")"

    source_dataset_expr = coalesce_expr(["source_dataset", "source", "source_system"], cast_text=True)
    source_id_expr = coalesce_expr(["source_id", "raw_payload_hash", "deal_id", "facilityid", "ISSUE_ID"], cast_text=True)
    mapping_source_expr = coalesce_expr(["mapping_source"], cast_text=True)
    ticker_expr = coalesce_expr(["ticker", "tic", "acquiror_ticker", "target_ticker"], cast_text=True)
    action_subtype_expr = coalesce_expr(["action_subtype", "action_code", "source_action_subtype"], cast_text=True)
    size_expr = coalesce_expr(
        [
            "size",
            "amount",
            "ratio",
            "split_factor",
            "facpr",
            "divamt",
            "deal_value",
            "offering_amt_k",
            "principal_amt",
            "dealamount",
        ],
        cast_text=False,
    )
    if size_expr != "NULL":
        size_expr = f"CAST({size_expr} AS DOUBLE)"
    positive_split_size_expr = coalesce_expr(
        [
            "split_factor",
            "facpr",
            "ratio",
            "size",
            "amount",
            "divamt",
            "deal_value",
            "offering_amt_k",
            "principal_amt",
            "dealamount",
        ],
        cast_text=False,
    )
    if positive_split_size_expr != "NULL":
        split_candidates = [
            "split_factor",
            "facpr",
            "ratio",
            "size",
            "amount",
            "divamt",
            "deal_value",
            "offering_amt_k",
            "principal_amt",
            "dealamount",
        ]
        positive_parts = [
            f"CASE WHEN CAST({col} AS DOUBLE) > 0 THEN CAST({col} AS DOUBLE) ELSE NULL END"
            for col in split_candidates
            if has(col)
        ]
        positive_split_size_expr = "COALESCE(" + ",".join(positive_parts) + ")" if positive_parts else "NULL"
    if size_expr != "NULL" and positive_split_size_expr != "NULL":
        size_expr = (
            "CASE "
            "WHEN lower(CAST(action_type AS VARCHAR)) IN ('split','stock_split','reverse_split') "
            f"THEN COALESCE({positive_split_size_expr}, {size_expr}) "
            f"ELSE {size_expr} END"
        )

    company_raw_expr = coalesce_expr(
        [
            "company_id",
            "gvkey",
            "acquiror_gvkey",
            "target_gvkey",
            "acquiror_permno",
            "permno",
        ],
        cast_text=True,
    )
    if company_raw_expr == "NULL":
        raise RuntimeError("No company identifier columns found in actions_path.")
    company_id_expr = (
        f"CASE WHEN {company_raw_expr} IS NULL THEN NULL "
        f"ELSE lpad(regexp_extract(CAST({company_raw_expr} AS VARCHAR), '[0-9]+', 0), 6, '0') END"
    )
    action_filter = ""
    if action_types:
        action_filter = f"AND action_type IN ({_sql_in_list(action_types)})"

    date_filter = ""
    if start_date:
        date_filter += f" AND {date_expr} >= DATE '{start_date}'"
    if end_date:
        date_filter += f" AND {date_expr} <= DATE '{end_date}'"

    horizons = sorted(set(int(h) for h in horizons))
    horizon_ctes = []
    horizon_selects = []
    base_pe_expr = "(base.price / base.eps_ttm)"
    base_ev_expr = "((COALESCE(base.market_cap_raw, base.price * base.shares_out) + (base.debt - COALESCE(base.cash,0))) / base.ebitda_ttm)"
    for h in horizons:
        alias = f"h{h}"
        horizon_ctes.append(
            f"""{alias} AS (
                SELECT d.action_id, f.*
                FROM dates d
                LEFT JOIN fund_snap f
                  ON f.gvkey = d.company_id
                 AND f.datadate <= d.action_date + INTERVAL '{h} months'
                QUALIFY row_number() OVER (PARTITION BY d.action_id ORDER BY f.datadate DESC) = 1
            )"""
        )
        horizon_selects.append(
            f"""
            CASE
              WHEN base.price IS NOT NULL AND base.eps_ttm IS NOT NULL AND base.eps_ttm != 0
                   AND {alias}.price IS NOT NULL AND {alias}.eps_ttm IS NOT NULL AND {alias}.eps_ttm != 0
              THEN (({alias}.price / {alias}.eps_ttm) - {base_pe_expr}) / ABS({base_pe_expr})
              ELSE NULL
            END AS outcome_pe_{h}m,
            CASE
              WHEN COALESCE(base.market_cap_raw, base.price * base.shares_out) IS NOT NULL
                   AND base.debt IS NOT NULL AND base.ebitda_ttm IS NOT NULL AND base.ebitda_ttm != 0
                   AND COALESCE({alias}.market_cap_raw, {alias}.price * {alias}.shares_out) IS NOT NULL
                   AND {alias}.debt IS NOT NULL AND {alias}.ebitda_ttm IS NOT NULL AND {alias}.ebitda_ttm != 0
              THEN ((
                    (COALESCE({alias}.market_cap_raw, {alias}.price * {alias}.shares_out)
                     + ({alias}.debt - COALESCE({alias}.cash,0))) / {alias}.ebitda_ttm
                   ) - {base_ev_expr}) / ABS({base_ev_expr})
              ELSE NULL
            END AS outcome_ev_ebitda_{h}m
            """
        )

    macro_selects = []
    if include_macro:
        for name, series_id in macro_series.items():
            macro_selects.append(
                f"arg_max(value, event_time) FILTER (WHERE entity_id = '{series_id}') AS macro_{name}"
            )

    macro_cte = ""
    macro_join = ""
    macro_cols = ""
    if include_macro and macro_series:
        macro_cte = f""",
    macro AS (
        SELECT
            d.action_id,
            {", ".join(macro_selects)}
        FROM dates d
        LEFT JOIN read_parquet('{(DATA_DIR / "warehouse" / "warehouse_macro.parquet").as_posix()}') m
          ON m.entity_id IN ({_sql_in_list(macro_ids)})
         AND CAST(m.event_time AS TIMESTAMP) <= d.action_date
        GROUP BY d.action_id
    )
        """
        macro_join = "LEFT JOIN macro ON macro.action_id = d.action_id"
        macro_cols = ", " + ", ".join([f"macro.macro_{name}" for name in macro_series.keys()])

    query = f"""
    WITH actions AS (
        SELECT
            row_number() OVER () AS action_id,
            {company_id_expr} AS company_id,
            action_type,
            {action_subtype_expr} AS action_subtype,
            {size_expr} AS size,
            {ticker_expr} AS ticker,
            {mapping_source_expr} AS mapping_source,
            {source_id_expr} AS source_id,
            {source_dataset_expr} AS source_dataset,
            CAST({date_expr} AS TIMESTAMP) AS action_date
        FROM read_parquet('{actions_path.as_posix()}')
        WHERE {company_id_expr} IS NOT NULL
          AND {date_expr} IS NOT NULL
          {action_filter}
          {date_filter}
    ),
    actions_ids AS (
        SELECT DISTINCT company_id FROM actions
    ),
    fund_snap AS (
        SELECT
            f.gvkey,
            CAST(datadate AS TIMESTAMP) AS datadate,
            -- TTM sums over last 4 quarters
            SUM(revtq) OVER w AS revenue_ttm,
            SUM(oibdpq) OVER w AS ebitda_ttm,
            SUM(niq) OVER w AS net_income_ttm,
            SUM(epspxq) OVER w AS eps_ttm,
            cshoq AS shares_out,
            cheq AS cash,
            (COALESCE(dlttq,0) + COALESCE(dlcq,0)) AS debt,
            atq AS total_assets,
            oancfy AS oancfy,
            capxy AS capxy,
            prccq AS price,
            mkvaltq AS market_cap_raw
        FROM read_parquet('{Path(fundamentals_path).as_posix()}') f
        JOIN actions_ids a
          ON a.company_id = f.gvkey
        WINDOW w AS (
            PARTITION BY gvkey
            ORDER BY datadate
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        )
    ),
    dates AS (
        SELECT
            action_id,
            company_id,
            action_type,
            action_subtype,
            action_date,
            size,
            ticker,
            mapping_source,
            source_id,
            source_dataset
        FROM actions
    ),
    base AS (
        SELECT d.action_id, f.*
        FROM dates d
        LEFT JOIN fund_snap f
          ON f.gvkey = d.company_id
         AND f.datadate <= d.action_date
        QUALIFY row_number() OVER (PARTITION BY d.action_id ORDER BY f.datadate DESC) = 1
    ),
    t1 AS (
        SELECT d.action_id, f.*
        FROM dates d
        LEFT JOIN fund_snap f
          ON f.gvkey = d.company_id
         AND f.datadate <= d.action_date + INTERVAL '{t1_offset} months'
        QUALIFY row_number() OVER (PARTITION BY d.action_id ORDER BY f.datadate DESC) = 1
    ),
    {", ".join(horizon_ctes)}
    {macro_cte}
    SELECT
        d.company_id,
        d.action_type,
        d.action_subtype,
        d.action_date,
        d.size AS action_size,
        d.ticker,
        d.mapping_source,
        d.source_id,
        d.source_dataset,
        -- base metrics
        COALESCE(base.market_cap_raw, base.price * base.shares_out) AS base_market_cap,
        base.cash AS base_cash,
        base.debt AS base_total_debt,
        base.cash AS base_available_liquidity,
        CASE
          WHEN base.revenue_ttm IS NOT NULL AND base.ebitda_ttm IS NOT NULL AND base.revenue_ttm != 0
          THEN base.ebitda_ttm / base.revenue_ttm
          ELSE NULL
        END AS base_margin,
        CASE
          WHEN base.debt IS NOT NULL THEN base.debt - COALESCE(base.cash,0)
          ELSE NULL
        END AS base_net_debt,
        CASE
          WHEN base.debt IS NOT NULL AND base.ebitda_ttm IS NOT NULL AND base.ebitda_ttm != 0
          THEN (base.debt - COALESCE(base.cash,0)) / base.ebitda_ttm
          ELSE NULL
        END AS base_leverage,
        base.revenue_ttm AS base_revenue_ttm,
        CASE
          WHEN base.net_income_ttm IS NOT NULL AND base.total_assets IS NOT NULL AND base.total_assets != 0
          THEN base.net_income_ttm / base.total_assets
          ELSE NULL
        END AS base_roic,
        CASE
          WHEN base.oancfy IS NOT NULL AND base.capxy IS NOT NULL AND base.revenue_ttm IS NOT NULL AND base.revenue_ttm != 0
          THEN (base.oancfy - base.capxy) / base.revenue_ttm
          ELSE NULL
        END AS base_fcf_margin,
        CASE
          WHEN base.price IS NOT NULL AND base.eps_ttm IS NOT NULL AND base.eps_ttm != 0
          THEN base.price / base.eps_ttm
          ELSE NULL
        END AS base_pe,
        CASE
          WHEN COALESCE(base.market_cap_raw, base.price * base.shares_out) IS NOT NULL
               AND base.debt IS NOT NULL AND base.ebitda_ttm IS NOT NULL AND base.ebitda_ttm != 0
          THEN (COALESCE(base.market_cap_raw, base.price * base.shares_out) + (base.debt - COALESCE(base.cash,0))) / base.ebitda_ttm
          ELSE NULL
        END AS base_ev_ebitda,
        -- deltas (t1 vs base)
        CASE
          WHEN base.revenue_ttm IS NOT NULL AND base.revenue_ttm != 0 AND t1.revenue_ttm IS NOT NULL
          THEN (t1.revenue_ttm - base.revenue_ttm) / ABS(base.revenue_ttm)
          ELSE NULL
        END AS revenue_delta,
        CASE
          WHEN base.revenue_ttm IS NOT NULL AND base.ebitda_ttm IS NOT NULL AND base.revenue_ttm != 0
               AND t1.revenue_ttm IS NOT NULL AND t1.ebitda_ttm IS NOT NULL AND t1.revenue_ttm != 0
          THEN (t1.ebitda_ttm / t1.revenue_ttm) - (base.ebitda_ttm / base.revenue_ttm)
          ELSE NULL
        END AS margin_delta,
        CASE
          WHEN base.debt IS NOT NULL AND base.ebitda_ttm IS NOT NULL AND base.ebitda_ttm != 0
               AND t1.debt IS NOT NULL AND t1.ebitda_ttm IS NOT NULL AND t1.ebitda_ttm != 0
          THEN ((t1.debt - COALESCE(t1.cash,0)) / t1.ebitda_ttm) - ((base.debt - COALESCE(base.cash,0)) / base.ebitda_ttm)
          ELSE NULL
        END AS leverage_delta,
        CASE
          WHEN base.eps_ttm IS NOT NULL AND base.eps_ttm != 0 AND t1.eps_ttm IS NOT NULL
          THEN (t1.eps_ttm - base.eps_ttm) / ABS(base.eps_ttm)
          ELSE NULL
        END AS eps_delta,
        CASE
          WHEN base.net_income_ttm IS NOT NULL AND base.total_assets IS NOT NULL AND base.total_assets != 0
               AND t1.net_income_ttm IS NOT NULL AND t1.total_assets IS NOT NULL AND t1.total_assets != 0
          THEN (t1.net_income_ttm / t1.total_assets) - (base.net_income_ttm / base.total_assets)
          ELSE NULL
        END AS roic_delta,
        CASE
          WHEN base.oancfy IS NOT NULL AND base.capxy IS NOT NULL AND base.revenue_ttm IS NOT NULL AND base.revenue_ttm != 0
               AND t1.oancfy IS NOT NULL AND t1.capxy IS NOT NULL AND t1.revenue_ttm IS NOT NULL AND t1.revenue_ttm != 0
          THEN ((t1.oancfy - t1.capxy) / t1.revenue_ttm) - ((base.oancfy - base.capxy) / base.revenue_ttm)
          ELSE NULL
        END AS fcf_margin_delta,
        {", ".join(horizon_selects)}
        {macro_cols}
    FROM dates d
    LEFT JOIN base ON base.action_id = d.action_id
    LEFT JOIN t1 ON t1.action_id = d.action_id
    {macro_join}
    {"" if not horizon_ctes else "LEFT JOIN " + " LEFT JOIN ".join([f"h{h} ON h{h}.action_id = d.action_id" for h in horizons])}
    WHERE base.revenue_ttm IS NOT NULL OR (base.price IS NOT NULL AND base.eps_ttm IS NOT NULL AND base.eps_ttm != 0)
    """

    con = duckdb.connect()
    if duckdb_memory:
        con.execute(f"SET memory_limit='{duckdb_memory}'")
    if duckdb_threads:
        con.execute(f"SET threads={int(duckdb_threads)}")
    if duckdb_preserve_order is not None:
        flag = "true" if duckdb_preserve_order else "false"
        con.execute(f"SET preserve_insertion_order={flag}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT 'parquet');")
    con.close()
    df = pd.read_parquet(out_path)
    if include_macro and macro_series:
        df = _enrich_macro_columns_from_helper(df, macro_series=macro_series)
    df = augment_action_outcomes_df(df)
    df.to_parquet(out_path, index=False)
    print(f"[build_action_outcomes_fast] wrote {out_path}", flush=True)

if __name__ == "__main__":
    main()
