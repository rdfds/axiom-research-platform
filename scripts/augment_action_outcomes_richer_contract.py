#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import load_config


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    raw = df.get(column)
    if isinstance(raw, pd.Series):
        return pd.to_numeric(raw, errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment an existing action outcomes parquet with richer contract-era macro and debt/liquidity columns."
    )
    parser.add_argument(
        "--in-path",
        default=str(ROOT / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet"),
    )
    parser.add_argument(
        "--out-path",
        default=str(ROOT / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.rich_contract_v3.parquet"),
    )
    parser.add_argument(
        "--fundamentals-path",
        default=str(ROOT / "data" / "curated" / "fundamentals_master.parquet"),
    )
    parser.add_argument(
        "--raw-timeseries-path",
        default=str(ROOT / "data" / "inputs_layer" / "raw_timeseries.parquet"),
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "pipeline_v1.json"),
    )
    parser.add_argument("--duckdb-memory", default="4GB")
    parser.add_argument("--duckdb-threads", type=int, default=1)
    return parser.parse_args()


def _macro_subquery(*, macro_path: Path, series_id: Optional[str], value_alias: str) -> str:
    if not series_id:
        return f"SELECT CAST(NULL AS DOUBLE) AS {value_alias}, CAST(NULL AS TIMESTAMP) AS event_time WHERE FALSE"
    return f"""
        SELECT
            CAST(value AS DOUBLE) AS {value_alias},
            CAST(event_time AS TIMESTAMP) AS event_time
        FROM read_parquet('{macro_path.as_posix()}')
        WHERE entity_id = '{series_id}'
        ORDER BY event_time
    """


def _enrich(
    *,
    in_path: Path,
    out_path: Path,
    fundamentals_path: Path,
    raw_timeseries_path: Path,
    config_path: Path,
    duckdb_memory: str,
    duckdb_threads: int,
) -> None:
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input outcomes parquet: {in_path}")
    if not fundamentals_path.exists():
        raise FileNotFoundError(f"Missing fundamentals parquet: {fundamentals_path}")
    if not raw_timeseries_path.exists():
        raise FileNotFoundError(f"Missing raw timeseries parquet: {raw_timeseries_path}")

    config = load_config(str(config_path))
    macro_series = dict(config.get("macro_series", {}) or {})
    macro_path = ROOT / "data" / "warehouse" / "warehouse_macro.parquet"
    if not macro_path.exists():
        raise FileNotFoundError(f"Missing macro warehouse parquet: {macro_path}")

    fed_series = macro_series.get("fed_funds_effective")
    gdp_series = macro_series.get("real_gdp") or macro_series.get("real_gdp_level")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{duckdb_memory}'")
    con.execute(f"SET threads={int(duckdb_threads)}")
    con.execute("SET preserve_insertion_order=false")

    row_count = con.execute(
        f"SELECT count(*) FROM read_parquet('{in_path.as_posix()}')"
    ).fetchone()[0]
    print(f"[augment_richer_contract] loaded {in_path} rows={row_count:,}", flush=True)

    fed_subquery = _macro_subquery(
        macro_path=macro_path,
        series_id=fed_series,
        value_alias="macro_fed_funds_effective",
    )
    gdp_subquery = _macro_subquery(
        macro_path=macro_path,
        series_id=gdp_series,
        value_alias="macro_real_gdp",
    )

    query = f"""
    WITH actions AS (
        SELECT
            row_number() OVER () AS __row_id,
            *,
            lpad(regexp_extract(CAST(company_id AS VARCHAR), '[0-9]+', 0), 6, '0') AS company_id_norm,
            CAST(action_date AS TIMESTAMP) AS action_ts
        FROM read_parquet('{in_path.as_posix()}')
    ),
    actions_ids AS (
        SELECT DISTINCT company_id_norm AS gvkey
        FROM actions
    ),
    fundamentals AS (
        SELECT
            f.gvkey,
            CAST(datadate AS TIMESTAMP) AS datadate,
            SUM(revtq) OVER w AS revenue_ttm,
            SUM(oibdpq) OVER w AS ebitda_ttm,
            SUM(xintq) OVER w AS interest_expense_ttm,
            CAST(cheq AS DOUBLE) AS cash,
            CAST(COALESCE(dlcq, 0) AS DOUBLE) AS current_debt,
            CAST(COALESCE(dlttq, 0) + COALESCE(dlcq, 0) AS DOUBLE) AS total_debt,
            CAST(oancfy AS DOUBLE) AS operating_cash_flow,
            CAST(capxy AS DOUBLE) AS capex,
            CAST(COALESCE(mkvaltq, prccq * cshoq) AS DOUBLE) AS market_cap
        FROM read_parquet('{fundamentals_path.as_posix()}') f
        JOIN actions_ids a
          ON a.gvkey = f.gvkey
        WINDOW w AS (
            PARTITION BY f.gvkey
            ORDER BY datadate
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        )
    ),
    base AS (
        SELECT
            a.__row_id,
            f.revenue_ttm AS base_revenue_ttm,
            f.ebitda_ttm AS base_ebitda_ttm,
            f.cash AS base_cash,
            f.total_debt AS base_total_debt,
            f.current_debt AS base_current_debt,
            f.cash AS base_available_liquidity,
            f.interest_expense_ttm AS base_interest_expense,
            CASE
              WHEN f.operating_cash_flow IS NOT NULL
               AND f.capex IS NOT NULL
               AND f.market_cap IS NOT NULL
               AND f.market_cap != 0
              THEN (f.operating_cash_flow - f.capex) / f.market_cap
              ELSE NULL
            END AS base_fcf_yield
        FROM actions a
        LEFT JOIN fundamentals f
          ON f.gvkey = a.company_id_norm
         AND f.datadate <= a.action_ts
        QUALIFY row_number() OVER (PARTITION BY a.__row_id ORDER BY f.datadate DESC) = 1
    ),
    base_prior AS (
        SELECT
            a.__row_id,
            f.revenue_ttm AS base_revenue_ttm_lag_1y
        FROM (
            SELECT __row_id, company_id_norm, action_ts - INTERVAL '1 year' AS prior_ts
            FROM actions
        ) a
        LEFT JOIN fundamentals f
          ON f.gvkey = a.company_id_norm
         AND f.datadate <= a.prior_ts
        QUALIFY row_number() OVER (PARTITION BY a.__row_id ORDER BY f.datadate DESC) = 1
    ),
    price_points AS (
        SELECT
            lpad(regexp_extract(CAST(company_id AS VARCHAR), '[0-9]+', 0), 6, '0') AS gvkey,
            CAST(COALESCE(trade_date, event_time) AS TIMESTAMP) AS price_ts,
            CAST(COALESCE(adjusted_close, close) AS DOUBLE) AS price
        FROM read_parquet('{raw_timeseries_path.as_posix()}')
        WHERE series_type = 'price'
          AND company_id IS NOT NULL
          AND COALESCE(adjusted_close, close) IS NOT NULL
          AND lpad(regexp_extract(CAST(company_id AS VARCHAR), '[0-9]+', 0), 6, '0') IN (SELECT gvkey FROM actions_ids)
    ),
    price_enriched AS (
        SELECT
            gvkey,
            price_ts,
            price,
            CASE
              WHEN lag(price) OVER (PARTITION BY gvkey ORDER BY price_ts) > 0
              THEN (price / lag(price) OVER (PARTITION BY gvkey ORDER BY price_ts)) - 1.0
              ELSE NULL
            END AS ret
        FROM price_points
    ),
    price_windows AS (
        SELECT
            gvkey,
            price_ts,
            price,
            stddev_pop(ret) OVER (
                PARTITION BY gvkey
                ORDER BY price_ts
                RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW
            ) * sqrt(252.0) AS base_volatility_30d,
            count(ret) OVER (
                PARTITION BY gvkey
                ORDER BY price_ts
                RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW
            ) AS ret_count_30d,
            stddev_pop(ret) OVER (
                PARTITION BY gvkey
                ORDER BY price_ts
                RANGE BETWEEN INTERVAL 90 DAYS PRECEDING AND CURRENT ROW
            ) * sqrt(252.0) AS base_volatility_90d,
            count(ret) OVER (
                PARTITION BY gvkey
                ORDER BY price_ts
                RANGE BETWEEN INTERVAL 90 DAYS PRECEDING AND CURRENT ROW
            ) AS ret_count_90d,
            min(price) OVER (
                PARTITION BY gvkey
                ORDER BY price_ts
                RANGE BETWEEN INTERVAL 90 DAYS PRECEDING AND CURRENT ROW
            ) AS min_price_90d,
            max(price) OVER (
                PARTITION BY gvkey
                ORDER BY price_ts
                RANGE BETWEEN INTERVAL 90 DAYS PRECEDING AND CURRENT ROW
            ) AS max_price_90d,
            count(price) OVER (
                PARTITION BY gvkey
                ORDER BY price_ts
                RANGE BETWEEN INTERVAL 90 DAYS PRECEDING AND CURRENT ROW
            ) AS price_count_90d
        FROM price_enriched
    ),
    price_with_momentum AS (
        SELECT
            p.gvkey,
            p.price_ts,
            p.price,
            CASE
              WHEN p.ret_count_30d >= 10 THEN p.base_volatility_30d
              ELSE NULL
            END AS base_volatility_30d,
            CASE
              WHEN p.ret_count_90d >= 20 THEN p.base_volatility_90d
              ELSE NULL
            END AS base_volatility_90d,
            CASE
              WHEN p.price_count_90d >= 20 AND p.max_price_90d > 0
              THEN (p.min_price_90d / p.max_price_90d) - 1.0
              ELSE NULL
            END AS base_drawdown_90d,
            prior.price AS price_60d
        FROM price_windows p
        ASOF LEFT JOIN price_points prior
          ON p.gvkey = prior.gvkey
         AND (p.price_ts - INTERVAL 60 DAYS) >= prior.price_ts
    ),
    price_asof AS (
        SELECT
            a.__row_id,
            p.base_volatility_30d,
            p.base_volatility_90d,
            p.base_drawdown_90d,
            CASE
              WHEN p.price_60d IS NOT NULL AND p.price_60d != 0
              THEN (p.price / p.price_60d) - 1.0
              ELSE NULL
            END AS base_momentum_60d
        FROM actions a
        ASOF LEFT JOIN price_with_momentum p
          ON a.company_id_norm = p.gvkey
         AND a.action_ts >= p.price_ts
    ),
    fed_series AS ({fed_subquery}),
    fed AS (
        SELECT
            a.__row_id,
            f.macro_fed_funds_effective
        FROM actions a
        ASOF LEFT JOIN fed_series f
          ON a.action_ts >= f.event_time
    ),
    gdp_series AS ({gdp_subquery}),
    gdp_current AS (
        SELECT
            a.__row_id,
            g.macro_real_gdp
        FROM actions a
        ASOF LEFT JOIN gdp_series g
          ON a.action_ts >= g.event_time
    ),
    gdp_prior AS (
        SELECT
            a.__row_id,
            g.macro_real_gdp AS macro_real_gdp_prior
        FROM (
            SELECT __row_id, action_ts - INTERVAL '1 year' AS prior_ts
            FROM actions
        ) a
        ASOF LEFT JOIN gdp_series g
          ON a.prior_ts >= g.event_time
    )
    SELECT
        a.* EXCLUDE (__row_id, company_id_norm, action_ts),
        base.base_revenue_ttm,
        base.base_ebitda_ttm,
        base.base_cash,
        base.base_total_debt,
        base.base_current_debt,
        base.base_available_liquidity,
        base.base_interest_expense,
        base.base_fcf_yield,
        base_prior.base_revenue_ttm_lag_1y,
        price_asof.base_volatility_30d,
        price_asof.base_volatility_90d,
        price_asof.base_drawdown_90d,
        price_asof.base_momentum_60d,
        CASE
          WHEN base.base_revenue_ttm IS NOT NULL
           AND base_prior.base_revenue_ttm_lag_1y IS NOT NULL
           AND base_prior.base_revenue_ttm_lag_1y != 0
          THEN (base.base_revenue_ttm / base_prior.base_revenue_ttm_lag_1y) - 1.0
          ELSE NULL
        END AS base_revenue_growth_yoy,
        fed.macro_fed_funds_effective,
        gdp_current.macro_real_gdp,
        CASE
          WHEN gdp_current.macro_real_gdp IS NOT NULL
           AND gdp_prior.macro_real_gdp_prior IS NOT NULL
           AND gdp_prior.macro_real_gdp_prior != 0
          THEN (gdp_current.macro_real_gdp / gdp_prior.macro_real_gdp_prior) - 1.0
          ELSE NULL
        END AS macro_real_gdp_growth_yoy
    FROM actions a
    LEFT JOIN base ON base.__row_id = a.__row_id
    LEFT JOIN base_prior ON base_prior.__row_id = a.__row_id
    LEFT JOIN price_asof ON price_asof.__row_id = a.__row_id
    LEFT JOIN fed ON fed.__row_id = a.__row_id
    LEFT JOIN gdp_current ON gdp_current.__row_id = a.__row_id
    LEFT JOIN gdp_prior ON gdp_prior.__row_id = a.__row_id
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(".tmp.parquet")
    con.execute(f"COPY ({query}) TO '{temp_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd')")
    con.close()

    df = pd.read_parquet(temp_path)

    leverage_proxy = _numeric_series(df, "base_leverage")
    debt = _numeric_series(df, "base_total_debt")
    ebitda = _numeric_series(df, "base_ebitda_ttm")
    leverage_proxy = leverage_proxy.where(leverage_proxy.notna(), debt / ebitda.where(ebitda > 0))

    vol_30 = _numeric_series(df, "base_volatility_30d")
    drawdown_90 = _numeric_series(df, "base_drawdown_90d")
    liquidity = _numeric_series(df, "base_available_liquidity")
    macro_ig = _numeric_series(df, "macro_ig_oas")
    macro_hy = _numeric_series(df, "macro_hy_oas")
    ev_ebitda = _numeric_series(df, "base_ev_ebitda")
    momentum_60d = _numeric_series(df, "base_momentum_60d")

    leverage_score = ((leverage_proxy - 1.5) / 5.0).clip(lower=0.0, upper=1.0)
    vol_score = ((vol_30 - 0.15) / 0.45).clip(lower=0.0, upper=1.0)
    draw_score = ((drawdown_90.clip(upper=0.0).abs() - 0.10) / 0.40).clip(lower=0.0, upper=1.0)
    liquidity_coverage = liquidity / debt.where(debt > 0)
    liquidity_score = (1.0 - liquidity_coverage).clip(lower=0.0, upper=1.0)

    risk_components = pd.DataFrame(
        {
            "leverage": leverage_score * 0.35,
            "volatility": vol_score * 0.25,
            "drawdown": draw_score * 0.15,
            "liquidity": liquidity_score * 0.15,
        }
    )
    risk_weights = pd.DataFrame(
        {
            "leverage": leverage_score.notna().astype(float) * 0.35,
            "volatility": vol_score.notna().astype(float) * 0.25,
            "drawdown": draw_score.notna().astype(float) * 0.15,
            "liquidity": liquidity_score.notna().astype(float) * 0.15,
        }
    )
    total_weight = risk_weights.sum(axis=1)
    risk_score = (risk_components.sum(axis=1) / total_weight.where(total_weight > 0)).where(total_weight > 0)

    base_credit_spread_level = ((macro_ig + (risk_score * (macro_hy - macro_ig))) / 100.0).where(
        macro_ig.notna() & macro_hy.notna() & risk_score.notna()
    )
    spread_access = (1.0 - (base_credit_spread_level / 0.08)).clip(lower=0.0, upper=1.0)

    eq_components = pd.DataFrame(
        {
            "volatility_component": (1.0 - (vol_30 / 0.8)).clip(lower=0.0, upper=1.0),
            "momentum_component": ((momentum_60d + 0.2) / 0.4).clip(lower=0.0, upper=1.0),
            "valuation_component": (ev_ebitda / 20.0).clip(lower=0.0, upper=1.0),
        }
    )
    base_equity_window_proxy = eq_components.mean(axis=1, skipna=True).where(eq_components.notna().any(axis=1))

    credit_components = pd.DataFrame(
        {
            "spread_component": (1.0 - (base_credit_spread_level / 0.10)).clip(lower=0.0, upper=1.0),
            "volatility_component": (1.0 - (vol_30 / 1.0)).clip(lower=0.0, upper=1.0),
        }
    )
    base_credit_window_proxy = credit_components.mean(axis=1, skipna=True).where(
        credit_components.notna().any(axis=1)
    )

    df["base_credit_spread_level"] = base_credit_spread_level
    df["base_equity_window_proxy"] = base_equity_window_proxy
    df["base_credit_window_proxy"] = base_credit_window_proxy
    df["base_credit_spread_access"] = spread_access

    df.to_parquet(out_path, index=False)
    try:
        temp_path.unlink()
    except FileNotFoundError:
        pass

    print(f"[augment_richer_contract] wrote {out_path}", flush=True)
    for col in (
        "base_revenue_ttm",
        "base_ebitda_ttm",
        "base_cash",
        "base_total_debt",
        "base_current_debt",
        "base_available_liquidity",
        "base_interest_expense",
        "base_fcf_yield",
        "base_revenue_ttm_lag_1y",
        "base_revenue_growth_yoy",
        "base_volatility_30d",
        "base_volatility_90d",
        "base_drawdown_90d",
        "base_momentum_60d",
        "base_credit_spread_level",
        "base_equity_window_proxy",
        "base_credit_window_proxy",
        "macro_fed_funds_effective",
        "macro_real_gdp_growth_yoy",
    ):
        coverage = int(df[col].notna().sum()) if col in df.columns else 0
        print(f"[augment_richer_contract] coverage {col}={coverage:,}", flush=True)


def main() -> None:
    args = _parse_args()
    _enrich(
        in_path=Path(args.in_path).resolve(),
        out_path=Path(args.out_path).resolve(),
        fundamentals_path=Path(args.fundamentals_path).resolve(),
        raw_timeseries_path=Path(args.raw_timeseries_path).resolve(),
        config_path=Path(args.config).resolve(),
        duckdb_memory=str(args.duckdb_memory),
        duckdb_threads=int(args.duckdb_threads),
    )


if __name__ == "__main__":
    main()
