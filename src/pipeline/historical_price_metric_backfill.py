from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Optional

import duckdb
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RAW_TIMESERIES_PATH = REPO_ROOT / "data/inputs_layer/raw_timeseries.parquet"
_PRICE_BACKFILL_COLS = (
    "base_volatility_30d",
    "base_volatility_90d",
    "base_drawdown_90d",
    "base_momentum_60d",
)
_STALE_STATE_VECTOR_COLS = (
    "state_vector_v1.market_stress",
    "state_vector_v1.market_access",
)


def _normalize_company_id_for_price_join(value: object) -> str:
    digits = re.search(r"[0-9]+", str(value or ""))
    if digits is None:
        return ""
    return digits.group(0).rjust(6, "0")[:6]


def default_raw_timeseries_path() -> Path:
    return _DEFAULT_RAW_TIMESERIES_PATH


def _missing_metric_mask(frame: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for col in cols:
        if col not in frame.columns:
            mask = mask | True
            continue
        mask = mask | pd.to_numeric(frame[col], errors="coerce").isna()
    return mask


def backfill_historical_price_window_metrics(
    frame: pd.DataFrame,
    *,
    raw_timeseries_path: Optional[Path] = None,
) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out

    if "company_id" not in out.columns or "action_date" not in out.columns:
        return out

    raw_path = Path(raw_timeseries_path or default_raw_timeseries_path())
    if not raw_path.exists():
        return out

    out = out.reset_index(drop=True)
    missing_mask = _missing_metric_mask(out, _PRICE_BACKFILL_COLS)
    action_ts = pd.to_datetime(out["action_date"], utc=True, errors="coerce")
    company_id_norm = out["company_id"].map(_normalize_company_id_for_price_join)
    eligible_mask = missing_mask & action_ts.notna() & company_id_norm.astype(bool)
    if not bool(eligible_mask.any()):
        return out

    actions = pd.DataFrame(
        {
            "__row_id": out.index[eligible_mask].to_numpy(dtype=int),
            "company_id_norm": company_id_norm[eligible_mask].astype(str).to_numpy(dtype=object),
            "action_ts": action_ts[eligible_mask].to_numpy(dtype="datetime64[ns]"),
        }
    )
    if actions.empty:
        return out

    con = duckdb.connect()
    try:
        schema = con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [str(raw_path)],
        ).df()
        available_cols = {str(value) for value in schema.get("column_name", pd.Series(dtype=str)).tolist()}
        if "trade_date" in available_cols and "event_time" in available_cols:
            price_ts_expr = "COALESCE(trade_date, event_time)"
        elif "trade_date" in available_cols:
            price_ts_expr = "trade_date"
        elif "event_time" in available_cols:
            price_ts_expr = "event_time"
        else:
            return out

        con.register("actions_df", actions)
        query = f"""
            WITH actions AS (
                SELECT __row_id, company_id_norm, action_ts
                FROM actions_df
            ),
            actions_ids AS (
                SELECT DISTINCT company_id_norm AS gvkey
                FROM actions
            ),
            price_points AS (
                SELECT
                    lpad(regexp_extract(CAST(company_id AS VARCHAR), '[0-9]+', 0), 6, '0') AS gvkey,
                    CAST({price_ts_expr} AS TIMESTAMP) AS price_ts,
                    CAST(COALESCE(adjusted_close, close) AS DOUBLE) AS price
                FROM read_parquet(?)
                WHERE series_type = 'price'
                  AND company_id IS NOT NULL
                  AND COALESCE(adjusted_close, close) IS NOT NULL
                  AND lpad(regexp_extract(CAST(company_id AS VARCHAR), '[0-9]+', 0), 6, '0') IN (
                      SELECT gvkey FROM actions_ids
                  )
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
                    stddev_pop(ret) OVER (
                        PARTITION BY gvkey
                        ORDER BY price_ts
                        RANGE BETWEEN INTERVAL 90 DAYS PRECEDING AND CURRENT ROW
                    ) * sqrt(12.0) AS sparse_volatility_90d,
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
                      WHEN p.ret_count_90d >= 2 THEN p.sparse_volatility_90d
                      ELSE NULL
                    END AS sparse_volatility_90d,
                    CASE
                      WHEN p.price_count_90d >= 20 AND p.max_price_90d > 0
                      THEN (p.min_price_90d / p.max_price_90d) - 1.0
                      ELSE NULL
                    END AS base_drawdown_90d,
                    CASE
                      WHEN p.price_count_90d >= 3 AND p.max_price_90d > 0
                      THEN (p.min_price_90d / p.max_price_90d) - 1.0
                      ELSE NULL
                    END AS sparse_drawdown_90d,
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
                    COALESCE(p.base_volatility_90d, p.sparse_volatility_90d) AS base_volatility_90d,
                    COALESCE(p.base_drawdown_90d, p.sparse_drawdown_90d) AS base_drawdown_90d,
                    CASE
                      WHEN p.price_60d IS NOT NULL AND p.price_60d != 0
                      THEN (p.price / p.price_60d) - 1.0
                      ELSE NULL
                    END AS base_momentum_60d
                FROM actions a
                ASOF LEFT JOIN price_with_momentum p
                  ON a.company_id_norm = p.gvkey
                 AND a.action_ts >= p.price_ts
            )
            SELECT *
            FROM price_asof
        """
        price_asof = con.execute(query, [str(raw_path)]).df()
    finally:
        con.close()

    if price_asof.empty:
        return out

    price_asof = price_asof.rename(
        columns={col: f"{col}__backfill" for col in _PRICE_BACKFILL_COLS}
    )
    out = out.merge(price_asof, how="left", left_index=True, right_on="__row_id")

    any_backfilled = pd.Series(False, index=out.index, dtype=bool)
    for col in _PRICE_BACKFILL_COLS:
        incoming_col = f"{col}__backfill"
        current = (
            pd.to_numeric(out[col], errors="coerce")
            if col in out.columns
            else pd.Series(float("nan"), index=out.index, dtype="float64")
        )
        incoming = pd.to_numeric(out.get(incoming_col), errors="coerce")
        fill_mask = current.isna() & incoming.notna()
        if col not in out.columns:
            out[col] = pd.NA
        out.loc[fill_mask, col] = incoming[fill_mask]
        any_backfilled = any_backfilled | fill_mask

    if bool(any_backfilled.any()):
        for col in _STALE_STATE_VECTOR_COLS:
            if col in out.columns:
                out.loc[any_backfilled, col] = pd.NA

    drop_cols = ["__row_id"] + [f"{col}__backfill" for col in _PRICE_BACKFILL_COLS]
    existing_drop_cols = [col for col in drop_cols if col in out.columns]
    if existing_drop_cols:
        out = out.drop(columns=existing_drop_cols)
    return out
