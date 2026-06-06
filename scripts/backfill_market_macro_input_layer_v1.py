#!/usr/bin/env python3
"""Backfill market and macro inputs into the v1 company input-layer artifact.

Preferred path:
- CRSP daily market cache for point-in-time-safe prices, returns, and shares

Fallback path:
- local month-end parquet, which remains a proxy-only source
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
import signal
from typing import Any, Dict, Iterable

import duckdb
import pandas as pd
from bs4 import BeautifulSoup

try:
    from repair_statement_debt_override_artifact import _fetch_sec_primary_document, _latest_sec_filing, _sec_session
except Exception:  # noqa: BLE001
    try:
        from scripts.repair_statement_debt_override_artifact import _fetch_sec_primary_document, _latest_sec_filing, _sec_session
    except Exception:  # noqa: BLE001
        _fetch_sec_primary_document = None
        _latest_sec_filing = None
        _sec_session = None


MAX_SEC_FACT_AGE_DAYS = 550
MAX_MONTHLY_GAP_DAYS = 45
MAX_DAILY_ANCHOR_GAP_DAYS = 7
MAX_ISSUER_SHARES_AGE_DAYS = 130
ISSUER_SHARES_OVERRIDE_MIN_RATIO = 1.05
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_COMPANYFACTS_ROOT = REPO_ROOT / "data" / "sec" / "companyfacts"
DEFAULT_LOCAL_CRSP_DAILY_ROOT = REPO_ROOT / "data" / "wrds" / "crsp"
SHARES_OUT_CONCEPTS = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
]
SHARE_CLASS_SEGMENT_RE = re.compile(
    r"(statementclassofstockaxis|classesofsharecapitalaxis|commonclass[a-z0-9]*member|class[a-z0-9]*member|ordinaryshareclass)",
    re.IGNORECASE,
)

MARKET_METRICS = {
    "market.price_spot": {"unit": "usd_per_share"},
    "market.total_return_1m_standardized": {"unit": "ratio", "months": 1},
    "market.total_return_3m_standardized": {"unit": "ratio", "months": 3},
    "market.total_return_6m_standardized": {"unit": "ratio", "months": 6},
    "market.total_return_12m_standardized": {"unit": "ratio", "months": 12},
}

MACRO_SERIES_SPECS = {
    "macro.fed_funds_effective": {"instrument_id": "DFF", "unit": "pct"},
    "macro.sofr": {"instrument_id": "SOFR", "unit": "pct"},
    "macro.ust_2y_yield": {"instrument_id": "DGS2", "unit": "pct"},
    "macro.ust_10y_yield": {"instrument_id": "DGS10", "unit": "pct"},
    "macro.ig_oas": {"instrument_id": "BAMLC0A0CM", "unit": "pct"},
    "macro.hy_oas": {"instrument_id": "BAMLH0A0HYM2", "unit": "pct"},
    "macro.unemployment_rate": {"instrument_id": "UNRATE", "unit": "pct"},
    "macro.wti_crude": {"instrument_id": "DCOILWTICO", "unit": "usd_bbl"},
}

MACRO_YOY_SPECS = {
    "macro.cpi_yoy": {"instrument_id": "CPIAUCSL", "unit": "ratio"},
    "macro.retail_sales_yoy": {"instrument_id": "RSAFS", "unit": "ratio"},
    "macro.real_gdp_growth_yoy": {"instrument_id": "GDPC1", "unit": "ratio", "lag_observations": 4},
}

MACRO_BASE_LOOKBACK_DAYS = 400
MACRO_LAGGED_SERIES_LOOKBACK_DAYS = max(
    MACRO_BASE_LOOKBACK_DAYS,
    max(int(spec.get("lag_observations") or 0) for spec in MACRO_YOY_SPECS.values()) * 120 + 180,
)

MACRO_METRIC_UNITS = {
    "macro.sofr_or_fed_funds": "pct",
    **{metric_name: spec["unit"] for metric_name, spec in MACRO_SERIES_SPECS.items()},
    "macro.curve_2s10s": "pct",
    **{metric_name: spec["unit"] for metric_name, spec in MACRO_YOY_SPECS.items()},
}


class _CompanyProcessingTimeout(RuntimeError):
    """Raised when one company exceeds the allowed processing timeout."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", required=True, help="Input company snapshot JSONL")
    parser.add_argument("--entity-identifier-path", required=True, help="Entity identifier parquet")
    parser.add_argument("--raw-timeseries-path", required=True, help="Local raw_timeseries parquet")
    parser.add_argument("--crsp-market-cache-path", help="Optional filtered CRSP daily market parquet cache")
    parser.add_argument(
        "--crsp-daily-root",
        help="Optional CRSP daily parquet folder. Defaults to the local canonical WRDS CRSP folder when present.",
    )
    parser.add_argument(
        "--allow-monthly-market-proxy",
        action="store_true",
        help="Allow the older monthly raw-timeseries proxy path when exact CRSP daily data is unavailable.",
    )
    parser.add_argument(
        "--companyfacts-root",
        help="Optional SEC companyfacts folder. Defaults to the local canonical companyfacts root when present.",
    )
    parser.add_argument(
        "--sec-filing-cache-root",
        default="/tmp/sec_filing_debt_cache",
        help="Optional SEC filing HTML cache for issuer-level shares fallback",
    )
    parser.add_argument(
        "--company-processing-timeout-seconds",
        type=float,
        default=30.0,
        help="Fail open on a single company if market-cap construction exceeds this timeout. Use 0 to disable.",
    )
    parser.add_argument(
        "--price-load-batch-size",
        type=int,
        default=32,
        help="Number of rows to batch together when loading CRSP/monthly price history. Smaller batches write earlier; larger batches reduce repeated parquet scans.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _feature_template(
    *,
    metric_name: str,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    provenance_artifact_type: str,
    primary_source_basis: str,
    support_mode: str,
    value: Any,
    unit: str,
    missing_reason: str | None,
    component_breakdown: Dict[str, Any] | None,
    quality_flags: list[str] | None,
) -> Dict[str, Any]:
    return {
        "name": metric_name,
        "value": value,
        "unit": unit,
        "computed_at": computed_at,
        "as_of_time": as_of_time,
        "window": None,
        "confidence": 1.0 if value is not None else None,
        "provenance": [
            {
                "artifact_type": provenance_artifact_type,
                "artifact_id": f"{primary_source_basis}:{Path(provenance_source).name}",
                "source": provenance_source,
                "published_at": as_of_time,
                "ingested_at": computed_at,
                "hash": None,
            }
        ],
        "missing_reason": missing_reason,
        "fallback_used": None,
        "metric_policy_id": None,
        "market_owner": None,
        "primary_source_basis": primary_source_basis,
        "methodology_registry_id": None,
        "methodology_metric_id": None,
        "canonical_owner_id": None,
        "canonical_owner_name": None,
        "canonical_classification": None,
        "market_layer_status": None,
        "current_alignment_status": None,
        "primary_source_document_id": None,
        "recommended_metric_name": None,
        "input_source_registry_id": None,
        "input_source_owner_id": None,
        "input_source_owner_name": None,
        "input_source_classification": primary_source_basis,
        "input_source_formula_basis": None,
        "input_source_alignment_status": "aligned",
        "input_source_document_ids": None,
        "definition_requirement": None,
        "definition_requirement_reason": None,
        "methodology_execution_decision": None,
        "methodology_execution_reason": None,
        "input_layer_bucket": "market_macro",
        "input_layer_bucket_reason": primary_source_basis,
        "strict_market_defined": None,
        "archetype": None,
        "sector": None,
        "subsector": None,
        "override_level_applied": None,
        "support_mode": support_mode,
        "applicability_status": None,
        "component_breakdown": component_breakdown,
        "quality_flags": quality_flags,
        "view_type": None,
    }


def iter_snapshot_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_row_batches(rows: Iterable[Dict[str, Any]], batch_size: int) -> Iterable[list[Dict[str, Any]]]:
    batch: list[Dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _permno_map(entity_identifier_path: Path) -> pd.DataFrame:
    ids = pd.read_parquet(entity_identifier_path)
    ids = ids[ids["identifier_type"].astype(str).str.lower() == "permno"].copy()
    ids["permno"] = ids["identifier_value"].astype(str).str.strip()
    return ids[["entity_id", "permno"]].drop_duplicates()


def _load_price_history(raw_timeseries_path: Path, permnos: list[str]) -> pd.DataFrame:
    permno_sql = ",".join(f"'{permno}'" for permno in sorted(set(permnos)))
    query = f"""
        SELECT
            CAST(entity_id AS VARCHAR) AS permno,
            CAST(trade_date AS DATE) AS trade_date,
            adjusted_close,
            close,
            volume,
            ret,
            retx
        FROM read_parquet('{raw_timeseries_path}')
        WHERE series_type = 'price'
          AND CAST(entity_id AS VARCHAR) IN ({permno_sql})
    """
    prices = duckdb.sql(query).fetchdf()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], utc=True).dt.normalize()
    prices = prices.sort_values(["permno", "trade_date"]).drop_duplicates(["permno", "trade_date"], keep="last")
    return prices


def _load_crsp_market_cache(crsp_market_cache_path: Path, permnos: list[str]) -> pd.DataFrame:
    permno_sql = ",".join(f"'{permno}'" for permno in sorted(set(permnos)))
    query = f"""
        SELECT
            CAST(permno AS VARCHAR) AS permno,
            CAST(trade_date AS DATE) AS trade_date,
            close_price,
            price_proxy,
            total_return,
            price_return,
            shares_outstanding,
            daily_cap,
            delist_flag
        FROM read_parquet('{crsp_market_cache_path}')
        WHERE CAST(permno AS VARCHAR) IN ({permno_sql})
    """
    prices = duckdb.sql(query).fetchdf()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], utc=True).dt.normalize()
    prices["date_key"] = prices["trade_date"]
    prices = prices.sort_values(["permno", "trade_date"]).drop_duplicates(["permno", "trade_date"], keep="last")
    return prices


def _load_price_history_for_row(
    *,
    permno: str | None,
    as_of_time: str,
    crsp_market_cache_path: Path | None,
    crsp_daily_root: Path | None,
    raw_timeseries_path: Path,
    allow_monthly_market_proxy: bool,
) -> pd.DataFrame:
    if not permno:
        return pd.DataFrame()
    as_of_date = pd.Timestamp(as_of_time).tz_convert("UTC").normalize()
    if crsp_market_cache_path is not None:
        return _load_crsp_market_cache(crsp_market_cache_path, [permno])
    if crsp_daily_root is not None:
        return _load_crsp_daily_from_repo(
            crsp_daily_root,
            [permno],
            min_asof_date=as_of_date,
            max_asof_date=as_of_date,
        )
    if allow_monthly_market_proxy:
        return _load_price_history(raw_timeseries_path, [permno])
    return pd.DataFrame()


def _load_price_history_for_batch(
    *,
    permnos: list[str],
    as_of_times: list[str],
    crsp_market_cache_path: Path | None,
    crsp_daily_root: Path | None,
    raw_timeseries_path: Path,
    allow_monthly_market_proxy: bool,
) -> Dict[str, pd.DataFrame]:
    permnos = sorted({str(permno).strip() for permno in permnos if permno})
    if not permnos:
        return {}

    if crsp_market_cache_path is not None:
        prices = _load_crsp_market_cache(crsp_market_cache_path, permnos)
    elif crsp_daily_root is not None:
        as_of_dates = [pd.Timestamp(as_of_time).tz_convert("UTC").normalize() for as_of_time in as_of_times]
        prices = _load_crsp_daily_from_repo(
            crsp_daily_root,
            permnos,
            min_asof_date=min(as_of_dates),
            max_asof_date=max(as_of_dates),
        )
    elif allow_monthly_market_proxy:
        prices = _load_price_history(raw_timeseries_path, permnos)
    else:
        prices = pd.DataFrame()

    if prices.empty:
        return {}
    return {
        str(permno): frame.reset_index(drop=True)
        for permno, frame in prices.groupby("permno", sort=False)
    }


def _load_crsp_daily_from_repo(
    crsp_daily_root: Path,
    permnos: list[str],
    *,
    min_asof_date: pd.Timestamp,
    max_asof_date: pd.Timestamp,
) -> pd.DataFrame:
    if not permnos:
        return pd.DataFrame()
    start_year = int(min_asof_date.year) - 1
    end_year = int(max_asof_date.year)
    files: list[Path] = []
    for year in range(start_year, end_year + 1):
        candidate = crsp_daily_root / f"dsf_{year:04d}-01-01_to_{year:04d}-12-31.parquet"
        if candidate.exists():
            files.append(candidate)
    if not files:
        return pd.DataFrame()

    permno_sql = ",".join(f"'{permno}'" for permno in sorted(set(permnos)))
    min_trade_date = (min_asof_date - pd.Timedelta(days=370)).date().isoformat()
    max_trade_date = max_asof_date.date().isoformat()
    selects = []
    for file_path in files:
        selects.append(
            f"""
            SELECT
                CAST(permno AS VARCHAR) AS permno,
                CAST(date AS DATE) AS trade_date,
                ABS(prc) AS close_price,
                ABS(prc) AS price_proxy,
                ret AS total_return,
                retx AS price_return,
                shrout AS shares_outstanding,
                ABS(prc) * shrout AS daily_cap,
                FALSE AS delist_flag
            FROM read_parquet('{file_path.as_posix()}')
            WHERE CAST(permno AS VARCHAR) IN ({permno_sql})
              AND CAST(date AS DATE) >= DATE '{min_trade_date}'
              AND CAST(date AS DATE) <= DATE '{max_trade_date}'
            """
        )
    query = " UNION ALL ".join(selects)
    prices = duckdb.sql(query).fetchdf()
    if prices.empty:
        return prices
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], utc=True).dt.normalize()
    prices["date_key"] = prices["trade_date"]
    prices = prices.sort_values(["permno", "trade_date"]).drop_duplicates(["permno", "trade_date"], keep="last")
    return prices


def _load_macro_history(
    raw_timeseries_path: Path,
    *,
    min_asof_date: pd.Timestamp | None = None,
    max_asof_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    wanted_ids = {
        "SOFR",
        "DFF",
        "DGS2",
        "DGS10",
        "BAMLC0A0CM",
        "BAMLH0A0HYM2",
        "CPIAUCSL",
        "UNRATE",
        "RSAFS",
        "DCOILWTICO",
        "GDPC1",
    }
    wanted_sql = ",".join(f"'{instrument_id}'" for instrument_id in sorted(wanted_ids))
    date_filters = ""
    if min_asof_date is not None and max_asof_date is not None:
        # Monthly YoY metrics only need about a year of history, but quarterly GDP growth
        # is keyed off release observations and needs a wider lookback so the prior-year
        # release still survives the as-of-safe filter.
        min_event_date = (min_asof_date - pd.Timedelta(days=MACRO_LAGGED_SERIES_LOOKBACK_DAYS)).date().isoformat()
        max_event_date = max_asof_date.date().isoformat()
        date_filters = (
            f"\n          AND CAST(event_time AS DATE) >= DATE '{min_event_date}'"
            f"\n          AND CAST(event_time AS DATE) <= DATE '{max_event_date}'"
        )
    query = f"""
        SELECT
            instrument_id,
            CAST(event_time AS DATE) AS event_date,
            value,
            units
        FROM read_parquet('{raw_timeseries_path}')
        WHERE series_type = 'macro'
          AND instrument_id IN ({wanted_sql})
          {date_filters}
    """
    macro = duckdb.sql(query).fetchdf()
    macro["event_date"] = pd.to_datetime(macro["event_date"], utc=True).dt.normalize()
    macro = macro.sort_values(["instrument_id", "event_date"]).drop_duplicates(["instrument_id", "event_date"], keep="last")
    return macro


def _latest_value_on_or_before(df: pd.DataFrame, date_key: pd.Timestamp, value_col: str) -> float | None:
    eligible = df[df["date_key"] <= date_key]
    if eligible.empty:
        return None
    value = eligible.iloc[-1][value_col]
    return None if pd.isna(value) else float(value)


def _latest_row_on_or_before(df: pd.DataFrame, date_key: pd.Timestamp) -> pd.Series | None:
    eligible = df[df["date_key"] <= date_key]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _has_dense_monthly_coverage(window: pd.DataFrame, months: int) -> tuple[bool, Dict[str, Any]]:
    if window.empty:
        return False, {"periods_used": 0}
    unique_dates = sorted(pd.to_datetime(window["date_key"], utc=True).dt.normalize().unique())
    gaps = []
    for idx in range(1, len(unique_dates)):
        gaps.append(int((unique_dates[idx] - unique_dates[idx - 1]).days))
    diagnostics = {
        "periods_used": int(len(unique_dates)),
        "max_gap_days": max(gaps) if gaps else 0,
        "required_months": months,
    }
    if len(unique_dates) < months:
        return False, diagnostics
    if gaps and max(gaps) > MAX_MONTHLY_GAP_DAYS:
        return False, diagnostics
    return True, diagnostics


def _compound_trailing_return(price_history: pd.DataFrame, as_of_date: pd.Timestamp, months: int) -> tuple[float | None, Dict[str, Any], str | None]:
    current_row = _latest_row_on_or_before(price_history, as_of_date)
    if current_row is None:
        return None, {"lookback_months": months}, "market_timeseries_unavailable"

    current_trade_date = current_row["date_key"]
    target_date = current_trade_date - pd.DateOffset(months=months)
    window = price_history[(price_history["date_key"] > target_date) & (price_history["date_key"] <= current_trade_date)].copy()
    if window.empty:
        return None, {
            "lookback_months": months,
            "current_trade_date": str(current_trade_date.date()),
            "target_trade_date": str(target_date.date()),
        }, "market_timeseries_unavailable"

    dense_enough, coverage_meta = _has_dense_monthly_coverage(window, months)
    if not dense_enough:
        return None, {
            "lookback_months": months,
            "current_trade_date": str(current_trade_date.date()),
            "target_trade_date": str(target_date.date()),
            **coverage_meta,
        }, "market_timeseries_sparse"

    return_col = None
    # This metric is meant to reflect market total return, so prefer `ret`
    # (which includes distributions) over `retx` (which excludes them).
    if window["ret"].notna().all():
        return_col = "ret"
    elif window["retx"].notna().all():
        return_col = "retx"
    else:
        return None, {
            "lookback_months": months,
            "current_trade_date": str(current_trade_date.date()),
            "target_trade_date": str(target_date.date()),
            **coverage_meta,
        }, "market_return_series_unavailable"

    compounded = float((1.0 + window[return_col].astype(float)).prod() - 1.0)
    components = {
        "lookback_months": months,
        "current_trade_date": str(current_trade_date.date()),
        "target_trade_date": str(target_date.date()),
        **coverage_meta,
        "return_column": return_col,
        "formula": f"compound_{return_col}_over_dense_monthly_window",
    }
    return compounded, components, None


def _recent_enough_trade_date(observed_date: pd.Timestamp, anchor_date: pd.Timestamp) -> bool:
    return 0 <= int((anchor_date - observed_date).days) <= MAX_DAILY_ANCHOR_GAP_DAYS


def _compound_trailing_crsp_return(
    price_history: pd.DataFrame,
    as_of_date: pd.Timestamp,
    months: int,
) -> tuple[float | None, Dict[str, Any], str | None, str]:
    current_row = _latest_row_on_or_before(price_history, as_of_date)
    if current_row is None:
        return None, {"lookback_months": months}, "market_timeseries_unavailable", "unsupported"

    current_trade_date = current_row["trade_date"]
    if not _recent_enough_trade_date(current_trade_date, as_of_date):
        return None, {
            "lookback_months": months,
            "current_trade_date": str(current_trade_date.date()),
            "formula": "compound_daily_total_return",
        }, "market_timeseries_stale", "unsupported"

    target_date = current_trade_date - pd.DateOffset(months=months)
    start_row = _latest_row_on_or_before(price_history, target_date)
    if start_row is None:
        return None, {
            "lookback_months": months,
            "current_trade_date": str(current_trade_date.date()),
            "target_trade_date": str(target_date.date()),
            "formula": "compound_daily_total_return",
        }, "market_timeseries_unavailable", "unsupported"

    start_trade_date = start_row["trade_date"]
    if not _recent_enough_trade_date(start_trade_date, target_date.normalize()):
        return None, {
            "lookback_months": months,
            "current_trade_date": str(current_trade_date.date()),
            "target_trade_date": str(target_date.date()),
            "anchor_trade_date": str(start_trade_date.date()),
            "formula": "compound_daily_total_return",
        }, "market_timeseries_sparse", "unsupported"

    window = price_history[
        (price_history["trade_date"] > start_trade_date) & (price_history["trade_date"] <= current_trade_date)
    ].copy()
    if window.empty:
        return None, {
            "lookback_months": months,
            "current_trade_date": str(current_trade_date.date()),
            "target_trade_date": str(target_date.date()),
            "anchor_trade_date": str(start_trade_date.date()),
            "formula": "compound_daily_total_return",
        }, "market_timeseries_unavailable", "unsupported"

    if window["total_return"].notna().all():
        return_col = "total_return"
        support_mode = "exact"
    elif window["price_return"].notna().all():
        return_col = "price_return"
        support_mode = "proxy_missing_component"
    else:
        return None, {
            "lookback_months": months,
            "current_trade_date": str(current_trade_date.date()),
            "target_trade_date": str(target_date.date()),
            "anchor_trade_date": str(start_trade_date.date()),
            "formula": "compound_daily_total_return",
        }, "market_return_series_unavailable", "unsupported"

    compounded = float((1.0 + window[return_col].astype(float)).prod() - 1.0)
    components = {
        "lookback_months": months,
        "current_trade_date": str(current_trade_date.date()),
        "target_trade_date": str(target_date.date()),
        "anchor_trade_date": str(start_trade_date.date()),
        "rows_used": int(len(window)),
        "return_column": return_col,
        "formula": f"compound_{return_col}_from_crsp_daily_window",
    }
    missing_reason = None if support_mode == "exact" else "total_return_component_unavailable"
    return compounded, components, missing_reason, support_mode


def _price_feature(
    *,
    metric_name: str,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    unit: str,
    value: float | None,
    components: Dict[str, Any],
    missing_reason: str | None,
    support_mode: str,
    quality_flags: list[str] | None = None,
) -> Dict[str, Any]:
    return _feature_template(
        metric_name=metric_name,
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        provenance_artifact_type="MarketTimeseries",
        primary_source_basis="market_timeseries",
        support_mode=support_mode,
        value=value,
        unit=unit,
        missing_reason=missing_reason,
        component_breakdown=components,
        quality_flags=quality_flags if value is not None else [missing_reason or "market_timeseries_unavailable"],
    )


def _build_price_metrics_from_crsp(
    permno: str | None,
    price_history: pd.DataFrame | None,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
) -> Dict[str, Dict[str, Any]]:
    as_of_date = pd.Timestamp(as_of_time).tz_convert("UTC").normalize()
    metrics: Dict[str, Dict[str, Any]] = {}

    if permno is None or price_history is None or price_history.empty:
        for metric_name, spec in MARKET_METRICS.items():
            metrics[metric_name] = _price_feature(
                metric_name=metric_name,
                as_of_time=as_of_time,
                computed_at=computed_at,
                provenance_source=provenance_source,
                unit=spec["unit"],
                value=None,
                components={"permno": permno},
                missing_reason="market_timeseries_unavailable",
                support_mode="unsupported",
            )
        return metrics

    current_row = _latest_row_on_or_before(price_history, as_of_date)
    current_trade_date = None if current_row is None else current_row["trade_date"]
    close_price = None if current_row is None or pd.isna(current_row["close_price"]) else float(current_row["close_price"])
    price_proxy = None if current_row is None or pd.isna(current_row["price_proxy"]) else float(current_row["price_proxy"])
    if close_price is not None:
        current_price = close_price
        price_support_mode = "exact"
        quality_flags = None
        missing_reason = None
        formula = "latest_crsp_close_on_or_before_asof"
    elif price_proxy is not None:
        current_price = price_proxy
        price_support_mode = "proxy_missing_component"
        quality_flags = ["used_abs_dlyprc_proxy"]
        missing_reason = "close_component_unavailable"
        formula = "latest_abs_crsp_price_on_or_before_asof"
    else:
        current_price = None
        price_support_mode = "unsupported"
        quality_flags = ["market_price_unavailable"]
        missing_reason = "market_price_unavailable"
        formula = "latest_crsp_price_on_or_before_asof"

    if current_trade_date is not None and current_price is not None and not _recent_enough_trade_date(current_trade_date, as_of_date):
        current_price = None
        price_support_mode = "unsupported"
        quality_flags = ["market_timeseries_stale"]
        missing_reason = "market_timeseries_stale"

    metrics["market.price_spot"] = _price_feature(
        metric_name="market.price_spot",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="usd_per_share",
        value=current_price,
        components={
            "permno": permno,
            "current_trade_date": None if current_trade_date is None else str(current_trade_date.date()),
            "formula": formula,
        },
        missing_reason=missing_reason,
        support_mode=price_support_mode,
        quality_flags=quality_flags,
    )

    for metric_name, spec in MARKET_METRICS.items():
        if metric_name == "market.price_spot":
            continue
        value, return_components, missing_reason, support_mode = _compound_trailing_crsp_return(
            price_history,
            as_of_date,
            spec["months"],
        )
        metrics[metric_name] = _price_feature(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            unit=spec["unit"],
            value=value,
            components={
                "permno": permno,
                "current_close": current_price,
                **return_components,
            },
            missing_reason=missing_reason,
            support_mode=support_mode,
            quality_flags=None if support_mode == "exact" else ([missing_reason] if value is None else ["price_return_only"]),
        )

    return metrics


def _macro_feature(
    *,
    metric_name: str,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    unit: str,
    value: float | None,
    components: Dict[str, Any],
    missing_reason: str | None,
) -> Dict[str, Any]:
    return _feature_template(
        metric_name=metric_name,
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        provenance_artifact_type="MacroTimeseries",
        primary_source_basis="macro_timeseries",
        support_mode="exact" if value is not None else "unsupported",
        value=value,
        unit=unit,
        missing_reason=missing_reason,
        component_breakdown=components,
        quality_flags=None if value is not None else [missing_reason or "macro_series_unavailable"],
    )


@contextmanager
def _company_processing_guard(timeout_seconds: float | None):
    if (
        timeout_seconds is None
        or timeout_seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(signum, frame):  # noqa: ARG001
        raise _CompanyProcessingTimeout(f"company_processing_timeout_after_{timeout_seconds:g}s")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_fail_open_market_metrics(
    *,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    error_type: str,
    error_message: str,
) -> Dict[str, Dict[str, Any]]:
    error_text = str(error_message).strip()[:240]
    missing_reason = "company_processing_timeout" if error_type == "company_processing_timeout" else "company_processing_failed"
    breakdown = {
        "error_type": error_type,
        "error_message": error_text,
    }
    quality_flags = ["company_processing_fail_open", error_type]
    metrics: Dict[str, Dict[str, Any]] = {}
    for metric_name, spec in MARKET_METRICS.items():
        metrics[metric_name] = _price_feature(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            unit=spec["unit"],
            value=None,
            components=breakdown,
            missing_reason=missing_reason,
            support_mode="unsupported",
            quality_flags=quality_flags,
        )
    metrics["market.market_cap_provider_direct"] = _feature_template(
        metric_name="market.market_cap_provider_direct",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        provenance_artifact_type="DerivedComputation",
        primary_source_basis="market_processing_fail_open",
        support_mode="unsupported",
        value=None,
        unit="usd",
        missing_reason=missing_reason,
        component_breakdown=breakdown,
        quality_flags=quality_flags,
    )
    return metrics


def _build_fail_open_market_cap_metric(
    *,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    error_type: str,
    error_message: str,
) -> Dict[str, Any]:
    error_text = str(error_message).strip()[:240]
    missing_reason = "company_processing_timeout" if error_type == "company_processing_timeout" else "company_processing_failed"
    breakdown = {
        "error_type": error_type,
        "error_message": error_text,
    }
    quality_flags = ["market_cap_processing_fail_open", error_type]
    return _feature_template(
        metric_name="market.market_cap_provider_direct",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        provenance_artifact_type="DerivedComputation",
        primary_source_basis="market_cap_processing_fail_open",
        support_mode="unsupported",
        value=None,
        unit="usd",
        missing_reason=missing_reason,
        component_breakdown=breakdown,
        quality_flags=quality_flags,
    )


def _build_fail_open_macro_metrics(
    *,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    error_type: str,
    error_message: str,
) -> Dict[str, Dict[str, Any]]:
    error_text = str(error_message).strip()[:240]
    missing_reason = "macro_build_failed"
    breakdown = {
        "error_type": error_type,
        "error_message": error_text,
    }
    return {
        metric_name: _macro_feature(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            unit=unit,
            value=None,
            components=breakdown,
            missing_reason=missing_reason,
        )
        for metric_name, unit in MACRO_METRIC_UNITS.items()
    }


def _parse_iso_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(str(text)[:10])
    except ValueError:
        return None


def _load_companyfacts(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _candidate_units_map(companyfacts: dict, concept_name: str, taxonomy: str | None = None) -> dict | None:
    taxonomies = [taxonomy] if taxonomy else ["dei", "us-gaap", "ifrs-full"]
    for current_taxonomy in taxonomies:
        facts = (companyfacts.get("facts") or {}).get(current_taxonomy) or {}
        if concept_name in facts:
            return facts[concept_name].get("units") or {}
    return None


def _latest_shares_outstanding(companyfacts: dict, as_of_date: str) -> tuple[float | None, dict[str, Any] | None]:
    as_of_dt = date.fromisoformat(as_of_date)
    for taxonomy, concept_name in SHARES_OUT_CONCEPTS:
        units_map = _candidate_units_map(companyfacts, concept_name, taxonomy)
        if not units_map:
            continue
        candidates = []
        for unit, entries in units_map.items():
            if unit.lower() != "shares":
                continue
            for entry in entries:
                end_dt = _parse_iso_date(entry.get("end"))
                filed_dt = _parse_iso_date(entry.get("filed"))
                value = entry.get("val")
                if end_dt is None or value is None or end_dt > as_of_dt:
                    continue
                if filed_dt is not None and filed_dt > as_of_dt:
                    continue
                if (as_of_dt - end_dt).days > MAX_SEC_FACT_AGE_DAYS:
                    continue
                candidates.append((end_dt, filed_dt or end_dt, entry, taxonomy, concept_name))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        end_dt, filed_dt, chosen, chosen_taxonomy, chosen_concept = candidates[-1]
        return float(chosen["val"]), {
            "concept": chosen_concept,
            "taxonomy": chosen_taxonomy,
            "end": end_dt.isoformat(),
            "filed": filed_dt.isoformat(),
            "fy": chosen.get("fy"),
            "fp": chosen.get("fp"),
            "frame": chosen.get("frame"),
            "form": chosen.get("form"),
            "unit": "shares",
            "formula": "latest_shares_outstanding_on_or_before_asof",
        }
    return None, None


def _shares_support_mode(*, reference_date: date | None, as_of_date: date) -> tuple[str, str | None]:
    if reference_date is None:
        return "unsupported", "shares_outstanding_unavailable"
    age_days = (as_of_date - reference_date).days
    if age_days < 0:
        return "unsupported", "shares_outstanding_unavailable"
    if age_days <= MAX_ISSUER_SHARES_AGE_DAYS:
        return "exact", None
    return "proxy_missing_component", "issuer_shares_stale"


def _parse_xbrl_numeric(tag: Any) -> float | None:
    text = " ".join(tag.stripped_strings)
    if not text:
        return None
    cleaned = text.replace(",", "").replace("$", "").replace("(", "").replace(")", "").strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    scale_text = tag.get("scale")
    if scale_text not in (None, ""):
        try:
            value *= 10 ** int(scale_text)
        except ValueError:
            return None
    if tag.get("sign") == "-" or ("(" in text and ")" in text):
        value *= -1.0
    return value


def _context_date_and_segment(context_tag: Any) -> tuple[date | None, str]:
    instant = context_tag.find(lambda t: t.name and t.name.lower().endswith("instant"))
    end_date = context_tag.find(lambda t: t.name and t.name.lower().endswith("enddate"))
    context_date = _parse_iso_date(instant.get_text(" ", strip=True) if instant else end_date.get_text(" ", strip=True) if end_date else None)
    segment = context_tag.find(lambda t: t.name and t.name.lower().endswith("segment"))
    segment_text = " ".join(segment.stripped_strings) if segment else ""
    return context_date, segment_text


def _extract_issuer_shares_from_filing(
    *,
    filing: dict[str, Any],
    html: str,
    as_of_date: date,
) -> tuple[float | None, dict[str, Any] | None]:
    soup = BeautifulSoup(html, "html.parser")
    contexts: dict[str, tuple[date | None, str]] = {}
    for context_tag in soup.find_all(lambda t: t.name and t.name.lower().endswith("context")):
        context_id = context_tag.get("id")
        if not context_id:
            continue
        contexts[str(context_id)] = _context_date_and_segment(context_tag)

    share_facts: dict[tuple[str, str], dict[str, Any]] = {}
    for tag in soup.find_all(lambda t: t.name and t.name.lower().endswith("nonfraction")):
        fact_name = str(tag.get("name") or "")
        if not any(fact_name.endswith(concept_name) for _, concept_name in SHARES_OUT_CONCEPTS):
            continue
        context_ref = str(tag.get("contextref") or "")
        if not context_ref or context_ref not in contexts:
            continue
        value = _parse_xbrl_numeric(tag)
        if value is None or value <= 0:
            continue
        context_date, segment_text = contexts[context_ref]
        if context_date is None or context_date > as_of_date:
            continue
        share_facts[(context_ref, fact_name)] = {
            "value": float(value),
            "context_ref": context_ref,
            "date": context_date,
            "segment_text": segment_text,
            "fact_name": fact_name,
        }

    if not share_facts:
        return None, None

    latest_date = max(item["date"] for item in share_facts.values())
    latest_facts = [item for item in share_facts.values() if item["date"] == latest_date]

    class_facts: dict[str, dict[str, Any]] = {}
    aggregate_facts: dict[str, dict[str, Any]] = {}
    for item in latest_facts:
        segment_key = " ".join(str(item["segment_text"]).split())
        if segment_key and SHARE_CLASS_SEGMENT_RE.search(segment_key):
            class_facts[segment_key] = item
        else:
            aggregate_facts[item["context_ref"]] = item

    aggregate_value = max((item["value"] for item in aggregate_facts.values()), default=None)
    class_value = None
    if class_facts:
        class_value = float(sum(item["value"] for item in class_facts.values()))

    selected_value = None
    selected_mode = None
    if class_value is not None and aggregate_value is not None:
        if abs(class_value - aggregate_value) / max(class_value, aggregate_value) <= 0.02:
            selected_value = max(class_value, aggregate_value)
            selected_mode = "class_sum_confirmed_by_aggregate"
        elif class_value > aggregate_value * 1.02:
            selected_value = class_value
            selected_mode = "share_class_sum"
        else:
            selected_value = aggregate_value
            selected_mode = "aggregate_total"
    elif class_value is not None:
        selected_value = class_value
        selected_mode = "share_class_sum" if len(class_facts) > 1 else "single_share_class_context"
    elif aggregate_value is not None:
        selected_value = aggregate_value
        selected_mode = "aggregate_total"

    if selected_value is None:
        return None, None

    support_mode, missing_reason = _shares_support_mode(reference_date=latest_date, as_of_date=as_of_date)
    return float(selected_value), {
        "selected_filing": filing,
        "share_reference_date": latest_date.isoformat(),
        "selected_mode": selected_mode,
        "share_class_count": int(len(class_facts)),
        "aggregate_context_count": int(len(aggregate_facts)),
        "share_class_contexts": [
            {
                "context_ref": item["context_ref"],
                "segment_text": item["segment_text"],
                "value": item["value"],
            }
            for item in class_facts.values()
        ],
        "aggregate_contexts": [
            {
                "context_ref": item["context_ref"],
                "value": item["value"],
            }
            for item in aggregate_facts.values()
        ],
        "support_mode": support_mode,
        "missing_reason": missing_reason,
        "formula": "sum_latest_share_class_contexts_or_use_latest_aggregate_total",
    }


def _latest_issuer_shares_from_sec_filing(
    *,
    cik: str,
    as_of_date: date,
    session: Any,
    cache_dir: Path | None,
) -> tuple[float | None, dict[str, Any] | None]:
    if cache_dir is not None and cache_dir.exists():
        best_cached: tuple[date, float, dict[str, Any]] | None = None
        for cached_html_path in sorted(cache_dir.glob(f"{cik}_*.htm*")):
            try:
                html = cached_html_path.read_text(errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            filing = {
                "cik": cik,
                "filing_date": None,
                "form": None,
                "accession_number": cached_html_path.name.split("_", 2)[1] if "_" in cached_html_path.name else None,
                "primary_document": cached_html_path.name,
                "cache_path": str(cached_html_path),
            }
            shares_out, shares_meta = _extract_issuer_shares_from_filing(filing=filing, html=html, as_of_date=as_of_date)
            if shares_out is None or shares_meta is None:
                continue
            reference_date = _parse_iso_date(shares_meta.get("share_reference_date"))
            if reference_date is None or reference_date > as_of_date:
                continue
            if best_cached is None or reference_date > best_cached[0] or (
                reference_date == best_cached[0] and float(shares_out) > best_cached[1]
            ):
                best_cached = (reference_date, float(shares_out), shares_meta)
        if best_cached is not None:
            return best_cached[1], best_cached[2]
    if _latest_sec_filing is None or _fetch_sec_primary_document is None or session is None:
        return None, None
    filing = _latest_sec_filing(cik=cik, as_of_date=as_of_date, session=session, cache_dir=cache_dir)
    if filing is None:
        return None, None
    html = _fetch_sec_primary_document(filing, session=session, cache_dir=cache_dir)
    if not html:
        return None, None
    return _extract_issuer_shares_from_filing(filing=filing, html=html, as_of_date=as_of_date)


def _build_price_metrics(
    permno: str | None,
    price_history: pd.DataFrame | None,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
) -> Dict[str, Dict[str, Any]]:
    as_of_date = pd.Timestamp(as_of_time).tz_convert("UTC").normalize()
    metrics: Dict[str, Dict[str, Any]] = {}

    if permno is None or price_history is None or price_history.empty:
        for metric_name, spec in MARKET_METRICS.items():
            metrics[metric_name] = _price_feature(
                metric_name=metric_name,
                as_of_time=as_of_time,
                computed_at=computed_at,
                provenance_source=provenance_source,
                unit=spec["unit"],
                value=None,
                components={"permno": permno},
                missing_reason="market_timeseries_unavailable",
                support_mode="unsupported",
            )
        return metrics

    price_history = price_history.copy()
    price_history["date_key"] = price_history["trade_date"]
    current_row = _latest_row_on_or_before(price_history, as_of_date)
    current_price = None if current_row is None or pd.isna(current_row["close"]) else float(current_row["close"])
    current_trade_date = None if current_row is None else str(current_row["date_key"].date())

    metrics["market.price_spot"] = _price_feature(
        metric_name="market.price_spot",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="usd_per_share",
        value=current_price,
        components={
            "permno": permno,
            "current_trade_date": current_trade_date,
            "formula": "latest_close_on_or_before_asof",
        },
        missing_reason="market_price_unavailable" if current_price is None else None,
        support_mode="proxy_missing_component" if current_price is not None else "unsupported",
        quality_flags=["monthly_timeseries_price_proxy"] if current_price is not None else None,
    )

    for metric_name, spec in MARKET_METRICS.items():
        if metric_name == "market.price_spot":
            continue
        value, return_components, missing_reason = _compound_trailing_return(
            price_history,
            as_of_date,
            spec["months"],
        )
        metrics[metric_name] = _price_feature(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            unit=spec["unit"],
            value=value,
            components={
                "permno": permno,
                "current_close": current_price,
                **return_components,
            },
            missing_reason=missing_reason,
            support_mode="exact" if value is not None else "unsupported",
        )

    return metrics


def _build_market_cap_metric(
    *,
    price_history: pd.DataFrame | None,
    price_node: Dict[str, Any],
    issuer_shares_outstanding: float | None,
    issuer_shares_meta: dict[str, Any] | None,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
) -> Dict[str, Any] | None:
    as_of_date = pd.Timestamp(as_of_time).tz_convert("UTC").normalize()
    if price_history is None or price_history.empty:
        return None
    current_row = _latest_row_on_or_before(price_history, as_of_date)
    if current_row is None:
        return None

    close_price = None if pd.isna(current_row["close_price"]) else float(current_row["close_price"])
    price_proxy = None if pd.isna(current_row["price_proxy"]) else float(current_row["price_proxy"])
    crsp_shares = None if pd.isna(current_row["shares_outstanding"]) else float(current_row["shares_outstanding"]) * 1000.0
    share_reference_support = (issuer_shares_meta or {}).get("support_mode") or "unsupported"
    share_reference_missing = (issuer_shares_meta or {}).get("missing_reason")
    share_reference_age = (issuer_shares_meta or {}).get("share_reference_date")

    market_cap = None
    formula = None
    quality_flags: list[str] | None = None
    missing_reason = None
    support_mode = "unsupported"

    issuer_override = (
        issuer_shares_outstanding is not None
        and issuer_shares_outstanding > 0
        and crsp_shares is not None
        and issuer_shares_outstanding > crsp_shares * ISSUER_SHARES_OVERRIDE_MIN_RATIO
    )

    if issuer_shares_outstanding is not None and close_price is not None and (issuer_override or crsp_shares is None):
        market_cap = close_price * float(issuer_shares_outstanding)
        formula = "close_price * issuer_level_shares_outstanding"
        support_mode = "exact" if share_reference_support == "exact" else "proxy_missing_component"
        missing_reason = share_reference_missing
        quality_flags = ["issuer_level_shares_override"] if issuer_override else ["issuer_level_shares_fallback"]
    elif issuer_shares_outstanding is not None and price_proxy is not None and (issuer_override or crsp_shares is None):
        market_cap = price_proxy * float(issuer_shares_outstanding)
        formula = "price_proxy * issuer_level_shares_outstanding"
        support_mode = "proxy_missing_component"
        missing_reason = share_reference_missing or "close_component_unavailable"
        quality_flags = ["issuer_level_shares_override", "used_abs_dlyprc_proxy"] if issuer_override else ["issuer_level_shares_fallback", "used_abs_dlyprc_proxy"]
    elif close_price is not None and pd.notna(current_row["shares_outstanding"]):
        market_cap = close_price * float(current_row["shares_outstanding"]) * 1000.0
        formula = "close_price * shares_outstanding_thousands * 1000"
        support_mode = "exact"
    elif price_proxy is not None and pd.notna(current_row["shares_outstanding"]):
        market_cap = price_proxy * float(current_row["shares_outstanding"]) * 1000.0
        formula = "price_proxy * shares_outstanding_thousands * 1000"
        support_mode = "proxy_missing_component"
        missing_reason = "close_component_unavailable"
        quality_flags = ["used_abs_dlyprc_proxy"]
    elif pd.notna(current_row["daily_cap"]):
        market_cap = float(current_row["daily_cap"]) * 1000.0
        formula = "crsp_daily_cap_thousands * 1000"
        support_mode = "exact"
    else:
        return None

    current_trade_date = current_row["trade_date"]
    if not _recent_enough_trade_date(current_trade_date, as_of_date):
        return _feature_template(
            metric_name="market.market_cap_provider_direct",
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            provenance_artifact_type="CRSPDailyStockFile",
            primary_source_basis="crsp_daily_stock_file",
            support_mode="unsupported",
            value=None,
            unit="usd",
            missing_reason="market_timeseries_stale",
            component_breakdown={
                "current_trade_date": str(current_trade_date.date()),
                "formula": formula,
            },
            quality_flags=["market_timeseries_stale"],
        )

    return _feature_template(
        metric_name="market.market_cap_provider_direct",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        provenance_artifact_type="CRSPDailyStockFile",
        primary_source_basis="crsp_daily_stock_file",
        support_mode=support_mode,
        value=market_cap,
        unit="usd",
        missing_reason=missing_reason,
        component_breakdown={
            "current_trade_date": str(current_trade_date.date()),
            "close_price": close_price,
            "price_proxy": price_proxy,
            "shares_outstanding": None if pd.isna(current_row["shares_outstanding"]) else float(current_row["shares_outstanding"]),
            "issuer_shares_outstanding": issuer_shares_outstanding,
            "issuer_shares_reference_date": share_reference_age,
            "issuer_shares_reference": issuer_shares_meta,
            "daily_cap": None if pd.isna(current_row["daily_cap"]) else float(current_row["daily_cap"]),
            "formula": formula,
        },
        quality_flags=quality_flags,
    )


def _build_market_cap_metric_from_companyfacts(
    *,
    companyfacts: dict | None,
    price_node: Dict[str, Any],
    as_of_time: str,
    computed_at: str,
    companyfacts_path: Path | None,
) -> Dict[str, Any]:
    price = price_node.get("value")
    price_support = price_node.get("support_mode") or "missing_metric"
    if companyfacts is None or companyfacts_path is None:
        return _feature_template(
            metric_name="market.market_cap_provider_direct",
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=str(companyfacts_path or "companyfacts_unavailable"),
            provenance_artifact_type="SecCompanyFacts",
            primary_source_basis="sec_companyfacts",
            support_mode="unsupported",
            value=None,
            unit="usd",
            missing_reason="companyfacts_unavailable",
            component_breakdown=None,
            quality_flags=["companyfacts_unavailable"],
        )
    shares_out, shares_meta = _latest_shares_outstanding(companyfacts, as_of_time[:10])
    if price is None or shares_out is None:
        return _feature_template(
            metric_name="market.market_cap_provider_direct",
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=str(companyfacts_path),
            provenance_artifact_type="SecCompanyFacts",
            primary_source_basis="sec_companyfacts",
            support_mode="unsupported",
            value=None,
            unit="usd",
            missing_reason="component_unavailable",
            component_breakdown={
                "price_spot": price,
                "shares_outstanding": shares_meta,
                "formula": "price_spot * shares_outstanding",
            },
            quality_flags=["component_unavailable"],
        )
    support_mode = "exact" if price_support == "exact" else "proxy_missing_component"
    quality_flags = None if support_mode == "exact" else ["price_component_not_exact"]
    return _feature_template(
        metric_name="market.market_cap_provider_direct",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=str(companyfacts_path),
        provenance_artifact_type="SecCompanyFacts",
        primary_source_basis="sec_companyfacts",
        support_mode=support_mode,
        value=float(price) * float(shares_out),
        unit="usd",
        missing_reason=None if support_mode == "exact" else "price_component_not_exact",
        component_breakdown={
            "price_spot": price,
            "shares_outstanding": shares_meta,
            "formula": "price_spot * shares_outstanding",
        },
        quality_flags=quality_flags,
    )


def _build_macro_metrics(
    macro_history: pd.DataFrame,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
) -> Dict[str, Dict[str, Any]]:
    as_of_date = pd.Timestamp(as_of_time).tz_convert("UTC").normalize()
    macro = macro_history.copy()
    macro["date_key"] = macro["event_date"]
    by_instrument = {
        instrument_id: frame.sort_values("date_key").reset_index(drop=True)
        for instrument_id, frame in macro.groupby("instrument_id")
    }
    metrics: Dict[str, Dict[str, Any]] = {}

    def latest_series_value(instrument_id: str) -> float | None:
        frame = by_instrument.get(instrument_id)
        if frame is None or frame.empty:
            return None
        return _latest_value_on_or_before(frame, as_of_date, "value")

    for metric_name, spec in MACRO_SERIES_SPECS.items():
        value = latest_series_value(spec["instrument_id"])
        metrics[metric_name] = _macro_feature(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            unit=spec["unit"],
            value=value,
            components={
                "instrument_id": spec["instrument_id"],
                "formula": "latest_value_on_or_before_asof",
            },
            missing_reason="macro_series_unavailable" if value is None else None,
        )

    sofr = latest_series_value("SOFR")
    dff = latest_series_value("DFF")
    policy_rate = sofr if sofr is not None else dff
    policy_source = "SOFR" if sofr is not None else ("DFF" if dff is not None else None)
    metrics["macro.sofr_or_fed_funds"] = _macro_feature(
        metric_name="macro.sofr_or_fed_funds",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="pct",
        value=policy_rate,
        components={
            "preferred_instrument": "SOFR",
            "fallback_instrument": "DFF",
            "selected_instrument": policy_source,
            "formula": "latest_value_on_or_before_asof",
        },
        missing_reason="macro_series_unavailable" if policy_rate is None else None,
    )

    ust2 = metrics["macro.ust_2y_yield"]["value"]
    ust10 = metrics["macro.ust_10y_yield"]["value"]
    curve = None if ust2 is None or ust10 is None else float(ust10) - float(ust2)
    metrics["macro.curve_2s10s"] = _macro_feature(
        metric_name="macro.curve_2s10s",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="pct",
        value=curve,
        components={
            "macro.ust_10y_yield": ust10,
            "macro.ust_2y_yield": ust2,
            "formula": "macro.ust_10y_yield - macro.ust_2y_yield",
        },
        missing_reason="component_unavailable" if curve is None else None,
    )

    for metric_name, spec in MACRO_YOY_SPECS.items():
        instrument_id = spec["instrument_id"]
        frame = by_instrument.get(instrument_id)
        eligible = None if frame is None or frame.empty else frame[frame["date_key"] <= as_of_date].reset_index(drop=True)
        current = None if eligible is None or eligible.empty else float(eligible.iloc[-1]["value"])
        lag_observations = int(spec.get("lag_observations") or 0)
        if lag_observations > 0:
            prior = None if eligible is None or len(eligible) <= lag_observations else float(eligible.iloc[-1 - lag_observations]["value"])
            formula = f"(current_value / value_{lag_observations}_observations_prior) - 1"
            prior_component_key = f"value_{lag_observations}_observations_prior"
        else:
            prior_date = as_of_date - pd.DateOffset(months=12)
            prior = None if frame is None or frame.empty else _latest_value_on_or_before(frame, prior_date, "value")
            formula = "(current_value / value_12m_prior) - 1"
            prior_component_key = "value_12m_prior"
        if current is None or prior is None:
            value = None
            missing_reason = "macro_series_unavailable"
        elif prior <= 0:
            value = None
            missing_reason = "non_positive_base_value"
        else:
            value = (current / prior) - 1.0
            missing_reason = None
        metrics[metric_name] = _macro_feature(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            unit=spec["unit"],
            value=value,
            components={
                "instrument_id": instrument_id,
                "current_value": current,
                prior_component_key: prior,
                "formula": formula,
            },
            missing_reason=missing_reason,
        )

    return metrics


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.snapshot_path)
    entity_identifier_path = Path(args.entity_identifier_path)
    raw_timeseries_path = Path(args.raw_timeseries_path)
    crsp_market_cache_path = Path(args.crsp_market_cache_path) if args.crsp_market_cache_path else None
    crsp_daily_root = Path(args.crsp_daily_root) if args.crsp_daily_root else (
        DEFAULT_LOCAL_CRSP_DAILY_ROOT if DEFAULT_LOCAL_CRSP_DAILY_ROOT.exists() else None
    )
    companyfacts_root = Path(args.companyfacts_root) if args.companyfacts_root else (
        DEFAULT_LOCAL_COMPANYFACTS_ROOT if DEFAULT_LOCAL_COMPANYFACTS_ROOT.exists() else None
    )
    sec_filing_cache_root = Path(args.sec_filing_cache_root) if args.sec_filing_cache_root else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    permnos = _permno_map(entity_identifier_path)
    permno_by_entity = permnos.set_index("entity_id")["permno"].to_dict()
    computed_at = _now_iso()
    snapshot_rows = iter_snapshot_rows(snapshot_path)

    exact_market_price_source = crsp_market_cache_path or crsp_daily_root

    if crsp_market_cache_path is not None:
        market_provenance_source = str(crsp_market_cache_path)
        market_builder = _build_price_metrics_from_crsp
    elif crsp_daily_root is not None:
        market_provenance_source = str(crsp_daily_root)
        market_builder = _build_price_metrics_from_crsp
    elif args.allow_monthly_market_proxy:
        market_provenance_source = str(raw_timeseries_path)
        market_builder = _build_price_metrics
    else:
        market_provenance_source = str(raw_timeseries_path)
        market_builder = _build_price_metrics_from_crsp
    macro_history: pd.DataFrame | None = None
    macro_history_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
    macro_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    companyfacts_cache: Dict[str, dict | None] = {}
    issuer_shares_cache: Dict[str, tuple[float | None, dict[str, Any] | None]] = {}
    sec_session = _sec_session() if sec_filing_cache_root is not None and _sec_session is not None else None
    counters: Counter[str] = Counter()
    empty_price_history = pd.DataFrame()

    # Line-buffered writes make progress visible on disk as rows complete.
    with out_path.open("w", buffering=1) as out_handle:
        for row_batch in _iter_row_batches(snapshot_rows, max(1, int(args.price_load_batch_size))):
            batch_permnos = [permno_by_entity.get(row.get("company_id")) for row in row_batch]
            batch_price_history = _load_price_history_for_batch(
                permnos=[permno for permno in batch_permnos if permno],
                as_of_times=[row["as_of_time"] for row in row_batch],
                crsp_market_cache_path=crsp_market_cache_path,
                crsp_daily_root=crsp_daily_root,
                raw_timeseries_path=raw_timeseries_path,
                allow_monthly_market_proxy=args.allow_monthly_market_proxy,
            )
            for row in row_batch:
                as_of_time = row["as_of_time"]
                entity_id = row.get("company_id")
                permno = permno_by_entity.get(entity_id)
                features = row.setdefault("features", {})

                companyfacts_path = (companyfacts_root / f"CIK{entity_id}.json") if companyfacts_root is not None else None

                try:
                    with _company_processing_guard(args.company_processing_timeout_seconds):
                        price_metrics = market_builder(
                            permno=permno,
                            price_history=batch_price_history.get(str(permno), empty_price_history),
                            as_of_time=as_of_time,
                            computed_at=computed_at,
                            provenance_source=market_provenance_source,
                        )
                except Exception as exc:  # noqa: BLE001
                    error_type = "company_processing_timeout" if isinstance(exc, _CompanyProcessingTimeout) else "company_processing_failed"
                    counters[f"row:{error_type}"] += 1
                    price_metrics = _build_fail_open_market_metrics(
                        as_of_time=as_of_time,
                        computed_at=computed_at,
                        provenance_source=str(companyfacts_path or market_provenance_source),
                        error_type=error_type,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    try:
                        with _company_processing_guard(args.company_processing_timeout_seconds):
                            market_cap_node = None
                            if companyfacts_root is not None and entity_id not in companyfacts_cache:
                                companyfacts_cache[entity_id] = _load_companyfacts(companyfacts_path)
                            companyfacts = companyfacts_cache.get(entity_id)
                            if entity_id not in issuer_shares_cache:
                                issuer_shares = None
                                issuer_shares_meta = None
                                if companyfacts is not None:
                                    issuer_shares, issuer_shares_meta = _latest_shares_outstanding(companyfacts, as_of_time[:10])
                                    if issuer_shares_meta is not None:
                                        support_mode, missing = _shares_support_mode(
                                            reference_date=_parse_iso_date(issuer_shares_meta.get("end")),
                                            as_of_date=date.fromisoformat(as_of_time[:10]),
                                        )
                                        issuer_shares_meta = {
                                            **issuer_shares_meta,
                                            "support_mode": support_mode,
                                            "missing_reason": missing,
                                            "source": "sec_companyfacts",
                                        }
                                if issuer_shares is None and sec_session is not None and sec_filing_cache_root is not None:
                                    issuer_shares, issuer_shares_meta = _latest_issuer_shares_from_sec_filing(
                                        cik=str(entity_id).zfill(10),
                                        as_of_date=date.fromisoformat(as_of_time[:10]),
                                        session=sec_session,
                                        cache_dir=sec_filing_cache_root,
                                    )
                                    if issuer_shares_meta is not None:
                                        issuer_shares_meta = {
                                            **issuer_shares_meta,
                                            "source": "sec_filing_cover_page",
                                        }
                                issuer_shares_cache[entity_id] = (issuer_shares, issuer_shares_meta)
                            issuer_shares, issuer_shares_meta = issuer_shares_cache[entity_id]

                            if market_builder is _build_price_metrics_from_crsp and exact_market_price_source is not None:
                                market_cap_node = _build_market_cap_metric(
                                    price_history=batch_price_history.get(str(permno), empty_price_history),
                                    price_node=price_metrics["market.price_spot"],
                                    issuer_shares_outstanding=issuer_shares,
                                    issuer_shares_meta=issuer_shares_meta,
                                    as_of_time=as_of_time,
                                    computed_at=computed_at,
                                    provenance_source=market_provenance_source,
                                )

                            if market_cap_node is None:
                                market_cap_node = _build_market_cap_metric_from_companyfacts(
                                    companyfacts=companyfacts,
                                    price_node=price_metrics["market.price_spot"],
                                    as_of_time=as_of_time,
                                    computed_at=computed_at,
                                    companyfacts_path=companyfacts_path,
                                )
                    except Exception as exc:  # noqa: BLE001
                        error_type = "company_processing_timeout" if isinstance(exc, _CompanyProcessingTimeout) else "company_processing_failed"
                        counters[f"market_cap:{error_type}"] += 1
                        market_cap_node = _build_fail_open_market_cap_metric(
                            as_of_time=as_of_time,
                            computed_at=computed_at,
                            provenance_source=str(companyfacts_path or market_provenance_source),
                            error_type=error_type,
                            error_message=f"{type(exc).__name__}: {exc}",
                        )
                    price_metrics["market.market_cap_provider_direct"] = market_cap_node

                for metric_name, node in price_metrics.items():
                    features[metric_name] = node
                    counters[f"{metric_name}:{node['support_mode']}"] += 1

                if as_of_time not in macro_cache:
                    try:
                        as_of_date = pd.Timestamp(as_of_time).tz_convert("UTC").normalize()
                        if (
                            macro_history is None
                            or macro_history_range is None
                            or as_of_date < macro_history_range[0]
                            or as_of_date > macro_history_range[1]
                        ):
                            macro_history = _load_macro_history(
                                raw_timeseries_path,
                                min_asof_date=as_of_date,
                                max_asof_date=as_of_date,
                            )
                            macro_history_range = (as_of_date, as_of_date)
                        macro_cache[as_of_time] = _build_macro_metrics(
                            macro_history=macro_history,
                            as_of_time=as_of_time,
                            computed_at=computed_at,
                            provenance_source=str(raw_timeseries_path),
                        )
                    except Exception as exc:  # noqa: BLE001
                        counters["row:macro_build_failed"] += 1
                        macro_cache[as_of_time] = _build_fail_open_macro_metrics(
                            as_of_time=as_of_time,
                            computed_at=computed_at,
                            provenance_source=str(raw_timeseries_path),
                            error_type="macro_build_failed",
                            error_message=f"{type(exc).__name__}: {exc}",
                        )
                for metric_name, node in macro_cache[as_of_time].items():
                    features[metric_name] = node
                    counters[f"{metric_name}:{node['support_mode']}"] += 1

                out_handle.write(json.dumps(row) + "\n")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {}
        tracked_metrics = (
            ["market.market_cap_provider_direct"]
            + list(MARKET_METRICS)
            + [
                "macro.sofr_or_fed_funds",
                *MACRO_SERIES_SPECS.keys(),
                "macro.curve_2s10s",
                *MACRO_YOY_SPECS.keys(),
            ]
        )
        for metric_name in tracked_metrics:
            summary[metric_name] = {
                "exact": counters[f"{metric_name}:exact"],
                "proxy_missing_component": counters[f"{metric_name}:proxy_missing_component"],
                "unsupported": counters[f"{metric_name}:unsupported"],
            }
        if counters["row:company_processing_failed"] or counters["row:company_processing_timeout"] or counters["row:macro_build_failed"]:
            summary["row_fail_open"] = {
                "company_processing_failed": counters["row:company_processing_failed"],
                "company_processing_timeout": counters["row:company_processing_timeout"],
                "macro_build_failed": counters["row:macro_build_failed"],
            }
        if counters["market_cap:company_processing_failed"] or counters["market_cap:company_processing_timeout"]:
            summary["market_cap_fail_open"] = {
                "company_processing_failed": counters["market_cap:company_processing_failed"],
                "company_processing_timeout": counters["market_cap:company_processing_timeout"],
            }
        summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote market/macro input-layer snapshots -> {out_path}")
    if counters["row:company_processing_failed"] or counters["row:company_processing_timeout"] or counters["row:macro_build_failed"]:
        print(
            "row_fail_open:"
            f" company_processing_failed={counters['row:company_processing_failed']}"
            f" company_processing_timeout={counters['row:company_processing_timeout']}"
            f" macro_build_failed={counters['row:macro_build_failed']}"
        )
    if counters["market_cap:company_processing_failed"] or counters["market_cap:company_processing_timeout"]:
        print(
            "market_cap_fail_open:"
            f" company_processing_failed={counters['market_cap:company_processing_failed']}"
            f" company_processing_timeout={counters['market_cap:company_processing_timeout']}"
        )


if __name__ == "__main__":
    main()
