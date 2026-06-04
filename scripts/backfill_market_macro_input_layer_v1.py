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


