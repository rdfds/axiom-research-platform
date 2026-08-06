#!/usr/bin/env python
"""
As-Of Coverage Audit
====================
Generates coverage reports for warehouse tables:
- Missing required fields
- Missing months (prices)
- Missing sizes by action type (corporate actions)
"""

from __future__ import annotations

import argparse
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asof_store import AsOfWarehouse


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"


REQUIRED_FIELDS = {
    "warehouse_financials": [
        "company_id",
        "event_time",
        "available_time",
        "statement_type",
        "line_item",
        "value",
        "fiscal_period_end",
        "fiscal_year",
        "fiscal_quarter",
    ],
    "warehouse_prices": ["entity_id", "event_time", "available_time", "close", "adjusted_close", "volume"],
    "warehouse_prices_daily": [
        "entity_id",
        "event_time",
        "available_time",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "total_return_index",
    ],
    "warehouse_prices_daily_rdp": [
        "entity_id",
        "event_time",
        "available_time",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "total_return_index",
    ],
    "warehouse_macro": [
        "entity_id",
        "event_time",
        "available_time",
        "instrument_id",
        "instrument_type",
        "tenor",
        "value",
        "units",
    ],
    "warehouse_estimates": [
        "entity_id",
        "event_time",
        "available_time",
        "metric",
        "period",
        "consensus_value",
        # "num_estimates" is optional for some sources (e.g., FMP).
        # Set AUDIT_ESTIMATES_REQUIRE_NUM=1 to enforce it.
    ],
    "warehouse_press_releases": [
        "entity_id",
        "event_time",
        "available_time",
        "document_id",
        "release_date",
        "headline",
        "text",
    ],
    "warehouse_corp_actions": ["entity_id", "event_time", "available_time", "action_type", "announcement_date", "effective_date", "size"],
    "warehouse_mna_deals": ["deal_id", "announcement_date", "deal_value", "status"],
    "warehouse_13f_holdings": [
        "company_id",
        "event_time",
        "available_time",
        "holding_cusip",
        "value_k",
        "shares",
    ],
    "warehouse_13f_filings": [
        "company_id",
        "event_time",
        "available_time",
    ],
}

if os.getenv("AUDIT_ESTIMATES_REQUIRE_NUM", "0") in ("1", "true", "True"):
    REQUIRED_FIELDS["warehouse_estimates"].append("num_estimates")


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def missingness(df: pd.DataFrame, fields: List[str]) -> Dict[str, float]:
    stats = {}
    total = len(df)
    if total == 0:
        return {f"missing_{f}_pct": 1.0 for f in fields}
    for field in fields:
        if field not in df.columns:
            stats[f"missing_{field}_pct"] = 1.0
        else:
            stats[f"missing_{field}_pct"] = float(df[field].isna().mean())
    return stats


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if pd.isna(start) or pd.isna(end):
        return 0
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def audit_prices(df: pd.DataFrame) -> Dict[str, float]:
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    coverage = df.groupby("entity_id")["event_time"].agg(["min", "max", "nunique"]).reset_index()
    coverage["expected_months"] = coverage.apply(lambda r: months_between(r["min"], r["max"]), axis=1)
    coverage["missing_months"] = coverage["expected_months"] - coverage["nunique"]
    coverage["missing_months"] = coverage["missing_months"].clip(lower=0)

    total_missing = coverage["missing_months"].sum()
    avg_missing = coverage["missing_months"].mean()
    coverage_pct = (coverage["nunique"].sum() / coverage["expected_months"].sum()) if coverage["expected_months"].sum() else 0
    pct_with_gaps = (coverage["missing_months"] > 0).mean()

    coverage_path = WAREHOUSE_DIR / "coverage_prices_missing_months.csv"
    coverage.to_csv(coverage_path, index=False)

    return {
        "price_missing_months_total": float(total_missing),
        "price_missing_months_avg": float(avg_missing),
        "price_coverage_pct": float(coverage_pct),
        "price_entities_with_gaps_pct": float(pct_with_gaps),
    }


def audit_corp_actions(df: pd.DataFrame) -> None:
    if "action_type" not in df.columns or "size" not in df.columns:
        return
    summary = (
        df.groupby("action_type")["size"]
        .apply(lambda s: float(s.isna().mean()))
        .reset_index()
        .rename(columns={"size": "missing_size_pct"})
    )
    summary.to_csv(WAREHOUSE_DIR / "coverage_corp_actions_missing_size_by_type.csv", index=False)


def audit_mna(df: pd.DataFrame) -> None:
    if "status" not in df.columns:
        return
    summary = (
        df.groupby("status")["deal_value"]
        .apply(lambda s: float(s.isna().mean()))
        .reset_index()
        .rename(columns={"deal_value": "missing_deal_value_pct"})
    )
    summary.to_csv(WAREHOUSE_DIR / "coverage_mna_missing_value_by_status.csv", index=False)


def audit_financials(df: pd.DataFrame) -> None:
    if df.empty:
        return
    line_summary = (
        df.groupby(["statement_type", "line_item"])["value"]
        .apply(lambda s: float(s.isna().mean()))
        .reset_index()
        .rename(columns={"value": "missing_value_pct"})
    )
    line_summary["count"] = (
        df.groupby(["statement_type", "line_item"]).size().values
    )
    line_summary.to_csv(WAREHOUSE_DIR / "coverage_financials_line_items.csv", index=False)

def get_table_columns(path: Path) -> List[str]:
    if duckdb is None:
        # fallback: read only schema
        try:
            return list(pd.read_parquet(path, engine="pyarrow", columns=[]).columns)
        except Exception:
            return list(pd.read_parquet(path).columns)
    con = duckdb.connect(database=":memory:")
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{path.as_posix()}') LIMIT 0").df()
        return list(df.columns)
    finally:
        con.close()


def audit_large_table(
    table: str,
    required: List[str],
    asof: pd.Timestamp | None,
    store: AsOfWarehouse,
) -> Dict[str, object]:
    path = store.table_path(table)
    if not path.exists():
        return {
            "table": table,
            "rows": 0,
            "min_event_time": None,
            "max_event_time": None,
            "unique_entities": 0,
            **{f"missing_{field}_pct": 1.0 for field in required},
        }

    if duckdb is None:
        # fallback to full load (slow)
        df = store.query(table, as_of=asof)
        row = {
            "table": table,
            "rows": len(df),
            "min_event_time": df["event_time"].min() if "event_time" in df.columns else None,
            "max_event_time": df["event_time"].max() if "event_time" in df.columns else None,
            "unique_entities": df["entity_id"].nunique() if "entity_id" in df.columns else 0,
        }
        row.update(missingness(df, required))
        return row

    cols = get_table_columns(path)
    select_exprs = [
        "count(*) as rows",
        "min(event_time) as min_event_time",
        "max(event_time) as max_event_time",
    ]
    if "entity_id" in cols:
        select_exprs.append("count(distinct entity_id) as unique_entities")
    else:
        select_exprs.append("0 as unique_entities")

    present_required = []
    missing_required = []
    for field in required:
        if field in cols:
            present_required.append(field)
            select_exprs.append(f"avg(CASE WHEN {field} IS NULL THEN 1 ELSE 0 END) as missing_{field}_pct")
        else:
            missing_required.append(field)

    where_clauses = []
    params: List[object] = []
    if asof is not None and "available_time" in cols:
        where_clauses.append("available_time <= ?")
        params.append(asof)

    query = f"SELECT {', '.join(select_exprs)} FROM read_parquet('{path.as_posix()}', union_by_name=True)"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    con = duckdb.connect(database=":memory:")
    try:
        result: Dict[str, object] = {}
        err: Dict[str, Exception] = {}
        done = threading.Event()

        def _run():
            try:
                result["row"] = con.execute(query, params).df().iloc[0].to_dict()
            except Exception as exc:  # pragma: no cover
                err["error"] = exc
            finally:
                done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        start = time.perf_counter()
        while not done.wait(timeout=10):
            elapsed = time.perf_counter() - start
            log(f"  still aggregating {table}... {elapsed:.0f}s elapsed")
        t.join()
        if "error" in err:
            raise err["error"]
        row = result["row"] if "row" in result else {}
    finally:
        con.close()

    # Fill missing required fields (not present)
    for field in missing_required:
        row[f"missing_{field}_pct"] = 1.0
    row["table"] = table
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=None, help="Optional as-of date (YYYY-MM-DD)")
    parser.add_argument(
        "--skip",
        default=os.getenv("AUDIT_SKIP_TABLES", ""),
        help="Comma-separated table names to skip (or set AUDIT_SKIP_TABLES)",
    )
    args = parser.parse_args()

    asof = pd.to_datetime(args.as_of) if args.as_of else None
    store = AsOfWarehouse(WAREHOUSE_DIR)

    skip_tables = {t.strip() for t in args.skip.split(",") if t.strip()}
    summary_rows = []

    def query_with_progress(table: str, as_of, columns):
        result = {}
        err = {}
        done = threading.Event()

        def _run():
            try:
                result["df"] = store.query(table, as_of=as_of, columns=columns)
            except Exception as exc:  # pragma: no cover
                err["error"] = exc
            finally:
                done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        start = time.perf_counter()
        while not done.wait(timeout=10):
            elapsed = time.perf_counter() - start
            log(f"  still loading {table}... {elapsed:.0f}s elapsed")
        t.join()
        if "error" in err:
            raise err["error"]
        return result.get("df", pd.DataFrame())

    large_tables = {"warehouse_prices_daily", "warehouse_prices_daily_rdp", "warehouse_financials", "warehouse_13f_holdings"}

    def audit_press_releases_streaming(path: Path, required: List[str], asof: pd.Timestamp | None) -> Dict[str, object]:
        compact_only = os.getenv("AUDIT_PRESS_RELEASES_COMPACT_ONLY", "1") == "1"
        if compact_only:
            files = sorted(path.glob("year=*/part_compact.parquet"))
        else:
            files = sorted(path.glob("year=*/part_*.parquet"))
        if not files:
            return {
                "table": "warehouse_press_releases",
                "rows": 0,
                "min_event_time": None,
                "max_event_time": None,
                "unique_entities": 0,
                **{f"missing_{field}_pct": 1.0 for field in required},
            }

        skip_text = os.getenv("AUDIT_PRESS_RELEASES_SKIP_TEXT", "0") == "1"
        log_every = int(os.getenv("AUDIT_PRESS_RELEASES_LOG_EVERY", "200"))
        verbose = os.getenv("AUDIT_PRESS_RELEASES_VERBOSE", "0") == "1"
        file_log_every = int(os.getenv("AUDIT_PRESS_RELEASES_FILE_LOG_EVERY", "0"))
        slow_log_seconds = float(os.getenv("AUDIT_PRESS_RELEASES_SLOW_LOG_SECONDS", "10"))
        start_file = int(os.getenv("AUDIT_PRESS_RELEASES_START_FILE", "0"))
        max_mb = float(os.getenv("AUDIT_PRESS_RELEASES_MAX_MB", "0"))

        if start_file:
            files = files[start_file:]

        total_rows = 0
        min_event = None
        max_event = None
        entity_ids = set()
        missing_counts = {field: 0 for field in required}

        cols = set(required)
        cols.update(["event_time", "available_time", "entity_id"])
        if skip_text:
            cols.discard("text")

        start = time.perf_counter()
        def _read_with_progress(fpath: Path) -> pd.DataFrame | None:
            if max_mb and fpath.stat().st_size > max_mb * 1024 * 1024:
                log(f"  Skipping large parquet (> {max_mb} MB): {fpath}")
                return None

            result: Dict[str, object] = {}
            done = threading.Event()

            def _run() -> None:
                try:
                    result["df"] = pd.read_parquet(fpath, columns=sorted(cols))
                except Exception as exc:  # pragma: no cover - defensive
                    result["error"] = exc
                finally:
                    done.set()

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            start_ts = time.perf_counter()
            while not done.wait(timeout=slow_log_seconds):
                elapsed = time.perf_counter() - start_ts
                log(f"  still reading {fpath.name}... {elapsed:.0f}s")
            t.join()
            if "error" in result:
                raise result["error"]  # type: ignore[arg-type]
            return result.get("df")  # type: ignore[return-value]

        for idx, fpath in enumerate(files, start=1):
            if verbose or (file_log_every and idx % file_log_every == 0):
                log(f"  reading {idx}/{len(files)}: {fpath.name}")
            try:
                df = _read_with_progress(fpath)
            except Exception as exc:
                log(f"  Skipping unreadable parquet {fpath}: {exc}")
                continue

            if df is None or df.empty:
                continue

            # df is non-empty at this point

            if asof is not None and "available_time" in df.columns:
                df["available_time"] = pd.to_datetime(df["available_time"], errors="coerce")
                df = df[df["available_time"] <= asof]
                if df.empty:
                    continue

            if "event_time" in df.columns:
                df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
                min_event = df["event_time"].min() if min_event is None else min(min_event, df["event_time"].min())
                max_event = df["event_time"].max() if max_event is None else max(max_event, df["event_time"].max())

            if "entity_id" in df.columns:
                entity_ids.update(df["entity_id"].dropna().astype(str).tolist())

            total_rows += len(df)
            for field in required:
                if field not in df.columns:
                    missing_counts[field] += len(df)
                else:
                    missing_counts[field] += int(df[field].isna().sum())

            if log_every and idx % log_every == 0:
                elapsed = time.perf_counter() - start
                rate = idx / elapsed if elapsed > 0 else 0.0
                remaining = len(files) - idx
                eta = (remaining / rate) if rate > 0 else 0.0
                log(
                    f"  press_releases progress: {idx}/{len(files)} files | rows {total_rows:,} | ETA {eta/60:.1f}m"
                )

        row = {
            "table": "warehouse_press_releases",
            "rows": total_rows,
            "min_event_time": min_event,
            "max_event_time": max_event,
            "unique_entities": len(entity_ids),
        }
        for field in required:
            if total_rows == 0:
                row[f"missing_{field}_pct"] = 1.0
            else:
                row[f"missing_{field}_pct"] = float(missing_counts[field] / total_rows)
        return row

    for table, required in REQUIRED_FIELDS.items():
        if table in skip_tables:
            log(f"Skipping {table} (per --skip/AUDIT_SKIP_TABLES)")
            continue
        log(f"Auditing {table}...")
        if table in large_tables:
            t0 = time.perf_counter()
            try:
                row = audit_large_table(table, required, asof, store)
            except Exception as exc:
                log(f"  Audit failed for {table}: {exc}. Skipping.")
                continue
            t1 = time.perf_counter()
            log(f"  Aggregated {row.get('rows', 0):,} rows in {t1 - t0:.1f}s")
            summary_rows.append(row)
            continue

        if table == "warehouse_press_releases":
            t0 = time.perf_counter()
            row = audit_press_releases_streaming(store.table_path(table), required, asof)
            t1 = time.perf_counter()
            log(f"  Aggregated {row.get('rows', 0):,} rows in {t1 - t0:.1f}s")
            summary_rows.append(row)
            continue

        t0 = time.perf_counter()
        # Only pull necessary columns to reduce I/O
        select_cols = set(required)
        select_cols.update(["event_time", "entity_id"])
        df = query_with_progress(table, asof, sorted(select_cols))
        t1 = time.perf_counter()
        log(f"  Loaded {len(df):,} rows in {t1 - t0:.1f}s")
        if df.empty:
            summary_rows.append(
                {
                    "table": table,
                    "rows": 0,
                    "min_event_time": None,
                    "max_event_time": None,
                    "unique_entities": 0,
                    **{f"missing_{field}_pct": 1.0 for field in required},
                }
            )
            continue

        df["event_time"] = pd.to_datetime(df["event_time"])
        min_event = df["event_time"].min()
        max_event = df["event_time"].max()
        entity_col = "entity_id" if "entity_id" in df.columns else None
        unique_entities = df[entity_col].nunique() if entity_col else 0
        log(
            f"  event_time: {min_event.date() if pd.notna(min_event) else 'n/a'}"
            f" -> {max_event.date() if pd.notna(max_event) else 'n/a'} |"
            f" entities: {unique_entities:,}"
        )

        row = {
            "table": table,
            "rows": len(df),
            "min_event_time": min_event,
            "max_event_time": max_event,
            "unique_entities": unique_entities,
        }
        t2 = time.perf_counter()
        row.update(missingness(df, required))
        log(f"  missingness computed in {time.perf_counter() - t2:.1f}s")
        summary_rows.append(row)

        if table == "warehouse_prices":
            t3 = time.perf_counter()
            row.update(audit_prices(df))
            log(f"  price coverage computed in {time.perf_counter() - t3:.1f}s")
        if table == "warehouse_financials":
            t3 = time.perf_counter()
            audit_financials(df)
            log(f"  financial line-item coverage in {time.perf_counter() - t3:.1f}s")
        if table == "warehouse_corp_actions":
            t3 = time.perf_counter()
            audit_corp_actions(df)
            log(f"  corp action coverage in {time.perf_counter() - t3:.1f}s")
        if table == "warehouse_mna_deals":
            t3 = time.perf_counter()
            audit_mna(df)
            log(f"  M&A coverage in {time.perf_counter() - t3:.1f}s")

    summary = pd.DataFrame(summary_rows)
    summary_path = WAREHOUSE_DIR / "coverage_audit_summary.csv"
    summary.to_csv(summary_path, index=False)
    log(f"Saved audit summary -> {summary_path}")


if __name__ == "__main__":
    main()
