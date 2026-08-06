#!/usr/bin/env python3
"""Overlay issuer bond-spread truth onto a flat quantitative export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import duckdb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-path", required=True, help="Input flat parquet export")
    parser.add_argument("--entity-identifier-path", required=True, help="Entity identifier parquet")
    parser.add_argument("--trace-daily-path", required=True, help="TRACE daily parquet")
    parser.add_argument("--bond-issuances-path", required=True, help="FISD bond issuances parquet")
    parser.add_argument("--raw-timeseries-path", required=True, help="Raw timeseries parquet with Treasury yields")
    parser.add_argument("--as-of-date", default="2024-12-31", help="As-of date in YYYY-MM-DD")
    parser.add_argument(
        "--current-lookback-days",
        type=int,
        default=90,
        help="Recency window for the current spread level",
    )
    parser.add_argument(
        "--history-lookback-days",
        type=int,
        default=730,
        help="History window for issuer spread percentile",
    )
    parser.add_argument("--out-parquet", required=True, help="Output parquet path")
    parser.add_argument("--out-csv", help="Optional output CSV path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _support_counts(series: pd.Series) -> Dict[str, int]:
    support = series.fillna("unsupported").astype(str)
    return {
        "exact": int((support == "exact").sum()),
        "proxy_missing_component": int((support == "proxy_missing_component").sum()),
        "unsupported": int((support == "unsupported").sum()),
    }


def _json_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            item = value.item()
            if isinstance(item, pd.Timestamp):
                return item.isoformat()
            return item
        except Exception:
            pass
    return value


def build_bond_credit_overlay(
    *,
    flat_path: Path,
    entity_identifier_path: Path,
    trace_daily_path: Path,
    bond_issuances_path: Path,
    raw_timeseries_path: Path,
    as_of_date: str,
    current_lookback_days: int,
    history_lookback_days: int,
) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of_date)

    companies = pd.read_parquet(flat_path, columns=["company_id"]).copy()
    companies["company_id"] = companies["company_id"].astype(str)
    companies = companies.drop_duplicates()

    identifiers = pd.read_parquet(
        entity_identifier_path,
        columns=["entity_id", "identifier_type", "identifier_value"],
    ).copy()
    identifiers = identifiers[identifiers["identifier_type"].astype(str).str.lower() == "permno"].copy()
    identifiers["company_id"] = identifiers["entity_id"].astype(str)
    identifiers["permno"] = pd.to_numeric(identifiers["identifier_value"], errors="coerce")
    identifiers = identifiers[identifiers["permno"].notna()][["company_id", "permno"]].drop_duplicates()

    issues = pd.read_parquet(
        bond_issuances_path,
        columns=[
            "permno",
            "COMPLETE_CUSIP",
            "offering_date",
            "maturity_date",
            "amount",
            "offering_amt_k",
            "CONVERTIBLE",
            "ASSET_BACKED",
            "PERPETUAL",
            "PRIVATE_PLACEMENT",
        ],
    ).copy()
    issues["permno"] = pd.to_numeric(issues["permno"], errors="coerce")
    issues["offering_date"] = pd.to_datetime(issues["offering_date"], errors="coerce").dt.normalize()
    issues["maturity_date"] = pd.to_datetime(issues["maturity_date"], errors="coerce").dt.normalize()
    issues["cusip9"] = issues["COMPLETE_CUSIP"].astype(str).str.upper().str.strip()
    issues["issue_amount"] = pd.to_numeric(issues["amount"], errors="coerce")
    missing_amount_mask = issues["issue_amount"].isna()
    issues.loc[missing_amount_mask, "issue_amount"] = (
        pd.to_numeric(issues.loc[missing_amount_mask, "offering_amt_k"], errors="coerce") * 1000.0
    )

    eligible_issues = (
        issues.merge(identifiers, on="permno", how="inner")
        .merge(companies, on="company_id", how="inner")
        .loc[
            lambda df: df["cusip9"].ne("NAN")
            & df["offering_date"].notna()
            & df["maturity_date"].notna()
            & (df["offering_date"] <= as_of_ts)
            & (df["maturity_date"] > as_of_ts)
            & df["CONVERTIBLE"].fillna("N").eq("N")
            & df["ASSET_BACKED"].fillna("N").eq("N")
            & df["PERPETUAL"].fillna("N").eq("N")
            & df["PRIVATE_PLACEMENT"].fillna("N").eq("N"),
            ["company_id", "cusip9", "maturity_date", "issue_amount"],
        ]
        .drop_duplicates()
    )
    if eligible_issues.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "credit_spread_trade_date",
                "credit_spread_level",
                "spread_percentile_2y",
                "current_issue_count",
                "history_obs",
            ]
        )

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.register("eligible_issues_df", eligible_issues)
    trace_query = f"""
    SELECT
      e.company_id,
      e.cusip9,
      e.maturity_date,
      e.issue_amount,
      CAST(t.trade_date AS DATE) AS trade_date,
      t.yld_pt_avg,
      COALESCE(t.volume_total, 0.0) AS volume_total,
      COALESCE(t.trades, 0) AS trades
    FROM read_parquet('{_sql_path(trace_daily_path)}') t
    JOIN eligible_issues_df e
      ON upper(trim(t.cusip_id)) = e.cusip9
    WHERE CAST(t.trade_date AS DATE) <= DATE '{as_of_date}'
      AND CAST(t.trade_date AS DATE) >= DATE '{as_of_date}' - INTERVAL '{int(history_lookback_days)} days'
      AND t.yld_pt_avg IS NOT NULL
    """
    try:
        trace_df = con.execute(trace_query).fetchdf()
    finally:
        con.close()

    if trace_df.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "credit_spread_trade_date",
                "credit_spread_level",
                "spread_percentile_2y",
                "current_issue_count",
                "history_obs",
            ]
        )

    trace_df["trade_date"] = pd.to_datetime(trace_df["trade_date"], errors="coerce").dt.normalize()
    trace_df["maturity_date"] = pd.to_datetime(trace_df["maturity_date"], errors="coerce").dt.normalize()

    issue_daily = (
        trace_df.groupby(["company_id", "cusip9", "issue_amount", "maturity_date", "trade_date"], as_index=False)
        .agg(
            yld_pt_avg=("yld_pt_avg", "mean"),
            volume_total=("volume_total", "sum"),
            trades=("trades", "sum"),
        )
    )
    issue_daily["years_to_maturity"] = (
        (issue_daily["maturity_date"] - issue_daily["trade_date"]).dt.days.clip(lower=1) / 365.25
    ).clip(lower=0.25)
    issue_daily["treasury_id"] = np.select(
        [
            issue_daily["years_to_maturity"] <= 1.5,
            issue_daily["years_to_maturity"] <= 3.5,
            issue_daily["years_to_maturity"] <= 7.5,
            issue_daily["years_to_maturity"] <= 20.0,
        ],
        ["DGS1", "DGS2", "DGS5", "DGS10"],
        default="DGS30",
    )

    treasuries = pd.read_parquet(
        raw_timeseries_path,
        columns=["instrument_id", "event_time", "value"],
        filters=[("instrument_id", "in", ["DGS1", "DGS2", "DGS5", "DGS10", "DGS30"])],
    ).copy()
    treasuries["trade_date"] = pd.to_datetime(treasuries["event_time"], errors="coerce").dt.normalize()
    treasuries = treasuries.loc[treasuries["value"].notna() & treasuries["trade_date"].notna(), ["instrument_id", "trade_date", "value"]]
    treasuries = treasuries.sort_values(["instrument_id", "trade_date"])

    matched_chunks = []
    for treasury_id, chunk in issue_daily.groupby("treasury_id", sort=False):
        treasury_chunk = treasuries[treasuries["instrument_id"] == treasury_id][["trade_date", "value"]].copy()
        if treasury_chunk.empty:
            continue
        chunk_sorted = chunk.sort_values("trade_date").copy()
        chunk_sorted["trade_date"] = pd.to_datetime(chunk_sorted["trade_date"]).astype("datetime64[ns]")
        treasury_chunk["trade_date"] = pd.to_datetime(treasury_chunk["trade_date"]).astype("datetime64[ns]")
        matched = pd.merge_asof(
            chunk_sorted,
            treasury_chunk.sort_values("trade_date"),
            on="trade_date",
            direction="backward",
        )
        matched["treasury_id"] = treasury_id
        matched = matched.rename(columns={"value": "treasury_yield"})
        matched_chunks.append(matched)

    if not matched_chunks:
        return pd.DataFrame(
            columns=[
                "company_id",
                "credit_spread_trade_date",
                "credit_spread_level",
                "spread_percentile_2y",
                "current_issue_count",
                "history_obs",
            ]
        )

    issue_spreads = pd.concat(matched_chunks, ignore_index=True)
    issue_spreads = issue_spreads[issue_spreads["treasury_yield"].notna()].copy()
    issue_spreads["spread_pct"] = issue_spreads["yld_pt_avg"] - issue_spreads["treasury_yield"]
    issue_spreads["weight"] = np.where(
        issue_spreads["volume_total"] > 0,
        issue_spreads["volume_total"],
        np.where(issue_spreads["issue_amount"] > 0, issue_spreads["issue_amount"], 1.0),
    )

    issuer_daily = (
        issue_spreads.assign(weighted_spread=issue_spreads["spread_pct"] * issue_spreads["weight"])
        .groupby(["company_id", "trade_date"], as_index=False)
        .agg(
            issuer_spread_pct=("weighted_spread", "sum"),
            total_weight=("weight", "sum"),
            issue_count=("cusip9", "nunique"),
        )
    )
    issuer_daily["issuer_spread_pct"] = issuer_daily["issuer_spread_pct"] / issuer_daily["total_weight"]
    issuer_daily = issuer_daily.drop(columns=["total_weight"]).sort_values(["company_id", "trade_date"])

    current_cutoff = as_of_ts - pd.Timedelta(days=int(current_lookback_days))
    current_spread = (
        issuer_daily[issuer_daily["trade_date"] >= current_cutoff]
        .sort_values(["company_id", "trade_date"])
        .groupby("company_id", as_index=False)
        .tail(1)
        .rename(
            columns={
                "trade_date": "credit_spread_trade_date",
                "issuer_spread_pct": "current_spread_pct",
                "issue_count": "current_issue_count",
            }
        )
    )
    if current_spread.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "credit_spread_trade_date",
                "credit_spread_level",
                "spread_percentile_2y",
                "current_issue_count",
                "history_obs",
            ]
        )

    history = issuer_daily.merge(
        current_spread[["company_id", "current_spread_pct"]],
        on="company_id",
        how="inner",
    )
    percentile = history.groupby("company_id", group_keys=False).apply(
        lambda g: float((g["issuer_spread_pct"] <= g["current_spread_pct"].iloc[0]).mean())
    )
    history_obs = history.groupby("company_id").size().astype(int)

    overlay = current_spread[["company_id", "credit_spread_trade_date", "current_spread_pct", "current_issue_count"]].copy()
    overlay["credit_spread_level"] = overlay["current_spread_pct"] / 100.0
    overlay["spread_percentile_2y"] = overlay["company_id"].map(percentile)
    overlay["history_obs"] = overlay["company_id"].map(history_obs)
    return overlay[
        [
            "company_id",
            "credit_spread_trade_date",
            "credit_spread_level",
            "spread_percentile_2y",
            "current_issue_count",
            "history_obs",
        ]
    ]


def main() -> None:
    args = parse_args()

    flat_path = Path(args.flat_path)
    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    merged = pd.read_parquet(flat_path).copy()
    merged["company_id"] = merged["company_id"].astype(str)

    overlay = build_bond_credit_overlay(
        flat_path=flat_path,
        entity_identifier_path=Path(args.entity_identifier_path),
        trace_daily_path=Path(args.trace_daily_path),
        bond_issuances_path=Path(args.bond_issuances_path),
        raw_timeseries_path=Path(args.raw_timeseries_path),
        as_of_date=args.as_of_date,
        current_lookback_days=args.current_lookback_days,
        history_lookback_days=args.history_lookback_days,
    )
    overlay["company_id"] = overlay["company_id"].astype(str)

    merged = merged.merge(overlay, on="company_id", how="left")

    if "market__credit_spread_percentile_2y__value" not in merged.columns:
        merged["market__credit_spread_percentile_2y__value"] = None
    if "market__credit_spread_percentile_2y__unit" not in merged.columns:
        merged["market__credit_spread_percentile_2y__unit"] = "ratio"
    if "market__credit_spread_percentile_2y__support_mode" not in merged.columns:
        merged["market__credit_spread_percentile_2y__support_mode"] = "unsupported"
    if "market__credit_spread_percentile_2y__fallback_used" not in merged.columns:
        merged["market__credit_spread_percentile_2y__fallback_used"] = None

    merged["market__credit_spread_trade_date"] = merged["credit_spread_trade_date"]
    merged["market__credit_spread_issue_count__value"] = merged["current_issue_count"]
    merged["market__credit_spread_issue_count__unit"] = "count"
    merged["market__credit_spread_issue_count__support_mode"] = merged["current_issue_count"].notna().map(
        {True: "exact", False: "unsupported"}
    )
    merged["market__credit_spread_history_obs__value"] = merged["history_obs"]
    merged["market__credit_spread_history_obs__unit"] = "trading_days"
    merged["market__credit_spread_history_obs__support_mode"] = merged["history_obs"].notna().map(
        {True: "exact", False: "unsupported"}
    )

    exact_mask = merged["credit_spread_level"].notna()
    percentile_mask = merged["spread_percentile_2y"].notna()

    merged.loc[exact_mask, "market__credit_spread_level__value"] = merged.loc[exact_mask, "credit_spread_level"]
    merged.loc[exact_mask, "market__credit_spread_level__unit"] = "pct"
    merged.loc[exact_mask, "market__credit_spread_level__support_mode"] = "exact"
    merged.loc[exact_mask, "market__credit_spread_level__fallback_used"] = (
        f"trace_fisd_volume_weighted_spread_to_matched_treasury_{int(args.current_lookback_days)}d"
    )

    merged.loc[percentile_mask, "market__credit_spread_percentile_2y__value"] = (
        merged.loc[percentile_mask, "spread_percentile_2y"]
    )
    merged.loc[percentile_mask, "market__credit_spread_percentile_2y__unit"] = "ratio"
    merged.loc[percentile_mask, "market__credit_spread_percentile_2y__support_mode"] = "exact"
    merged.loc[percentile_mask, "market__credit_spread_percentile_2y__fallback_used"] = (
        f"issuer_spread_percentile_over_{int(args.history_lookback_days)}d_trace_history"
    )

    merged.loc[percentile_mask, "market__credit_window_proxy__value"] = 1.0 - merged.loc[
        percentile_mask, "spread_percentile_2y"
    ]
    merged.loc[percentile_mask, "market__credit_window_proxy__unit"] = "ratio"
    merged.loc[percentile_mask, "market__credit_window_proxy__support_mode"] = "proxy_missing_component"
    merged.loc[percentile_mask, "market__credit_window_proxy__fallback_used"] = (
        f"one_minus_real_bond_spread_percentile_{int(args.history_lookback_days)}d"
    )

    merged = merged.drop(
        columns=[
            "credit_spread_trade_date",
            "credit_spread_level",
            "spread_percentile_2y",
            "current_issue_count",
            "history_obs",
        ],
        errors="ignore",
    )

    merged.to_parquet(out_parquet, index=False)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_csv, index=False)

    if args.summary_out:
        sample_ids = ["0000006201", "0000003453", "0001637459", "0000909832", "0000003197", "0001639438"]
        sample = {}
        for cid in sample_ids:
            sub = merged[merged["company_id"] == cid]
            if sub.empty:
                continue
            row = sub.iloc[0]
            sample[cid] = {
                "company_name": row["company_name"],
                "credit_spread_level": _json_scalar(row.get("market__credit_spread_level__value")),
                "credit_spread_support_mode": _json_scalar(row.get("market__credit_spread_level__support_mode")),
                "credit_spread_trade_date": _json_scalar(row.get("market__credit_spread_trade_date")),
                "credit_spread_issue_count": _json_scalar(row.get("market__credit_spread_issue_count__value")),
                "credit_spread_history_obs": _json_scalar(row.get("market__credit_spread_history_obs__value")),
                "credit_spread_percentile_2y": _json_scalar(row.get("market__credit_spread_percentile_2y__value")),
                "credit_window_proxy": _json_scalar(row.get("market__credit_window_proxy__value")),
            }

        summary = {
            "rows": int(len(merged)),
            "bond_spread_matches": int(exact_mask.sum()),
            "bond_spread_percentile_matches": int(percentile_mask.sum()),
            "current_lookback_days": int(args.current_lookback_days),
            "history_lookback_days": int(args.history_lookback_days),
            "market.credit_spread_level": _support_counts(merged["market__credit_spread_level__support_mode"]),
            "market.credit_spread_percentile_2y": _support_counts(
                merged["market__credit_spread_percentile_2y__support_mode"]
            ),
            "market.credit_window_proxy": _support_counts(merged["market__credit_window_proxy__support_mode"]),
            "sample": sample,
        }
        Path(args.summary_out).write_text(json.dumps(summary, indent=2))

    print(f"Overlayed bond credit spreads -> {out_parquet}")


if __name__ == "__main__":
    main()
