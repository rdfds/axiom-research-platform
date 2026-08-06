#!/usr/bin/env python3
"""Overlay market-grade Refinitiv market/pricing metrics onto a flat quantitative export."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from scripts.repair_rating_state_artifact import (
    _resolve_ratings_path,
    build_rating_index,
    load_issuer_ratings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-path", required=True, help="Input flat parquet export")
    parser.add_argument("--provider-reference-path", required=True, help="Refinitiv fundamentals parquet")
    parser.add_argument("--entity-identifier-path", required=True, help="Entity identifier parquet")
    parser.add_argument("--ratings-path", help="Optional issuer ratings parquet/csv.gz path")
    parser.add_argument("--out-parquet", required=True, help="Output parquet path")
    parser.add_argument("--out-csv", help="Optional output CSV path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


def _ticker_identifier_map(path: Path) -> pd.DataFrame:
    identifiers = _read_parquet_with_retries(path)
    identifiers = identifiers[identifiers["identifier_type"].astype(str).str.lower() == "ticker"].copy()
    identifiers["ticker"] = identifiers["identifier_value"].astype(str).str.upper().str.strip()
    return identifiers[["entity_id", "ticker"]].drop_duplicates()


def _read_parquet_with_retries(path: Path, attempts: int = 4, sleep_seconds: float = 3.0) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return pd.read_parquet(path).copy()
        except TimeoutError as exc:
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(sleep_seconds)
    raise last_error


def _refinitiv_map(provider_reference_path: Path, entity_identifier_path: Path) -> pd.DataFrame:
    ref = _read_parquet_with_retries(provider_reference_path)
    ref["ticker"] = ref["Instrument"].astype(str).str.replace(r"\..*$", "", regex=True).str.upper().str.strip()
    tickers = _ticker_identifier_map(entity_identifier_path)
    merged = ref.merge(tickers, on="ticker", how="inner")
    merged["entity_id"] = merged["entity_id"].astype(str)
    return merged.sort_values(["entity_id", "Instrument"]).drop_duplicates("entity_id", keep="first")


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
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def main() -> None:
    args = parse_args()

    flat_path = Path(args.flat_path)
    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = _read_parquet_with_retries(flat_path)
    df["company_id"] = df["company_id"].astype(str)

    # Allow repeated overlays by removing raw provider columns that may already exist
    # from a prior run, so the merge does not create suffixed duplicates.
    existing_provider_columns = [
        "Instrument",
        "Company Common Name",
        "provider_revenue",
        "provider_ebitda",
        "provider_free_cash_flow",
        "provider_cash_and_short_term_investments",
        "provider_market_cap",
        "provider_ev_ebitda",
    ]
    existing_rating_columns = [
        "capital_structure__rating_state__rating",
        "capital_structure__rating_state__rating_support_mode",
        "capital_structure__rating_state__outlook",
        "capital_structure__rating_state__watchlist",
        "capital_structure__rating_state__score__value",
        "capital_structure__rating_state__score__unit",
        "capital_structure__rating_state__score__support_mode",
        "capital_structure__rating_state__score__fallback_used",
        "capital_structure__rating_state__source_type",
        "capital_structure__rating_state__rating_date",
    ]
    df = df.drop(columns=[col for col in existing_rating_columns if col in df.columns], errors="ignore")

    provider_fields = [
        "provider_revenue",
        "provider_ebitda",
        "provider_free_cash_flow",
        "provider_cash_and_short_term_investments",
        "provider_market_cap",
        "provider_ev_ebitda",
    ]
    needs_provider_refresh = not all(col in df.columns for col in provider_fields)
    if needs_provider_refresh:
        ref = _refinitiv_map(Path(args.provider_reference_path), Path(args.entity_identifier_path))
        ref = ref.rename(
            columns={
                "entity_id": "company_id",
                "Revenue": "provider_revenue",
                "EBITDA": "provider_ebitda",
                "Free Cash Flow": "provider_free_cash_flow",
                "Cash and Short Term Investments": "provider_cash_and_short_term_investments",
                "Company Market Cap": "provider_market_cap",
                "Enterprise Value To EBITDA (Daily Time Series Ratio)": "provider_ev_ebitda",
            }
        )
        df = df.drop(columns=[col for col in existing_provider_columns if col in df.columns], errors="ignore")
        merged = df.merge(
            ref[
                [
                    "company_id",
                    "Instrument",
                    "Company Common Name",
                    "provider_revenue",
                    "provider_ebitda",
                    "provider_free_cash_flow",
                    "provider_cash_and_short_term_investments",
                    "provider_market_cap",
                    "provider_ev_ebitda",
                ]
            ],
            on="company_id",
            how="left",
        )
    else:
        merged = df.copy()

    merged["operating__ebitda_provider_direct__value"] = merged["provider_ebitda"]
    merged["operating__ebitda_provider_direct__unit"] = "usd"
    merged["operating__ebitda_provider_direct__support_mode"] = merged["provider_ebitda"].notna().map(
        {True: "exact", False: "unsupported"}
    )
    merged["operating__revenue_provider_direct__value"] = merged["provider_revenue"]
    merged["operating__revenue_provider_direct__unit"] = "usd"
    merged["operating__revenue_provider_direct__support_mode"] = merged["provider_revenue"].notna().map(
        {True: "exact", False: "unsupported"}
    )
    merged["operating__free_cash_flow_provider_direct__value"] = merged["provider_free_cash_flow"]
    merged["operating__free_cash_flow_provider_direct__unit"] = "usd"
    merged["operating__free_cash_flow_provider_direct__support_mode"] = merged["provider_free_cash_flow"].notna().map(
        {True: "exact", False: "unsupported"}
    )
    merged["liquidity__cash_and_short_term_investments_provider_direct__value"] = (
        merged["provider_cash_and_short_term_investments"]
    )
    merged["liquidity__cash_and_short_term_investments_provider_direct__unit"] = "usd"
    merged["liquidity__cash_and_short_term_investments_provider_direct__support_mode"] = (
        merged["provider_cash_and_short_term_investments"].notna().map({True: "exact", False: "unsupported"})
    )
    merged["market__market_cap_provider_direct__value"] = merged["provider_market_cap"]
    merged["market__market_cap_provider_direct__unit"] = "usd"
    merged["market__market_cap_provider_direct__support_mode"] = merged["provider_market_cap"].notna().map(
        {True: "exact", False: "unsupported"}
    )

    ratings_path = None
    ratings_df = None
    rating_index = {}
    rating_match_count = 0
    if args.ratings_path:
        ratings_path = _resolve_ratings_path(args.ratings_path)
        ratings_df = load_issuer_ratings(ratings_path)
        rating_index = build_rating_index(ratings_df)

        merged["capital_structure__rating_state__rating"] = None
        merged["capital_structure__rating_state__rating_support_mode"] = "unsupported"
        merged["capital_structure__rating_state__outlook"] = None
        merged["capital_structure__rating_state__watchlist"] = None
        merged["capital_structure__rating_state__score__value"] = None
        merged["capital_structure__rating_state__score__unit"] = "ordinal_notch"
        merged["capital_structure__rating_state__score__support_mode"] = "unsupported"
        merged["capital_structure__rating_state__score__fallback_used"] = None
        merged["capital_structure__rating_state__source_type"] = None
        merged["capital_structure__rating_state__rating_date"] = None

        for idx, company_id in merged["company_id"].items():
            record = rating_index.get(company_id)
            if not record:
                continue
            payload = record["payload"]
            rating_match_count += 1
            merged.at[idx, "capital_structure__rating_state__rating"] = payload.get("rating")
            merged.at[idx, "capital_structure__rating_state__rating_support_mode"] = (
                "exact" if payload.get("rating") is not None else "unsupported"
            )
            merged.at[idx, "capital_structure__rating_state__outlook"] = payload.get("outlook")
            merged.at[idx, "capital_structure__rating_state__watchlist"] = payload.get("watchlist")
            merged.at[idx, "capital_structure__rating_state__score__value"] = payload.get("score")
            merged.at[idx, "capital_structure__rating_state__score__support_mode"] = (
                "exact" if payload.get("score") is not None else "unsupported"
            )
            merged.at[idx, "capital_structure__rating_state__score__fallback_used"] = (
                None if payload.get("score") is not None else "rating_symbol_without_numeric_score"
            )
            merged.at[idx, "capital_structure__rating_state__source_type"] = record.get("source_type")
            merged.at[idx, "capital_structure__rating_state__rating_date"] = record.get("rating_date")

    margin_mask = merged["provider_ebitda"].notna() & merged["provider_revenue"].notna() & (merged["provider_revenue"] != 0)
    fcf_yield_mask = (
        merged["provider_free_cash_flow"].notna()
        & merged["provider_market_cap"].notna()
        & (merged["provider_market_cap"] != 0)
    )
    fcf_conversion_mask = (
        merged["provider_free_cash_flow"].notna()
        & merged["provider_ebitda"].notna()
        & (merged["provider_ebitda"] != 0)
    )
    ev_ebitda_mask = merged["provider_ev_ebitda"].notna()

    merged.loc[margin_mask, "operating__ebitda_margin_ttm__value"] = (
        merged.loc[margin_mask, "provider_ebitda"] / merged.loc[margin_mask, "provider_revenue"]
    )
    merged.loc[margin_mask, "operating__ebitda_margin_ttm__support_mode"] = "exact"
    merged.loc[margin_mask, "operating__ebitda_margin_ttm__fallback_used"] = "refinitiv_direct_ebitda_over_revenue"

    merged.loc[fcf_yield_mask, "market__fcf_yield__value"] = (
        merged.loc[fcf_yield_mask, "provider_free_cash_flow"] / merged.loc[fcf_yield_mask, "provider_market_cap"]
    )
    merged.loc[fcf_yield_mask, "market__fcf_yield__support_mode"] = "exact"
    merged.loc[fcf_yield_mask, "market__fcf_yield__fallback_used"] = "refinitiv_direct_fcf_over_market_cap"

    merged.loc[fcf_conversion_mask, "operating__fcf_conversion__value"] = (
        merged.loc[fcf_conversion_mask, "provider_free_cash_flow"] / merged.loc[fcf_conversion_mask, "provider_ebitda"]
    )
    merged.loc[fcf_conversion_mask, "operating__fcf_conversion__support_mode"] = "exact"
    merged.loc[fcf_conversion_mask, "operating__fcf_conversion__fallback_used"] = "refinitiv_direct_fcf_over_ebitda"

    merged.loc[ev_ebitda_mask, "market__ev_ebitda__value"] = merged.loc[ev_ebitda_mask, "provider_ev_ebitda"]
    merged.loc[ev_ebitda_mask, "market__ev_ebitda__support_mode"] = "exact"
    merged.loc[ev_ebitda_mask, "market__ev_ebitda__fallback_used"] = "refinitiv_direct_ev_ebitda"

    merged.to_parquet(out_parquet, index=False)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_csv, index=False)

    if args.summary_out:
        sample_ids = ["0000006201", "0000003453", "0001611647", "0001637459", "0000909832", "0000003197", "0001639438", "0000007431"]
        sample = {}
        for cid in sample_ids:
            sub = merged[merged["company_id"] == cid]
            if sub.empty:
                continue
            row = sub.iloc[0]
            sample[cid] = {
                "company_name": row["company_name"],
                "provider_instrument": _json_scalar(row.get("Instrument")),
                "provider_company_name": _json_scalar(row.get("Company Common Name")),
                "provider_ebitda": _json_scalar(row.get("provider_ebitda")),
                "provider_revenue": _json_scalar(row.get("provider_revenue")),
                "provider_free_cash_flow": _json_scalar(row.get("provider_free_cash_flow")),
                "provider_cash_and_short_term_investments": _json_scalar(
                    row.get("provider_cash_and_short_term_investments")
                ),
                "provider_market_cap": _json_scalar(row.get("provider_market_cap")),
                "ebitda_margin_ttm": _json_scalar(row.get("operating__ebitda_margin_ttm__value")),
                "fcf_yield": _json_scalar(row.get("market__fcf_yield__value")),
                "fcf_conversion": _json_scalar(row.get("operating__fcf_conversion__value")),
                "ev_ebitda": _json_scalar(row.get("market__ev_ebitda__value")),
                "rating": _json_scalar(row.get("capital_structure__rating_state__rating")),
                "rating_support_mode": _json_scalar(row.get("capital_structure__rating_state__rating_support_mode")),
                "rating_outlook": _json_scalar(row.get("capital_structure__rating_state__outlook")),
                "rating_watchlist": _json_scalar(row.get("capital_structure__rating_state__watchlist")),
                "rating_score": _json_scalar(row.get("capital_structure__rating_state__score__value")),
                "rating_score_support_mode": _json_scalar(row.get("capital_structure__rating_state__score__support_mode")),
            }

        summary = {
            "rows": int(len(merged)),
            "provider_ebitda_coverage": int(merged["provider_ebitda"].notna().sum()),
            "provider_free_cash_flow_coverage": int(merged["provider_free_cash_flow"].notna().sum()),
            "provider_cash_and_short_term_investments_coverage": int(
                merged["provider_cash_and_short_term_investments"].notna().sum()
            ),
            "provider_market_cap_coverage": int(merged["provider_market_cap"].notna().sum()),
            "provider_ev_ebitda_coverage": int(merged["provider_ev_ebitda"].notna().sum()),
            "ebitda_margin_overrides": int(margin_mask.sum()),
            "fcf_yield_overrides": int(fcf_yield_mask.sum()),
            "fcf_conversion_overrides": int(fcf_conversion_mask.sum()),
            "ev_ebitda_overrides": int(ev_ebitda_mask.sum()),
            "operating.ebitda_margin_ttm": _support_counts(merged["operating__ebitda_margin_ttm__support_mode"]),
            "market.fcf_yield": _support_counts(merged["market__fcf_yield__support_mode"]),
            "operating.fcf_conversion": _support_counts(merged["operating__fcf_conversion__support_mode"]),
            "market.ev_ebitda": _support_counts(merged["market__ev_ebitda__support_mode"]),
            "sample": sample,
        }
        if args.ratings_path:
            summary["ratings_path"] = str(ratings_path)
            summary["ratings_rows"] = int(len(ratings_df)) if ratings_df is not None else 0
            summary["rating_index_size"] = int(len(rating_index))
            summary["rating_matches"] = int(rating_match_count)
            summary["capital_structure.rating_state.rating"] = _support_counts(
                merged["capital_structure__rating_state__rating_support_mode"]
            )
            summary["capital_structure.rating_state.score"] = _support_counts(
                merged["capital_structure__rating_state__score__support_mode"]
            )
        Path(args.summary_out).write_text(json.dumps(summary, indent=2))

    print(f"Overlayed Refinitiv EBITDA metrics -> {out_parquet}")


if __name__ == "__main__":
    main()
