#!/usr/bin/env python3
"""Overlay WRDS CDS spreads onto the flat quantitative comps export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TIER_RANK = {"SNRFOR": 3, "SUBLT2": 2, "SECDOM": 1}


def _company_id_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(10)


def _count_support(series: pd.Series) -> dict[str, int]:
    vc = series.fillna("unsupported").value_counts()
    return {
        "exact": int(vc.get("exact", 0)),
        "proxy_missing_component": int(vc.get("proxy_missing_component", 0)),
        "unsupported": int(vc.get("unsupported", 0)),
    }


def _prepare_cds(cds_path: Path, as_of_date: str) -> pd.DataFrame:
    cds = pd.read_csv(cds_path)
    cds["date"] = pd.to_datetime(cds["date"])
    cds = cds[
        (cds["tenor"] == "5Y")
        & (cds["currency"] == "USD")
        & (cds["date"] <= pd.Timestamp(as_of_date))
        & cds["parspread"].notna()
    ].copy()

    cds["tier_rank"] = cds["tier"].map(TIER_RANK).fillna(0)
    cds["primarycurve_rank"] = (cds["primarycurve"] == "Y").astype(int)
    cds["liq_nonnull"] = cds["curveliquidityscore"].notna().astype(int)

    # Prefer senior curves, then primary curves, then rows with actual liquidity scores.
    cds = cds.sort_values(
        [
            "redcode",
            "date",
            "tier_rank",
            "primarycurve_rank",
            "liq_nonnull",
            "curveliquidityscore",
        ],
        ascending=[True, True, False, False, False, False],
    )
    return cds.groupby(["redcode", "date"], as_index=False).head(1).copy()


def _latest_cds_snapshot(best_daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for redcode, group in best_daily.groupby("redcode"):
        group = group.sort_values("date")
        latest = group.iloc[-1]
        current = float(latest["parspread"])
        percentile = float((group["parspread"] <= current).mean())
        rows.append(
            {
                "redcode": redcode,
                "market__credit_spread_level__value": current,
                "market__credit_spread_level__unit": "spread",
                "market__credit_spread_level__support_mode": "exact",
                "market__credit_spread_level__fallback_used": "wrds_markit_cds_5y_usd_redcode",
                "market__credit_spread_percentile_2y__value": percentile,
                "market__credit_spread_percentile_2y__unit": "percentile_0_1",
                "market__credit_spread_percentile_2y__support_mode": "exact",
                "market__credit_spread_percentile_2y__fallback_used": "wrds_markit_cds_5y_usd_redcode",
                "market__credit_window_proxy__value": 1.0 - percentile,
                "market__credit_window_proxy__unit": "index_0_1",
                "market__credit_window_proxy__support_mode": "proxy_missing_component",
                "market__credit_window_proxy__fallback_used": "one_minus_cds_spread_percentile_2y",
                "market__cds_redcode": redcode,
                "market__cds_ticker": latest["ticker"],
                "market__cds_shortname": latest["shortname"],
                "market__cds_trade_date": latest["date"].strftime("%Y-%m-%d"),
                "market__cds_tier": latest["tier"],
                "market__cds_primarycurve": latest["primarycurve"],
                "market__cds_docclause": latest["docclause"],
                "market__cds_liquidity_score__value": (
                    None if pd.isna(latest["curveliquidityscore"]) else float(latest["curveliquidityscore"])
                ),
                "market__cds_liquidity_score__unit": "score",
                "market__cds_liquidity_score__support_mode": (
                    "exact" if not pd.isna(latest["curveliquidityscore"]) else "unsupported"
                ),
                "market__cds_history_obs__value": int(len(group)),
                "market__cds_history_obs__unit": "days",
                "market__cds_history_obs__support_mode": "exact",
            }
        )
    return pd.DataFrame(rows)


def overlay_cds(flat_path: Path, cds_path: Path, redcode_map_path: Path, as_of_date: str) -> tuple[pd.DataFrame, dict]:
    flat = pd.read_parquet(flat_path)
    prior_exact_ids = set(
        flat.loc[
            flat["market__credit_spread_level__support_mode"] == "exact",
            "company_id",
        ].astype(str)
    )

    mapping = pd.read_csv(redcode_map_path)
    mapping["company_id"] = _company_id_series(mapping["company_id"])

    best_daily = _prepare_cds(cds_path, as_of_date)
    latest = _latest_cds_snapshot(best_daily)

    overlay = mapping.merge(latest, on="redcode", how="left")
    overlay = overlay.dropna(subset=["market__credit_spread_level__value"]).copy()
    overlay = overlay.drop_duplicates(subset=["company_id"])

    flat["company_id"] = _company_id_series(flat["company_id"])
    flat = flat.merge(overlay.drop(columns=["company_name", "equity_ticker", "cds_ticker", "shortname", "match_type"], errors="ignore"), on="company_id", how="left", suffixes=("", "__cds_new"))

    for col in [c for c in flat.columns if c.endswith("__cds_new")]:
        target = col[:-9]
        flat[target] = flat[col].where(flat[col].notna(), flat.get(target))
        flat.drop(columns=[col], inplace=True)

    flat["market__credit_truth_tier"] = "unsupported"
    flat.loc[
        flat["market__credit_spread_level__support_mode"] == "proxy_missing_component",
        "market__credit_truth_tier",
    ] = "proxy"
    flat.loc[
        flat["market__credit_spread_level__support_mode"] == "exact",
        "market__credit_truth_tier",
    ] = "bond_exact"
    flat.loc[flat["market__cds_redcode"].notna(), "market__credit_truth_tier"] = "cds_exact"
    flat["market__credit_truth_tier_rank__value"] = flat["market__credit_truth_tier"].map(
        {
            "unsupported": 0,
            "proxy": 1,
            "bond_exact": 2,
            "cds_exact": 3,
        }
    )
    flat["market__credit_truth_tier_rank__unit"] = "ordinal_0_3"
    flat["market__credit_truth_tier_rank__support_mode"] = "exact"

    cds_ids = set(overlay["company_id"].astype(str))
    summary = {
        "rows": int(len(flat)),
        "cds_redcode_map_rows": int(len(mapping)),
        "cds_daily_rows": int(len(best_daily)),
        "cds_exact_matches": int(len(cds_ids)),
        "prior_exact_credit_spread_rows": int(len(prior_exact_ids)),
        "overlap_with_prior_exact_credit_spread": int(len(cds_ids & prior_exact_ids)),
        "new_exact_cds_beyond_prior_exact": int(len(cds_ids - prior_exact_ids)),
        "market.credit_truth_tier": flat["market__credit_truth_tier"].value_counts(dropna=False).to_dict(),
        "market.credit_spread_level": _count_support(flat["market__credit_spread_level__support_mode"]),
        "market.credit_spread_percentile_2y": _count_support(flat["market__credit_spread_percentile_2y__support_mode"]),
        "market.credit_window_proxy": _count_support(flat["market__credit_window_proxy__support_mode"]),
    }
    return flat, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-path", type=Path, required=True)
    parser.add_argument("--cds-path", type=Path, required=True)
    parser.add_argument("--redcode-map-path", type=Path, required=True)
    parser.add_argument("--out-parquet", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--as-of-date", default="2024-12-31")
    args = parser.parse_args()

    repaired, summary = overlay_cds(
        flat_path=args.flat_path,
        cds_path=args.cds_path,
        redcode_map_path=args.redcode_map_path,
        as_of_date=args.as_of_date,
    )
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_parquet(args.out_parquet, index=False)
    repaired.to_csv(args.out_csv, index=False)
    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote repaired flat quant export -> {args.out_parquet}")


if __name__ == "__main__":
    main()
