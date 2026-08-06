#!/usr/bin/env python3
"""Overlay USD public-bond maturity schedule lower bounds onto a flat quantitative export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-path", required=True, help="Input flat parquet export")
    parser.add_argument("--entity-identifier-path", required=True, help="Entity identifier parquet")
    parser.add_argument("--bond-issuances-path", required=True, help="FISD bond issuances parquet")
    parser.add_argument("--bond-redemptions-path", help="Optional FISD bond redemptions parquet")
    parser.add_argument("--as-of-date", default="2024-12-31", help="As-of date in YYYY-MM-DD")
    parser.add_argument("--out-parquet", required=True, help="Output parquet path")
    parser.add_argument("--out-csv", help="Optional output CSV path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


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


def build_public_bond_maturity_overlay(
    *,
    flat_path: Path,
    entity_identifier_path: Path,
    bond_issuances_path: Path,
    bond_redemptions_path: Path | None,
    as_of_date: str,
) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of_date).normalize()
    end_12m = as_of_ts + pd.DateOffset(years=1)
    end_24m = as_of_ts + pd.DateOffset(years=2)

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
            "ISSUE_ID",
            "permno",
            "offering_date",
            "maturity_date",
            "amount",
            "offering_amt_k",
            "currency",
            "CONVERTIBLE",
            "ASSET_BACKED",
            "PERPETUAL",
            "PRIVATE_PLACEMENT",
        ],
    ).copy()
    issues["permno"] = pd.to_numeric(issues["permno"], errors="coerce")
    issues["offering_date"] = pd.to_datetime(issues["offering_date"], errors="coerce").dt.normalize()
    issues["maturity_date"] = pd.to_datetime(issues["maturity_date"], errors="coerce").dt.normalize()
    issues["issue_amount"] = pd.to_numeric(issues["amount"], errors="coerce")
    missing_amount = issues["issue_amount"].isna()
    issues.loc[missing_amount, "issue_amount"] = (
        pd.to_numeric(issues.loc[missing_amount, "offering_amt_k"], errors="coerce") * 1000.0
    )

    eligible = (
        issues.merge(identifiers, on="permno", how="inner")
        .merge(companies, on="company_id", how="inner")
        .loc[
            lambda df: df["offering_date"].notna()
            & df["maturity_date"].notna()
            & df["issue_amount"].notna()
            & (df["offering_date"] <= as_of_ts)
            & (df["maturity_date"] > as_of_ts)
            & df["currency"].fillna("N").isin(["N", "USD"])
            & df["CONVERTIBLE"].fillna("N").eq("N")
            & df["ASSET_BACKED"].fillna("N").eq("N")
            & df["PERPETUAL"].fillna("N").eq("N")
            & df["PRIVATE_PLACEMENT"].fillna("N").eq("N"),
            ["company_id", "ISSUE_ID", "maturity_date", "issue_amount"],
        ]
        .drop_duplicates()
    )

    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "public_bond_outstanding",
                "public_bond_issue_count",
                "public_bond_due_0_12m",
                "public_bond_due_12_24m",
            ]
        )

    if bond_redemptions_path is not None and Path(bond_redemptions_path).exists():
        redemptions = pd.read_parquet(
            bond_redemptions_path,
            columns=["ISSUE_ID", "action_date", "amount"],
        ).copy()
        redemptions["action_date"] = pd.to_datetime(redemptions["action_date"], errors="coerce").dt.normalize()
        # Curated FISD redemptions preserve the source amount in thousands.
        redemptions["redeemed_amount"] = pd.to_numeric(redemptions["amount"], errors="coerce") * 1000.0
        redeemed = (
            redemptions.loc[
                lambda df: df["action_date"].notna()
                & (df["action_date"] <= as_of_ts)
                & df["redeemed_amount"].notna(),
                ["ISSUE_ID", "redeemed_amount"],
            ]
            .groupby("ISSUE_ID", as_index=False)["redeemed_amount"]
            .sum()
        )
        eligible = eligible.merge(redeemed, on="ISSUE_ID", how="left")
        eligible["redeemed_amount"] = eligible["redeemed_amount"].fillna(0.0)
    else:
        eligible["redeemed_amount"] = 0.0

    eligible["outstanding_amount"] = (eligible["issue_amount"] - eligible["redeemed_amount"]).clip(lower=0.0)
    eligible = eligible[eligible["outstanding_amount"] > 0].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "public_bond_outstanding",
                "public_bond_issue_count",
                "public_bond_due_0_12m",
                "public_bond_due_12_24m",
            ]
        )

    totals = eligible.groupby("company_id", as_index=False).agg(
        public_bond_outstanding=("outstanding_amount", "sum"),
        public_bond_issue_count=("ISSUE_ID", "nunique"),
    )
    due_0_12m = (
        eligible.loc[eligible["maturity_date"] <= end_12m, ["company_id", "outstanding_amount"]]
        .groupby("company_id", as_index=False)["outstanding_amount"]
        .sum()
        .rename(columns={"outstanding_amount": "public_bond_due_0_12m"})
    )
    due_12_24m = (
        eligible.loc[
            (eligible["maturity_date"] > end_12m) & (eligible["maturity_date"] <= end_24m),
            ["company_id", "outstanding_amount"],
        ]
        .groupby("company_id", as_index=False)["outstanding_amount"]
        .sum()
        .rename(columns={"outstanding_amount": "public_bond_due_12_24m"})
    )

    overlay = companies.merge(totals, on="company_id", how="left")
    overlay = overlay.merge(due_0_12m, on="company_id", how="left")
    overlay = overlay.merge(due_12_24m, on="company_id", how="left")

    has_public_bonds = overlay["public_bond_outstanding"].notna()
    overlay.loc[has_public_bonds, "public_bond_due_0_12m"] = overlay.loc[has_public_bonds, "public_bond_due_0_12m"].fillna(0.0)
    overlay.loc[has_public_bonds, "public_bond_due_12_24m"] = overlay.loc[has_public_bonds, "public_bond_due_12_24m"].fillna(0.0)
    return overlay


def main() -> None:
    args = parse_args()

    flat_path = Path(args.flat_path)
    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    merged = pd.read_parquet(flat_path).copy()
    merged["company_id"] = merged["company_id"].astype(str)

    overlay = build_public_bond_maturity_overlay(
        flat_path=flat_path,
        entity_identifier_path=Path(args.entity_identifier_path),
        bond_issuances_path=Path(args.bond_issuances_path),
        bond_redemptions_path=Path(args.bond_redemptions_path) if args.bond_redemptions_path else None,
        as_of_date=args.as_of_date,
    )
    overlay["company_id"] = overlay["company_id"].astype(str)
    merged = merged.merge(overlay, on="company_id", how="left")

    # Raw public-bond schedule transparency columns.
    raw_cols = [
        "capital_structure__public_bond_outstanding__value",
        "capital_structure__public_bond_outstanding__unit",
        "capital_structure__public_bond_outstanding__support_mode",
        "capital_structure__public_bond_issue_count__value",
        "capital_structure__public_bond_issue_count__unit",
        "capital_structure__public_bond_issue_count__support_mode",
        "capital_structure__public_bond_due_0_12m__value",
        "capital_structure__public_bond_due_0_12m__unit",
        "capital_structure__public_bond_due_0_12m__support_mode",
        "capital_structure__public_bond_due_12_24m__value",
        "capital_structure__public_bond_due_12_24m__unit",
        "capital_structure__public_bond_due_12_24m__support_mode",
    ]
    merged = merged.drop(columns=[c for c in raw_cols if c in merged.columns], errors="ignore")

    has_public_schedule = merged["public_bond_outstanding"].notna()
    merged["capital_structure__public_bond_outstanding__value"] = merged["public_bond_outstanding"]
    merged["capital_structure__public_bond_outstanding__unit"] = "usd"
    merged["capital_structure__public_bond_outstanding__support_mode"] = has_public_schedule.map(
        {True: "exact", False: "unsupported"}
    )
    merged["capital_structure__public_bond_issue_count__value"] = merged["public_bond_issue_count"]
    merged["capital_structure__public_bond_issue_count__unit"] = "count"
    merged["capital_structure__public_bond_issue_count__support_mode"] = has_public_schedule.map(
        {True: "exact", False: "unsupported"}
    )
    merged["capital_structure__public_bond_due_0_12m__value"] = merged["public_bond_due_0_12m"]
    merged["capital_structure__public_bond_due_0_12m__unit"] = "usd"
    merged["capital_structure__public_bond_due_0_12m__support_mode"] = has_public_schedule.map(
        {True: "exact", False: "unsupported"}
    )
    merged["capital_structure__public_bond_due_12_24m__value"] = merged["public_bond_due_12_24m"]
    merged["capital_structure__public_bond_due_12_24m__unit"] = "usd"
    merged["capital_structure__public_bond_due_12_24m__support_mode"] = has_public_schedule.map(
        {True: "exact", False: "unsupported"}
    )

    if "capital_structure__debt_due_12_24m__value" not in merged.columns:
        merged["capital_structure__debt_due_12_24m__value"] = np.nan
    if "capital_structure__debt_due_12_24m__unit" not in merged.columns:
        merged["capital_structure__debt_due_12_24m__unit"] = "usd"
    if "capital_structure__debt_due_12_24m__support_mode" not in merged.columns:
        merged["capital_structure__debt_due_12_24m__support_mode"] = "unsupported"
    if "capital_structure__debt_due_12_24m__fallback_used" not in merged.columns:
        merged["capital_structure__debt_due_12_24m__fallback_used"] = None

    existing_due_0_support = merged["capital_structure__debt_due_0_12m__support_mode"].fillna("unsupported")
    existing_due_12_support = merged["capital_structure__debt_due_12_24m__support_mode"].fillna("unsupported")

    due_0_proxy_mask = (
        existing_due_0_support.eq("unsupported")
        & has_public_schedule
        & merged["public_bond_due_0_12m"].fillna(0.0).gt(0)
    )
    merged.loc[due_0_proxy_mask, "capital_structure__debt_due_0_12m__value"] = merged.loc[
        due_0_proxy_mask, "public_bond_due_0_12m"
    ]
    merged.loc[due_0_proxy_mask, "capital_structure__debt_due_0_12m__unit"] = "usd"
    merged.loc[due_0_proxy_mask, "capital_structure__debt_due_0_12m__support_mode"] = "proxy_missing_component"
    merged.loc[due_0_proxy_mask, "capital_structure__debt_due_0_12m__fallback_used"] = (
        "fisd_usd_public_bond_maturity_due_0_12m_lower_bound"
    )

    due_12_proxy_mask = (
        existing_due_12_support.eq("unsupported")
        & has_public_schedule
        & merged["public_bond_due_12_24m"].fillna(0.0).gt(0)
    )
    merged.loc[due_12_proxy_mask, "capital_structure__debt_due_12_24m__value"] = merged.loc[
        due_12_proxy_mask, "public_bond_due_12_24m"
    ]
    merged.loc[due_12_proxy_mask, "capital_structure__debt_due_12_24m__unit"] = "usd"
    merged.loc[due_12_proxy_mask, "capital_structure__debt_due_12_24m__support_mode"] = "proxy_missing_component"
    merged.loc[due_12_proxy_mask, "capital_structure__debt_due_12_24m__fallback_used"] = (
        "fisd_usd_public_bond_maturity_due_12_24m_lower_bound"
    )

    due_0_supported = merged["capital_structure__debt_due_0_12m__support_mode"].fillna("unsupported").ne("unsupported")
    due_0_lower_bound = np.where(
        due_0_supported,
        pd.to_numeric(merged["capital_structure__debt_due_0_12m__value"], errors="coerce"),
        np.where(has_public_schedule, merged["public_bond_due_0_12m"].fillna(0.0), np.nan),
    )
    due_12_lower_bound = np.where(
        merged["capital_structure__debt_due_12_24m__support_mode"].fillna("unsupported").ne("unsupported"),
        pd.to_numeric(merged["capital_structure__debt_due_12_24m__value"], errors="coerce"),
        np.where(has_public_schedule, merged["public_bond_due_12_24m"].fillna(0.0), 0.0),
    )
    denominator = pd.to_numeric(
        merged["capital_structure__debt_like_obligations_normalized__value"],
        errors="coerce",
    )
    ratio_update_mask = has_public_schedule & denominator.notna() & denominator.gt(0) & pd.notna(due_0_lower_bound)
    ratio_values = (pd.Series(due_0_lower_bound, index=merged.index) + pd.Series(due_12_lower_bound, index=merged.index)) / denominator

    existing_ratio_cols = [
        "capital_structure__maturity_wall_ratio_24m__value",
        "capital_structure__maturity_wall_ratio_24m__unit",
        "capital_structure__maturity_wall_ratio_24m__support_mode",
        "capital_structure__maturity_wall_ratio_24m__fallback_used",
    ]
    for col in existing_ratio_cols:
        if col not in merged.columns:
            merged[col] = np.nan if col.endswith("__value") else None
    merged.loc[ratio_update_mask, "capital_structure__maturity_wall_ratio_24m__value"] = ratio_values[ratio_update_mask]
    merged.loc[ratio_update_mask, "capital_structure__maturity_wall_ratio_24m__unit"] = "ratio"
    merged.loc[ratio_update_mask, "capital_structure__maturity_wall_ratio_24m__support_mode"] = "proxy_missing_component"
    merged.loc[ratio_update_mask, "capital_structure__maturity_wall_ratio_24m__fallback_used"] = (
        "statement_and_fisd_usd_public_bond_24m_lower_bound_over_debt_like_obligations"
    )

    merged = merged.drop(
        columns=[
            "public_bond_outstanding",
            "public_bond_issue_count",
            "public_bond_due_0_12m",
            "public_bond_due_12_24m",
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
                "company_name": row.get("company_name"),
                "public_bond_outstanding": _json_scalar(row.get("capital_structure__public_bond_outstanding__value")),
                "public_bond_due_0_12m": _json_scalar(row.get("capital_structure__public_bond_due_0_12m__value")),
                "public_bond_due_12_24m": _json_scalar(row.get("capital_structure__public_bond_due_12_24m__value")),
                "debt_due_0_12m": _json_scalar(row.get("capital_structure__debt_due_0_12m__value")),
                "debt_due_0_12m_support_mode": _json_scalar(row.get("capital_structure__debt_due_0_12m__support_mode")),
                "debt_due_12_24m": _json_scalar(row.get("capital_structure__debt_due_12_24m__value")),
                "debt_due_12_24m_support_mode": _json_scalar(row.get("capital_structure__debt_due_12_24m__support_mode")),
                "maturity_wall_ratio_24m": _json_scalar(row.get("capital_structure__maturity_wall_ratio_24m__value")),
                "maturity_wall_ratio_24m_support_mode": _json_scalar(
                    row.get("capital_structure__maturity_wall_ratio_24m__support_mode")
                ),
            }

        summary = {
            "rows": int(len(merged)),
            "public_bond_schedule_matches": int(has_public_schedule.sum()),
            "proxy_due_0_12m_matches": int(due_0_proxy_mask.sum()),
            "proxy_due_12_24m_matches": int(due_12_proxy_mask.sum()),
            "maturity_ratio_public_bond_matches": int(ratio_update_mask.sum()),
            "capital_structure.debt_due_0_12m": _support_counts(merged["capital_structure__debt_due_0_12m__support_mode"]),
            "capital_structure.debt_due_12_24m": _support_counts(merged["capital_structure__debt_due_12_24m__support_mode"]),
            "capital_structure.maturity_wall_ratio_24m": _support_counts(
                merged["capital_structure__maturity_wall_ratio_24m__support_mode"]
            ),
            "sample": sample,
        }
        Path(args.summary_out).write_text(json.dumps(summary, indent=2))

    print(f"Overlayed public-bond maturity lower bounds -> {out_parquet}")


if __name__ == "__main__":
    main()
