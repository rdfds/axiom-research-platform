#!/usr/bin/env python3
"""Overlay USD public-bond maturity schedule lower bounds onto a flat quantitative export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def parse_args() :
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


