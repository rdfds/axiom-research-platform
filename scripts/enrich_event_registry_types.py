#!/usr/bin/env python
"""
Enrich EventRegistry with normalized action taxonomy.

Adds:
  - action_family (capital_structure, capital_return, mna, restructuring, governance, other)
  - action_type_norm (normalized to Step 1 spec)
  - mapping_source (action_type | subtype_keyword | unmapped)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", default="data/inputs_layer/event_registry.parquet")
    parser.add_argument("--out-path", default="data/inputs_layer/event_registry_enriched.parquet")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    in_path = ROOT / args.in_path
    out_path = ROOT / args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        print(f"Output exists: {out_path}")
        return

    df = pd.read_parquet(in_path)

    # Base mapping from action_type
    base_map = {
        # Capital structure
        "bond_issuance": ("capital_structure", "debt_issuance"),
        "loan_issuance": ("capital_structure", "debt_issuance"),
        "loan_refinancing": ("capital_structure", "refinancing"),
        "acquisition_financing": ("capital_structure", "debt_issuance"),
        "lbo_financing": ("capital_structure", "debt_issuance"),
        "equity_offering_public_proxy": ("capital_structure", "equity_offering"),
        "equity_offering_private": ("capital_structure", "equity_offering"),
        "rights_offering": ("capital_structure", "equity_offering"),
        "issuer_rating": ("capital_structure", "rating_action"),
        # Capital return
        "buyback": ("capital_return", "buyback"),
        "dividend_regular": ("capital_return", "dividend_regular"),
        "dividend_increase": ("capital_return", "dividend_increase"),
        "dividend_cut": ("capital_return", "dividend_cut"),
        "dividend_initiate": ("capital_return", "dividend_initiate"),
        "dividend_special": ("capital_return", "dividend_special"),
        "dividend_recap": ("capital_return", "dividend_recap"),
        "distribution_other": ("capital_return", "distribution_other"),
        # M&A
        "acquisition": ("mna", "acquisition"),
        "divestiture": ("mna", "divestiture"),
        "spinoff": ("mna", "spinoff"),
        # Restructuring
        "bankruptcy": ("restructuring", "bankruptcy"),
        "delisting": ("restructuring", "delisting"),
        # Governance
        "reverse_split": ("governance", "reverse_split"),
        "stock_split": ("governance", "stock_split"),
        "ticker_change": ("governance", "ticker_change"),
    }

    df["action_family"] = df["action_type"].map({k: v[0] for k, v in base_map.items()})
    df["action_type_norm"] = df["action_type"].map({k: v[1] for k, v in base_map.items()})
    df["mapping_source"] = pd.NA
    df.loc[df["action_family"].notna(), "mapping_source"] = "action_type"

    # Subtype keyword mapping for unmapped rows
    subtype = df.get("action_subtype")
    subtype = subtype.astype("string").str.lower().fillna("")
    mask_unmapped = df["action_family"].isna()

    keyword_map = [
        (r"tender", "capital_structure", "tender_offer"),
        (r"exchange", "capital_structure", "exchange_offer"),
        (r"liability", "capital_structure", "liability_management"),
        (r"refinan", "capital_structure", "refinancing"),
        (r"rating", "capital_structure", "rating_action"),
        (r"activist", "governance", "activist_entry"),
        (r"poison", "governance", "poison_pill"),
        (r"board", "governance", "board_change"),
        (r"bankrupt", "restructuring", "bankruptcy"),
        (r"restruct", "restructuring", "restructuring"),
        (r"impair", "restructuring", "impairment"),
        (r"workforce|layoff|reduction", "restructuring", "workforce_reduction"),
        (r"cost savings|cost program", "restructuring", "cost_savings_program"),
        (r"joint venture|jv", "mna", "joint_venture"),
        (r"asset sale", "mna", "asset_sale"),
        (r"spin", "mna", "spinoff"),
        (r"dividend", "capital_return", "dividend_other"),
        (r"buyback|repurchase", "capital_return", "buyback"),
    ]

    for pattern, family, norm in keyword_map:
        m = mask_unmapped & subtype.str.contains(pattern, regex=True)
        if not m.any():
            continue
        df.loc[m, "action_family"] = family
        df.loc[m, "action_type_norm"] = norm
        df.loc[m, "mapping_source"] = "subtype_keyword"
        mask_unmapped = df["action_family"].isna()

    df.loc[df["mapping_source"].isna(), "mapping_source"] = "unmapped"

    df.to_parquet(out_path, index=False)
    print(f"Wrote enriched EventRegistry -> {out_path}")


if __name__ == "__main__":
    main()
