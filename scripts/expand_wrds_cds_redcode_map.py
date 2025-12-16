#!/usr/bin/env python3
"""Expand the partial WRDS CDS RED-code map with safe alias rules."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ABBREVIATIONS = {
    "AMERN": "AMERICAN",
    "AIRLS": "AIRLINES",
    "AWYS": "AIRWAYS",
    "BK": "BANK",
    "BKS": "BANKS",
    "FINL": "FINANCIAL",
    "FIN": "FINANCIAL",
    "HLDGS": "HOLDINGS",
    "INTL": "INTERNATIONAL",
    "TECH": "TECHNOLOGY",
    "TECHS": "TECHNOLOGIES",
    "COMMN": "COMMUNICATIONS",
    "COMMS": "COMMUNICATIONS",
    "SVCS": "SERVICES",
    "SVS": "SERVICES",
    "MFG": "MANUFACTURING",
    "WHSL": "WHOLESALE",
    "PRODS": "PRODUCTS",
    "MTG": "MORTGAGE",
    "PWR": "POWER",
    "GEN": "GENERAL",
    "INDS": "INDUSTRIES",
    "INDL": "INDUSTRIAL",
    "GRP": "GROUP",
    "COS": "COMPANIES",
    "CTRL": "CONTROL",
    "ELEC": "ELECTRIC",
    "ELECN": "ELECTRONICS",
    "ELECS": "ELECTRONICS",
    "AMER": "AMERICAN",
}

STOPWORDS = {
    "THE",
    "CORPORATION",
    "CORP",
    "INCORPORATED",
    "INC",
    "COMPANY",
    "CO",
    "GROUP",
    "HOLDINGS",
    "HOLDING",
    "PLC",
    "LTD",
    "LIMITED",
    "NV",
    "N",
    "V",
    "SA",
    "SPA",
    "AG",
    "SE",
    "LLC",
    "LP",
    "NEW",
    "CLASS",
    "A",
    "B",
    "DE",
    "US",
    "COMPANIES",
}

ALLOWED_EXTRA_TOKENS = {
    "FOODS",
    "HOMES",
    "HOLDCO",
    "MOTORS",
    "ENERGY",
    "PETROLEUM",
    "COMMUNICATIONS",
    "HEALTHCARE",
    "TECHNOLOGY",
    "TECHNOLOGIES",
    "PROPERTIES",
    "BRANDS",
    "FINANCIAL",
    "INDUSTRIES",
    "INDUSTRIAL",
    "AIRWAYS",
    "AIRLINES",
    "PRODUCTS",
    "SYSTEMS",
    "SERVICES",
    "GROUP",
    "FOODSVC",
}


def _company_id_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(10)


def _tokens(value: object) -> list[str]:
    text = "" if pd.isna(value) else str(value).upper()
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    out: list[str] = []
    for token in text.split():
        token = ABBREVIATIONS.get(token, token)
        if len(token) < 3 or token in STOPWORDS or token.isdigit():
            continue
        out.append(token)
    return out


def _build_additions(broad: pd.DataFrame, unresolved: pd.DataFrame) -> pd.DataFrame:
    broad = broad.copy()
    unresolved = unresolved.copy()
    broad["tokens"] = broad["shortname"].map(_tokens)
    unresolved["tokens"] = unresolved["company_name"].map(_tokens)

    additions = []
    for _, row in unresolved.iterrows():
        query = set(row["tokens"])
        if not query:
            continue

        candidates = []
        for _, candidate in broad.iterrows():
            cand_tokens = set(candidate["tokens"])
            if not cand_tokens:
                continue
            if query == cand_tokens:
                candidates.append(("token_exact_abbrev", 0, candidate))
            else:
                extra = cand_tokens - query
                if query.issubset(cand_tokens) and len(extra) == 1 and all(
                    token in ALLOWED_EXTRA_TOKENS for token in extra
                ):
                    candidates.append(("token_subset_one_extra", 1, candidate))

        if not candidates:
            continue

        redcodes = {item[2]["redcode"] for item in candidates}
        if len(redcodes) != 1:
            continue

        candidates.sort(key=lambda item: (item[1], item[2]["shortname"]))
        best = candidates[0]
        additions.append(
            {
                "company_id": row["company_id"],
                "company_name": row["company_name"],
                "equity_ticker": row["equity_ticker"],
                "cds_ticker": best[2]["ticker"],
                "redcode": best[2]["redcode"],
                "shortname": best[2]["shortname"],
                "match_type": best[0],
                "company_tokens": " ".join(row["tokens"]),
                "shortname_tokens": " ".join(best[2]["tokens"]),
            }
        )

    if not additions:
        return pd.DataFrame(
            columns=[
                "company_id",
                "company_name",
                "equity_ticker",
                "cds_ticker",
                "redcode",
                "shortname",
                "match_type",
                "company_tokens",
                "shortname_tokens",
            ]
        )
    return pd.DataFrame(additions).drop_duplicates(subset=["company_id", "redcode"])


def expand_map(
    broad_cds_path: Path,
    partial_map_path: Path,
    unresolved_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    broad = pd.read_csv(broad_cds_path, usecols=["ticker", "shortname", "redcode"]).drop_duplicates()
    partial = pd.read_csv(partial_map_path)
    unresolved = pd.read_csv(unresolved_path)

    partial["company_id"] = _company_id_series(partial["company_id"])
    unresolved["company_id"] = _company_id_series(unresolved["company_id"])

    additions = _build_additions(broad, unresolved)
    additions["company_id"] = _company_id_series(additions["company_id"])

    expanded = pd.concat(
        [
            partial,
            additions[
                ["company_id", "company_name", "equity_ticker", "cds_ticker", "redcode", "shortname", "match_type"]
            ],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["company_id", "redcode"])

    remaining = unresolved[~unresolved["company_id"].isin(set(expanded["company_id"]))].copy()

    summary = {
        "partial_rows": int(len(partial)),
        "new_safe_alias_rows": int(len(additions)),
        "expanded_rows": int(len(expanded)),
        "remaining_unresolved": int(len(remaining)),
        "new_match_types": additions["match_type"].value_counts().to_dict(),
    }
    return additions, expanded, remaining, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-cds-path", type=Path, required=True)
    parser.add_argument("--partial-map-path", type=Path, required=True)
    parser.add_argument("--unresolved-path", type=Path, required=True)
    parser.add_argument("--out-additions", type=Path, required=True)
    parser.add_argument("--out-expanded-map", type=Path, required=True)
    parser.add_argument("--out-remaining-unresolved", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    additions, expanded, remaining, summary = expand_map(
        broad_cds_path=args.broad_cds_path,
        partial_map_path=args.partial_map_path,
        unresolved_path=args.unresolved_path,
    )

    args.out_additions.parent.mkdir(parents=True, exist_ok=True)
    additions.to_csv(args.out_additions, index=False)
    expanded.to_csv(args.out_expanded_map, index=False)
    remaining.to_csv(args.out_remaining_unresolved, index=False)
    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote expanded RED map -> {args.out_expanded_map}")


if __name__ == "__main__":
    main()
