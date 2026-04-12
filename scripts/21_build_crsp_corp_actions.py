#!/usr/bin/env python
"""
Build unified CRSP corporate actions table from WRDS pulls.

Inputs:
  data/wrds/crsp/msedist_*.parquet
  data/wrds/crsp/msedelist_*.parquet

Outputs:
  data/curated/corporate_actions_crsp.parquet
  data/curated/corporate_actions_crsp_summary.csv
"""

from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds


CRSP_DIR = Path(__file__).parent.parent / "data" / "wrds" / "crsp"
OUT_DIR = Path(__file__).parent.parent / "data" / "curated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Distcd lookup derived from CRSP documentation (see README / notes in summary output)
CRSP_DISTCD_DOCS = {
    # Spin-off (reorganization)
    3763: {
        "action_type": "spinoff",
        "action_subtype": "reorganization",
        "description": "Issue on file received as a spin-off in reorganization (non-taxable)",
        "source": "CRSP US Stock & Indexes Databases Guide - Flat File Format 1.0",
    },
    # Rights offerings
    4523: {
        "action_type": "rights_offering",
        "action_subtype": "market_value",
        "description": "Rights to buy more of this security at market value (non-taxable)",
        "source": "CRSP US Stock & Indexes Databases Guide - Flat File Format 1.0",
    },
    4533: {
        "action_type": "rights_offering",
        "action_subtype": "indicated_value",
        "description": "Rights to buy more of this security at indicated value (non-taxable)",
        "source": "CRSP US Stock & Indexes Databases Guide - Flat File Format 1.0",
    },
    4563: {
        "action_type": "rights_offering",
        "action_subtype": "non_transferable",
        "description": "Rights to buy more of this security, non-transferable value at ex-date (non-taxable)",
        "source": "CRSP US Stock & Indexes Databases Guide - Flat File Format 1.0",
    },
    4623: {
        "action_type": "rights_offering",
        "action_subtype": "units",
        "description": "Rights to buy units that include this security (non-taxable)",
        "source": "CRSP US Stock & Indexes Databases Guide - Flat File Format 1.0",
    },
    4823: {
        "action_type": "rights_offering",
        "action_subtype": "other_securities",
        "description": "Rights to buy other securities (non-taxable)",
        "source": "CRSP US Stock & Indexes Databases Guide - Flat File Format 1.0",
    },
    4999: {
        "action_type": "rights_offering",
        "action_subtype": "missing_rights_distribution",
        "description": "Missing rights distribution; dividend reinvestment plan tax treatment",
        "source": "CRSP US Stock & Indexes Databases Guide - Flat File Format 1.0",
    },
    # Stock distributions (to avoid misclassifying 57xx as splits)
    5763: {
        "action_type": "stock_distribution",
        "action_subtype": "same_company_other_issue",
        "description": "Stock distribution in different issue of same company (non-taxable)",
        "source": "CRSP US Stock & Indexes Databases Guide - Flat File Format 1.0",
    },
    5773: {
        "action_type": "stock_distribution",
        "action_subtype": "other_class_common",
        "description": "Initial stock distribution of other class of common (same company, on file)",
        "source": "CRSP US Stock & Indexes Databases Guide - Flat File Format 1.0",
    },
}


def map_distcd(row):
    distcd = row.get("distcd")
    facpr = row.get("facpr")
    if pd.isna(distcd):
        return "distribution_other", "unknown"
    distcd = int(distcd)

    if distcd in CRSP_DISTCD_DOCS:
        doc = CRSP_DISTCD_DOCS[distcd]
        return doc["action_type"], doc["action_subtype"]

    # Special / irregular dividend codes within 1xxx
    if distcd in (1262, 1263):
        return "dividend", "irregular"
    if distcd in (1272, 1273):
        return "dividend", "special"
    if distcd == 1292:
        return "dividend", "liquidating"

    if 1000 <= distcd <= 1999:
        return "dividend", "regular"
    # Spin-offs / reorganizations (CRSP 41xx series)
    if 4100 <= distcd <= 4199:
        return "spinoff", f"distcd_{distcd}"
    if 4500 <= distcd <= 4599:
        return "return_of_capital", "distribution"
    if 4000 <= distcd <= 4999:
        return "distribution_other", f"distcd_{distcd}"
    if 5500 <= distcd <= 5599:
        return "split", "forward"
    if 5600 <= distcd <= 5699:
        return "split", "reverse"
    # Use factor as a fallback to identify reverse splits
    if pd.notna(facpr) and facpr < 1:
        return "split", "reverse"

    return "distribution_other", f"distcd_{distcd}"


def map_dlstcd(code):
    if pd.isna(code):
        return "delisting", "unknown"
    code = int(code)
    if 200 <= code <= 299:
        return "delisting", "merger_or_exchange"
    if 300 <= code <= 399:
        return "delisting", "liquidation"
    if 400 <= code <= 499:
        return "delisting", "dropped"
    if 500 <= code <= 599:
        return "delisting", "bankruptcy_or_insufficient"
    if 600 <= code <= 699:
        return "delisting", "foreign_listing"
    if 700 <= code <= 799:
        return "delisting", "ceased_trading"
    return "delisting", f"dlstcd_{code}"


def load_msedist():
    files = sorted(CRSP_DIR.glob("msedist_*.parquet"))
    if not files:
        return pd.DataFrame()

    dataset = ds.dataset(files, format="parquet")
    cols = [
        "permno",
        "permco",
        "distcd",
        "divamt",
        "facpr",
        "facshr",
        "dclrdt",
        "exdt",
        "rcrddt",
        "paydt",
        "acperm",
        "accomp",
        "hexcd",
        "hsiccd",
        "cusip",
    ]
    df = dataset.to_table(columns=cols).to_pandas()
    df["distcd"] = pd.to_numeric(df["distcd"], errors="coerce")
    mapped = df.apply(map_distcd, axis=1, result_type="expand")
    df["action_type"] = mapped[0]
    df["action_subtype"] = mapped[1]
    df["action_code"] = df["distcd"]
    df["action_code_type"] = "distcd"
    df["action_date"] = df["exdt"]
    df["source"] = "wrds_crsp"
    df["source_table"] = "msedist"
    return df


def load_msedelist():
    files = sorted(CRSP_DIR.glob("msedelist_*.parquet"))
    if not files:
        return pd.DataFrame()

    dataset = ds.dataset(files, format="parquet")
    cols = [
        "permno",
        "permco",
        "dlstdt",
        "dlstcd",
        "dlamt",
        "dlret",
        "dlretx",
        "dlprc",
        "dlpdt",
        "acperm",
        "accomp",
        "hexcd",
        "hsiccd",
        "cusip",
    ]
    df = dataset.to_table(columns=cols).to_pandas()
    df["dlstcd"] = pd.to_numeric(df["dlstcd"], errors="coerce")
    mapped = df["dlstcd"].apply(map_dlstcd)
    df["action_type"] = mapped.map(lambda x: x[0])
    df["action_subtype"] = mapped.map(lambda x: x[1])
    df["action_code"] = df["dlstcd"]
    df["action_code_type"] = "dlstcd"
    df["action_date"] = df["dlstdt"]
    df["source"] = "wrds_crsp"
    df["source_table"] = "msedelist"
    return df


def main():
    print("Building unified CRSP corporate actions...")
    dist = load_msedist()
    dlst = load_msedelist()

    if dist.empty and dlst.empty:
        raise SystemExit("No CRSP msedist/msedelist files found.")

    df = pd.concat([dist, dlst], ignore_index=True, sort=False)
    df["action_date"] = pd.to_datetime(df["action_date"], errors="coerce")

    out_path = OUT_DIR / "corporate_actions_crsp.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} rows -> {out_path}")

    summary = (
        df.groupby(["action_type", "action_subtype"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary_path = OUT_DIR / "corporate_actions_crsp_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")

    # Build a distcd lookup table for observed codes
    if not dist.empty:
        distcd_values = (
            dist[["distcd"]]
            .dropna()
            .assign(distcd=lambda d: d["distcd"].astype(int))
            .drop_duplicates()
            .sort_values("distcd")
        )
        lookup_rows = []
        for distcd in distcd_values["distcd"].tolist():
            doc = CRSP_DISTCD_DOCS.get(distcd)
            row = {
                "distcd": distcd,
                "action_type": doc["action_type"] if doc else None,
                "action_subtype": doc["action_subtype"] if doc else None,
                "description": doc["description"] if doc else None,
                "source": doc["source"] if doc else None,
            }
            lookup_rows.append(row)
        lookup = pd.DataFrame(lookup_rows)
        lookup_path = OUT_DIR / "crsp_distcd_lookup.csv"
        lookup.to_csv(lookup_path, index=False)
        print(f"Saved distcd lookup -> {lookup_path}")


if __name__ == "__main__":
    main()
