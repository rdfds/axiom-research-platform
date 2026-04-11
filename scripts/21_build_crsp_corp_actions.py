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


