"""
Ingest WRDS FISD Bond Issues + Ratings CSVs into curated parquet files.

Inputs:
  data/wrds/fisd/fisd_issues.csv.gz
  data/wrds/fisd/fisd_issuers.csv.gz
  data/wrds/fisd/fisd_ratings.csv.gz
  data/wrds/fisd/fisd_redemptions.csv.gz

Outputs:
  data/curated/bond_issuances_fisd.parquet
  data/curated/bond_ratings_fisd.parquet
  data/curated/bond_redemptions_fisd.parquet

Environment:
  FISD_ISSUES_PATH (default: data/wrds/fisd/fisd_issues.csv.gz)
  FISD_ISSUERS_PATH (default: data/wrds/fisd/fisd_issuers.csv.gz)
  FISD_RATINGS_PATH (default: data/wrds/fisd/fisd_ratings.csv.gz)
  FISD_REDEMPTIONS_PATH (default: data/wrds/fisd/fisd_redemptions.csv.gz)
  FISD_OUT_ISSUES (default: data/curated/bond_issuances_fisd.parquet)
  FISD_OUT_RATINGS (default: data/curated/bond_ratings_fisd.parquet)
  FISD_OUT_REDEMPTIONS (default: data/curated/bond_redemptions_fisd.parquet)
"""

import os
from datetime import datetime
from pathlib import Path

import duckdb


DATA_DIR = Path(__file__).parent.parent / "data"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
CIQ_MAP = DATA_DIR / "wrds" / "ciq" / "ciq_identifiers_map.parquet"

ISSUES_PATH = Path(os.getenv("FISD_ISSUES_PATH", DATA_DIR / "wrds" / "fisd" / "fisd_issues.csv.gz"))
ISSUERS_PATH = Path(os.getenv("FISD_ISSUERS_PATH", DATA_DIR / "wrds" / "fisd" / "fisd_issuers.csv.gz"))
RATINGS_PATH = Path(os.getenv("FISD_RATINGS_PATH", DATA_DIR / "wrds" / "fisd" / "fisd_ratings.csv.gz"))
REDEMPTIONS_PATH = Path(os.getenv("FISD_REDEMPTIONS_PATH", DATA_DIR / "wrds" / "fisd" / "fisd_redemptions.csv.gz"))

OUT_ISSUES = Path(os.getenv("FISD_OUT_ISSUES", DATA_DIR / "curated" / "bond_issuances_fisd.parquet"))
OUT_RATINGS = Path(os.getenv("FISD_OUT_RATINGS", DATA_DIR / "curated" / "bond_ratings_fisd.parquet"))
OUT_REDEMPTIONS = Path(os.getenv("FISD_OUT_REDEMPTIONS", DATA_DIR / "curated" / "bond_redemptions_fisd.parquet"))


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


