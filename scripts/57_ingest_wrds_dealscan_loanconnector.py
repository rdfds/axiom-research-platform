"""
Normalize WRDS DealScan / LoanConnector exports into stable local parquet files.

Inputs:
  - main facility export CSV/CSV.GZ from the WRDS DealScan web query
  - lpc_loanconnector_company_id_map.csv
  - wrds_loanconnector_ids.csv
  - wrds_financial_covenants.csv

Outputs:
  - data/wrds/dealscan/loanconnector_facilities.parquet
  - data/wrds/dealscan/loanconnector_revolver_facilities.parquet
  - data/wrds/dealscan/loanconnector_company_id_map.parquet
  - data/wrds/dealscan/loanconnector_id_map.parquet
  - data/wrds/dealscan/loanconnector_financial_covenants.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "data" / "wrds" / "dealscan"


def _slug(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "").strip())
    return re.sub(r"_+", "_", text).strip("_").lower()


def _normalize_id(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    raw = raw.mask(raw.isin({"", "nan", "None", "<NA>"}))
    raw = raw.str.replace(r"\.0$", "", regex=True)
    return raw


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _to_float(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .replace({"": None, "nan": None, "None": None, "<NA>": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _is_revolver_like(series: pd.Series) -> pd.Series:
    pattern = re.compile(
        r"revolv|line\s*(?:>=|<)?|364-day|credit facility|asset[- ]based|abl|rcf|swingline",
        re.I,
    )
    return series.fillna("").astype(str).str.contains(pattern)


