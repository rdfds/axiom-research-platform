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


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False)


def build_normalized_outputs(
    *,
    facilities_path: Path,
    company_map_path: Path,
    id_map_path: Path,
    covenants_path: Path,
    out_root: Path,
) -> dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)

    facilities = _read_csv(facilities_path)
    facilities = facilities.rename(columns={col: _slug(col) for col in facilities.columns})
    facilities["borrower_id"] = _normalize_id(facilities["borrower_id"])
    facilities["lpc_deal_id"] = _normalize_id(facilities["lpc_deal_id"])
    facilities["lpc_tranche_id"] = _normalize_id(facilities["lpc_tranche_id"])
    facilities["ticker"] = facilities["ticker"].fillna("").astype(str).str.upper().str.strip()
    facilities["borrower_name_norm"] = facilities["borrower_name"].fillna("").astype(str).str.upper().str.strip()
    facilities["parent_norm"] = facilities["parent"].fillna("").astype(str).str.upper().str.strip()
    for col in ("deal_active_date", "tranche_active_date", "tranche_maturity_date"):
        facilities[col] = _to_datetime(facilities[col])
    for col in (
        "tranche_amount",
        "tranche_amount_converted",
        "all_in_spread_undrawn_bps",
        "annual_fee_bps",
        "commitment_fee_bps",
        "utilization_fee_bps",
        "letter_of_credit",
        "swingline",
    ):
        facilities[col] = _to_float(facilities[col])
    facilities["loanconnector_company_id"] = facilities["borrower_id"]
    facilities["loanconnector_deal_id"] = facilities["lpc_deal_id"]
    facilities["loanconnector_tranche_id"] = facilities["lpc_tranche_id"]
    facilities["tranche_amount_converted_usd"] = facilities["tranche_amount_converted"] * 1_000_000.0
    facilities["letter_of_credit_usd"] = facilities["letter_of_credit"] * 1_000_000.0
    facilities["swingline_usd"] = facilities["swingline"] * 1_000_000.0
    facilities["is_revolver_like"] = _is_revolver_like(facilities["tranche_type"])

    company_map = _read_csv(company_map_path)
    company_map = company_map.rename(columns={col: _slug(col) for col in company_map.columns})
    company_map["loanconnector_company_id"] = _normalize_id(company_map["loanconnector_company_id"])
    company_map["lpc_company_id"] = _normalize_id(company_map["lpc_company_id"])

    id_map = _read_csv(id_map_path)
    id_map = id_map.rename(columns={col: _slug(col) for col in id_map.columns})
    for col in (
        "loanconnector_deal_id",
        "wrds_package_id",
        "loanconnector_tranche_id",
        "wrds_facility_id",
    ):
        id_map[col] = _normalize_id(id_map[col])

    covenants = _read_csv(covenants_path)
    covenants = covenants.rename(columns={col: _slug(col) for col in covenants.columns})
    covenants["lpc_deal_id"] = _normalize_id(covenants["lpc_deal_id"])
    covenants["lpc_tranche_id"] = _normalize_id(covenants["lpc_tranche_id"])
    for col in ("deal_active_date", "deal_input_date", "tranche_active_date"):
        if col in covenants.columns:
            covenants[col] = _to_datetime(covenants[col])

    facilities = facilities.merge(
        company_map[["loanconnector_company_id", "company_name", "lpc_company_id"]],
        on="loanconnector_company_id",
        how="left",
    )
    facilities["company_name_norm"] = facilities["company_name"].fillna("").astype(str).str.upper().str.strip()
    facilities = facilities.merge(
        id_map[
            [
                "loanconnector_deal_id",
                "loanconnector_tranche_id",
                "wrds_package_id",
                "wrds_facility_id",
            ]
        ].drop_duplicates(),
        on=["loanconnector_deal_id", "loanconnector_tranche_id"],
        how="left",
    )
    facilities = facilities.merge(
        covenants.drop_duplicates(subset=["lpc_deal_id", "lpc_tranche_id"], keep="last"),
        on=["lpc_deal_id", "lpc_tranche_id"],
        how="left",
        suffixes=("", "_covenant"),
    )

    facilities_out = out_root / "loanconnector_facilities.parquet"
    revolver_out = out_root / "loanconnector_revolver_facilities.parquet"
    company_map_out = out_root / "loanconnector_company_id_map.parquet"
    id_map_out = out_root / "loanconnector_id_map.parquet"
    covenants_out = out_root / "loanconnector_financial_covenants.parquet"

    facilities.to_parquet(facilities_out, index=False)
    facilities[facilities["is_revolver_like"]].copy().to_parquet(revolver_out, index=False)
    company_map.to_parquet(company_map_out, index=False)
    id_map.to_parquet(id_map_out, index=False)
    covenants.to_parquet(covenants_out, index=False)

    return {
        "facilities": facilities_out,
        "revolver_facilities": revolver_out,
        "company_map": company_map_out,
        "id_map": id_map_out,
        "covenants": covenants_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize WRDS DealScan / LoanConnector downloads.")
    parser.add_argument("--facilities-path", required=True, help="Path to WRDS DealScan main CSV/CSV.GZ export")
    parser.add_argument("--company-map-path", required=True, help="Path to lpc_loanconnector_company_id_map.csv")
    parser.add_argument("--id-map-path", required=True, help="Path to wrds_loanconnector_ids.csv")
    parser.add_argument("--covenants-path", required=True, help="Path to wrds_financial_covenants.csv")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output directory for parquet artifacts")
    args = parser.parse_args()

    outputs = build_normalized_outputs(
        facilities_path=Path(args.facilities_path),
        company_map_path=Path(args.company_map_path),
        id_map_path=Path(args.id_map_path),
        covenants_path=Path(args.covenants_path),
        out_root=Path(args.out_root),
    )
    print("Saved normalized WRDS DealScan artifacts:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
