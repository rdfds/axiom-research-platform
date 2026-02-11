from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_PATH = ROOT / "data" / "wrds" / "dealscan" / "loanconnector_revolver_facilities.parquet"
DEFAULT_OUT_DIR = ROOT / "data" / "wrds" / "dealscan"


def build_active_revolver_artifact(
    *,
    in_path: Path,
    out_path: Path,
    as_of: str,
    tickers: Optional[str] = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    as_of_ts = duckdb.sql(f"SELECT CAST('{as_of}' AS TIMESTAMP)").fetchone()[0]
    as_of_literal = as_of_ts.strftime("%Y-%m-%d %H:%M:%S")
    columns = set(
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{in_path.as_posix()}')").fetchdf()["column_name"].astype(str).tolist()
    )

    ticker_filter = ""
    if tickers:
        ticker_values = [token.strip().upper() for token in tickers.split(",") if token.strip()]
        if ticker_values:
            quoted = ", ".join("'" + token.replace("'", "''") + "'" for token in ticker_values)
            ticker_filter = f"AND upper(coalesce(ticker, '')) IN ({quoted})"

    def col_expr(name: str, *, cast: Optional[str] = None) -> str:
        if name in columns:
            if cast:
                return f"TRY_CAST({name} AS {cast}) AS {name}"
            return name
        if cast:
            return f"CAST(NULL AS {cast}) AS {name}"
        return f"NULL AS {name}"

    query = f"""
    COPY (
        WITH active AS (
            SELECT
                {col_expr('ticker')},
                {col_expr('borrower_name_norm')},
                {col_expr('parent_norm')},
                {col_expr('company_name_norm')},
                {col_expr('loanconnector_company_id')},
                {col_expr('loanconnector_deal_id')},
                {col_expr('loanconnector_tranche_id')},
                {col_expr('lpc_deal_id')},
                {col_expr('lpc_tranche_id')},
                {col_expr('wrds_package_id')},
                {col_expr('wrds_facility_id')},
                {col_expr('tranche_type')},
                {col_expr('tranche_active_date', cast='TIMESTAMP')},
                {col_expr('tranche_maturity_date', cast='TIMESTAMP')},
                {col_expr('tranche_amount_converted_usd', cast='DOUBLE')},
                {col_expr('max_leverage_ratio')},
                {col_expr('min_interest_coverage_ratio')},
                {col_expr('min_fixed_charge_coverage_ratio')},
                {col_expr('min_current_ratio')},
                {col_expr('all_covenants_financial')},
                ROW_NUMBER() OVER (
                    PARTITION BY coalesce({('loanconnector_tranche_id' if 'loanconnector_tranche_id' in columns else 'NULL')}, {('lpc_tranche_id' if 'lpc_tranche_id' in columns else 'NULL')}, {('wrds_facility_id' if 'wrds_facility_id' in columns else 'NULL')})
                    ORDER BY TRY_CAST(tranche_active_date AS TIMESTAMP) DESC NULLS LAST
                ) AS rn
            FROM read_parquet('{in_path.as_posix()}')
            WHERE (TRY_CAST(tranche_active_date AS TIMESTAMP) IS NULL OR TRY_CAST(tranche_active_date AS TIMESTAMP) <= TIMESTAMP '{as_of_literal}')
              AND (TRY_CAST(tranche_maturity_date AS TIMESTAMP) IS NULL OR TRY_CAST(tranche_maturity_date AS TIMESTAMP) >= TIMESTAMP '{as_of_literal}')
              {ticker_filter}
        )
        SELECT *
        EXCLUDE (rn)
        FROM active
        WHERE rn = 1
    ) TO '{out_path.as_posix()}' (FORMAT PARQUET)
    """
    con.execute(query)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact active WRDS DealScan revolver artifact for a specific as-of date.")
    parser.add_argument("--in-path", default=str(DEFAULT_IN_PATH), help="Input DealScan revolver parquet")
    parser.add_argument("--asof", required=True, help="As-of date, e.g. 2024-12-31")
    parser.add_argument("--tickers", default=None, help="Optional comma-separated tickers to retain")
    parser.add_argument("--out-path", default=None, help="Output parquet path")
    args = parser.parse_args()

    out_path = Path(args.out_path) if args.out_path else (
        DEFAULT_OUT_DIR / f"loanconnector_revolver_facilities_active_{args.asof.replace('-', '_')}.parquet"
    )
    path = build_active_revolver_artifact(
        in_path=Path(args.in_path),
        out_path=out_path,
        as_of=args.asof,
        tickers=args.tickers,
    )
    print(path)


if __name__ == "__main__":
    main()
