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


def _date_expr(col: str) -> str:
    # Handle common FISD date formats.
    return (
        f"coalesce("
        f"try_strptime({col}, '%Y-%m-%d'),"
        f"try_strptime({col}, '%m/%d/%Y'),"
        f"try_strptime({col}, '%Y%m%d')"
        f")"
    )


def build_bond_issuances(con: duckdb.DuckDBPyConnection) -> None:
    if not ISSUES_PATH.exists():
        raise FileNotFoundError(f"Missing issues file: {ISSUES_PATH}")
    if not ISSUERS_PATH.exists():
        raise FileNotFoundError(f"Missing issuers file: {ISSUERS_PATH}")

    log("Building bond issuances (FISD)...")

    query = f"""
    WITH issues_raw AS (
        SELECT * FROM read_csv_auto('{ISSUES_PATH.as_posix()}', union_by_name=true, all_varchar=true)
    ),
    issues AS (
        SELECT
            ISSUE_ID,
            ISSUER_ID,
            PROSPECTUS_ISSUER_NAME,
            ISSUER_CUSIP,
            ISSUE_CUSIP,
            COMPLETE_CUSIP,
            ISSUE_NAME,
            CUSIP_NAME,
            ISIN,
            SEDOL,
            OFFERING_DATE,
            OFFERING_AMT,
            PRINCIPAL_AMT,
            CURRENCY,
            FOREIGN_CURRENCY,
            MATURITY,
            COUPON,
            COUPON_TYPE,
            SECURITY_LEVEL,
            BOND_TYPE,
            CONVERTIBLE,
            PRIVATE_PLACEMENT,
            RULE_144A,
            ASSET_BACKED,
            PERPETUAL
        FROM issues_raw
    ),
    issues2 AS (
        SELECT
            ISSUE_ID,
            ISSUER_ID,
            PROSPECTUS_ISSUER_NAME,
            ISSUER_CUSIP,
            ISSUE_CUSIP,
            COMPLETE_CUSIP,
            ISSUE_NAME,
            CUSIP_NAME,
            ISIN,
            SEDOL,
            {_date_expr('OFFERING_DATE')} AS offering_date,
            {_date_expr('MATURITY')} AS maturity_date,
            try_cast(OFFERING_AMT as double) AS offering_amt_k,
            try_cast(PRINCIPAL_AMT as double) AS principal_amt,
            CURRENCY,
            FOREIGN_CURRENCY,
            try_cast(COUPON as double) AS coupon,
            COUPON_TYPE,
            SECURITY_LEVEL,
            BOND_TYPE,
            CONVERTIBLE,
            PRIVATE_PLACEMENT,
            RULE_144A,
            ASSET_BACKED,
            PERPETUAL,
            coalesce(COMPLETE_CUSIP, ISSUE_CUSIP, ISSUER_CUSIP) AS cusip_raw,
            substr(coalesce(COMPLETE_CUSIP, ISSUE_CUSIP, ISSUER_CUSIP), 1, 8) AS cusip8
        FROM issues
    ),
    issuers AS (
        SELECT
            ISSUER_ID,
            LEGAL_NAME,
            COUNTRY_DOMICILE,
            SIC_CODE,
            NAICS_CODE
        FROM read_csv_auto('{ISSUERS_PATH.as_posix()}', union_by_name=true, all_varchar=true)
    ),
    ciq AS (
        SELECT gvkey, cusip8 FROM read_parquet('{CIQ_MAP.as_posix()}')
    ),
    link AS (
        SELECT
            gvkey,
            lpermno AS permno,
            coalesce(cast(linkdt as timestamp), timestamp '1900-01-01') AS linkdt,
            coalesce(cast(linkenddt as timestamp), timestamp '2099-12-31') AS linkenddt
        FROM read_parquet('{(CRSP_DIR / "ccmxpf_lnkhist.parquet").as_posix()}')
    )
    SELECT
        i.ISSUE_ID,
        i.ISSUER_ID,
        i.PROSPECTUS_ISSUER_NAME,
        i.ISSUER_CUSIP,
        i.ISSUE_CUSIP,
        i.COMPLETE_CUSIP,
        i.ISSUE_NAME,
        i.CUSIP_NAME,
        i.ISIN,
        i.SEDOL,
        i.offering_date,
        i.maturity_date,
        i.offering_amt_k,
        i.principal_amt,
        coalesce(i.CURRENCY, i.FOREIGN_CURRENCY) AS currency,
        i.coupon,
        i.COUPON_TYPE,
        i.SECURITY_LEVEL,
        i.BOND_TYPE,
        i.CONVERTIBLE,
        i.PRIVATE_PLACEMENT,
        i.RULE_144A,
        i.ASSET_BACKED,
        i.PERPETUAL,
        iss.LEGAL_NAME,
        iss.COUNTRY_DOMICILE,
        iss.SIC_CODE,
        iss.NAICS_CODE,
        ciq.gvkey,
        link.permno,
        case
            when i.offering_amt_k is not null then i.offering_amt_k * 1000
            else i.principal_amt
        end AS amount
    FROM issues2 i
    LEFT JOIN issuers iss ON iss.ISSUER_ID = i.ISSUER_ID
    LEFT JOIN ciq ON ciq.cusip8 = i.cusip8
    LEFT JOIN link
        ON link.gvkey = ciq.gvkey
       AND i.offering_date BETWEEN link.linkdt AND link.linkenddt
    WHERE i.offering_date IS NOT NULL
    """

    OUT_ISSUES.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO '{OUT_ISSUES.as_posix()}' (FORMAT 'parquet');")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{OUT_ISSUES.as_posix()}')").fetchone()[0]
    log(f"Saved {n:,} bond issuances -> {OUT_ISSUES}")


