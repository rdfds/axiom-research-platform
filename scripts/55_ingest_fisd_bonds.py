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


def build_bond_ratings(con: duckdb.DuckDBPyConnection) -> None:
    if not RATINGS_PATH.exists():
        log("Ratings file not found; skipping ratings.")
        return

    log("Building bond ratings (FISD)...")

    query = f"""
    WITH ratings_raw AS (
        SELECT * FROM read_csv_auto(
            '{RATINGS_PATH.as_posix()}',
            union_by_name=true,
            all_varchar=true,
            strict_mode=false,
            ignore_errors=true,
            null_padding=true
        )
    ),
    ratings AS (
        SELECT
            ISSUE_ID,
            ISSUER_ID,
            RATING_TYPE,
            RATING_DATE,
            RATING,
            RATING_STATUS,
            INVESTMENT_GRADE,
            ISSUE_CUSIP,
            COMPLETE_CUSIP,
            OFFERING_DATE,
            MATURITY,
            {_date_expr('RATING_DATE')} AS rating_date,
            {_date_expr('OFFERING_DATE')} AS offering_date,
            {_date_expr('MATURITY')} AS maturity_date,
            substr(coalesce(COMPLETE_CUSIP, ISSUE_CUSIP), 1, 8) AS cusip8
        FROM ratings_raw
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
        r.ISSUE_ID,
        r.ISSUER_ID,
        r.RATING_TYPE,
        r.rating_date,
        r.RATING,
        r.RATING_STATUS,
        r.INVESTMENT_GRADE,
        r.ISSUE_CUSIP,
        r.COMPLETE_CUSIP,
        r.offering_date,
        r.maturity_date,
        ciq.gvkey,
        link.permno
    FROM ratings r
    LEFT JOIN ciq ON ciq.cusip8 = r.cusip8
    LEFT JOIN link
        ON link.gvkey = ciq.gvkey
       AND cast(r.rating_date as timestamp) BETWEEN link.linkdt AND link.linkenddt
    WHERE r.rating_date IS NOT NULL
    """

    OUT_RATINGS.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO '{OUT_RATINGS.as_posix()}' (FORMAT 'parquet');")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{OUT_RATINGS.as_posix()}')").fetchone()[0]
    log(f"Saved {n:,} bond ratings -> {OUT_RATINGS}")


def build_bond_redemptions(con: duckdb.DuckDBPyConnection) -> None:
    if not REDEMPTIONS_PATH.exists():
        log("Redemptions file not found; skipping redemptions.")
        return

    log("Building bond redemptions (FISD)...")

    query = f"""
    WITH red_raw AS (
        SELECT * FROM read_csv_auto(
            '{REDEMPTIONS_PATH.as_posix()}',
            union_by_name=true,
            all_varchar=true,
            strict_mode=false,
            ignore_errors=true,
            null_padding=true
        )
    ),
    red AS (
        SELECT
            ISSUE_ID,
            ISSUER_ID,
            PROSPECTUS_ISSUER_NAME,
            ISSUER_CUSIP,
            ISSUE_CUSIP,
            COMPLETE_CUSIP,
            ISSUE_NAME,
            ACTION_TYPE,
            CALL_DATE,
            CALL_AMOUNT,
            CALL_PRICE,
            MR_DATE,
            MR_PRICE,
            NEXT_CALL_DATE,
            NEXT_CALL_PRICE,
            NEXT_SF_DATE,
            NEXT_SF_AMOUNT,
            MATURITY,
            OFFERING_DATE,
            substr(coalesce(COMPLETE_CUSIP, ISSUE_CUSIP, ISSUER_CUSIP), 1, 8) AS cusip8
        FROM red_raw
    ),
    red2 AS (
        SELECT
            ISSUE_ID,
            ISSUER_ID,
            PROSPECTUS_ISSUER_NAME,
            ISSUER_CUSIP,
            ISSUE_CUSIP,
            COMPLETE_CUSIP,
            ISSUE_NAME,
            ACTION_TYPE,
            {_date_expr('CALL_DATE')} AS call_date,
            try_cast(CALL_AMOUNT as double) AS call_amount,
            try_cast(CALL_PRICE as double) AS call_price,
            {_date_expr('MR_DATE')} AS mr_date,
            try_cast(MR_PRICE as double) AS mr_price,
            {_date_expr('NEXT_CALL_DATE')} AS next_call_date,
            try_cast(NEXT_CALL_PRICE as double) AS next_call_price,
            {_date_expr('NEXT_SF_DATE')} AS next_sf_date,
            try_cast(NEXT_SF_AMOUNT as double) AS next_sf_amount,
            {_date_expr('MATURITY')} AS maturity_date,
            {_date_expr('OFFERING_DATE')} AS offering_date,
            cusip8
        FROM red
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
        r.ISSUE_ID,
        r.ISSUER_ID,
        r.PROSPECTUS_ISSUER_NAME,
        r.ISSUER_CUSIP,
        r.ISSUE_CUSIP,
        r.COMPLETE_CUSIP,
        r.ISSUE_NAME,
        r.ACTION_TYPE,
        -- Use realized redemption dates only; next_call/next_sf are scheduled (future) dates
        coalesce(r.call_date, r.mr_date) AS action_date,
        r.call_amount,
        r.call_price,
        r.mr_price,
        r.next_call_price,
        r.next_sf_amount,
        r.maturity_date,
        r.offering_date,
        ciq.gvkey,
        link.permno,
        coalesce(r.call_amount, r.next_sf_amount) AS amount
    FROM red2 r
    LEFT JOIN ciq ON ciq.cusip8 = r.cusip8
    LEFT JOIN link
        ON link.gvkey = ciq.gvkey
       AND cast(coalesce(r.call_date, r.mr_date, r.next_call_date, r.next_sf_date) as timestamp) BETWEEN link.linkdt AND link.linkenddt
    WHERE coalesce(r.call_date, r.mr_date, r.next_call_date, r.next_sf_date) IS NOT NULL
    """

    OUT_REDEMPTIONS.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO '{OUT_REDEMPTIONS.as_posix()}' (FORMAT 'parquet');")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{OUT_REDEMPTIONS.as_posix()}')").fetchone()[0]
    log(f"Saved {n:,} bond redemptions -> {OUT_REDEMPTIONS}")


def main() -> None:
    con = duckdb.connect()
    build_bond_issuances(con)
    build_bond_ratings(con)
    build_bond_redemptions(con)
    con.close()
    log("Done.")


if __name__ == "__main__":
    main()
