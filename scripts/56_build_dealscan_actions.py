"""
Build curated loan actions from DealScan facilities.

Inputs:
  data/dealscan_facilities.parquet
  data/fundamentals_quarterly.parquet (for ticker -> gvkey mapping)

Output:
  data/curated/loan_actions_dealscan.parquet
"""

from datetime import datetime
from pathlib import Path

import duckdb


DATA_DIR = Path(__file__).parent.parent / "data"
FACILITIES_PATH = DATA_DIR / "dealscan_facilities.parquet"
FUND_PATH = DATA_DIR / "fundamentals_quarterly.parquet"
NAMES_PATH = DATA_DIR / "wrds" / "crsp" / "msenames_2000-01-01_to_2024-12-31.parquet"
LINK_PATH = DATA_DIR / "wrds" / "crsp" / "ccmxpf_lnkhist.parquet"
CIQ_PATH = DATA_DIR / "wrds" / "ciq" / "ciq_identifiers_map.parquet"
OUT_PATH = DATA_DIR / "curated" / "loan_actions_dealscan.parquet"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    if not FACILITIES_PATH.exists():
        raise FileNotFoundError(f"Missing facilities file: {FACILITIES_PATH}")
    if not FUND_PATH.exists():
        raise FileNotFoundError(f"Missing fundamentals file: {FUND_PATH}")
    if not NAMES_PATH.exists():
        raise FileNotFoundError(f"Missing CRSP names file: {NAMES_PATH}")
    if not LINK_PATH.exists():
        raise FileNotFoundError(f"Missing CRSP link file: {LINK_PATH}")

    log("Building DealScan loan actions with ticker->gvkey mapping...")

    con = duckdb.connect()
    query = f"""
    WITH facilities AS (
        SELECT
            facilityid,
            packageid,
            cast(facilitystartdate as timestamp) AS facilitystartdate,
            facilityenddate,
            facilityamt,
            currency,
            primarypurpose,
            secondarypurpose,
            loantype,
            maturity,
            secured,
            seniority,
            facility_company,
            targetcompany,
            dealactivedate,
            borrower_name,
            dealamount,
            dealpurpose,
            dealstatus,
            salesatclose,
            ticker,
            primarysiccode,
            country,
            sales,
            publicprivate
        FROM read_parquet('{FACILITIES_PATH.as_posix()}')
    ),
    fund AS (
        SELECT gvkey, cast(datadate as timestamp) AS datadate, tic
        FROM read_parquet('{FUND_PATH.as_posix()}')
        WHERE tic IS NOT NULL
    ),
    names AS (
        SELECT
            permno,
            upper(ticker) AS ticker,
            cast(namedt as timestamp) AS namedt,
            cast(nameendt as timestamp) AS nameendt
        FROM read_parquet('{NAMES_PATH.as_posix()}')
        WHERE ticker IS NOT NULL
    ),
    link AS (
        SELECT
            gvkey,
            lpermno AS permno,
            coalesce(cast(linkdt as timestamp), timestamp '1900-01-01') AS linkdt,
            coalesce(cast(linkenddt as timestamp), timestamp '2099-12-31') AS linkenddt
        FROM read_parquet('{LINK_PATH.as_posix()}')
    ),
    ciq AS (
        SELECT
            upper(ticker) AS ticker,
            min(gvkey) AS gvkey
        FROM read_parquet('{CIQ_PATH.as_posix()}')
        WHERE ticker IS NOT NULL
        GROUP BY 1
    ),
    mapped AS (
        SELECT
            f.*,
            fund.gvkey,
            fund.datadate,
            row_number() OVER (
                PARTITION BY f.facilityid
                ORDER BY fund.datadate DESC
            ) AS rn
        FROM facilities f
        JOIN fund
          ON upper(f.ticker) = upper(fund.tic)
         AND fund.datadate <= f.facilitystartdate
    ),
    crsp_map AS (
        SELECT
            f.facilityid,
            link.gvkey,
            row_number() OVER (
                PARTITION BY f.facilityid
                ORDER BY names.namedt DESC
            ) AS rn
        FROM facilities f
        JOIN names
          ON upper(f.ticker) = names.ticker
         AND f.facilitystartdate BETWEEN names.namedt AND names.nameendt
        JOIN link
          ON link.permno = names.permno
         AND f.facilitystartdate BETWEEN link.linkdt AND link.linkenddt
    ),
    base AS (
        SELECT
            f.*,
            m.gvkey AS gvkey_cs,
            crsp.gvkey AS gvkey_crsp,
            ciq.gvkey AS gvkey_ciq
        FROM facilities f
        LEFT JOIN mapped m
          ON f.facilityid = m.facilityid AND m.rn = 1
        LEFT JOIN crsp_map crsp
          ON f.facilityid = crsp.facilityid AND crsp.rn = 1
        LEFT JOIN ciq
          ON upper(f.ticker) = ciq.ticker
    )
    SELECT
        facilityid,
        packageid,
        borrower_name,
        ticker,
        coalesce(gvkey_cs, gvkey_crsp, gvkey_ciq) AS gvkey,
        facilitystartdate AS action_date,
        facilityamt AS amount,
        currency,
        loantype,
        maturity,
        secured,
        seniority,
        primarypurpose,
        secondarypurpose,
        dealpurpose,
        dealamount,
        dealstatus,
        case
            when gvkey_cs is not null then 'compustat_ticker'
            when gvkey_crsp is not null then 'crsp_names'
            when gvkey_ciq is not null then 'ciq_ticker'
            else null
        end as mapping_source,
        CASE
            WHEN lower(coalesce(primarypurpose, dealpurpose, '')) LIKE '%lbo%' THEN 'lbo_financing'
            WHEN lower(coalesce(primarypurpose, dealpurpose, '')) LIKE '%takeover%' THEN 'acquisition_financing'
            WHEN lower(coalesce(primarypurpose, dealpurpose, '')) LIKE '%acquis%' THEN 'acquisition_financing'
            WHEN lower(coalesce(primarypurpose, dealpurpose, '')) LIKE '%merger%' THEN 'acquisition_financing'
            WHEN lower(coalesce(primarypurpose, dealpurpose, '')) LIKE '%dividend%' THEN 'dividend_recap'
            WHEN lower(coalesce(primarypurpose, dealpurpose, '')) LIKE '%recap%' THEN 'dividend_recap'
            WHEN lower(coalesce(primarypurpose, dealpurpose, '')) LIKE '%refin%' THEN 'loan_refinancing'
            WHEN lower(coalesce(primarypurpose, dealpurpose, '')) LIKE '%debt repay%' THEN 'loan_refinancing'
            ELSE 'loan_issuance'
        END AS action_type,
        loantype AS action_subtype,
        'dealscan' AS source
    FROM base
    """

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO '{OUT_PATH.as_posix()}' (FORMAT 'parquet');")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{OUT_PATH.as_posix()}')").fetchone()[0]
    con.close()
    log(f"Saved {n:,} loan actions -> {OUT_PATH}")


if __name__ == "__main__":
    main()
