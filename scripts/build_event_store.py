#!/usr/bin/env python
"""
Build Canonical Event Store with lifecycle scaffolding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", default="data/inputs_layer/event_registry_enriched.parquet")
    parser.add_argument("--mna-path", default="data/warehouse/warehouse_mna_deals.parquet")
    parser.add_argument("--enrich-mna-params", action="store_true")
    parser.add_argument("--corp-actions-path", default="data/curated/corporate_actions_master.parquet")
    parser.add_argument("--enrich-corp-params", action="store_true")
    parser.add_argument("--out-path", default="data/inputs_layer/event_store.parquet")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    in_path = ROOT / args.in_path
    out_path = ROOT / args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        print(f"Output exists: {out_path}")
        return

    df = pd.read_parquet(in_path)

    # Build canonical fields
    announced_at = pd.to_datetime(df.get("announcement_date"), errors="coerce", utc=True)
    effective_at = pd.to_datetime(df.get("effective_date"), errors="coerce", utc=True)
    status = df.get("status").astype("string").str.lower()

    closed_at = pd.to_datetime(df.get("closed_date"), errors="coerce", utc=True) if "closed_date" in df.columns else pd.NaT
    withdrawn_at = (
        pd.to_datetime(df.get("withdrawn_date"), errors="coerce", utc=True)
        if "withdrawn_date" in df.columns
        else pd.NaT
    )

    # Fill lifecycle timestamps from status if explicit dates not provided
    if isinstance(closed_at, pd.Series):
        closed_at = closed_at.where(~status.isin(["closed", "completed", "consummated"]), effective_at)
    else:
        closed_at = effective_at.where(status.isin(["closed", "completed", "consummated"]))

    if isinstance(withdrawn_at, pd.Series):
        withdrawn_at = withdrawn_at.where(
            ~status.isin(["withdrawn", "terminated", "canceled", "cancelled", "abandoned"]),
            effective_at.combine_first(announced_at),
        )
    else:
        withdrawn_at = effective_at.combine_first(announced_at).where(
            status.isin(["withdrawn", "terminated", "canceled", "cancelled", "abandoned"])
        )

    event_type = df.get("action_type_norm", df.get("action_type")).astype("string")
    event_type = event_type.fillna(df.get("action_type")).fillna("unknown")

    out = pd.DataFrame(
        {
            "event_id": df["event_id"].astype("string"),
            "company_id": df["company_id"].astype("string"),
            "event_type": event_type,
            "event_subtype": df.get("action_subtype"),
            "params": df.get("parameters"),
            "status": df.get("status"),
            "announced_at": announced_at,
            "effective_at": effective_at,
            "closed_at": closed_at,
            "withdrawn_at": withdrawn_at,
            "evidence": df.get("evidence_links"),
            "created_at": pd.to_datetime(df.get("ingested_at"), errors="coerce", utc=True),
            # keep source_type for enrichment joins (extra fields allowed by schema)
            "source_type": df.get("source_type"),
            # keep raw_pointer for corporate-actions joins
            "raw_pointer": df.get("raw_pointer"),
        }
    )

    if args.enrich_mna_params:
        mna_path = ROOT / args.mna_path
        if not mna_path.exists():
            raise FileNotFoundError(f"Missing M&A file: {mna_path}")
        con = duckdb.connect()
        # only need event_id + source_type for M&A enrichment
        con.register("events_base", out[["event_id", "source_type"]])
        query = f"""
        WITH mna AS (
            SELECT
                deal_id,
                try_cast(acquirer_company_id AS BIGINT) AS acquirer_company_id,
                try_cast(target_company_id AS BIGINT) AS target_company_id,
                CAST(announcement_date AS TIMESTAMPTZ) AS announcement_date,
                try_cast(close_date AS TIMESTAMPTZ) AS close_date,
                deal_value,
                consideration_type,
                deal_type,
                status,
                target_name,
                acquiror_name
            FROM read_parquet('{mna_path.as_posix()}')
            WHERE acquirer_company_id IS NOT NULL AND announcement_date IS NOT NULL
        ),
        events_mna AS (
            SELECT
                e.event_id,
                regexp_extract(e.event_id, 'evt_deal_id:([0-9]+)', 1) AS deal_id
            FROM events_base e
            WHERE e.source_type IN ('refinitiv_mna', 'refinitiv_mna_target')
        ),
        joined AS (
            SELECT
                e.*,
                m.*,
                row_number() OVER (
                    PARTITION BY e.event_id
                    ORDER BY m.deal_value DESC NULLS LAST, m.close_date DESC NULLS LAST
                ) AS rn
            FROM events_mna e
            LEFT JOIN mna m
              ON e.deal_id = m.deal_id
        )
        SELECT
            event_id,
            CASE
                WHEN deal_id IS NOT NULL THEN
                    struct_pack(
                        deal_id := deal_id,
                        deal_value := deal_value,
                        consideration_type := consideration_type,
                        deal_type := deal_type,
                        status := status,
                        target_name := target_name,
                        acquiror_name := acquiror_name,
                        target_company_id := target_company_id,
                        acquirer_company_id := acquirer_company_id,
                        close_date := close_date
                    )
                ELSE NULL
            END AS params,
            NULL AS source_type
        FROM joined
        WHERE rn = 1 OR rn IS NULL
        """
        mna_params = con.execute(query).df()
        if not mna_params.empty:
            out = out.merge(mna_params[["event_id", "params"]], on="event_id", how="left", suffixes=("", "_mna"))
            out["params"] = out["params_mna"].combine_first(out["params"])
            out = out.drop(columns=["params_mna"])

    if args.enrich_corp_params:
        corp_path = ROOT / args.corp_actions_path
        if not corp_path.exists():
            raise FileNotFoundError(f"Missing corporate actions file: {corp_path}")
        con = duckdb.connect()
        con.register("events_base", out[["event_id", "source_type", "raw_pointer"]])
        query = f"""
        WITH events_ca AS (
            SELECT
                event_id,
                source_type,
                try_cast(regexp_extract(raw_pointer, '#row=([0-9]+)', 1) AS BIGINT) AS row_id
            FROM events_base
            WHERE source_type IN ('fisd', 'dealscan', 'compustat_proxy')
        ),
        corp AS (
            SELECT
                row_number() OVER () - 1 AS row_id,
                -- bond issuance (FISD)
                ISSUE_ID,
                ISSUE_CUSIP,
                COMPLETE_CUSIP,
                ISSUE_NAME,
                offering_date,
                maturity_date,
                offering_amt_k,
                principal_amt,
                currency,
                coupon,
                COUPON_TYPE,
                SECURITY_LEVEL,
                BOND_TYPE,
                CONVERTIBLE,
                PRIVATE_PLACEMENT,
                RULE_144A,
                ASSET_BACKED,
                PERPETUAL,
                call_amount,
                call_price,
                mr_price,
                next_call_price,
                next_sf_amount,
                -- equity offering proxy (Compustat)
                offering_amount,
                amount_sold,
                amount_remaining,
                equity_flag,
                form_type,
                filing_date,
                accepted_date,
                issuer_state,
                issuer_country,
                -- loan issuance (Dealscan)
                facilityid,
                packageid,
                borrower_name,
                loantype,
                maturity,
                secured,
                seniority,
                primarypurpose,
                secondarypurpose,
                dealpurpose,
                dealamount,
                dealstatus
            FROM read_parquet('{corp_path.as_posix()}')
        ),
        joined AS (
            SELECT
                e.*,
                c.*
            FROM events_ca e
            LEFT JOIN corp c
              ON e.row_id = c.row_id
        )
        SELECT
            event_id,
            CASE
                WHEN source_type = 'fisd' AND ISSUE_ID IS NOT NULL THEN
                    struct_pack(
                        issue_id := ISSUE_ID,
                        issue_cusip := ISSUE_CUSIP,
                        complete_cusip := COMPLETE_CUSIP,
                        issue_name := ISSUE_NAME,
                        offering_date := offering_date,
                        maturity_date := maturity_date,
                        offering_amt_k := offering_amt_k,
                        principal_amt := principal_amt,
                        currency := currency,
                        coupon := coupon,
                        coupon_type := COUPON_TYPE,
                        security_level := SECURITY_LEVEL,
                        bond_type := BOND_TYPE,
                        convertible := CONVERTIBLE,
                        private_placement := PRIVATE_PLACEMENT,
                        rule_144a := RULE_144A,
                        asset_backed := ASSET_BACKED,
                        perpetual := PERPETUAL,
                        call_amount := call_amount,
                        call_price := call_price,
                        mr_price := mr_price,
                        next_call_price := next_call_price,
                        next_sf_amount := next_sf_amount
                    )
                WHEN source_type = 'dealscan' AND facilityid IS NOT NULL THEN
                    struct_pack(
                        facilityid := facilityid,
                        packageid := packageid,
                        borrower_name := borrower_name,
                        loantype := loantype,
                        maturity := maturity,
                        secured := secured,
                        seniority := seniority,
                        primarypurpose := primarypurpose,
                        secondarypurpose := secondarypurpose,
                        dealpurpose := dealpurpose,
                        dealamount := dealamount,
                        dealstatus := dealstatus
                    )
                WHEN source_type = 'compustat_proxy' THEN
                    struct_pack(
                        offering_amount := offering_amount,
                        amount_sold := amount_sold,
                        amount_remaining := amount_remaining,
                        equity_flag := equity_flag,
                        form_type := form_type,
                        filing_date := filing_date,
                        accepted_date := accepted_date,
                        issuer_state := issuer_state,
                        issuer_country := issuer_country
                    )
                ELSE NULL
            END AS params,
            source_type
        FROM joined
        """
        corp_params = con.execute(query).df()
        if not corp_params.empty:
            out = out.merge(corp_params[["event_id", "params"]], on="event_id", how="left", suffixes=("", "_corp"))
            out["params"] = out["params_corp"].combine_first(out["params"])
            out = out.drop(columns=["params_corp"])

    # drop raw_pointer if present
    if "raw_pointer" in out.columns:
        out = out.drop(columns=["raw_pointer"])

    out.to_parquet(out_path, index=False)
    print(f"Wrote Canonical Event Store -> {out_path}")


if __name__ == "__main__":
    main()
