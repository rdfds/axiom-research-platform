#!/usr/bin/env python
"""
Build financial-statement facts into ExtractedFactRegistry-compatible rows.

Outputs per-year parquet parts that can live alongside the enriched fact registry,
so downstream reads see both text-derived and financial-derived facts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--financials-root", default="data/warehouse/warehouse_financials")
    parser.add_argument("--out-root", default="data/inputs_layer/extracted_fact_registry_enriched")
    parser.add_argument("--years", default=None, help="Comma-separated years (default: all).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", default="6GB")
    args = parser.parse_args()

    fin_root = ROOT / args.financials_root
    out_root = ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")

    # resolve years
    if args.years:
        years = [y.strip() for y in args.years.split(",") if y.strip()]
    else:
        years = []
        for p in fin_root.glob("year=*"):
            m = re.search(r"year=(\d{4})", p.as_posix())
            if m:
                years.append(m.group(1))
        years = sorted(set(years))

    # mapping: line_item -> (fact_type, priority)
    # priority lower = preferred
    mapping_rows = [
        ("Revenue", "financial.revenue", 1),
        ("Revenues", "financial.revenue", 2),
        ("SalesRevenueNet", "financial.revenue", 3),
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "financial.revenue", 4),
        ("EBITDA", "financial.ebitda", 1),
        ("OperatingIncome", "financial.ebit", 1),
        ("OperatingIncomeLoss", "financial.ebit", 2),
        ("NetIncome", "financial.net_income", 1),
        ("NetIncomeLoss", "financial.net_income", 2),
        ("Cash", "financial.cash", 1),
        ("CashAndCashEquivalentsAtCarryingValue", "financial.cash", 2),
        ("CashAndCashEquivalentsPeriodIncreaseDecrease", "financial.cash_delta", 1),
        ("RestrictedCash", "financial.restricted_cash", 1),
        ("RestrictedCashAndCashEquivalents", "financial.restricted_cash", 2),
        ("RestrictedCashAndCashEquivalentsAtCarryingValue", "financial.restricted_cash", 3),
        ("RestrictedCashAndCashEquivalentsCurrent", "financial.restricted_cash_current", 1),
        ("RestrictedCashAndCashEquivalentsNoncurrent", "financial.restricted_cash_noncurrent", 1),
        ("RestrictedCashAndInvestmentsNoncurrent", "financial.restricted_cash_noncurrent", 2),
        ("MarketableSecurities", "financial.marketable_securities", 1),
        ("MarketableSecuritiesCurrent", "financial.marketable_securities", 2),
        ("ShortTermInvestments", "financial.marketable_securities", 3),
        ("TradingSecuritiesShortTermInvestmentsAmortizedCost", "financial.marketable_securities", 4),
        ("CashAndShortTermInvestments", "financial.cash_and_short_term_investments", 1),
        ("CashCashEquivalentsAndShortTermInvestments", "financial.cash_and_short_term_investments", 2),
        ("UnusedCommitmentUnderLineOfCreditFacility", "financial.revolver_undrawn", 1),
        ("UnusedCommitmentUnderRevolvingCreditFacility", "financial.revolver_undrawn", 2),
        ("AvailableBorrowingsUnderLineOfCreditFacility", "financial.revolver_undrawn", 3),
        ("DebtCurrent", "financial.debt_current", 1),
        ("ShortTermDebt", "financial.debt_current", 2),
        ("DebtLongTerm", "financial.debt_long_term", 1),
        ("LongTermDebt", "financial.debt_long_term", 2),
        ("SharesOut", "financial.shares_out", 1),
        ("Capex", "financial.capex", 1),
        ("OperatingCashFlow", "financial.operating_cash_flow", 1),
        ("NetCashProvidedByUsedInOperatingActivities", "financial.operating_cash_flow", 2),
        ("InterestExpense", "financial.interest_expense", 1),
        ("CurrentAssets", "financial.current_assets", 1),
        ("CurrentLiabilities", "financial.current_liabilities", 1),
        ("CashDividends", "financial.dividends_cash", 1),
        ("CommonStockDividendsPerShareCashPaid", "financial.dividends_per_share_cash", 1),
    ]

    map_values = ",\n            ".join(
        [f"('{li}', '{ft}', {prio})" for li, ft, prio in mapping_rows]
    )
    derived_line_item_rows = [
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations",
        "LineOfCreditFacilityMaximumBorrowingCapacity",
        "LineOfCreditFacilityAmountOutstanding",
        "LineOfCredit",
    ]
    derived_line_item_values = ",\n            ".join([f"('{li}')" for li in derived_line_item_rows])

    for y in years:
        out_dir = out_root / f"year={y}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "part_financials.parquet"
        if out_file.exists() and not args.overwrite:
            print(f"Skipping year={y}, output exists.")
            continue

        fin_dir = fin_root / f"year={y}"
        fin_files = sorted(fin_dir.glob("*.parquet"))
        if not fin_files:
            print(f"No financial files found for year={y}, skipping.")
            continue

        def build_query(fin_source_sql: str) -> str:
            return f"""
        WITH mapping AS (
            SELECT * FROM (VALUES
            {map_values}
            ) AS t(line_item, fact_type, priority)
        ),
        derived_line_items AS (
            SELECT * FROM (VALUES
            {derived_line_item_values}
            ) AS t(line_item)
        ),
        fin AS (
            SELECT
                source_system,
                entity_id,
                company_id,
                event_time,
                available_time,
                ingestion_time,
                version_id,
                fiscal_period_end,
                fiscal_year,
                fiscal_quarter,
                statement_type,
                line_item,
                value,
                currency,
                units
            FROM {fin_source_sql}
            WHERE line_item IN (
                SELECT line_item FROM mapping
                UNION
                SELECT line_item FROM derived_line_items
            )
        ),
        base AS (
            SELECT
                f.*,
                m.fact_type,
                m.priority
            FROM fin f
            JOIN mapping m
              ON f.line_item = m.line_item
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY entity_id, fiscal_period_end, fact_type
                    ORDER BY priority ASC, available_time DESC
                ) AS rn
            FROM base
        ),
        base_facts AS (
            SELECT * FROM ranked WHERE rn = 1
        ),
        agg AS (
            SELECT
                entity_id,
                company_id,
                fiscal_period_end,
                max(CASE WHEN line_item = 'OperatingCashFlow' THEN value END) AS ocf,
                max(CASE WHEN line_item = 'NetCashProvidedByUsedInOperatingActivities' THEN value END) AS ocf_alt,
                max(CASE WHEN line_item = 'Capex' THEN value END) AS capex,
                max(CASE WHEN line_item = 'CurrentAssets' THEN value END) AS curr_assets,
                max(CASE WHEN line_item = 'CurrentLiabilities' THEN value END) AS curr_liab,
                max(CASE WHEN line_item IN ('Cash', 'CashAndCashEquivalentsAtCarryingValue') THEN value END) AS cash_direct,
                max(CASE WHEN line_item = 'DebtCurrent' THEN value END) AS debt_current,
                max(CASE WHEN line_item = 'ShortTermDebt' THEN value END) AS debt_current_alt,
                max(CASE WHEN line_item = 'DebtLongTerm' THEN value END) AS debt_long,
                max(CASE WHEN line_item = 'LongTermDebt' THEN value END) AS debt_long_alt,
                max(CASE WHEN line_item IN ('RestrictedCash', 'RestrictedCashAndCashEquivalents', 'RestrictedCashAndCashEquivalentsAtCarryingValue') THEN value END) AS restricted_cash_direct,
                max(CASE WHEN line_item = 'RestrictedCashAndCashEquivalentsCurrent' THEN value END) AS restricted_cash_current,
                max(CASE WHEN line_item IN ('RestrictedCashAndCashEquivalentsNoncurrent', 'RestrictedCashAndInvestmentsNoncurrent') THEN value END) AS restricted_cash_noncurrent,
                max(CASE WHEN line_item IN ('CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations') THEN value END) AS cash_and_restricted_total,
                max(CASE WHEN line_item IN ('MarketableSecurities', 'MarketableSecuritiesCurrent', 'ShortTermInvestments', 'TradingSecuritiesShortTermInvestmentsAmortizedCost') THEN value END) AS marketable_direct,
                max(CASE WHEN line_item IN ('CashAndShortTermInvestments', 'CashCashEquivalentsAndShortTermInvestments') THEN value END) AS cash_and_short_term_investments,
                max(CASE WHEN line_item IN ('UnusedCommitmentUnderLineOfCreditFacility', 'UnusedCommitmentUnderRevolvingCreditFacility', 'AvailableBorrowingsUnderLineOfCreditFacility') THEN value END) AS revolver_undrawn_direct,
                max(CASE WHEN line_item = 'LineOfCreditFacilityMaximumBorrowingCapacity' THEN value END) AS revolver_capacity,
                max(CASE WHEN line_item = 'LineOfCreditFacilityAmountOutstanding' THEN value END) AS revolver_outstanding,
                max(CASE WHEN line_item = 'LineOfCredit' THEN value END) AS line_of_credit_reported,
                max(CASE WHEN line_item = 'LineOfCreditFacilityMaximumBorrowingCapacity' THEN available_time END) AS revolver_capacity_available_time,
                max(CASE WHEN line_item = 'LineOfCreditFacilityAmountOutstanding' THEN available_time END) AS revolver_outstanding_available_time,
                max(CASE WHEN line_item = 'LineOfCredit' THEN available_time END) AS line_of_credit_available_time,
                max(CASE WHEN line_item = 'LineOfCreditFacilityMaximumBorrowingCapacity' THEN event_time END) AS revolver_capacity_event_time,
                max(CASE WHEN line_item = 'LineOfCreditFacilityAmountOutstanding' THEN event_time END) AS revolver_outstanding_event_time,
                max(CASE WHEN line_item = 'LineOfCredit' THEN event_time END) AS line_of_credit_event_time,
                max(CASE WHEN line_item = 'LineOfCreditFacilityMaximumBorrowingCapacity' THEN ingestion_time END) AS revolver_capacity_ingestion_time,
                max(CASE WHEN line_item = 'LineOfCreditFacilityAmountOutstanding' THEN ingestion_time END) AS revolver_outstanding_ingestion_time,
                max(CASE WHEN line_item = 'LineOfCredit' THEN ingestion_time END) AS line_of_credit_ingestion_time,
                max(available_time) AS available_time,
                max(event_time) AS event_time,
                max(ingestion_time) AS ingestion_time,
                max(source_system) AS source_system,
                max(version_id) AS version_id,
                max(currency) AS currency,
                max(units) AS units
            FROM fin
            GROUP BY 1,2,3
        ),
        derived AS (
            SELECT
                entity_id,
                company_id,
                fiscal_period_end,
                COALESCE(ocf, ocf_alt) AS operating_cash_flow,
                capex,
                curr_assets,
                curr_liab,
                cash_direct,
                COALESCE(debt_current, debt_current_alt) AS debt_current,
                COALESCE(debt_long, debt_long_alt) AS debt_long,
                restricted_cash_direct,
                restricted_cash_current,
                restricted_cash_noncurrent,
                cash_and_restricted_total,
                marketable_direct,
                cash_and_short_term_investments,
                revolver_undrawn_direct,
                revolver_capacity,
                revolver_outstanding,
                line_of_credit_reported,
                revolver_capacity_available_time,
                revolver_outstanding_available_time,
                line_of_credit_available_time,
                revolver_capacity_event_time,
                revolver_outstanding_event_time,
                line_of_credit_event_time,
                revolver_capacity_ingestion_time,
                revolver_outstanding_ingestion_time,
                line_of_credit_ingestion_time,
                available_time,
                event_time,
                ingestion_time,
                source_system,
                version_id,
                currency,
                units
            FROM agg
        ),
        derived_facts AS (
            SELECT
                entity_id,
                company_id,
                fiscal_period_end,
                'financial.free_cash_flow' AS fact_type,
                (operating_cash_flow - capex) AS value,
                source_system,
                version_id,
                available_time,
                event_time,
                ingestion_time,
                currency,
                units
            FROM derived
            WHERE operating_cash_flow IS NOT NULL AND capex IS NOT NULL
            UNION ALL
            SELECT
                entity_id,
                company_id,
                fiscal_period_end,
                'financial.working_capital' AS fact_type,
                (curr_assets - curr_liab) AS value,
                source_system,
                version_id,
                available_time,
                event_time,
                ingestion_time,
                currency,
                units
            FROM derived
            WHERE curr_assets IS NOT NULL AND curr_liab IS NOT NULL
            UNION ALL
            SELECT
                entity_id,
                company_id,
                fiscal_period_end,
                'financial.total_debt' AS fact_type,
                (COALESCE(debt_current, 0) + COALESCE(debt_long, 0)) AS value,
                source_system,
                version_id,
                available_time,
                event_time,
                ingestion_time,
                currency,
                units
            FROM derived
            WHERE debt_current IS NOT NULL OR debt_long IS NOT NULL
            UNION ALL
            SELECT
                entity_id,
                company_id,
                fiscal_period_end,
                'financial.restricted_cash' AS fact_type,
                greatest(0, cash_and_restricted_total - COALESCE(cash_direct, 0)) AS value,
                source_system,
                version_id,
                available_time,
                event_time,
                ingestion_time,
                currency,
                units
            FROM derived
            WHERE restricted_cash_direct IS NULL
              AND restricted_cash_current IS NULL
              AND restricted_cash_noncurrent IS NULL
              AND cash_and_restricted_total IS NOT NULL
            UNION ALL
            SELECT
                entity_id,
                company_id,
                fiscal_period_end,
                'financial.marketable_securities' AS fact_type,
                greatest(0, cash_and_short_term_investments - COALESCE(cash_direct, 0)) AS value,
                source_system,
                version_id,
                available_time,
                event_time,
                ingestion_time,
                currency,
                units
            FROM derived
            WHERE marketable_direct IS NULL
              AND cash_and_short_term_investments IS NOT NULL
            UNION ALL
            SELECT
                entity_id,
                company_id,
                fiscal_period_end,
                'financial.revolver_undrawn' AS fact_type,
                greatest(0, revolver_capacity - COALESCE(revolver_outstanding, 0)) AS value,
                source_system,
                version_id,
                coalesce(
                    greatest(revolver_capacity_available_time, revolver_outstanding_available_time),
                    revolver_capacity_available_time,
                    revolver_outstanding_available_time
                ) AS available_time,
                coalesce(
                    greatest(revolver_capacity_event_time, revolver_outstanding_event_time),
                    revolver_capacity_event_time,
                    revolver_outstanding_event_time
                ) AS event_time,
                coalesce(
                    greatest(revolver_capacity_ingestion_time, revolver_outstanding_ingestion_time),
                    revolver_capacity_ingestion_time,
                    revolver_outstanding_ingestion_time
                ) AS ingestion_time,
                currency,
                units
            FROM derived
            WHERE revolver_undrawn_direct IS NULL
              AND revolver_capacity IS NOT NULL
            UNION ALL
            SELECT
                entity_id,
                company_id,
                fiscal_period_end,
                'financial.revolver_undrawn' AS fact_type,
                line_of_credit_reported AS value,
                source_system,
                version_id,
                line_of_credit_available_time AS available_time,
                line_of_credit_event_time AS event_time,
                line_of_credit_ingestion_time AS ingestion_time,
                currency,
                units
            FROM derived
            WHERE revolver_undrawn_direct IS NULL
              AND revolver_capacity IS NULL
              AND line_of_credit_reported IS NOT NULL
        ),
        all_facts AS (
            SELECT
                entity_id,
                company_id,
                fiscal_period_end,
                fact_type,
                value,
                source_system,
                version_id,
                available_time,
                event_time,
                ingestion_time,
                currency,
                units,
                statement_type,
                line_item,
                fiscal_year,
                fiscal_quarter
            FROM base_facts
            UNION ALL
            SELECT
                entity_id,
                company_id,
                fiscal_period_end,
                fact_type,
                value,
                source_system,
                version_id,
                available_time,
                event_time,
                ingestion_time,
                currency,
                units,
                'derived' AS statement_type,
                NULL AS line_item,
                NULL AS fiscal_year,
                NULL AS fiscal_quarter
            FROM derived_facts
        )
        SELECT
            md5(
                concat_ws('|',
                    coalesce(source_system, ''),
                    coalesce(entity_id, ''),
                    coalesce(CAST(fiscal_period_end AS VARCHAR), ''),
                    coalesce(fact_type, ''),
                    coalesce(CAST(version_id AS VARCHAR), '')
                )
            ) AS fact_id,
            concat(
                'finstmt:',
                coalesce(source_system, 'unknown'),
                ':',
                coalesce(entity_id, 'unknown'),
                ':',
                CAST(fiscal_period_end AS VARCHAR)
            ) AS document_id,
            entity_id,
            fact_type,
            value AS fact_value,
            COALESCE(units, currency) AS unit,
            concat(
                'statement_type=', coalesce(statement_type, 'unknown'),
                '; line_item=', coalesce(line_item, 'derived'),
                '; fiscal_period_end=', coalesce(CAST(fiscal_period_end AS VARCHAR), ''),
                '; fiscal_year=', coalesce(CAST(fiscal_year AS VARCHAR), ''),
                '; fiscal_quarter=', coalesce(CAST(fiscal_quarter AS VARCHAR), '')
            ) AS context,
            1.0 AS confidence_score,
            NULL AS citation_span,
            NULL AS paragraph_index,
            NULL AS speaker,
            NULL AS transcript_timestamp,
            NULL AS is_qa,
            CAST(version_id AS VARCHAR) AS source_id,
            coalesce(source_system, 'financials') AS source_type,
            CAST(available_time AS TIMESTAMPTZ) AS published_at,
            CAST(event_time AS TIMESTAMPTZ) AS effective_at,
            CAST(ingestion_time AS TIMESTAMPTZ) AS ingested_at,
            concat(
                'warehouse_financials:',
                coalesce(source_system, 'unknown'),
                ':',
                coalesce(CAST(version_id AS VARCHAR), '')
            ) AS raw_pointer
        FROM all_facts
        WHERE value IS NOT NULL
        """

        file_list = ", ".join([f"'{p.as_posix()}'" for p in fin_files])
        fin_source = f"read_parquet([{file_list}], union_by_name=True)"
        query = build_query(fin_source)

        try:
            con.execute(f"COPY ({query}) TO '{out_file.as_posix()}' (FORMAT 'parquet');")
            print(f"Wrote financial facts year={y} -> {out_file}")
        except Exception as e:
            msg = str(e)
            m = re.search(r'file \"([^\"]+\\.parquet)\"', msg) or re.search(r"file '([^']+\\.parquet)'", msg)
            if not m:
                raise
            bad_file = m.group(1)
            fin_files = [p for p in fin_files if p.as_posix() != bad_file]
            if not fin_files:
                raise
            file_list = ", ".join([f"'{p.as_posix()}'" for p in fin_files])
            fin_source = f"read_parquet([{file_list}], union_by_name=True)"
            query = build_query(fin_source)
            con.execute(f"COPY ({query}) TO '{out_file.as_posix()}' (FORMAT 'parquet');")
            print(f"Wrote financial facts year={y} -> {out_file} (skipped bad file {bad_file})")

    print(f"Saved financial facts into {out_root}")


if __name__ == "__main__":
    main()
