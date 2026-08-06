from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_financial_rows(root: Path, year: int, rows: list[dict]) -> Path:
    year_dir = root / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    out = year_dir / "part_test.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _row(
    *,
    entity_id: str,
    line_item: str,
    value: float,
    fiscal_period_end: str = "2025-09-30",
    statement_type: str = "balance_sheet",
    available_time: str = "2025-11-05T00:00:00Z",
    ingestion_time: str = "2025-11-05T00:00:00Z",
) -> dict:
    return {
        "source_system": "sec_edgar_xbrl",
        "entity_id": entity_id,
        "company_id": entity_id,
        "event_time": fiscal_period_end,
        "available_time": available_time,
        "ingestion_time": ingestion_time,
        "version_id": f"{entity_id}:{line_item}:{fiscal_period_end}",
        "fiscal_period_end": fiscal_period_end,
        "fiscal_year": 2025,
        "fiscal_quarter": 3,
        "statement_type": statement_type,
        "line_item": line_item,
        "value": value,
        "currency": "USD",
        "units": "USD",
    }


def _run_registry_build(fin_root: Path, out_root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_financial_facts_registry.py"),
            "--financials-root",
            str(fin_root),
            "--out-root",
            str(out_root),
            "--years",
            "2025",
            "--overwrite",
            "--threads",
            "1",
            "--memory",
            "1GB",
        ],
        cwd=ROOT,
        check=True,
    )


def _load_output(out_root: Path) -> pd.DataFrame:
    return pd.read_parquet(out_root / "year=2025" / "part_financials.parquet")


def test_build_financial_facts_registry_derives_restricted_cash_and_revolver(tmp_path: Path):
    fin_root = tmp_path / "warehouse_financials"
    out_root = tmp_path / "extracted_fact_registry_enriched"
    _write_financial_rows(
        fin_root,
        2025,
        [
            _row(entity_id="ABC", line_item="CashAndCashEquivalentsAtCarryingValue", value=100.0),
            _row(entity_id="ABC", line_item="CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", value=112.0),
            _row(entity_id="ABC", line_item="LineOfCreditFacilityMaximumBorrowingCapacity", value=500.0),
            _row(entity_id="ABC", line_item="LineOfCreditFacilityAmountOutstanding", value=120.0),
        ],
    )

    _run_registry_build(fin_root, out_root)
    out = _load_output(out_root)

    restricted = out[out["fact_type"] == "financial.restricted_cash"]["fact_value"].tolist()
    revolver = out[out["fact_type"] == "financial.revolver_undrawn"]["fact_value"].tolist()

    assert 12.0 in restricted
    assert 380.0 in revolver


def test_build_financial_facts_registry_maps_marketable_and_restricted_components(tmp_path: Path):
    fin_root = tmp_path / "warehouse_financials"
    out_root = tmp_path / "extracted_fact_registry_enriched"
    _write_financial_rows(
        fin_root,
        2025,
        [
            _row(entity_id="ABC", line_item="MarketableSecuritiesCurrent", value=65.0),
            _row(entity_id="ABC", line_item="RestrictedCashAndCashEquivalentsNoncurrent", value=14.0),
            _row(entity_id="ABC", line_item="RestrictedCashAndCashEquivalentsCurrent", value=6.0),
        ],
    )

    _run_registry_build(fin_root, out_root)
    out = _load_output(out_root)

    marketable = out[out["fact_type"] == "financial.marketable_securities"]["fact_value"].tolist()
    restricted_current = out[out["fact_type"] == "financial.restricted_cash_current"]["fact_value"].tolist()
    restricted_noncurrent = out[out["fact_type"] == "financial.restricted_cash_noncurrent"]["fact_value"].tolist()

    assert 65.0 in marketable
    assert 6.0 in restricted_current
    assert 14.0 in restricted_noncurrent


def test_build_financial_facts_registry_revolver_keeps_component_availability(tmp_path: Path):
    fin_root = tmp_path / "warehouse_financials"
    out_root = tmp_path / "extracted_fact_registry_enriched"
    _write_financial_rows(
        fin_root,
        2025,
        [
            _row(
                entity_id="ABC",
                line_item="LineOfCreditFacilityMaximumBorrowingCapacity",
                value=500.0,
                available_time="2025-02-01T00:00:00Z",
                ingestion_time="2025-02-01T00:00:00Z",
            ),
            _row(
                entity_id="ABC",
                line_item="LineOfCreditFacilityAmountOutstanding",
                value=120.0,
                available_time="2025-02-01T00:00:00Z",
                ingestion_time="2025-02-01T00:00:00Z",
            ),
            _row(
                entity_id="ABC",
                line_item="CashAndCashEquivalentsAtCarryingValue",
                value=100.0,
                available_time="2026-02-01T00:00:00Z",
                ingestion_time="2026-02-01T00:00:00Z",
            ),
        ],
    )

    _run_registry_build(fin_root, out_root)
    out = _load_output(out_root)
    revolver = out[out["fact_type"] == "financial.revolver_undrawn"].copy()

    assert revolver["fact_value"].tolist() == [380.0]
    assert str(revolver.iloc[0]["published_at"]) == "2025-02-01 00:00:00+00:00"
