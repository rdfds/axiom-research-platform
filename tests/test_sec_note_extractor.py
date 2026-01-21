from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "59_extract_sec_credit_note_patterns.py"
    spec = importlib.util.spec_from_file_location("extract_sec_credit_note_patterns", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_revolver_note_rows_derives_undrawn_from_capacity_and_outstanding():
    module = _load_module()
    doc = {
        "document_id": "sec:abc:10k:2024",
        "source_type": "sec_edgar_filing",
        "doc_type": "10-K",
        "title": "Annual Report on Form 10-K",
        "raw_text": """
        Liquidity and Capital Resources
        Our revolving credit facility provides aggregate commitments of $2.0 billion.
        At December 31, 2024, $500 million was outstanding under the revolving credit facility.
        """,
    }
    rows = module.extract_revolver_note_rows(doc)
    metrics = {(row["metric_key"], row["pattern_name"]): row["value"] for row in rows}
    assert metrics[("financial.revolver_capacity", "capacity")] == 2_000_000_000.0
    assert metrics[("financial.revolver_outstanding", "outstanding")] == 500_000_000.0
    assert metrics[("financial.revolver_undrawn", "capacity_minus_outstanding")] == 1_500_000_000.0


def test_extract_lease_note_rows_captures_costs_liabilities_and_schedule():
    module = _load_module()
    doc = {
        "document_id": "sec:abc:10q:2024",
        "source_type": "sec_edgar_filing",
        "doc_type": "10-Q",
        "title": "Quarterly Report on Form 10-Q",
        "raw_text": """
        Leases
        Operating lease cost was $125 million and finance lease cost was $20 million.
        Lease liabilities current were $80 million. Lease liabilities noncurrent were $300 million.
        Maturity analysis of lease liabilities
        2025 $90 million
        2026 $85 million
        Thereafter $300 million
        """,
    }
    rows = module.extract_lease_note_rows(doc)
    df = pd.DataFrame(rows)
    assert float(df.loc[df["metric_key"] == "financial.lease_expense_operating", "value"].iloc[0]) == 125_000_000.0
    assert float(df.loc[df["metric_key"] == "financial.lease_expense_finance", "value"].iloc[0]) == 20_000_000.0
    assert float(df.loc[df["metric_key"] == "financial.lease_liability_current", "value"].iloc[0]) == 80_000_000.0
    assert float(df.loc[df["metric_key"] == "financial.lease_liability_noncurrent", "value"].iloc[0]) == 300_000_000.0
    schedule = df[df["metric_key"] == "financial.lease_payment_due"].set_index("bucket_label")["value"].to_dict()
    assert schedule["2025"] == 90_000_000.0
    assert schedule["2026"] == 85_000_000.0
    assert schedule["Thereafter"] == 300_000_000.0


def test_extract_debt_maturity_rows_captures_year_buckets():
    module = _load_module()
    doc = {
        "document_id": "sec:abc:10k:2024",
        "source_type": "sec_edgar_filing",
        "doc_type": "10-K",
        "title": "Annual Report on Form 10-K",
        "raw_text": """
        Long-term debt maturities are as follows:
        2025 $200 million
        2026 $350 million
        Thereafter $1.2 billion
        """,
    }
    rows = module.extract_debt_maturity_rows(doc)
    df = pd.DataFrame(rows)
    buckets = df.set_index("bucket_label")["value"].to_dict()
    assert buckets["2025"] == 200_000_000.0
    assert buckets["2026"] == 350_000_000.0
    assert buckets["Thereafter"] == 1_200_000_000.0


def test_extract_note_pattern_rows_skips_non_sec_documents():
    module = _load_module()
    doc = {
        "document_id": "fmp:ABC:2024Q4",
        "source_type": "fmp_transcripts",
        "doc_type": "earnings_call",
        "title": "ABC 2024Q4 Earnings Call",
        "raw_text": "Our revolving credit facility provides commitments of $2.0 billion.",
    }
    extracted = module.extract_note_pattern_rows(doc)
    assert extracted == {"revolver": [], "lease": [], "debt_maturity": []}
