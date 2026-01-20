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


