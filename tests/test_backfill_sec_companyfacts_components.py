from pathlib import Path

import pytest

from scripts.backfill_sec_companyfacts_components import (
    _cash_sti_proxy_represents_cash_only,
    _extract_lease_liabilities,
    _extract_revolver_undrawn,
    _extract_restricted_cash,
    _repair_cash_sti_from_statement_cash,
    _repair_restricted_cash_from_total_cash_reconciliation,
    _statement_fact_node_is_fresh_enough,
)


def _companyfacts_with_entries(entries):
    facts = {}
    for concept, value in entries.items():
        facts[concept] = {
            "units": {
                "USD": [
                    {
                        "val": value,
                        "end": "2024-12-31",
                        "filed": "2024-12-31",
                        "fy": 2024,
                        "fp": "FY",
                        "form": "10-K",
                    }
                ]
            }
        }
    return {"facts": {"us-gaap": facts}}


def _companyfacts_with_fact_rows(entries):
    facts = {}
    for concept, rows in entries.items():
        facts[concept] = {"units": {"USD": rows}}
    return {"facts": {"us-gaap": facts}}


