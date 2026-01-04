from datetime import date
from pathlib import Path
from unittest.mock import patch

import requests

from scripts.repair_statement_debt_override_artifact import (
    _CompanyProcessingTimeoutError,
    _extract_filing_table_debt_candidate,
    _fetch_sec_primary_document,
    _load_sec_submissions,
    _process_repair_row,
    _recompute_smart,
    _should_override_total_debt_with_filing_candidate,
)


class _OfflineSession:
    def get(self, url, timeout):  # noqa: D401, ANN001
        raise requests.ConnectionError(f"offline: {url}")


def test_load_sec_submissions_returns_none_when_network_unavailable(tmp_path):
    payload = _load_sec_submissions("0000001750", session=_OfflineSession(), cache_dir=tmp_path)
    assert payload is None


def test_fetch_sec_primary_document_returns_none_when_network_unavailable(tmp_path):
    html = _fetch_sec_primary_document(
        {
            "cik": "0000001750",
            "accession_number": "0000001750-24-000001",
            "primary_document": "form10k.htm",
        },
        session=_OfflineSession(),
        cache_dir=tmp_path,
    )
    assert html is None


def test_extract_filing_table_debt_candidate_sums_segmented_current_and_long_term_debt_rows():
    html = """
    <html>
      <body>
        <h1>Consolidated Balance Sheets (in millions)</h1>
        <table>
          <tr><th>Liabilities and stockholders' equity</th><th>Amount</th></tr>
          <tr><td>Company excluding Ford Credit debt payable within one year</td><td>21,919</td></tr>
          <tr><td>Ford Credit debt payable within one year</td><td>51,752</td></tr>
          <tr><td>Company excluding Ford Credit long-term debt</td><td>21,919</td></tr>
          <tr><td>Ford Credit long-term debt</td><td>67,665</td></tr>
        </table>
      </body>
    </html>
    """

    candidate = _extract_filing_table_debt_candidate(
        filing={"filing_date": "2026-02-11"},
        html=html,
        as_of_date=date(2026, 3, 28),
    )

    assert candidate is not None
    assert candidate["mode"] == "balance_sheet_current_plus_long"
    assert candidate["value"] == 163_255_000_000.0
    assert sum(1 for row in candidate["matched_rows"] if row["bucket"] == "long_total") == 2


def test_extract_filing_table_debt_candidate_handles_securitized_and_long_term_borrowings_table():
    html = """
    <html>
      <body>
        <h1>Condensed Consolidated Balance Sheets (in millions)</h1>
        <table>
          <tr><th>Current liabilities</th><th>Amount</th></tr>
          <tr><td>Short-term borrowings</td><td>14,392</td></tr>
          <tr><td>Short-term securitization borrowings</td><td>6,283</td></tr>
          <tr><td>Long-term borrowings</td><td>41,804</td></tr>
        </table>
      </body>
    </html>
    """

    candidate = _extract_filing_table_debt_candidate(
        filing={"filing_date": "2026-02-19"},
        html=html,
        as_of_date=date(2026, 3, 28),
    )

    assert candidate is not None
    assert candidate["mode"] == "balance_sheet_current_plus_long"
    assert candidate["current_mode"] == "current_total_plus_extras"
    assert candidate["value"] == 62_479_000_000.0


def test_extract_filing_table_debt_candidate_captures_vehicle_program_debt_from_balance_sheet():
    html = """
    <html>
      <body>
        <h1>Consolidated Balance Sheets (in millions)</h1>
        <table>
          <tr><th>Liabilities and stockholders' equity</th><th>Amount</th></tr>
          <tr><td>Short-term debt and current portion of long-term debt</td><td>540</td></tr>
          <tr><td>Long-term debt</td><td>5,465</td></tr>
          <tr><td>Liabilities under vehicle programs:</td></tr>
          <tr><td>Debt</td><td>4,056</td></tr>
          <tr><td>Debt due to Avis Budget Rental Car Funding (AESOP) LLC-related party</td><td>13,837</td></tr>
        </table>
      </body>
    </html>
    """

    candidate = _extract_filing_table_debt_candidate(
        filing={"filing_date": "2024-11-01"},
        html=html,
        as_of_date=date(2024, 12, 31),
    )

    assert candidate is not None
    assert candidate["mode"] == "balance_sheet_current_plus_long_plus_vehicle_program_debt"
    assert candidate["value"] == 23_898_000_000.0
    assert sum(1 for row in candidate["matched_rows"] if row["bucket"] == "vehicle_program_debt") == 2


def test_extract_filing_table_debt_candidate_ignores_vehicle_program_available_funding_table():
    html = """
    <html>
      <body>
        <h1>Available funding under our debt arrangements related to our vehicle programs (in millions)</h1>
        <table>
          <tr><th>Arrangement</th><th>Total</th><th>Outstanding</th><th>Available</th></tr>
          <tr><td>Americas - Debt due to Avis Budget Rental Car Funding (AESOP) LLC-related party</td><td>17,695</td><td>13,525</td><td>4,170</td></tr>
          <tr><td>Americas - Debt borrowings</td><td>7,730</td><td>4,356</td><td>3,374</td></tr>
        </table>
      </body>
    </html>
    """

    candidate = _extract_filing_table_debt_candidate(
        filing={"filing_date": "2024-11-01"},
        html=html,
        as_of_date=date(2024, 12, 31),
    )

    assert candidate is None


def test_filing_candidate_does_not_override_exact_current_debt_when_it_is_materially_smaller():
    assert not _should_override_total_debt_with_filing_candidate(
        current_value=131_574_000_000.0,
        current_support="exact",
        parsed_value=46_913_000_000.0,
    )


def test_filing_candidate_still_overrides_when_current_debt_is_missing_or_partial():
    assert _should_override_total_debt_with_filing_candidate(
        current_value=None,
        current_support="unsupported",
        parsed_value=62_479_000_000.0,
    )
    assert _should_override_total_debt_with_filing_candidate(
        current_value=20_675_000_000.0,
        current_support="proxy_missing_component",
        parsed_value=62_479_000_000.0,
    )


def test_filing_candidate_does_not_override_partial_current_debt_when_it_is_implausibly_smaller():
    assert not _should_override_total_debt_with_filing_candidate(
        current_value=11_211_618_000.0,
        current_support="proxy_missing_component",
        parsed_value=11_786_562.0,
    )


def test_filing_candidate_can_still_override_partial_current_debt_when_modestly_smaller():
    assert _should_override_total_debt_with_filing_candidate(
        current_value=7_677_000_000.0,
        current_support="proxy_missing_component",
        parsed_value=7_603_000_000.0,
    )


def test_process_repair_row_fail_opens_on_timeout():
    row = {
        "company_id": "0000723612",
        "features": {
            "capital_structure.total_debt_provider_direct": {
                "value": 5_995_000_000.0,
                "support_mode": "proxy_missing_component",
            }
        },
    }
    original = dict(row["features"]["capital_structure.total_debt_provider_direct"])
    with patch(
        "scripts.repair_statement_debt_override_artifact._repair_total_debt_node",
        side_effect=_CompanyProcessingTimeoutError("boom"),
    ):
        row_repaired, sec_row_repaired, failure_reason = _process_repair_row(
            row=row,
            candidates={},
            computed_at="2026-03-31T00:00:00Z",
            provenance_source="test",
            sec_session=None,
            sec_cache_dir=Path("/tmp"),
            timeout_seconds=1,
            skip_fact_registry_repair=False,
        )
    assert not row_repaired
    assert not sec_row_repaired
    assert failure_reason == "company_processing_timeout"
    assert row["features"]["capital_structure.total_debt_provider_direct"] == original


def test_process_repair_row_fail_opens_on_exception():
    row = {
        "company_id": "0000723612",
        "features": {
            "capital_structure.total_debt_provider_direct": {
                "value": 5_995_000_000.0,
                "support_mode": "proxy_missing_component",
            }
        },
    }
    original = dict(row["features"]["capital_structure.total_debt_provider_direct"])
    with patch(
        "scripts.repair_statement_debt_override_artifact._repair_total_debt_node",
        side_effect=RuntimeError("boom"),
    ):
        row_repaired, sec_row_repaired, failure_reason = _process_repair_row(
            row=row,
            candidates={},
            computed_at="2026-03-31T00:00:00Z",
            provenance_source="test",
            sec_session=None,
            sec_cache_dir=Path("/tmp"),
            timeout_seconds=1,
            skip_fact_registry_repair=False,
        )
    assert not row_repaired
    assert not sec_row_repaired
    assert failure_reason == "company_processing_failed"
    assert row["features"]["capital_structure.total_debt_provider_direct"] == original


def test_recompute_smart_ignores_negative_lease_input_when_building_debt_like():
    features = {
        "capital_structure.total_debt_provider_direct": {
            "value": 100.0,
            "support_mode": "exact",
        },
        "capital_structure.lease_liabilities_sec_exact": {
            "value": -20.0,
            "support_mode": "exact",
        },
        "liquidity.available_liquidity_normalized": {
            "value": 10.0,
            "support_mode": "exact",
        },
        "operating.operating_earnings_normalized": {
            "value": 50.0,
            "support_mode": "exact",
        },
    }

    _recompute_smart(
        features,
        row={"as_of_time": "2024-12-31T00:00:00Z"},
        computed_at="2026-04-01T00:00:00Z",
        provenance_source="test",
    )

    debt_like = features["capital_structure.debt_like_obligations_normalized"]
    assert debt_like["value"] == 100.0
    assert debt_like["support_mode"] == "proxy_missing_component"
    assert debt_like["component_breakdown"]["lease_negative_input_ignored"] is True
    assert features["capital_structure.net_debt_normalized"]["value"] == 90.0
    assert features["capital_structure.gross_leverage_normalized"]["value"] == 2.0
    assert features["capital_structure.net_leverage_normalized"]["value"] == 1.8
