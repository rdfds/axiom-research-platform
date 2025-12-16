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


