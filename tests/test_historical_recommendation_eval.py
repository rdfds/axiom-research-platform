from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from src.historical_recommendation_eval import (
    _aggregate_historical_cases,
    _cached_snapshot_loader,
    _filter_excluded_historical_cases,
    _historical_case_key,
    _load_excluded_historical_case_keys,
    _load_fixed_historical_cases,
    _normalize_fixed_historical_case,
    _prefilter_support_is_eligible,
    _prioritize_historical_cases,
    build_historical_recommendation_report,
    render_historical_recommendation_markdown,
    _resolve_supported_historical_entities,
    _score_ex_post_alignment,
    _select_historical_cases_from_frame,
    _snapshot_coverage_summary,
    _snapshot_has_meaningful_coverage,
    _summarize_historical_selection_pool,
    _summarize_case_support_by_family,
)
from src.model_feature_bundle import feature_view_from_snapshot


@dataclass
class _DummySnapshot:
    company_id: str
    as_of_time: str
    features: dict


def test_select_historical_cases_from_frame_stratifies_and_limits_company_reuse():
    frame = pd.DataFrame(
        [
            {"company_id": "A", "action_date": "2024-06-01T00:00:00Z", "normalized_action_id": "capital_return.open_market_buyback", "normalized_action_family": "capital_return"},
            {"company_id": "B", "action_date": "2024-05-01T00:00:00Z", "normalized_action_id": "capital_structure.refinancing", "normalized_action_family": "capital_structure"},
            {"company_id": "C", "action_date": "2024-04-01T00:00:00Z", "normalized_action_id": "mna.tuck_in_acquisition", "normalized_action_family": "mna"},
            {"company_id": "A", "action_date": "2024-03-01T00:00:00Z", "normalized_action_id": "capital_return.special_dividend", "normalized_action_family": "capital_return"},
        ]
    )
    frame["action_date"] = pd.to_datetime(frame["action_date"], utc=True)

    cases = _select_historical_cases_from_frame(
        frame=frame,
        case_count=3,
        lookback_days=90,
        max_cases_per_company=1,
    )

    assert len(cases) == 3
    assert sorted(case["company_id"] for case in cases) == ["A", "B", "C"]
    assert all(case["as_of_time"] < case["anchor_action_date"] for case in cases)


def test_filter_excluded_historical_cases_removes_matching_anchor_rows():
    frame = pd.DataFrame(
        [
            {
                "company_id": "A",
                "action_date": pd.Timestamp("2024-06-01T00:00:00Z"),
                "normalized_action_id": "capital_return.open_market_buyback",
                "normalized_action_family": "capital_return",
            },
            {
                "company_id": "B",
                "action_date": pd.Timestamp("2024-05-01T00:00:00Z"),
                "normalized_action_id": "capital_structure.refinancing",
                "normalized_action_family": "capital_structure",
            },
        ]
    )
    excluded = {
        ("A", pd.Timestamp("2024-06-01T00:00:00Z"), "capital_return.open_market_buyback"),
    }

    filtered = _filter_excluded_historical_cases(frame, excluded)

    assert len(filtered) == 1
    assert filtered.iloc[0]["company_id"] == "B"


def test_load_excluded_historical_case_keys_reads_source_company_ids(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        """
        {
          "cases": [
            {
              "company_id": "resolved-A",
              "source_company_id": "source-A",
              "anchor_action_id": "capital_return.dividend_increase",
              "anchor_action_date": "2024-06-01T00:00:00+00:00"
            }
          ]
        }
        """.strip()
    )

    excluded = _load_excluded_historical_case_keys([report_path])

    assert excluded == {
        ("source-A", pd.Timestamp("2024-06-01T00:00:00Z"), "capital_return.dividend_increase"),
    }


