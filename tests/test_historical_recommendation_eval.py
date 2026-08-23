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


def test_normalize_fixed_historical_case_uses_source_company_id_when_present():
    normalized = _normalize_fixed_historical_case(
        {
            "company_id": "resolved-A",
            "source_company_id": "source-A",
            "ticker": "AAA",
            "mapping_method": "ticker",
            "anchor_action_id": "capital_return.dividend_increase",
            "anchor_action_family": "capital_return",
            "anchor_action_date": "2024-06-01T00:00:00+00:00",
            "as_of_time": "2024-02-02T00:00:00+00:00",
        }
    )

    assert normalized == {
        "company_id": "resolved-A",
        "source_company_id": "source-A",
        "ticker": "AAA",
        "mapping_method": "ticker",
        "anchor_action_date": "2024-06-01T00:00:00+00:00",
        "anchor_action_id": "capital_return.dividend_increase",
        "anchor_action_family": "capital_return",
        "as_of_time": "2024-02-02T00:00:00+00:00",
    }


def test_load_fixed_historical_cases_dedupes_and_preserves_order(tmp_path):
    report_a = tmp_path / "report_a.json"
    report_a.write_text(
        """
        {
          "cases": [
            {
              "company_id": "resolved-A",
              "source_company_id": "source-A",
              "anchor_action_id": "capital_return.dividend_increase",
              "anchor_action_family": "capital_return",
              "anchor_action_date": "2024-06-01T00:00:00+00:00",
              "as_of_time": "2024-02-02T00:00:00+00:00"
            },
            {
              "company_id": "resolved-B",
              "source_company_id": "source-B",
              "anchor_action_id": "capital_structure.refinancing",
              "anchor_action_family": "capital_structure",
              "anchor_action_date": "2024-05-01T00:00:00+00:00",
              "as_of_time": "2024-01-02T00:00:00+00:00"
            }
          ]
        }
        """.strip()
    )
    report_b = tmp_path / "report_b.json"
    report_b.write_text(
        """
        {
          "cases": [
            {
              "company_id": "resolved-B",
              "source_company_id": "source-B",
              "anchor_action_id": "capital_structure.refinancing",
              "anchor_action_family": "capital_structure",
              "anchor_action_date": "2024-05-01T00:00:00+00:00",
              "as_of_time": "2024-01-02T00:00:00+00:00"
            },
            {
              "company_id": "resolved-C",
              "source_company_id": "source-C",
              "anchor_action_id": "capital_return.special_dividend",
              "anchor_action_family": "capital_return",
              "anchor_action_date": "2024-04-01T00:00:00+00:00",
              "as_of_time": "2023-12-02T00:00:00+00:00"
            }
          ]
        }
        """.strip()
    )

    cases = _load_fixed_historical_cases([report_a, report_b], case_count=5)

    assert [case["source_company_id"] for case in cases] == ["source-A", "source-B", "source-C"]
    assert [case["anchor_action_id"] for case in cases] == [
        "capital_return.dividend_increase",
        "capital_structure.refinancing",
        "capital_return.special_dividend",
    ]


def test_load_fixed_historical_cases_uses_all_manifest_cases_when_case_count_omitted(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
        {
          "case_count": 3,
          "cases": [
            {
              "company_id": "resolved-A",
              "source_company_id": "source-A",
              "anchor_action_id": "capital_return.dividend_increase",
              "anchor_action_family": "capital_return",
              "anchor_action_date": "2024-06-01T00:00:00+00:00",
              "as_of_time": "2024-02-02T00:00:00+00:00"
            },
            {
              "company_id": "resolved-B",
              "source_company_id": "source-B",
              "anchor_action_id": "capital_structure.refinancing",
              "anchor_action_family": "capital_structure",
              "anchor_action_date": "2024-05-01T00:00:00+00:00",
              "as_of_time": "2024-01-02T00:00:00+00:00"
            },
            {
              "company_id": "resolved-C",
              "source_company_id": "source-C",
              "anchor_action_id": "capital_return.special_dividend",
              "anchor_action_family": "capital_return",
              "anchor_action_date": "2024-04-01T00:00:00+00:00",
              "as_of_time": "2023-12-02T00:00:00+00:00"
            }
          ]
        }
        """.strip()
    )

    cases = _load_fixed_historical_cases([manifest])

    assert [case["source_company_id"] for case in cases] == ["source-A", "source-B", "source-C"]


def test_build_historical_recommendation_report_skips_prefilter_for_fixed_cases(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
        {
          "cases": [
            {
              "company_id": "resolved-B",
              "source_company_id": "source-B",
              "anchor_action_id": "capital_structure.refinancing",
              "anchor_action_family": "capital_structure",
              "anchor_action_date": "2024-05-01T00:00:00+00:00",
              "as_of_time": "2024-01-02T00:00:00+00:00"
            },
            {
              "company_id": "resolved-A",
              "source_company_id": "source-A",
              "anchor_action_id": "capital_return.dividend_increase",
              "anchor_action_family": "capital_return",
              "anchor_action_date": "2024-06-01T00:00:00+00:00",
              "as_of_time": "2024-02-02T00:00:00+00:00"
            }
          ]
        }
        """.strip()
    )

    monkeypatch.setattr(
        "src.historical_recommendation_eval._summarize_historical_selection_pool",
        lambda **_: {"family_counts": {}},
    )
    monkeypatch.setattr(
        "src.historical_recommendation_eval._load_action_support_summary",
        lambda **_: {"support_mode_counts": {}, "exact_status_counts": {}, "actions": {}},
    )
    monkeypatch.setattr(
        "src.historical_recommendation_eval._prefilter_case_support",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prefilter should be skipped for fixed cases")),
    )
    monkeypatch.setattr(
        "src.historical_recommendation_eval._load_realized_outcomes_lookup",
        lambda *_args, **_kwargs: {},
    )

    class _FakeBuilder:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr("src.historical_recommendation_eval.CompanyStateBuilder", _FakeBuilder)
    monkeypatch.setattr(
        "src.historical_recommendation_eval._build_historical_alias_overrides",
        lambda cases: {str(case["company_id"]): str(case["source_company_id"]) for case in cases},
    )
    monkeypatch.setattr(
        "src.historical_recommendation_eval._cached_snapshot_loader",
        lambda *args, **kwargs: (lambda company_id, as_of_dt: {"company_id": company_id, "features": []}),
    )
    monkeypatch.setattr(
        "src.historical_recommendation_eval._snapshot_coverage_summary",
        lambda _snapshot: {"non_missing_core_features": 5},
    )
    monkeypatch.setattr(
        "src.historical_recommendation_eval._snapshot_has_meaningful_coverage",
        lambda _coverage, min_non_missing_core_features=3: True,
    )
    monkeypatch.setattr(
        "src.historical_recommendation_eval._run_store_bindings",
        lambda: (
            lambda root: object(),
            None,
            None,
            None,
            None,
            None,
            lambda **kwargs: f"run-{kwargs['company_id']}",
            None,
        ),
    )
    monkeypatch.setattr(
        "src.historical_recommendation_eval.execute_recommendation_run",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "artifacts": {},
        },
    )
    monkeypatch.setattr(
        "src.historical_recommendation_eval._top_action_ids",
        lambda _package: ["capital_structure.equity_issuance"],
    )
    monkeypatch.setattr(
        "src.historical_recommendation_eval._score_ex_post_alignment",
        lambda **kwargs: {"score": 1.0, "reason": "anchor_primary_exact"},
    )

    report = build_historical_recommendation_report(
        runs_root=tmp_path / "runs",
        outcomes_path=tmp_path / "outcomes.parquet",
        entity_graph_path=tmp_path / "entity_graph.parquet",
        entity_identifier_path=tmp_path / "entity_identifier.parquet",
        entity_table_path=tmp_path / "entity.parquet",
        fixed_case_paths=[manifest],
        case_count=2,
        raw_timeseries_path=tmp_path / "raw_timeseries.parquet",
        event_store_path=tmp_path / "event_store.parquet",
        facts_path=tmp_path / "facts",
        ownership_summary_path=tmp_path / "ownership.parquet",
        issuer_ratings_path=tmp_path / "ratings.parquet",
    )

    assert report["selection_mode"] == "fixed_cases"
    assert report["family_prefilter_summary"] == {}
    assert [case["source_company_id"] for case in report["cases"]] == ["source-B", "source-A"]


def test_summarize_historical_selection_pool_tracks_missing_action_ids(tmp_path):
    outcomes_path = tmp_path / "outcomes.parquet"
    frame = pd.DataFrame(
        [
            {
                "company_id": "A",
                "action_date": "2024-06-01T00:00:00Z",
                "normalized_action_id": "mna.tuck_in_acquisition",
                "normalized_action_family": "mna",
            },
            {
                "company_id": "B",
                "action_date": "2024-05-01T00:00:00Z",
                "normalized_action_id": None,
                "normalized_action_family": "mna",
            },
            {
                "company_id": "C",
                "action_date": "2024-04-01T00:00:00Z",
                "normalized_action_id": None,
                "normalized_action_family": "portfolio",
            },
        ]
    )
    frame["action_date"] = pd.to_datetime(frame["action_date"], utc=True)
    frame.to_parquet(outcomes_path, index=False)

    summary = _summarize_historical_selection_pool(
        outcomes_path=outcomes_path,
        families=["mna", "portfolio"],
        alignment_horizon_days=30,
    )

    assert summary["families"] == ["mna", "portfolio"]
    assert summary["total_rows"] == 3
    assert summary["with_action_id_count"] == 1
    assert summary["missing_action_id_count"] == 2
    assert summary["family_counts"] == {
        "mna": {
            "row_count": 2,
            "with_action_id_count": 1,
            "missing_action_id_count": 1,
        },
        "portfolio": {
            "row_count": 1,
            "with_action_id_count": 0,
            "missing_action_id_count": 1,
        },
    }


def test_score_ex_post_alignment_prefers_exact_primary_match():
    lookup = {
        "A": [
            (pd.Timestamp("2024-06-01T00:00:00Z"), "capital_structure.refinancing", "capital_structure"),
            (pd.Timestamp("2024-07-15T00:00:00Z"), "capital_return.open_market_buyback", "capital_return"),
        ]
    }
    score = _score_ex_post_alignment(
        company_id="A",
        as_of_time="2024-03-01T00:00:00Z",
        recommended_action_ids=["capital_structure.refinancing"],
        outcomes_lookup=lookup,
        alignment_horizon_days=180,
        anchor_action_id="capital_structure.refinancing",
        anchor_action_family="capital_structure",
    )

    assert score["score"] == 1.0
    assert score["primary_exact_match"] is True
    assert score["reason"] == "anchor_primary_exact"


def test_score_ex_post_alignment_gives_family_credit_without_exact_match():
    lookup = {
        "A": [
            (pd.Timestamp("2024-06-01T00:00:00Z"), "capital_return.tender_offer_buyback", "capital_return"),
        ]
    }
    score = _score_ex_post_alignment(
        company_id="A",
        as_of_time="2024-03-01T00:00:00Z",
        recommended_action_ids=["capital_return.open_market_buyback"],
        outcomes_lookup=lookup,
        alignment_horizon_days=180,
        anchor_action_id="capital_return.tender_offer_buyback",
        anchor_action_family="capital_return",
    )

    assert score["score"] == 0.6
    assert score["primary_exact_match"] is False
    assert score["primary_family_match"] is True
    assert score["reason"] == "anchor_primary_family_match"


def test_score_ex_post_alignment_downshifts_to_family_only_for_family_only_actions():
    lookup = {
        "A": [
            (pd.Timestamp("2024-06-01T00:00:00Z"), "capital_structure.equity_issuance", "capital_structure"),
        ]
    }
    score = _score_ex_post_alignment(
        company_id="A",
        as_of_time="2024-03-01T00:00:00Z",
        recommended_action_ids=["capital_structure.refinancing"],
        outcomes_lookup=lookup,
        alignment_horizon_days=180,
        anchor_action_id="capital_structure.refinancing",
        anchor_action_family="capital_structure",
        anchor_action_support={"support_mode": "family_only"},
        recommended_action_support=[{"support_mode": "family_only"}],
    )

    assert score["score"] == 1.0
    assert score["primary_exact_match"] is False
    assert score["primary_family_match"] is True
    assert score["primary_support_adjusted_match"] is True
    assert score["primary_benchmark_mode"] == "family_only"
    assert score["reason"] == "anchor_primary_family_support_adjusted"


