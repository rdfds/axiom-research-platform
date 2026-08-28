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


def test_aggregate_historical_cases_tracks_action_support_modes():
    aggregate = _aggregate_historical_cases(
        [
            {
                "company_id": "A",
                "anchor_action_support": {"support_mode": "exact_supported"},
                "recommended_action_support": [{"support_mode": "exact_supported"}],
                "recommended_posture": "act",
                "top_action_ids": ["capital_return.open_market_buyback"],
                "anchor_action_family": "capital_return",
                "historical_alignment": {
                    "score": 1.0,
                    "primary_exact_match": True,
                    "primary_family_match": True,
                    "primary_support_adjusted_match": True,
                    "any_exact_match": True,
                    "any_family_match": True,
                    "any_support_adjusted_match": True,
                },
            },
            {
                "company_id": "B",
                "anchor_action_support": {"support_mode": "family_only"},
                "unsupported_reason": "no_feasible_plan_generated",
            },
        ]
    )

    assert aggregate["anchor_support_mode_counts"] == {
        "exact_supported": 1,
    }
    assert aggregate["recommended_support_mode_counts"] == {
        "exact_supported": 1,
    }
    assert aggregate["anchor_primary_support_adjusted_rate"] == 1.0
    assert aggregate["future_any_support_adjusted_rate"] == 1.0


def test_aggregate_historical_cases_anchor_support_modes_do_not_exceed_completed_cases():
    aggregate = _aggregate_historical_cases(
        [
            {
                "company_id": "A",
                "anchor_action_support": {"support_mode": "exact_supported"},
                "recommended_action_support": [{"support_mode": "exact_supported"}],
                "recommended_posture": "act",
                "top_action_ids": ["capital_return.open_market_buyback"],
                "anchor_action_family": "capital_return",
                "historical_alignment": {
                    "score": 1.0,
                    "primary_exact_match": True,
                    "primary_family_match": True,
                    "primary_support_adjusted_match": True,
                    "any_exact_match": True,
                    "any_family_match": True,
                    "any_support_adjusted_match": True,
                },
            },
            {
                "company_id": "B",
                "anchor_action_support": {"support_mode": "family_only"},
                "error": "runtime failure",
            },
            {
                "company_id": "C",
                "anchor_action_support": {"support_mode": "family_only"},
                "unsupported_reason": "no_feasible_plan_generated",
            },
        ]
    )

    assert aggregate["completed_case_count"] == 1
    assert sum(aggregate["anchor_support_mode_counts"].values()) == aggregate["completed_case_count"]


def test_cached_snapshot_loader_uses_persistent_cache(tmp_path):
    class DummyBuilder:
        def __init__(self):
            self.calls = 0

        def build(self, *, company_id: str, as_of_time: str, extra_aliases=None):
            self.calls += 1
            return _DummySnapshot(
                company_id=company_id,
                as_of_time=as_of_time,
                features={
                    "marker": "built",
                    "operating.revenue_ttm": {"value": 900.0, "support_mode": "exact"},
                    "macro.hy_oas": {"value": 3.2, "support_mode": "exact"},
                },
            )

    builder = DummyBuilder()
    events = []
    loader = _cached_snapshot_loader(
        builder,
        cache_dir=tmp_path / "snapshots",
        progress_logger=events.append,
    )

    first = loader("ABC", pd.Timestamp("2024-01-01T00:00:00Z").to_pydatetime())
    assert builder.calls == 1
    assert first["company_id"] == "ABC"
    precedent_view = feature_view_from_snapshot(first, view_name="precedent")
    assert precedent_view["state_vector_v1.size_log_revenue"]["value"] == math.log10(900.0)
    assert any(event["event"] == "snapshot_build_complete" for event in events)

    builder_2 = DummyBuilder()
    events_2 = []
    loader_2 = _cached_snapshot_loader(
        builder_2,
        cache_dir=tmp_path / "snapshots",
        progress_logger=events_2.append,
    )
    second = loader_2("ABC", pd.Timestamp("2024-01-01T00:00:00Z").to_pydatetime())
    assert builder_2.calls == 0
    assert second["company_id"] == "ABC"
    precedent_view_2 = feature_view_from_snapshot(second, view_name="precedent")
    assert precedent_view_2["state_vector_v1.credit_spread"]["value"] == 3.2
    assert any(event["event"] == "snapshot_cache_hit" for event in events_2)


def test_snapshot_coverage_rejects_empty_state_with_only_intent_defaults():
    snapshot = {
        "features": {
            "strategic.intent.return_capital_priority": {"value": 0.0, "missing_reason": None},
            "strategic.intent.deleveraging_priority": {"value": 0.0, "missing_reason": None},
            "liquidity.cash": {"value": None, "missing_reason": "unavailable"},
            "capital_structure.net_leverage": {"value": None, "missing_reason": "unavailable"},
        }
    }
    coverage = _snapshot_coverage_summary(snapshot)
    assert coverage["non_missing_core_feature_count"] == 0
    assert _snapshot_has_meaningful_coverage(coverage, min_non_missing_core_features=1) is False


def test_snapshot_coverage_accepts_snapshot_with_real_core_features():
    snapshot = {
        "features": {
            "liquidity.cash": {"value": 100.0, "missing_reason": None},
            "capital_structure.net_leverage": {"value": 2.5, "missing_reason": None},
            "market.market_cap": {"value": 5000.0, "missing_reason": None},
            "strategic.intent.return_capital_priority": {"value": 0.0, "missing_reason": None},
        }
    }
    coverage = _snapshot_coverage_summary(snapshot)
    assert coverage["non_missing_core_feature_count"] == 3
    assert _snapshot_has_meaningful_coverage(coverage, min_non_missing_core_features=3) is True


def test_aggregate_historical_cases_separates_unsupported_cases():
    aggregate = _aggregate_historical_cases(
        [
            {"company_id": "A", "unsupported_reason": "insufficient_snapshot_coverage"},
            {
                "company_id": "B",
                "recommended_posture": "act_now",
                "top_action_ids": ["capital_structure.refinancing"],
                "anchor_action_family": "capital_structure",
                "historical_alignment": {
                    "score": 1.0,
                    "primary_exact_match": True,
                    "primary_family_match": True,
                    "any_exact_match": True,
                    "any_family_match": True,
                },
            },
        ]
    )
    assert aggregate["completed_case_count"] == 1
    assert aggregate["unsupported_case_count"] == 1
    assert aggregate["scored_case_count"] == 1
    assert aggregate["coverage_skip_rate"] == 0.5


def test_resolve_supported_historical_entities_maps_by_ticker(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "company_id": "005606",
                "ticker": "HPQ",
                "action_date": pd.Timestamp("2024-12-30T00:00:00Z"),
                "normalized_action_id": "capital_structure.equity_issuance",
                "normalized_action_family": "capital_structure",
            }
        ]
    )
    entity_identifier_path = tmp_path / "entity_identifier.parquet"
    entity_table_path = tmp_path / "entity.parquet"
    pd.DataFrame(
        [
            {
                "entity_id": "0000047217",
                "identifier_type": "ticker",
                "identifier_value": "HPQ",
                "valid_from": "2024-01-01T00:00:00Z",
                "valid_to": None,
            }
        ]
    ).to_parquet(entity_identifier_path, index=False)
    pd.DataFrame(
        [{"entity_id": "0000047217", "entity_type": "company"}]
    ).to_parquet(entity_table_path, index=False)

    resolved = _resolve_supported_historical_entities(
        frame=frame,
        entity_identifier_path=entity_identifier_path,
        entity_table_path=entity_table_path,
        lookback_days=120,
    )
    assert len(resolved) == 1
    assert resolved.iloc[0]["resolved_company_id"] == "0000047217"
    assert resolved.iloc[0]["mapping_method"] == "ticker_identifier"


def test_prefilter_support_requires_real_core_sources():
    assert _prefilter_support_is_eligible({"facts_hits": 10, "timeseries_hits": 5, "estimated_supported": True}) is True
    assert _prefilter_support_is_eligible({"timeseries_hits": 5, "ownership_hits": 1, "estimated_supported": False}) is False


def test_prioritize_historical_cases_prefers_supported_profiles():
    cases = [
        {
            "company_id": "A",
            "as_of_time": "2024-01-01T00:00:00+00:00",
            "anchor_action_id": "capital_structure.refinancing",
            "anchor_action_family": "capital_structure",
        },
        {
            "company_id": "B",
            "as_of_time": "2024-01-02T00:00:00+00:00",
            "anchor_action_id": "capital_return.open_market_buyback",
            "anchor_action_family": "capital_return",
        },
    ]
    profiles = {
        _historical_case_key(cases[0]): {"estimated_supported": False, "score": 0.2, "strong_source_count": 0},
        _historical_case_key(cases[1]): {"estimated_supported": True, "score": 4.5, "strong_source_count": 3},
    }
    family_summary = _summarize_case_support_by_family(cases, profiles)
    ordered = _prioritize_historical_cases(
        cases,
        case_support_prefilter=profiles,
        family_prefilter_summary=family_summary,
    )
    assert [case["company_id"] for case in ordered] == ["B", "A"]


def test_summarize_case_support_by_family_sorts_best_family_first():
    cases = [
        {
            "company_id": "A",
            "as_of_time": "2024-01-01T00:00:00+00:00",
            "anchor_action_id": "capital_structure.refinancing",
            "anchor_action_family": "capital_structure",
        },
        {
            "company_id": "B",
            "as_of_time": "2024-01-02T00:00:00+00:00",
            "anchor_action_id": "capital_return.open_market_buyback",
            "anchor_action_family": "capital_return",
        },
    ]
    profiles = {
        _historical_case_key(cases[0]): {"estimated_supported": False, "score": 0.1},
        _historical_case_key(cases[1]): {"estimated_supported": True, "score": 3.2},
    }
    summary = _summarize_case_support_by_family(cases, profiles)
    assert list(summary.keys())[0] == "capital_return"
    assert summary["capital_return"]["estimated_supported_count"] == 1


def test_render_historical_recommendation_markdown_handles_null_alignment_score():
    report = {
        "case_count_requested": 1,
        "candidate_case_count": 1,
        "runs_analyzed": 1,
        "supported_case_count": 1,
        "family_prefilter_summary": {},
        "aggregate": {
            "completed_case_count": 1,
            "scored_case_count": 0,
            "unsupported_case_count": 0,
            "mean_alignment_score": 0.0,
            "strong_alignment_rate": 0.0,
            "anchor_primary_exact_rate": 0.0,
            "anchor_primary_family_rate": 0.0,
            "future_any_exact_rate": 0.0,
            "future_any_family_rate": 0.0,
        },
        "cases": [
            {
                "company_id": "A",
                "as_of_time": "2024-01-01T00:00:00Z",
                "anchor_action_id": "capital_structure.refinancing",
                "top_action_ids": ["capital_structure.refinancing"],
                "historical_alignment": {"score": None, "reason": "no_company_events"},
            }
        ],
    }
    rendered = render_historical_recommendation_markdown(report)
    assert "score=`n/a`" in rendered
