from __future__ import annotations

import importlib.util
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


MODULE_PATH = Path("./scripts/generate_one_company_precedent_audit.py")
SPEC = importlib.util.spec_from_file_location("generate_one_company_precedent_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_confidence_lines_include_coverage_and_action_score() -> None:
    lines = audit._confidence_lines(
        "Learned",
        {
            "calibration_confidence": 0.61,
            "confidence_label": "medium",
            "retrieval_tier": "exact",
            "out_of_sample_flag": False,
            "exact_match_count": 42,
            "minimum_exact_support": 5,
            "top_similarity_mean": 0.88,
            "top_weighted_feature_coverage": 0.97,
            "top_critical_feature_coverage": 0.91,
            "top_action_match_score": 1.0,
        },
    )
    joined = "\n".join(lines)
    assert "top-support critical coverage" in joined
    assert "top-support action match score" in joined


def test_action_params_from_outcome_row_carries_refinancing_family_hints() -> None:
    params = audit._action_params_from_outcome_row(
        {
            "normalized_action_id": "capital_structure.refinancing",
            "raw_action_subtype": "Revolver/Line >= 1 Yr.",
            "action_size": 180_000_000.0,
        }
    )
    assert params["amount_usd"] == 180_000_000.0
    assert params["source_action_subtype"] == "Revolver/Line >= 1 Yr."
    assert params["instrument_type"] == "revolver"


def test_match_explanation_lines_call_out_closest_and_gaps() -> None:
    target_values = {key: 0.0 for key in audit._STATE_VECTOR_V1_FEATURES}
    target_values["state_vector_v1.valuation_multiple"] = 60.0
    target_values["state_vector_v1.growth"] = 0.15
    target_values["state_vector_v1.market_stress"] = 0.40
    match_row = pd.Series(
        {
            "state_vector_v1.valuation_multiple": 58.0,
            "state_vector_v1.growth": 0.12,
            "state_vector_v1.market_stress": 0.05,
        }
    )
    feature_scales = {key: 1.0 for key in audit._STATE_VECTOR_V1_FEATURES}
    lines = audit._match_explanation_lines(
        action_id="capital_return.open_market_buyback",
        target_values=target_values,
        match_row=match_row,
        feature_scales=feature_scales,
    )
    joined = "\n".join(lines)
    assert "Why it matched" in joined
    assert "Main gaps" in joined
    assert "valuation multiple" in joined
    assert "market stress" in joined


def test_synthesized_snapshot_row_from_outcome_row_preserves_core_features() -> None:
    row = audit._synthesized_snapshot_row_from_outcome_row(
        {
            "action_size": 250000000.0,
            "base_revenue_ttm": 1000.0,
            "base_revenue_ttm_lag_1y": 900.0,
            "base_ebitda_ttm": 200.0,
            "base_margin": 0.2,
            "base_fcf_margin": 0.1,
            "base_total_debt": 300.0,
            "base_net_debt": 120.0,
            "base_cash": 180.0,
            "base_available_liquidity": 260.0,
            "base_current_debt": 40.0,
            "base_interest_expense": 20.0,
            "base_market_cap": 2500.0,
            "base_ev_ebitda": 12.5,
            "base_fcf_yield": 0.04,
            "base_volatility_30d": 0.2,
            "base_volatility_90d": 0.3,
            "base_drawdown_90d": -0.2,
            "base_momentum_60d": 0.08,
            "base_credit_spread_level": 0.035,
            "base_credit_window_proxy": 0.7,
            "base_equity_window_proxy": 0.8,
            "base_revenue_growth_yoy": 0.11,
            "macro_vix": 18.0,
            "macro_fed_funds_effective": 0.0525,
            "macro_hy_oas": 0.038,
            "macro_ig_oas": 0.012,
            "macro_real_gdp_growth_yoy": 0.021,
            "macro_sofr": 0.053,
            "macro_rate_10y": 0.041,
            "macro_rate_2y": 0.045,
            "sector": "Industrials",
            "subsector": "Electrical Equipment",
        },
        company_id="0000012345",
        as_of_time="2024-08-01T00:00:00+00:00",
        outcomes_path=Path("/tmp/mock_outcomes.parquet"),
    )

    features = row["features"]
    assert row["action_params"]["amount_usd"] == 250000000.0
    assert row["action_params"]["action_size"] == 250000000.0
    assert features["market.ev_ebitda"]["value"] == 12.5
    assert features["cash_flow.free_cash_flow_ttm"]["value"] == 100.0 * 1_000_000.0
    assert features["capital_structure.current_debt_provider_direct"]["value"] == 40.0 * 1_000_000.0
    assert features["capital_structure.debt_due_next_24m"]["value"] == 40.0 * 1_000_000.0
    assert features["capital_structure.debt_due_next_24m"]["support_mode"] == "proxy_missing_component"
    assert features["capital_structure.interest_coverage"]["value"] == 10.0
    assert features["market.enterprise_value"]["value"] == 2620.0 * 1_000_000.0
    assert features["market.vix"]["value"] == 18.0
    assert features["macro.fed_funds_effective"]["value"] == 0.0525
    assert features["macro.hy_oas"]["value"] == 0.038
    assert features["taxonomy.sector"]["value"] == "Industrials"
    assert features["operating.revenue_yoy_last_q"]["value"] == 0.11


def test_synthesized_snapshot_row_uses_direct_ticker_taxonomy_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PRECEDENT_DISABLE_HISTORICAL_TAXONOMY_LOOKUP", "1")
    monkeypatch.setattr(audit, "_enrich_missing_historical_taxonomy", lambda df: df)
    monkeypatch.setattr(
        audit,
        "_historical_taxonomy_for_ticker",
        lambda ticker, allow_sec_identity_heuristics=False: {
            "taxonomy.sector": "Information Technology",
            "taxonomy.subsector": "Semiconductors",
        },
    )

    row = audit._synthesized_snapshot_row_from_outcome_row(
        {
            "ticker": "FLNC",
            "action_size": 400000000.0,
            "base_revenue_ttm": 1000.0,
            "base_ebitda_ttm": 100.0,
            "base_market_cap": 2500.0,
        },
        company_id="0001868941",
        as_of_time="2024-08-01T00:00:00+00:00",
        outcomes_path=Path("/tmp/mock_outcomes.parquet"),
    )

    assert row["features"]["taxonomy.sector"]["value"] == "Information Technology"
    assert row["features"]["taxonomy.subsector"]["value"] == "Semiconductors"


def test_coerce_snapshot_row_for_audit_synthesizes_flat_outcome_rows() -> None:
    flat_row = {
        "company_id": "005338",
        "ticker": "GEF",
        "normalized_action_id": "capital_structure.refinancing",
        "action_date": "2016-11-03T00:00:00+00:00",
        "base_revenue_ttm": 3425.2,
        "base_ebitda_ttm": 468.6,
        "base_total_debt": 1026.2,
        "base_cash": 103.7,
        "base_market_cap": 2490.227,
        "base_ev_ebitda": 7.2828,
        "base_fcf_yield": 0.0761,
        "macro_fed_funds_effective": 0.41,
        "sector": "Containers & Packaging",
        "subsector": "SIC 2673",
    }

    row = audit._coerce_snapshot_row_for_audit(
        flat_row,
        company_id="005338",
        snapshot_as_of_time="2016-11-03T00:00:00+00:00",
        outcomes_path=Path("/tmp/mock_outcomes.parquet"),
    )

    assert "features" in row
    assert row["features"]["operating.revenue_ttm_provider_direct"]["value"] == 3425.2 * 1_000_000.0

    _, _, payload = audit._build_target_payload(row)

    assert payload["target_values"]["state_vector_v1.size_log_revenue"] is not None
    assert payload["target_values"]["state_vector_v1.valuation_multiple"] is not None


def test_build_target_payload_backfills_market_cap_and_taxonomy_aliases() -> None:
    row = {
        "company_id": "0000012345",
        "as_of_time": "2024-08-01T00:00:00+00:00",
        "features": {
            "market.market_cap_provider_direct": {"value": 2_500_000_000.0, "support_mode": "exact"},
            "taxonomy.sector": {"value": "Industrials", "support_mode": "exact"},
            "taxonomy.subsector": {"value": "Electrical Equipment", "support_mode": "exact"},
            "operating.revenue_ttm_provider_direct": {"value": 1_000_000_000.0, "support_mode": "exact"},
            "operating.ebitda_ltm_provider_direct": {"value": 200_000_000.0, "support_mode": "exact"},
        },
    }

    _, _, payload = audit._build_target_payload(row)

    assert payload["precedent_features"]["market_cap"]["value"] == 2_500_000_000.0
    assert payload["precedent_features"]["sector"]["value"] == "Industrials"
    assert payload["precedent_features"]["subsector"]["value"] == "Electrical Equipment"


def test_build_target_payload_caps_liquidity_when_current_debt_is_only_proxy() -> None:
    row = audit._synthesized_snapshot_row_from_outcome_row(
        {
            "action_size": 200_000_000.0,
            "base_revenue_ttm": 3802.0,
            "base_revenue_ttm_lag_1y": 3575.8,
            "base_ebitda_ttm": 451.6,
            "base_margin": 0.1188,
            "base_fcf_margin": -0.0156,
            "base_total_debt": 1100.4,
            "base_net_debt": 754.7,
            "base_cash": 345.7,
            "base_available_liquidity": 345.7,
            "base_current_debt": 0.8,
            "base_interest_expense": 164.0,
            "base_market_cap": 1677.7422,
            "base_ev_ebitda": 5.3863,
            "base_fcf_yield": -0.0353,
            "base_credit_spread_level": 0.0121,
            "base_credit_window_proxy": 0.8793,
            "base_equity_window_proxy": 0.2693,
            "macro_vix": 13.58,
            "macro_fed_funds_effective": 4.58,
            "macro_hy_oas": 2.64,
            "sector": "Industrials",
            "subsector": "Commercial Services & Supplies",
        },
        company_id="0000028823",
        as_of_time="2024-12-11T00:00:00+00:00",
        outcomes_path=Path("/tmp/mock_outcomes.parquet"),
    )

    _, bundle, payload = audit._build_target_payload(row)
    support_meta = (bundle.get("state_vector_v1", {}) or {}).get("support", {})

    assert payload["target_values"]["state_vector_v1.liquidity_flexibility"] == 25.0
    assert "current_debt_proxy_ratio_capped" in set(
        (support_meta.get("state_vector_v1.liquidity_flexibility") or {}).get("quality_flags") or []
    )


def test_load_historical_outcome_target_row_prefers_forward_nearest_action_date(tmp_path, monkeypatch) -> None:
    outcomes_path = tmp_path / "outcomes.parquet"
    monkeypatch.setenv("PRECEDENT_DISABLE_HISTORICAL_TAXONOMY_LOOKUP", "1")
    frame = pd.DataFrame(
        [
            {
                "company_id": "src123",
                "ticker": "TEST",
                "normalized_action_id": "capital_structure.new_debt_issuance",
                "action_date": "2024-10-01T00:00:00+00:00",
                "sector": "Industrials",
                "subsector": "Electrical Equipment",
                "base_revenue_ttm": 1000.0,
                "base_ebitda_ttm": 100.0,
                "base_ev_ebitda": 9.0,
                "base_total_debt": 500.0,
                "base_cash": 50.0,
                "base_market_cap": 1200.0,
            },
            {
                "company_id": "src123",
                "ticker": "TEST",
                "normalized_action_id": "capital_structure.new_debt_issuance",
                "action_date": "2024-12-15T00:00:00+00:00",
                "sector": "Industrials",
                "subsector": "Electrical Equipment",
                "base_revenue_ttm": 1000.0,
                "base_ebitda_ttm": 100.0,
                "base_ev_ebitda": 13.0,
                "base_total_debt": 500.0,
                "base_cash": 50.0,
                "base_market_cap": 1200.0,
            },
        ]
    )
    frame.to_parquet(outcomes_path, index=False)

    row = audit._load_historical_outcome_target_row(
        outcomes_path,
        action_id="capital_structure.new_debt_issuance",
        company_id="0000012345",
        snapshot_as_of_time="2024-08-01T00:00:00+00:00",
        source_company_id="src123",
        target_ticker="",
    )

    assert row is not None
    assert row["features"]["market.ev_ebitda"]["value"] == 9.0
    assert row["snapshot_catalog_source"] == "historical_outcome_fallback"


def test_load_snapshot_row_requires_exact_as_of_match_when_requested(tmp_path) -> None:
    snapshot_path = tmp_path / "snapshots.jsonl.gz"
    rows = [
        {"company_id": "0000012345", "as_of_time": "2024-08-15T00:00:00+00:00"},
        {"company_id": "0000012345", "as_of_time": "2024-09-01T00:00:00+00:00"},
    ]
    with gzip.open(snapshot_path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    try:
        audit._load_snapshot_row(
            snapshot_path,
            company_id="0000012345",
            snapshot_as_of_time="2024-08-14T00:00:00+00:00",
        )
    except ValueError as exc:
        assert "snapshot_as_of_time" in str(exc)
    else:
        raise AssertionError("expected exact as-of lookup to fail when no exact match exists")


def test_filter_historical_precedents_as_of_excludes_future_rows() -> None:
    frame = pd.DataFrame(
        [
            {"action_date": "2024-08-13T00:00:00+00:00", "ticker": "OLD"},
            {"action_date": "2024-08-14T00:00:00+00:00", "ticker": "SAME_DAY"},
            {"action_date": "2024-08-15T00:00:00+00:00", "ticker": "FUTURE"},
        ]
    )

    filtered = audit._filter_historical_precedents_as_of(
        frame,
        snapshot_as_of_time="2024-08-14T00:00:00+00:00",
    )

    assert list(filtered["ticker"]) == ["OLD"]


def test_locate_match_row_handles_utc_normalized_dates() -> None:
    frame = pd.DataFrame(
        [
            {
                "company_id": "163627",
                "ticker": "ALLY",
                "action_date": "2014-02-19T00:00:00+00:00",
                "normalized_action_id": "capital_structure.refinancing",
            }
        ]
    )
    case = SimpleNamespace(
        company_id="163627",
        decision_time="2014-02-19 00:00:00",
        action_id="capital_structure.refinancing",
    )

    row = audit._locate_match_row(frame, case)

    assert row["ticker"] == "ALLY"


def test_resolve_snapshot_policy_keeps_requested_cutoff_for_debt_issuance() -> None:
    policy = audit._resolve_snapshot_policy(
        "capital_structure.new_debt_issuance",
        "2024-12-11T00:00:00+00:00",
    )

    assert policy["target_snapshot_as_of_time"] == "2024-12-11T00:00:00+00:00"
    assert policy["historical_precedent_cutoff_time"] == "2024-12-11T00:00:00+00:00"
    assert policy["target_snapshot_cutoff_date"] == "2024-12-11"
    assert policy["cutoff_policy"] == "requested_snapshot_as_of_time"


def test_render_doc_surfaces_cutoff_policy_and_support_tiers() -> None:
    target_values = {key: 0.0 for key in audit._STATE_VECTOR_V1_FEATURES}
    row = {
        "features": {
            "taxonomy.sector": {"value": "Industrials", "support_mode": "exact"},
            "taxonomy.subsector": {"value": "Commercial Services & Supplies", "support_mode": "exact"},
        }
    }
    bundle = {
        "state_vector_v1": {
            "values": target_values,
            "support": {},
            "meta": {"sector": "Industrials", "subsector": "Commercial Services & Supplies"},
        }
    }
    same_company_hist = pd.Series({"ticker": "DBD"})
    peer_hist = pd.Series({"ticker": "XRX"})
    learned = {
        "state_weight_scope": "capital_structure.new_debt_issuance",
        "calibration_confidence": 0.13,
        "confidence_label": "low",
        "retrieval_tier": "global",
        "out_of_sample_flag": True,
        "exact_match_count": 1,
        "minimum_exact_support": 5,
        "top_similarity_mean": 0.42,
        "top_weighted_feature_coverage": 0.81,
        "top_critical_feature_coverage": 0.66,
        "top_action_match_score": 0.91,
        "matches": [
            {
                "precedent_id": "xrx::2024-01-01",
                "company_id": "0000101010",
                "action_id": "capital_structure.new_debt_issuance",
                "decision_time": "2024-01-01T00:00:00+00:00",
                "similarity_score": 0.43,
                "nonnull_compact_features": len(audit._STATE_VECTOR_V1_FEATURES),
                "explanation_lines": ["- Why it matched: `borrower quality`"],
                "historical_row": peer_hist,
            },
            {
                "precedent_id": "dbd::2023-01-01",
                "company_id": "0000028823",
                "action_id": "capital_structure.new_debt_issuance",
                "decision_time": "2023-01-01T00:00:00+00:00",
                "similarity_score": 0.41,
                "nonnull_compact_features": len(audit._STATE_VECTOR_V1_FEATURES),
                "explanation_lines": ["- Why it matched: `same-company history`"],
                "historical_row": same_company_hist,
            },
        ],
    }
    prior_only = dict(learned)
    doc = audit._render_doc(
        company_name="Diebold Nixdorf",
        company_id="0000028823",
        action_id="capital_structure.new_debt_issuance",
        row=row,
        bundle=bundle,
        learned=learned,
        prior_only=prior_only,
        outcomes_path=Path("/tmp/mock_outcomes.parquet"),
        snapshot_path=Path("/tmp/mock_snapshot.jsonl.gz"),
        snapshot_source_note="historical_outcome_fallback",
        target_snapshot_cutoff_date="2024-12-31",
        historical_precedent_cutoff_time="2025-01-01T00:00:00+00:00",
        cutoff_policy="fixed_calendar_year_end_2024",
    )

    assert "- Target snapshot cutoff date: `2024-12-31`" in doc
    assert "- Historical precedent cutoff: `< 2025-01-01T00:00:00+00:00`" in doc
    assert "- Cutoff policy: `fixed_calendar_year_end_2024`" in doc
    assert "## Support Tiers" in doc
    assert "Peer Primary" in doc
    assert "Same Company History Primary" in doc
    assert "No high-confidence precedent set was found" in doc
    assert "- Support tier: `peer_primary`" in doc
    assert "- Support tier: `same_company_history_primary`" in doc
