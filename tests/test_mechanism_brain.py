from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.action_ontology import build_default_action_schema_registry
from src.causal_impact_model import CausalImpactModel
from src.mechanism_brain import MechanismBrain
from src.recommendation_run import RecommendationRunStore, create_recommendation_run


def _write_entity_files(tmp_path: Path) -> tuple[Path, Path]:
    entity_graph = tmp_path / "entity_graph.parquet"
    entity_identifier = tmp_path / "entity_identifier.parquet"

    pd.DataFrame(
        [
            {
                "entity_id": "0000320193",
                "related_id": "001690",
                "valid_from": "2001-01-01T00:00:00Z",
                "effective_at": "2001-01-01T00:00:00Z",
                "published_at": "2001-01-01T00:00:00Z",
                "ingested_at": "2001-01-01T00:00:00Z",
            }
        ]
    ).to_parquet(entity_graph, index=False)

    pd.DataFrame([{"entity_id": "0000320193", "identifier_value": "001690"}]).to_parquet(
        entity_identifier, index=False
    )
    return entity_graph, entity_identifier


def _write_snapshot(tmp_path: Path, features: dict) -> tuple[Path, dict]:
    root = tmp_path / "snapshots"
    keyed = root / "keyed" / "as_of_date=2026-02-28"
    keyed.mkdir(parents=True, exist_ok=True)
    row = {
        "snapshot_id": "snap-123",
        "company_id": "0000320193",
        "as_of_time": "2026-02-28T00:00:00+00:00",
        "features": features,
        "regime": {
            "credit_regime": "neutral",
            "risk_regime": "neutral",
            "vol_regime": "normal",
            "sector_cycle": "neutral",
        },
        "constraint_set": {"hard": [], "soft": []},
        "provenance": {"computation_version": "state_builder_v5"},
    }
    p = keyed / "company_id=0000320193.json"
    p.write_text(json.dumps(row) + "\n")
    return root, row


def _make_run(tmp_path: Path, snapshot_root: Path, objectives: dict | None = None) -> object:
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    store = RecommendationRunStore(root=tmp_path / "runs")
    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        objectives=objectives,
        run_store=store,
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )
    run = store.get_run(run_id)
    assert run is not None
    return run


def _candidate(action_id: str, params: dict) -> dict:
    at, st = action_id.split(".", 1)
    return {
        "candidate_id": "cand-1",
        "candidate_signature": f"sig::{action_id}",
        "action_id": action_id,
        "action_type": at,
        "action_subtype": st,
        "parameters": params,
        "params": params,
        "created_at": "2026-02-28T00:00:00+00:00",
    }


def test_infeasible_action_flagged_for_liquidity_shortfall(tmp_path: Path):
    features = {
        "liquidity.runway_months": {"value": 4.0},
        "liquidity.available_for_actions": {"value": 50_000_000.0},
        "market.market_cap": {"value": 1_000_000_000.0},
        "capital_structure.net_debt": {"value": 300_000_000.0},
        "operating.ebitda_ttm": {"value": 120_000_000.0},
        "capital_structure.maturity_wall_ratio_24m": {"value": 0.10},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    brain = MechanismBrain(action_registry=registry)

    evaluated = brain.evaluate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        candidates=[
            _candidate(
                "capital_return.open_market_buyback",
                {"size_pct_market_cap": 0.10, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
            )
        ],
    )[0]

    assert evaluated.feasibility.feasibility_status == "infeasible"
    assert any(b.blocker_type == "liquidity_shortfall" for b in evaluated.feasibility.blockers)


def test_mechanism_brain_uses_capital_structure_debt_liquidity_aliases_when_context_matches(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv(
        "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
        "normalized_available_liquidity,normalized_net_debt,normalized_net_leverage,normalized_operating_earnings_fill",
    )
    features = {
        "liquidity.runway_months": {"value": 24.0},
        "liquidity.available_liquidity_normalized": {"value": 300_000_000.0, "support_mode": "exact"},
        "market.market_cap": {"value": 1_000_000_000.0},
        "capital_structure.net_debt_normalized": {"value": 300_000_000.0, "support_mode": "exact"},
        "capital_structure.net_leverage_normalized": {"value": 3.0, "support_mode": "exact"},
        "operating.operating_earnings_normalized": {"value": 100_000_000.0, "support_mode": "exact"},
        "capital_structure.maturity_wall_ratio_24m": {"value": 0.10},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    brain = MechanismBrain(action_registry=registry)

    evaluated = brain.evaluate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        candidates=[
            _candidate(
                "capital_structure.new_debt_issuance",
                {"size_pct_market_cap": 0.20, "funding_mix": {"cash": 0.0, "debt": 1.0, "equity": 0.0}},
            )
        ],
    )[0]

    proforma_signal = next(
        (sig for sig in evaluated.feasibility.gating_signals if sig.feature_name == "capital_structure.proforma_leverage"),
        None,
    )
    assert proforma_signal is not None
    assert abs(float(proforma_signal.value) - 5.0) < 1e-6


def test_proforma_leverage_calculation_and_breach(tmp_path: Path):
    features = {
        "liquidity.runway_months": {"value": 24.0},
        "liquidity.available_for_actions": {"value": 300_000_000.0},
        "market.market_cap": {"value": 1_000_000_000.0},
        "capital_structure.net_debt": {"value": 300_000_000.0},
        "capital_structure.net_leverage": {"value": 3.0},
        "operating.ebitda_ttm": {"value": 100_000_000.0},
        "capital_structure.maturity_wall_ratio_24m": {"value": 0.10},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    brain = MechanismBrain(action_registry=registry)

    evaluated = brain.evaluate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        candidates=[
            _candidate(
                "capital_return.open_market_buyback",
                {"size_pct_market_cap": 0.20, "funding_mix": {"cash": 0.0, "debt": 1.0, "equity": 0.0}},
            )
        ],
    )[0]

    proforma_signal = None
    for sig in evaluated.feasibility.gating_signals:
        if sig.feature_name == "capital_structure.proforma_leverage":
            proforma_signal = sig
            break
    assert proforma_signal is not None
    assert abs(float(proforma_signal.value) - 5.0) < 1e-6
    assert any(b.blocker_type == "leverage_breach" for b in evaluated.feasibility.blockers)
    assert evaluated.feasibility.feasibility_status == "infeasible"


def test_dividend_continuity_exception_softens_liquidity_shortfall(tmp_path: Path):
    features = {
        "liquidity.runway_months": {"value": 3.0},
        "liquidity.available_for_actions": {"value": 0.0},
        "liquidity.cash": {"value": 3_625_000.0},
        "market.market_cap": {"value": 3_006_115_060.0},
        "capital_structure.total_debt": {"value": 814_582_000.0},
        "capital_structure.net_debt": {"value": 810_957_000.0},
        "capital_structure.net_leverage": {"value": 2.90},
        "capital_structure.interest_coverage": {"value": 3.35},
        "capital_structure.debt_due_0_12m": {"value": 0.0},
        "capital_structure.debt_due_12_24m": {"value": 0.0},
        "capital_structure.maturity_wall_ratio_24m": {"value": 0.0},
        "strategic.intent.return_capital_priority": {"value": 1.0},
        "strategic.last_action_type": {"value": "buyback"},
        "strategic.action_frequency_24m": {"value": 0.2916666667},
        "strategic.recent_actions_count_24m": {"value": 7.0},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    brain = MechanismBrain(action_registry=registry)

    evaluated = brain.evaluate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        candidates=[_candidate("capital_return.dividend_increase", {"annualized_cash_commitment_usd": 3_006_115.06, "percent_change": 0.02})],
    )[0]

    liquidity_blockers = [b for b in evaluated.feasibility.blockers if b.blocker_type == "liquidity_shortfall"]
    assert liquidity_blockers
    assert all(b.severity == "soft" for b in liquidity_blockers)
    assert evaluated.feasibility.feasibility_status == "conditional"
    assert any(s.feature_name == "capital_return.incremental_quarterly_cash_commitment_usd" for s in evaluated.feasibility.gating_signals)


def test_dividend_continuity_exception_does_not_apply_to_large_commitment(tmp_path: Path):
    features = {
        "liquidity.runway_months": {"value": 3.0},
        "liquidity.available_for_actions": {"value": 0.0},
        "liquidity.cash": {"value": 3_625_000.0},
        "market.market_cap": {"value": 3_006_115_060.0},
        "capital_structure.total_debt": {"value": 814_582_000.0},
        "capital_structure.net_debt": {"value": 810_957_000.0},
        "capital_structure.net_leverage": {"value": 2.90},
        "capital_structure.interest_coverage": {"value": 3.35},
        "capital_structure.debt_due_0_12m": {"value": 0.0},
        "capital_structure.debt_due_12_24m": {"value": 0.0},
        "capital_structure.maturity_wall_ratio_24m": {"value": 0.0},
        "strategic.intent.return_capital_priority": {"value": 1.0},
        "strategic.last_action_type": {"value": "buyback"},
        "strategic.action_frequency_24m": {"value": 0.2916666667},
        "strategic.recent_actions_count_24m": {"value": 7.0},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    brain = MechanismBrain(action_registry=registry)

    evaluated = brain.evaluate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        candidates=[_candidate("capital_return.dividend_increase", {"annualized_cash_commitment_usd": 15_030_575.30, "percent_change": 0.08})],
    )[0]

    assert evaluated.feasibility.feasibility_status == "infeasible"
    assert any(b.blocker_type == "liquidity_shortfall" and b.severity == "hard" for b in evaluated.feasibility.blockers)
    assert not any(s.feature_name == "capital_return.incremental_quarterly_cash_commitment_usd" for s in evaluated.feasibility.gating_signals)


def test_deleveraging_equity_issuance_can_stay_feasible_in_high_vol_when_window_is_still_open(tmp_path: Path):
    features = {
        "liquidity.runway_months": {"value": 18.0},
        "liquidity.available_for_actions": {"value": 50_000_000.0},
        "market.market_cap": {"value": 800_000_000.0},
        "market.equity_window_proxy": {"value": 0.32},
        "market.credit_window_proxy": {"value": 0.10},
        "capital_structure.total_debt": {"value": 350_000_000.0},
        "capital_structure.net_debt": {"value": 320_000_000.0},
        "capital_structure.net_leverage": {"value": 3.6},
        "capital_structure.interest_coverage": {"value": 5.4},
        "capital_structure.maturity_wall_ratio_24m": {"value": 0.18},
        "operating.ebitda_ttm": {"value": 90_000_000.0},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    snapshot["regime"]["vol_regime"] = "high"
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    brain = MechanismBrain(action_registry=registry)

    evaluated = brain.evaluate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        candidates=[
            _candidate(
                "capital_structure.equity_issuance",
                {"amount_usd": 100_000_000.0, "use_of_proceeds": "deleveraging"},
            )
        ],
    )[0]

    assert evaluated.feasibility.feasibility_status == "feasible"
    assert not any(
        b.blocker_type == "market_access_closed" and "Equity-dependent action" in b.explanation
        for b in evaluated.feasibility.blockers
    )


def test_mechanism_rules_trigger_buyback_interaction(tmp_path: Path):
    features = {
        "liquidity.runway_months": {"value": 20.0},
        "liquidity.available_for_actions": {"value": 250_000_000.0},
        "market.market_cap": {"value": 2_000_000_000.0},
        "capital_structure.net_debt": {"value": 200_000_000.0},
        "capital_structure.net_leverage": {"value": 1.5},
        "operating.ebitda_ttm": {"value": 130_000_000.0},
        "operating.fcf_conversion": {"value": 0.45},
        "market.ev_ebitda_vs_peer_z": {"value": -1.3},
        "market.fcf_yield_percentile_peers": {"value": 80.0},
        "capital_structure.maturity_wall_ratio_24m": {"value": 0.10},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    brain = MechanismBrain(action_registry=registry)

    evaluated = brain.evaluate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        candidates=[
            _candidate(
                "capital_return.open_market_buyback",
                {"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
            )
        ],
    )[0]

    undervaluation = None
    for mech in evaluated.mechanism_activation.mechanisms:
        if mech.mechanism_id == "undervaluation_arbitrage":
            undervaluation = mech
            break
    assert undervaluation is not None
    assert undervaluation.activation_strength > 0.7
    assert any(i.direction == "positive" for i in evaluated.mechanism_activation.key_interactions)


def test_impact_distributions_are_well_formed(tmp_path: Path):
    features = {
        "liquidity.runway_months": {"value": 18.0},
        "liquidity.available_for_actions": {"value": 300_000_000.0},
        "market.market_cap": {"value": 2_500_000_000.0},
        "capital_structure.net_debt": {"value": 500_000_000.0},
        "capital_structure.net_leverage": {"value": 2.8},
        "operating.ebitda_ttm": {"value": 180_000_000.0},
        "capital_structure.interest_coverage": {"value": 3.0},
        "capital_structure.maturity_wall_ratio_24m": {"value": 0.30},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    brain = MechanismBrain(action_registry=registry)

    evaluated = brain.evaluate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        candidates=[_candidate("capital_structure.refinancing", {"amount": 250_000_000.0, "new_tenor_years": 5})],
    )[0]

    for dist in evaluated.impact_distribution.objectives.values():
        assert dist.p10 <= dist.p25 <= dist.median <= dist.p75 <= dist.p90


def test_sanity_checks_trigger_objective_contradiction(tmp_path: Path):
    features = {
        "liquidity.runway_months": {"value": 20.0},
        "liquidity.available_for_actions": {"value": 400_000_000.0},
        "market.market_cap": {"value": 1_500_000_000.0},
        "capital_structure.net_debt": {"value": 250_000_000.0},
        "operating.ebitda_ttm": {"value": 150_000_000.0},
        "capital_structure.maturity_wall_ratio_24m": {"value": 0.10},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(
        tmp_path,
        snapshot_root,
        objectives={
            "value_creation_weight": 0.1,
            "risk_reduction_weight": 0.6,
            "growth_weight": 0.1,
            "rating_preservation_weight": 0.1,
            "optionality_weight": 0.1,
        },
    )
    registry = build_default_action_schema_registry("v1.0")
    brain = MechanismBrain(action_registry=registry)

    evaluated = brain.evaluate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        candidates=[
            _candidate(
                "capital_return.open_market_buyback",
                {"size_pct_market_cap": 0.10, "funding_mix": {"cash": 0.0, "debt": 1.0, "equity": 0.0}},
            )
        ],
    )[0]

    contradiction = [s for s in evaluated.structural_sanity_flags if s.check_type == "objective_contradiction"]
    assert contradiction
    assert contradiction[0].status == "fail"


def test_negative_revisions_warn_on_capital_return_actions(tmp_path: Path):
    features = {
        "liquidity.runway_months": {"value": 20.0},
        "liquidity.available_for_actions": {"value": 300_000_000.0},
        "market.market_cap": {"value": 2_000_000_000.0},
        "capital_structure.net_debt": {"value": 250_000_000.0},
        "capital_structure.net_leverage": {"value": 1.8},
        "operating.ebitda_ttm": {"value": 150_000_000.0},
        "operating.fcf_conversion": {"value": 0.35},
        "market.ev_ebitda_vs_peer_z": {"value": -1.2},
        "market.fcf_yield_percentile_peers": {"value": 0.82},
        "capital_structure.maturity_wall_ratio_24m": {"value": 0.10},
        "expectations.analyst_coverage_count": {"value": 11.0},
        "expectations.revision_signal": {"value": -0.09},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    brain = MechanismBrain(action_registry=registry)

    evaluated = brain.evaluate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        candidates=[
            _candidate(
                "capital_return.open_market_buyback",
                {"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
            )
        ],
    )[0]

    expectation_checks = [s for s in evaluated.structural_sanity_flags if s.check_type == "expectations_contradiction"]
    assert expectation_checks
    assert expectation_checks[0].status == "warning"


