from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.action_ontology import build_default_action_schema_registry
from src.candidate_generation import CandidateGenerationEngine, _feature_value
from src.recommendation_run import RecommendationRunStore, create_recommendation_run


def _candidate_action_ids(out: dict) -> list[str]:
    return [row["action_id"] for row in out["candidates"]]


def test_feature_value_blocks_unsupported_metric_inputs():
    features = {
        "capital_structure.net_leverage": {
            "value": 6.0,
            "support_mode": "unsupported",
            "applicability_status": "unsupported",
            "quality_flags": ["unsupported_metric"],
        }
    }
    assert _feature_value(features, "capital_structure.net_leverage") is None
    assert _feature_value(features, "capital_structure.net_leverage", default=1.5) == 1.5


def test_feature_value_prefers_fixed_charge_for_lease_heavy_coverage():
    features = {
        "capital_structure.interest_coverage": {
            "value": 5.0,
            "support_mode": "exact",
            "applicability_status": "secondary",
        },
        "capital_structure.fixed_charge_coverage": {
            "value": 2.25,
            "support_mode": "exact",
            "applicability_status": "primary",
        },
    }
    assert _feature_value(features, "capital_structure.interest_coverage") == 2.25


def test_feature_value_only_applies_global_leverage_aliases_without_action_context(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv(
        "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
        "normalized_net_leverage,normalized_available_liquidity",
    )
    features = {
        "capital_structure.net_leverage_normalized": {"value": 2.1, "support_mode": "exact"},
        "liquidity.available_liquidity_normalized": {"value": 350.0, "support_mode": "exact"},
    }

    assert _feature_value(features, "capital_structure.net_leverage") == 2.1
    assert _feature_value(features, "liquidity.available_for_actions") is None


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

    pd.DataFrame(
        [
            {
                "entity_id": "0000320193",
                "identifier_value": "001690",
            }
        ]
    ).to_parquet(entity_identifier, index=False)

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
        "regime": {"credit_regime": "neutral"},
        "constraint_set": {"hard": [], "soft": []},
        "provenance": {
            "computation_version": "state_builder_v5",
            "inputs_used": {"facts": True, "timeseries": True, "events": True},
        },
    }
    p = keyed / "company_id=0000320193.json"
    p.write_text(json.dumps(row) + "\n")
    return root, row


def _make_run(tmp_path: Path, snapshot_root: Path) -> object:
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    runs_root = tmp_path / "runs"
    store = RecommendationRunStore(root=runs_root)
    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        run_store=store,
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )
    run = store.get_run(run_id)
    assert run is not None
    return run


def _rich_feature_set() -> dict:
    return {
        "capital_structure.maturity_wall_ratio_24m": {"value": 0.35},
        "capital_structure.debt_due_next_24m": {"value": 500_000_000.0},
        "capital_structure.total_debt": {"value": 4_000_000_000.0},
        "capital_structure.net_leverage": {"value": 4.2},
        "capital_structure.interest_coverage": {"value": 2.0},
        "capital_structure.rating_state": {"value": {"rating": "BB+", "outlook": "neg"}},
        "market.credit_window_proxy": {"value": 0.6},
        "market.market_cap": {"value": 5_000_000_000.0},
        "market.equity_window_proxy": {"value": 0.7},
        "market.ev_ebitda_vs_peer_z": {"value": -1.5},
        "market.fcf_yield_percentile_peers": {"value": 0.85},
        "market.volatility_30d": {"value": 0.2},
        "market.conglomerate_discount_signal": {"value": 0.7},
        "liquidity.available_for_actions": {"value": 900_000_000.0},
        "liquidity.runway_months": {"value": 24.0},
        "operating.fcf_conversion": {"value": 0.45},
        "operating.ebitda_margin_ttm": {"value": 0.18},
        "operating.margin_volatility_8q": {"value": 0.15},
        "operating.ebitda_margin_percentile_peers": {"value": 0.2},
        "operating.segment_margin_divergence": {"value": 0.4},
        "operating.revenue_cagr_3y": {"value": 0.01},
        "strategic.constraint_set": {"value": {"hard": [], "soft": []}},
        "strategic.intent_vector": {"value": {"pursue_mna_priority": 0.8}},
        "strategic.intent.pursue_mna_priority": {"value": 0.8},
        "strategic.segment_count": {"value": 3},
        "strategic.segment_references": {"value": ["segment_A", "segment_B", "segment_C"]},
        "peer_context.relative_positioning.market_share_percentile": {"value": 0.2},
        "peer_context.consolidation_wave_score": {"value": 0.8},
        "segment_disclosure": {"value": True},
    }


def _capital_return_feature_set() -> dict:
    features = _rich_feature_set()
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.05}
    features["capital_structure.debt_due_next_24m"] = {"value": 50_000_000.0}
    features["capital_structure.total_debt"] = {"value": 600_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 1.5}
    features["capital_structure.interest_coverage"] = {"value": 8.0}
    features["capital_structure.rating_state"] = {"value": {"rating": "BBB", "outlook": "stable"}}
    return features


def _mna_capacity_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["liquidity.available_for_actions"] = {"value": 900_000_000.0}
    features["market.market_cap"] = {"value": 5_000_000_000.0}
    features["market.credit_window_proxy"] = {"value": 0.7}
    features["market.equity_window_proxy"] = {"value": 0.7}
    return features


def _absolute_maturity_wall_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["market.market_cap"] = {"value": 600_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 60_000_000.0}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 397_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": None}
    features["capital_structure.total_debt"] = {"value": 0.0}
    return features


def _equity_backstop_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.55}
    features["market.market_cap"] = {"value": 9_500_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 125_000_000.0}
    features["capital_structure.total_debt"] = {"value": 350_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 1.25}
    features["capital_structure.interest_coverage"] = {"value": 5.0}
    return features


def _dividend_confidence_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 123_317_673_766.0}
    features["liquidity.available_for_actions"] = {"value": 3_080_620_000.0}
    features["capital_structure.total_debt"] = {"value": 769_000_000.0}
    features["capital_structure.net_debt"] = {"value": -2_610_000_000.0}
    features["capital_structure.net_leverage"] = {"value": -1.1651785714285714}
    features["capital_structure.interest_coverage"] = {"value": 17.23148148148148}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 1_500_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 1_750_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 4.22626788036411}
    features["operating.fcf_conversion"] = {"value": 0.85}
    return features


def _durable_dividend_growth_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 52_477_254_000.0}
    features["liquidity.cash"] = {"value": 2_016_400_000.0}
    features["liquidity.available_for_actions"] = {"value": 1_442_947_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 573_453_000.0}
    features["capital_structure.total_debt"] = {"value": 8_591_500_000.0}
    features["capital_structure.net_debt"] = {"value": 6_575_100_000.0}
    features["capital_structure.net_leverage"] = {"value": 9.095448886429658}
    features["capital_structure.interest_coverage"] = {"value": 173.70149253731344}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": None}
    features["capital_structure.debt_due_12_24m"] = {"value": None}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": None}
    features["operating.fcf_conversion"] = {"value": 4.150643242495504}
    return features


def _near_min_cash_dividend_growth_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.6573364236049207}
    features["market.market_cap"] = {"value": 3_187_306_733.88}
    features["liquidity.cash"] = {"value": 106_200_000.0}
    features["liquidity.available_for_actions"] = {"value": 0.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 108_501_000.0}
    features["capital_structure.total_debt"] = {"value": 1_146_900_000.0}
    features["capital_structure.net_debt"] = {"value": 1_040_700_000.0}
    features["capital_structure.net_leverage"] = {"value": 16.36320754716981}
    features["capital_structure.interest_coverage"] = {"value": 10.365591397849462}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.0}
    features["operating.fcf_conversion"] = {"value": 3.680817610062893}
    return features


def _balance_sheet_pressure_dividend_cut_feature_set() -> dict:
    features = _near_min_cash_dividend_growth_feature_set()
    features["market.drawdown_90d"] = {"value": -0.6420062695924765}
    return features


def _extreme_volatility_dividend_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 21_913_842_960.0}
    features["market.volatility_90d"] = {"value": 25.128087066656516}
    features["market.fcf_yield"] = {"value": 0.019792922710622544}
    features["liquidity.cash"] = {"value": 264_705_000.0}
    features["liquidity.available_for_actions"] = {"value": 215_517_660.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 49_187_340.0}
    features["capital_structure.total_debt"] = {"value": 0.0}
    features["capital_structure.net_debt"] = {"value": -264_705_000.0}
    features["capital_structure.net_leverage"] = {"value": -9.45780334429041}
    features["capital_structure.interest_coverage"] = {"value": 129.51439232409382}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": None}
    features["capital_structure.debt_due_12_24m"] = {"value": None}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": None}
    features["operating.ebitda_margin_ttm"] = {"value": 0.017070246124307596}
    features["operating.fcf_conversion"] = {"value": 15.497320280120052}
    features["operating.revenue_yoy_last_q"] = {"value": 0.0}
    return features


def _anomalous_regular_dividend_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.5}
    features["market.market_cap"] = {"value": 919_534_584.72}
    features["market.drawdown_90d"] = {"value": -0.8321995464852607}
    features["liquidity.cash"] = {"value": 11_742_490.0}
    features["liquidity.available_for_actions"] = {"value": 11_667_490.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 75_000.0}
    features["capital_structure.total_debt"] = {"value": 23_223_090.0}
    features["capital_structure.net_debt"] = {"value": 11_480_600.0}
    features["capital_structure.net_leverage"] = {"value": None}
    features["capital_structure.interest_coverage"] = {"value": -12.246334546770981}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 700_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 30.142414295427525}
    features["operating.ebitda_margin_ttm"] = {"value": -2.3888}
    features["operating.fcf_conversion"] = {"value": -0.04855994641661085}
    features["operating.revenue_yoy_last_q"] = {"value": 11_544.12}
    return features


def _leveraged_dividend_growth_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.34006645680307807}
    features["market.market_cap"] = {"value": 19_683_756_000.0}
    features["market.drawdown_90d"] = {"value": -0.6901066925315228}
    features["liquidity.cash"] = {"value": 112_600_000.0}
    features["liquidity.available_for_actions"] = {"value": 0.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 128_889_000.0}
    features["capital_structure.total_debt"] = {"value": 1_256_200_000.0}
    features["capital_structure.net_debt"] = {"value": 1_143_600_000.0}
    features["capital_structure.net_leverage"] = {"value": 4.766986244268445}
    features["capital_structure.interest_coverage"] = {"value": 10.288930581613508}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 750_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.5970386881069893}
    features["operating.ebitda_margin_ttm"] = {"value": 0.055838744966599164}
    features["operating.fcf_conversion"] = {"value": 2.7027928303459774}
    features["operating.revenue_yoy_last_q"] = {"value": 2.574590232132457}
    return features


def _cash_rich_regular_dividend_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 3_557_278_380.0}
    features["market.drawdown_90d"] = {"value": -0.8692121384812651}
    features["market.volatility_90d"] = {"value": 1.9758993390895303}
    features["liquidity.cash"] = {"value": 498_614_000.0}
    features["liquidity.available_for_actions"] = {"value": 482_519_870.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 16_094_130.0}
    features["capital_structure.total_debt"] = {"value": 0.0}
    features["capital_structure.net_debt"] = {"value": -498_614_000.0}
    features["capital_structure.net_leverage"] = {"value": -8.422533783783784}
    features["capital_structure.interest_coverage"] = {"value": 32.62130177514793}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": None}
    features["capital_structure.debt_due_12_24m"] = {"value": None}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": None}
    features["operating.ebitda_margin_ttm"] = {"value": 0.11035079249390926}
    features["operating.fcf_conversion"] = {"value": 1.5413006756756757}
    features["operating.revenue_yoy_last_q"] = {"value": 0.0}
    return features


def _special_dividend_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.5176704008918035}
    features["market.drawdown_90d"] = {"value": -0.8649724433557868}
    features["market.market_cap"] = {"value": 7_735_520_999.999999}
    features["liquidity.available_for_actions"] = {"value": 436_633_910.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 86_538_090.0}
    features["capital_structure.total_debt"] = {"value": 76_770_000.0}
    features["capital_structure.net_debt"] = {"value": -446_402_000.0}
    features["capital_structure.net_leverage"] = {"value": -0.6773529335896662}
    features["capital_structure.interest_coverage"] = {"value": 2.0248750107484614}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.0}
    features["operating.fcf_conversion"] = {"value": 1.1581651465239537}
    features["operating.ebitda_margin_ttm"] = {"value": 0.2284678342219016}
    return features






def _missing_schedule_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["capital_structure.total_debt"] = {"value": 5_266_024_000.0}
    features["capital_structure.net_debt"] = {"value": 5_143_608_000.0}
    features["capital_structure.net_leverage"] = {"value": 18.181782190817216}
    features["capital_structure.interest_coverage"] = {"value": 2.193585574004143}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": None}
    features["capital_structure.debt_due_12_24m"] = {"value": None}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": None}
    features["liquidity.cash"] = {"value": 122_416_000.0}
    features["liquidity.available_for_actions"] = {"value": 0.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 429_874_050.0}
    features["liquidity.runway_months"] = {"value": 60.0}
    features["market.market_cap"] = {"value": 8_143_100_320.0}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.3831326308747283}
    features["market.drawdown_90d"] = {"value": -0.5640689805719276}
    features["market.volatility_90d"] = {"value": 1.883070918060035}
    features["operating.fcf_conversion"] = {"value": 3.0276706527771395}
    features["operating.ebitda_margin_ttm"] = {"value": 0.019742922374588558}
    features["operating.revenue_yoy_last_q"] = {"value": 3.040983916307948}
    features["strategic.intent.return_capital_priority"] = {"value": 0.0}
    features["strategic.last_action_type"] = {"value": "spinoff"}
    features["strategic.action_frequency_24m"] = {"value": 0.041666666666666664}
    features["strategic.recent_actions_count_24m"] = {"value": 1.0}
    return features


def _stable_debt_bearing_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.14270944219552495}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 240_664_710_932.0}
    features["market.drawdown_90d"] = {"value": -0.6860024798096579}
    features["market.volatility_90d"] = {"value": 0.9152819046437158}
    features["liquidity.cash"] = {"value": 7_685_500_000.0}
    features["liquidity.available_for_actions"] = {"value": 6_923_110_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 762_390_000.0}
    features["capital_structure.total_debt"] = {"value": 24_122_100_000.0}
    features["capital_structure.net_debt"] = {"value": 16_436_600_000.0}
    features["capital_structure.net_leverage"] = {"value": 1.8785329782735407}
    features["capital_structure.interest_coverage"] = {"value": 11.194579351402162}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 2_450_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 2_250_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.19484207428043163}
    features["operating.ebitda_margin_ttm"] = {"value": 0.34430016133474994}
    features["operating.fcf_conversion"] = {"value": 0.8150908031132496}
    features["operating.revenue_yoy_last_q"] = {"value": 0.0}
    return features


def _large_cap_coverage_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 117_919_000_000.0}
    features["market.drawdown_90d"] = {"value": -0.9342}
    features["market.volatility_90d"] = {"value": 1.7026}
    features["liquidity.cash"] = {"value": 886_591_000.0}
    features["liquidity.available_for_actions"] = {"value": 865_283_650.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 21_307_350.0}
    features["liquidity.runway_months"] = {"value": 60.0}
    features["capital_structure.total_debt"] = {"value": 3_131_676_000.0}
    features["capital_structure.net_debt"] = {"value": 2_245_085_000.0}
    features["capital_structure.net_leverage"] = {"value": 9.6609}
    features["capital_structure.interest_coverage"] = {"value": 7.0225}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 1_250_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.3991}
    features["operating.ebitda_margin_ttm"] = {"value": 0.3272}
    features["operating.fcf_conversion"] = {"value": 0.4659}
    features["operating.revenue_yoy_last_q"] = {"value": 0.0}
    return features


def _no_maturity_pressure_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 2_834_518_720.0}
    features["market.drawdown_90d"] = {"value": -0.8536}
    features["market.volatility_90d"] = {"value": 2.5625}
    features["liquidity.cash"] = {"value": 49_968_000.0}
    features["liquidity.available_for_actions"] = {"value": 39_399_690.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 10_568_310.0}
    features["liquidity.runway_months"] = {"value": 60.0}
    features["capital_structure.total_debt"] = {"value": 901_217_000.0}
    features["capital_structure.net_debt"] = {"value": 851_249_000.0}
    features["capital_structure.net_leverage"] = {"value": 20.321}
    features["capital_structure.interest_coverage"] = {"value": 2.063}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.0}
    features["operating.ebitda_margin_ttm"] = {"value": 0.1189}
    features["operating.fcf_conversion"] = {"value": 0.1725}
    features["operating.revenue_yoy_last_q"] = {"value": -0.2871}
    return features


def _low_margin_no_maturity_regular_payer_feature_set() -> dict:
    features = _no_maturity_pressure_regular_payer_feature_set()
    features["market.market_cap"] = {"value": 12_950_437_710.0}
    features["liquidity.cash"] = {"value": 349_825_000.0}
    features["liquidity.available_for_actions"] = {"value": 317_391_370.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 32_433_630.0}
    features["capital_structure.total_debt"] = {"value": 667_287_000.0}
    features["capital_structure.net_debt"] = {"value": 317_462_000.0}
    features["capital_structure.net_leverage"] = {"value": 6.379863344051447}
    features["capital_structure.interest_coverage"] = {"value": 3.0534564357203817}
    features["market.drawdown_90d"] = {"value": -0.9472207009857613}
    features["market.volatility_90d"] = {"value": 1.6956828858311133}
    features["operating.fcf_conversion"] = {"value": 2.7572146302250804}
    features["operating.ebitda_margin_ttm"] = {"value": 0.046026300478854816}
    features["operating.revenue_yoy_last_q"] = {"value": 3.052390305338361}
    return features


def _schedule_anomaly_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 51_029_750_340.0}
    features["market.drawdown_90d"] = {"value": -0.9639226558612355}
    features["market.volatility_90d"] = {"value": 38.244284987211536}
    features["market.volatility_30d"] = {"value": 65.41545849042024}
    features["liquidity.cash"] = {"value": 1_583_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 1_493_420_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 89_580_000.0}
    features["capital_structure.total_debt"] = {"value": 1_951_000_000.0}
    features["capital_structure.net_debt"] = {"value": 368_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 0.5644171779141104}
    features["capital_structure.interest_coverage"] = {"value": 3.6288659793814433}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 2_350_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 826_192_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 1.6279815479241415}
    features["operating.ebitda_margin_ttm"] = {"value": 0.21835231078365708}
    features["operating.fcf_conversion"] = {"value": 0.9263803680981595}
    features["operating.revenue_yoy_last_q"] = {"value": 0.0}
    return features


def _coverage_outlier_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 15_334_172_000.0}
    features["market.drawdown_90d"] = {"value": -0.6147547762798776}
    features["liquidity.cash"] = {"value": 293_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 3_560_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 289_440_000.0}
    features["capital_structure.total_debt"] = {"value": 7_039_000_000.0}
    features["capital_structure.net_debt"] = {"value": 6_746_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 20.567073170731707}
    features["capital_structure.interest_coverage"] = {"value": 20.057971014492754}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 800_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.1136525074584458}
    features["operating.ebitda_margin_ttm"] = {"value": 0.03399668325041459}
    features["operating.fcf_conversion"] = {"value": 5.634146341463414}
    features["operating.revenue_yoy_last_q"] = {"value": 3.336179775280899}
    return features


def _financing_anomaly_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.27}
    features["market.market_cap"] = {"value": 24_000_000_000.0}
    features["market.drawdown_90d"] = {"value": -0.854}
    features["liquidity.cash"] = {"value": 688_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 0.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 1_240_000_000.0}
    features["capital_structure.total_debt"] = {"value": 715_000_000.0}
    features["capital_structure.net_debt"] = {"value": 27_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 0.009}
    features["capital_structure.interest_coverage"] = {"value": 7.4}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 1_250_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 800_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 2.867132867132867}
    features["operating.fcf_conversion"] = {"value": 0.95}
    features["operating.ebitda_margin_ttm"] = {"value": 0.18}
    features["operating.revenue_yoy_last_q"] = {"value": 0.02}
    return features


def _weak_fcf_financing_anomaly_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 8_665_891_860.0}
    features["market.drawdown_90d"] = {"value": -0.8490164050848661}
    features["market.volatility_90d"] = {"value": 2.215284578116092}
    features["liquidity.cash"] = {"value": 73_799_000.0}
    features["liquidity.available_for_actions"] = {"value": 24_520_190.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 49_278_810.0}
    features["capital_structure.total_debt"] = {"value": 348_452_000.0}
    features["capital_structure.net_debt"] = {"value": 274_653_000.0}
    features["capital_structure.net_leverage"] = {"value": 2.251567841420526}
    features["capital_structure.interest_coverage"] = {"value": 30.2278431372549}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 484_290_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 1.3898327459736204}
    features["operating.fcf_conversion"] = {"value": 0.26238902142101767}
    features["operating.ebitda_margin_ttm"] = {"value": 0.0742609247260638}
    features["operating.revenue_yoy_last_q"] = {"value": 0.0}
    return features


def _no_market_dislocation_financing_anomaly_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.11936164230699982}
    features["market.market_cap"] = {"value": 37_514_695_350.0}
    features["market.drawdown_90d"] = {"value": None}
    features["market.volatility_90d"] = {"value": None}
    features["liquidity.cash"] = {"value": 3_074_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 104_750_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 2_969_250_000.0}
    features["capital_structure.total_debt"] = {"value": 44_000_000.0}
    features["capital_structure.net_debt"] = {"value": -3_030_000_000.0}
    features["capital_structure.net_leverage"] = {"value": -0.4065476989131893}
    features["capital_structure.interest_coverage"] = {"value": 13.396774193548387}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 1_998_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 500_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 56.77272727272727}
    features["operating.fcf_conversion"] = {"value": 1.0990205286461827}
    features["operating.ebitda_margin_ttm"] = {"value": 0.07530184389997474}
    features["operating.revenue_yoy_last_q"] = {"value": 0.16395987440170287}
    return features


def _distressed_private_placement_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.0}
    features["market.market_cap"] = {"value": 50_834_400.0}
    features["market.drawdown_90d"] = {"value": -0.9414551607445009}
    features["market.volatility_30d"] = {"value": 5.191502294826686}
    features["liquidity.cash"] = {"value": 15_087_000.0}
    features["liquidity.available_for_actions"] = {"value": 0.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 20_018_400.0}
    features["capital_structure.total_debt"] = {"value": 820_073_000.0}
    features["capital_structure.net_debt"] = {"value": 804_986_000.0}
    features["capital_structure.interest_coverage"] = {"value": 0.0019606517476844026}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": None}
    features["capital_structure.debt_due_12_24m"] = {"value": None}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": None}
    features["operating.ebitda_margin_ttm"] = {"value": -0.018593394077448748}
    features["operating.fcf_conversion"] = {"value": -3.670911582171355}
    features["operating.revenue_yoy_last_q"] = {"value": 6.4075554223421145}
    return features


def _relaxed_window_equity_backstop_feature_set() -> dict:
    features = _equity_backstop_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["market.equity_window_proxy"] = {"value": 1.0 / 3.0}
    features["market.drawdown_90d"] = {"value": -0.98}
    return features


def _distressed_nonpayer_public_recap_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.5}
    features["market.market_cap"] = {"value": 213_480_000.0}
    features["market.drawdown_90d"] = {"value": -0.9776559865092749}
    features["market.volatility_90d"] = {"value": 56.90818708301117}
    features["liquidity.cash"] = {"value": 234_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 221_454_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 12_546_000.0}
    features["capital_structure.total_debt"] = {"value": 247_600_000.0}
    features["capital_structure.net_debt"] = {"value": 13_600_000.0}
    features["capital_structure.interest_coverage"] = {"value": -19.36}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.0}
    features["operating.ebitda_margin_ttm"] = {"value": -0.27235772357723576}
    features["operating.fcf_conversion"] = {"value": -0.47146619841966636}
    features["operating.revenue_yoy_last_q"] = {"value": 1.648511716276124}
    return features


def _distressed_nonpayer_private_recap_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.0}
    features["market.market_cap"] = {"value": 2_877_647_620.76}
    features["market.drawdown_90d"] = {"value": -0.850457984866587}
    features["market.volatility_90d"] = {"value": 2.6754643248993712}
    features["liquidity.cash"] = {"value": 149_300_000.0}
    features["liquidity.available_for_actions"] = {"value": 127_133_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 22_167_000.0}
    features["capital_structure.total_debt"] = {"value": 3_900_000.0}
    features["capital_structure.net_debt"] = {"value": -145_400_000.0}
    features["capital_structure.interest_coverage"] = {"value": -12.650684931506849}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 250_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 500_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.19230769230769232}
    features["operating.ebitda_margin_ttm"] = {"value": -0.01204493165516308}
    features["operating.fcf_conversion"] = {"value": -19.831460674157302}
    features["operating.revenue_yoy_last_q"] = {"value": -0.2515952597994531}
    return features


def _nonpayer_public_recap_preference_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 1_029_471_300.0}
    features["market.drawdown_90d"] = {"value": -0.8851228978007761}
    features["market.volatility_90d"] = {"value": 3.4605519232574187}
    features["liquidity.cash"] = {"value": 295_899_330.0}
    features["liquidity.available_for_actions"] = {"value": 295_899_330.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 0.0}
    features["capital_structure.total_debt"] = {"value": 448_054_000.0}
    features["capital_structure.net_debt"] = {"value": 63_187_000.0}
    features["capital_structure.net_leverage"] = {"value": 1.9416464370217865}
    features["capital_structure.interest_coverage"] = {"value": 1.5130803151166823}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 445_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.9931838573029144}
    features["operating.ebitda_margin_ttm"] = {"value": 0.010973536791510894}
    features["operating.fcf_conversion"] = {"value": 1.1265095412223827}
    features["operating.revenue_yoy_last_q"] = {"value": 3.008018499447233}
    return features


def _nonpayer_public_recap_low_stress_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 69_071_149_770.37}
    features["market.drawdown_90d"] = {"value": -0.941210032357939}
    features["market.volatility_90d"] = {"value": 2.399449240862341}
    features["liquidity.cash"] = {"value": 751_023_560.0}
    features["liquidity.available_for_actions"] = {"value": 751_023_560.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 0.0}
    features["capital_structure.total_debt"] = {"value": 1_090_578_000.0}
    features["capital_structure.net_debt"] = {"value": 321_808_000.0}
    features["capital_structure.net_leverage"] = {"value": 1.842071219640639}
    features["capital_structure.interest_coverage"] = {"value": 20.77622276727745}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 500_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.4584724797309317}
    features["operating.ebitda_margin_ttm"] = {"value": 0.29532514690270273}
    features["operating.fcf_conversion"] = {"value": 3.979456093051477}
    features["operating.revenue_yoy_last_q"] = {"value": -0.06977793856119875}
    return features


def _market_shutdown_regular_payer_recap_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.09}
    features["market.ev_ebitda_vs_peer_z"] = {"value": -0.2}
    features["market.fcf_yield_percentile_peers"] = {"value": 0.4}
    features["market.market_cap"] = {"value": 4_726_330_000.0}
    features["market.drawdown_90d"] = {"value": -0.8383333333333334}
    features["market.volatility_90d"] = {"value": 2.4365760788302686}
    features["liquidity.cash"] = {"value": 1_476_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 982_710_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 493_290_000.0}
    features["capital_structure.total_debt"] = {"value": 6_295_000_000.0}
    features["capital_structure.net_debt"] = {"value": 4_819_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 2.7631880733944953}
    features["capital_structure.interest_coverage"] = {"value": 3.8013698630136985}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 700_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.11119936457505956}
    features["operating.ebitda_margin_ttm"] = {"value": 0.10606337043118652}
    features["operating.fcf_conversion"] = {"value": 1.1548165137614679}
    return features


def _market_shutdown_low_debt_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.13}
    features["market.market_cap"] = {"value": 202_299_210.0}
    features["market.drawdown_90d"] = {"value": -0.7268190175991595}
    features["market.volatility_90d"] = {"value": 2.379903638473696}
    features["liquidity.cash"] = {"value": 150_725_000.0}
    features["liquidity.available_for_actions"] = {"value": 148_801_490.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 1_923_510.0}
    features["capital_structure.total_debt"] = {"value": 0.0}
    features["capital_structure.net_debt"] = {"value": -150_725_000.0}
    features["capital_structure.net_leverage"] = {"value": -23.310392824002474}
    features["capital_structure.interest_coverage"] = {"value": 28.906666666666666}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 195_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": None}
    features["operating.ebitda_margin_ttm"] = {"value": 0.10084688928053402}
    features["operating.fcf_conversion"] = {"value": 0.8829260748530776}
    return features


def _strategic_regular_payer_recap_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["strategic.last_action_type"] = {"value": "debt_issuance"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.34}
    features["market.market_cap"] = {"value": 1_526_663_300.0}
    features["market.drawdown_90d"] = {"value": -0.6838145587637251}
    features["liquidity.cash"] = {"value": 5_246_000.0}
    features["liquidity.available_for_actions"] = {"value": 0.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 9_000_000.0}
    features["capital_structure.total_debt"] = {"value": 1_731_521_000.0}
    features["capital_structure.net_debt"] = {"value": 1_726_275_000.0}
    features["capital_structure.net_leverage"] = {"value": 8.587450192266557}
    features["capital_structure.interest_coverage"] = {"value": 9.987831730212076}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.0}
    features["operating.ebitda_margin_ttm"] = {"value": 0.124}
    features["operating.fcf_conversion"] = {"value": 0.668251891574596}
    features["operating.revenue_yoy_last_q"] = {"value": -0.08}
    return features


def _strategic_regular_payer_buyback_feature_set() -> dict:
    features = _strategic_regular_payer_recap_feature_set()
    features["strategic.last_action_type"] = {"value": "buyback"}
    return features


def _buyback_regular_payer_recap_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["strategic.last_action_type"] = {"value": "buyback"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.5}
    features["market.market_cap"] = {"value": 2_486_200_000.0}
    features["liquidity.cash"] = {"value": 556_400_000.0}
    features["liquidity.available_for_actions"] = {"value": 541_196_000.0}
    features["liquidity.runway_months"] = {"value": 60.0}
    features["capital_structure.total_debt"] = {"value": 999_000_000.0}
    features["capital_structure.net_debt"] = {"value": 442_600_000.0}
    features["capital_structure.net_leverage"] = {"value": 3.842013888888889}
    features["capital_structure.interest_coverage"] = {"value": 8.51937984496124}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 187_500_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.18768768768768768}
    features["operating.fcf_conversion"] = {"value": 0.8237847222222222}
    return features


def _strategic_nonpayer_recap_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["strategic.last_action_type"] = {"value": "buyback"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 5_663_112_000.0}
    features["market.drawdown_90d"] = {"value": -0.8511376783648283}
    features["liquidity.cash"] = {"value": 372_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 343_785_000.0}
    features["liquidity.runway_months"] = {"value": 60.0}
    features["capital_structure.total_debt"] = {"value": 238_700_000.0}
    features["capital_structure.net_debt"] = {"value": -133_300_000.0}
    features["capital_structure.net_leverage"] = {"value": -1.9777448071216617}
    features["capital_structure.interest_coverage"] = {"value": 98.875}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 1_025_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 4.294093003770423}
    features["operating.fcf_conversion"] = {"value": 1.4792284866468843}
    return features


def _low_quality_strategic_nonpayer_feature_set() -> dict:
    features = _strategic_nonpayer_recap_feature_set()
    features["capital_structure.interest_coverage"] = {"value": 3.0}
    features["operating.fcf_conversion"] = {"value": 0.3}
    return features


def _growth_missing_liquidity_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["market.credit_window_proxy"] = {"value": 0.7}
    features["market.equity_window_proxy"] = {"value": 0.7}
    features["market.ev_ebitda_vs_peer_z"] = {"value": -0.2}
    features["market.fcf_yield_percentile_peers"] = {"value": 0.4}
    features["liquidity.available_for_actions"] = {"value": 1_500_000_000.0}
    features["market.market_cap"] = {"value": 40_000_000_000.0}
    features["operating.revenue_cagr_3y"] = {"value": None}
    features["strategic.intent_vector"] = {"value": {"pursue_mna_priority": 0.0}}
    features["strategic.intent.pursue_mna_priority"] = {"value": 0.0}
    features["strategic.segment_count"] = {"value": 1}
    features["strategic.segment_references"] = {"value": ["segment_A"]}
    features["peer_context.consolidation_wave_score"] = {"value": 0.0}
    return features


def _dividend_cut_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 15_265_996_999.999998}
    features["liquidity.cash"] = {"value": 100_110_000.0}
    features["liquidity.available_for_actions"] = {"value": 0.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 139_113_480.0}
    features["capital_structure.total_debt"] = {"value": 940_785_000.0}
    features["capital_structure.net_debt"] = {"value": 840_675_000.0}
    features["capital_structure.net_leverage"] = {"value": 5.921706054309161}
    features["capital_structure.interest_coverage"] = {"value": 5.097834857450242}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.0}
    features["operating.fcf_conversion"] = {"value": 0.5368506321980769}
    return features


def _maturity_wall_dividend_cut_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 5_519_486_050.0}
    features["market.drawdown_90d"] = {"value": -0.8821873460482793}
    features["market.volatility_90d"] = {"value": 2.2166079812258888}
    features["liquidity.cash"] = {"value": 184_496_000.0}
    features["liquidity.available_for_actions"] = {"value": 75_493_550.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 109_002_450.0}
    features["liquidity.runway_months"] = {"value": 60.0}
    features["capital_structure.total_debt"] = {"value": 344_589_000.0}
    features["capital_structure.net_debt"] = {"value": 160_093_000.0}
    features["capital_structure.net_leverage"] = {"value": 7.082820864487015}
    features["capital_structure.interest_coverage"] = {"value": 4.5796644771879995}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 350_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 1.0157027647429255}
    features["operating.fcf_conversion"] = {"value": 4.917887006149626}
    features["operating.ebitda_margin_ttm"] = {"value": 0.006220869347432099}
    features["operating.revenue_yoy_last_q"] = {"value": 3.145202416332491}
    return features


def _buyback_reset_dividend_cut_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["strategic.last_action_type"] = {"value": "buyback"}
    features["market.market_cap"] = {"value": 41_542_328_467.2}
    features["market.credit_window_proxy"] = {"value": None}
    features["market.equity_window_proxy"] = {"value": 0.5093425745784695}
    features["liquidity.cash"] = {"value": None}
    features["liquidity.available_for_actions"] = {"value": None}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": None}
    features["capital_structure.total_debt"] = {"value": 7_317_000_000.0}
    features["capital_structure.net_debt"] = {"value": None}
    features["capital_structure.net_leverage"] = {"value": None}
    features["capital_structure.interest_coverage"] = {"value": 6.881533101045296}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 500_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.06833401667350007}
    features["operating.fcf_conversion"] = {"value": -0.7716455696202532}
    features["operating.ebitda_margin_ttm"] = {"value": 0.13789010682119668}
    return features


def _real_financing_stress_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.52}
    features["market.market_cap"] = {"value": 4_500_000_000.0}
    features["market.drawdown_90d"] = {"value": -0.78}
    features["liquidity.cash"] = {"value": 110_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 15_000_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 150_000_000.0}
    features["capital_structure.total_debt"] = {"value": 2_400_000_000.0}
    features["capital_structure.net_debt"] = {"value": 2_290_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 6.4}
    features["capital_structure.interest_coverage"] = {"value": 1.1}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 500_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 400_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.375}
    features["capital_structure.rating_state"] = {"value": {"rating": "BB", "outlook": "neg"}}
    features["operating.fcf_conversion"] = {"value": 0.18}
    features["operating.ebitda_margin_ttm"] = {"value": 0.07}
    features["operating.revenue_yoy_last_q"] = {"value": -0.08}
    return features


def _mild_maturity_wall_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.7899059015422136}
    features["market.equity_window_proxy"] = {"value": 0.7602797553111594}
    features["market.market_cap"] = {"value": 25_931_450_290.88}
    features["market.drawdown_90d"] = {"value": -0.06351573520828346}
    features["market.volatility_90d"] = {"value": 0.32334714253693525}
    features["liquidity.cash"] = {"value": 397_200_000.0}
    features["liquidity.available_for_actions"] = {"value": 1_147_200_000.0}
    features["capital_structure.total_debt"] = {"value": 1_568_100_000.0}
    features["capital_structure.net_debt"] = {"value": 1_170_900_000.0}
    features["capital_structure.net_leverage"] = {"value": 0.8801774035931744}
    features["capital_structure.interest_coverage"] = {"value": 19.1685878962536}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 200_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 200_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.2550857725910337}
    features["operating.fcf_conversion"] = {"value": 0.4118619860181914}
    features["operating.ebitda_margin_ttm"] = {"value": 0.23635071511059785}
    features["operating.revenue_yoy_last_q"] = {"value": None}
    return features


def _continuity_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 2.0 / 3.0}
    features["market.market_cap"] = {"value": 2_632_476_600.0}
    features["market.drawdown_90d"] = {"value": -0.9906321315850718}
    features["market.volatility_90d"] = {"value": 228.45575104408334}
    features["liquidity.cash"] = {"value": 87_000.0}
    features["liquidity.available_for_actions"] = {"value": 0.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 27_609_990.0}
    features["liquidity.runway_months"] = {"value": 60.0}
    features["capital_structure.total_debt"] = {"value": 204_484_000.0}
    features["capital_structure.net_debt"] = {"value": 204_397_000.0}
    features["capital_structure.net_leverage"] = {"value": 2.3577105417969157}
    features["capital_structure.interest_coverage"] = {"value": 16.1884695531373}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.0}
    features["operating.fcf_conversion"] = {"value": 0.7942625125442654}
    features["operating.ebitda_margin_ttm"] = {"value": 0.0941974263663261}
    features["operating.revenue_yoy_last_q"] = {"value": 2.7011851571831302}
    return features


def _net_cash_continuity_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.0}
    features["market.equity_window_proxy"] = {"value": 0.45790279372368925}
    features["market.market_cap"] = {"value": 5_201_162_941.8}
    features["market.drawdown_90d"] = {"value": -0.7253012048192771}
    features["market.volatility_90d"] = {"value": 2.551628756369442}
    features["liquidity.cash"] = {"value": 179_317_000.0}
    features["liquidity.available_for_actions"] = {"value": 148_406_470.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 30_910_530.0}
    features["liquidity.runway_months"] = {"value": 60.0}
    features["capital_structure.total_debt"] = {"value": 0.0}
    features["capital_structure.net_debt"] = {"value": -179_317_000.0}
    features["capital_structure.net_leverage"] = {"value": -2.011024257853242}
    features["capital_structure.interest_coverage"] = {"value": 446.7471264367816}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": None}
    features["operating.fcf_conversion"] = {"value": 0.3904359236040239}
    features["operating.ebitda_margin_ttm"] = {"value": 0.08654041195670213}
    features["operating.revenue_yoy_last_q"] = {"value": None}
    return features


def _healthy_low_growth_regular_payer_feature_set() -> dict:
    features = _dividend_confidence_feature_set()
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["operating.revenue_cagr_3y"] = {"value": -0.20}
    features["strategic.intent_vector"] = {"value": {"pursue_mna_priority": 0.0}}
    features["strategic.intent.pursue_mna_priority"] = {"value": 0.0}
    features["strategic.segment_count"] = {"value": 1}
    features["market.conglomerate_discount_signal"] = {"value": 0.0}
    return features


def _high_coverage_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["strategic.last_action_type"] = {"value": "buyback"}
    features["market.market_cap"] = {"value": 47_340_149_525.55}
    features["market.credit_window_proxy"] = {"value": 0.7184835168640267}
    features["market.equity_window_proxy"] = {"value": 0.6355018752847249}
    features["market.drawdown_90d"] = {"value": -0.0966395080903435}
    features["market.volatility_90d"] = {"value": 0.35249623981431105}
    features["liquidity.available_for_actions"] = {"value": 1_970_300_000.0}
    features["liquidity.cash"] = {"value": 470_300_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 49_525_000.0}
    features["capital_structure.total_debt"] = {"value": 3_224_000_000.0}
    features["capital_structure.net_debt"] = {"value": 2_753_700_000.0}
    features["capital_structure.net_leverage"] = {"value": 1.713565650280025}
    features["capital_structure.interest_coverage"] = {"value": 40.994897959183675}
    features["capital_structure.debt_due_0_12m"] = {"value": 300_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.09305210918114144}
    features["operating.fcf_conversion"] = {"value": 0.18481642812694463}
    features["operating.ebitda_margin_ttm"] = {"value": 0.1926396547590506}
    features["operating.revenue_cagr_3y"] = {"value": 0.05551837854168662}
    return features


def _mega_cap_high_coverage_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["strategic.last_action_type"] = {"value": "buyback"}
    features["market.market_cap"] = {"value": 887_295_454_500.0}
    features["market.credit_window_proxy"] = {"value": 0.9150952178444158}
    features["market.equity_window_proxy"] = {"value": 0.9358076425872679}
    features["market.drawdown_90d"] = {"value": -0.14194139489982016}
    features["market.volatility_90d"] = {"value": 0.2621253960496523}
    features["liquidity.available_for_actions"] = {"value": 5_881_000_000.0}
    features["liquidity.cash"] = {"value": 2_483_000_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 183_050_000.0}
    features["capital_structure.total_debt"] = {"value": 14_048_000_000.0}
    features["capital_structure.net_debt"] = {"value": 9_167_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 1.1263054429291068}
    features["capital_structure.interest_coverage"] = {"value": 70.16379310344827}
    features["capital_structure.debt_due_0_12m"] = {"value": 1_050_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.07474373576309795}
    features["operating.fcf_conversion"] = {"value": -0.31318343776876767}
    features["operating.ebitda_margin_ttm"] = {"value": 0.4837476099426386}
    features["operating.revenue_cagr_3y"] = {"value": -0.1359799797184732}
    return features


def _coverage_supported_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.credit_window_proxy"] = {"value": 0.7162147603111676}
    features["market.equity_window_proxy"] = {"value": 0.6153600679746636}
    features["market.market_cap"] = {"value": 23_065_204_794.46}
    features["market.drawdown_90d"] = {"value": -0.05993431855500819}
    features["market.volatility_90d"] = {"value": 0.31680118793620654}
    features["liquidity.cash"] = {"value": 1_229_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 3_679_000_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 250_000_000.0}
    features["capital_structure.total_debt"] = {"value": 8_830_000_000.0}
    features["capital_structure.net_debt"] = {"value": 7_601_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 2.683968926553672}
    features["capital_structure.interest_coverage"] = {"value": 13.11111111111111}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 1_250_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 800_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.232163080407701}
    features["operating.fcf_conversion"] = {"value": 0.06426553672316385}
    features["operating.ebitda_margin_ttm"] = {"value": 0.052019617567642035}
    features["operating.revenue_cagr_3y"] = {"value": 0.07008533328578515}
    return features


def _preemptive_deleveraging_dividend_cut_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.market_cap"] = {"value": 18_196_560_000.0}
    features["market.credit_window_proxy"] = {"value": 0.7081801682041196}
    features["market.equity_window_proxy"] = {"value": 0.6548303273494266}
    features["market.drawdown_90d"] = {"value": -0.1555610601932127}
    features["market.volatility_90d"] = {"value": 0.3777520419406822}
    features["liquidity.cash"] = {"value": 3_026_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 6_312_960_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 500_000_000.0}
    features["capital_structure.total_debt"] = {"value": 13_726_000_000.0}
    features["capital_structure.net_debt"] = {"value": 10_700_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 5.201750121536218}
    features["capital_structure.interest_coverage"] = {"value": 19.97087378640777}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 3_984_588_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 750_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.3449357423867114}
    features["operating.fcf_conversion"] = {"value": -0.17744287797763733}
    features["operating.ebitda_margin_ttm"] = {"value": 0.11692650334075724}
    return features


def _sparse_data_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.market_cap"] = {"value": 5_189_187_706.78}
    features["market.credit_window_proxy"] = {"value": 0.8774575396913192}
    features["market.equity_window_proxy"] = {"value": 0.7591570195515213}
    features["market.drawdown_90d"] = {"value": -0.09733201581027662}
    features["market.volatility_90d"] = {"value": 0.17212376172507898}
    features["liquidity.available_for_actions"] = {"value": 754_420_000.0}
    features["capital_structure.total_debt"] = {"value": 2_198_376_000.0}
    features["capital_structure.net_debt"] = {"value": 2_193_956_000.0}
    features["capital_structure.net_leverage"] = {"value": 3.9289057939091787}
    features["capital_structure.interest_coverage"] = {"value": 20.42928221262896}
    features["capital_structure.debt_due_0_12m"] = {"value": None}
    features["capital_structure.debt_due_12_24m"] = {"value": None}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": None}
    features["operating.fcf_conversion"] = {"value": None}
    features["operating.ebitda_margin_ttm"] = {"value": None}
    features["operating.revenue_cagr_3y"] = {"value": None}
    return features


def _coverage_gap_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.market_cap"] = {"value": 10_325_962_368.08}
    features["market.credit_window_proxy"] = {"value": 0.9403787079850169}
    features["market.equity_window_proxy"] = {"value": 0.6493341798224167}
    features["market.drawdown_90d"] = {"value": -0.009120456026058599}
    features["market.volatility_90d"] = {"value": 0.2866755549851585}
    features["liquidity.available_for_actions"] = {"value": 116_900_000.0}
    features["capital_structure.total_debt"] = {"value": 5_876_000_000.0}
    features["capital_structure.net_debt"] = {"value": 5_759_100_000.0}
    features["capital_structure.net_leverage"] = {"value": 3.9050040683482505}
    features["capital_structure.interest_coverage"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 300_000_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 1_750_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.3488767869298843}
    features["operating.fcf_conversion"] = {"value": 0.29719283970707894}
    features["operating.ebitda_margin_ttm"] = {"value": 0.04842522787569939}
    features["operating.revenue_cagr_3y"] = {"value": None}
    return features


def _balance_sheet_light_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.market_cap"] = {"value": 25_931_450_290.88}
    features["market.credit_window_proxy"] = {"value": None}
    features["market.equity_window_proxy"] = {"value": 0.7717284445028556}
    features["market.drawdown_90d"] = {"value": None}
    features["market.volatility_90d"] = {"value": None}
    features["liquidity.available_for_actions"] = {"value": 1_147_200_000.0}
    features["liquidity.cash"] = {"value": 397_200_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 0.0}
    features["capital_structure.total_debt"] = {"value": 2_021_900_000.0}
    features["capital_structure.net_debt"] = {"value": 1_624_700_000.0}
    features["capital_structure.net_leverage"] = {"value": 1.221303465383748}
    features["capital_structure.interest_coverage"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 400_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.19783372075770314}
    features["operating.fcf_conversion"] = {"value": 0.4118619860181914}
    features["operating.ebitda_margin_ttm"] = {"value": 0.23635071511059785}
    features["operating.revenue_cagr_3y"] = {"value": None}
    features["strategic.last_action_type"] = {"value": "buyback"}
    return features


def _balance_sheet_light_regular_payer_negative_feature_set() -> dict:
    features = _balance_sheet_light_regular_payer_feature_set()
    features["liquidity.available_for_actions"] = {"value": None}
    features["liquidity.cash"] = {"value": None}
    features["operating.fcf_conversion"] = {"value": -0.15}
    return features


def _liquidity_supported_regular_payer_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["market.market_cap"] = {"value": 1_483_105_344.84}
    features["market.credit_window_proxy"] = {"value": 0.8296065679699798}
    features["market.equity_window_proxy"] = {"value": 0.7493464073683587}
    features["market.drawdown_90d"] = {"value": -0.10385558279474028}
    features["market.volatility_90d"] = {"value": 0.25019076442627564}
    features["liquidity.available_for_actions"] = {"value": 745_700_000.0}
    features["capital_structure.total_debt"] = {"value": 3_101_100_000.0}
    features["capital_structure.net_debt"] = {"value": 2_355_400_000.0}
    features["capital_structure.net_leverage"] = {"value": 8.23854494578524}
    features["capital_structure.interest_coverage"] = {"value": 7.919667590027701}
    features["capital_structure.debt_due_0_12m"] = {"value": None}
    features["capital_structure.debt_due_12_24m"] = {"value": None}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": None}
    features["operating.fcf_conversion"] = {"value": 0.3906960475690801}
    features["operating.ebitda_margin_ttm"] = {"value": 0.10177999288002848}
    features["operating.revenue_cagr_3y"] = {"value": None}
    return features


def _sparse_reset_recap_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["strategic.last_action_type"] = {"value": "debt_issuance"}
    features["market.market_cap"] = {"value": 610_097_252.76}
    features["market.credit_window_proxy"] = {"value": 0.5074821251650947}
    features["market.equity_window_proxy"] = {"value": 0.3025264989846301}
    features["market.drawdown_90d"] = {"value": -0.1556603773584906}
    features["liquidity.available_for_actions"] = {"value": 377_784_000.0}
    features["capital_structure.total_debt"] = {"value": 506_172_000.0}
    features["capital_structure.net_debt"] = {"value": 450_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 2.7799505305420134}
    features["capital_structure.interest_coverage"] = {"value": 4.698075870489664}
    return features


def _acquisition_reset_recap_feature_set() -> dict:
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    features["strategic.last_action_type"] = {"value": "acquisition"}
    features["market.market_cap"] = {"value": 205_894_294.83}
    features["market.credit_window_proxy"] = {"value": 0.633400794648336}
    features["market.equity_window_proxy"] = {"value": 0.24308728224966272}
    features["market.drawdown_90d"] = {"value": -0.2028985507246377}
    features["liquidity.available_for_actions"] = {"value": 137_436_000.0}
    features["capital_structure.total_debt"] = {"value": 50_540_000.0}
    features["capital_structure.net_debt"] = {"value": -19_396_000.0}
    features["capital_structure.net_leverage"] = {"value": -0.3900261411622763}
    features["capital_structure.interest_coverage"] = {"value": 4.7119575516391885}
    features["capital_structure.debt_due_0_12m"] = {"value": None}
    features["capital_structure.debt_due_12_24m"] = {"value": None}
    return features


def test_generated_candidates_pass_schema_validation(tmp_path: Path):
    snapshot_root, snapshot = _write_snapshot(tmp_path, _rich_feature_set())
    run = _make_run(tmp_path, snapshot_root)

    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)
    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)

    candidates = out["candidates"]
    assert len(candidates) > 0

    for c in candidates:
        v = registry.validate_candidate(
            {
                "action_id": c["action_id"],
                "parameters": c["parameters"],
                "available_features": list(snapshot["features"].keys()),
                "available_evidence_classes": [
                    "financial_disclosure",
                    "market_signal",
                    "management_statement",
                    "capital_policy_statement",
                    "liquidity_disclosure",
                    "segment_disclosure",
                    "peer_context_signal",
                    "recent_action_history",
                    "rating_disclosure",
                ],
                "known_segments": ["segment_A", "segment_B", "segment_C"],
                "constraints": [],
            }
        )
        assert v.valid is True


def test_parameter_ranges_respected(tmp_path: Path):
    snapshot_root, snapshot = _write_snapshot(tmp_path, _rich_feature_set())
    run = _make_run(tmp_path, snapshot_root)

    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)
    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=600)

    for c in out["candidates"]:
        schema = registry.get_action(c["action_id"])
        assert schema is not None
        for pname, pval in c["parameters"].items():
            pdef = schema["parameter_schema"].get(pname)
            if not pdef:
                continue
            if pdef["type"] in {"percent", "numeric"}:
                if "min" in pdef:
                    assert float(pval) >= float(pdef["min"])
                if "max" in pdef:
                    assert float(pval) <= float(pdef["max"])


def test_duplicate_candidates_removed(tmp_path: Path):
    features = {
        "market.market_cap": {"value": 2_000_000_000.0},
        "liquidity.available_for_actions": {"value": 100_000_000.0},
        "capital_structure.net_leverage": {"value": 2.0},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)

    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)
    llm = [
        {
            "action_id": "capital_return.open_market_buyback",
            "parameters": {
                "size_pct_market_cap": 0.1,
                "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0},
            },
            "evidence_refs": [],
        },
        {
            "action_id": "capital_return.open_market_buyback",
            "parameters": {
                "size_pct_market_cap": 0.1,
                "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0},
            },
            "evidence_refs": [],
        },
    ]
    out = engine.generate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        llm_proposals=llm,
        max_candidates=100,
    )

    buybacks = [x for x in out["candidates"] if x["action_id"] == "capital_return.open_market_buyback"]
    signatures = {x["candidate_signature"] for x in buybacks}
    assert len(signatures) == len(buybacks)
    assert out["llm_trace"]["accepted_count"] == 2
    assert len(buybacks) == 1


def test_deterministic_generation_reproducible(tmp_path: Path):
    snapshot_root, snapshot = _write_snapshot(tmp_path, _rich_feature_set())
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    a = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=400)
    b = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=400)

    sig_a = [x["candidate_signature"] for x in a["candidates"]]
    sig_b = [x["candidate_signature"] for x in b["candidates"]]
    assert sig_a == sig_b


def test_dividend_initiate_generated_only_for_non_payers(tmp_path: Path):
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_initiate" in action_ids

    features_true = _capital_return_feature_set()
    features_true["capital_return.dividend_payer_flag"] = {"value": True}
    snapshot_root_true, snapshot_true = _write_snapshot(tmp_path / "payer_true", features_true)
    run_true = _make_run(tmp_path / "payer_true", snapshot_root_true)
    out_true = engine.generate_candidate_set(run=run_true, state_snapshot=snapshot_true, max_candidates=500)
    action_ids_true = {row["action_id"] for row in out_true["candidates"]}
    assert "capital_return.dividend_initiate" not in action_ids_true


def test_dividend_initiate_generated_for_nonpayer_without_market_cap(tmp_path: Path):
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["market.market_cap"] = {"value": None}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine._gen_liquidity_excess(run=run, features=snapshot["features"])
    action_ids = {row["action_id"] for row in out}
    assert "capital_return.dividend_initiate" in action_ids
    assert "capital_return.open_market_buyback" not in action_ids


def test_dividend_initiate_generated_when_payer_history_is_sparse_but_no_positive_signal_exists(tmp_path: Path):
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {
        "value": None,
        "missing_reason": "unavailable",
        "quality_flags": ["event_history_unavailable"],
    }
    features["capital_return.last_dividend_event_type"] = {"value": None}
    features["market.market_cap"] = {"value": None}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine._gen_liquidity_excess(run=run, features=snapshot["features"])
    action_ids = {row["action_id"] for row in out}
    assert "capital_return.dividend_initiate" in action_ids
    assert "capital_return.open_market_buyback" not in action_ids


def test_capital_return_suppressed_when_financing_stress_is_active(tmp_path: Path):
    features = _rich_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    blocked = {
        "capital_return.open_market_buyback",
        "capital_return.accelerated_share_repurchase",
        "capital_return.special_dividend",
        "capital_return.dividend_initiate",
        "capital_return.dividend_increase",
    }
    assert blocked.isdisjoint(action_ids)
    assert "capital_structure.refinancing" in action_ids


def test_capital_return_generated_when_financing_stress_is_mild(tmp_path: Path):
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.open_market_buyback" in action_ids
    assert "capital_return.accelerated_share_repurchase" in action_ids
    assert "capital_return.dividend_initiate" in action_ids


def test_mna_suppressed_when_financing_stress_is_active(tmp_path: Path):
    features = _rich_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    blocked = {
        "mna.tuck_in_acquisition",
        "mna.platform_acquisition",
        "mna.go_private_lbo",
    }
    assert blocked.isdisjoint(action_ids)
    assert "capital_structure.refinancing" in action_ids


def test_mna_generated_when_capacity_is_clear(tmp_path: Path):
    features = _mna_capacity_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "mna.tuck_in_acquisition" in action_ids
    assert "mna.platform_acquisition" in action_ids
    assert "mna.go_private_lbo" in action_ids


def test_maturity_wall_generated_from_absolute_debt_due_when_ratio_missing(tmp_path: Path):
    features = _absolute_maturity_wall_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.refinancing" in action_ids
    assert "capital_structure.exchange_offer" in action_ids


def test_small_absolute_debt_due_does_not_trigger_maturity_wall(tmp_path: Path):
    features = _capital_return_feature_set()
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["capital_structure.debt_due_0_12m"] = {"value": 500_000.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 0.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": None}
    features["capital_structure.total_debt"] = {"value": 0.0}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.refinancing" not in action_ids


def test_equity_issuance_generated_when_equity_window_is_only_viable_backstop(tmp_path: Path):
    features = _equity_backstop_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" in action_ids


def test_equity_issuance_generated_without_net_leverage_when_coverage_is_broken(tmp_path: Path):
    features = _equity_backstop_feature_set()
    features.pop("capital_structure.net_leverage", None)
    features["capital_structure.total_debt"] = {"value": 3_900_000.0}
    features["capital_structure.interest_coverage"] = {"value": -12.0}
    features["liquidity.available_for_actions"] = {"value": 127_000_000.0}
    features["market.market_cap"] = {"value": 6_500_000_000.0}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" in action_ids


def test_equity_issuance_not_generated_when_liquidity_is_plentiful(tmp_path: Path):
    features = _equity_backstop_feature_set()
    features["liquidity.available_for_actions"] = {"value": 700_000_000.0}
    features["market.market_cap"] = {"value": 5_000_000_000.0}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" not in action_ids


def test_undervaluation_capital_return_is_suppressed_when_revisions_are_sharply_negative(tmp_path: Path):
    features = _capital_return_feature_set()
    features["liquidity.available_for_actions"] = {"value": 100_000_000.0}
    features["market.ev_ebitda_vs_peer_z"] = {"value": -1.6}
    features["market.fcf_yield_percentile_peers"] = {"value": 0.86}
    features["expectations.analyst_coverage_count"] = {"value": 12.0}
    features["expectations.revision_signal"] = {"value": -0.10}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.open_market_buyback" not in action_ids
    assert "capital_return.accelerated_share_repurchase" not in action_ids
    assert "capital_return.dividend_increase" not in action_ids


def test_undervaluation_capital_return_survives_when_revisions_are_supportive(tmp_path: Path):
    features = _capital_return_feature_set()
    features["liquidity.available_for_actions"] = {"value": 100_000_000.0}
    features["market.ev_ebitda_vs_peer_z"] = {"value": -1.6}
    features["market.fcf_yield_percentile_peers"] = {"value": 0.86}
    features["expectations.analyst_coverage_count"] = {"value": 12.0}
    features["expectations.revision_signal"] = {"value": 0.08}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.open_market_buyback" in action_ids


def test_dividend_increase_generated_for_healthy_existing_payer(tmp_path: Path):
    features = _dividend_confidence_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids


def test_special_dividend_generated_for_net_cash_regular_payer_with_excess_liquidity(tmp_path: Path):
    features = _special_dividend_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.special_dividend" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids






def test_dividend_increase_generated_for_missing_schedule_regular_payer(tmp_path: Path):
    features = _missing_schedule_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.refinancing" not in action_ids


def test_healthy_existing_payer_not_forced_into_equity_backstop(tmp_path: Path):
    features = _dividend_confidence_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" not in action_ids


def test_special_dividend_generated_for_durable_payer_with_leverage_optics(tmp_path: Path):
    features = _durable_dividend_growth_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = set(_candidate_action_ids(out))
    assert "capital_return.special_dividend" in action_ids


def test_dividend_increase_generated_when_cash_is_near_policy_floor(tmp_path: Path):
    features = _near_min_cash_dividend_growth_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids


def test_dividend_increase_not_generated_when_durable_payer_faces_maturity_wall(tmp_path: Path):
    features = _durable_dividend_growth_feature_set()
    features["liquidity.available_for_actions"] = {"value": 0.0}
    features["capital_structure.debt_due_0_12m"] = {"value": 0.0}
    features["capital_structure.debt_due_12_24m"] = {"value": 750_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.5970386881069893}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" not in action_ids
    assert "capital_structure.equity_issuance" in action_ids or "capital_structure.exchange_offer" in action_ids


def test_extreme_volatility_payer_still_surfaces_equity_backstop(tmp_path: Path):
    features = _extreme_volatility_dividend_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = set(_candidate_action_ids(out))
    assert "capital_structure.equity_issuance" in action_ids


def test_anomalous_regular_dividend_payer_prefers_dividend_increase_over_financing(tmp_path: Path):
    features = _anomalous_regular_dividend_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.exchange_offer" not in action_ids


def test_leveraged_regular_payer_can_still_generate_dividend_increase(tmp_path: Path):
    features = _leveraged_dividend_growth_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.liability_management_exercise" not in action_ids
    assert "capital_structure.exchange_offer" not in action_ids


def test_cash_rich_regular_payer_surfaces_dividend_increase(tmp_path: Path):
    features = _cash_rich_regular_dividend_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = set(_candidate_action_ids(out))
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids


def test_stable_debt_bearing_regular_payer_prefers_dividend_increase(tmp_path: Path):
    features = _stable_debt_bearing_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids


def test_large_cap_coverage_regular_payer_prefers_dividend_increase(tmp_path: Path):
    features = _large_cap_coverage_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.refinancing" not in action_ids
    assert "capital_structure.exchange_offer" not in action_ids
    assert "capital_structure.liability_management_exercise" not in action_ids


def test_no_maturity_pressure_regular_payer_prefers_dividend_increase(tmp_path: Path):
    features = _no_maturity_pressure_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids


def test_low_margin_no_maturity_regular_payer_does_not_force_dividend_increase(tmp_path: Path):
    features = _low_margin_no_maturity_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" not in action_ids


def test_schedule_anomaly_regular_payer_prefers_dividend_increase(tmp_path: Path):
    features = _schedule_anomaly_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids


def test_coverage_outlier_regular_payer_prefers_dividend_increase(tmp_path: Path):
    features = _coverage_outlier_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids


def test_financing_anomaly_regular_payer_prefers_dividend_increase_over_financing(tmp_path: Path):
    features = _financing_anomaly_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.refinancing" not in action_ids
    assert "capital_structure.exchange_offer" not in action_ids
    assert "capital_structure.liability_management_exercise" not in action_ids


def test_weak_fcf_schedule_anomaly_regular_payer_keeps_financing_actions(tmp_path: Path):
    features = _weak_fcf_financing_anomaly_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" not in action_ids
    assert "capital_structure.refinancing" in action_ids or "capital_structure.equity_issuance" in action_ids


def test_no_market_dislocation_schedule_anomaly_regular_payer_keeps_financing_actions(tmp_path: Path):
    features = _no_market_dislocation_financing_anomaly_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" not in action_ids
    assert "capital_structure.refinancing" in action_ids or "capital_structure.equity_issuance" in action_ids


def test_equity_issuance_generated_for_extreme_volatility_equity_financing_profile(tmp_path: Path):
    features = _extreme_volatility_dividend_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" in action_ids


def test_equity_issuance_generated_for_distressed_private_placement_profile(tmp_path: Path):
    features = _distressed_private_placement_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    placements = [
        row
        for row in out["candidates"]
        if row["action_id"] == "capital_structure.equity_issuance"
        and row["parameters"].get("offering_type") == "private_placement"
    ]
    assert placements


def test_equity_issuance_generated_for_relaxed_public_window_backstop(tmp_path: Path):
    features = _relaxed_window_equity_backstop_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" in action_ids


def test_equity_issuance_generated_for_distressed_nonpayer_public_recap(tmp_path: Path):
    features = _distressed_nonpayer_public_recap_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" in action_ids


def test_equity_issuance_generated_for_distressed_nonpayer_private_recap_without_cash_shortfall(tmp_path: Path):
    features = _distressed_nonpayer_private_recap_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    placements = [
        row
        for row in out["candidates"]
        if row["action_id"] == "capital_structure.equity_issuance"
        and row["parameters"].get("offering_type") == "private_placement"
    ]
    assert placements


def test_distressed_regular_payer_does_not_use_nonpayer_equity_recap_override(tmp_path: Path):
    features = _distressed_nonpayer_public_recap_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": True}
    features["capital_return.last_dividend_event_type"] = {"value": "dividend_regular"}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" not in action_ids


def test_nonpayer_public_recap_preference_suppresses_debt_only_maturity_actions(tmp_path: Path):
    features = _nonpayer_public_recap_preference_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" in action_ids
    assert "capital_structure.refinancing" not in action_ids
    assert "capital_structure.exchange_offer" not in action_ids
    assert "capital_structure.liability_management_exercise" not in action_ids


def test_nonpayer_public_recap_preference_does_not_fire_for_low_debt_pressure(tmp_path: Path):
    features = _nonpayer_public_recap_low_stress_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" not in action_ids


def test_market_shutdown_regular_payer_surfaces_equity_backstop(tmp_path: Path):
    features = _market_shutdown_regular_payer_recap_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = set(_candidate_action_ids(out))
    assert "capital_structure.equity_issuance" in action_ids


def test_market_shutdown_regular_payer_guard_does_not_fire_for_low_debt_profile(tmp_path: Path):
    features = _market_shutdown_low_debt_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" not in action_ids


def test_strategic_regular_payer_recap_surfaces_equity(tmp_path: Path):
    features = _strategic_regular_payer_recap_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=12)
    action_ids = _candidate_action_ids(out)
    assert "capital_structure.equity_issuance" in action_ids


def test_buyback_oriented_regular_payer_does_not_use_strategic_recap_override(tmp_path: Path):
    features = _strategic_regular_payer_buyback_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=12)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.refinancing" in action_ids


def test_buyback_regular_payer_recap_surfaces_equity(tmp_path: Path):
    features = _buyback_regular_payer_recap_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=12)
    action_ids = _candidate_action_ids(out)
    assert "capital_structure.equity_issuance" in action_ids


def test_strategic_nonpayer_recap_prefers_equity_over_debt_tools(tmp_path: Path):
    features = _strategic_nonpayer_recap_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=12)
    action_ids = [row["action_id"] for row in out["candidates"]]
    assert "capital_structure.equity_issuance" in action_ids
    assert "capital_structure.refinancing" not in action_ids
    assert "capital_structure.exchange_offer" not in action_ids
    assert "capital_structure.liability_management_exercise" not in action_ids


def test_low_quality_strategic_nonpayer_does_not_trigger_equity_override(tmp_path: Path):
    features = _low_quality_strategic_nonpayer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=12)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.refinancing" in action_ids


def test_growth_substitution_requires_observed_weak_growth(tmp_path: Path):
    features = _growth_missing_liquidity_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = set(_candidate_action_ids(out))
    assert "mna.tuck_in_acquisition" not in action_ids
    assert "mna.platform_acquisition" not in action_ids


def test_dividend_cut_generated_for_stressed_existing_payer(tmp_path: Path):
    features = _dividend_cut_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_cut" in action_ids


def test_dividend_cut_generated_for_balance_sheet_pressure_regular_payer(tmp_path: Path):
    features = _balance_sheet_pressure_dividend_cut_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_cut" in action_ids
    assert "capital_return.dividend_increase" not in action_ids
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.refinancing" not in action_ids


def test_dividend_cut_profile_prioritizes_dividend_cut_variants(tmp_path: Path):
    features = _dividend_cut_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = _candidate_action_ids(out)
    assert action_ids[:12] == ["capital_return.dividend_cut"] * 12


def test_dividend_cut_not_generated_when_coverage_is_still_strong(tmp_path: Path):
    features = _dividend_cut_feature_set()
    features["capital_structure.interest_coverage"] = {"value": 10.5}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_cut" not in action_ids


def test_dividend_cut_generated_for_regular_payer_with_extreme_maturity_wall(tmp_path: Path):
    features = _maturity_wall_dividend_cut_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = _candidate_action_ids(out)
    assert "capital_return.dividend_cut" in action_ids
    assert action_ids[:12] == ["capital_return.dividend_cut"] * 12


def test_dividend_cut_generated_for_buyback_reset_regular_payer(tmp_path: Path):
    features = _buyback_reset_dividend_cut_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_cut" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids


def test_dividend_cut_not_generated_for_buyback_reset_when_cashflow_recovers(tmp_path: Path):
    features = _buyback_reset_dividend_cut_feature_set()
    features["operating.fcf_conversion"] = {"value": 0.10}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_cut" not in action_ids


def test_real_financing_stress_regular_payer_still_surfaces_financing_actions(tmp_path: Path):
    features = _real_financing_stress_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" not in action_ids
    assert (
        "capital_structure.equity_issuance" in action_ids
        or "capital_structure.refinancing" in action_ids
        or "capital_structure.exchange_offer" in action_ids
    )


def test_mild_maturity_wall_regular_payer_can_still_surface_dividend_increase(tmp_path: Path):
    features = _mild_maturity_wall_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.refinancing" not in action_ids
    assert "capital_structure.exchange_offer" not in action_ids
    assert "capital_structure.liability_management_exercise" not in action_ids


def test_undercovered_mild_maturity_wall_regular_payer_keeps_financing_actions(tmp_path: Path):
    features = _mild_maturity_wall_regular_payer_feature_set()
    features["liquidity.available_for_actions"] = {"value": 300_000_000.0}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" not in action_ids
    assert (
        "capital_structure.equity_issuance" in action_ids
        or "capital_structure.refinancing" in action_ids
        or "capital_structure.exchange_offer" in action_ids
    )


def test_continuity_regular_payer_generates_dividend_increase_without_financing_actions(tmp_path: Path):
    features = _continuity_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.refinancing" not in action_ids


def test_net_cash_continuity_regular_payer_generates_dividend_increase(tmp_path: Path):
    features = _net_cash_continuity_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids


def test_healthy_regular_payer_low_growth_does_not_trigger_growth_substitution_playbook(tmp_path: Path):
    features = _healthy_low_growth_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "mna.tuck_in_acquisition" not in action_ids
    assert "mna.platform_acquisition" not in action_ids


def test_high_coverage_regular_payer_surfaces_dividend_increase_without_runway(tmp_path: Path):
    features = _high_coverage_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids


def test_high_coverage_regular_payer_negative_revisions_block_dividend_increase(tmp_path: Path):
    features = _high_coverage_regular_payer_feature_set()
    features["expectations.analyst_coverage_count"] = {"value": 14.0}
    features["expectations.revision_signal"] = {"value": -0.08}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" not in action_ids


def test_mega_cap_high_coverage_regular_payer_can_surface_dividend_increase_despite_noisy_fcf(tmp_path: Path):
    features = _mega_cap_high_coverage_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids


def test_coverage_supported_regular_payer_prefers_dividend_increase_over_financing(tmp_path: Path):
    features = _coverage_supported_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.refinancing" not in action_ids
    assert "capital_structure.equity_issuance" not in action_ids


def test_undercovered_coverage_supported_regular_payer_keeps_financing_actions(tmp_path: Path):
    features = _coverage_supported_regular_payer_feature_set()
    features["capital_structure.interest_coverage"] = {"value": 6.0}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" not in action_ids
    assert (
        "capital_structure.equity_issuance" in action_ids
        or "capital_structure.refinancing" in action_ids
        or "capital_structure.exchange_offer" in action_ids
    )


def test_sparse_data_regular_payer_prefers_dividend_increase_over_equity_backstop(tmp_path: Path):
    features = _sparse_data_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = _candidate_action_ids(out)
    assert action_ids[:9] == ["capital_return.dividend_increase"] * 9
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.refinancing" not in action_ids


def test_coverage_gap_regular_payer_prefers_dividend_increase_over_financing(tmp_path: Path):
    features = _coverage_gap_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.refinancing" not in action_ids


def test_liquidity_supported_regular_payer_prefers_dividend_increase_over_financing(tmp_path: Path):
    features = _liquidity_supported_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.refinancing" not in action_ids


def test_balance_sheet_light_regular_payer_prefers_dividend_increase_over_financing(tmp_path: Path):
    features = _balance_sheet_light_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" in action_ids
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.refinancing" not in action_ids


def test_balance_sheet_light_regular_payer_requires_cash_or_cashflow_support(tmp_path: Path):
    features = _balance_sheet_light_regular_payer_negative_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    assert not engine._balance_sheet_light_regular_payer_dividend_increase_profile(snapshot["features"])

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" not in action_ids


def test_sparse_data_regular_payer_with_deep_drawdown_keeps_financing_actions(tmp_path: Path):
    features = _sparse_data_regular_payer_feature_set()
    features["market.drawdown_90d"] = {"value": -0.35}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" not in action_ids
    assert "capital_structure.equity_issuance" in action_ids


def test_sparse_reset_recap_prefers_equity_issuance_over_capital_return(tmp_path: Path):
    features = _sparse_reset_recap_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = _candidate_action_ids(out)
    assert action_ids[:12] == ["capital_structure.equity_issuance"] * 12
    assert "capital_return.open_market_buyback" not in action_ids
    assert "capital_return.accelerated_share_repurchase" not in action_ids


def test_sparse_reset_recap_requires_real_balance_sheet_burden(tmp_path: Path):
    features = _sparse_reset_recap_feature_set()
    features["capital_structure.total_debt"] = {"value": 50_000_000.0}
    features["capital_structure.net_debt"] = {"value": 20_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 0.4}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_return.open_market_buyback" in action_ids or "capital_return.accelerated_share_repurchase" in action_ids


def test_sparse_reset_recap_handles_missing_credit_window(tmp_path: Path):
    features = _sparse_reset_recap_feature_set()
    features["market.credit_window_proxy"] = {"value": None}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = _candidate_action_ids(out)
    assert action_ids[:12] == ["capital_structure.equity_issuance"] * 12
    assert "capital_return.open_market_buyback" not in action_ids
    assert "capital_return.accelerated_share_repurchase" not in action_ids


def test_net_cash_nonpayer_is_not_blocked_by_maturity_wall_alone(tmp_path: Path):
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["market.market_cap"] = {"value": 385_000_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 7_500_000_000.0}
    features["capital_structure.total_debt"] = {"value": 1_700_000_000.0}
    features["capital_structure.net_debt"] = {"value": -5_800_000_000.0}
    features["capital_structure.net_leverage"] = {"value": -1.1372549019607843}
    features["capital_structure.gross_leverage"] = {"value": 0.33}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.41}
    features["capital_structure.interest_coverage"] = {"value": None}
    features["market.ev_ebitda_vs_peer_z"] = {"value": None}
    features["market.fcf_yield_percentile_peers"] = {"value": None}
    features["market.fcf_yield"] = {"value": 0.01}

    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    assert not engine._capital_return_blocked_by_financing_stress(snapshot["features"])

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.open_market_buyback" in action_ids


def test_buyback_supported_net_cash_nonpayer_can_clear_low_liquidity_ratio_gate(tmp_path: Path):
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["market.market_cap"] = {"value": 233_768_640_000.0}
    features["liquidity.available_for_actions"] = {"value": 5_340_000_000.0}
    features["liquidity.cash"] = {"value": 4_113_000_000.0}
    features["capital_structure.total_debt"] = {"value": 2_468_000_000.0}
    features["capital_structure.net_debt"] = {"value": -2_872_000_000.0}
    features["capital_structure.net_leverage"] = {"value": -2.5460992907801416}
    features["capital_structure.gross_leverage"] = {"value": 2.1879432624113475}
    features["capital_structure.interest_coverage"] = {"value": 45.12}
    features["capital_structure.debt_due_next_24m"] = {"value": 700_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.28363047001620745}
    features["market.fcf_yield"] = {"value": 0.0031954671079919023}
    features["capital_return.buyback_capacity_proxy"] = {"value": 0.0028430982017091794}
    features["market.ev_ebitda_vs_peer_z"] = {"value": None}
    features["market.fcf_yield_percentile_peers"] = {"value": None}

    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    assert not engine._net_cash_maturity_override_profile(snapshot["features"])
    assert engine._buyback_supported_net_cash_override_profile(snapshot["features"])
    assert not engine._capital_return_blocked_by_financing_stress(snapshot["features"])

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.open_market_buyback" in action_ids
    assert "capital_return.accelerated_share_repurchase" in action_ids


def test_buyback_supported_net_cash_nonpayer_keeps_small_buyback_under_candidate_cap(tmp_path: Path):
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["market.market_cap"] = {"value": 233_768_640_000.0}
    features["liquidity.available_for_actions"] = {"value": 5_340_000_000.0}
    features["liquidity.cash"] = {"value": 4_113_000_000.0}
    features["capital_structure.total_debt"] = {"value": 2_468_000_000.0}
    features["capital_structure.net_debt"] = {"value": -2_872_000_000.0}
    features["capital_structure.net_leverage"] = {"value": -2.5460992907801416}
    features["capital_structure.gross_leverage"] = {"value": 2.1879432624113475}
    features["capital_structure.interest_coverage"] = {"value": 45.12}
    features["capital_structure.debt_due_next_24m"] = {"value": 700_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.28363047001620745}
    features["market.fcf_yield"] = {"value": 0.0031954671079919023}
    features["capital_return.buyback_capacity_proxy"] = {"value": 0.0028430982017091794}
    features["market.ev_ebitda_vs_peer_z"] = {"value": None}
    features["market.fcf_yield_percentile_peers"] = {"value": None}

    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=12)
    buybacks = [
        row["parameters"].get("size_pct_market_cap")
        for row in out["candidates"]
        if row["action_id"] == "capital_return.open_market_buyback"
    ]

    assert 0.02 in buybacks


def test_missing_market_cap_nonpayer_with_strong_coverage_can_initiate_dividend(tmp_path: Path):
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["capital_return.last_dividend_event_type"] = {"value": None}
    features["market.market_cap"] = {"value": None}
    features["liquidity.available_for_actions"] = {"value": 1_445_000_000.0}
    features["liquidity.cash"] = {"value": 1_445_000_000.0}
    features["capital_structure.total_debt"] = {"value": 8_353_000_000.0}
    features["capital_structure.net_debt"] = {"value": 6_908_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 7.3646}
    features["capital_structure.interest_coverage"] = {"value": 15.63}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.1195}
    features["capital_structure.debt_due_next_24m"] = {"value": 998_601_000.0}
    features["cash_flow.free_cash_flow_ttm"] = {"value": 2_423_000_000.0}

    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    assert not engine._capital_return_blocked_by_financing_stress(snapshot["features"])

    out = engine._gen_liquidity_excess(run=run, features=snapshot["features"])
    action_ids = {row["action_id"] for row in out}
    assert "capital_return.dividend_initiate" in action_ids
    assert "capital_return.open_market_buyback" not in action_ids


def test_missing_market_cap_buyback_transition_nonpayer_can_initiate_dividend(tmp_path: Path):
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["capital_return.last_dividend_event_type"] = {"value": None}
    features["market.market_cap"] = {"value": None}
    features["market.equity_window_proxy"] = {"value": None}
    features["market.credit_window_proxy"] = {"value": None}
    features["liquidity.available_for_actions"] = {"value": 369_400_000.0}
    features["liquidity.cash"] = {"value": 369_400_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 22_506_000.0}
    features["capital_structure.total_debt"] = {"value": 2_101_200_000.0}
    features["capital_structure.net_debt"] = {"value": 1_731_800_000.0}
    features["capital_structure.net_leverage"] = {"value": 16.635926993275696}
    features["capital_structure.gross_leverage"] = {"value": 20.18443804034582}
    features["capital_structure.interest_coverage"] = {"value": 2.899286878476516}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": None}
    features["capital_structure.debt_due_next_24m"] = {"value": None}
    features["cash_flow.free_cash_flow_ttm"] = {"value": None}
    features["operating.fcf_conversion"] = {"value": 2.2151777137367916}
    features["strategic.last_action_type"] = {"value": "buyback"}
    features["ownership_governance.activist_presence_flag"] = {"value": True}
    features["capital_return.buyback_capacity_proxy"] = {"value": None}
    features["capital_return.share_count_trend"] = {"value": None}
    features["market.ev_ebitda_vs_peer_z"] = {"value": None}
    features["market.fcf_yield_percentile_peers"] = {"value": None}

    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    assert engine._missing_market_cap_dividend_initiate_override_profile(snapshot["features"])
    assert not engine._capital_return_blocked_by_financing_stress(snapshot["features"])

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=12)
    action_ids = [row["action_id"] for row in out["candidates"]]
    assert "capital_return.dividend_initiate" in action_ids
    assert "capital_return.open_market_buyback" not in action_ids
    assert "capital_return.accelerated_share_repurchase" not in action_ids
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.refinancing" not in action_ids


def test_coverage_supported_nonpayer_can_initiate_dividend_despite_elevated_leverage(tmp_path: Path):
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["capital_return.last_dividend_event_type"] = {"value": None}
    features["market.market_cap"] = {"value": 37_235_561_182.17}
    features["market.equity_window_proxy"] = {"value": 1.0}
    features["market.credit_window_proxy"] = {"value": None}
    features["market.ev_ebitda_vs_peer_z"] = {"value": -0.2}
    features["market.fcf_yield_percentile_peers"] = {"value": 0.5}
    features["liquidity.available_for_actions"] = {"value": 2_863_000_000.0}
    features["liquidity.cash"] = {"value": 2_324_000_000.0}
    features["capital_structure.total_debt"] = {"value": 10_234_000_000.0}
    features["capital_structure.net_debt"] = {"value": 7_371_000_000.0}
    features["capital_structure.net_leverage"] = {"value": 7.858208955223881}
    features["capital_structure.gross_leverage"] = {"value": 10.91044776119403}
    features["capital_structure.interest_coverage"] = {"value": 15.633333333333333}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.09757680281414892}
    features["capital_structure.debt_due_next_24m"] = {"value": 998_601_000.0}
    features["cash_flow.free_cash_flow_ttm"] = {"value": 2_423_000_000.0}
    features["operating.fcf_conversion"] = {"value": 2.5831556503198296}
    features["market.fcf_yield"] = {"value": 0.06507220310567624}
    features["strategic.last_action_type"] = {"value": "debt_issuance"}

    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    assert engine._coverage_supported_dividend_initiate_override_profile(snapshot["features"])
    assert not engine._capital_return_blocked_by_financing_stress(snapshot["features"])

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=12)
    action_ids = [row["action_id"] for row in out["candidates"]]
    assert "capital_return.dividend_initiate" in action_ids
    assert "capital_return.open_market_buyback" not in action_ids
    assert "capital_return.accelerated_share_repurchase" not in action_ids
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.refinancing" not in action_ids


def test_debt_bearing_buyback_transition_nonpayer_can_initiate_dividend(tmp_path: Path):
    features = _capital_return_feature_set()
    features["capital_return.dividend_payer_flag"] = {"value": False}
    features["capital_return.last_dividend_event_type"] = {"value": None}
    features["market.market_cap"] = {"value": 8_400_535_720.0}
    features["market.equity_window_proxy"] = {"value": 0.7039568306316377}
    features["market.credit_window_proxy"] = {"value": None}
    features["liquidity.available_for_actions"] = {"value": 681_400_000.0}
    features["liquidity.cash"] = {"value": 527_300_000.0}
    features["liquidity.minimum_cash_policy_proxy"] = {"value": 642_603_000.0}
    features["capital_structure.total_debt"] = {"value": 5_416_500_000.0}
    features["capital_structure.net_debt"] = {"value": 4_735_100_000.0}
    features["capital_structure.net_leverage"] = {"value": 2.9408732376871}
    features["capital_structure.gross_leverage"] = {"value": 3.364076765418297}
    features["capital_structure.interest_coverage"] = {"value": 5.469089673913044}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.0}
    features["capital_structure.debt_due_next_24m"] = {"value": 0.0}
    features["cash_flow.free_cash_flow_ttm"] = {"value": None}
    features["operating.fcf_conversion"] = {"value": 0.3236264828271536}
    features["strategic.last_action_type"] = {"value": "buyback"}
    features["ownership_governance.activist_presence_flag"] = {"value": True}
    features["capital_return.buyback_capacity_proxy"] = {"value": 0.0}
    features["capital_return.share_count_trend"] = {"value": None}
    features["market.ev_ebitda_vs_peer_z"] = {"value": None}
    features["market.fcf_yield_percentile_peers"] = {"value": None}

    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    assert engine._debt_bearing_dividend_initiate_override_profile(snapshot["features"])
    assert not engine._capital_return_blocked_by_financing_stress(snapshot["features"])

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=12)
    action_ids = [row["action_id"] for row in out["candidates"]]
    assert "capital_return.dividend_initiate" in action_ids
    assert "capital_return.open_market_buyback" not in action_ids
    assert "capital_return.accelerated_share_repurchase" not in action_ids
    assert "capital_structure.equity_issuance" not in action_ids
    assert "capital_structure.refinancing" not in action_ids


def test_acquisition_reset_recap_allows_moderate_debt_burden(tmp_path: Path):
    features = _acquisition_reset_recap_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = _candidate_action_ids(out)
    assert action_ids[:12] == ["capital_structure.equity_issuance"] * 12
    assert "capital_return.open_market_buyback" not in action_ids
    assert "capital_return.accelerated_share_repurchase" not in action_ids


def test_coverage_supported_regular_payer_is_not_blocked_by_financing_stress(tmp_path: Path):
    features = _coverage_supported_regular_payer_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    assert engine._coverage_supported_dividend_increase_profile(snapshot["features"])
    assert not engine._capital_return_blocked_by_financing_stress(snapshot["features"])


def test_small_cap_negative_fcf_regular_payer_does_not_force_dividend_increase(tmp_path: Path):
    features = _mega_cap_high_coverage_regular_payer_feature_set()
    features["market.market_cap"] = {"value": 5_000_000_000.0}
    features["liquidity.available_for_actions"] = {"value": 200_000_000.0}
    features["market.ev_ebitda_vs_peer_z"] = {"value": 0.0}
    features["market.fcf_yield_percentile_peers"] = {"value": 0.4}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_increase" not in action_ids


def test_dividend_cut_not_generated_when_near_term_wall_is_covered(tmp_path: Path):
    features = _maturity_wall_dividend_cut_feature_set()
    features["liquidity.available_for_actions"] = {"value": 175_000_000.0}
    features["capital_structure.debt_due_0_12m"] = {"value": 75_000_000.0}
    features["capital_structure.maturity_wall_ratio_24m"] = {"value": 0.21764916399235046}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "capital_return.dividend_cut" not in action_ids


def test_preemptive_deleveraging_dividend_cut_suppresses_financing_actions(tmp_path: Path):
    features = _preemptive_deleveraging_dividend_cut_feature_set()
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = _candidate_action_ids(out)
    assert action_ids[:12] == ["capital_return.dividend_cut"] * 12


def test_working_capital_program_requires_real_liquidity_stress(tmp_path: Path):
    features = _rich_feature_set()
    features["capital_structure.net_leverage"] = {"value": 6.0}
    features["capital_structure.rating_state"] = {"value": {"rating": "BB", "outlook": "stable"}}
    features["liquidity.runway_months"] = {"value": 60.0}
    features["operating.fcf_conversion"] = {"value": 2.0}
    features["liquidity.available_for_actions"] = {"value": 6_000_000_000.0}
    features["capital_structure.debt_due_next_24m"] = {"value": 1_000_000_000.0}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "restructuring.working_capital_program" not in action_ids


def test_working_capital_program_generated_when_liquidity_is_constrained(tmp_path: Path):
    features = _rich_feature_set()
    features["capital_structure.net_leverage"] = {"value": 6.0}
    features["capital_structure.rating_state"] = {"value": {"rating": "BB", "outlook": "stable"}}
    features["liquidity.runway_months"] = {"value": 9.0}
    features["operating.fcf_conversion"] = {"value": 0.45}
    features["liquidity.available_for_actions"] = {"value": 100_000_000.0}
    features["capital_structure.debt_due_next_24m"] = {"value": 500_000_000.0}
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    action_ids = {row["action_id"] for row in out["candidates"]}
    assert "restructuring.working_capital_program" in action_ids


def test_refinancing_tenor_variants_use_year_scale(tmp_path: Path):
    snapshot_root, snapshot = _write_snapshot(tmp_path, _rich_feature_set())
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(run=run, state_snapshot=snapshot, max_candidates=500)
    tenors = sorted(
        {
            row["parameters"].get("new_tenor_years")
            for row in out["candidates"]
            if row["action_id"] == "capital_structure.refinancing"
        }
    )
    assert tenors
    assert tenors == [3.0, 5.0, 7.0]


def test_llm_proposals_validated(tmp_path: Path):
    features = {
        "market.market_cap": {"value": 2_000_000_000.0},
        "liquidity.available_for_actions": {"value": 120_000_000.0},
        "capital_structure.net_leverage": {"value": 2.5},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    llm = [
        {
            "action_id": "capital_return.open_market_buyback",
            "parameters": {
                "size_pct_market_cap": 0.05,
                "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0},
            },
            "evidence_refs": [
                {
                    "reference_type": "state_feature",
                    "reference_id": "market.market_cap",
                    "explanation": "Valid feature reference.",
                }
            ],
        },
        {
            "action_id": "unknown.action",
            "parameters": {},
            "evidence_refs": [],
        },
        {
            "action_id": "capital_return.open_market_buyback",
            "parameters": {"size_pct_market_cap": 0.1},
            "evidence_refs": [{"reference_type": "not_allowed", "reference_id": "x", "explanation": "bad"}],
        },
    ]

    out = engine.generate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        llm_proposals=llm,
        llm_metadata={"prompt": "p", "response": "r", "temperature": 0.2, "seed": 7},
        max_candidates=100,
    )
    assert "llm_trace" in out
    assert out["llm_trace"]["proposal_count"] == 3
    assert out["llm_trace"]["accepted_count"] == 1
    assert len(out["llm_trace"]["discarded"]) == 2


def test_min_candidates_target_expands_coverage(tmp_path: Path):
    features = {
        "market.market_cap": {"value": 1_000_000_000.0},
        "liquidity.available_for_actions": {"value": 10_000_000.0},
        "capital_structure.net_leverage": {"value": 2.0},
        "strategic.segment_count": {"value": 1},
    }
    snapshot_root, snapshot = _write_snapshot(tmp_path, features)
    run = _make_run(tmp_path, snapshot_root)
    registry = build_default_action_schema_registry("v1.0")
    engine = CandidateGenerationEngine(registry)

    out = engine.generate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        max_candidates=80,
        min_candidates_target=40,
    )
    assert out["counts"]["deduped"] >= 40
    assert len(out["candidates"]) >= 40
