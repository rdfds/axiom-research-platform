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


