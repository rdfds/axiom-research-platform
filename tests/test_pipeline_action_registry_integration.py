from __future__ import annotations

import json

from src.action_ontology import build_default_action_schema_registry
from src.pipeline.actions import build_change_vector
from src.company_state_store import SnapshotStore
from src.pipeline.run import (
    _baseline_from_world_model_features,
    _id_aliases,
    _is_materialized_local,
    _load_company_state_keyed_snapshot_row,
    _load_company_state_snapshot_row,
    _materialize_action_params,
    _resolve_company_id_aliases_from_cik_gvkey,
    _resolve_company_id_aliases_from_entity_identifier,
    _resolve_action_schema,
)
from src.pipeline.types import ActionCandidate


def test_resolve_action_schema_from_legacy_alias():
    registry = build_default_action_schema_registry()
    schema = _resolve_action_schema(
        registry=registry,
        action_type="buyback",
        action_subtype=None,
        action_id=None,
    )
    assert schema["action_id"] == "capital_return.open_market_buyback"


def test_resolve_action_schema_from_action_id():
    registry = build_default_action_schema_registry()
    schema = _resolve_action_schema(
        registry=registry,
        action_type=None,
        action_subtype=None,
        action_id="capital_structure.refinancing",
    )
    assert schema["action_type"] == "capital_structure"
    assert schema["action_subtype"] == "refinancing"


def test_resolve_action_schema_from_stock_split_alias():
    registry = build_default_action_schema_registry()
    schema = _resolve_action_schema(
        registry=registry,
        action_type="stock_split",
        action_subtype=None,
        action_id=None,
    )
    assert schema["action_id"] == "governance.stock_split"


def test_build_change_vector_uses_stock_split_alias():
    action = ActionCandidate(
        action_type="governance",
        action_subtype="stock_split",
        action_id="governance.stock_split",
        params={},
    )

    change = build_change_vector(
        action,
        {"action_effects": {"stock_split": {"trading_liquidity": 0.08}}},
    )

    assert change == {"trading_liquidity": 0.08}


def test_materialize_action_params_sets_required_defaults():
    registry = build_default_action_schema_registry()
    schema = registry.get_action("capital_return.open_market_buyback")
    assert schema is not None

    params, assumptions = _materialize_action_params(schema, action_params={})
    assert "size_pct_market_cap" in params
    assert "funding_mix" in params
    assert "default_param:size_pct_market_cap" in assumptions
    assert "default_param:funding_mix" in assumptions
    assert params["funding_mix"]["cash"] == 1.0


def test_baseline_mapping_from_world_model_features():
    baseline = _baseline_from_world_model_features(
        {
            "market.market_cap": {"value": 100.0},
            "operating.ebitda_margin_ttm": {"value": 0.2},
            "capital_structure.net_leverage": {"value": 2.5},
            "operating.fcf_conversion": {"value": 0.3},
        }
    )
    assert baseline["market_cap"] == 100.0
    assert baseline["ebitda_margin"] == 0.2
    assert baseline["leverage_net_debt_ebitda"] == 2.5
    assert baseline["fcf_margin"] == 0.3


def test_baseline_mapping_uses_runtime_aliases(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv(
        "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
        "normalized_net_leverage,pe_ratio_compatibility_alias",
    )
    baseline = _baseline_from_world_model_features(
        {
            "market.market_cap": {"value": 100.0},
            "capital_structure.net_leverage_normalized": {"value": 2.1, "support_mode": "exact"},
            "market.pe_ratio": {"value": 17.0, "support_mode": "exact"},
        }
    )

    assert baseline["market_cap"] == 100.0
    assert baseline["leverage_net_debt_ebitda"] == 2.1
    assert baseline["pe"] == 17.0


