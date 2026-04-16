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


