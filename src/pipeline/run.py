from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..action_ontology import ActionSchemaRegistry, build_default_action_schema_registry
from ..model_feature_bundle import _STATE_VECTOR_V1_FEATURES, attach_model_feature_bundle, feature_view_from_snapshot
from ..runtime_feature_adapter import adapt_snapshot, resolve_feature_value
from .types import ActionCandidate, CompanyStateSnapshot, PrecedentPack


DATA_DIR = Path(__file__).parent.parent.parent / "data"
_DEFAULT_REGISTRY: Optional[ActionSchemaRegistry] = None
_PRECEDENT_DEBUG = os.getenv("RECO_PRECEDENT_DEBUG", "").strip().lower() not in {"", "0", "false", "no"}
_DEFAULT_PRECEDENT_OUTCOMES_CANDIDATES: Tuple[Path, ...] = (
    DATA_DIR / "curated" / "action_outcomes_with_credit_ratings.normalized_full.rich_contract_v3.parquet",
    DATA_DIR / "curated" / "action_outcomes_with_credit_ratings.normalized_full.rich_contract_v2.parquet",
    DATA_DIR / "curated" / "action_outcomes_with_credit_ratings.normalized_full.rich_contract_v1.parquet",
    DATA_DIR / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet",
    DATA_DIR / "curated" / "action_outcomes_with_credit_ratings.parquet",
    DATA_DIR / "curated" / "action_outcomes.parquet",
)


def _precedent_debug(stage: str, **details: Any) -> None:
    if not _PRECEDENT_DEBUG:
        return
    payload = {"ok": True, "event": "precedent_wrapper_debug", "stage": stage}
    payload.update(details)
    print(json.dumps(payload, default=str), flush=True)


def _default_precedent_outcomes_path() -> Path:
    for candidate in _DEFAULT_PRECEDENT_OUTCOMES_CANDIDATES:
        if candidate.exists():
            return candidate
    return _DEFAULT_PRECEDENT_OUTCOMES_CANDIDATES[0]


@lru_cache(maxsize=8)
def _load_outcomes_table_cached(path_str: str) -> pd.DataFrame:
    started = time.perf_counter()
    _precedent_debug("load_outcomes_table_cached:start", path=path_str)
    table = pd.read_parquet(path_str)
    _precedent_debug(
        "load_outcomes_table_cached:done",
        path=path_str,
        rows=int(len(table)),
        elapsed_seconds=round(time.perf_counter() - started, 6),
    )
    return table


def _load_outcomes_table(path: Path) -> pd.DataFrame:
    # Keep one in-memory copy per outcomes path for faster repeated precedent calls.
    return _load_outcomes_table_cached(str(path.resolve()))


@lru_cache(maxsize=8)
def _load_precedent_runtime_cached(path_str: str) -> Tuple[Dict[str, object], object]:
    """Build once per outcomes path: historical stores + retrieval index."""
    from .historical_stores import build_historical_stores_from_outcomes
    from .precedent_brain import augment_precedent_state_vector_columns, build_precedent_retrieval_index

    started = time.perf_counter()
    _precedent_debug("load_precedent_runtime_cached:start", path=path_str)
    full_df = _load_outcomes_table_cached(path_str)
    full_df = augment_precedent_state_vector_columns(full_df)
    stores_started = time.perf_counter()
    _precedent_debug("historical_stores:start", path=path_str, rows=int(len(full_df)))
    stores = build_historical_stores_from_outcomes(full_df, dataset_version=path_str)
    _precedent_debug(
        "historical_stores:done",
        path=path_str,
        elapsed_seconds=round(time.perf_counter() - stores_started, 6),
    )
    index_started = time.perf_counter()
    _precedent_debug("retrieval_index:start", path=path_str, rows=int(len(full_df)))
    retrieval_index = build_precedent_retrieval_index(full_df)
    _precedent_debug(
        "retrieval_index:done",
        path=path_str,
        index_rows=int(getattr(retrieval_index, "n_rows", 0) or 0),
        elapsed_seconds=round(time.perf_counter() - index_started, 6),
    )
    _precedent_debug(
        "load_precedent_runtime_cached:done",
        path=path_str,
        elapsed_seconds=round(time.perf_counter() - started, 6),
    )
    return stores, retrieval_index


def _load_precedent_runtime(path: Path) -> Tuple[Dict[str, object], object]:
    return _load_precedent_runtime_cached(str(path.resolve()))


@lru_cache(maxsize=1)
def _load_precedent_bindings() -> Tuple[Any, Any, Any, Any, Any]:
    from .actions import build_change_vector
    from .config import load_config
    from .features import FeatureBuilder
    from .precedent import build_precedent_pack
    from .precedent_brain import build_precedent_pack_v2

    return build_change_vector, load_config, FeatureBuilder, build_precedent_pack, build_precedent_pack_v2


def warm_precedent_runtime(path: str | Path) -> Dict[str, Any]:
    """Warm precedent caches (outcomes table, stores, retrieval index) for this outcomes path."""
    p = Path(path)
    _load_precedent_bindings()
    stores, retrieval_index = _load_precedent_runtime(p)
    event_store = stores.get("historical_event_store")
    n_events = 0
    if event_store is not None and hasattr(event_store, "events"):
        try:
            n_events = int(len(getattr(event_store, "events")))
        except Exception:
            n_events = 0
    n_index = int(getattr(retrieval_index, "n_rows", 0) or 0)
    return {
        "ok": True,
        "outcomes_path": str(p.resolve()),
        "historical_events": n_events,
        "index_rows": n_index,
    }


def _default_registry() -> ActionSchemaRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_default_action_schema_registry(version="v1.0")
    return _DEFAULT_REGISTRY


def _parse_date(value: str) -> datetime:
    return pd.to_datetime(value).to_pydatetime()


_LEGACY_ACTION_ALIASES: Dict[str, str] = {
    "buyback": "capital_return.open_market_buyback",
    "asr": "capital_return.accelerated_share_repurchase",
    "dividend": "capital_return.dividend_increase",
    "dividend_initiate": "capital_return.dividend_initiate",
    "debt_issuance": "capital_structure.new_debt_issuance",
    "refinancing": "capital_structure.refinancing",
    "equity_issuance": "capital_structure.equity_issuance",
    "acquisition": "mna.tuck_in_acquisition",
    "lbo": "mna.go_private_lbo",
    "go_private_lbo": "mna.go_private_lbo",
    "divestiture": "portfolio.divestiture_full",
    "spin_off": "portfolio.spin_off",
    "asset_sale": "portfolio.asset_sale",
    "stock_split": "governance.stock_split",
    "cost_program": "restructuring.cost_program",
}


def _resolve_action_schema(
    registry: ActionSchemaRegistry,
    action_type: Optional[str],
    action_subtype: Optional[str],
    action_id: Optional[str],
) -> Dict[str, Any]:
    if action_id:
        schema = registry.get_action(action_id)
        if schema is None:
            raise ValueError(f"Unknown action_id: {action_id}")
        return schema

    if action_type and "." in action_type:
        schema = registry.get_action(action_type)
        if schema is not None:
            return schema

    if action_type and action_subtype:
        aid = f"{action_type}.{action_subtype}"
        schema = registry.get_action(aid)
        if schema is not None:
            return schema

    if action_type:
        mapped = _LEGACY_ACTION_ALIASES.get(action_type)
        if mapped:
            schema = registry.get_action(mapped)
            if schema is not None:
                return schema
        # If an action_type root is provided and has exactly one action, resolve directly.
        cands = registry.get_actions_by_type(action_type)
        if len(cands) == 1:
            return cands[0]

    if action_subtype:
        cands = registry.get_actions_by_subtype(action_subtype)
        if len(cands) == 1:
            return cands[0]

    raise ValueError(
        "Could not resolve action schema. Provide --action-id or a resolvable --action-type/--action-subtype."
    )


def _default_param_value(pdef: Dict[str, Any]) -> Any:
    ptype = pdef.get("type")
    if ptype == "percent":
        lo = pdef.get("min")
        hi = pdef.get("max")
        if lo is not None and hi is not None:
            return float(lo) if float(lo) == float(hi) else float(lo + (hi - lo) * 0.25)
        if lo is not None:
            return float(lo)
        return 0.1
    if ptype == "numeric":
        if pdef.get("min") is not None:
            v = float(pdef.get("min"))
            return 1.0 if v == 0.0 else v
        return 1.0
    if ptype == "boolean":
        return False
    if ptype == "enum":
        vals = pdef.get("values", [])
        return vals[0] if vals else None
    if ptype == "funding_mix_object":
        return {"cash": 1.0, "debt": 0.0, "equity": 0.0}
    if ptype == "date_window":
        return {"start": None, "end": None}
    if ptype == "range":
        return {"min": 0.0, "max": 1.0}
    return None


def _materialize_action_params(schema: Dict[str, Any], action_params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    merged = dict(action_params or {})
    assumptions: List[str] = []
    for pname, pdef in schema.get("parameter_schema", {}).items():
        if pname in merged:
            continue
        if bool(pdef.get("required", False)):
            default_value = _default_param_value(pdef)
            if default_value is None and pdef.get("type") in {"entity_reference", "segment_reference"}:
                continue
            merged[pname] = default_value
            assumptions.append(f"default_param:{pname}")
    return merged, assumptions


def _snapshot_constraint_tokens(snapshot: CompanyStateSnapshot) -> List[str]:
    tokens: List[str] = []
    cs = snapshot.constraint_set
    if isinstance(cs, dict):
        for bucket in ("hard", "soft"):
            for item in cs.get(bucket, []) or []:
                if isinstance(item, dict):
                    if item.get("name"):
                        tokens.append(str(item["name"]))
                    if item.get("constraint_id"):
                        tokens.append(str(item["constraint_id"]))
                elif item is not None:
                    tokens.append(str(item))
    elif isinstance(cs, list):
        for item in cs:
            if isinstance(item, dict):
                if item.get("name"):
                    tokens.append(str(item["name"]))
                if item.get("constraint_id"):
                    tokens.append(str(item["constraint_id"]))
            elif item is not None:
                tokens.append(str(item))
    return list(dict.fromkeys(tokens))


def _infer_evidence_classes(snapshot: CompanyStateSnapshot) -> List[str]:
    classes = {"financial_disclosure"}
    prov = snapshot.provenance if isinstance(snapshot.provenance, dict) else {}
    inputs = prov.get("inputs_used", {}) if isinstance(prov.get("inputs_used"), dict) else {}
    if inputs.get("facts"):
        classes.update({"management_statement", "capital_policy_statement", "liquidity_disclosure"})
    if inputs.get("timeseries") or inputs.get("macro"):
        classes.add("market_signal")
    if inputs.get("events"):
        classes.update({"recent_action_history", "peer_context_signal"})
    if inputs.get("issuer_ratings"):
        classes.add("rating_disclosure")
    if inputs.get("ownership"):
        classes.add("recent_action_history")
    return sorted(classes)


def _extract_feature_value(feature_obj: Any) -> Any:
    if isinstance(feature_obj, dict):
        return feature_obj.get("value")
    return feature_obj


def _baseline_from_world_model_features(features: Dict[str, Any]) -> Dict[str, Any]:
    # Map world-model features into precedent baseline keys expected by stage1/stage2.
    out: Dict[str, Any] = {}
    if not isinstance(features, dict):
        return out

    def fv(name: str) -> Any:
        return resolve_feature_value(features, name)

    def first_value(*names: str) -> Any:
        for name in names:
            value = fv(name)
            if value is not None:
                return value
        return None

    out["market_cap"] = first_value(
        "scale.market_cap",
        "market.market_cap_provider_direct",
        "market.market_cap",
        "base_market_cap",
    )
    out["ebitda_margin"] = fv("operating.ebitda_margin_ttm")
    out["leverage_net_debt_ebitda"] = fv("capital_structure.net_leverage")
    out["fcf_margin"] = fv("operating.fcf_conversion")
    out["pe"] = fv("market.pe")
    out["revenue_ttm"] = fv("operating.revenue_ttm")
    out["roic"] = fv("operating.roic")
    out["sector"] = first_value("taxonomy.sector", "sector")
    out["subsector"] = first_value("taxonomy.subsector", "subsector", "industry")
    for key in _STATE_VECTOR_V1_FEATURES:
        out[key] = fv(key)
    return out


def _id_aliases(raw_id: str) -> List[str]:
    cid = str(raw_id)
    out: List[str] = [cid]
    if cid.isdigit():
        stripped = cid.lstrip("0")
        if stripped:
            out.append(stripped)
            for w in (6, 8, 9, 10):
                out.append(stripped.zfill(w))
        for w in (6, 8, 9, 10):
            out.append(cid.zfill(w))
    return list(dict.fromkeys(out))


def _is_materialized_local(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        st = path.stat()
    except OSError:
        return False
    if st.st_size <= 0:
        return False
    try:
        with path.open("rb") as f:
            f.read(1024)
    except Exception:
        return False
    return True


def _resolve_company_id_aliases_from_entity_identifier(
    company_id: str,
    entity_identifier_path: Optional[Path] = None,
) -> List[str]:
    path = entity_identifier_path or (DATA_DIR / "inputs_layer" / "entity_identifier.parquet")
    if not _is_materialized_local(path):
        return []

    try:
        ids = pd.read_parquet(path, columns=["entity_id", "identifier_value"])
    except Exception:
        return []
    if ids.empty:
        return []

    ids["entity_id"] = ids["entity_id"].astype(str)
    ids["identifier_value"] = ids["identifier_value"].astype(str)
    aliases = set(_id_aliases(company_id))

    matched = ids[ids["identifier_value"].isin(aliases) | ids["entity_id"].isin(aliases)]
    if matched.empty:
        return []

    entity_ids = set(matched["entity_id"].dropna().astype(str).tolist())
    expanded = ids[ids["entity_id"].isin(entity_ids)]

    out: List[str] = []
    for ent in sorted(entity_ids):
        out.extend(_id_aliases(ent))
    for ident in expanded["identifier_value"].dropna().astype(str).tolist():
        out.extend(_id_aliases(ident))
    return list(dict.fromkeys(out))


