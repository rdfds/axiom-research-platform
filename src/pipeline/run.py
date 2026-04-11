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


def _resolve_company_id_aliases_from_cik_gvkey(
    company_id: str,
    cik_gvkey_path: Optional[Path] = None,
) -> List[str]:
    path = cik_gvkey_path or (DATA_DIR / "wrds" / "compustat" / "cik_gvkey.csv.gz")
    if not _is_materialized_local(path):
        return []
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return []
    if df.empty:
        return []
    df.columns = [c.lower() for c in df.columns]
    if "gvkey" not in df.columns or "cik" not in df.columns:
        return []

    aliases = set(_id_aliases(company_id))
    numeric_aliases = {a for a in aliases if a.isdigit()}
    if not numeric_aliases:
        return []

    matches = df[
        df["gvkey"].astype(str).str.zfill(6).isin({a.zfill(6) for a in numeric_aliases})
        | df["gvkey"].astype(str).isin(numeric_aliases)
        | df["cik"].astype(str).isin(numeric_aliases)
    ]
    if matches.empty:
        return []

    out: List[str] = []
    for _, row in matches.iterrows():
        gv = str(row.get("gvkey", "")).strip()
        cik = str(row.get("cik", "")).strip()
        if gv:
            out.extend(_id_aliases(gv))
        if cik:
            out.extend(_id_aliases(cik))
    return list(dict.fromkeys(out))


def _load_company_state_snapshot_row(
    snapshot_jsonl_path: Path,
    company_id: str,
    as_of: datetime,
) -> Optional[Dict[str, Any]]:
    if not snapshot_jsonl_path.exists():
        return None
    company_id_str = str(company_id)
    aliases: set[str] = set(_id_aliases(company_id_str))
    as_of_date = pd.to_datetime(as_of).date()

    with snapshot_jsonl_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = str(row.get("company_id"))
            cid_match = cid in aliases
            if not cid_match and cid.isdigit() and company_id_str.isdigit():
                cid_match = cid.lstrip("0") == company_id_str.lstrip("0")
            if not cid_match:
                continue
            row_asof_raw = row.get("as_of_time")
            row_asof = pd.to_datetime(row_asof_raw, errors="coerce")
            if pd.isna(row_asof) or row_asof.date() != as_of_date:
                continue
            return row
    return None


def _load_company_state_keyed_snapshot_row(
    snapshot_root: Path,
    company_id: str,
    as_of: datetime,
) -> Optional[Dict[str, Any]]:
    from ..company_state_store import SnapshotStore

    store = SnapshotStore(root=snapshot_root)
    as_of_date = pd.to_datetime(as_of).strftime("%Y-%m-%d")
    aliases = _id_aliases(company_id)
    for alias in aliases:
        row = store.load_keyed_snapshot(company_id=str(alias), as_of=as_of_date)
        if row is not None:
            return row
    return None


def _outcome_aliases(action: ActionCandidate) -> List[str]:
    out: List[str] = []
    if action.action_id:
        out.append(action.action_id)
        legacy = {
            "capital_return.open_market_buyback": "buyback",
            "capital_return.accelerated_share_repurchase": "buyback",
            "capital_return.tender_offer_buyback": "buyback",
            "capital_return.dividend_increase": "dividend",
            "capital_return.dividend_cut": "dividend",
            "capital_return.dividend_initiate": "dividend",
            "capital_return.special_dividend": "dividend",
            "capital_structure.new_debt_issuance": "debt_issuance",
            "capital_structure.refinancing": "refinancing",
            "capital_structure.equity_issuance": "equity_issuance",
            "mna.tuck_in_acquisition": "acquisition",
            "mna.platform_acquisition": "acquisition",
            "mna.go_private_lbo": "acquisition",
            "mna.transformational_acquisition": "acquisition",
            "portfolio.divestiture_full": "divestiture",
            "portfolio.divestiture_partial": "divestiture",
            "portfolio.spin_off": "spin_off",
            "portfolio.asset_sale": "asset_sale",
            "governance.stock_split": "stock_split",
            "restructuring.cost_program": "cost_program",
        }.get(action.action_id)
        if legacy:
            out.append(legacy)
    if action.action_subtype:
        out.append(action.action_subtype)
    out.append(action.action_type)
    if action.action_id and "." in action.action_id:
        family, _, leaf = action.action_id.partition(".")
        if family:
            out.append(family)
        if leaf:
            out.append(leaf)
    return list(dict.fromkeys([x for x in out if x]))


def run_precedent(
    company_id: str,
    as_of_date: str,
    action_type: Optional[str] = None,
    action_subtype: Optional[str] = None,
    action_id: Optional[str] = None,
    action_params: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    outcomes_path: Optional[str] = None,
    state_snapshot_root: Optional[str] = None,
    state_snapshot_path: Optional[str] = None,
    state_snapshot: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
) -> PrecedentPack:
    wrapper_started = time.perf_counter()
    _precedent_debug(
        "run_precedent:start",
        company_id=company_id,
        as_of_date=as_of_date,
        action_id=action_id,
        action_type=action_type,
        action_subtype=action_subtype,
        run_id=run_id,
        candidate_id=candidate_id,
    )
    build_change_vector, load_config, FeatureBuilder, build_precedent_pack, build_precedent_pack_v2 = (
        _load_precedent_bindings()
    )

    config_started = time.perf_counter()
    _precedent_debug("config_load:start", config_path=config_path)
    config = load_config(config_path)
    _precedent_debug("config_load:done", elapsed_seconds=round(time.perf_counter() - config_started, 6))
    as_of = _parse_date(as_of_date)
    action_params = action_params or {}

    available_features: List[str] = []
    snapshot_for_validation: CompanyStateSnapshot
    baseline_features: Dict[str, Any]

    snapshot_row: Optional[Dict[str, Any]] = None
    if isinstance(state_snapshot, dict) and state_snapshot:
        snapshot_row = dict(state_snapshot)
    if snapshot_row is None and state_snapshot_root:
        snapshot_root = Path(state_snapshot_root)
        keyed_started = time.perf_counter()
        _precedent_debug(
            "keyed_snapshot_lookup:start",
            snapshot_root=str(snapshot_root),
            company_id=company_id,
            as_of_date=as_of_date,
        )
        snapshot_row = _load_company_state_keyed_snapshot_row(
            snapshot_root=snapshot_root,
            company_id=company_id,
            as_of=as_of,
        )
        _precedent_debug(
            "keyed_snapshot_lookup:done",
            found=bool(snapshot_row),
            elapsed_seconds=round(time.perf_counter() - keyed_started, 6),
        )
        if snapshot_row is None:
            alias_started = time.perf_counter()
            aliases = []
            aliases.extend(_resolve_company_id_aliases_from_entity_identifier(company_id))
            aliases.extend(_resolve_company_id_aliases_from_cik_gvkey(company_id))
            aliases = list(dict.fromkeys(aliases))
            _precedent_debug("keyed_snapshot_aliases:resolved", alias_count=len(aliases), aliases=aliases[:20])
            for alias in aliases:
                snapshot_row = _load_company_state_keyed_snapshot_row(
                    snapshot_root=snapshot_root,
                    company_id=alias,
                    as_of=as_of,
                )
                if snapshot_row is not None:
                    break
            _precedent_debug(
                "keyed_snapshot_aliases:done",
                found=bool(snapshot_row),
                elapsed_seconds=round(time.perf_counter() - alias_started, 6),
            )

    if snapshot_row is None and state_snapshot_path:
        snapshot_path = Path(state_snapshot_path)
        # Prefer keyed snapshot root if available; otherwise use jsonl path fallback.
        if snapshot_row is None:
            snapshot_started = time.perf_counter()
            _precedent_debug(
                "snapshot_path_lookup:start",
                snapshot_path=str(snapshot_path),
                company_id=company_id,
                as_of_date=as_of_date,
            )
            snapshot_row = _load_company_state_snapshot_row(snapshot_path, company_id=company_id, as_of=as_of)
            _precedent_debug(
                "snapshot_path_lookup:done",
                found=bool(snapshot_row),
                elapsed_seconds=round(time.perf_counter() - snapshot_started, 6),
            )
        if snapshot_row is None:
            alias_started = time.perf_counter()
            aliases = []
            aliases.extend(_resolve_company_id_aliases_from_entity_identifier(company_id))
            aliases.extend(_resolve_company_id_aliases_from_cik_gvkey(company_id))
            aliases = list(dict.fromkeys(aliases))
            _precedent_debug("snapshot_path_aliases:resolved", alias_count=len(aliases), aliases=aliases[:20])
            for alias in aliases:
                snapshot_row = _load_company_state_snapshot_row(snapshot_path, company_id=alias, as_of=as_of)
                if snapshot_row is not None:
                    break
            _precedent_debug(
                "snapshot_path_aliases:done",
                found=bool(snapshot_row),
                elapsed_seconds=round(time.perf_counter() - alias_started, 6),
            )
            if snapshot_row is None:
                tried = [company_id] + aliases[:20]
                raise ValueError(
                    f"company_id={company_id} as_of={as_of_date} not found in snapshot file: {snapshot_path}. "
                    f"Tried aliases: {tried}"
                )

    if snapshot_row is not None:
        snapshot_build_started = time.perf_counter()
        _precedent_debug("snapshot_materialize:start", source="snapshot_row")
        adapted_snapshot_row, _ = adapt_snapshot(snapshot_row)
        adapted_snapshot_row = attach_model_feature_bundle(adapted_snapshot_row)
        raw_features = feature_view_from_snapshot(adapted_snapshot_row, view_name="precedent")
        available_features = list(raw_features.keys())
        baseline_features = _baseline_from_world_model_features(raw_features)
        snapshot_for_validation = CompanyStateSnapshot(
            company_id=str(adapted_snapshot_row.get("company_id")),
            as_of_time=pd.to_datetime(adapted_snapshot_row.get("as_of_time")).to_pydatetime(),
            features=baseline_features,
            regime=(
                adapted_snapshot_row.get("regime", {})
                if isinstance(adapted_snapshot_row.get("regime"), dict)
                else {}
            ),
            constraint_set=adapted_snapshot_row.get("constraint_set", []),
            provenance=(
                adapted_snapshot_row.get("provenance", {})
                if isinstance(adapted_snapshot_row.get("provenance"), dict)
                else {}
            ),
        )
        _precedent_debug(
            "snapshot_materialize:done",
            source="snapshot_row",
            feature_count=len(available_features),
            elapsed_seconds=round(time.perf_counter() - snapshot_build_started, 6),
        )
    else:
        snapshot_build_started = time.perf_counter()
        _precedent_debug("snapshot_materialize:start", source="feature_builder")
        builder = FeatureBuilder()
        macro_series = config.get("macro_series", {})
        snapshot_for_validation = builder.build_company_state(company_id, as_of, macro_series)
        adapted_snapshot, _ = adapt_snapshot(snapshot_for_validation.to_dict())
        adapted_snapshot = attach_model_feature_bundle(adapted_snapshot)
        precedent_features = feature_view_from_snapshot(adapted_snapshot, view_name="precedent")
        snapshot_for_validation = CompanyStateSnapshot(
            company_id=str(adapted_snapshot.get("company_id", snapshot_for_validation.company_id)),
            as_of_time=pd.to_datetime(
                adapted_snapshot.get("as_of_time", snapshot_for_validation.as_of_time)
            ).to_pydatetime(),
            features=precedent_features,
            regime=adapted_snapshot.get("regime", {}) if isinstance(adapted_snapshot.get("regime"), dict) else {},
            constraint_set=adapted_snapshot.get("constraint_set", []),
            provenance=(
                adapted_snapshot.get("provenance", {})
                if isinstance(adapted_snapshot.get("provenance"), dict)
                else {}
            ),
        )
        baseline_features = snapshot_for_validation.features
        available_features = list((snapshot_for_validation.features or {}).keys())
        _precedent_debug(
            "snapshot_materialize:done",
            source="feature_builder",
            feature_count=len(available_features),
            elapsed_seconds=round(time.perf_counter() - snapshot_build_started, 6),
        )

    schema_started = time.perf_counter()
    _precedent_debug("resolve_action_schema:start", action_id=action_id, action_type=action_type, action_subtype=action_subtype)
    registry = _default_registry()
    schema = _resolve_action_schema(registry, action_type=action_type, action_subtype=action_subtype, action_id=action_id)
    resolved_params, assumptions = _materialize_action_params(schema, action_params)
    _precedent_debug(
        "resolve_action_schema:done",
        resolved_action_id=str(schema["action_id"]),
        elapsed_seconds=round(time.perf_counter() - schema_started, 6),
    )

    validation_started = time.perf_counter()
    _precedent_debug("candidate_validation:start", resolved_action_id=str(schema["action_id"]))
    validation = registry.validate_candidate(
        {
            "action_id": schema["action_id"],
            "parameters": resolved_params,
            "available_features": available_features,
            "available_evidence_classes": _infer_evidence_classes(snapshot_for_validation),
            "constraints": _snapshot_constraint_tokens(snapshot_for_validation),
        },
        strict_evidence=False,
    )
    _precedent_debug(
        "candidate_validation:done",
        valid=bool(validation.valid),
        error_count=len(validation.errors),
        elapsed_seconds=round(time.perf_counter() - validation_started, 6),
    )
    if not validation.valid:
        raise ValueError(
            f"Action candidate validation failed for {schema['action_id']}: "
            + "; ".join(validation.errors)
        )

    action = ActionCandidate(
        action_type=schema["action_type"],
        action_subtype=schema["action_subtype"],
        action_id=schema["action_id"],
        params=resolved_params,
        assumed_preconditions=assumptions,
    )
    change_vector = build_change_vector(action, config)

    outcomes_path = Path(outcomes_path) if outcomes_path else _default_precedent_outcomes_path()
    if not outcomes_path.exists():
        raise FileNotFoundError(f"Missing action outcomes dataset: {outcomes_path}")
    runtime_started = time.perf_counter()
    _precedent_debug("precedent_runtime:start", outcomes_path=str(outcomes_path))
    stores, retrieval_index = _load_precedent_runtime(outcomes_path)
    _precedent_debug(
        "precedent_runtime:done",
        elapsed_seconds=round(time.perf_counter() - runtime_started, 6),
        index_rows=int(getattr(retrieval_index, "n_rows", 0) or 0),
    )
    # Precedent Brain v2 artifact (keeps legacy distributions for compatibility).
    try:
        v2_started = time.perf_counter()
        _precedent_debug("precedent_brain_v2:start", action_id=str(schema["action_id"]))
        pack = build_precedent_pack_v2(
            candidate_id=str(candidate_id or f"{company_id}:{schema['action_id']}:{as_of_date}"),
            run_id=str(run_id or ""),
            company_id=str(company_id),
            action_id=str(schema["action_id"]),
            action_subtype=str(schema.get("action_subtype") or ""),
            action_params=resolved_params,
            candidate_features=baseline_features if isinstance(baseline_features, dict) else {},
            candidate_regime=snapshot_for_validation.regime if isinstance(snapshot_for_validation.regime, dict) else {},
            historical_event_store=stores["historical_event_store"],
            historical_state_store=stores["historical_state_store"],
            historical_outcome_store=stores["historical_outcome_store"],
            regime_history=stores["regime_history"],
            retrieval_index=retrieval_index,
            top_k=30,
            min_k=10,
        )
        _precedent_debug(
            "precedent_brain_v2:done",
            action_id=str(schema["action_id"]),
            elapsed_seconds=round(time.perf_counter() - v2_started, 6),
            total_wrapper_seconds=round(time.perf_counter() - wrapper_started, 6),
        )
        return pack
    except Exception as exc:
        # Conservative fallback to legacy pack to avoid pipeline interruptions.
        _precedent_debug(
            "precedent_brain_v2:fallback_legacy",
            action_id=str(schema["action_id"]),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(limit=20) if _PRECEDENT_DEBUG else None,
            total_wrapper_seconds=round(time.perf_counter() - wrapper_started, 6),
        )
        outcomes_started = time.perf_counter()
        _precedent_debug("outcomes_table:start", outcomes_path=str(outcomes_path))
        full_df = _load_outcomes_table(outcomes_path)
        _precedent_debug(
            "outcomes_table:done",
            rows=int(len(full_df)),
            elapsed_seconds=round(time.perf_counter() - outcomes_started, 6),
        )
        df = full_df
        aliases = _outcome_aliases(action)
        _precedent_debug("outcome_aliases", aliases=aliases)
        filter_columns = (
            ("normalized_action_id", "outcomes_filter:normalized_action_id"),
            ("normalized_action_subfamily", "outcomes_filter:normalized_action_subfamily"),
            ("normalized_action_family", "outcomes_filter:normalized_action_family"),
            ("action_id", "outcomes_filter:action_id"),
            ("raw_action_subtype", "outcomes_filter:raw_action_subtype"),
            ("action_subtype", "outcomes_filter:action_subtype"),
            ("raw_action_type", "outcomes_filter:raw_action_type"),
            ("action_type", "outcomes_filter:action_type"),
        )
        for col, stage in filter_columns:
            if col not in df.columns:
                continue
            matched = df[df[col].astype(str).isin(aliases)]
            if matched.empty:
                continue
            df = matched
            _precedent_debug(stage, rows=int(len(df)))
            break

        outcome_cfg = config.get("outcome", {})
        metric = outcome_cfg.get("primary_metric", "pe")
        horizons = outcome_cfg.get("horizons_months", [3, 6, 12])
        outcome_cols = [f"outcome_{metric}_{h}m" for h in horizons if h]

        target_col = outcome_cols[-1] if outcome_cols else f"outcome_{metric}_12m"
        legacy_started = time.perf_counter()
        _precedent_debug(
            "legacy_precedent_pack:start",
            rows=int(len(df)),
            target_col=target_col,
            outcome_cols=outcome_cols,
        )
        legacy_pack = build_precedent_pack(
            df=df,
            change_vector=change_vector,
            baseline=baseline_features,
            config=config.get("similarity", {}),
            target_col=target_col,
            outcome_cols=outcome_cols,
            top_n=50,
        )
        _precedent_debug(
            "legacy_precedent_pack:done",
            elapsed_seconds=round(time.perf_counter() - legacy_started, 6),
        )
        return legacy_pack
