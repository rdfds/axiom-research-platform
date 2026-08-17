"""Causal-style impact model utilities for Mechanism Brain.

This module provides:
- Offline training data serialization format
- Fast deterministic inference for objective impact distributions
- Action-id -> legacy action-type normalization for outcomes data alignment
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from .causal_feature_contract import (
    INFERENCE_SOURCE_KEYS,
    build_contract_feature_map,
    canonicalize_feature_name,
)
from .model_feature_bundle import build_model_feature_bundle, feature_view_from_snapshot, get_bundle_value
from .runtime_feature_adapter import adapt_snapshot, resolve_feature_value


_Z = {
    "p10": -1.2815515655446004,
    "p25": -0.6744897501960817,
    "median": 0.0,
    "p75": 0.6744897501960817,
    "p90": 1.2815515655446004,
}

DEFAULT_CAUSAL_IMPACT_MODEL_ARTIFACT = Path(
    "data/models/causal_impact_model_v6_bundle_contract_hgb_actiontype.json"
)
DEFAULT_CAUSAL_ROUTING_CONFIG_PATH = Path("configs/causal_capital_routing_prod_dividend_v2.json")

# Backward-compatible feature normalization defaults for legacy artifacts
# that predate explicit transform metadata.
_DEFAULT_USD_MILLIONS_FEATURES = {
    "base_market_cap",
    "base_net_debt",
    "base_revenue_ttm",
    "action_size",
}
_DEFAULT_RATE_PERCENT_FEATURES = {
    "macro_rate_10y",
    "macro_rate_2y",
    "macro_sofr",
}
_DEFAULT_OAS_PERCENT_FEATURES = {
    "macro_ig_oas",
    "macro_hy_oas",
}


class _RidgePredictor:
    """Compatibility wrapper for legacy pickled ridge predictors."""

    def __init__(self, beta: Any = None) -> None:
        self.beta = beta

    def predict(self, X: Any) -> list[float]:  # noqa: N803
        beta_raw = self.beta
        if beta_raw is None:
            return [0.0 for _ in list(X or [])]
        try:
            beta = [float(v) for v in list(beta_raw)]
        except Exception:
            return [0.0 for _ in list(X or [])]
        if not beta:
            return [0.0 for _ in list(X or [])]
        out: list[float] = []
        for row_raw in list(X or []):
            try:
                row = [float(v) for v in list(row_raw)]
            except Exception:
                row = []
            y = beta[0]
            width = min(len(row), max(0, len(beta) - 1))
            for idx in range(width):
                y += beta[idx + 1] * row[idx]
            out.append(float(y))
        return out


class _BundleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        # Legacy training artifacts may pickle _RidgePredictor under __main__
        # when the trainer is executed as a script path.
        if module == "__main__" and name == "_RidgePredictor":
            return _RidgePredictor
        return super().find_class(module, name)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    qq = _clip(float(q), 0.0, 1.0)
    pos = qq * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _scalar_float(value: Any) -> Optional[float]:
    raw = _to_float(value)
    if raw is not None:
        return float(raw)
    try:
        seq = list(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if not seq:
        return None
    return _scalar_float(seq[0])


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    if isinstance(v, bool):
        return float(v)
    try:
        out = float(v)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _feature_value(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw.get("value")
    return raw


def _extract_feature(features: Dict[str, Any], name: str, default: Any = None) -> Any:
    return resolve_feature_value(features, name, default=default)


def _nested_get(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class CausalActionPolicy:
    action_id: str
    status: str
    model_action_alias: str
    model_subtype_alias: str
    objective_allowlist: tuple[str, ...]
    strict_gate_primary_objectives: tuple[str, ...]
    model_artifact_path_override: str
    max_blend_weight: Optional[float]
    notes: str
    future_action_alias: str
    future_action_aliases: tuple[str, ...]
    quality_floor_override: Optional[float]
    support_floor_override: Optional[float]
    min_train_rows_override: Optional[int]
    min_oos_r2_override: Optional[float]
    min_treated_rows_override: Optional[int]
    min_control_rows_override: Optional[int]


def _legacy_action_id_to_outcomes_action_type(action_id: str, action_type: str = "") -> str:
    aid = str(action_id or "")
    if aid in {
        "capital_return.open_market_buyback",
        "capital_return.accelerated_share_repurchase",
        "capital_return.tender_offer_buyback",
    }:
        return "buyback"
    if aid == "capital_return.dividend_increase":
        return "dividend_increase"
    if aid == "capital_return.dividend_cut":
        return "dividend_cut"
    if aid == "capital_return.dividend_initiate":
        # There is no dedicated dividend-initiation cell in the current model.
        # Use the broader regular-dividend family as the conservative causal
        # prior rather than dropping causal support entirely.
        return "dividend_regular"
    if aid == "capital_return.special_dividend":
        # There is no durable special-dividend cell in the current causal model.
        # Use the broader regular-dividend family as a conservative fallback so
        # we can still blend a finance-policy prior instead of dropping causal
        # support entirely.
        return "dividend_regular"
    if aid in {
        "capital_structure.new_debt_issuance",
        "capital_structure.convertible_issuance",
        "capital_structure.refinancing",
        "capital_structure.tender_offer_debt",
        "capital_structure.exchange_offer",
        "capital_structure.liability_management_exercise",
    }:
        return "bond_issuance"
    if aid == "capital_structure.revolver_draw_or_resize":
        return "loan_issuance"
    if aid in {
        "capital_structure.equity_issuance",
        "capital_structure.preferred_issuance",
    }:
        return "equity_offering_public_proxy"
    if aid == "mna.go_private_lbo":
        return "acquisition"
    if aid.startswith("mna.") or aid == "portfolio.joint_venture":
        return "acquisition"
    if aid in {"portfolio.spin_off", "portfolio.carve_out_ipo"}:
        return "spin_off"
    if aid.startswith("portfolio."):
        return "divestiture"
    if aid.startswith("restructuring.") or aid.startswith("governance."):
        return "cost_program"

    at = str(action_type or "").strip()
    if at:
        return at
    if "." in aid:
        return aid.split(".", 1)[0]
    return aid


def _canonical_subtype(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return "unknown"
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _legacy_action_subtype_to_outcomes_subtype(action_id: str, action_type: str = "", action_subtype: str = "") -> str:
    aid = str(action_id or "")
    alias = _legacy_action_id_to_outcomes_action_type(action_id=aid, action_type=action_type)
    st = str(action_subtype or "")
    if not st and "." in aid:
        st = aid.split(".", 1)[1]
    st_key = _canonical_subtype(st)

    # Hand-tuned bridge from ontology subtypes to historical subtype taxonomy.
    if alias == "buyback":
        return "buyback"
    if alias == "dividend_regular":
        return "regular"
    if alias == "dividend_special":
        return "special"
    if alias == "dividend_cut":
        return "dividend_cut"
    if alias == "dividend_increase":
        return "dividend_increase"
    if alias == "equity_offering_public_proxy":
        return "share_issuance_proxy"
    if alias == "bond_issuance":
        return "unknown"
    if aid == "mna.go_private_lbo":
        return "acquisition_lbo"
    if aid == "capital_structure.revolver_draw_or_resize":
        # The current outcomes corpus has usable coverage primarily in the
        # long-dated revolver bucket; map the recommendation action there so
        # subtype-level rescue cells can be selected at runtime.
        return "revolver_line_1_yr"
    return st_key


def _default_causal_routing_config_path() -> Path:
    env = str(os.environ.get("CAUSAL_ROUTING_CONFIG_PATH", "")).strip()
    if env:
        return Path(env)
    return DEFAULT_CAUSAL_ROUTING_CONFIG_PATH


@lru_cache(maxsize=2)
def load_causal_routing_config(path_str: Optional[str] = None) -> Dict[str, Any]:
    path = Path(path_str) if path_str else _default_causal_routing_config_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def get_causal_action_policy(
    action_id: str,
    action_type: str = "",
    action_subtype: str = "",
) -> CausalActionPolicy:
    aid = str(action_id or "").strip()
    legacy_action_alias = _legacy_action_id_to_outcomes_action_type(aid, action_type)
    legacy_subtype_alias = _legacy_action_subtype_to_outcomes_subtype(aid, action_type, action_subtype)
    config = load_causal_routing_config()
    actions = dict(config.get("actions", {}) or {})
    status_blend_caps = dict(config.get("status_max_blend_weight", {}) or {})
    spec = dict(actions.get(aid, {}) or {})

    status = str(spec.get("status", "unconfigured") or "unconfigured").strip().lower()
    if not status:
        status = "unconfigured"
    model_action_alias = str(spec.get("model_action_alias", legacy_action_alias) or legacy_action_alias).strip()
    model_subtype_alias = str(spec.get("model_subtype_alias", legacy_subtype_alias) or legacy_subtype_alias).strip()
    objective_allowlist_raw = spec.get("objective_allowlist", [])
    objective_allowlist = tuple(
        str(x).strip()
        for x in list(objective_allowlist_raw or [])
        if str(x).strip()
    )
    strict_gate_primary_objectives_raw = spec.get("strict_gate_primary_objectives", [])
    strict_gate_primary_objectives = tuple(
        str(x).strip()
        for x in list(strict_gate_primary_objectives_raw or [])
        if str(x).strip()
    )
    model_artifact_path_override = str(spec.get("model_artifact_path_override", "") or "").strip()
    max_blend_weight = _to_float(spec.get("max_blend_weight"))
    if max_blend_weight is None:
        max_blend_weight = _to_float(status_blend_caps.get(status))
    notes = str(spec.get("notes", "") or "")
    future_action_aliases_raw = list(spec.get("future_action_aliases", []) or [])
    future_action_alias = str(spec.get("future_action_alias", "") or "")
    gate_overrides = dict(spec.get("strict_gate_overrides", {}) or {})
    future_action_aliases: list[str] = []
    for item in future_action_aliases_raw:
        alias = str(item or "").strip()
        if alias and alias not in future_action_aliases:
            future_action_aliases.append(alias)
    if future_action_alias and future_action_alias not in future_action_aliases:
        future_action_aliases.insert(0, future_action_alias)
    return CausalActionPolicy(
        action_id=aid,
        status=status,
        model_action_alias=model_action_alias or legacy_action_alias,
        model_subtype_alias=model_subtype_alias or legacy_subtype_alias,
        objective_allowlist=objective_allowlist,
        strict_gate_primary_objectives=strict_gate_primary_objectives,
        model_artifact_path_override=model_artifact_path_override,
        max_blend_weight=max_blend_weight,
        notes=notes,
        future_action_alias=future_action_alias,
        future_action_aliases=tuple(future_action_aliases),
        quality_floor_override=_to_float(gate_overrides.get("quality_floor")),
        support_floor_override=_to_float(gate_overrides.get("support_floor")),
        min_train_rows_override=(
            int(max(0.0, _to_float(gate_overrides.get("min_train_rows"), 0.0) or 0.0))
            if gate_overrides.get("min_train_rows") is not None
            else None
        ),
        min_oos_r2_override=_to_float(gate_overrides.get("min_oos_r2")),
        min_treated_rows_override=(
            int(max(0.0, _to_float(gate_overrides.get("min_treated_rows"), 0.0) or 0.0))
            if gate_overrides.get("min_treated_rows") is not None
            else None
        ),
        min_control_rows_override=(
            int(max(0.0, _to_float(gate_overrides.get("min_control_rows"), 0.0) or 0.0))
            if gate_overrides.get("min_control_rows") is not None
            else None
        ),
    )


def action_id_to_outcomes_action_type(action_id: str, action_type: str = "") -> str:
    policy = get_causal_action_policy(action_id=action_id, action_type=action_type)
    return str(policy.model_action_alias or "")


def action_subtype_to_outcomes_subtype(action_id: str, action_type: str = "", action_subtype: str = "") -> str:
    policy = get_causal_action_policy(
        action_id=action_id,
        action_type=action_type,
        action_subtype=action_subtype,
    )
    return str(policy.model_subtype_alias or "")


@dataclass
class CausalPrediction:
    objectives: Dict[str, Dict[str, float]]
    blend_weight: float
    coverage_score: float
    n_train: int
    model_version: str
    model_quality: float
    support_score: float
    out_of_sample_flag: bool
    min_oos_r2: Optional[float]
    min_treated_rows: int
    min_control_rows: int
    selected_model_keys: list[str]
    gate_reason: str
    action_status: str
    objective_allowlist: list[str]
    strict_gate_primary_objectives: list[str]
    model_artifact_path_override: str
    max_blend_weight: Optional[float]
    future_action_alias: str
    future_action_aliases: list[str]
    quality_floor_override: Optional[float]
    support_floor_override: Optional[float]
    min_train_rows_override: Optional[int]
    min_oos_r2_override: Optional[float]
    min_treated_rows_override: Optional[int]
    min_control_rows_override: Optional[int]


@dataclass
class CausalRoutingDiagnostics:
    action_alias: str
    subtype_alias: str
    blend_weight: float
    coverage_score: float
    n_train: int
    model_version: str
    model_quality: float
    support_score: float
    out_of_sample_flag: bool
    min_oos_r2: Optional[float]
    min_treated_rows: int
    min_control_rows: int
    selected_model_keys: list[str]
    selected_models_by_objective: Dict[str, Dict[str, Any]]
    gate_reason: str
    action_status: str
    objective_allowlist: list[str]
    strict_gate_primary_objectives: list[str]
    model_artifact_path_override: str
    max_blend_weight: Optional[float]
    future_action_alias: str
    future_action_aliases: list[str]
    quality_floor_override: Optional[float]
    support_floor_override: Optional[float]
    min_train_rows_override: Optional[int]
    min_oos_r2_override: Optional[float]
    min_treated_rows_override: Optional[int]
    min_control_rows_override: Optional[int]


class CausalImpactModel:
    def __init__(self, payload: Dict[str, Any], artifact_path: Optional[Path] = None) -> None:
        self.payload = payload if isinstance(payload, dict) else {}
        self.feature_order = list(self.payload.get("feature_order", []) or [])
        self.feature_stats = dict(self.payload.get("feature_stats", {}) or {})
        self.objectives = dict(self.payload.get("objectives", {}) or {})
        spec = dict(self.payload.get("feature_transform_spec", {}) or {})
        if spec:
            self._usd_millions_features = set(spec.get("usd_millions_features", []) or [])
            self._rate_percent_features = set(spec.get("rate_percent_features", []) or [])
            self._oas_percent_features = set(spec.get("oas_percent_features", []) or [])
            self._signed_log1p_features = set(spec.get("signed_log1p_features", []) or [])
        else:
            # Legacy model artifacts do not include transform metadata.
            # Apply only unit harmonization defaults; keep value transform empty
            # to avoid changing model semantics.
            self._usd_millions_features = set(_DEFAULT_USD_MILLIONS_FEATURES)
            self._rate_percent_features = set(_DEFAULT_RATE_PERCENT_FEATURES)
            self._oas_percent_features = set(_DEFAULT_OAS_PERCENT_FEATURES)
            self._signed_log1p_features = set()
        self.version = str(self.payload.get("version", "causal_impact_model_v1"))
        self.artifact_path = artifact_path
        self._bundle_cache: Optional[Dict[str, Any]] = None
        min_obj_oos_env = _to_float(os.environ.get("CAUSAL_MIN_OBJECTIVE_OOS_R2"))
        min_obj_oos_payload = _to_float(self.payload.get("inference_min_objective_oos_r2"))
        # Keep default permissive for backward compatibility.
        self.min_objective_oos_r2 = float(
            min_obj_oos_env if min_obj_oos_env is not None else (min_obj_oos_payload if min_obj_oos_payload is not None else -1.0)
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "CausalImpactModel":
        p = Path(path)
        payload = json.loads(p.read_text())
        return cls(payload, artifact_path=p)

    def _override_model_for_action(
        self,
        *,
        action_id: str,
        action_type: str,
        action_subtype: str,
    ) -> Optional["CausalImpactModel"]:
        policy = get_causal_action_policy(
            action_id=action_id,
            action_type=action_type,
            action_subtype=action_subtype,
        )
        override_raw = str(policy.model_artifact_path_override or "").strip()
        if not override_raw:
            return None
        override_path = Path(override_raw)
        current_path = self.artifact_path.resolve() if isinstance(self.artifact_path, Path) else None
        try:
            resolved_override = override_path.resolve()
        except Exception:
            resolved_override = override_path
        if current_path is not None and resolved_override == current_path:
            return None
        return load_default_causal_impact_model(str(override_path))

    def predict(
        self,
        action_id: str,
        action_type: str,
        params: Dict[str, Any],
        features: Dict[str, Any],
        regime: Dict[str, Any],
        action_subtype: str = "",
    ) -> Optional[CausalPrediction]:
        override_model = self._override_model_for_action(
            action_id=action_id,
            action_type=action_type,
            action_subtype=action_subtype,
        )
        if override_model is not None:
            return override_model.predict(
                action_id=action_id,
                action_type=action_type,
                params=params,
                features=features,
                regime=regime,
                action_subtype=action_subtype,
            )
        features, bundle = self._resolve_causal_inputs(
            features,
            action_id=action_id,
            action_type=action_type,
            regime=regime,
        )
        routing = self.diagnose(
            action_id=action_id,
            action_type=action_type,
            params=params,
            features=features,
            regime=regime,
            action_subtype=action_subtype,
        )
        if routing is None:
            return None

        action_alias = routing.action_alias
        subtype_alias = routing.subtype_alias
        objective_allowlist = set(routing.objective_allowlist)
        alias_pairs = self._routing_alias_pairs(
            action_id=action_id,
            action_type=action_type,
            action_subtype=action_subtype,
            action_alias=action_alias,
            subtype_alias=subtype_alias,
            future_action_aliases=routing.future_action_aliases,
        )
        x_std = self._standardized_feature_vector(params=params, features=features, regime=regime, bundle=bundle)
        if x_std is None:
            return None

        out: Dict[str, Dict[str, float]] = {}
        for objective_name, objective_payload in self.objectives.items():
            if objective_allowlist and objective_name not in objective_allowlist:
                continue
            payload_obj = dict(objective_payload or {})
            dr_models = dict(payload_obj.get("dr_models", {}) or {})
            models = dict(payload_obj.get("models", {}) or {})
            keys = self._candidate_model_keys_multi(alias_pairs)
            model, selected_key = self._select_model(
                dr_models=dr_models,
                models=models,
                keys=keys,
            )
            if not isinstance(model, dict):
                continue
            model_oos_r2 = _to_float(model.get("oos_r2"))
            if self.min_objective_oos_r2 > -1.0:
                if model_oos_r2 is None or float(model_oos_r2) < float(self.min_objective_oos_r2):
                    continue
            pred = self._predict_objective(model, x_std, selected_key=selected_key)
            if pred is None:
                continue
            out[objective_name] = pred

        if not out:
            return None

        return CausalPrediction(
            objectives=out,
            blend_weight=routing.blend_weight,
            coverage_score=routing.coverage_score,
            n_train=routing.n_train,
            model_version=routing.model_version,
            model_quality=routing.model_quality,
            support_score=routing.support_score,
            out_of_sample_flag=routing.out_of_sample_flag,
            min_oos_r2=routing.min_oos_r2,
            min_treated_rows=routing.min_treated_rows,
            min_control_rows=routing.min_control_rows,
            selected_model_keys=list(routing.selected_model_keys),
            gate_reason=routing.gate_reason,
            action_status=routing.action_status,
            objective_allowlist=list(routing.objective_allowlist),
            strict_gate_primary_objectives=list(routing.strict_gate_primary_objectives),
            model_artifact_path_override=routing.model_artifact_path_override,
            max_blend_weight=routing.max_blend_weight,
            future_action_alias=routing.future_action_alias,
            future_action_aliases=list(routing.future_action_aliases),
            quality_floor_override=routing.quality_floor_override,
            support_floor_override=routing.support_floor_override,
            min_train_rows_override=routing.min_train_rows_override,
            min_oos_r2_override=routing.min_oos_r2_override,
            min_treated_rows_override=routing.min_treated_rows_override,
            min_control_rows_override=routing.min_control_rows_override,
        )

    def diagnose(
        self,
        action_id: str,
        action_type: str,
        params: Dict[str, Any],
        features: Dict[str, Any],
        regime: Dict[str, Any],
        action_subtype: str = "",
    ) -> Optional[CausalRoutingDiagnostics]:
        override_model = self._override_model_for_action(
            action_id=action_id,
            action_type=action_type,
            action_subtype=action_subtype,
        )
        if override_model is not None:
            return override_model.diagnose(
                action_id=action_id,
                action_type=action_type,
                params=params,
                features=features,
                regime=regime,
                action_subtype=action_subtype,
            )
        if not self.feature_order or not self.objectives:
            return None
        features, bundle = self._resolve_causal_inputs(
            features,
            action_id=action_id,
            action_type=action_type,
            regime=regime,
        )
        policy = get_causal_action_policy(
            action_id=action_id,
            action_type=action_type,
            action_subtype=action_subtype,
        )
        action_alias = str(policy.model_action_alias or "")
        subtype_alias = str(policy.model_subtype_alias or "")
        objective_allowlist = set(policy.objective_allowlist)
        alias_pairs = self._routing_alias_pairs(
            action_id=action_id,
            action_type=action_type,
            action_subtype=action_subtype,
            action_alias=action_alias,
            subtype_alias=subtype_alias,
            future_action_aliases=list(policy.future_action_aliases),
        )
        x_std = self._standardized_feature_vector(params=params, features=features, regime=regime, bundle=bundle)
        if x_std is None:
            return None
        z_abs = [abs(float(v)) for v in x_std.values()]
        z_mean = sum(z_abs) / max(1, len(z_abs))
        z_p90 = _quantile(z_abs, 0.90)
        support_score = _clip(1.0 - (z_mean / 4.0), 0.0, 1.0)
        out_of_sample_flag = bool(support_score < 0.20 or z_p90 > 4.5)

        objective_stats: Dict[str, Dict[str, Any]] = {}
        selected_model_keys: list[str] = []
        selected_models_by_objective: Dict[str, Dict[str, Any]] = {}
        gate_reasons: list[str] = []
        for objective_name, objective_payload in self.objectives.items():
            if objective_allowlist and objective_name not in objective_allowlist:
                continue
            payload_obj = dict(objective_payload or {})
            dr_models = dict(payload_obj.get("dr_models", {}) or {})
            models = dict(payload_obj.get("models", {}) or {})
            keys = self._candidate_model_keys_multi(alias_pairs)
            model, selected_key = self._select_model(
                dr_models=dr_models,
                models=models,
                keys=keys,
            )
            if not isinstance(model, dict):
                continue
            model_oos_r2 = _to_float(model.get("oos_r2"))
            # Accuracy-first optional filter: drop weak objective cells before blending.
            if self.min_objective_oos_r2 > -1.0:
                if model_oos_r2 is None or float(model_oos_r2) < float(self.min_objective_oos_r2):
                    continue
            if selected_key:
                selected_model_keys.append(str(selected_key))
            selected_models_by_objective[objective_name] = {
                "selected_key": str(selected_key),
                "enabled": bool(model.get("enabled", True)),
                "gate_reason": str(model.get("gate_reason", "") or ""),
                "oos_r2": _to_float(model.get("oos_r2")),
                "treated_rows": int(_to_float(model.get("treated_rows"), 0.0) or 0.0),
                "control_rows": int(_to_float(model.get("control_rows"), 0.0) or 0.0),
                "n_train": int(_to_float(model.get("n_train"), 0.0) or 0.0),
                "model_family": str(model.get("model_family", "") or ""),
            }
            n_train = int(_to_float(model.get("n_train"), 0.0) or 0)
            treated_rows = int(_to_float(model.get("treated_rows"), 0.0) or 0.0)
            control_rows = int(_to_float(model.get("control_rows"), 0.0) or 0.0)
            oos_r2 = _to_float(model.get("oos_r2"))
            reason = str(model.get("gate_reason", "")).strip()
            if reason and reason not in gate_reasons:
                gate_reasons.append(reason)
            quality_raw = _to_float(model.get("oos_r2"))
            if quality_raw is None:
                quality_raw = _to_float(model.get("r2"), 0.0) or 0.0
            quality_raw = _clip(float(quality_raw), -1.0, 1.0)
            objective_stats[objective_name] = {
                "n_train": n_train,
                "treated_rows": treated_rows,
                "control_rows": control_rows,
                "oos_r2": float(_clip(oos_r2, -1.0, 1.0)) if oos_r2 is not None else None,
                "quality_raw": float(quality_raw),
                # Map model quality into [0,1], with values below -0.2 treated as uninformative.
                "quality": _clip((float(quality_raw) + 0.20) / 0.80, 0.0, 1.0),
            }

        if not selected_models_by_objective:
            return None

        gate_objectives = [
            objective_name
            for objective_name in policy.strict_gate_primary_objectives
            if objective_name in objective_stats
        ]
        if not gate_objectives:
            gate_objectives = list(objective_stats)

        n_train_min = None
        quality_accum: list[float] = []
        quality_raw_accum: list[float] = []
        oos_values: list[float] = []
        treated_values: list[int] = []
        control_values: list[int] = []
        for objective_name in gate_objectives:
            stats = objective_stats.get(objective_name, {})
            n_train = int(stats.get("n_train", 0) or 0)
            n_train_min = n_train if n_train_min is None else min(n_train_min, n_train)
            treated_rows = int(stats.get("treated_rows", 0) or 0)
            control_rows = int(stats.get("control_rows", 0) or 0)
            if treated_rows > 0:
                treated_values.append(treated_rows)
            if control_rows > 0:
                control_values.append(control_rows)
            oos_r2 = stats.get("oos_r2")
            if oos_r2 is not None:
                oos_values.append(float(oos_r2))
            quality_raw_accum.append(float(stats.get("quality_raw", 0.0) or 0.0))
            quality_accum.append(float(stats.get("quality", 0.0) or 0.0))

        min_n = int(n_train_min or 0)
        quality = sum(quality_accum) / max(1, len(quality_accum))
        quality_raw = sum(quality_raw_accum) / max(1, len(quality_raw_accum))
        # Conservative blend weighting to keep deterministic/mechanistic priors dominant.
        blend_weight = 0.05 + 0.55 * (min_n / (min_n + 1200.0)) * quality
        if min_n < 250:
            blend_weight = min(blend_weight, 0.25)
        if quality_raw < 0.0:
            # If out-of-sample quality is negative, keep ML contribution minimal.
            blend_weight = min(blend_weight, 0.12)
        blend_weight = _clip(float(blend_weight), 0.05, 0.55)
        coverage_score = _clip(float(quality), 0.0, 1.0)
        min_oos_r2 = min(oos_values) if oos_values else None
        min_treated_rows = min(treated_values) if treated_values else 0
        min_control_rows = min(control_values) if control_values else 0
        dedup_keys: list[str] = []
        for k in selected_model_keys:
            if k not in dedup_keys:
                dedup_keys.append(k)
        return CausalRoutingDiagnostics(
            action_alias=str(action_alias),
            subtype_alias=str(subtype_alias),
            blend_weight=round(float(blend_weight), 6),
            coverage_score=round(float(coverage_score), 6),
            n_train=min_n,
            model_version=self.version,
            model_quality=round(float(quality_raw), 6),
            support_score=round(float(support_score), 6),
            out_of_sample_flag=bool(out_of_sample_flag),
            min_oos_r2=round(float(min_oos_r2), 6) if min_oos_r2 is not None else None,
            min_treated_rows=int(min_treated_rows),
            min_control_rows=int(min_control_rows),
            selected_model_keys=dedup_keys,
            selected_models_by_objective=selected_models_by_objective,
            gate_reason="|".join(gate_reasons) if gate_reasons else "selected",
            action_status=str(policy.status or "unconfigured"),
            objective_allowlist=list(policy.objective_allowlist),
            strict_gate_primary_objectives=list(policy.strict_gate_primary_objectives),
            model_artifact_path_override=str(policy.model_artifact_path_override or ""),
            max_blend_weight=policy.max_blend_weight,
            future_action_alias=str(policy.future_action_alias or ""),
            future_action_aliases=list(policy.future_action_aliases),
            quality_floor_override=policy.quality_floor_override,
            support_floor_override=policy.support_floor_override,
            min_train_rows_override=policy.min_train_rows_override,
            min_oos_r2_override=policy.min_oos_r2_override,
            min_treated_rows_override=policy.min_treated_rows_override,
            min_control_rows_override=policy.min_control_rows_override,
        )

    def _routing_alias_pairs(
        self,
        *,
        action_id: str,
        action_type: str,
        action_subtype: str,
        action_alias: str,
        subtype_alias: str,
        future_action_aliases: Sequence[str] | None = None,
    ) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []

        def _add_pair(alias: str, subtype: str) -> None:
            pair = (str(alias or "").strip(), str(subtype or "").strip())
            if pair == ("", ""):
                return
            if pair not in pairs:
                pairs.append(pair)

        _add_pair(action_alias, subtype_alias)

        action_id_text = str(action_id or "").strip()
        action_type_text = str(action_type or "").strip()
        action_subtype_text = str(action_subtype or "").strip()
        if action_id_text and "." in action_id_text:
            canonical_type, canonical_subtype = action_id_text.split(".", 1)
            _add_pair(canonical_type, canonical_subtype)
            _add_pair(canonical_subtype, canonical_subtype)

        if action_type_text and action_subtype_text:
            _add_pair(action_type_text, action_subtype_text)
            _add_pair(action_subtype_text, action_subtype_text)

        for raw_alias in list(future_action_aliases or []):
            future_alias_text = str(raw_alias or "").strip()
            if not future_alias_text:
                continue
            if "." in future_alias_text:
                future_type, future_subtype = future_alias_text.split(".", 1)
                _add_pair(future_type, future_subtype)
                _add_pair(future_subtype, future_subtype)
            else:
                if action_type_text:
                    _add_pair(action_type_text, future_alias_text)
                _add_pair(future_alias_text, future_alias_text)

        return pairs

    def _candidate_model_keys(self, action_alias: str, subtype_alias: str) -> list[str]:
        # Support both action-type keyed cells (e.g. "dividend::dividend_increase")
        # and subtype-keyed cells (e.g. "dividend_increase::dividend_increase")
        # because training artifacts may be emitted in either style.
        alias_prefixes = [str(action_alias or "").strip()]
        st = str(subtype_alias or "").strip()
        if st:
            alias_prefixes.append(st)

        keys: list[str] = []
        for prefix in alias_prefixes:
            if not prefix:
                continue
            if st:
                keys.append(f"{prefix}::{st}")
            keys.append(f"{prefix}::all")
            keys.append(prefix)
        keys.append("__global__")

        # Keep order, remove dupes.
        out: list[str] = []
        for k in keys:
            if k not in out:
                out.append(k)
        return out

    def _candidate_model_keys_multi(self, alias_pairs: Sequence[Tuple[str, str]]) -> list[str]:
        out: list[str] = []
        for action_alias, subtype_alias in list(alias_pairs or []):
            for key in self._candidate_model_keys(action_alias=action_alias, subtype_alias=subtype_alias):
                if key not in out:
                    out.append(key)
        if "__global__" not in out:
            out.append("__global__")
        return out

    def _select_model(
        self,
        dr_models: Dict[str, Any],
        models: Dict[str, Any],
        keys: list[str],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        for k in keys:
            m = dr_models.get(k)
            if isinstance(m, dict) and bool(m.get("enabled", True)):
                return m, k
        for k in keys:
            m = models.get(k)
            if isinstance(m, dict) and bool(m.get("enabled", True)):
                return m, k
        return None, ""

    def _predict_objective(self, model: Dict[str, Any], x_std: Dict[str, float], selected_key: str = "") -> Optional[Dict[str, float]]:
        model_family = str(model.get("model_family", "linear")).strip().lower()
        if model_family == "hgb":
            bundle_key = str(model.get("bundle_key") or "")
            if not bundle_key:
                bundle_key = selected_key
            predictor = self._load_bundle_model(bundle_key)
            if predictor is None:
                return None
            vec = [float(x_std.get(f, 0.0)) for f in self.feature_order]
            mu_fallback = self._predict_hgb_fallback(predictor, vec)
            if mu_fallback is not None:
                mu = float(mu_fallback)
            else:
                try:
                    mu = float(predictor.predict([vec])[0])
                except Exception:
                    return None
            sigma = max(1e-6, float(_to_float(model.get("residual_std"), 0.15) or 0.15))
            out = {q: round(float(mu + z * sigma), 6) for q, z in _Z.items()}
            ordered = sorted([out["p10"], out["p25"], out["median"], out["p75"], out["p90"]])
            return {
                "p10": ordered[0],
                "p25": ordered[1],
                "median": ordered[2],
                "p75": ordered[3],
                "p90": ordered[4],
            }

        intercept = _to_float(model.get("intercept"))
        if intercept is None:
            return None
        coeffs = dict(model.get("coefficients", {}) or {})
        sigma = max(1e-6, float(_to_float(model.get("residual_std"), 0.15) or 0.15))

        mu = intercept
        for feature_name, beta_raw in coeffs.items():
            beta = _to_float(beta_raw, 0.0) or 0.0
            mu += beta * float(x_std.get(feature_name, 0.0))

        out: Dict[str, float] = {}
        for q, z in _Z.items():
            out[q] = round(float(mu + z * sigma), 6)
        # enforce monotone quantiles
        ordered = sorted([out["p10"], out["p25"], out["median"], out["p75"], out["p90"]])
        return {
            "p10": ordered[0],
            "p25": ordered[1],
            "median": ordered[2],
            "p75": ordered[3],
            "p90": ordered[4],
        }

    def _predict_hgb_fallback(self, predictor: Any, vec: Sequence[float]) -> Optional[float]:
        trees = getattr(predictor, "_predictors", None)
        if trees is None:
            return None
        baseline = _scalar_float(getattr(predictor, "_baseline_prediction", 0.0))
        if baseline is None:
            baseline = 0.0
        learning_rate = float(_to_float(getattr(predictor, "learning_rate", 1.0), 1.0) or 1.0)
        total = float(baseline)
        for stage in list(trees or []):
            for tree in list(stage or []):
                value = self._predict_hgb_tree(tree, vec)
                if value is None:
                    return None
                total += learning_rate * float(value)
        return float(total)

    def _predict_hgb_tree(self, tree: Any, vec: Sequence[float]) -> Optional[float]:
        nodes = getattr(tree, "nodes", None)
        if nodes is None:
            return None
        idx = 0
        max_steps = max(1, len(nodes) + 1)
        for _ in range(max_steps):
            node = nodes[idx]
            if bool(node["is_leaf"]):
                return float(node["value"])
            if bool(node["is_categorical"]):
                return None
            feature_idx = int(node["feature_idx"])
            if feature_idx < 0 or feature_idx >= len(vec):
                return None
            value = float(vec[feature_idx])
            if math.isnan(value):
                go_left = bool(node["missing_go_to_left"])
            else:
                go_left = value <= float(node["num_threshold"])
            idx = int(node["left"] if go_left else node["right"])
            if idx < 0 or idx >= len(nodes):
                return None
        return None

    def _load_bundle_model(self, bundle_key: str) -> Optional[Any]:
        if not bundle_key:
            return None
        if self._bundle_cache is None:
            bundle_path_raw = str(self.payload.get("model_bundle_path", "")).strip()
            if not bundle_path_raw:
                self._bundle_cache = {}
            else:
                bundle_path = Path(bundle_path_raw)
                if not bundle_path.is_absolute() and self.artifact_path is not None:
                    bundle_path = self.artifact_path.parent / bundle_path
                try:
                    with open(bundle_path, "rb") as fh:
                        loaded = pickle.load(fh)
                    self._bundle_cache = loaded if isinstance(loaded, dict) else {}
                except Exception:
                    try:
                        with open(bundle_path, "rb") as fh:
                            loaded = _BundleUnpickler(fh).load()
                        self._bundle_cache = loaded if isinstance(loaded, dict) else {}
                    except Exception:
                        self._bundle_cache = {}
        return self._bundle_cache.get(bundle_key) if isinstance(self._bundle_cache, dict) else None

    def _standardized_feature_vector(
        self,
        params: Dict[str, Any],
        features: Dict[str, Any],
        regime: Dict[str, Any],
        bundle: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        if isinstance(features, dict) and isinstance(features.get("features"), dict):
            features = dict(features.get("features", {}) or {})
        feature_source: Dict[str, Any] = {}
        if isinstance(features, dict):
            feature_source.update(features)
        if isinstance(bundle, dict):
            feature_source.update(dict(bundle.get("canonical", {}) or {}))
        for canonical_key, legacy_key in INFERENCE_SOURCE_KEYS.items():
            value = self._feature_input(
                bundle,
                features,
                canonical_key=canonical_key,
                legacy_key=legacy_key,
                default=None,
            )
            if value is not None:
                feature_source[canonical_key] = value

        contract_raw = build_contract_feature_map(
            feature_source,
            params=params,
            regime=regime,
        )

        x_std: Dict[str, float] = {}
        for name in self.feature_order:
            stats = dict(self.feature_stats.get(name, {}) or {})
            mean = _to_float(stats.get("mean"), 0.0) or 0.0
            std = max(1e-9, float(_to_float(stats.get("std"), 1.0) or 1.0))
            median = _to_float(stats.get("median"), mean) or mean
            canonical_name = canonicalize_feature_name(name)
            val = _to_float(
                self._normalize_feature_value(
                    name,
                    contract_raw.get(canonical_name),
                ),
                median,
            )
            if val is None:
                val = median
            x_std[name] = float((val - mean) / std)
        return x_std

    def _normalize_feature_value(self, feature_name: str, value: Any) -> Optional[float]:
        v = _to_float(value)
        if v is None:
            return None

        names = {str(feature_name), canonicalize_feature_name(str(feature_name))}
        # Unit harmonization (must mirror training-time rules).
        if names & self._usd_millions_features and abs(v) >= 1e7:
            v = v / 1e6
        if names & self._rate_percent_features and abs(v) <= 1.0:
            v = v * 100.0
        if names & self._oas_percent_features and abs(v) >= 50.0:
            v = v / 100.0

        # Value transform for heavy-tailed financial features.
        if names & self._signed_log1p_features:
            v = math.copysign(math.log1p(abs(v)), v)
        return float(v)

    def _resolve_causal_inputs(
        self,
        features: Dict[str, Any],
        *,
        action_id: str,
        action_type: str,
        regime: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        if not isinstance(features, dict):
            return {}, None

        if isinstance(features.get("features"), dict):
            snapshot = dict(features)
            bundle = build_model_feature_bundle(
                snapshot,
                action_id=action_id,
                action_type=action_type,
                regime=regime,
            )
            return (
                feature_view_from_snapshot(
                    snapshot,
                    view_name="causal",
                    action_id=action_id,
                    action_type=action_type,
                    regime=regime,
                ),
                bundle,
            )

        if "views" in features and "canonical" in features:
            bundle = dict(features)
            causal_view = dict((bundle.get("views", {}) or {}).get("causal", {}) or {})
            return (causal_view or dict((bundle.get("raw_features", {}) or {})), bundle)

        adapted_snapshot, _ = adapt_snapshot(
            {"features": features},
            action_family=action_type,
            action_id=action_id,
        )
        snapshot = {
            "features": adapted_snapshot.get("features", {}) if isinstance(adapted_snapshot, dict) else features,
            "regime": dict(regime or {}),
        }
        bundle = build_model_feature_bundle(
            snapshot,
            action_id=action_id,
            action_type=action_type,
            regime=regime,
        )
        return dict((bundle.get("views", {}) or {}).get("causal", {}) or {}), bundle

    def _feature_input(
        self,
        bundle: Optional[Dict[str, Any]],
        features: Dict[str, Any],
        *,
        canonical_key: str,
        legacy_key: str,
        default: Any = None,
    ) -> Any:
        if isinstance(bundle, dict):
            value = get_bundle_value(bundle, canonical_key, default=None)
            if value is not None:
                return value
        return _extract_feature(features, legacy_key, default=default)


def _default_model_path() -> Path:
    env = str(os.environ.get("CAUSAL_IMPACT_MODEL_PATH", "")).strip()
    if env:
        return Path(env)
    return DEFAULT_CAUSAL_IMPACT_MODEL_ARTIFACT


@lru_cache(maxsize=2)
def load_default_causal_impact_model(path_str: Optional[str] = None) -> Optional[CausalImpactModel]:
    path = Path(path_str) if path_str else _default_model_path()
    if not path.exists():
        return None
    try:
        return CausalImpactModel.from_path(path)
    except Exception:
        return None


__all__ = [
    "CausalActionPolicy",
    "CausalImpactModel",
    "CausalPrediction",
    "DEFAULT_CAUSAL_ROUTING_CONFIG_PATH",
    "DEFAULT_CAUSAL_IMPACT_MODEL_ARTIFACT",
    "action_id_to_outcomes_action_type",
    "action_subtype_to_outcomes_subtype",
    "get_causal_action_policy",
    "load_causal_routing_config",
    "load_default_causal_impact_model",
]
