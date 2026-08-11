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


