"""Mechanism Brain: evaluates action candidates without ranking.

Transforms ActionCandidateDraft-like records into ActionCandidate evaluations with:
- feasibility gating
- mechanism activation
- probabilistic counterfactual impact
- structural sanity flags
- risks and assumptions
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .causal_impact_model import (
    CausalImpactModel,
    get_causal_action_policy,
    load_default_causal_impact_model,
)
from .model_feature_bundle import feature_view_from_snapshot
from .runtime_feature_adapter import resolve_feature_value


_HARD_ACTIONS_HIGH_COMPLEXITY = {
    "mna.go_private_lbo",
    "mna.transformational_acquisition",
    "portfolio.spin_off",
    "portfolio.carve_out_ipo",
    "restructuring.chapter_pathway",
    "restructuring.out_of_court_restructuring",
}

_CASH_CONSUMING_ACTION_PREFIXES = (
    "capital_return.",
    "mna.",
)

_DEBT_REQUIRING_ACTION_IDS = {
    "capital_structure.new_debt_issuance",
    "capital_structure.refinancing",
    "capital_structure.tender_offer_debt",
    "capital_structure.exchange_offer",
    "capital_structure.liability_management_exercise",
    "capital_structure.revolver_draw_or_resize",
    "capital_structure.convertible_issuance",
}

_EQUITY_REQUIRING_ACTION_IDS = {
    "capital_structure.equity_issuance",
    "capital_structure.convertible_issuance",
    "capital_structure.preferred_issuance",
    "portfolio.carve_out_ipo",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


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


def _parse_action_id_tokens(raw: str) -> set[str]:
    tokens: set[str] = set()
    text = str(raw or "").strip()
    if not text:
        return tokens
    for part in re.split(r"[,\s]+", text):
        tok = str(part or "").strip().lower()
        if tok:
            tokens.add(tok)
    return tokens


def _load_action_id_tokens_from_file(path_value: str) -> set[str]:
    path = Path(str(path_value or "").strip())
    if not str(path):
        return set()
    if not path.exists() or not path.is_file():
        return set()
    try:
        body = path.read_text()
    except Exception:
        return set()
    out: set[str] = set()
    for line in body.splitlines():
        line_clean = str(line).split("#", 1)[0].strip()
        if not line_clean:
            continue
        out.update(_parse_action_id_tokens(line_clean))
    return out


@dataclass
class Signal:
    feature_name: str
    value: Any
    threshold: Any
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Blocker:
    blocker_type: str
    severity: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Remediation:
    action_required: str
    expected_effect: str
    estimated_delay_days: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeasibilityResult:
    feasibility_status: str
    pass_probability: float
    blockers: List[Blocker] = field(default_factory=list)
    remediation_steps: List[Remediation] = field(default_factory=list)
    lead_time_prior_days: int = 0
    gating_signals: List[Signal] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasibility_status": self.feasibility_status,
            "pass_probability": float(self.pass_probability),
            "blockers": [b.to_dict() for b in self.blockers],
            "remediation_steps": [r.to_dict() for r in self.remediation_steps],
            "lead_time_prior_days": int(self.lead_time_prior_days),
            "gating_signals": [s.to_dict() for s in self.gating_signals],
        }


@dataclass
class Mechanism:
    mechanism_id: str
    channel_type: str
    activation_strength: float
    positive_signals: List[Signal] = field(default_factory=list)
    negative_signals: List[Signal] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "channel_type": self.channel_type,
            "activation_strength": float(self.activation_strength),
            "positive_signals": [s.to_dict() for s in self.positive_signals],
            "negative_signals": [s.to_dict() for s in self.negative_signals],
        }


@dataclass
class Interaction:
    feature_combination: str
    direction: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MechanismActivation:
    mechanisms: List[Mechanism] = field(default_factory=list)
    key_interactions: List[Interaction] = field(default_factory=list)
    narrative_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanisms": [m.to_dict() for m in self.mechanisms],
            "key_interactions": [k.to_dict() for k in self.key_interactions],
            "narrative_explanation": self.narrative_explanation,
        }


@dataclass
class Distribution:
    median: float
    p10: float
    p25: float
    p75: float
    p90: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegimeImpact:
    regime_condition: str
    effect_shift: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Driver:
    driver_name: str
    contribution: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImpactDistribution:
    objectives: Dict[str, Distribution]
    regime_sensitivity: List[RegimeImpact]
    key_drivers: List[Driver]
    uncertainty_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objectives": {k: v.to_dict() for k, v in self.objectives.items()},
            "regime_sensitivity": [r.to_dict() for r in self.regime_sensitivity],
            "key_drivers": [d.to_dict() for d in self.key_drivers],
            "uncertainty_score": float(self.uncertainty_score),
        }


@dataclass
class SanityCheck:
    check_type: str
    status: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskItem:
    risk_type: str
    probability: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_type": self.risk_type,
            "probability": float(self.probability),
            "explanation": self.explanation,
        }


@dataclass
class Assumption:
    assumption_type: str
    description: str
    sensitivity: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionCandidate:
    candidate_id: str
    run_id: str
    action_id: str
    action_type: str
    action_subtype: str
    parameters: Dict[str, Any]
    feasibility: FeasibilityResult
    mechanism_activation: MechanismActivation
    impact_distribution: ImpactDistribution
    structural_sanity_flags: List[SanityCheck]
    risks: List[RiskItem]
    assumptions: List[Assumption]
    evaluation_confidence: float
    created_at: str
    evaluation_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "action_id": self.action_id,
            "action_type": self.action_type,
            "action_subtype": self.action_subtype,
            "parameters": self.parameters,
            "feasibility": self.feasibility.to_dict(),
            "mechanism_activation": self.mechanism_activation.to_dict(),
            "impact_distribution": self.impact_distribution.to_dict(),
            "structural_sanity_flags": [x.to_dict() for x in self.structural_sanity_flags],
            "risks": [x.to_dict() for x in self.risks],
            "assumptions": [x.to_dict() for x in self.assumptions],
            "evaluation_confidence": float(self.evaluation_confidence),
            "created_at": self.created_at,
            "evaluation_profile": dict(self.evaluation_profile or {}),
        }


class MechanismBrain:
    """Deterministic evaluator that transforms draft candidates into evaluated candidates."""

    def __init__(
        self,
        action_registry: Any,
        max_leverage_threshold_default: float = 4.5,
        maturity_wall_threshold: float = 0.25,
        ig_leverage_boundary: float = 3.5,
        coverage_safety_band: float = 2.5,
        expectations_coverage_min: float = 5.0,
        expectations_revision_negative_floor: float = -0.05,
        expectations_revision_positive_floor: float = 0.02,
        ownership_crowding_support_min: float = 0.55,
        causal_model: Optional[CausalImpactModel] = None,
        causal_mode: Optional[str] = None,
        causal_quality_floor: float = 0.08,
        causal_support_floor: float = 0.35,
        causal_min_train_rows: int = 1000,
        causal_min_oos_r2: float = 0.0,
        causal_min_treated_rows: int = 1000,
        causal_min_control_rows: int = 5000,
    ) -> None:
        self.registry = action_registry
        self.max_leverage_threshold_default = max_leverage_threshold_default
        self.maturity_wall_threshold = maturity_wall_threshold
        self.ig_leverage_boundary = ig_leverage_boundary
        self.coverage_safety_band = coverage_safety_band
        self.expectations_coverage_min = expectations_coverage_min
        self.expectations_revision_negative_floor = expectations_revision_negative_floor
        self.expectations_revision_positive_floor = expectations_revision_positive_floor
        self.ownership_crowding_support_min = ownership_crowding_support_min
        self.causal_model = causal_model if causal_model is not None else load_default_causal_impact_model()
        raw_mode = str(causal_mode or os.environ.get("CAUSAL_IMPACT_MODE", "blend")).strip().lower()
        self.causal_mode = raw_mode if raw_mode in {"blend", "standalone"} else "blend"
        self.causal_quality_floor = (
            _to_float(os.environ.get("CAUSAL_STRICT_QUALITY_FLOOR"), causal_quality_floor) or causal_quality_floor
        )
        self.causal_support_floor = (
            _to_float(os.environ.get("CAUSAL_STRICT_SUPPORT_FLOOR"), causal_support_floor) or causal_support_floor
        )
        min_train_env = _to_float(os.environ.get("CAUSAL_STRICT_MIN_TRAIN_ROWS"))
        min_train_raw = min_train_env if min_train_env is not None else float(causal_min_train_rows)
        self.causal_min_train_rows = int(max(0.0, min_train_raw))
        min_oos_env = _to_float(os.environ.get("CAUSAL_STRICT_MIN_OOS_R2"))
        min_oos_raw = min_oos_env if min_oos_env is not None else float(causal_min_oos_r2)
        self.causal_min_oos_r2 = float(min_oos_raw)
        min_treated_env = _to_float(os.environ.get("CAUSAL_STRICT_MIN_TREATED_ROWS"))
        min_treated_raw = min_treated_env if min_treated_env is not None else float(causal_min_treated_rows)
        self.causal_min_treated_rows = int(max(0.0, min_treated_raw))
        min_control_env = _to_float(os.environ.get("CAUSAL_STRICT_MIN_CONTROL_ROWS"))
        min_control_raw = min_control_env if min_control_env is not None else float(causal_min_control_rows)
        self.causal_min_control_rows = int(max(0.0, min_control_raw))
        blocklist_text = ",".join(
            x
            for x in (
                str(os.environ.get("CAUSAL_ACTION_BLOCKLIST", "") or "").strip(),
                str(os.environ.get("CAUSAL_ACTION_DENYLIST", "") or "").strip(),
            )
            if x
        )
        self.causal_action_blocklist: set[str] = _parse_action_id_tokens(blocklist_text)
        self.causal_action_blocklist.update(
            _load_action_id_tokens_from_file(str(os.environ.get("CAUSAL_ACTION_BLOCKLIST_PATH", "") or "").strip())
        )
        self.last_evaluation_set_profile: Dict[str, Any] = {}

    def _expectations_context(self, features: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
        coverage = _to_float(_extract_feature(features, "expectations.analyst_coverage_count"))
        revision_signal = _to_float(_extract_feature(features, "expectations.revision_signal"))
        if coverage is not None and coverage < 0:
            coverage = None
        return coverage, revision_signal

    def _capital_return_expectations_shift_by_objective(
        self, action_id: str, features: Dict[str, Any]
    ) -> tuple[Dict[str, float], Optional[Driver]]:
        out = {
            "value_creation": 0.0,
            "risk_reduction": 0.0,
            "growth": 0.0,
            "rating_preservation": 0.0,
            "optionality": 0.0,
        }
        if not action_id.startswith("capital_return."):
            return out, None

        coverage, revision_signal = self._expectations_context(features)
        if coverage is None or coverage < self.expectations_coverage_min or revision_signal is None:
            return out, None

        normalized = _clip(revision_signal / 0.10, -1.0, 1.0)
        if action_id in {"capital_return.open_market_buyback", "capital_return.accelerated_share_repurchase"}:
            value_shift = 0.03 * normalized
            optionality_shift = 0.01 * normalized
        elif action_id in {"capital_return.dividend_increase", "capital_return.dividend_initiate"}:
            value_shift = 0.025 * normalized
            optionality_shift = 0.008 * normalized
        else:
            value_shift = 0.02 * normalized
            optionality_shift = 0.006 * normalized

        out["value_creation"] = value_shift
        out["optionality"] = optionality_shift
        driver = Driver(
            driver_name="expectations_revision_signal",
            contribution=round(revision_signal, 6),
            explanation="Analyst revisions tilt capital-return reception toward support when improving and caution when deteriorating.",
        )
        return out, driver

    def _capital_return_positioning_shift_by_objective(
        self, action_id: str, features: Dict[str, Any]
    ) -> tuple[Dict[str, float], Optional[Driver]]:
        out = {
            "value_creation": 0.0,
            "risk_reduction": 0.0,
            "growth": 0.0,
            "rating_preservation": 0.0,
            "optionality": 0.0,
        }
        if not action_id.startswith("capital_return."):
            return out, None

        crowding_signal = _to_float(_extract_feature(features, "ownership_governance.crowding_signal"))
        activist_present = bool(_extract_feature(features, "ownership_governance.activist_presence_flag"))
        institutional_pct = _to_float(_extract_feature(features, "ownership_governance.institutional_pct"))
        top5_holder_pct = _to_float(_extract_feature(features, "ownership_governance.top5_holder_pct"))

        if activist_present:
            positioning_strength = 1.0
        else:
            components = []
            if crowding_signal is not None:
                components.append(float(crowding_signal))
            if institutional_pct is not None:
                components.append(_clip((institutional_pct - 0.50) / 0.40, 0.0, 1.0))
            if top5_holder_pct is not None:
                components.append(_clip((top5_holder_pct - 0.35) / 0.35, 0.0, 1.0))
            positioning_strength = float(sum(components) / len(components)) if components else 0.0

        if positioning_strength < self.ownership_crowding_support_min:
            return out, None

        if action_id in {"capital_return.open_market_buyback", "capital_return.accelerated_share_repurchase"}:
            out["value_creation"] = 0.02 * positioning_strength
            out["optionality"] = 0.008 * positioning_strength
        elif action_id in {"capital_return.dividend_increase", "capital_return.special_dividend", "capital_return.dividend_initiate"}:
            out["value_creation"] = 0.012 * positioning_strength
            out["optionality"] = 0.004 * positioning_strength
        else:
            out["value_creation"] = 0.008 * positioning_strength

        driver = Driver(
            driver_name="ownership_positioning_signal",
            contribution=round(positioning_strength, 6),
            explanation="Concentrated or activist holder bases can amplify market demand for visible capital-allocation actions.",
        )
        return out, driver

    def evaluate_candidate_set(
        self,
        run: Any,
        state_snapshot: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
        peer_set: Optional[Dict[str, Any]] = None,
        extracted_facts: Optional[List[Dict[str, Any]]] = None,
        event_store: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ActionCandidate]:
        def _pct(values: Sequence[float], q: float) -> float:
            if not values:
                return 0.0
            arr = sorted(float(v) for v in values)
            idx = min(len(arr) - 1, max(0, int(round(q * (len(arr) - 1)))))
            return float(arr[idx])

        total_t0 = time.perf_counter()
        out: List[ActionCandidate] = []
        loop_latencies: List[float] = []
        for row in candidates:
            row_t0 = time.perf_counter()
            out.append(
                self.evaluate_candidate(
                    run=run,
                    state_snapshot=state_snapshot,
                    candidate=row,
                    peer_set=peer_set,
                    extracted_facts=extracted_facts,
                    event_store=event_store,
                )
            )
            loop_latencies.append(time.perf_counter() - row_t0)
        total_seconds = time.perf_counter() - total_t0
        per_candidate_totals = [
            float(getattr(item, "evaluation_profile", {}).get("total_seconds", 0.0) or 0.0)
            for item in out
        ]
        self.last_evaluation_set_profile = {
            "count": int(len(out)),
            "total_seconds": round(total_seconds, 6),
            "loop_p50_seconds": round(_pct(loop_latencies, 0.50), 6),
            "loop_p95_seconds": round(_pct(loop_latencies, 0.95), 6),
            "loop_sum_seconds": round(sum(loop_latencies), 6),
            "candidate_profile_total_sum_seconds": round(sum(per_candidate_totals), 6),
            "candidate_profile_total_p50_seconds": round(_pct(per_candidate_totals, 0.50), 6),
            "candidate_profile_total_p95_seconds": round(_pct(per_candidate_totals, 0.95), 6),
            "unattributed_seconds": round(max(0.0, total_seconds - sum(per_candidate_totals)), 6),
        }
        return out

    def evaluate_candidate(
        self,
        run: Any,
        state_snapshot: Dict[str, Any],
        candidate: Dict[str, Any],
        peer_set: Optional[Dict[str, Any]] = None,
        extracted_facts: Optional[List[Dict[str, Any]]] = None,
        event_store: Optional[List[Dict[str, Any]]] = None,
    ) -> ActionCandidate:
        wall_t0 = time.perf_counter()
        action_id = str(candidate.get("action_id", ""))
        action_type = str(candidate.get("action_type", action_id.split(".")[0] if "." in action_id else ""))
        action_subtype = str(candidate.get("action_subtype", action_id.split(".", 1)[1] if "." in action_id else ""))
        features = feature_view_from_snapshot(
            state_snapshot,
            view_name="mechanism",
            action_id=action_id,
            action_type=action_type,
            regime=(state_snapshot.get("regime", {}) if isinstance(state_snapshot, dict) else {}),
        )
        regime = state_snapshot.get("regime", {}) if isinstance(state_snapshot, dict) else {}
        params = dict(candidate.get("parameters", candidate.get("params", {})) or {})
        schema_t0 = time.perf_counter()
        schema = self.registry.get_action(action_id) if hasattr(self.registry, "get_action") else {}
        schema_seconds = time.perf_counter() - schema_t0
        if not isinstance(schema, dict):
            schema = {}

        feasibility_t0 = time.perf_counter()
        feasibility = self._evaluate_feasibility(run, action_id, action_type, params, schema, features, regime)
        feasibility_seconds = time.perf_counter() - feasibility_t0
        mechanism_activation_t0 = time.perf_counter()
        mechanism_activation = self._activate_mechanisms(action_id, action_type, params, schema, features, regime, feasibility)
        mechanism_activation_seconds = time.perf_counter() - mechanism_activation_t0
        impact_distribution_t0 = time.perf_counter()
        impact_distribution = self._estimate_impact_distribution(
            action_id=action_id,
            action_type=action_type,
            action_subtype=action_subtype,
            params=params,
            features=features,
            state_snapshot=state_snapshot,
            regime=regime,
            feasibility=feasibility,
            mechanism_activation=mechanism_activation,
        )
        impact_distribution_seconds = time.perf_counter() - impact_distribution_t0
        structural_checks_t0 = time.perf_counter()
        sanity = self._run_structural_checks(
            run=run,
            action_id=action_id,
            action_type=action_type,
            params=params,
            features=features,
            mechanism_activation=mechanism_activation,
            feasibility=feasibility,
        )
        structural_checks_seconds = time.perf_counter() - structural_checks_t0
        risk_identification_t0 = time.perf_counter()
        risks = self._identify_risks(
            action_id=action_id,
            action_type=action_type,
            feasibility=feasibility,
            mechanism_activation=mechanism_activation,
            regime=regime,
            schema=schema,
        )
        risk_identification_seconds = time.perf_counter() - risk_identification_t0
        assumptions_t0 = time.perf_counter()
        assumptions = self._collect_assumptions(
            action_id=action_id,
            action_type=action_type,
            params=params,
            feasibility=feasibility,
            regime=regime,
        )
        assumptions_seconds = time.perf_counter() - assumptions_t0
        evaluation_confidence_t0 = time.perf_counter()
        eval_conf = self._evaluation_confidence(
            candidate=candidate,
            features=features,
            mechanism_activation=mechanism_activation,
            impact_distribution=impact_distribution,
            feasibility=feasibility,
        )
        evaluation_confidence_seconds = time.perf_counter() - evaluation_confidence_t0

        candidate_id = str(candidate.get("candidate_id") or "")
        candidate_id_seconds = 0.0
        if not candidate_id:
            candidate_id_t0 = time.perf_counter()
            signature = str(candidate.get("candidate_signature", action_id))
            namespace = uuid.UUID(str(run.run_id)) if self._is_uuid(str(run.run_id)) else uuid.NAMESPACE_URL
            candidate_id = str(uuid.uuid5(namespace, signature))
            candidate_id_seconds = time.perf_counter() - candidate_id_t0

        total_seconds = time.perf_counter() - wall_t0

        return ActionCandidate(
            candidate_id=candidate_id,
            run_id=str(run.run_id),
            action_id=action_id,
            action_type=action_type,
            action_subtype=action_subtype,
            parameters=params,
            feasibility=feasibility,
            mechanism_activation=mechanism_activation,
            impact_distribution=impact_distribution,
            structural_sanity_flags=sanity,
            risks=risks,
            assumptions=assumptions,
            evaluation_confidence=eval_conf,
            created_at=str(candidate.get("created_at") or getattr(run, "created_at", _now_iso())),
            evaluation_profile={
                "schema_lookup_seconds": round(schema_seconds, 6),
                "feasibility_seconds": round(feasibility_seconds, 6),
                "mechanism_activation_seconds": round(mechanism_activation_seconds, 6),
                "impact_distribution_seconds": round(impact_distribution_seconds, 6),
                "structural_checks_seconds": round(structural_checks_seconds, 6),
                "risk_identification_seconds": round(risk_identification_seconds, 6),
                "assumptions_seconds": round(assumptions_seconds, 6),
                "evaluation_confidence_seconds": round(evaluation_confidence_seconds, 6),
                "candidate_id_seconds": round(candidate_id_seconds, 6),
                "total_seconds": round(total_seconds, 6),
            },
        )

    def _evaluate_feasibility(
        self,
        run: Any,
        action_id: str,
        action_type: str,
        params: Dict[str, Any],
        schema: Dict[str, Any],
        features: Dict[str, Any],
        regime: Dict[str, Any],
    ) -> FeasibilityResult:
        blockers: List[Blocker] = []
        remediations: List[Remediation] = []
        signals: List[Signal] = []

        runway = _to_float(_extract_feature(features, "liquidity.runway_months"))
        available_liq = _to_float(_extract_feature(features, "liquidity.available_for_actions"), 0.0) or 0.0
        mcap = _to_float(_extract_feature(features, "market.market_cap"), 0.0) or 0.0
        total_debt = _to_float(_extract_feature(features, "capital_structure.total_debt"))
        cash = _to_float(_extract_feature(features, "liquidity.cash"), 0.0) or 0.0
        net_debt = _to_float(_extract_feature(features, "capital_structure.net_debt"))
        ebitda = _to_float(_extract_feature(features, "operating.ebitda_ttm"))
        if ebitda is None:
            ebitda = _to_float(_extract_feature(features, "capital_structure.ebitda_ttm"))
        if net_debt is None and total_debt is not None:
            net_debt = total_debt - cash
        if net_debt is None and ebitda is not None:
            lev = _to_float(_extract_feature(features, "capital_structure.net_leverage"))
            if lev is not None:
                net_debt = lev * ebitda
        net_debt = net_debt if net_debt is not None else 0.0

        interest_coverage = _to_float(_extract_feature(features, "capital_structure.interest_coverage"))
        maturity_ratio = _to_float(_extract_feature(features, "capital_structure.maturity_wall_ratio_24m"), 0.0) or 0.0
        credit_regime = str(regime.get("credit_regime", "neutral"))
        vol_regime = str(regime.get("vol_regime", "normal"))

        action_size_usd = self._estimate_action_size_usd(action_id, params, mcap)
        cash_usage = self._estimate_cash_usage_usd(action_id, params, action_size_usd)
        debt_delta = self._estimate_debt_delta_usd(action_id, params, action_size_usd)
        equity_raise = self._estimate_equity_raise_usd(action_id, params, action_size_usd)
        proforma_net_debt = net_debt + debt_delta - equity_raise

        proforma_leverage: Optional[float] = None
        if ebitda is not None and ebitda > 0:
            proforma_leverage = proforma_net_debt / ebitda

        proforma_interest_coverage = interest_coverage
        if interest_coverage is not None and abs(net_debt) > 1e-9:
            debt_ratio = debt_delta / max(abs(net_debt), 1.0)
            if debt_delta >= 0:
                proforma_interest_coverage = interest_coverage / (1.0 + 0.6 * max(0.0, debt_ratio))
            else:
                proforma_interest_coverage = interest_coverage * (1.0 + 0.3 * abs(debt_ratio))

        runway_adj = runway
        if runway is not None and available_liq > 0 and cash_usage > 0:
            remaining_frac = _clip((available_liq - cash_usage) / available_liq, 0.0, 1.0)
            runway_adj = runway * remaining_frac

        dividend_continuity_liquidity_exception = self._allows_dividend_continuity_liquidity_exception(
            action_id=action_id,
            params=params,
            features=features,
            runway_adj=runway_adj,
        )

        if runway_adj is not None:
            signals.append(
                Signal(
                    feature_name="liquidity.runway_months_proforma",
                    value=round(float(runway_adj), 4),
                    threshold=">=12 preferred",
                    interpretation="Liquidity runway after action cash usage.",
                )
            )
            if dividend_continuity_liquidity_exception:
                annualized_cash_commitment = _to_float(params.get("annualized_cash_commitment_usd"), 0.0) or 0.0
                signals.append(
                    Signal(
                        feature_name="capital_return.incremental_quarterly_cash_commitment_usd",
                        value=round(float(annualized_cash_commitment / 4.0), 4),
                        threshold="<=25% of cash",
                        interpretation="Incremental quarterly payout burden for a continuity-style dividend step-up.",
                    )
                )
            if runway_adj < 6:
                if dividend_continuity_liquidity_exception:
                    blockers.append(
                        Blocker(
                            blocker_type="liquidity_shortfall",
                            severity="soft",
                            explanation="Baseline runway is tight, but the incremental dividend step is de minimis for an established payer with no near-term maturities.",
                        )
                    )
                else:
                    blockers.append(
                        Blocker(
                            blocker_type="liquidity_shortfall",
                            severity="hard",
                            explanation="Pro-forma liquidity runway falls below 6 months.",
                        )
                    )
            elif runway_adj < 12:
                blockers.append(
                    Blocker(
                        blocker_type="liquidity_shortfall",
                        severity="soft",
                        explanation="Pro-forma liquidity runway is between 6 and 12 months.",
                    )
                )

        max_lev = self._resolve_max_leverage_threshold(run)
        if proforma_leverage is not None:
            signals.append(
                Signal(
                    feature_name="capital_structure.proforma_leverage",
                    value=round(float(proforma_leverage), 4),
                    threshold=round(float(max_lev), 4),
                    interpretation="Pro-forma leverage vs threshold.",
                )
            )
            if proforma_leverage > max_lev:
                blockers.append(
                    Blocker(
                        blocker_type="leverage_breach",
                        severity="hard",
                        explanation=f"Pro-forma leverage {proforma_leverage:.2f}x exceeds limit {max_lev:.2f}x.",
                    )
                )
            elif proforma_leverage > 0.9 * max_lev:
                blockers.append(
                    Blocker(
                        blocker_type="leverage_breach",
                        severity="soft",
                        explanation=f"Pro-forma leverage {proforma_leverage:.2f}x is close to limit {max_lev:.2f}x.",
                    )
                )

        signals.append(
            Signal(
                feature_name="capital_structure.maturity_wall_ratio_24m",
                value=round(float(maturity_ratio), 4),
                threshold=self.maturity_wall_threshold,
                interpretation="Near-term maturity pressure indicator.",
            )
        )
        if cash_usage > 0 and maturity_ratio > self.maturity_wall_threshold:
            blockers.append(
                Blocker(
                    blocker_type="maturity_wall_conflict",
                    severity="hard" if maturity_ratio >= 0.40 else "soft",
                    explanation="Action consumes liquidity while near-term maturities are elevated.",
                )
            )

        rating_state = _extract_feature(features, "capital_structure.rating_state")
        starts_ig = self._is_investment_grade(rating_state)
        rating_risk = False
        if starts_ig and proforma_leverage is not None and proforma_leverage > self.ig_leverage_boundary:
            rating_risk = True
        if proforma_interest_coverage is not None and proforma_interest_coverage < self.coverage_safety_band:
            rating_risk = True
        if rating_risk:
            blockers.append(
                Blocker(
                    blocker_type="rating_downgrade_risk",
                    severity="soft",
                    explanation="Pro-forma metrics increase downgrade probability near/through IG safety bands.",
                )
            )

        debt_required = self._requires_debt(action_id, params)
        equity_required = self._requires_equity(action_id, params)
        if debt_required and credit_regime == "tight":
            blockers.append(
                Blocker(
                    blocker_type="market_access_closed",
                    severity="soft",
                    explanation="Debt-dependent action in tight credit regime.",
                )
            )
        if equity_required and vol_regime == "high":
            equity_window = _to_float(_extract_feature(features, "market.equity_window_proxy"), None)
            use_of_proceeds = str((params or {}).get("use_of_proceeds", "") or "").strip().lower()
            recap_equity_window_still_serviceable = bool(
                action_id == "capital_structure.equity_issuance"
                and use_of_proceeds == "deleveraging"
                and equity_window is not None
                and equity_window >= 0.25
            )
            if not recap_equity_window_still_serviceable:
                blockers.append(
                    Blocker(
                        blocker_type="market_access_closed",
                        severity="soft",
                        explanation="Equity-dependent action in high volatility regime.",
                    )
                )

        base_complexity = _to_float(_nested_get(schema, "execution_complexity_prior.base_complexity_score"), 3.0) or 3.0
        if action_id in _HARD_ACTIONS_HIGH_COMPLEXITY or base_complexity >= 4:
            blockers.append(
                Blocker(
                    blocker_type="operational_capacity_limit",
                    severity="soft",
                    explanation="High-complexity action likely needs extended execution bandwidth.",
                )
            )

        if proforma_interest_coverage is not None:
            signals.append(
                Signal(
                    feature_name="capital_structure.proforma_interest_coverage",
                    value=round(float(proforma_interest_coverage), 4),
                    threshold=self.coverage_safety_band,
                    interpretation="Coverage cushion after financing effects.",
                )
            )
        signals.append(
            Signal(
                feature_name="regime.credit_regime",
                value=credit_regime,
                threshold="tight",
                interpretation="Credit market access proxy.",
            )
        )
        signals.append(
            Signal(
                feature_name="regime.vol_regime",
                value=vol_regime,
                threshold="high",
                interpretation="Equity market access proxy.",
            )
        )

        for b in blockers:
            remediations.extend(self._remediation_for_blocker(b))

        status = "feasible"
        if any(b.severity == "hard" for b in blockers):
            status = "infeasible"
        elif blockers:
            status = "conditional"

        lead_time_days = int(_to_float(_nested_get(schema, "lead_time_prior.median_days"), 30.0) or 30.0)
        if any(b.blocker_type == "operational_capacity_limit" for b in blockers):
            lead_time_days += 30
        if debt_required and credit_regime == "tight":
            lead_time_days += 10
        if equity_required and vol_regime == "high":
            lead_time_days += 10

        pass_probability = 0.90
        if status == "conditional":
            pass_probability = 0.62
        elif status == "infeasible":
            pass_probability = 0.12
        pass_probability -= 0.20 * sum(1 for b in blockers if b.severity == "hard")
        pass_probability -= 0.08 * sum(1 for b in blockers if b.severity == "soft")
        if runway_adj is not None:
            if runway_adj > 24:
                pass_probability += 0.05
            elif runway_adj < 9:
                pass_probability -= 0.05
        pass_probability = _clip(pass_probability, 0.01, 0.99)

        return FeasibilityResult(
            feasibility_status=status,
            pass_probability=pass_probability,
            blockers=blockers,
            remediation_steps=remediations,
            lead_time_prior_days=lead_time_days,
            gating_signals=signals,
        )

    def _activate_mechanisms(
        self,
        action_id: str,
        action_type: str,
        params: Dict[str, Any],
        schema: Dict[str, Any],
        features: Dict[str, Any],
        regime: Dict[str, Any],
        feasibility: FeasibilityResult,
    ) -> MechanismActivation:
        channels = schema.get("mechanism_channels", []) if isinstance(schema, dict) else []
        mechanisms: List[Mechanism] = []

        for ch in channels:
            mech_id = str(ch.get("channel_id", ""))
            channel_type = str(ch.get("channel_type", ""))
            pos_exprs = list(ch.get("activation_signals", []) or [])
            neg_exprs = list(ch.get("negative_signals", []) or [])

            pos_signals: List[Signal] = []
            neg_signals: List[Signal] = []
            pos_hits = 0
            neg_hits = 0

            for expr in pos_exprs:
                ok, signal = self._evaluate_signal_expression(str(expr), features, regime)
                if ok and signal is not None:
                    pos_hits += 1
                    pos_signals.append(signal)
            for expr in neg_exprs:
                ok, signal = self._evaluate_signal_expression(str(expr), features, regime)
                if ok and signal is not None:
                    neg_hits += 1
                    neg_signals.append(signal)

            pos_ratio = pos_hits / max(1, len(pos_exprs))
            neg_ratio = neg_hits / max(1, len(neg_exprs)) if neg_exprs else 0.0
            strength = _clip(0.10 + 0.90 * (pos_ratio - 0.60 * neg_ratio), 0.0, 1.0)
            if feasibility.feasibility_status == "infeasible":
                strength *= 0.75

            mechanisms.append(
                Mechanism(
                    mechanism_id=mech_id,
                    channel_type=channel_type,
                    activation_strength=round(strength, 4),
                    positive_signals=pos_signals,
                    negative_signals=neg_signals,
                )
            )

        interactions = self._derive_interactions(action_id, action_type, params, features, regime)
        avg_strength = sum(m.activation_strength for m in mechanisms) / max(1, len(mechanisms))
        narrative = (
            f"Activated {len(mechanisms)} mechanisms "
            f"(avg strength {avg_strength:.2f}) with {len(interactions)} key interactions."
        )
        return MechanismActivation(
            mechanisms=mechanisms,
            key_interactions=interactions,
            narrative_explanation=narrative,
        )

    def _estimate_impact_distribution(
        self,
        action_id: str,
        action_type: str,
        action_subtype: str,
        params: Dict[str, Any],
        features: Dict[str, Any],
        state_snapshot: Optional[Dict[str, Any]],
        regime: Dict[str, Any],
        feasibility: FeasibilityResult,
        mechanism_activation: MechanismActivation,
    ) -> ImpactDistribution:
        objective_names = [
            "value_creation",
            "risk_reduction",
            "growth",
            "rating_preservation",
            "optionality",
        ]

        priors = self._objective_priors(action_type)
        mechanism_strength = (
            sum(m.activation_strength for m in mechanism_activation.mechanisms)
            / max(1, len(mechanism_activation.mechanisms))
        )
        interaction_shift = 0.0
        for x in mechanism_activation.key_interactions:
            interaction_shift += 0.03 if x.direction == "positive" else -0.03

        feasible_mult = 0.5 + 0.5 * feasibility.pass_probability
        regime_shift_by_obj = self._regime_shift_by_objective(action_id, action_type, regime)
        expectations_shift_by_obj, expectations_driver = self._capital_return_expectations_shift_by_objective(
            action_id, features
        )
        positioning_shift_by_obj, positioning_driver = self._capital_return_positioning_shift_by_objective(
            action_id, features
        )
        regime_sensitivity = [
            RegimeImpact(regime_condition=k, effect_shift=v) for k, v in sorted(regime_shift_by_obj.items()) if abs(v) > 1e-12
        ]

        out_obj: Dict[str, Distribution] = {}
        for obj in objective_names:
            prior = priors.get(obj, 0.0)
            causal = 0.02 * (mechanism_strength - 0.5) + interaction_shift
            median = (
                (0.60 * prior + 0.40 * causal) * feasible_mult
                + regime_shift_by_obj.get(obj, 0.0)
                + expectations_shift_by_obj.get(obj, 0.0)
                + positioning_shift_by_obj.get(obj, 0.0)
            )
            spread = 0.06 + 0.12 * (1.0 - feasibility.pass_probability) + 0.05 * (1.0 - mechanism_strength)
            p10 = median - 1.5 * spread
            p25 = median - 0.75 * spread
            p75 = median + 0.75 * spread
            p90 = median + 1.5 * spread
            out_obj[obj] = Distribution(
                median=round(median, 6),
                p10=round(min(p10, p25), 6),
                p25=round(p25, 6),
                p75=round(p75, 6),
                p90=round(max(p90, p75), 6),
            )

        maturity_ratio = _to_float(_extract_feature(features, "capital_structure.maturity_wall_ratio_24m"), 0.0) or 0.0
        drivers = [
            Driver(
                driver_name="mechanism_strength",
                contribution=round(mechanism_strength, 6),
                explanation="Average activation strength across ontology channels.",
            ),
            Driver(
                driver_name="feasibility_pass_probability",
                contribution=round(feasibility.pass_probability, 6),
                explanation="Feasibility confidence scales expected outcomes.",
            ),
            Driver(
                driver_name="maturity_wall_ratio_24m",
                contribution=round(-maturity_ratio, 6),
                explanation="Higher near-term debt wall reduces optionality and downside protection.",
            ),
        ]
        if expectations_driver is not None:
            drivers.append(expectations_driver)
        if positioning_driver is not None:
            drivers.append(positioning_driver)

        required_feats = [
            "liquidity.runway_months",
            "liquidity.available_for_actions",
            "capital_structure.net_leverage",
            "capital_structure.maturity_wall_ratio_24m",
            "market.market_cap",
            "operating.fcf_conversion",
        ]
        present = sum(1 for f in required_feats if _extract_feature(features, f) is not None)
        missing_ratio = 1.0 - (present / max(1, len(required_feats)))
        uncertainty = _clip(0.20 + 0.35 * missing_ratio + 0.25 * (1.0 - feasibility.pass_probability), 0.0, 1.0)

        causal = self._predict_causal_impact(
            action_id=action_id,
            action_type=action_type,
            action_subtype=action_subtype,
            params=params,
            features=state_snapshot if isinstance(state_snapshot, dict) else features,
            regime=regime,
        )
        if causal is not None and self.causal_mode == "standalone":
            causal_obj = dict(causal.get("objectives", {}) or {})
            causal_quality = float(causal.get("model_quality", 0.0) or 0.0)
            causal_support = float(causal.get("support_score", 0.0) or 0.0)
            causal_oos = bool(causal.get("out_of_sample_flag"))
            standalone_allowed, gate_reason = self._passes_strict_causal_gate(causal)
            action_status = str(causal.get("action_status", "enabled") or "enabled").strip().lower()
            if action_status != "enabled":
                standalone_allowed = False
                gate_reason = f"{gate_reason}|status={action_status}" if gate_reason else f"status={action_status}"
            used = 0
            if standalone_allowed:
                for objective_name, causal_dist in causal_obj.items():
                    current = out_obj.get(objective_name)
                    if current is None:
                        continue
                    merged = {}
                    for q in ("p10", "p25", "median", "p75", "p90"):
                        cur = _to_float(getattr(current, q), 0.0) or 0.0
                        merged[q] = _to_float(causal_dist.get(q), cur) or cur
                    ordered = sorted([merged["p10"], merged["p25"], merged["median"], merged["p75"], merged["p90"]])
                    out_obj[objective_name] = Distribution(
                        p10=round(ordered[0], 6),
                        p25=round(ordered[1], 6),
                        median=round(ordered[2], 6),
                        p75=round(ordered[3], 6),
                        p90=round(ordered[4], 6),
                    )
                    used += 1

            drivers.append(
                Driver(
                    driver_name="causal_model_mode",
                    contribution=1.0 if standalone_allowed else 0.0,
                    explanation=(
                        "Standalone causal mode enabled; causal outputs applied only when model quality/support "
                        "pass minimum safety thresholds."
                    ),
                )
            )
            drivers.append(
                Driver(
                    driver_name="causal_model_quality",
                    contribution=round(causal_quality, 6),
                    explanation=(
                        "Average out-of-sample quality for the selected causal models "
                        f"(version={str(causal.get('model_version', 'unknown'))})."
                    ),
                )
            )
            drivers.append(
                Driver(
                    driver_name="causal_model_support_score",
                    contribution=round(causal_support, 6),
                    explanation="Support score based on standardized feature distance from training support.",
                )
            )
            drivers.append(
                Driver(
                    driver_name="causal_model_min_oos_r2",
                    contribution=round(float(_to_float(causal.get("min_oos_r2"), 0.0) or 0.0), 6),
                    explanation="Minimum selected-cell out-of-sample R2 across scored objectives.",
                )
            )
            drivers.append(
                Driver(
                    driver_name="causal_model_min_treated_rows",
                    contribution=round(float(_to_float(causal.get("min_treated_rows"), 0.0) or 0.0), 6),
                    explanation="Minimum treated observations in selected causal cells.",
                )
            )
            drivers.append(
                Driver(
                    driver_name="causal_model_min_control_rows",
                    contribution=round(float(_to_float(causal.get("min_control_rows"), 0.0) or 0.0), 6),
                    explanation="Minimum control observations in selected causal cells.",
                )
            )
            coverage = _clip(float(causal.get("coverage_score", 0.0) or 0.0), 0.0, 1.0)
            support = _clip(causal_support, 0.0, 1.0)
            quality_scaled = _clip((causal_quality + 0.20) / 0.80, 0.0, 1.0)
            n_train = max(0.0, float(causal.get("n_train", 0.0) or 0.0))
            n_factor = _clip(n_train / (n_train + 1000.0), 0.0, 1.0)
            uncertainty = _clip(1.0 - (0.35 * coverage + 0.30 * support + 0.20 * quality_scaled + 0.15 * n_factor), 0.0, 1.0)
            if causal_oos:
                uncertainty = max(uncertainty, 0.65)
                drivers.append(
                    Driver(
                        driver_name="causal_model_oos_penalty",
                        contribution=-0.25,
                        explanation="Out-of-support causal prediction; uncertainty floor applied.",
                    )
                )
            if used == 0:
                drivers.append(
                    Driver(
                        driver_name="causal_model_standalone_fallback",
                        contribution=0.0,
                        explanation=(
                            "Retained deterministic fallback for this candidate "
                            f"(strict_causal_gate={gate_reason})."
                        ),
                    )
                )
        elif causal is not None:
            raw_blend_weight = float(causal.get("blend_weight", 0.0) or 0.0)
            action_status = str(causal.get("action_status", "enabled") or "enabled").strip().lower()
            max_blend_weight = _to_float(causal.get("max_blend_weight"))
            if action_status == "weak_prior_only" and max_blend_weight is not None:
                raw_blend_weight = min(raw_blend_weight, float(max_blend_weight))
            blend_allowed, gate_reason = self._passes_strict_causal_gate(causal)
            blend_weight = raw_blend_weight if blend_allowed else 0.0
            if blend_weight > 0.0:
                for objective_name, causal_dist in dict(causal.get("objectives", {}) or {}).items():
                    current = out_obj.get(objective_name)
                    if current is None:
                        continue
                    merged = {}
                    for q in ("p10", "p25", "median", "p75", "p90"):
                        cur = _to_float(getattr(current, q), 0.0) or 0.0
                        ml = _to_float(causal_dist.get(q), cur) or cur
                        merged[q] = (1.0 - blend_weight) * cur + blend_weight * ml
                    ordered = sorted([merged["p10"], merged["p25"], merged["median"], merged["p75"], merged["p90"]])
                    out_obj[objective_name] = Distribution(
                        p10=round(ordered[0], 6),
                        p25=round(ordered[1], 6),
                        median=round(ordered[2], 6),
                        p75=round(ordered[3], 6),
                        p90=round(ordered[4], 6),
                    )
            else:
                uncertainty = max(uncertainty, 0.60)

            drivers.append(
                Driver(
                    driver_name="causal_model_blend_weight",
                    contribution=round(blend_weight, 6),
                    explanation=(
                        "Effective blend weight after strict causal quality/support gate "
                        f"(strict_causal_gate={gate_reason})."
                    ),
                )
            )
            drivers.append(
                Driver(
                    driver_name="causal_model_quality",
                    contribution=round(float(causal.get("model_quality", 0.0) or 0.0), 6),
                    explanation=(
                        "Average out-of-sample quality for the selected causal models "
                        f"(version={str(causal.get('model_version', 'unknown'))})."
                    ),
                )
            )
            drivers.append(
                Driver(
                    driver_name="causal_model_support_score",
                    contribution=round(float(causal.get("support_score", 0.0) or 0.0), 6),
                    explanation="Support score based on standardized feature distance from training support.",
                )
            )
            drivers.append(
                Driver(
                    driver_name="causal_model_min_oos_r2",
                    contribution=round(float(_to_float(causal.get("min_oos_r2"), 0.0) or 0.0), 6),
                    explanation="Minimum selected-cell out-of-sample R2 across scored objectives.",
                )
            )
            drivers.append(
                Driver(
                    driver_name="causal_model_min_treated_rows",
                    contribution=round(float(_to_float(causal.get("min_treated_rows"), 0.0) or 0.0), 6),
                    explanation="Minimum treated observations in selected causal cells.",
                )
            )
            drivers.append(
                Driver(
                    driver_name="causal_model_min_control_rows",
                    contribution=round(float(_to_float(causal.get("min_control_rows"), 0.0) or 0.0), 6),
                    explanation="Minimum control observations in selected causal cells.",
                )
            )
            if bool(causal.get("out_of_sample_flag")):
                drivers.append(
                    Driver(
                        driver_name="causal_model_oos_penalty",
                        contribution=-0.25,
                        explanation="Out-of-support causal prediction; blend gate disabled causal contribution.",
                    )
                )
            uncertainty = _clip(
                uncertainty * (1.0 - 0.25 * blend_weight) * (1.0 - 0.10 * float(causal.get("coverage_score", 0.0) or 0.0)),
                0.0,
                1.0,
            )

        return ImpactDistribution(
            objectives=out_obj,
            regime_sensitivity=regime_sensitivity,
            key_drivers=drivers,
            uncertainty_score=round(uncertainty, 6),
        )

    def _predict_causal_impact(
        self,
        action_id: str,
        action_type: str,
        action_subtype: str,
        params: Dict[str, Any],
        features: Dict[str, Any],
        regime: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self.causal_model is None:
            return None
        if self._is_causal_action_blocked(
            action_id=action_id,
            action_type=action_type,
            action_subtype=action_subtype,
        ):
            return None
        try:
            pred = self.causal_model.predict(
                action_id=action_id,
                action_type=action_type,
                params=params,
                features=features,
                regime=regime,
                action_subtype=action_subtype,
            )
        except Exception:
            return None
        if pred is None:
            return None
        return {
            "objectives": pred.objectives,
            "blend_weight": pred.blend_weight,
            "coverage_score": pred.coverage_score,
            "n_train": pred.n_train,
            "model_version": pred.model_version,
            "model_quality": pred.model_quality,
            "support_score": pred.support_score,
            "out_of_sample_flag": pred.out_of_sample_flag,
            "min_oos_r2": pred.min_oos_r2,
            "min_treated_rows": pred.min_treated_rows,
            "min_control_rows": pred.min_control_rows,
            "selected_model_keys": list(pred.selected_model_keys),
            "model_gate_reason": pred.gate_reason,
            "action_status": pred.action_status,
            "objective_allowlist": list(pred.objective_allowlist),
            "strict_gate_primary_objectives": list(pred.strict_gate_primary_objectives),
            "model_artifact_path_override": pred.model_artifact_path_override,
            "max_blend_weight": pred.max_blend_weight,
            "future_action_alias": pred.future_action_alias,
            "quality_floor_override": pred.quality_floor_override,
            "support_floor_override": pred.support_floor_override,
            "min_train_rows_override": pred.min_train_rows_override,
            "min_oos_r2_override": pred.min_oos_r2_override,
            "min_treated_rows_override": pred.min_treated_rows_override,
            "min_control_rows_override": pred.min_control_rows_override,
        }

    def _is_causal_action_blocked(
        self,
        action_id: str,
        action_type: str,
        action_subtype: str,
    ) -> bool:
        policy = get_causal_action_policy(
            action_id=action_id,
            action_type=action_type,
            action_subtype=action_subtype,
        )
        if str(policy.status or "").strip().lower() == "blocked":
            return True
        if not self.causal_action_blocklist:
            return False
        aid = str(action_id or "").strip().lower()
        at = str(action_type or "").strip().lower()
        st = str(action_subtype or "").strip().lower()
        keys: set[str] = set()
        if aid:
            keys.add(aid)
        if at and st:
            keys.add(f"{at}.{st}")
        if at:
            keys.add(f"{at}.*")
        keys.add("*")
        return any(k in self.causal_action_blocklist for k in keys)

    def _passes_strict_causal_gate(self, causal: Dict[str, Any]) -> Tuple[bool, str]:
        quality_floor = _to_float(causal.get("quality_floor_override"))
        if quality_floor is None:
            quality_floor = float(self.causal_quality_floor)
        support_floor = _to_float(causal.get("support_floor_override"))
        if support_floor is None:
            support_floor = float(self.causal_support_floor)
        min_train_rows = _to_float(causal.get("min_train_rows_override"))
        if min_train_rows is None:
            min_train_rows = float(self.causal_min_train_rows)
        min_oos_r2_floor = _to_float(causal.get("min_oos_r2_override"))
        if min_oos_r2_floor is None:
            min_oos_r2_floor = float(self.causal_min_oos_r2)
        min_treated_rows = _to_float(causal.get("min_treated_rows_override"))
        if min_treated_rows is None:
            min_treated_rows = float(self.causal_min_treated_rows)
        min_control_rows = _to_float(causal.get("min_control_rows_override"))
        if min_control_rows is None:
            min_control_rows = float(self.causal_min_control_rows)

        quality = float(causal.get("model_quality", 0.0) or 0.0)
        support = float(causal.get("support_score", 0.0) or 0.0)
        n_train = int(_to_float(causal.get("n_train"), 0.0) or 0.0)
        out_of_sample = bool(causal.get("out_of_sample_flag"))
        min_oos_r2 = _to_float(causal.get("min_oos_r2"))
        treated_rows = int(_to_float(causal.get("min_treated_rows"), 0.0) or 0.0)
        control_rows = int(_to_float(causal.get("min_control_rows"), 0.0) or 0.0)

        failures: List[str] = []
        if out_of_sample:
            failures.append("out_of_support")
        if quality < quality_floor:
            failures.append(f"quality<{quality_floor:.2f}")
        if support < support_floor:
            failures.append(f"support<{support_floor:.2f}")
        if n_train < int(max(0.0, min_train_rows)):
            failures.append(f"n_train<{int(max(0.0, min_train_rows))}")
        if min_oos_r2 is None:
            failures.append("oos_unavailable")
        elif float(min_oos_r2) < min_oos_r2_floor:
            failures.append(f"oos_r2<{min_oos_r2_floor:.2f}")
        if treated_rows < int(max(0.0, min_treated_rows)):
            failures.append(f"treated<{int(max(0.0, min_treated_rows))}")
        if control_rows < int(max(0.0, min_control_rows)):
            failures.append(f"control<{int(max(0.0, min_control_rows))}")

        if failures:
            return False, "|".join(failures)
        return True, "pass"

    def _run_structural_checks(
        self,
        run: Any,
        action_id: str,
        action_type: str,
        params: Dict[str, Any],
        features: Dict[str, Any],
        mechanism_activation: MechanismActivation,
        feasibility: FeasibilityResult,
    ) -> List[SanityCheck]:
        checks: List[SanityCheck] = []

        debt_pct = _to_float(_nested_get(params, "funding_mix.debt"), 0.0) or 0.0
        runway = _to_float(_extract_feature(features, "liquidity.runway_months"))
        ev_z = _to_float(_extract_feature(features, "market.ev_ebitda_vs_peer_z"))
        fcf_pct = _to_float(_extract_feature(features, "market.fcf_yield_percentile_peers"))
        expectations_coverage, expectations_revision = self._expectations_context(features)
        risk_weight = float(getattr(run.objectives, "risk_reduction_weight", 0.0))
        rating_weight = float(getattr(run.objectives, "rating_preservation_weight", 0.0))

        if action_id.startswith("capital_return.") and debt_pct > 0.5 and risk_weight >= 0.35:
            checks.append(
                SanityCheck(
                    check_type="objective_contradiction",
                    status="fail",
                    explanation="Debt-funded capital return conflicts with high risk-reduction objective weight.",
                )
            )
        elif action_type == "mna" and rating_weight >= 0.30 and feasibility.feasibility_status != "feasible":
            checks.append(
                SanityCheck(
                    check_type="objective_contradiction",
                    status="warning",
                    explanation="Growth action conflicts with elevated rating-preservation priority under constrained feasibility.",
                )
            )
        else:
            checks.append(
                SanityCheck(
                    check_type="objective_contradiction",
                    status="pass",
                    explanation="No material objective contradiction detected.",
                )
            )

        if action_id.startswith(_CASH_CONSUMING_ACTION_PREFIXES):
            if runway is not None and runway < 6:
                checks.append(
                    SanityCheck(
                        check_type="liquidity_contradiction",
                        status="fail",
                        explanation="Cash-consuming action while liquidity runway is already fragile (<6 months).",
                    )
                )
            elif runway is not None and runway < 12:
                checks.append(
                    SanityCheck(
                        check_type="liquidity_contradiction",
                        status="warning",
                        explanation="Cash-consuming action with moderate runway cushion (6-12 months).",
                    )
                )
            else:
                checks.append(
                    SanityCheck(
                        check_type="liquidity_contradiction",
                        status="pass",
                        explanation="Liquidity profile does not contradict action cash usage.",
                    )
                )

        if action_id in {"capital_return.open_market_buyback", "capital_return.accelerated_share_repurchase"}:
            undervaluation_strength = 0.0
            for m in mechanism_activation.mechanisms:
                if m.mechanism_id == "undervaluation_arbitrage":
                    undervaluation_strength = m.activation_strength
                    break
            if undervaluation_strength < 0.25 and (ev_z is None or ev_z >= 0) and (fcf_pct is None or fcf_pct < 70):
                checks.append(
                    SanityCheck(
                        check_type="mechanism_contradiction",
                        status="warning",
                        explanation="Buyback selected without strong undervaluation support.",
                    )
                )
            else:
                checks.append(
                    SanityCheck(
                        check_type="mechanism_contradiction",
                        status="pass",
                        explanation="Mechanism activation aligns with action rationale.",
                    )
                )

        if action_id.startswith("capital_return."):
            if (
                expectations_coverage is not None
                and expectations_coverage >= self.expectations_coverage_min
                and expectations_revision is not None
                and expectations_revision <= self.expectations_revision_negative_floor
            ):
                checks.append(
                    SanityCheck(
                        check_type="expectations_contradiction",
                        status="warning",
                        explanation="Capital-return action runs against sharply negative FY1 estimate revisions with meaningful analyst coverage.",
                    )
                )
            else:
                checks.append(
                    SanityCheck(
                        check_type="expectations_contradiction",
                        status="pass",
                        explanation="No material expectations contradiction detected for the proposed capital-return action.",
                    )
                )

        return checks

    def _identify_risks(
        self,
        action_id: str,
        action_type: str,
        feasibility: FeasibilityResult,
        mechanism_activation: MechanismActivation,
        regime: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> List[RiskItem]:
        risks: List[RiskItem] = []
        for blocker in feasibility.blockers:
            if blocker.blocker_type == "rating_downgrade_risk":
                risks.append(RiskItem("rating_downgrade_risk", 0.65, blocker.explanation))
            elif blocker.blocker_type == "market_access_closed":
                risks.append(RiskItem("market_access_risk", 0.50, blocker.explanation))
            elif blocker.blocker_type == "operational_capacity_limit":
                risks.append(RiskItem("execution_risk", 0.45, blocker.explanation))
            elif blocker.blocker_type == "maturity_wall_conflict":
                risks.append(RiskItem("refinancing_risk", 0.55, blocker.explanation))

        if action_type == "mna":
            risks.append(
                RiskItem(
                    risk_type="integration_risk",
                    probability=0.45 if action_id == "mna.transformational_acquisition" else 0.30,
                    explanation="M&A integration execution may dilute expected synergies.",
                )
            )

        credit_regime = str(regime.get("credit_regime", "neutral"))
        if credit_regime == "tight":
            risks.append(
                RiskItem(
                    risk_type="regime_shift_risk",
                    probability=0.35,
                    explanation="Tight credit regime can weaken financing-dependent outcomes.",
                )
            )

        complexity = _to_float(_nested_get(schema, "execution_complexity_prior.base_complexity_score"), 3.0) or 3.0
        if complexity >= 4:
            risks.append(
                RiskItem(
                    risk_type="execution_risk",
                    probability=0.40,
                    explanation="High execution complexity increases implementation slippage risk.",
                )
            )

        # Deduplicate by risk_type, keep max probability.
        merged: Dict[str, RiskItem] = {}
        for r in risks:
            prev = merged.get(r.risk_type)
            if prev is None or r.probability > prev.probability:
                merged[r.risk_type] = r
        return list(merged.values())

    def _collect_assumptions(
        self,
        action_id: str,
        action_type: str,
        params: Dict[str, Any],
        feasibility: FeasibilityResult,
        regime: Dict[str, Any],
    ) -> List[Assumption]:
        out: List[Assumption] = []
        if self._requires_debt(action_id, params):
            out.append(
                Assumption(
                    assumption_type="funding_access",
                    description=f"Debt financing remains available under credit_regime={regime.get('credit_regime', 'neutral')}.",
                    sensitivity="high",
                )
            )
        if self._requires_equity(action_id, params):
            out.append(
                Assumption(
                    assumption_type="equity_window",
                    description=f"Equity issuance window remains serviceable under vol_regime={regime.get('vol_regime', 'normal')}.",
                    sensitivity="high",
                )
            )
        if action_type == "portfolio":
            out.append(
                Assumption(
                    assumption_type="asset_marketability",
                    description="Portfolio assets can be monetized near modeled valuation ranges.",
                    sensitivity="medium",
                )
            )
        if feasibility.feasibility_status == "conditional":
            out.append(
                Assumption(
                    assumption_type="remediation_execution",
                    description="Remediation actions complete before primary action launch.",
                    sensitivity="high",
                )
            )
        return out

    def _evaluation_confidence(
        self,
        candidate: Dict[str, Any],
        features: Dict[str, Any],
        mechanism_activation: MechanismActivation,
        impact_distribution: ImpactDistribution,
        feasibility: FeasibilityResult,
    ) -> float:
        key_features = [
            "liquidity.runway_months",
            "liquidity.available_for_actions",
            "capital_structure.net_leverage",
            "capital_structure.maturity_wall_ratio_24m",
            "market.market_cap",
            "operating.fcf_conversion",
            "regime.credit_regime",
        ]
        present = 0
        for name in key_features:
            if name.startswith("regime."):
                # Regime is carried separately; treat as present in this context.
                present += 1
            elif _extract_feature(features, name) is not None:
                present += 1
        data_completeness = present / max(1, len(key_features))

        mech_strength = (
            sum(m.activation_strength for m in mechanism_activation.mechanisms)
            / max(1, len(mechanism_activation.mechanisms))
        )
        mechanism_clarity = _clip(0.5 + abs(mech_strength - 0.5), 0.0, 1.0)

        precedent_coverage = 0.65
        refs = candidate.get("rationale_refs", []) if isinstance(candidate, dict) else []
        if isinstance(refs, list) and any(isinstance(r, dict) and r.get("reference_type") == "precedent_signal" for r in refs):
            precedent_coverage = 0.80

        medians = [d.median for d in impact_distribution.objectives.values()]
        dispersion = (max(medians) - min(medians)) if medians else 0.0
        model_agreement = _clip(1.0 - abs(dispersion) / 0.6, 0.0, 1.0)

        conf = (
            0.35 * data_completeness
            + 0.25 * mechanism_clarity
            + 0.20 * precedent_coverage
            + 0.20 * model_agreement
        )
        conf *= 0.85 + 0.15 * feasibility.pass_probability
        return round(_clip(conf, 0.0, 1.0), 6)

    def _evaluate_signal_expression(
        self,
        expr: str,
        features: Dict[str, Any],
        regime: Dict[str, Any],
    ) -> Tuple[bool, Optional[Signal]]:
        text = str(expr or "").strip()
        if not text:
            return False, None

        m = re.match(r"^([A-Za-z0-9_.]+)\s*(<=|>=|==|!=|<|>)\s*(.+)$", text)
        if m:
            lhs_name = str(m.group(1)).strip()
            op = str(m.group(2)).strip()
            rhs_raw = str(m.group(3)).strip()

            lhs_val = self._resolve_signal_lhs(lhs_name, features, regime)
            rhs_val = self._resolve_signal_rhs(rhs_raw)
            ok = self._compare(lhs_val, op, rhs_val, lhs_name)
            if not ok:
                return False, None
            return (
                True,
                Signal(
                    feature_name=lhs_name,
                    value=lhs_val,
                    threshold=rhs_val,
                    interpretation=f"{lhs_name} {op} {rhs_val}",
                ),
            )

        # Fallback: treat as boolean flag lookup.
        lhs_val = self._resolve_signal_lhs(text, features, regime)
        ok = bool(lhs_val)
        if not ok:
            return False, None
        return (
            True,
            Signal(
                feature_name=text,
                value=lhs_val,
                threshold=True,
                interpretation=f"{text} interpreted as truthy flag.",
            ),
        )

    def _resolve_signal_lhs(self, name: str, features: Dict[str, Any], regime: Dict[str, Any]) -> Any:
        if name.startswith("regime."):
            return regime.get(name.split(".", 1)[1])
        return _extract_feature(features, name)

    def _resolve_signal_rhs(self, raw: str) -> Any:
        token = raw.strip().strip("'").strip('"')
        alias = {
            "target_band_high": 3.0,
            "threshold": 0.0,
            "high": "high",
            "low": "low",
            "tight": "tight",
            "neutral": "neutral",
            "loose": "loose",
            "risk_off": "risk_off",
            "risk_on": "risk_on",
            "true": True,
            "false": False,
        }
        if token in alias:
            return alias[token]
        if token.lower() in alias:
            return alias[token.lower()]
        n = _to_float(token)
        if n is not None:
            return n
        return token

    def _compare(self, lhs: Any, op: str, rhs: Any, lhs_name: str) -> bool:
        lnum = _to_float(lhs)
        rnum = _to_float(rhs)
        if lnum is not None and rnum is not None:
            # Percentile harmonization: ontology often uses 0-100 while features may be 0-1.
            if "percentile" in lhs_name and rnum > 1.0 and lnum <= 1.0:
                lnum = lnum * 100.0
            if op == "<":
                return lnum < rnum
            if op == "<=":
                return lnum <= rnum
            if op == ">":
                return lnum > rnum
            if op == ">=":
                return lnum >= rnum
            if op == "==":
                return abs(lnum - rnum) <= 1e-12
            if op == "!=":
                return abs(lnum - rnum) > 1e-12
            return False

        ls = str(lhs).lower()
        rs = str(rhs).lower()
        if op == "==":
            return ls == rs
        if op == "!=":
            return ls != rs
        return False

    def _derive_interactions(
        self,
        action_id: str,
        action_type: str,
        params: Dict[str, Any],
        features: Dict[str, Any],
        regime: Dict[str, Any],
    ) -> List[Interaction]:
        out: List[Interaction] = []
        ev_z = _to_float(_extract_feature(features, "market.ev_ebitda_vs_peer_z"))
        fcf_conv = _to_float(_extract_feature(features, "operating.fcf_conversion"))
        net_lev = _to_float(_extract_feature(features, "capital_structure.net_leverage"))
        maturity_ratio = _to_float(_extract_feature(features, "capital_structure.maturity_wall_ratio_24m"), 0.0) or 0.0
        credit_regime = str(regime.get("credit_regime", "neutral"))

        if action_id in {"capital_return.open_market_buyback", "capital_return.accelerated_share_repurchase"}:
            if (ev_z is not None and ev_z < -1.0) and (fcf_conv is not None and fcf_conv > 0.20) and (
                net_lev is not None and net_lev < 3.0
            ):
                out.append(
                    Interaction(
                        feature_combination="low valuation + strong FCF + low leverage",
                        direction="positive",
                        explanation="Undervaluation channel is amplified when cash generation is strong and leverage is conservative.",
                    )
                )

        cash_usage = self._estimate_cash_usage_usd(
            action_id,
            params,
            self._estimate_action_size_usd(action_id, params, _to_float(_extract_feature(features, "market.market_cap"), 0.0) or 0.0),
        )
        if credit_regime == "tight" and maturity_ratio > self.maturity_wall_threshold and cash_usage > 0:
            out.append(
                Interaction(
                    feature_combination="tight credit + elevated maturity wall + liquidity usage",
                    direction="negative",
                    explanation="Liquidity-consuming actions become fragile when refinancing conditions tighten.",
                )
            )

        if action_id == "capital_structure.refinancing" and maturity_ratio > self.maturity_wall_threshold:
            out.append(
                Interaction(
                    feature_combination="refinancing + elevated maturity wall",
                    direction="positive",
                    explanation="Refinancing directly addresses near-term debt concentration.",
                )
            )
        return out

    def _objective_priors(self, action_type: str) -> Dict[str, float]:
        priors: Dict[str, Dict[str, float]] = {
            "capital_return": {
                "value_creation": 0.08,
                "risk_reduction": -0.03,
                "growth": 0.00,
                "rating_preservation": -0.02,
                "optionality": -0.04,
            },
            "capital_structure": {
                "value_creation": 0.03,
                "risk_reduction": 0.09,
                "growth": 0.01,
                "rating_preservation": 0.06,
                "optionality": 0.05,
            },
            "mna": {
                "value_creation": 0.07,
                "risk_reduction": -0.04,
                "growth": 0.10,
                "rating_preservation": -0.03,
                "optionality": 0.01,
            },
            "portfolio": {
                "value_creation": 0.06,
                "risk_reduction": 0.05,
                "growth": 0.02,
                "rating_preservation": 0.03,
                "optionality": 0.06,
            },
            "restructuring": {
                "value_creation": 0.05,
                "risk_reduction": 0.04,
                "growth": 0.01,
                "rating_preservation": 0.02,
                "optionality": 0.03,
            },
            "governance": {
                "value_creation": 0.03,
                "risk_reduction": 0.03,
                "growth": 0.01,
                "rating_preservation": 0.01,
                "optionality": 0.04,
            },
        }
        return priors.get(
            action_type,
            {
                "value_creation": 0.02,
                "risk_reduction": 0.02,
                "growth": 0.01,
                "rating_preservation": 0.01,
                "optionality": 0.01,
            },
        )

    def _regime_shift_by_objective(self, action_id: str, action_type: str, regime: Dict[str, Any]) -> Dict[str, float]:
        out = {k: 0.0 for k in ["value_creation", "risk_reduction", "growth", "rating_preservation", "optionality"]}
        credit = str(regime.get("credit_regime", "neutral"))
        risk = str(regime.get("risk_regime", "neutral"))
        vol = str(regime.get("vol_regime", "normal"))

        if credit == "tight" and self._requires_debt(action_id, {}):
            out["value_creation"] -= 0.03
            out["optionality"] -= 0.05
            out["risk_reduction"] -= 0.02
        if vol == "high" and self._requires_equity(action_id, {}):
            out["value_creation"] -= 0.02
            out["growth"] -= 0.02
        if risk == "risk_off" and action_type == "mna":
            out["growth"] -= 0.03
            out["value_creation"] -= 0.02
        return out

    def _estimate_action_size_usd(self, action_id: str, params: Dict[str, Any], market_cap: float) -> float:
        size_abs = _to_float(params.get("size_absolute_usd"))
        if size_abs is not None and size_abs > 0:
            return size_abs
        pct = _to_float(params.get("size_pct_market_cap"))
        if pct is not None and market_cap > 0:
            return max(0.0, pct) * market_cap
        amount = _to_float(params.get("amount"))
        if amount is not None and amount > 0:
            return amount
        target_pct_ev = _to_float(params.get("target_size_pct_ev"))
        if target_pct_ev is not None and market_cap > 0:
            return max(0.0, target_pct_ev) * market_cap
        return 0.0

    def _allows_dividend_continuity_liquidity_exception(
        self,
        action_id: str,
        params: Dict[str, Any],
        features: Dict[str, Any],
        runway_adj: Optional[float],
    ) -> bool:
        if action_id != "capital_return.dividend_increase":
            return False
        if runway_adj is None or runway_adj < 2.5 or runway_adj >= 6.0:
            return False

        annualized_cash_commitment = _to_float(params.get("annualized_cash_commitment_usd"))
        if annualized_cash_commitment is None or annualized_cash_commitment <= 0:
            return False

        available_liq = _to_float(_extract_feature(features, "liquidity.available_for_actions"), 0.0) or 0.0
        cash = _to_float(_extract_feature(features, "liquidity.cash"), 0.0) or 0.0
        market_cap = _to_float(_extract_feature(features, "market.market_cap"), 0.0) or 0.0
        total_debt = _to_float(_extract_feature(features, "capital_structure.total_debt"), 0.0) or 0.0
        net_leverage = _to_float(_extract_feature(features, "capital_structure.net_leverage"))
        interest_coverage = _to_float(_extract_feature(features, "capital_structure.interest_coverage"))
        debt_due_0_12m = _to_float(_extract_feature(features, "capital_structure.debt_due_0_12m"), 0.0) or 0.0
        debt_due_12_24m = _to_float(_extract_feature(features, "capital_structure.debt_due_12_24m"), 0.0) or 0.0
        maturity_ratio = _to_float(_extract_feature(features, "capital_structure.maturity_wall_ratio_24m"), 0.0) or 0.0
        return_capital_priority = _to_float(_extract_feature(features, "strategic.intent.return_capital_priority"), 0.0) or 0.0
        recent_actions_count = _to_float(_extract_feature(features, "strategic.recent_actions_count_24m"), 0.0) or 0.0
        action_frequency = _to_float(_extract_feature(features, "strategic.action_frequency_24m"), 0.0) or 0.0
        last_action_type = str(_extract_feature(features, "strategic.last_action_type") or "").strip().lower()

        if available_liq > 0.0 or cash <= 0.0 or market_cap <= 0.0:
            return False

        quarterly_cash_commitment = annualized_cash_commitment / 4.0
        if quarterly_cash_commitment > max(1_000_000.0, 0.25 * cash):
            return False
        if annualized_cash_commitment / market_cap > 0.005:
            return False
        if total_debt > 0.0 and total_debt / market_cap > 0.35:
            return False
        if net_leverage is not None and net_leverage > 3.25:
            return False
        if interest_coverage is None or interest_coverage < 3.0:
            return False
        if (debt_due_0_12m + debt_due_12_24m) > 0.02 * market_cap:
            return False
        if maturity_ratio > self.maturity_wall_threshold:
            return False
        if return_capital_priority < 0.8:
            return False
        if recent_actions_count < 5.0 or action_frequency < 0.20:
            return False
        if last_action_type not in {"buyback", "dividend_increase", "dividend_maintenance", "special_dividend"}:
            return False
        return True

    def _estimate_cash_usage_usd(self, action_id: str, params: Dict[str, Any], action_size_usd: float) -> float:
        if action_size_usd <= 0:
            return 0.0
        funding_cash = _to_float(_nested_get(params, "funding_mix.cash"), 1.0 if action_id.startswith(_CASH_CONSUMING_ACTION_PREFIXES) else 0.0)
        funding_cash = _clip(float(funding_cash or 0.0), 0.0, 1.0)
        if action_id.startswith(_CASH_CONSUMING_ACTION_PREFIXES):
            return action_size_usd * funding_cash
        if action_id in {"capital_structure.tender_offer_debt", "capital_structure.exchange_offer"}:
            return action_size_usd * max(0.2, funding_cash)
        return 0.0

    def _estimate_debt_delta_usd(self, action_id: str, params: Dict[str, Any], action_size_usd: float) -> float:
        if action_size_usd <= 0:
            return 0.0
        if action_id in _DEBT_REQUIRING_ACTION_IDS or action_id.startswith("capital_structure.new_debt_issuance"):
            if action_id in {"capital_structure.refinancing", "capital_structure.exchange_offer"}:
                return action_size_usd * 0.05
            return action_size_usd
        debt_mix = _to_float(_nested_get(params, "funding_mix.debt"), 0.0) or 0.0
        if debt_mix > 0 and action_id.startswith(_CASH_CONSUMING_ACTION_PREFIXES):
            return action_size_usd * _clip(debt_mix, 0.0, 1.0)
        return 0.0

    def _estimate_equity_raise_usd(self, action_id: str, params: Dict[str, Any], action_size_usd: float) -> float:
        if action_size_usd <= 0:
            return 0.0
        if action_id in _EQUITY_REQUIRING_ACTION_IDS:
            return action_size_usd
        eq_mix = _to_float(_nested_get(params, "funding_mix.equity"), 0.0) or 0.0
        if eq_mix > 0:
            return action_size_usd * _clip(eq_mix, 0.0, 1.0)
        return 0.0

    def _requires_debt(self, action_id: str, params: Dict[str, Any]) -> bool:
        if action_id in _DEBT_REQUIRING_ACTION_IDS:
            return True
        debt_mix = _to_float(_nested_get(params, "funding_mix.debt"), 0.0) or 0.0
        return debt_mix > 0.0

    def _requires_equity(self, action_id: str, params: Dict[str, Any]) -> bool:
        if action_id in _EQUITY_REQUIRING_ACTION_IDS:
            return True
        eq_mix = _to_float(_nested_get(params, "funding_mix.equity"), 0.0) or 0.0
        return eq_mix > 0.0

    def _resolve_max_leverage_threshold(self, run: Any) -> float:
        for c in getattr(run.constraints, "hard_constraints", []) or []:
            if getattr(c, "constraint_type", "") == "leverage_limit":
                v = _to_float((getattr(c, "parameters", {}) or {}).get("max_leverage"))
                if v is not None:
                    return float(v)
        return float(self.max_leverage_threshold_default)

    def _is_investment_grade(self, rating_state_value: Any) -> bool:
        rating = ""
        if isinstance(rating_state_value, dict):
            rating = str(rating_state_value.get("rating", "") or "")
        elif isinstance(rating_state_value, str):
            rating = rating_state_value
        r = rating.upper().strip()
        if not r:
            return False
        return not (r.startswith("BB") or r.startswith("B") or r.startswith("CCC") or r.startswith("CC") or r.startswith("C") or r.startswith("D"))

    def _remediation_for_blocker(self, blocker: Blocker) -> List[Remediation]:
        if blocker.blocker_type == "liquidity_shortfall":
            return [
                Remediation(
                    action_required="Add liquidity buffer via refinancing or working-capital program before action launch.",
                    expected_effect="Improves runway and reduces execution fragility.",
                    estimated_delay_days=21,
                )
            ]
        if blocker.blocker_type == "leverage_breach":
            return [
                Remediation(
                    action_required="Reduce action size or increase equity/asset-sale funding share.",
                    expected_effect="Keeps pro-forma leverage within policy bands.",
                    estimated_delay_days=14,
                )
            ]
        if blocker.blocker_type == "maturity_wall_conflict":
            return [
                Remediation(
                    action_required="Refinance near-term maturities before liquidity-consuming action.",
                    expected_effect="Lowers maturity concentration and refinancing risk.",
                    estimated_delay_days=30,
                )
            ]
        if blocker.blocker_type == "rating_downgrade_risk":
            return [
                Remediation(
                    action_required="Commit explicit deleveraging path and preserve coverage cushion.",
                    expected_effect="Reduces downgrade probability around boundary conditions.",
                    estimated_delay_days=20,
                )
            ]
        if blocker.blocker_type == "market_access_closed":
            return [
                Remediation(
                    action_required="Stage timing to a better market window or switch funding channel.",
                    expected_effect="Improves placement certainty and transaction economics.",
                    estimated_delay_days=15,
                )
            ]
        if blocker.blocker_type == "operational_capacity_limit":
            return [
                Remediation(
                    action_required="Sequence preparatory workstreams and governance approvals first.",
                    expected_effect="Increases execution reliability for complex transactions.",
                    estimated_delay_days=30,
                )
            ]
        return []

    @staticmethod
    def _is_uuid(value: str) -> bool:
        try:
            uuid.UUID(str(value))
            return True
        except Exception:
            return False

    @staticmethod
    def blend_precedent_into_action_candidate(
        action_candidate: Dict[str, Any],
        precedent_pack: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Blend precedent distribution into action-candidate impact distribution deterministically."""
        out = dict(action_candidate or {})
        impact = dict(out.get("impact_distribution", {}) or {})
        objectives = dict(impact.get("objectives", {}) or {})
        if not objectives:
            return out

        dist = MechanismBrain._select_precedent_distribution(precedent_pack)
        if not dist:
            return out

        target_obj = MechanismBrain._metric_to_objective(str(dist.get("metric", "")))
        target = dict(objectives.get(target_obj, {}) or {})
        if not target:
            return out

        n = int(_to_float(dist.get("n"), 0.0) or 0)
        p25 = _to_float(dist.get("p25"))
        p50 = _to_float(dist.get("p50"))
        p75 = _to_float(dist.get("p75"))
        if p25 is None or p50 is None or p75 is None:
            return out

        spread = max(1e-9, p75 - p25)
        precedent_quantiles = {
            "p10": p25 - 0.67 * spread,
            "p25": p25,
            "median": p50,
            "p75": p75,
            "p90": p75 + 0.67 * spread,
        }
        blend_weight = _clip((n / (n + 40.0)) if n > 0 else 0.0, 0.15, 0.70)

        merged = {}
        for q in ("p10", "p25", "median", "p75", "p90"):
            cur = _to_float(target.get(q), 0.0) or 0.0
            prec = _to_float(precedent_quantiles.get(q), cur) or cur
            merged[q] = round((1.0 - blend_weight) * cur + blend_weight * prec, 6)

        # Ensure monotone quantiles.
        ordered = sorted(
            [merged["p10"], merged["p25"], merged["median"], merged["p75"], merged["p90"]]
        )
        merged = {
            "p10": ordered[0],
            "p25": ordered[1],
            "median": ordered[2],
            "p75": ordered[3],
            "p90": ordered[4],
        }
        objectives[target_obj] = merged
        impact["objectives"] = objectives

        old_unc = _to_float(impact.get("uncertainty_score"), 0.5) or 0.5
        impact["uncertainty_score"] = round(_clip(old_unc * (1.0 - 0.30 * blend_weight), 0.0, 1.0), 6)

        drivers = list(impact.get("key_drivers", []) or [])
        drivers.append(
            {
                "driver_name": "precedent_blend_weight",
                "contribution": round(float(blend_weight), 6),
                "explanation": f"Blended {target_obj} distribution with precedent metric={dist.get('metric')} n={n}.",
            }
        )
        impact["key_drivers"] = drivers
        impact["blend_metadata"] = {
            "source": "precedent_distribution",
            "target_objective": target_obj,
            "metric": dist.get("metric"),
            "n": n,
            "blend_weight": round(float(blend_weight), 6),
        }

        out["impact_distribution"] = impact
        return out

    @staticmethod
    def _select_precedent_distribution(precedent_pack: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(precedent_pack, dict):
            return {}
        dists = precedent_pack.get("legacy_distributions")
        if not isinstance(dists, list) or not dists:
            dists = precedent_pack.get("distributions", [])
        if not isinstance(dists, list) or not dists:
            return {}
        preferred = ("outcome_pe_12m", "outcome_ev_ebitda_12m", "outcome_pe_6m")
        for metric in preferred:
            for d in dists:
                if str((d or {}).get("metric", "")) == metric:
                    return dict(d)
        for d in dists:
            if str((d or {}).get("metric", "")).endswith("_12m"):
                return dict(d)
        return dict(dists[0]) if isinstance(dists[0], dict) else {}

    @staticmethod
    def _metric_to_objective(metric: str) -> str:
        m = str(metric).lower()
        if "pe" in m or "ev_ebitda" in m or "valuation" in m:
            return "value_creation"
        if "drawdown" in m or "vol" in m or "default" in m:
            return "risk_reduction"
        if "revenue" in m or "growth" in m:
            return "growth"
        if "rating" in m or "spread" in m or "coverage" in m:
            return "rating_preservation"
        return "value_creation"


def evaluate_action_candidates(
    run: Any,
    state_snapshot: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    action_registry: Any,
    peer_set: Optional[Dict[str, Any]] = None,
    extracted_facts: Optional[List[Dict[str, Any]]] = None,
    event_store: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    brain = MechanismBrain(action_registry=action_registry)
    evaluated = brain.evaluate_candidate_set(
        run=run,
        state_snapshot=state_snapshot,
        candidates=candidates,
        peer_set=peer_set,
        extracted_facts=extracted_facts,
        event_store=event_store,
    )
    return [x.to_dict() for x in evaluated]


__all__ = [
    "ActionCandidate",
    "Assumption",
    "Blocker",
    "Distribution",
    "Driver",
    "FeasibilityResult",
    "ImpactDistribution",
    "Interaction",
    "Mechanism",
    "MechanismActivation",
    "MechanismBrain",
    "RegimeImpact",
    "Remediation",
    "RiskItem",
    "SanityCheck",
    "Signal",
    "evaluate_action_candidates",
]
