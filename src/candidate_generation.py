"""Step 6 Candidate Generation Engine.

Deterministic, schema-constrained candidate generation under a frozen RecommendationRun.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import uuid

from .model_feature_bundle import feature_view_from_snapshot
from .action_ontology import ActionSchemaRegistry
from .recommendation_run import RecommendationRun
from .runtime_feature_adapter import resolve_feature_record


_RELATION_MAP = {
    ">": "greater_than",
    ">=": "greater_than",
    "<": "less_than",
    "<=": "less_than",
    "==": "equal",
    "=": "equal",
    "between": "within_range",
    "in_range": "within_range",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_num(v: float) -> float:
    return float(round(float(v), 12))


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(v[k]) for k in sorted(v.keys(), key=str)}
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    if isinstance(v, tuple):
        return [_json_safe(x) for x in v]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _round_num(v)
    return v


def _candidate_signature(action_id: str, parameters: Dict[str, Any]) -> str:
    norm = _json_safe(parameters)
    key = f"{action_id}|{json.dumps(norm, sort_keys=True, separators=(',', ':'), ensure_ascii=True)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _feature_record(features: Dict[str, Any], feature_name: str) -> Optional[Dict[str, Any]]:
    if not isinstance(features, dict):
        return None
    raw = features.get(feature_name)
    return raw if isinstance(raw, dict) else None


def _feature_is_hard_blocked(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    support_mode = str(raw.get("support_mode") or "").strip().lower()
    applicability_status = str(raw.get("applicability_status") or "").strip().lower()
    quality_flags = {
        str(flag).strip().lower()
        for flag in (raw.get("quality_flags") or [])
        if flag is not None
    }
    if support_mode == "unsupported":
        return True
    if applicability_status in {"unsupported", "diagnostic"}:
        return True
    if "unsupported_metric" in quality_flags or "sector_native_metrics_required" in quality_flags:
        return True
    return False


def _feature_replacement_value(features: Dict[str, Any], feature_name: str, raw: Any) -> Any:
    if feature_name != "capital_structure.interest_coverage":
        return None
    replacement = _feature_record(features, "capital_structure.fixed_charge_coverage")
    if not isinstance(replacement, dict) or _feature_is_hard_blocked(replacement):
        return None
    replacement_applicability = str(replacement.get("applicability_status") or "").strip().lower()
    current_applicability = str(raw.get("applicability_status") or "").strip().lower() if isinstance(raw, dict) else ""
    if replacement_applicability != "primary":
        return None
    if _feature_is_hard_blocked(raw) or current_applicability in {"secondary", "diagnostic", "unsupported"}:
        return replacement.get("value")
    return None


def _feature_value(features: Dict[str, Any], feature_name: str, default: Any = None) -> Any:
    if not isinstance(features, dict):
        return default

    raw = resolve_feature_record(features, feature_name)
    if raw is not None:
        replacement_value = _feature_replacement_value(features, feature_name, raw)
        if replacement_value is not None:
            return replacement_value
        if _feature_is_hard_blocked(raw):
            return default
        if isinstance(raw, dict) and "value" in raw:
            return raw.get("value")
        return raw

    # Nested fallback: allow key lookups against parent feature values.
    parts = feature_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:i])
        suffix = parts[i:]
        if prefix not in features:
            continue
        raw = features.get(prefix)
        if _feature_is_hard_blocked(raw):
            continue
        if isinstance(raw, dict) and "value" in raw:
            raw = raw.get("value")
        cur = raw
        ok = True
        for tok in suffix:
            if isinstance(cur, dict) and tok in cur:
                cur = cur[tok]
            else:
                ok = False
                break
        if ok:
            return cur

    return default


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _default_funding_mixes() -> List[Dict[str, float]]:
    return [
        {"cash": 1.0, "debt": 0.0, "equity": 0.0},
        {"cash": 0.7, "debt": 0.3, "equity": 0.0},
        {"cash": 0.5, "debt": 0.5, "equity": 0.0},
    ]


def _is_explicit_false(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) == 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no"}
    return False


def _is_explicit_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _dividend_initiation_nonpayer_signal(features: Dict[str, Any]) -> bool:
    payer_value = _feature_value(features, "capital_return.dividend_payer_flag")
    if _is_explicit_true(payer_value):
        return False
    if _is_explicit_false(payer_value):
        return True

    last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
    if last_dividend_event:
        return False

    payer_record = _feature_record(features, "capital_return.dividend_payer_flag") or {}
    missing_reason = str(payer_record.get("missing_reason") or "").strip().lower()
    quality_flags = {
        str(flag).strip().lower()
        for flag in (payer_record.get("quality_flags") or [])
        if flag is not None
    }
    return bool(
        missing_reason == "unavailable"
        or "event_history_unavailable" in quality_flags
        or "dividend_event_schema_incomplete" in quality_flags
        or "dividend_fact_fallback_no_positive_values" in quality_flags
        or "dividend_fact_fallback_missing_date" in quality_flags
    )


def _product_params(spec: Dict[str, Any], max_variants: int = 96) -> List[Dict[str, Any]]:
    if not spec:
        return [{}]
    keys = sorted(spec.keys())
    values: List[List[Any]] = []
    for k in keys:
        v = spec[k]
        if isinstance(v, list):
            values.append(v if v else [None])
        else:
            values.append([v])
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*values):
        out.append({k: combo[i] for i, k in enumerate(keys) if combo[i] is not None})
        if len(out) >= max_variants:
            break
    return out


def _numeric_value_grid(parameter_name: str, parameter_def: Dict[str, Any], *, max_anchor: float) -> List[float]:
    minimum = _to_float(parameter_def.get("min"), 0.0) or 0.0
    maximum = _to_float(parameter_def.get("max"), None)
    unit = str(parameter_def.get("unit", "") or "").lower()

    if unit == "years" or parameter_name in {"tenor_years", "new_tenor_years"}:
        anchors = [3.0, 5.0, 7.0]
    elif unit == "months" or parameter_name.endswith("_months"):
        anchors = [6.0, 12.0, 24.0]
    elif parameter_name == "leverage_post_close":
        anchors = [2.0, 2.5, 3.0]
    else:
        anchors = [max(minimum, max_anchor * 0.1), max(minimum, max_anchor * 0.2), max(minimum, max_anchor * 0.35)]

    out: List[float] = []
    for anchor in anchors:
        value = max(minimum, float(anchor))
        if maximum is not None:
            value = min(value, maximum)
        out.append(_round_num(value))
    return sorted(set(out))


def _candidate_variant_sort_key(candidate: Dict[str, Any]) -> Any:
    params = dict(candidate.get("parameters", {}) or {})
    funding_mix = params.get("funding_mix")
    debt_share = None
    if isinstance(funding_mix, dict):
        debt_share = _to_float(funding_mix.get("debt"), 0.0)
    generation_confidence = _to_float(candidate.get("generation_confidence"), 0.0) or 0.0

    size_pct = _to_float(params.get("size_pct_market_cap"))
    size_abs = _to_float(params.get("size_absolute_usd"))
    initial_yield = _to_float(params.get("initial_yield_pct"))
    percent_change = _to_float(params.get("percent_change"))

    return (
        str(candidate.get("action_id", "")),
        -generation_confidence,
        0 if size_pct is not None else 1,
        float(size_pct) if size_pct is not None else float("inf"),
        0 if size_abs is not None else 1,
        float(size_abs) if size_abs is not None else float("inf"),
        0 if initial_yield is not None else 1,
        float(initial_yield) if initial_yield is not None else float("inf"),
        0 if percent_change is not None else 1,
        abs(float(percent_change)) if percent_change is not None else float("inf"),
        float(debt_share) if debt_share is not None else 0.0,
        str(candidate.get("generation_source", "")),
        str(candidate.get("candidate_signature", "")),
    )


@dataclass
class Precondition:
    feature_name: str
    assumed_relation: str
    value: Any
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RationaleReference:
    reference_type: str
    reference_id: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionCandidateDraft:
    candidate_id: str
    run_id: str
    action_type: str
    action_subtype: str
    action_id: str
    parameters: Dict[str, Any]
    assumed_preconditions: List[Precondition]
    generation_source: str
    rationale_refs: List[RationaleReference]
    generation_confidence: float
    created_at: str
    candidate_signature: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["assumed_preconditions"] = [p.to_dict() for p in self.assumed_preconditions]
        out["rationale_refs"] = [r.to_dict() for r in self.rationale_refs]
        if not out["params"]:
            out["params"] = dict(self.parameters)
        return out


@dataclass
class PlaybookTemplate:
    playbook_id: str
    label: str
    trigger_conditions: List[str]
    action_sequence_template: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PlaybookRegistry:
    def __init__(self, templates: Sequence[PlaybookTemplate]) -> None:
        self.templates = list(templates)
        self._by_id = {x.playbook_id: x for x in self.templates}

    def get(self, playbook_id: str) -> Optional[PlaybookTemplate]:
        return self._by_id.get(playbook_id)

    @classmethod
    def default(cls) -> "PlaybookRegistry":
        return cls(
            [
                PlaybookTemplate(
                    playbook_id="deleveraging_playbook",
                    label="Deleveraging Playbook",
                    trigger_conditions=["capital_structure.net_leverage high OR rating pressure high"],
                    action_sequence_template=[
                        "portfolio.divestiture_partial",
                        "portfolio.divestiture_full",
                        "capital_structure.refinancing",
                        "capital_structure.equity_issuance",
                        "restructuring.working_capital_program",
                    ],
                ),
                PlaybookTemplate(
                    playbook_id="simplification_playbook",
                    label="Simplification Playbook",
                    trigger_conditions=["segment_count high AND conglomerate discount signal"],
                    action_sequence_template=[
                        "portfolio.spin_off",
                        "portfolio.divestiture_partial",
                        "portfolio.divestiture_full",
                        "governance.capital_allocation_policy_reset",
                    ],
                ),
                PlaybookTemplate(
                    playbook_id="growth_substitution_playbook",
                    label="Growth Substitution Playbook",
                    trigger_conditions=["organic growth weak AND balance sheet capacity available"],
                    action_sequence_template=[
                        "mna.tuck_in_acquisition",
                        "mna.platform_acquisition",
                    ],
                ),
            ]
        )


class CandidateGenerationEngine:
    def __init__(
        self,
        registry: ActionSchemaRegistry,
        playbooks: Optional[PlaybookRegistry] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.registry = registry
        self.playbooks = playbooks or PlaybookRegistry.default()
        base_thresholds = {
            "maturity_wall_ratio_24m": 0.20,
            "maturity_wall_liquidity_cover_24m_max": 1.0,
            "maturity_wall_vs_mcap_min": 0.05,
            "liquidity_vs_mcap": 0.05,
            "equity_backstop_credit_window_max": 0.15,
            "equity_backstop_equity_window_min": 0.45,
            "equity_backstop_relaxed_equity_window_min": 0.30,
            "equity_backstop_relaxed_drawdown_90d_max": -0.85,
            "equity_backstop_liquidity_vs_mcap_max": 0.03,
            "equity_backstop_liquidity_to_debt_max": 0.5,
            "equity_backstop_liquidity_cover_24m_max": 1.0,
            "equity_backstop_interest_coverage_max": 1.5,
            "equity_backstop_net_leverage_min": 2.5,
            "equity_backstop_distressed_nonpayer_interest_coverage_max": 0.0,
            "equity_backstop_distressed_nonpayer_public_debt_vs_mcap_min": 0.5,
            "equity_private_placement_equity_window_max": 0.1,
            "equity_private_placement_interest_coverage_max": 0.5,
            "equity_private_placement_debt_vs_mcap_min": 2.0,
            "equity_private_placement_drawdown_90d_max": -0.85,
            "equity_private_placement_volatility_30d_min": 5.0,
            "nonpayer_recap_preference_equity_window_min": 0.20,
            "nonpayer_recap_preference_total_debt_vs_mcap_min": 0.25,
            "nonpayer_recap_preference_debt_due_vs_mcap_min": 0.25,
            "nonpayer_recap_preference_net_leverage_min": 5.0,
            "nonpayer_recap_preference_interest_coverage_max": 0.0,
            "equity_runway_drawdown_90d_max": -0.85,
            "equity_runway_volatility_90d_min": 5.0,
            "equity_runway_fcf_yield_max": 0.03,
            "equity_runway_ebitda_margin_max": 0.05,
            "equity_runway_revenue_yoy_max": 0.02,
            "dividend_increase_fcf_conversion_min": 0.4,
            "dividend_increase_interest_coverage_min": 6.0,
            "dividend_increase_net_leverage_max": 1.0,
            "dividend_increase_liquidity_vs_mcap_min": 0.015,
            "dividend_increase_debt_due_vs_mcap_max": 0.01,
            "dividend_increase_fcf_conversion_strong_min": 1.0,
            "dividend_increase_interest_coverage_strong_min": 8.0,
            "dividend_increase_net_debt_vs_mcap_max": 0.35,
            "dividend_increase_cash_buffer_tolerance": 0.97,
            "dividend_increase_debt_due_vs_mcap_relaxed_max": 0.03,
            "dividend_increase_volatility_90d_relaxed_max": 5.0,
            "dividend_increase_leveraged_payer_fcf_conversion_min": 2.0,
            "dividend_increase_leveraged_payer_interest_coverage_min": 8.0,
            "dividend_increase_leveraged_payer_net_debt_vs_mcap_max": 0.10,
            "dividend_increase_leveraged_payer_cash_buffer_tolerance": 0.85,
            "dividend_increase_leveraged_payer_debt_due_vs_mcap_max": 0.05,
            "dividend_increase_leveraged_payer_net_leverage_min": 3.0,
            "dividend_increase_leveraged_payer_net_leverage_max": 6.0,
            "dividend_increase_stable_payer_fcf_conversion_min": 0.75,
            "dividend_increase_stable_payer_interest_coverage_min": 8.0,
            "dividend_increase_stable_payer_net_debt_vs_mcap_max": 0.10,
            "dividend_increase_stable_payer_debt_due_vs_mcap_max": 0.025,
            "dividend_increase_stable_payer_cash_buffer_tolerance": 0.85,
            "dividend_increase_schedule_anomaly_fcf_conversion_min": 0.75,
            "dividend_increase_schedule_anomaly_interest_coverage_min": 3.0,
            "dividend_increase_schedule_anomaly_net_debt_vs_mcap_max": 0.05,
            "dividend_increase_schedule_anomaly_debt_due_vs_total_debt_min": 1.10,
            "dividend_increase_schedule_anomaly_volatility_90d_min": 20.0,
            "dividend_increase_coverage_outlier_fcf_conversion_min": 2.0,
            "dividend_increase_coverage_outlier_interest_coverage_min": 10.0,
            "dividend_increase_coverage_outlier_total_debt_vs_mcap_min": 0.20,
            "dividend_increase_coverage_outlier_total_debt_vs_mcap_max": 0.60,
            "dividend_increase_coverage_outlier_net_debt_vs_mcap_max": 0.50,
            "dividend_increase_coverage_outlier_debt_due_vs_mcap_max": 0.06,
            "dividend_increase_missing_schedule_net_leverage_min": 10.0,
            "dividend_increase_missing_schedule_interest_coverage_min": 2.0,
            "dividend_increase_missing_schedule_fcf_conversion_min": 2.0,
            "dividend_increase_missing_schedule_ebitda_margin_min": 0.015,
            "dividend_increase_missing_schedule_total_debt_vs_mcap_max": 0.75,
            "dividend_increase_missing_schedule_drawdown_90d_min": -0.75,
            "dividend_increase_missing_schedule_equity_window_min": 0.33,
            "dividend_increase_missing_schedule_runway_months_min": 24.0,
            "dividend_increase_missing_schedule_revenue_yoy_min": 0.0,
            "special_dividend_liquidity_vs_mcap_min": 0.04,
            "special_dividend_cash_buffer_multiple_min": 3.0,
            "special_dividend_total_debt_vs_mcap_max": 0.05,
            "special_dividend_debt_due_vs_mcap_max": 0.01,
            "special_dividend_interest_coverage_min": 1.5,
            "special_dividend_fcf_conversion_min": 0.75,
            "special_dividend_ebitda_margin_min": 0.0,
            "special_dividend_leveraged_payer_liquidity_vs_mcap_min": 0.02,
            "special_dividend_leveraged_payer_cash_buffer_multiple_min": 2.5,
            "special_dividend_leveraged_payer_net_leverage_min": 6.0,
            "special_dividend_leveraged_payer_interest_coverage_min": 20.0,
            "special_dividend_leveraged_payer_fcf_conversion_min": 2.0,
            "special_dividend_leveraged_payer_net_debt_vs_mcap_max": 0.15,
            "special_dividend_leveraged_payer_debt_due_vs_mcap_max": 0.01,
            "buyback_share_count_trend_max": -0.01,
            "buyback_existing_payer_liquidity_vs_mcap_override_min": 0.20,
            "capital_return_net_cash_maturity_override_gross_leverage_max": 1.0,
            "capital_return_net_cash_maturity_override_total_debt_vs_mcap_max": 0.10,
            "capital_return_net_cash_maturity_override_interest_coverage_min": 4.0,
            "capital_return_net_cash_liquidity_vs_mcap_min": 0.010,
            "capital_return_net_cash_fcf_yield_min": 0.005,
            "capital_return_missing_market_cap_dividend_initiate_interest_coverage_min": 4.0,
            "capital_return_missing_market_cap_dividend_initiate_relaxed_interest_coverage_min": 2.5,
            "capital_return_missing_market_cap_dividend_initiate_maturity_wall_ratio_24m_max": 0.20,
            "capital_return_missing_market_cap_dividend_initiate_liquidity_min_usd": 100_000_000.0,
            "capital_return_missing_market_cap_dividend_initiate_liquidity_cover_24m_min": 1.25,
            "capital_return_missing_market_cap_dividend_initiate_cash_buffer_multiple_min": 5.0,
            "capital_return_coverage_supported_dividend_initiate_liquidity_min_usd": 500_000_000.0,
            "capital_return_coverage_supported_dividend_initiate_interest_coverage_min": 10.0,
            "capital_return_coverage_supported_dividend_initiate_liquidity_cover_24m_min": 1.5,
            "capital_return_coverage_supported_dividend_initiate_maturity_wall_ratio_24m_max": 0.15,
            "capital_return_coverage_supported_dividend_initiate_total_debt_vs_mcap_max": 0.35,
            "capital_return_coverage_supported_dividend_initiate_net_debt_vs_mcap_max": 0.25,
            "capital_return_coverage_supported_dividend_initiate_equity_window_min": 0.5,
            "capital_return_debt_bearing_dividend_initiate_liquidity_min_usd": 250_000_000.0,
            "capital_return_debt_bearing_dividend_initiate_interest_coverage_min": 4.0,
            "capital_return_debt_bearing_dividend_initiate_fcf_conversion_min": 0.20,
            "capital_return_debt_bearing_dividend_initiate_net_leverage_max": 3.5,
            "capital_return_debt_bearing_dividend_initiate_total_debt_vs_mcap_max": 0.75,
            "capital_return_debt_bearing_dividend_initiate_net_debt_vs_mcap_max": 0.60,
            "capital_return_debt_bearing_dividend_initiate_maturity_wall_ratio_24m_max": 0.10,
            "capital_return_debt_bearing_dividend_initiate_debt_due_vs_mcap_max": 0.01,
            "capital_return_debt_bearing_dividend_initiate_equity_window_min": 0.5,
            "expectations_coverage_min": 5.0,
            "expectations_revision_negative_max": -0.05,
            "expectations_revision_positive_min": 0.02,
            "payer_anomaly_total_debt_vs_mcap_max": 0.05,
            "payer_anomaly_net_debt_vs_mcap_max": 0.05,
            "payer_anomaly_debt_due_vs_total_debt_min": 10.0,
            "payer_anomaly_maturity_ratio_24m_min": 10.0,
            "payer_anomaly_revenue_yoy_abs_min": 5.0,
            "payer_anomaly_ebitda_margin_abs_min": 1.0,
            "financing_anomaly_debt_due_vs_total_debt_min": 1.15,
            "financing_anomaly_total_debt_vs_mcap_max": 0.15,
            "financing_anomaly_net_leverage_max": 1.8,
            "financing_anomaly_interest_coverage_min": 5.0,
            "financing_anomaly_dividend_fcf_conversion_min": 0.6,
            "financing_anomaly_drawdown_90d_max": -0.55,
            "financing_anomaly_volatility_90d_min": 1.0,
            "dividend_increase_continuity_runway_months_min": 36.0,
            "dividend_increase_continuity_fcf_conversion_min": 0.35,
            "dividend_increase_continuity_interest_coverage_min": 8.0,
            "dividend_increase_continuity_net_leverage_max": 2.5,
            "dividend_increase_continuity_total_debt_vs_mcap_max": 0.10,
            "dividend_increase_continuity_drawdown_90d_max": -0.70,
            "dividend_increase_continuity_volatility_90d_min": 2.0,
            "dividend_increase_large_cap_market_cap_min": 50_000_000_000.0,
            "dividend_increase_large_cap_fcf_conversion_min": 0.4,
            "dividend_increase_large_cap_interest_coverage_min": 6.0,
            "dividend_increase_large_cap_total_debt_vs_mcap_max": 0.05,
            "dividend_increase_large_cap_net_debt_vs_mcap_max": 0.03,
            "dividend_increase_large_cap_debt_due_vs_mcap_max": 0.015,
            "dividend_increase_large_cap_liquidity_vs_mcap_min": 0.005,
            "dividend_increase_large_cap_liquidity_cover_24m_min": 0.5,
            "dividend_increase_large_cap_maturity_wall_ratio_24m_max": 0.5,
            "dividend_increase_large_cap_equity_window_min": 0.55,
            "dividend_increase_large_cap_drawdown_90d_max": -0.8,
            "dividend_increase_large_cap_volatility_90d_min": 1.0,
            "dividend_increase_no_maturity_runway_months_min": 36.0,
            "dividend_increase_no_maturity_fcf_conversion_min": 0.15,
            "dividend_increase_no_maturity_interest_coverage_min": 2.0,
            "dividend_increase_no_maturity_total_debt_vs_mcap_max": 0.35,
            "dividend_increase_no_maturity_net_debt_vs_mcap_max": 0.35,
            "dividend_increase_no_maturity_ebitda_margin_min": 0.08,
            "dividend_increase_no_maturity_revenue_yoy_min": -0.35,
            "dividend_increase_no_maturity_equity_window_min": 0.55,
            "dividend_increase_no_maturity_drawdown_90d_max": -0.7,
            "dividend_increase_no_maturity_volatility_90d_min": 1.0,
            "dividend_increase_mild_maturity_interest_coverage_min": 12.0,
            "dividend_increase_mild_maturity_fcf_conversion_min": 0.4,
            "dividend_increase_mild_maturity_total_debt_vs_mcap_max": 0.10,
            "dividend_increase_mild_maturity_net_debt_vs_mcap_max": 0.10,
            "dividend_increase_mild_maturity_debt_due_vs_mcap_max": 0.02,
            "dividend_increase_mild_maturity_maturity_wall_ratio_24m_max": 0.30,
            "dividend_increase_mild_maturity_liquidity_cover_24m_min": 2.0,
            "dividend_increase_high_coverage_interest_coverage_min": 15.0,
            "dividend_increase_high_coverage_net_leverage_max": 2.0,
            "dividend_increase_high_coverage_total_debt_vs_mcap_max": 0.10,
            "dividend_increase_high_coverage_debt_due_vs_mcap_max": 0.01,
            "dividend_increase_high_coverage_revenue_cagr_3y_min": -0.20,
            "dividend_increase_high_coverage_fcf_conversion_min": 0.15,
            "dividend_increase_high_coverage_megacap_market_cap_min": 25_000_000_000.0,
            "dividend_increase_high_coverage_megacap_interest_coverage_min": 25.0,
            "dividend_increase_high_coverage_megacap_net_debt_vs_mcap_max": 0.03,
            "dividend_increase_high_coverage_megacap_fcf_conversion_min": -0.50,
            "dividend_increase_coverage_supported_market_cap_min": 1_000_000_000.0,
            "dividend_increase_coverage_supported_equity_window_min": 0.60,
            "dividend_increase_coverage_supported_interest_coverage_min": 7.5,
            "dividend_increase_coverage_supported_fcf_conversion_min": 0.05,
            "dividend_increase_coverage_supported_net_leverage_min": 2.5,
            "dividend_increase_coverage_supported_net_leverage_max": 6.0,
            "dividend_increase_coverage_supported_total_debt_vs_mcap_max": 1.75,
            "dividend_increase_coverage_supported_debt_due_vs_mcap_max": 0.20,
            "dividend_increase_coverage_supported_liquidity_cover_24m_min": 0.05,
            "dividend_increase_coverage_supported_ebitda_margin_min": 0.04,
            "dividend_increase_coverage_supported_revenue_cagr_3y_min": -0.35,
            "dividend_increase_coverage_supported_override_interest_coverage_min": 20.0,
            "dividend_increase_coverage_supported_override_ebitda_margin_min": 0.15,
            "dividend_increase_sparse_market_cap_min": 1_000_000_000.0,
            "dividend_increase_sparse_equity_window_min": 0.60,
            "dividend_increase_sparse_interest_coverage_min": 7.5,
            "dividend_increase_sparse_net_leverage_min": 3.0,
            "dividend_increase_sparse_net_leverage_max": 6.0,
            "dividend_increase_sparse_total_debt_vs_mcap_max": 1.75,
            "dividend_increase_sparse_debt_due_vs_mcap_max": 0.05,
            "dividend_increase_sparse_liquidity_cover_24m_min": 0.25,
            "dividend_increase_sparse_drawdown_90d_max": -0.20,
            "dividend_increase_coverage_gap_market_cap_min": 1_000_000_000.0,
            "dividend_increase_coverage_gap_equity_window_min": 0.60,
            "dividend_increase_coverage_gap_fcf_conversion_min": 0.20,
            "dividend_increase_coverage_gap_net_leverage_min": 2.5,
            "dividend_increase_coverage_gap_net_leverage_max": 5.0,
            "dividend_increase_coverage_gap_total_debt_vs_mcap_max": 1.0,
            "dividend_increase_coverage_gap_debt_due_vs_mcap_max": 0.20,
            "dividend_increase_coverage_gap_liquidity_cover_24m_min": 0.05,
            "dividend_increase_coverage_gap_ebitda_margin_min": 0.04,
            "dividend_increase_coverage_gap_drawdown_90d_max": -0.20,
            "dividend_increase_balance_sheet_light_market_cap_min": 1_000_000_000.0,
            "dividend_increase_balance_sheet_light_equity_window_min": 0.50,
            "dividend_increase_balance_sheet_light_total_debt_vs_mcap_max": 0.25,
            "dividend_increase_balance_sheet_light_net_leverage_max": 2.0,
            "dividend_increase_balance_sheet_light_interest_coverage_min": 2.0,
            "dividend_increase_balance_sheet_light_fcf_conversion_min": 0.25,
            "dividend_increase_balance_sheet_light_ebitda_margin_min": 0.01,
            "dividend_increase_balance_sheet_light_liquidity_vs_mcap_min": 0.02,
            "dividend_increase_balance_sheet_light_debt_due_vs_mcap_max": 0.20,
            "dividend_increase_balance_sheet_light_drawdown_90d_max": -0.25,
            "dividend_increase_liquidity_supported_market_cap_min": 1_000_000_000.0,
            "dividend_increase_liquidity_supported_equity_window_min": 0.60,
            "dividend_increase_liquidity_supported_interest_coverage_min": 7.5,
            "dividend_increase_liquidity_supported_net_leverage_min": 6.0,
            "dividend_increase_liquidity_supported_net_leverage_max": 9.0,
            "dividend_increase_liquidity_supported_total_debt_vs_mcap_max": 2.5,
            "dividend_increase_liquidity_supported_liquidity_vs_mcap_min": 0.20,
            "dividend_increase_liquidity_supported_ebitda_margin_min": 0.08,
            "dividend_increase_liquidity_supported_drawdown_90d_max": -0.20,
            "dividend_increase_liquidity_supported_debt_due_vs_mcap_max": 0.05,
            "dividend_increase_liquidity_supported_liquidity_cover_24m_min": 0.25,
            "market_shutdown_regular_payer_equity_window_max": 0.15,
            "market_shutdown_regular_payer_drawdown_90d_max": -0.80,
            "market_shutdown_regular_payer_total_debt_vs_mcap_min": 0.50,
            "market_shutdown_regular_payer_debt_due_vs_mcap_min": 0.10,
            "strategic_regular_payer_recap_equity_window_min": 0.25,
            "strategic_regular_payer_recap_total_debt_vs_mcap_min": 0.25,
            "strategic_regular_payer_recap_debt_due_vs_mcap_min": 0.10,
            "strategic_regular_payer_recap_net_leverage_min": 3.5,
            "strategic_regular_payer_recap_interest_coverage_max": 1.0,
            "buyback_regular_payer_recap_equity_window_min": 0.45,
            "buyback_regular_payer_recap_total_debt_vs_mcap_min": 0.25,
            "buyback_regular_payer_recap_debt_due_vs_mcap_min": 0.05,
            "buyback_regular_payer_recap_interest_coverage_min": 6.0,
            "buyback_regular_payer_recap_fcf_conversion_min": 0.75,
            "buyback_regular_payer_recap_runway_months_min": 24.0,
            "strategic_nonpayer_recap_equity_window_min": 0.10,
            "strategic_nonpayer_recap_total_debt_vs_mcap_min": 0.015,
            "strategic_nonpayer_recap_debt_due_vs_mcap_min": 0.005,
            "strategic_nonpayer_recap_interest_coverage_min": 6.0,
            "strategic_nonpayer_recap_fcf_conversion_min": 0.75,
            "strategic_nonpayer_recap_runway_months_min": 24.0,
            "sparse_reset_recap_equity_window_min": 0.20,
            "sparse_reset_recap_total_debt_vs_mcap_min": 0.24,
            "sparse_reset_recap_total_debt_vs_mcap_strong_min": 0.40,
            "sparse_reset_recap_net_leverage_min": 2.5,
            "sparse_reset_recap_interest_coverage_min": 3.0,
            "sparse_reset_recap_credit_window_max": 0.65,
            "sparse_reset_recap_drawdown_90d_max": -0.45,
            "dividend_cut_interest_coverage_max": 6.0,
            "dividend_cut_net_leverage_min": 4.5,
            "dividend_cut_debt_vs_mcap_max": 0.1,
            "dividend_cut_liquidity_cover_24m_max": 0.3,
            "dividend_cut_debt_due_vs_mcap_min": 0.05,
            "dividend_cut_maturity_wall_ratio_24m_min": 0.75,
            "dividend_cut_debt_due_vs_total_debt_min": 0.75,
            "dividend_cut_balance_sheet_pressure_net_leverage_min": 10.0,
            "dividend_cut_balance_sheet_pressure_interest_coverage_min": 8.0,
            "dividend_cut_balance_sheet_pressure_fcf_conversion_min": 2.0,
            "dividend_cut_balance_sheet_pressure_net_debt_vs_mcap_min": 0.2,
            "dividend_cut_balance_sheet_pressure_total_debt_vs_mcap_min": 0.2,
            "dividend_cut_balance_sheet_pressure_debt_due_vs_mcap_max": 0.01,
            "dividend_cut_balance_sheet_pressure_drawdown_90d_max": -0.55,
            "dividend_cut_preemptive_total_debt_vs_mcap_min": 0.50,
            "dividend_cut_preemptive_net_leverage_min": 5.0,
            "dividend_cut_preemptive_interest_coverage_min": 10.0,
            "dividend_cut_preemptive_fcf_conversion_max": 0.0,
            "dividend_cut_preemptive_debt_due_vs_mcap_min": 0.20,
            "dividend_cut_preemptive_maturity_wall_ratio_24m_min": 0.25,
            "dividend_cut_preemptive_drawdown_90d_max": -0.10,
            "dividend_cut_buyback_reset_interest_coverage_max": 8.0,
            "dividend_cut_buyback_reset_fcf_conversion_max": -0.25,
            "dividend_cut_buyback_reset_total_debt_vs_mcap_min": 0.10,
            "dividend_cut_buyback_reset_total_debt_vs_mcap_max": 0.30,
            "dividend_cut_buyback_reset_debt_due_vs_mcap_max": 0.10,
            "dividend_cut_buyback_reset_maturity_wall_ratio_24m_max": 0.25,
            "margin_percentile_peers": 0.40,
            "market_share_percentile": 0.40,
            "consolidation_wave": 0.60,
            "segment_count_high": 2.0,
            "net_leverage_high": 3.5,
            "capital_return_interest_coverage_min": 2.5,
            "capital_return_liquidity_cover_24m_min": 1.25,
            "mna_interest_coverage_min": 3.0,
            "mna_net_leverage_max": 3.0,
            "mna_liquidity_vs_mcap_min": 0.03,
            "mna_liquidity_cover_24m_min": 1.5,
            "mna_market_window_min": 0.55,
            "working_capital_runway_months_max": 24.0,
            "working_capital_fcf_conversion_max": 0.85,
            "working_capital_liquidity_cover_24m_max": 0.75,
        }
        if thresholds:
            base_thresholds.update(thresholds)
        self.thresholds = base_thresholds

    def generate_candidate_set(
        self,
        run: RecommendationRun,
        state_snapshot: Dict[str, Any],
        action_ids: Optional[Sequence[str]] = None,
        action_type: Optional[str] = None,
        max_candidates: int = 1500,
        min_candidates_target: int = 0,
        strict_evidence: bool = False,
        extracted_facts: Optional[List[Dict[str, Any]]] = None,
        event_store: Optional[List[Dict[str, Any]]] = None,
        peer_set: Optional[Dict[str, Any]] = None,
        llm_proposals: Optional[List[Dict[str, Any]]] = None,
        llm_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        features = feature_view_from_snapshot(state_snapshot, view_name="candidate_generation")
        regime = state_snapshot.get("regime", {}) if isinstance(state_snapshot, dict) else {}
        peer_ctx = peer_set or (state_snapshot.get("peer_set") if isinstance(state_snapshot, dict) else {}) or {}

        known_segments = self._extract_segments(state_snapshot, extracted_facts)
        available_features = sorted(features.keys()) if isinstance(features, dict) else []
        available_evidence = self._infer_evidence_classes(state_snapshot, extracted_facts, event_store)
        constraints = self._constraint_tokens(run, state_snapshot)

        if action_ids:
            raw_candidates = self._generate_explicit_action_candidates(run, action_ids)
            playbook_triggered: List[str] = []
            llm_trace = {}
        else:
            det = self._deterministic_generators(run, features, regime, peer_ctx, known_segments)
            pb, playbook_triggered = self._playbook_instantiation(run, features, regime, known_segments)
            llm, llm_trace = self._validate_llm_proposals(run, llm_proposals, llm_metadata)
            raw_candidates = det + pb + llm

            if action_type:
                raw_candidates = [c for c in raw_candidates if c.get("action_type") == str(action_type)]
            if action_ids:
                wanted = {str(x) for x in action_ids}
                raw_candidates = [c for c in raw_candidates if c.get("action_id") in wanted]

        validated: List[Dict[str, Any]] = []
        for c in raw_candidates:
            schema = self.registry.get_action(str(c.get("action_id", ""))) or {}
            prereq = schema.get("feasibility_prerequisites", {}) if isinstance(schema, dict) else {}
            required_features = set(prereq.get("required_features", []) if isinstance(prereq, dict) else [])
            evidence_req = schema.get("minimum_evidence_requirements", {}) if isinstance(schema, dict) else {}
            required_classes = set(
                evidence_req.get("minimum_classes_required", []) if isinstance(evidence_req, dict) else []
            )
            validation = self.registry.validate_candidate(
                {
                    "action_id": c["action_id"],
                    "parameters": c["parameters"],
                    "known_segments": known_segments,
                    # Candidate generation is schema-first: feasibility/evidence gating happens downstream.
                    "available_features": sorted(set(available_features) | required_features),
                    "available_evidence_classes": sorted(set(available_evidence) | required_classes),
                    "constraints": [],
                },
                strict_evidence=False,
            )
            if not validation.valid:
                continue
            c["_validation_warnings"] = list(validation.warnings)
            validated.append(c)

        deduped = self._dedupe_candidates(validated)

        # Coverage expansion: if user did not pin action filters and we are below target,
        # broaden ontology instantiation with lower-confidence exploratory variants.
        if (not action_ids) and (not action_type):
            target = int(max(0, min_candidates_target))
            hard_cap = int(max(0, max_candidates))
            if target > 0 and len(deduped) < target:
                expanded = self._coverage_expand_candidates(
                    run=run,
                    features=features,
                    known_segments=known_segments,
                    existing=deduped,
                    target=target,
                    hard_cap=hard_cap,
                )
                deduped = self._dedupe_candidates(deduped + expanded)

        # Keep candidate caps deterministic, but prefer the more conservative
        # parameterizations before hash-order truncation can discard them.
        deduped.sort(key=_candidate_variant_sort_key)
        if max_candidates > 0:
            deduped = deduped[: int(max_candidates)]

        drafts = [self._to_draft(run.run_id, c) for c in deduped]
        by_type: Dict[str, List[str]] = {}
        by_action: Dict[str, List[str]] = {}
        for d in drafts:
            by_type.setdefault(d.action_type, []).append(d.candidate_id)
            by_action.setdefault(d.action_id, []).append(d.candidate_id)

        out = {
            "run_id": run.run_id,
            "generated_at": _now_iso(),
            "registry_version": self.registry.version,
            "snapshot_hash": run.frozen_state.snapshot_hash,
            "candidates": [d.to_dict() for d in drafts],
            "index_by_action_type": by_type,
            "index_by_action_id": by_action,
            "counts": {
                "raw": len(raw_candidates),
                "validated": len(validated),
                "deduped": len(drafts),
            },
            "generation_context": {
                "playbooks_triggered": playbook_triggered,
                "strict_evidence": bool(strict_evidence),
                "min_candidates_target": int(max(0, min_candidates_target)),
            },
        }
        if llm_trace:
            out["llm_trace"] = llm_trace
        return out

    def _coverage_expand_candidates(
        self,
        run: RecommendationRun,
        features: Dict[str, Any],
        known_segments: List[str],
        existing: List[Dict[str, Any]],
        target: int,
        hard_cap: int,
    ) -> List[Dict[str, Any]]:
        existing_sigs = {str(x.get("candidate_signature", "")) for x in existing}
        out: List[Dict[str, Any]] = []
        budget = hard_cap if hard_cap > 0 else max(target, 5000)
        schemas = sorted(self.registry.actions, key=lambda x: str(x.get("action_id", "")))
        total = len(schemas)
        for i, schema in enumerate(schemas):
            if len(existing) + len(out) >= target:
                break
            if len(existing) + len(out) >= budget:
                break

            deficit = max(0, target - (len(existing) + len(out)))
            remaining = max(1, total - i)
            per_schema = max(12, min(96, int((deficit / remaining) * 4)))
            params_list = self.generate_parameter_variants(
                schema,
                features,
                known_segments=known_segments,
                max_variants=per_schema,
                include_optional=True,
            )
            for params in params_list:
                cand = self._build_candidate_raw(
                    run_id=run.run_id,
                    action_schema=schema,
                    parameters=params,
                    generation_source="deterministic_rule",
                    rationale_refs=[
                        {
                            "reference_type": "state_feature",
                            "reference_id": "coverage_expansion",
                            "explanation": "Expanded ontology coverage to reach minimum candidate target.",
                        }
                    ],
                    trigger_strength=0.25,
                    playbook_relevance=0.25,
                )
                sig = str(cand.get("candidate_signature", ""))
                if sig in existing_sigs:
                    continue
                existing_sigs.add(sig)
                out.append(cand)
                if len(existing) + len(out) >= target:
                    break
                if len(existing) + len(out) >= budget:
                    break
        return out

    def _generate_explicit_action_candidates(
        self,
        run: RecommendationRun,
        action_ids: Sequence[str],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for aid in action_ids:
            schema = self.registry.get_action(str(aid))
            if schema is None:
                continue
            params = self.generate_parameter_variants(schema, {}, max_variants=1)[0]
            out.append(
                self._build_candidate_raw(
                    run_id=run.run_id,
                    action_schema=schema,
                    parameters=params,
                    generation_source="deterministic_rule",
                    rationale_refs=[
                        {
                            "reference_type": "state_feature",
                            "reference_id": "explicit_action_request",
                            "explanation": "Action was explicitly requested for this run.",
                        }
                    ],
                    trigger_strength=0.65,
                    playbook_relevance=0.6,
                )
            )
        return out

    def _deterministic_generators(
        self,
        run: RecommendationRun,
        features: Dict[str, Any],
        regime: Dict[str, Any],
        peer_ctx: Dict[str, Any],
        known_segments: List[str],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        out.extend(self._gen_maturity_wall(run, features))
        out.extend(self._gen_dividend_policy_actions(run, features))
        out.extend(self._gen_dividend_missing_schedule_actions(run, features))
        out.extend(self._gen_dividend_coverage_supported_actions(run, features))
        out.extend(self._gen_dividend_balance_sheet_light_actions(run, features))
        out.extend(self._gen_dividend_continuity_actions(run, features))
        out.extend(self._gen_special_dividend_actions(run, features))
        out.extend(self._gen_dividend_cut_actions(run, features))
        out.extend(self._gen_equity_backstop(run, features))
        out.extend(self._gen_liquidity_excess(run, features))
        out.extend(self._gen_undervaluation(run, features))
        out.extend(self._gen_conglomerate_discount(run, features, known_segments))
        out.extend(self._gen_subscale_position(run, features))
        out.extend(self._gen_margin_underperformance(run, features))
        return out

    def _playbook_instantiation(
        self,
        run: RecommendationRun,
        features: Dict[str, Any],
        regime: Dict[str, Any],
        known_segments: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        triggered: List[str] = []
        out: List[Dict[str, Any]] = []

        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), 0.0) or 0.0
        rating_state = _feature_value(features, "capital_structure.rating_state")
        rating_pressure = isinstance(rating_state, dict) and str(rating_state.get("outlook", "")).lower() == "neg"
        segment_count = _to_float(_feature_value(features, "strategic.segment_count"), 0.0) or 0.0
        cong_disc = _to_float(_feature_value(features, "market.conglomerate_discount_signal"), 0.0) or 0.0
        growth = _to_float(_feature_value(features, "operating.revenue_cagr_3y"), None)
        liquidity = _to_float(_feature_value(features, "liquidity.available_for_actions"), 0.0) or 0.0

        leverage_stress = net_leverage > self.thresholds["net_leverage_high"]
        payout_anomaly_profile = self._missing_schedule_dividend_increase_profile(features)
        financing_anomaly_override = self._regular_payer_financing_anomaly_override(features)
        supportive_dividend_profile = self._supportive_regular_payer_dividend_increase_profile(features)
        balance_sheet_dividend_cut_profile = self._dividend_cut_balance_sheet_pressure_profile(features)
        coverage_supported_dividend_initiate = bool(
            self._coverage_supported_dividend_initiate_override_profile(features)
            and not self._has_buyback_specific_support(features)
        )
        missing_market_cap_dividend_initiate = self._missing_market_cap_dividend_initiate_override_profile(features)
        if rating_pressure or (
            leverage_stress
            and not coverage_supported_dividend_initiate
            and not missing_market_cap_dividend_initiate
            and not self._healthy_dividend_increase_profile(features)
            and not supportive_dividend_profile
            and not payout_anomaly_profile
            and not financing_anomaly_override
            and not balance_sheet_dividend_cut_profile
        ):
            strategic_regular_payer_recap = self._strategic_regular_payer_recap_profile(features)
            buyback_regular_payer_recap = self._buyback_regular_payer_recap_profile(features)
            strategic_nonpayer_recap = self._strategic_nonpayer_recap_profile(features)
            triggered.append("deleveraging_playbook")
            deleveraging_actions = {
                "portfolio.divestiture_partial": {"percent_divested": [0.2, 0.4, 0.6]},
                "portfolio.divestiture_full": {"percent_divested": 1.0},
                "capital_structure.refinancing": {"new_tenor_years": [3, 5, 7]},
                "capital_structure.equity_issuance": {"use_of_proceeds": "deleveraging"},
                "restructuring.working_capital_program": {"horizon_months": [6, 12]},
            }
            if strategic_regular_payer_recap or buyback_regular_payer_recap or strategic_nonpayer_recap:
                deleveraging_actions.pop("capital_structure.refinancing", None)
            out.extend(
                self._instantiate_playbook_actions(
                    run,
                    "deleveraging_playbook",
                    deleveraging_actions,
                    features,
                    known_segments,
                    trigger_strength=0.85,
                )
            )

        if segment_count > self.thresholds["segment_count_high"] and cong_disc > 0:
            triggered.append("simplification_playbook")
            out.extend(
                self._instantiate_playbook_actions(
                    run,
                    "simplification_playbook",
                    {
                        "portfolio.spin_off": {},
                        "portfolio.divestiture_partial": {"percent_divested": [0.2, 0.4]},
                        "portfolio.divestiture_full": {"percent_divested": 1.0},
                        "governance.capital_allocation_policy_reset": {},
                    },
                    features,
                    known_segments,
                    trigger_strength=0.8,
                )
            )

        if (
            growth is not None
            and growth <= 0.02
            and liquidity > 0
            and not self._mna_blocked_by_financing_or_capacity(features)
            and not self._healthy_dividend_increase_profile(features)
        ):
            triggered.append("growth_substitution_playbook")
            out.extend(
                self._instantiate_playbook_actions(
                    run,
                    "growth_substitution_playbook",
                    {
                        "mna.tuck_in_acquisition": {"target_size_pct_ev": [0.05, 0.1, 0.2]},
                        "mna.platform_acquisition": {"target_size_pct_ev": [0.1, 0.2, 0.35]},
                    },
                    features,
                    known_segments,
                    trigger_strength=0.72,
                )
            )

        return out, triggered

    def _instantiate_playbook_actions(
        self,
        run: RecommendationRun,
        playbook_id: str,
        action_overrides: Dict[str, Dict[str, Any]],
        features: Dict[str, Any],
        known_segments: List[str],
        trigger_strength: float,
    ) -> List[Dict[str, Any]]:
        pb = self.playbooks.get(playbook_id)
        if pb is None:
            return []

        out: List[Dict[str, Any]] = []
        for aid in pb.action_sequence_template:
            if (
                playbook_id == "deleveraging_playbook"
                and aid == "restructuring.working_capital_program"
                and not self._should_include_working_capital_program(features)
            ):
                continue
            if (
                playbook_id == "deleveraging_playbook"
                and aid == "capital_structure.refinancing"
                and (
                    self._strategic_regular_payer_recap_profile(features)
                    or self._buyback_regular_payer_recap_profile(features)
                    or self._strategic_nonpayer_recap_profile(features)
                )
            ):
                continue
            if playbook_id == "growth_substitution_playbook" and self._mna_blocked_by_financing_or_capacity(features):
                continue
            schema = self.registry.get_action(aid)
            if schema is None:
                continue
            overrides = action_overrides.get(aid, {})
            params_list = self.generate_parameter_variants(
                schema,
                {},
                overrides=overrides,
                known_segments=known_segments,
                max_variants=16,
            )
            for params in params_list:
                out.append(
                    self._build_candidate_raw(
                        run_id=run.run_id,
                        action_schema=schema,
                        parameters=params,
                        generation_source="playbook_template",
                        rationale_refs=[
                            {
                                "reference_type": "precedent_signal",
                                "reference_id": playbook_id,
                                "explanation": f"Generated from playbook template: {pb.label}.",
                            }
                        ],
                        trigger_strength=trigger_strength,
                        playbook_relevance=1.0,
                    )
                )
        return out

    def _should_include_working_capital_program(self, features: Dict[str, Any]) -> bool:
        runway_months = _to_float(_feature_value(features, "liquidity.runway_months"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        debt_due_24m = _to_float(_feature_value(features, "capital_structure.debt_due_next_24m"), None)
        if debt_due_24m is None:
            maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
            total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
            if maturity_ratio is not None and total_debt is not None:
                debt_due_24m = maturity_ratio * total_debt

        runway_stress = runway_months is not None and runway_months <= self.thresholds["working_capital_runway_months_max"]
        conversion_stress = (
            fcf_conversion is not None
            and fcf_conversion <= self.thresholds["working_capital_fcf_conversion_max"]
        )
        liquidity_cover_stress = False
        if available_for_actions is not None and debt_due_24m is not None and debt_due_24m > 0:
            liquidity_cover_stress = (
                available_for_actions / debt_due_24m
                <= self.thresholds["working_capital_liquidity_cover_24m_max"]
            )

        return bool(runway_stress or conversion_stress or liquidity_cover_stress)

    def _rating_pressure(self, features: Dict[str, Any]) -> bool:
        rating_state = _feature_value(features, "capital_structure.rating_state")
        if not isinstance(rating_state, dict):
            return False

        outlook = str(rating_state.get("outlook") or "").strip().lower()
        watchlist = str(rating_state.get("watchlist") or "").strip().lower()
        rating = str(rating_state.get("rating") or "").strip().upper()
        if outlook in {"neg", "negative"} or "neg" in watchlist:
            return True
        return rating in {"CCC+", "CCC", "CCC-", "CC", "C", "D", "SD"}

    def _debt_schedule_inconsistency_flag(self, features: Dict[str, Any]) -> bool:
        explicit = _feature_value(features, "capital_structure.debt_schedule_inconsistency_flag")
        explicit_num = _to_float(explicit, None)
        if explicit_num is not None:
            return explicit_num >= 0.5
        if _is_explicit_true(explicit):
            return True

        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        debt_schedule_total = _to_float(_feature_value(features, "capital_structure.debt_schedule_total"), None)
        schedule_vs_total_debt = _to_float(
            _feature_value(features, "capital_structure.debt_schedule_vs_total_debt"),
            None,
        )
        if schedule_vs_total_debt is not None:
            return schedule_vs_total_debt >= 1.5
        if total_debt is not None and total_debt <= 0:
            return bool(debt_schedule_total is not None and debt_schedule_total > 0)
        return bool(
            (
                debt_due_24m is not None
                and total_debt is not None
                and total_debt > 0
                and debt_due_24m >= total_debt * self.thresholds["payer_anomaly_debt_due_vs_total_debt_min"]
            )
            or (
                maturity_ratio is not None
                and maturity_ratio >= self.thresholds["payer_anomaly_maturity_ratio_24m_min"]
            )
        )

    def _existing_payer_data_anomaly_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False

        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        if market_cap is None or market_cap <= 0 or total_debt is None or total_debt < 0:
            return False

        total_debt_vs_mcap = total_debt / market_cap
        if total_debt_vs_mcap > self.thresholds["payer_anomaly_total_debt_vs_mcap_max"]:
            return False

        if net_debt is not None:
            net_debt_vs_mcap = max(net_debt, 0.0) / market_cap
            if net_debt_vs_mcap > self.thresholds["payer_anomaly_net_debt_vs_mcap_max"]:
                return False

        debt_due_24m = self._effective_debt_due_24m(features)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        debt_schedule_anomaly = self._debt_schedule_inconsistency_flag(features)
        if not debt_schedule_anomaly:
            debt_schedule_anomaly = bool(
                (
                    debt_due_24m is not None
                    and total_debt > 0
                    and debt_due_24m >= total_debt * self.thresholds["payer_anomaly_debt_due_vs_total_debt_min"]
                )
                or (
                    maturity_ratio is not None
                    and maturity_ratio >= self.thresholds["payer_anomaly_maturity_ratio_24m_min"]
                )
            )
        if not debt_schedule_anomaly:
            return False

        revenue_yoy = _to_float(_feature_value(features, "operating.revenue_yoy_last_q"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        operating_anomaly = bool(
            (
                revenue_yoy is not None
                and abs(revenue_yoy) >= self.thresholds["payer_anomaly_revenue_yoy_abs_min"]
            )
            or (
                ebitda_margin is not None
                and abs(ebitda_margin) >= self.thresholds["payer_anomaly_ebitda_margin_abs_min"]
            )
        )
        return operating_anomaly

    def _regular_payer_financing_anomaly_override(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        if (
            market_cap is None
            or market_cap <= 0
            or total_debt is None
            or total_debt <= 0
            or debt_due_24m is None
            or debt_due_24m <= 0
        ):
            return False

        debt_due_vs_total_debt = debt_due_24m / total_debt
        schedule_inconsistency_flag = self._debt_schedule_inconsistency_flag(features)
        schedule_vs_total_debt = _to_float(
            _feature_value(features, "capital_structure.debt_schedule_vs_total_debt"),
            None,
        )
        if not schedule_inconsistency_flag and schedule_vs_total_debt is not None:
            schedule_inconsistency_flag = schedule_vs_total_debt >= self.thresholds["financing_anomaly_debt_due_vs_total_debt_min"]
        if not schedule_inconsistency_flag:
            if debt_due_vs_total_debt < self.thresholds["financing_anomaly_debt_due_vs_total_debt_min"]:
                return False

        total_debt_vs_mcap = total_debt / market_cap
        if total_debt_vs_mcap > self.thresholds["financing_anomaly_total_debt_vs_mcap_max"]:
            return False

        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        volatility_90d = _to_float(_feature_value(features, "market.volatility_90d"), None)

        if (
            fcf_conversion is None
            or fcf_conversion < self.thresholds["financing_anomaly_dividend_fcf_conversion_min"]
        ):
            return False

        market_dislocation = bool(
            (drawdown_90d is not None and drawdown_90d <= self.thresholds["financing_anomaly_drawdown_90d_max"])
            or (
                volatility_90d is not None
                and volatility_90d >= self.thresholds["financing_anomaly_volatility_90d_min"]
            )
        )
        if not market_dislocation:
            return False

        strength_signals = 0
        if net_debt is not None and net_debt <= 0:
            strength_signals += 1
        if (
            net_leverage is not None
            and net_leverage <= self.thresholds["financing_anomaly_net_leverage_max"]
        ):
            strength_signals += 1
        if (
            interest_coverage is not None
            and interest_coverage >= self.thresholds["financing_anomaly_interest_coverage_min"]
        ):
            strength_signals += 1
        return strength_signals >= 2

    def _continuity_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False
        if self._healthy_dividend_increase_profile(features):
            return False
        if self._dividend_cut_profile(features):
            return False
        if self._special_dividend_profile(features):
            return False
        if self._capital_return_blocked_by_financing_stress(features):
            return False

        runway_months = _to_float(_feature_value(features, "liquidity.runway_months"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        revenue_yoy = _to_float(_feature_value(features, "operating.revenue_yoy_last_q"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        volatility_90d = _to_float(_feature_value(features, "market.volatility_90d"), None)

        if (
            runway_months is None
            or runway_months < self.thresholds["dividend_increase_continuity_runway_months_min"]
            or market_cap is None
            or market_cap <= 0
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["dividend_increase_continuity_fcf_conversion_min"]
            or ebitda_margin is None
            or ebitda_margin <= 0
            or (revenue_yoy is not None and revenue_yoy < 0)
        ):
            return False

        no_near_term_pressure = bool(
            (debt_due_24m is None or debt_due_24m <= 0)
            and (maturity_ratio is None or maturity_ratio <= 0.0)
        )
        if not no_near_term_pressure:
            return False

        market_dislocation = bool(
            (drawdown_90d is not None and drawdown_90d <= self.thresholds["dividend_increase_continuity_drawdown_90d_max"])
            or (
                volatility_90d is not None
                and volatility_90d >= self.thresholds["dividend_increase_continuity_volatility_90d_min"]
            )
        )
        if not market_dislocation:
            return False

        total_debt_vs_mcap = 0.0
        if total_debt is not None and total_debt > 0:
            total_debt_vs_mcap = total_debt / market_cap
        if total_debt_vs_mcap > self.thresholds["dividend_increase_continuity_total_debt_vs_mcap_max"]:
            return False

        leverage_ok = bool(net_debt is not None and net_debt <= 0) or bool(
            net_leverage is not None
            and net_leverage <= self.thresholds["dividend_increase_continuity_net_leverage_max"]
        )
        coverage_ok = (
            interest_coverage is not None
            and interest_coverage >= self.thresholds["dividend_increase_continuity_interest_coverage_min"]
        )
        return bool(leverage_ok and coverage_ok)

    def _stable_debt_bearing_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        revenue_yoy = _to_float(_feature_value(features, "operating.revenue_yoy_last_q"), None)
        if (
            market_cap is None
            or market_cap <= 0
            or net_debt is None
            or net_debt <= 0
            or debt_due_24m is None
            or debt_due_24m <= 0
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_stable_payer_interest_coverage_min"]
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["dividend_increase_stable_payer_fcf_conversion_min"]
            or ebitda_margin is None
            or ebitda_margin <= 0
            or (revenue_yoy is not None and revenue_yoy < 0)
        ):
            return False

        cash = _to_float(_feature_value(features, "liquidity.cash"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        liquidity_buffer_ok = bool(available_for_actions is not None and available_for_actions > 0)
        if not liquidity_buffer_ok:
            liquidity_buffer_ok = bool(
                cash is not None
                and minimum_cash is not None
                and minimum_cash > 0
                and cash >= minimum_cash * self.thresholds["dividend_increase_stable_payer_cash_buffer_tolerance"]
            )
        if not liquidity_buffer_ok:
            return False

        net_debt_burden = net_debt / market_cap
        debt_due_vs_mcap = debt_due_24m / market_cap
        return bool(
            net_debt_burden <= self.thresholds["dividend_increase_stable_payer_net_debt_vs_mcap_max"]
            and debt_due_vs_mcap <= self.thresholds["dividend_increase_stable_payer_debt_due_vs_mcap_max"]
        )

    def _schedule_anomaly_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False

        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        volatility_90d = _to_float(_feature_value(features, "market.volatility_90d"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        if (
            total_debt is None
            or total_debt <= 0
            or market_cap is None
            or market_cap <= 0
            or net_debt is None
            or net_debt < 0
            or debt_due_24m is None
            or debt_due_24m <= 0
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_schedule_anomaly_interest_coverage_min"]
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["dividend_increase_schedule_anomaly_fcf_conversion_min"]
            or ebitda_margin is None
            or ebitda_margin <= 0
        ):
            return False

        schedule_anomaly = self._debt_schedule_inconsistency_flag(features)
        if not schedule_anomaly:
            schedule_anomaly = bool(
                debt_due_24m >= total_debt * self.thresholds["dividend_increase_schedule_anomaly_debt_due_vs_total_debt_min"]
                or (maturity_ratio is not None and maturity_ratio >= 1.0)
            )
        if not schedule_anomaly:
            return False

        market_anomaly = bool(
            (volatility_90d is not None and volatility_90d >= self.thresholds["dividend_increase_schedule_anomaly_volatility_90d_min"])
            or (drawdown_90d is not None and drawdown_90d <= -0.9)
        )
        if not market_anomaly:
            return False

        cash = _to_float(_feature_value(features, "liquidity.cash"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        liquidity_buffer_ok = bool(available_for_actions is not None and available_for_actions > 0)
        if not liquidity_buffer_ok:
            liquidity_buffer_ok = bool(cash is not None and minimum_cash is not None and minimum_cash > 0 and cash >= minimum_cash)
        if not liquidity_buffer_ok:
            return False

        net_debt_burden = net_debt / market_cap
        return net_debt_burden <= self.thresholds["dividend_increase_schedule_anomaly_net_debt_vs_mcap_max"]

    def _coverage_outlier_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        revenue_yoy = _to_float(_feature_value(features, "operating.revenue_yoy_last_q"), None)
        cash = _to_float(_feature_value(features, "liquidity.cash"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        if (
            market_cap is None
            or market_cap <= 0
            or total_debt is None
            or total_debt <= 0
            or net_debt is None
            or net_debt <= 0
            or debt_due_24m is None
            or debt_due_24m <= 0
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_coverage_outlier_interest_coverage_min"]
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["dividend_increase_coverage_outlier_fcf_conversion_min"]
            or ebitda_margin is None
            or ebitda_margin <= 0
            or (revenue_yoy is not None and revenue_yoy < 0)
            or cash is None
            or minimum_cash is None
            or minimum_cash <= 0
            or cash < minimum_cash
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        net_debt_burden = net_debt / market_cap
        debt_due_vs_mcap = debt_due_24m / market_cap
        return bool(
            total_debt_vs_mcap >= self.thresholds["dividend_increase_coverage_outlier_total_debt_vs_mcap_min"]
            and total_debt_vs_mcap <= self.thresholds["dividend_increase_coverage_outlier_total_debt_vs_mcap_max"]
            and net_debt_burden <= self.thresholds["dividend_increase_coverage_outlier_net_debt_vs_mcap_max"]
            and debt_due_vs_mcap <= self.thresholds["dividend_increase_coverage_outlier_debt_due_vs_mcap_max"]
        )

    def _missing_schedule_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False

        debt_due_24m = self._effective_debt_due_24m(features)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        if debt_due_24m is not None or maturity_ratio is not None:
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        runway_months = _to_float(_feature_value(features, "liquidity.runway_months"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        revenue_yoy = _to_float(_feature_value(features, "operating.revenue_yoy_last_q"), None)
        if (
            market_cap is None
            or market_cap <= 0
            or total_debt is None
            or total_debt <= 0
            or net_leverage is None
            or net_leverage < self.thresholds["dividend_increase_missing_schedule_net_leverage_min"]
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_missing_schedule_interest_coverage_min"]
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["dividend_increase_missing_schedule_fcf_conversion_min"]
            or ebitda_margin is None
            or ebitda_margin < self.thresholds["dividend_increase_missing_schedule_ebitda_margin_min"]
            or runway_months is None
            or runway_months < self.thresholds["dividend_increase_missing_schedule_runway_months_min"]
            or equity_window is None
            or equity_window < self.thresholds["dividend_increase_missing_schedule_equity_window_min"]
            or drawdown_90d is None
            or drawdown_90d < self.thresholds["dividend_increase_missing_schedule_drawdown_90d_min"]
            or (revenue_yoy is not None and revenue_yoy < self.thresholds["dividend_increase_missing_schedule_revenue_yoy_min"])
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        return total_debt_vs_mcap <= self.thresholds["dividend_increase_missing_schedule_total_debt_vs_mcap_max"]

    def _large_cap_coverage_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        cash = _to_float(_feature_value(features, "liquidity.cash"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        revenue_yoy = _to_float(_feature_value(features, "operating.revenue_yoy_last_q"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        volatility_90d = _to_float(_feature_value(features, "market.volatility_90d"), None)

        if (
            market_cap is None
            or market_cap < self.thresholds["dividend_increase_large_cap_market_cap_min"]
            or available_for_actions is None
            or available_for_actions <= 0
            or total_debt is None
            or total_debt <= 0
            or net_debt is None
            or net_debt <= 0
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_large_cap_interest_coverage_min"]
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["dividend_increase_large_cap_fcf_conversion_min"]
            or ebitda_margin is None
            or ebitda_margin <= 0
            or (revenue_yoy is not None and revenue_yoy < 0)
            or debt_due_24m is None
            or debt_due_24m <= 0
            or equity_window is None
            or equity_window < self.thresholds["dividend_increase_large_cap_equity_window_min"]
        ):
            return False

        if minimum_cash is not None and minimum_cash > 0 and cash is not None and cash < minimum_cash:
            return False

        market_dislocation = bool(
            (drawdown_90d is not None and drawdown_90d <= self.thresholds["dividend_increase_large_cap_drawdown_90d_max"])
            or (
                volatility_90d is not None
                and volatility_90d >= self.thresholds["dividend_increase_large_cap_volatility_90d_min"]
            )
        )
        if not market_dislocation:
            return False

        total_debt_vs_mcap = total_debt / market_cap
        net_debt_vs_mcap = net_debt / market_cap
        debt_due_vs_mcap = debt_due_24m / market_cap
        liquidity_vs_mcap = available_for_actions / market_cap
        liquidity_cover_24m = available_for_actions / debt_due_24m
        return bool(
            total_debt_vs_mcap <= self.thresholds["dividend_increase_large_cap_total_debt_vs_mcap_max"]
            and net_debt_vs_mcap <= self.thresholds["dividend_increase_large_cap_net_debt_vs_mcap_max"]
            and debt_due_vs_mcap <= self.thresholds["dividend_increase_large_cap_debt_due_vs_mcap_max"]
            and liquidity_vs_mcap >= self.thresholds["dividend_increase_large_cap_liquidity_vs_mcap_min"]
            and liquidity_cover_24m >= self.thresholds["dividend_increase_large_cap_liquidity_cover_24m_min"]
            and (
                maturity_ratio is None
                or maturity_ratio <= self.thresholds["dividend_increase_large_cap_maturity_wall_ratio_24m_max"]
            )
        )

    def _no_maturity_pressure_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False

        runway_months = _to_float(_feature_value(features, "liquidity.runway_months"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        cash = _to_float(_feature_value(features, "liquidity.cash"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        revenue_yoy = _to_float(_feature_value(features, "operating.revenue_yoy_last_q"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        volatility_90d = _to_float(_feature_value(features, "market.volatility_90d"), None)

        if (
            runway_months is None
            or runway_months < self.thresholds["dividend_increase_no_maturity_runway_months_min"]
            or available_for_actions is None
            or available_for_actions <= 0
            or market_cap is None
            or market_cap <= 0
            or total_debt is None
            or total_debt <= 0
            or net_debt is None
            or net_debt <= 0
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_no_maturity_interest_coverage_min"]
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["dividend_increase_no_maturity_fcf_conversion_min"]
            or ebitda_margin is None
            or ebitda_margin < self.thresholds["dividend_increase_no_maturity_ebitda_margin_min"]
            or (revenue_yoy is not None and revenue_yoy < self.thresholds["dividend_increase_no_maturity_revenue_yoy_min"])
            or equity_window is None
            or equity_window < self.thresholds["dividend_increase_no_maturity_equity_window_min"]
        ):
            return False

        if minimum_cash is not None and minimum_cash > 0 and cash is not None and cash < minimum_cash:
            return False

        no_near_term_pressure = bool(
            (debt_due_24m is None or debt_due_24m <= 0)
            and (maturity_ratio is None or maturity_ratio <= 0.0)
        )
        if not no_near_term_pressure:
            return False

        market_dislocation = bool(
            (drawdown_90d is not None and drawdown_90d <= self.thresholds["dividend_increase_no_maturity_drawdown_90d_max"])
            or (
                volatility_90d is not None
                and volatility_90d >= self.thresholds["dividend_increase_no_maturity_volatility_90d_min"]
            )
        )
        if not market_dislocation:
            return False

        total_debt_vs_mcap = total_debt / market_cap
        net_debt_vs_mcap = net_debt / market_cap
        return bool(
            total_debt_vs_mcap <= self.thresholds["dividend_increase_no_maturity_total_debt_vs_mcap_max"]
            and net_debt_vs_mcap <= self.thresholds["dividend_increase_no_maturity_net_debt_vs_mcap_max"]
        )

    def _mild_maturity_pressure_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        revenue_yoy = _to_float(_feature_value(features, "operating.revenue_yoy_last_q"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)

        if (
            market_cap is None
            or market_cap <= 0
            or total_debt is None
            or total_debt <= 0
            or net_debt is None
            or net_debt <= 0
            or net_leverage is None
            or net_leverage > self.thresholds["dividend_increase_net_leverage_max"]
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_mild_maturity_interest_coverage_min"]
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["dividend_increase_mild_maturity_fcf_conversion_min"]
            or ebitda_margin is None
            or ebitda_margin <= 0
            or available_for_actions is None
            or available_for_actions <= 0
            or debt_due_24m is None
            or debt_due_24m <= 0
        ):
            return False

        if revenue_yoy is not None and revenue_yoy < -0.10:
            return False

        total_debt_vs_mcap = total_debt / market_cap
        if total_debt_vs_mcap > self.thresholds["dividend_increase_mild_maturity_total_debt_vs_mcap_max"]:
            return False

        net_debt_vs_mcap = net_debt / market_cap
        if net_debt_vs_mcap > self.thresholds["dividend_increase_mild_maturity_net_debt_vs_mcap_max"]:
            return False

        debt_due_vs_mcap = debt_due_24m / market_cap
        if debt_due_vs_mcap > self.thresholds["dividend_increase_mild_maturity_debt_due_vs_mcap_max"]:
            return False

        if (
            maturity_ratio is not None
            and maturity_ratio > self.thresholds["dividend_increase_mild_maturity_maturity_wall_ratio_24m_max"]
        ):
            return False

        liquidity_cover_24m = available_for_actions / debt_due_24m
        return liquidity_cover_24m >= self.thresholds["dividend_increase_mild_maturity_liquidity_cover_24m_min"]

    def _high_coverage_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        # Historical replays sometimes miss runway while still showing a very strong
        # regular payer with modest leverage and ample balance-sheet flexibility.
        # Keep a measured dividend-increase path alive for those names so we do not
        # collapse into no-plan / buyback-only behavior from a single noisy cash-flow field.
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        revenue_cagr = _to_float(_feature_value(features, "operating.revenue_cagr_3y"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            market_cap is None
            or market_cap <= 0
            or total_debt is None
            or total_debt < 0
            or net_debt is None
            or net_leverage is None
            or net_leverage > self.thresholds["dividend_increase_high_coverage_net_leverage_max"]
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_high_coverage_interest_coverage_min"]
            or available_for_actions is None
            or available_for_actions <= 0
            or ebitda_margin is None
            or ebitda_margin <= 0
        ):
            return False

        if (
            revenue_cagr is not None
            and revenue_cagr < self.thresholds["dividend_increase_high_coverage_revenue_cagr_3y_min"]
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        if total_debt_vs_mcap > self.thresholds["dividend_increase_high_coverage_total_debt_vs_mcap_max"]:
            return False

        debt_due_vs_mcap = 0.0
        if debt_due_24m is not None and debt_due_24m > 0:
            debt_due_vs_mcap = debt_due_24m / market_cap
            if debt_due_vs_mcap > self.thresholds["dividend_increase_high_coverage_debt_due_vs_mcap_max"]:
                return False

        liquidity_buffer_ok = available_for_actions > 0
        if not liquidity_buffer_ok:
            cash = _to_float(_feature_value(features, "liquidity.cash"), None)
            minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
            liquidity_buffer_ok = bool(
                cash is not None and minimum_cash is not None and minimum_cash > 0 and cash >= minimum_cash
            )
        if not liquidity_buffer_ok:
            return False

        if (
            fcf_conversion is not None
            and fcf_conversion >= self.thresholds["dividend_increase_high_coverage_fcf_conversion_min"]
        ):
            return True

        net_debt_vs_mcap = max(net_debt, 0.0) / market_cap
        return bool(
            market_cap >= self.thresholds["dividend_increase_high_coverage_megacap_market_cap_min"]
            and interest_coverage >= self.thresholds["dividend_increase_high_coverage_megacap_interest_coverage_min"]
            and net_debt_vs_mcap <= self.thresholds["dividend_increase_high_coverage_megacap_net_debt_vs_mcap_max"]
            and fcf_conversion is not None
            and fcf_conversion >= self.thresholds["dividend_increase_high_coverage_megacap_fcf_conversion_min"]
        )

    def _coverage_supported_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(
            _feature_value(features, "capital_return.last_dividend_event_type") or ""
        ).strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False
        if self._market_shutdown_regular_payer_recap_profile(features):
            return False
        if self._strategic_regular_payer_recap_profile(features):
            return False
        if self._buyback_regular_payer_recap_profile(features):
            return False
        if self._dividend_cut_profile(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        revenue_cagr = _to_float(_feature_value(features, "operating.revenue_cagr_3y"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            market_cap is None
            or market_cap < self.thresholds["dividend_increase_coverage_supported_market_cap_min"]
            or equity_window is None
            or equity_window < self.thresholds["dividend_increase_coverage_supported_equity_window_min"]
            or available_for_actions is None
            or available_for_actions <= 0.0
            or total_debt is None
            or total_debt <= 0.0
            or net_debt is None
            or net_debt <= 0.0
            or net_leverage is None
            or net_leverage < self.thresholds["dividend_increase_coverage_supported_net_leverage_min"]
            or net_leverage > self.thresholds["dividend_increase_coverage_supported_net_leverage_max"]
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_coverage_supported_interest_coverage_min"]
            or ebitda_margin is None
            or ebitda_margin < self.thresholds["dividend_increase_coverage_supported_ebitda_margin_min"]
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        if total_debt_vs_mcap > self.thresholds["dividend_increase_coverage_supported_total_debt_vs_mcap_max"]:
            return False

        quality_ok = bool(
            fcf_conversion is not None
            and fcf_conversion >= self.thresholds["dividend_increase_coverage_supported_fcf_conversion_min"]
        )
        if not quality_ok:
            quality_ok = bool(
                interest_coverage
                >= self.thresholds["dividend_increase_coverage_supported_override_interest_coverage_min"]
                and ebitda_margin
                >= self.thresholds["dividend_increase_coverage_supported_override_ebitda_margin_min"]
                and (
                    revenue_cagr is None
                    or revenue_cagr >= self.thresholds["dividend_increase_coverage_supported_revenue_cagr_3y_min"]
                )
            )
        if not quality_ok:
            return False

        if debt_due_24m is not None and debt_due_24m > 0:
            debt_due_vs_mcap = debt_due_24m / market_cap
            liquidity_cover = available_for_actions / debt_due_24m
            return bool(
                debt_due_vs_mcap <= self.thresholds["dividend_increase_coverage_supported_debt_due_vs_mcap_max"]
                and liquidity_cover
                >= self.thresholds["dividend_increase_coverage_supported_liquidity_cover_24m_min"]
            )

        return True

    def _sparse_data_regular_payer_dividend_increase_profile(
        self, features: Dict[str, Any]
    ) -> bool:
        last_dividend_event = str(
            _feature_value(features, "capital_return.last_dividend_event_type") or ""
        ).strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False
        if self._market_shutdown_regular_payer_recap_profile(features):
            return False
        if self._strategic_regular_payer_recap_profile(features):
            return False
        if self._buyback_regular_payer_recap_profile(features):
            return False
        if self._dividend_cut_profile(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        available_for_actions = _to_float(
            _feature_value(features, "liquidity.available_for_actions"), None
        )
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(
            _feature_value(features, "capital_structure.interest_coverage"), None
        )
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            market_cap is None
            or market_cap < self.thresholds["dividend_increase_sparse_market_cap_min"]
            or equity_window is None
            or equity_window < self.thresholds["dividend_increase_sparse_equity_window_min"]
            or available_for_actions is None
            or available_for_actions <= 0.0
            or total_debt is None
            or total_debt <= 0.0
            or net_debt is None
            or net_debt <= 0.0
            or net_leverage is None
            or net_leverage < self.thresholds["dividend_increase_sparse_net_leverage_min"]
            or net_leverage > self.thresholds["dividend_increase_sparse_net_leverage_max"]
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_sparse_interest_coverage_min"]
        ):
            return False

        if (
            drawdown_90d is not None
            and drawdown_90d < self.thresholds["dividend_increase_sparse_drawdown_90d_max"]
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        if total_debt_vs_mcap > self.thresholds["dividend_increase_sparse_total_debt_vs_mcap_max"]:
            return False

        if debt_due_24m is not None and debt_due_24m > 0:
            debt_due_vs_mcap = debt_due_24m / market_cap
            liquidity_cover = available_for_actions / debt_due_24m
            return bool(
                debt_due_vs_mcap <= self.thresholds["dividend_increase_sparse_debt_due_vs_mcap_max"]
                and liquidity_cover
                >= self.thresholds["dividend_increase_sparse_liquidity_cover_24m_min"]
            )

        return True

    def _coverage_gap_regular_payer_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        last_dividend_event = str(
            _feature_value(features, "capital_return.last_dividend_event_type") or ""
        ).strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False
        if self._market_shutdown_regular_payer_recap_profile(features):
            return False
        if self._strategic_regular_payer_recap_profile(features):
            return False
        if self._buyback_regular_payer_recap_profile(features):
            return False
        if self._dividend_cut_profile(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        available_for_actions = _to_float(
            _feature_value(features, "liquidity.available_for_actions"), None
        )
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(
            _feature_value(features, "capital_structure.interest_coverage"), None
        )
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if interest_coverage is not None:
            return False
        if (
            market_cap is None
            or market_cap < self.thresholds["dividend_increase_coverage_gap_market_cap_min"]
            or equity_window is None
            or equity_window < self.thresholds["dividend_increase_coverage_gap_equity_window_min"]
            or available_for_actions is None
            or available_for_actions <= 0.0
            or total_debt is None
            or total_debt <= 0.0
            or net_debt is None
            or net_debt <= 0.0
            or net_leverage is None
            or net_leverage < self.thresholds["dividend_increase_coverage_gap_net_leverage_min"]
            or net_leverage > self.thresholds["dividend_increase_coverage_gap_net_leverage_max"]
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["dividend_increase_coverage_gap_fcf_conversion_min"]
            or ebitda_margin is None
            or ebitda_margin < self.thresholds["dividend_increase_coverage_gap_ebitda_margin_min"]
        ):
            return False

        if (
            drawdown_90d is not None
            and drawdown_90d < self.thresholds["dividend_increase_coverage_gap_drawdown_90d_max"]
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        if total_debt_vs_mcap > self.thresholds["dividend_increase_coverage_gap_total_debt_vs_mcap_max"]:
            return False

        if debt_due_24m is not None and debt_due_24m > 0:
            debt_due_vs_mcap = debt_due_24m / market_cap
            liquidity_cover = available_for_actions / debt_due_24m
            return bool(
                debt_due_vs_mcap <= self.thresholds["dividend_increase_coverage_gap_debt_due_vs_mcap_max"]
                and liquidity_cover
                >= self.thresholds["dividend_increase_coverage_gap_liquidity_cover_24m_min"]
            )

        return True

    def _liquidity_supported_regular_payer_dividend_increase_profile(
        self, features: Dict[str, Any]
    ) -> bool:
        last_dividend_event = str(
            _feature_value(features, "capital_return.last_dividend_event_type") or ""
        ).strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False
        if self._market_shutdown_regular_payer_recap_profile(features):
            return False
        if self._strategic_regular_payer_recap_profile(features):
            return False
        if self._buyback_regular_payer_recap_profile(features):
            return False
        if self._dividend_cut_profile(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        available_for_actions = _to_float(
            _feature_value(features, "liquidity.available_for_actions"), None
        )
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(
            _feature_value(features, "capital_structure.interest_coverage"), None
        )
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            market_cap is None
            or market_cap < self.thresholds["dividend_increase_liquidity_supported_market_cap_min"]
            or equity_window is None
            or equity_window < self.thresholds["dividend_increase_liquidity_supported_equity_window_min"]
            or available_for_actions is None
            or available_for_actions <= 0.0
            or total_debt is None
            or total_debt <= 0.0
            or net_debt is None
            or net_debt <= 0.0
            or net_leverage is None
            or net_leverage < self.thresholds["dividend_increase_liquidity_supported_net_leverage_min"]
            or net_leverage > self.thresholds["dividend_increase_liquidity_supported_net_leverage_max"]
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_liquidity_supported_interest_coverage_min"]
            or ebitda_margin is None
            or ebitda_margin < self.thresholds["dividend_increase_liquidity_supported_ebitda_margin_min"]
        ):
            return False

        if (
            drawdown_90d is not None
            and drawdown_90d < self.thresholds["dividend_increase_liquidity_supported_drawdown_90d_max"]
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        liquidity_vs_mcap = available_for_actions / market_cap
        if total_debt_vs_mcap > self.thresholds["dividend_increase_liquidity_supported_total_debt_vs_mcap_max"]:
            return False
        if liquidity_vs_mcap < self.thresholds["dividend_increase_liquidity_supported_liquidity_vs_mcap_min"]:
            return False

        if debt_due_24m is not None and debt_due_24m > 0:
            debt_due_vs_mcap = debt_due_24m / market_cap
            liquidity_cover = available_for_actions / debt_due_24m
            return bool(
                debt_due_vs_mcap <= self.thresholds["dividend_increase_liquidity_supported_debt_due_vs_mcap_max"]
                and liquidity_cover
                >= self.thresholds["dividend_increase_liquidity_supported_liquidity_cover_24m_min"]
            )

        return True

    def _balance_sheet_light_regular_payer_dividend_increase_profile(
        self, features: Dict[str, Any]
    ) -> bool:
        last_dividend_event = str(
            _feature_value(features, "capital_return.last_dividend_event_type") or ""
        ).strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False
        if self._market_shutdown_regular_payer_recap_profile(features):
            return False
        if self._strategic_regular_payer_recap_profile(features):
            return False
        if self._buyback_regular_payer_recap_profile(features):
            return False
        if self._dividend_cut_profile(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(
            _feature_value(features, "capital_structure.interest_coverage"), None
        )
        available_for_actions = _to_float(
            _feature_value(features, "liquidity.available_for_actions"), None
        )
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            market_cap is None
            or market_cap < self.thresholds["dividend_increase_balance_sheet_light_market_cap_min"]
            or equity_window is None
            or equity_window < self.thresholds["dividend_increase_balance_sheet_light_equity_window_min"]
            or total_debt is None
            or total_debt <= 0.0
            or ebitda_margin is None
            or ebitda_margin < self.thresholds["dividend_increase_balance_sheet_light_ebitda_margin_min"]
        ):
            return False

        if (
            drawdown_90d is not None
            and drawdown_90d < self.thresholds["dividend_increase_balance_sheet_light_drawdown_90d_max"]
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        if (
            total_debt_vs_mcap
            > self.thresholds["dividend_increase_balance_sheet_light_total_debt_vs_mcap_max"]
        ):
            return False

        if (
            net_leverage is not None
            and net_leverage > self.thresholds["dividend_increase_balance_sheet_light_net_leverage_max"]
        ):
            return False
        if (
            interest_coverage is not None
            and interest_coverage
            < self.thresholds["dividend_increase_balance_sheet_light_interest_coverage_min"]
        ):
            return False

        if debt_due_24m is not None and debt_due_24m > 0:
            debt_due_vs_mcap = debt_due_24m / market_cap
            if (
                debt_due_vs_mcap
                > self.thresholds["dividend_increase_balance_sheet_light_debt_due_vs_mcap_max"]
            ):
                return False

        liquidity_supported = False
        if available_for_actions is not None and available_for_actions > 0:
            liquidity_supported = (
                available_for_actions / market_cap
                >= self.thresholds["dividend_increase_balance_sheet_light_liquidity_vs_mcap_min"]
            )
        cashflow_supported = bool(
            fcf_conversion is not None
            and fcf_conversion >= self.thresholds["dividend_increase_balance_sheet_light_fcf_conversion_min"]
        )
        return bool(liquidity_supported or cashflow_supported)

    def _supportive_regular_payer_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        return bool(
            self._coverage_supported_dividend_increase_profile(features)
            or self._sparse_data_regular_payer_dividend_increase_profile(features)
            or self._coverage_gap_regular_payer_dividend_increase_profile(features)
            or self._balance_sheet_light_regular_payer_dividend_increase_profile(features)
            or self._liquidity_supported_regular_payer_dividend_increase_profile(features)
        )

    def _runway_equity_pressure_profile(self, features: Dict[str, Any]) -> bool:
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        volatility_90d = _to_float(_feature_value(features, "market.volatility_90d"), None)
        fcf_yield = _to_float(_feature_value(features, "market.fcf_yield"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        revenue_yoy = _to_float(_feature_value(features, "operating.revenue_yoy_last_q"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)

        market_stress = bool(
            (drawdown_90d is not None and drawdown_90d <= self.thresholds["equity_runway_drawdown_90d_max"])
            or (
                volatility_90d is not None
                and volatility_90d >= self.thresholds["equity_runway_volatility_90d_min"]
            )
        )
        operating_fragility = bool(
            (fcf_yield is not None and fcf_yield <= self.thresholds["equity_runway_fcf_yield_max"])
            or (
                ebitda_margin is not None
                and ebitda_margin <= self.thresholds["equity_runway_ebitda_margin_max"]
            )
            or (
                revenue_yoy is not None
                and revenue_yoy <= self.thresholds["equity_runway_revenue_yoy_max"]
            )
        )
        return bool(
            market_stress
            and operating_fragility
            and (net_debt is None or net_debt <= 0.0)
            and (total_debt is None or total_debt <= 0.0)
        )

    def _market_shutdown_regular_payer_recap_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._has_buyback_specific_support(features):
            return False

        credit_window = _to_float(_feature_value(features, "market.credit_window_proxy"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            credit_window is None
            or credit_window > self.thresholds["equity_backstop_credit_window_max"]
            or equity_window is None
            or equity_window > self.thresholds["market_shutdown_regular_payer_equity_window_max"]
            or drawdown_90d is None
            or drawdown_90d > self.thresholds["market_shutdown_regular_payer_drawdown_90d_max"]
            or market_cap is None
            or market_cap <= 0
            or total_debt is None
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        debt_due_vs_mcap = (debt_due_24m / market_cap) if debt_due_24m is not None and debt_due_24m > 0 else 0.0
        return bool(
            total_debt_vs_mcap >= self.thresholds["market_shutdown_regular_payer_total_debt_vs_mcap_min"]
            or debt_due_vs_mcap >= self.thresholds["market_shutdown_regular_payer_debt_due_vs_mcap_min"]
        )

    def _strategic_regular_payer_recap_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._has_buyback_specific_support(features):
            return False

        last_action_type = str(_feature_value(features, "strategic.last_action_type") or "").strip().lower()
        if last_action_type not in {"acquisition", "debt_issuance", "equity_offering"}:
            return False

        credit_window = _to_float(_feature_value(features, "market.credit_window_proxy"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            credit_window is None
            or credit_window > self.thresholds["equity_backstop_credit_window_max"]
            or equity_window is None
            or equity_window < self.thresholds["strategic_regular_payer_recap_equity_window_min"]
            or market_cap is None
            or market_cap <= 0.0
            or total_debt is None
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        debt_due_vs_mcap = (
            debt_due_24m / market_cap
            if debt_due_24m is not None and debt_due_24m > 0
            else 0.0
        )
        debt_pressure = bool(
            total_debt_vs_mcap >= self.thresholds["strategic_regular_payer_recap_total_debt_vs_mcap_min"]
            or debt_due_vs_mcap >= self.thresholds["strategic_regular_payer_recap_debt_due_vs_mcap_min"]
        )
        balance_sheet_distress = bool(
            (
                interest_coverage is not None
                and interest_coverage <= self.thresholds["strategic_regular_payer_recap_interest_coverage_max"]
            )
            or (
                net_leverage is not None
                and net_leverage >= self.thresholds["strategic_regular_payer_recap_net_leverage_min"]
            )
        )
        return bool(debt_pressure or balance_sheet_distress)

    def _buyback_regular_payer_recap_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._has_buyback_specific_support(features):
            return False
        last_action_type = str(_feature_value(features, "strategic.last_action_type") or "").strip().lower()
        if last_action_type != "buyback":
            return False

        credit_window = _to_float(_feature_value(features, "market.credit_window_proxy"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        runway_months = _to_float(_feature_value(features, "liquidity.runway_months"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            credit_window is None
            or credit_window > self.thresholds["equity_backstop_credit_window_max"]
            or equity_window is None
            or equity_window < self.thresholds["buyback_regular_payer_recap_equity_window_min"]
            or market_cap is None
            or market_cap <= 0.0
            or interest_coverage is None
            or interest_coverage < self.thresholds["buyback_regular_payer_recap_interest_coverage_min"]
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["buyback_regular_payer_recap_fcf_conversion_min"]
            or runway_months is None
            or runway_months < self.thresholds["buyback_regular_payer_recap_runway_months_min"]
            or available_for_actions is None
            or available_for_actions <= 0.0
        ):
            return False

        total_debt_vs_mcap = (total_debt / market_cap) if total_debt is not None and total_debt > 0 else 0.0
        debt_due_vs_mcap = (
            debt_due_24m / market_cap
            if debt_due_24m is not None and debt_due_24m > 0
            else 0.0
        )
        return bool(
            total_debt_vs_mcap >= self.thresholds["buyback_regular_payer_recap_total_debt_vs_mcap_min"]
            or debt_due_vs_mcap >= self.thresholds["buyback_regular_payer_recap_debt_due_vs_mcap_min"]
        )

    def _strategic_nonpayer_recap_profile(self, features: Dict[str, Any]) -> bool:
        if _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False

        last_action_type = str(_feature_value(features, "strategic.last_action_type") or "").strip().lower()
        if last_action_type not in {"buyback", "debt_issuance", "equity_offering"}:
            return False

        credit_window = _to_float(_feature_value(features, "market.credit_window_proxy"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        runway_months = _to_float(_feature_value(features, "liquidity.runway_months"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            credit_window is None
            or credit_window > self.thresholds["equity_backstop_credit_window_max"]
            or equity_window is None
            or equity_window < self.thresholds["strategic_nonpayer_recap_equity_window_min"]
            or market_cap is None
            or market_cap <= 0.0
            or interest_coverage is None
            or interest_coverage < self.thresholds["strategic_nonpayer_recap_interest_coverage_min"]
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["strategic_nonpayer_recap_fcf_conversion_min"]
            or runway_months is None
            or runway_months < self.thresholds["strategic_nonpayer_recap_runway_months_min"]
        ):
            return False

        total_debt_vs_mcap = (total_debt / market_cap) if total_debt is not None and total_debt > 0 else 0.0
        debt_due_vs_mcap = (
            debt_due_24m / market_cap
            if debt_due_24m is not None and debt_due_24m > 0
            else 0.0
        )
        return bool(
            total_debt_vs_mcap >= self.thresholds["strategic_nonpayer_recap_total_debt_vs_mcap_min"]
            or debt_due_vs_mcap >= self.thresholds["strategic_nonpayer_recap_debt_due_vs_mcap_min"]
        )

    def _sparse_reset_recap_profile(self, features: Dict[str, Any]) -> bool:
        if self._coverage_supported_dividend_initiate_override_profile(features) and not self._has_buyback_specific_support(
            features
        ):
            return False
        if self._debt_bearing_dividend_initiate_override_profile(features):
            return False

        last_action_type = str(_feature_value(features, "strategic.last_action_type") or "").strip().lower()
        last_dividend_event = str(
            _feature_value(features, "capital_return.last_dividend_event_type") or ""
        ).strip().lower()
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        credit_window = _to_float(_feature_value(features, "market.credit_window_proxy"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)

        if (
            equity_window is None
            or equity_window < self.thresholds["sparse_reset_recap_equity_window_min"]
            or market_cap is None
            or market_cap <= 0.0
            or total_debt is None
            or total_debt <= 0.0
            or interest_coverage is None
            or interest_coverage < self.thresholds["sparse_reset_recap_interest_coverage_min"]
        ):
            return False

        if (
            drawdown_90d is not None
            and drawdown_90d < self.thresholds["sparse_reset_recap_drawdown_90d_max"]
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        balance_sheet_burden = bool(
            total_debt_vs_mcap >= self.thresholds["sparse_reset_recap_total_debt_vs_mcap_min"]
            or (
                net_leverage is not None
                and net_leverage >= self.thresholds["sparse_reset_recap_net_leverage_min"]
            )
        )
        if not balance_sheet_burden:
            return False

        strategic_reset_signal = last_action_type in {
            "buyback",
            "debt_issuance",
            "acquisition",
            "equity_offering",
        }
        if not strategic_reset_signal:
            strategic_reset_signal = bool(
                total_debt_vs_mcap
                >= self.thresholds["sparse_reset_recap_total_debt_vs_mcap_strong_min"]
            )
        if not strategic_reset_signal and last_dividend_event in {"dividend_regular", "regular"}:
            strategic_reset_signal = bool(
                credit_window is None
                or credit_window <= self.thresholds["sparse_reset_recap_credit_window_max"]
            )
        if not strategic_reset_signal:
            return False

        return bool(
            credit_window is None
            or credit_window <= self.thresholds["sparse_reset_recap_credit_window_max"]
            or (
                net_leverage is not None
                and net_leverage >= self.thresholds["sparse_reset_recap_net_leverage_min"]
            )
        )

    def _nonpayer_recap_preference_profile(self, features: Dict[str, Any]) -> bool:
        if _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False

        credit_window = _to_float(_feature_value(features, "market.credit_window_proxy"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            credit_window is None
            or credit_window > self.thresholds["equity_backstop_credit_window_max"]
            or equity_window is None
            or equity_window <= 0.0
            or market_cap is None
            or market_cap <= 0.0
        ):
            return False

        public_like_window = (
            equity_window >= self.thresholds["nonpayer_recap_preference_equity_window_min"]
        )
        private_like_window = (
            0.0 < equity_window <= self.thresholds["equity_private_placement_equity_window_max"]
        )
        if not (public_like_window or private_like_window):
            return False

        total_debt_vs_mcap = (total_debt / market_cap) if total_debt is not None and total_debt > 0 else 0.0
        debt_due_vs_mcap = (
            debt_due_24m / market_cap
            if debt_due_24m is not None and debt_due_24m > 0
            else 0.0
        )
        debt_pressure = bool(
            total_debt_vs_mcap >= self.thresholds["nonpayer_recap_preference_total_debt_vs_mcap_min"]
            or debt_due_vs_mcap >= self.thresholds["nonpayer_recap_preference_debt_due_vs_mcap_min"]
        )
        balance_sheet_distress = bool(
            (
                interest_coverage is not None
                and interest_coverage <= self.thresholds["nonpayer_recap_preference_interest_coverage_max"]
            )
            or (
                net_leverage is not None
                and net_leverage >= self.thresholds["nonpayer_recap_preference_net_leverage_min"]
            )
        )
        return bool(debt_pressure or balance_sheet_distress)

    def _durable_leveraged_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        if self._rating_pressure(features):
            return False

        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        if (
            fcf_conversion is None
            or fcf_conversion < self.thresholds["dividend_increase_leveraged_payer_fcf_conversion_min"]
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_leveraged_payer_interest_coverage_min"]
            or net_debt is None
            or net_debt <= 0
            or net_leverage is None
            or net_leverage < self.thresholds["dividend_increase_leveraged_payer_net_leverage_min"]
            or net_leverage > self.thresholds["dividend_increase_leveraged_payer_net_leverage_max"]
            or market_cap is None
            or market_cap <= 0
        ):
            return False

        net_debt_burden = net_debt / market_cap
        if net_debt_burden > self.thresholds["dividend_increase_leveraged_payer_net_debt_vs_mcap_max"]:
            return False

        debt_due_24m = self._effective_debt_due_24m(features)
        if debt_due_24m is None or debt_due_24m <= 0:
            return False
        debt_due_vs_mcap = debt_due_24m / market_cap
        if debt_due_vs_mcap > self.thresholds["dividend_increase_leveraged_payer_debt_due_vs_mcap_max"]:
            return False

        cash = _to_float(_feature_value(features, "liquidity.cash"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        liquidity_buffer_ok = bool(
            available_for_actions is not None and available_for_actions > 0
        )
        if not liquidity_buffer_ok:
            liquidity_buffer_ok = bool(
                cash is not None
                and minimum_cash is not None
                and minimum_cash > 0
                and cash >= minimum_cash * self.thresholds["dividend_increase_leveraged_payer_cash_buffer_tolerance"]
            )
        if not liquidity_buffer_ok:
            return False

        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        revenue_yoy = _to_float(_feature_value(features, "operating.revenue_yoy_last_q"), None)
        return bool(
            ebitda_margin is not None
            and ebitda_margin > 0
            and (revenue_yoy is None or revenue_yoy >= 0)
        )

    def _healthy_dividend_increase_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        if not self._expectations_allow_capital_return(features):
            return False
        if self._runway_equity_pressure_profile(features):
            return False
        if self._market_shutdown_regular_payer_recap_profile(features):
            return False
        if self._strategic_regular_payer_recap_profile(features):
            return False
        if self._buyback_regular_payer_recap_profile(features):
            return False
        if self._dividend_cut_balance_sheet_pressure_profile(features):
            return False
        if self._leveraged_special_dividend_profile(features):
            return False

        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        anomaly_override = self._existing_payer_data_anomaly_profile(features)
        financing_anomaly_override = self._regular_payer_financing_anomaly_override(features)

        cash = _to_float(_feature_value(features, "liquidity.cash"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        liquidity_buffer_ok = False
        if available_for_actions is not None and available_for_actions > 0:
            liquidity_buffer_ok = True
        elif (
            cash is not None
            and minimum_cash is not None
            and minimum_cash > 0
            and cash >= minimum_cash * self.thresholds["dividend_increase_cash_buffer_tolerance"]
        ):
            liquidity_buffer_ok = True

        if financing_anomaly_override:
            return bool(
                market_cap is not None
                and market_cap > 0
                and fcf_conversion is not None
                and fcf_conversion >= self.thresholds["financing_anomaly_dividend_fcf_conversion_min"]
                and interest_coverage is not None
                and interest_coverage >= self.thresholds["financing_anomaly_interest_coverage_min"]
            )

        if anomaly_override:
            # Some payer snapshots have mathematically impossible debt schedules plus extreme
            # operating ratios. For established regular payers with tiny debt burdens, trust
            # the payer/liquidity evidence over obviously broken financing features.
            return liquidity_buffer_ok

        if self._durable_leveraged_dividend_increase_profile(features):
            return True
        if self._stable_debt_bearing_dividend_increase_profile(features):
            return True
        if self._schedule_anomaly_dividend_increase_profile(features):
            return True
        if self._coverage_outlier_dividend_increase_profile(features):
            return True
        if self._large_cap_coverage_dividend_increase_profile(features):
            return True
        if self._no_maturity_pressure_dividend_increase_profile(features):
            return True
        if self._mild_maturity_pressure_dividend_increase_profile(features):
            return True
        if self._high_coverage_dividend_increase_profile(features):
            return True

        if (
            fcf_conversion is None
            or fcf_conversion < self.thresholds["dividend_increase_fcf_conversion_min"]
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_increase_interest_coverage_min"]
            or self._rating_pressure(features)
        ):
            return False

        liquidity_vs_mcap = None
        if available_for_actions is not None and market_cap is not None and market_cap > 0:
            liquidity_vs_mcap = available_for_actions / market_cap
            if (
                net_debt is not None
                and net_debt <= 0
                and liquidity_vs_mcap >= self.thresholds["dividend_increase_liquidity_vs_mcap_min"]
            ):
                return True

        debt_due_vs_mcap = 0.0
        if debt_due_24m is not None and market_cap is not None and market_cap > 0:
            debt_due_vs_mcap = debt_due_24m / market_cap

        if (
            (net_debt is None or net_debt > 0)
            and net_leverage is not None
            and net_leverage <= self.thresholds["dividend_increase_net_leverage_max"]
            and debt_due_vs_mcap <= self.thresholds["dividend_increase_debt_due_vs_mcap_max"]
        ):
            return True

        if (
            fcf_conversion < self.thresholds["dividend_increase_fcf_conversion_strong_min"]
            or interest_coverage < self.thresholds["dividend_increase_interest_coverage_strong_min"]
            or market_cap is None
            or market_cap <= 0
        ):
            return False

        volatility_90d = _to_float(_feature_value(features, "market.volatility_90d"), None)
        if (
            volatility_90d is not None
            and volatility_90d > self.thresholds["dividend_increase_volatility_90d_relaxed_max"]
        ):
            return False

        net_debt_burden = None
        if net_debt is not None:
            net_debt_burden = max(net_debt, 0.0) / market_cap

        if not liquidity_buffer_ok:
            return False

        due_pressure = False
        if debt_due_24m is not None and debt_due_24m > 0:
            if debt_due_vs_mcap > self.thresholds["dividend_increase_debt_due_vs_mcap_relaxed_max"]:
                due_pressure = True
            elif available_for_actions is None or available_for_actions <= 0:
                due_pressure = True

        return bool(
            net_debt_burden is not None
            and net_debt_burden <= self.thresholds["dividend_increase_net_debt_vs_mcap_max"]
            and not due_pressure
        )

    def _special_dividend_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        if not self._expectations_allow_capital_return(features):
            return False
        if self._market_shutdown_regular_payer_recap_profile(features):
            return False
        if self._buyback_regular_payer_recap_profile(features):
            return False
        if self._leveraged_special_dividend_profile(features):
            return True
        if self._healthy_dividend_increase_profile(features):
            return False
        if self._dividend_cut_profile(features):
            return False
        if self._rating_pressure(features):
            return False

        last_dividend_event = str(
            _feature_value(features, "capital_return.last_dividend_event_type") or ""
        ).strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False

        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            available_for_actions is None
            or available_for_actions <= 0
            or minimum_cash is None
            or minimum_cash <= 0
            or market_cap is None
            or market_cap <= 0
            or total_debt is None
            or net_debt is None
            or interest_coverage is None
            or fcf_conversion is None
            or ebitda_margin is None
        ):
            return False

        liquidity_vs_mcap = available_for_actions / market_cap
        debt_due_vs_mcap = (debt_due_24m / market_cap) if debt_due_24m is not None and market_cap > 0 else 0.0
        total_debt_vs_mcap = total_debt / market_cap

        return bool(
            liquidity_vs_mcap >= self.thresholds["special_dividend_liquidity_vs_mcap_min"]
            and available_for_actions
            >= minimum_cash * self.thresholds["special_dividend_cash_buffer_multiple_min"]
            and net_debt <= 0
            and total_debt_vs_mcap <= self.thresholds["special_dividend_total_debt_vs_mcap_max"]
            and debt_due_vs_mcap <= self.thresholds["special_dividend_debt_due_vs_mcap_max"]
            and interest_coverage >= self.thresholds["special_dividend_interest_coverage_min"]
            and fcf_conversion >= self.thresholds["special_dividend_fcf_conversion_min"]
            and ebitda_margin > self.thresholds["special_dividend_ebitda_margin_min"]
        )

    def _leveraged_special_dividend_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        if self._rating_pressure(features):
            return False

        last_dividend_event = str(
            _feature_value(features, "capital_return.last_dividend_event_type") or ""
        ).strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False

        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            available_for_actions is None
            or available_for_actions <= 0
            or minimum_cash is None
            or minimum_cash <= 0
            or market_cap is None
            or market_cap <= 0
            or net_debt is None
            or net_debt <= 0
            or net_leverage is None
            or net_leverage < self.thresholds["special_dividend_leveraged_payer_net_leverage_min"]
            or interest_coverage is None
            or interest_coverage < self.thresholds["special_dividend_leveraged_payer_interest_coverage_min"]
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["special_dividend_leveraged_payer_fcf_conversion_min"]
        ):
            return False

        liquidity_vs_mcap = available_for_actions / market_cap
        net_debt_vs_mcap = net_debt / market_cap
        debt_due_vs_mcap = (debt_due_24m / market_cap) if debt_due_24m is not None and debt_due_24m > 0 else 0.0
        return bool(
            liquidity_vs_mcap >= self.thresholds["special_dividend_leveraged_payer_liquidity_vs_mcap_min"]
            and available_for_actions
            >= minimum_cash * self.thresholds["special_dividend_leveraged_payer_cash_buffer_multiple_min"]
            and net_debt_vs_mcap <= self.thresholds["special_dividend_leveraged_payer_net_debt_vs_mcap_max"]
            and debt_due_vs_mcap <= self.thresholds["special_dividend_leveraged_payer_debt_due_vs_mcap_max"]
        )

    def _has_buyback_specific_support(self, features: Dict[str, Any]) -> bool:
        buyback_capacity = _to_float(_feature_value(features, "capital_return.buyback_capacity_proxy"), None)
        share_count_trend = _to_float(_feature_value(features, "capital_return.share_count_trend"), None)
        activist_present = _is_explicit_true(_feature_value(features, "ownership_governance.activist_presence_flag"))
        ev_z = _to_float(_feature_value(features, "market.ev_ebitda_vs_peer_z"), None)
        fcf_percentile = _to_float(_feature_value(features, "market.fcf_yield_percentile_peers"), None)
        expectations_supportive = self._expectations_allow_capital_return(features)

        if buyback_capacity is not None and buyback_capacity > 0:
            return True
        if (
            share_count_trend is not None
            and share_count_trend <= self.thresholds["buyback_share_count_trend_max"]
        ):
            return True
        if activist_present:
            return True
        return bool(
            expectations_supportive
            and ev_z is not None
            and ev_z < -1.0
            and fcf_percentile is not None
            and fcf_percentile > 0.7
        )

    def _expectations_context(self, features: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
        coverage = _to_float(_feature_value(features, "expectations.analyst_coverage_count"), None)
        revision_signal = _to_float(_feature_value(features, "expectations.revision_signal"), None)
        if coverage is not None and coverage < 0:
            coverage = None
        return coverage, revision_signal

    def _expectations_allow_capital_return(self, features: Dict[str, Any]) -> bool:
        coverage, revision_signal = self._expectations_context(features)
        if coverage is None or coverage < self.thresholds["expectations_coverage_min"] or revision_signal is None:
            return True
        return revision_signal >= self.thresholds["expectations_revision_negative_max"]

    def _expectations_support_capital_return(self, features: Dict[str, Any]) -> bool:
        coverage, revision_signal = self._expectations_context(features)
        return bool(
            coverage is not None
            and coverage >= self.thresholds["expectations_coverage_min"]
            and revision_signal is not None
            and revision_signal >= self.thresholds["expectations_revision_positive_min"]
        )

    def _prefer_dividend_over_nonrecurring_capital_return(
        self, features: Dict[str, Any], liquidity_vs_mcap: float
    ) -> bool:
        if self._coverage_supported_dividend_initiate_override_profile(features):
            return not self._has_buyback_specific_support(features)
        if self._debt_bearing_dividend_initiate_override_profile(features):
            return True
        if liquidity_vs_mcap >= self.thresholds["buyback_existing_payer_liquidity_vs_mcap_override_min"]:
            return False
        if not self._healthy_dividend_increase_profile(features):
            return False

        last_dividend_event = str(
            _feature_value(features, "capital_return.last_dividend_event_type") or ""
        ).strip().lower()
        if last_dividend_event not in {"dividend_regular", "regular"}:
            return False
        return not self._has_buyback_specific_support(features)

    def _dividend_cut_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        if self._healthy_dividend_increase_profile(features):
            return False
        if self._buyback_reset_dividend_cut_profile(features):
            return True
        if self._preemptive_deleveraging_dividend_cut_profile(features):
            return True
        if self._dividend_cut_balance_sheet_pressure_profile(features):
            return True
        if not self._capital_return_blocked_by_financing_stress(
            features, allow_supportive_dividend_override=False
        ):
            return False

        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        cash = _to_float(_feature_value(features, "liquidity.cash"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)

        liquidity_squeeze = bool(
            (available_for_actions is not None and available_for_actions <= 0.0)
            or (
                cash is not None
                and minimum_cash is not None
                and minimum_cash > 0
                and cash <= minimum_cash
            )
        )
        debt_vs_mcap = None
        if total_debt is not None and market_cap is not None and market_cap > 0:
            debt_vs_mcap = total_debt / market_cap

        near_term_financing_squeeze = self._dividend_cut_near_term_financing_squeeze(features)

        return bool(
            (liquidity_squeeze or near_term_financing_squeeze)
            and net_debt is not None
            and net_debt > 0
            and net_leverage is not None
            and net_leverage >= self.thresholds["dividend_cut_net_leverage_min"]
            and interest_coverage is not None
            and interest_coverage <= self.thresholds["dividend_cut_interest_coverage_max"]
            and debt_vs_mcap is not None
            and debt_vs_mcap <= self.thresholds["dividend_cut_debt_vs_mcap_max"]
        )

    def _buyback_reset_dividend_cut_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        if self._regular_payer_financing_anomaly_override(features):
            return False

        last_action_type = str(_feature_value(features, "strategic.last_action_type") or "").strip().lower()
        if last_action_type != "buyback":
            return False

        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        cash = _to_float(_feature_value(features, "liquidity.cash"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)

        if (
            total_debt is None
            or total_debt <= 0
            or interest_coverage is None
            or interest_coverage > self.thresholds["dividend_cut_buyback_reset_interest_coverage_max"]
            or fcf_conversion is None
            or fcf_conversion > self.thresholds["dividend_cut_buyback_reset_fcf_conversion_max"]
            or market_cap is None
            or market_cap <= 0
        ):
            return False

        # This fallback is meant for sparse-liquidity regular payers that still
        # show debt burden and a clear cash-flow reset signal after recent buybacks.
        if available_for_actions is not None and available_for_actions > 0:
            return False
        if (
            cash is not None
            and minimum_cash is not None
            and minimum_cash > 0
            and cash > minimum_cash
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        if (
            total_debt_vs_mcap < self.thresholds["dividend_cut_buyback_reset_total_debt_vs_mcap_min"]
            or total_debt_vs_mcap > self.thresholds["dividend_cut_buyback_reset_total_debt_vs_mcap_max"]
        ):
            return False

        if debt_due_24m is not None and debt_due_24m > 0:
            debt_due_vs_mcap = debt_due_24m / market_cap
            if debt_due_vs_mcap > self.thresholds["dividend_cut_buyback_reset_debt_due_vs_mcap_max"]:
                return False
        elif (
            maturity_ratio is not None
            and maturity_ratio > self.thresholds["dividend_cut_buyback_reset_maturity_wall_ratio_24m_max"]
        ):
            return False

        return True

    def _dividend_cut_balance_sheet_pressure_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        if self._regular_payer_financing_anomaly_override(features):
            return False

        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        cash = _to_float(_feature_value(features, "liquidity.cash"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        debt_due_24m = self._effective_debt_due_24m(features)

        if (
            available_for_actions is None
            or available_for_actions > 0.0
            or cash is None
            or minimum_cash is None
            or minimum_cash <= 0
            or cash > minimum_cash
            or net_debt is None
            or net_debt <= 0
            or total_debt is None
            or total_debt <= 0
            or net_leverage is None
            or net_leverage < self.thresholds["dividend_cut_balance_sheet_pressure_net_leverage_min"]
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_cut_balance_sheet_pressure_interest_coverage_min"]
            or fcf_conversion is None
            or fcf_conversion < self.thresholds["dividend_cut_balance_sheet_pressure_fcf_conversion_min"]
            or market_cap is None
            or market_cap <= 0
            or drawdown_90d is None
            or drawdown_90d > self.thresholds["dividend_cut_balance_sheet_pressure_drawdown_90d_max"]
        ):
            return False

        net_debt_vs_mcap = net_debt / market_cap
        total_debt_vs_mcap = total_debt / market_cap
        debt_due_vs_mcap = (debt_due_24m / market_cap) if debt_due_24m is not None and debt_due_24m > 0 else 0.0
        return bool(
            net_debt_vs_mcap >= self.thresholds["dividend_cut_balance_sheet_pressure_net_debt_vs_mcap_min"]
            and total_debt_vs_mcap >= self.thresholds["dividend_cut_balance_sheet_pressure_total_debt_vs_mcap_min"]
            and debt_due_vs_mcap <= self.thresholds["dividend_cut_balance_sheet_pressure_debt_due_vs_mcap_max"]
        )

    def _preemptive_deleveraging_dividend_cut_profile(self, features: Dict[str, Any]) -> bool:
        if not _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        if self._regular_payer_financing_anomaly_override(features):
            return False

        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)

        if (
            total_debt is None
            or total_debt <= 0
            or net_debt is None
            or net_debt <= 0
            or net_leverage is None
            or net_leverage < self.thresholds["dividend_cut_preemptive_net_leverage_min"]
            or interest_coverage is None
            or interest_coverage < self.thresholds["dividend_cut_preemptive_interest_coverage_min"]
            or fcf_conversion is None
            or fcf_conversion > self.thresholds["dividend_cut_preemptive_fcf_conversion_max"]
            or market_cap is None
            or market_cap <= 0
            or drawdown_90d is None
            or drawdown_90d > self.thresholds["dividend_cut_preemptive_drawdown_90d_max"]
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        if total_debt_vs_mcap < self.thresholds["dividend_cut_preemptive_total_debt_vs_mcap_min"]:
            return False

        debt_due_vs_mcap = (
            debt_due_24m / market_cap if debt_due_24m is not None and debt_due_24m > 0 else 0.0
        )
        return bool(
            debt_due_vs_mcap >= self.thresholds["dividend_cut_preemptive_debt_due_vs_mcap_min"]
            or (
                maturity_ratio is not None
                and maturity_ratio >= self.thresholds["dividend_cut_preemptive_maturity_wall_ratio_24m_min"]
            )
        )

    def _dividend_cut_near_term_financing_squeeze(self, features: Dict[str, Any]) -> bool:
        if self._regular_payer_financing_anomaly_override(features):
            return False
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)

        if (
            available_for_actions is None
            or debt_due_24m is None
            or debt_due_24m <= 0
            or market_cap is None
            or market_cap <= 0
        ):
            return False

        liquidity_cover = available_for_actions / debt_due_24m
        debt_due_vs_mcap = debt_due_24m / market_cap
        debt_due_vs_total_debt = None
        if total_debt is not None and total_debt > 0:
            debt_due_vs_total_debt = debt_due_24m / total_debt

        extreme_maturity_pressure = bool(
            (maturity_ratio is not None and maturity_ratio >= self.thresholds["dividend_cut_maturity_wall_ratio_24m_min"])
            or (
                debt_due_vs_total_debt is not None
                and debt_due_vs_total_debt >= self.thresholds["dividend_cut_debt_due_vs_total_debt_min"]
            )
        )

        return bool(
            liquidity_cover <= self.thresholds["dividend_cut_liquidity_cover_24m_max"]
            and debt_due_vs_mcap >= self.thresholds["dividend_cut_debt_due_vs_mcap_min"]
            and extreme_maturity_pressure
        )

    def _net_cash_maturity_override_profile(self, features: Dict[str, Any]) -> bool:
        if _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        if self._strategic_nonpayer_recap_profile(features):
            return False
        if self._nonpayer_recap_preference_profile(features):
            return False

        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        if net_debt is None or net_debt > 0:
            return False

        gross_leverage = _to_float(_feature_value(features, "capital_structure.gross_leverage"), None)
        if (
            gross_leverage is not None
            and gross_leverage > self.thresholds["capital_return_net_cash_maturity_override_gross_leverage_max"]
        ):
            return False

        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        if total_debt is not None and market_cap is not None and market_cap > 0:
            total_debt_vs_mcap = total_debt / market_cap
            if (
                total_debt_vs_mcap
                > self.thresholds["capital_return_net_cash_maturity_override_total_debt_vs_mcap_max"]
            ):
                return False

        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        if (
            interest_coverage is not None
            and interest_coverage
            < self.thresholds["capital_return_net_cash_maturity_override_interest_coverage_min"]
        ):
            return False

        return True

    def _buyback_supported_net_cash_override_profile(self, features: Dict[str, Any]) -> bool:
        if _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag")):
            return False
        if self._strategic_nonpayer_recap_profile(features):
            return False
        if self._nonpayer_recap_preference_profile(features):
            return False
        if not self._has_buyback_specific_support(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        if (
            market_cap is None
            or market_cap <= 0
            or available_for_actions is None
            or available_for_actions <= 0
        ):
            return False

        liquidity_vs_mcap = available_for_actions / market_cap
        if liquidity_vs_mcap < self.thresholds["capital_return_net_cash_liquidity_vs_mcap_min"]:
            return False

        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        if net_debt is None or net_debt > 0:
            return False

        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        if total_debt is not None and total_debt > 0:
            total_debt_vs_mcap = total_debt / market_cap
            if (
                total_debt_vs_mcap
                > self.thresholds["capital_return_net_cash_maturity_override_total_debt_vs_mcap_max"]
            ):
                return False

        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        if (
            interest_coverage is not None
            and interest_coverage
            < self.thresholds["capital_return_net_cash_maturity_override_interest_coverage_min"]
        ):
            return False

        return True

    def _missing_market_cap_dividend_initiate_override_profile(self, features: Dict[str, Any]) -> bool:
        if not _dividend_initiation_nonpayer_signal(features):
            return False
        if self._rating_pressure(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        if market_cap is not None and market_cap > 0:
            return False

        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        if (
            available_for_actions is None
            or available_for_actions
            < self.thresholds["capital_return_missing_market_cap_dividend_initiate_liquidity_min_usd"]
        ):
            return False

        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        if (
            interest_coverage is None
            or interest_coverage
            < self.thresholds["capital_return_missing_market_cap_dividend_initiate_relaxed_interest_coverage_min"]
        ):
            return False

        strong_coverage = (
            interest_coverage
            >= self.thresholds["capital_return_missing_market_cap_dividend_initiate_interest_coverage_min"]
        )
        if not strong_coverage:
            cash = _to_float(_feature_value(features, "liquidity.cash"), None)
            minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
            if (
                cash is None
                or minimum_cash is None
                or minimum_cash <= 0
                or cash
                < minimum_cash
                * self.thresholds["capital_return_missing_market_cap_dividend_initiate_cash_buffer_multiple_min"]
            ):
                return False
            if not self._has_buyback_specific_support(features):
                return False

        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        if (
            maturity_ratio is not None
            and maturity_ratio
            > self.thresholds["capital_return_missing_market_cap_dividend_initiate_maturity_wall_ratio_24m_max"]
        ):
            return False

        debt_due_24m = self._effective_debt_due_24m(features)
        if debt_due_24m is not None and debt_due_24m > 0:
            liquidity_cover = available_for_actions / debt_due_24m
            if (
                liquidity_cover
                < self.thresholds["capital_return_missing_market_cap_dividend_initiate_liquidity_cover_24m_min"]
            ):
                return False

        free_cash_flow = _to_float(_feature_value(features, "cash_flow.free_cash_flow_ttm"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        if (free_cash_flow is None or free_cash_flow <= 0) and (fcf_conversion is None or fcf_conversion <= 0):
            return False

        return True

    def _coverage_supported_dividend_initiate_override_profile(self, features: Dict[str, Any]) -> bool:
        if not _dividend_initiation_nonpayer_signal(features):
            return False
        if self._rating_pressure(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        if (
            market_cap is None
            or market_cap <= 0
            or available_for_actions is None
            or available_for_actions
            < self.thresholds["capital_return_coverage_supported_dividend_initiate_liquidity_min_usd"]
        ):
            return False

        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        if (
            interest_coverage is None
            or interest_coverage
            < self.thresholds["capital_return_coverage_supported_dividend_initiate_interest_coverage_min"]
        ):
            return False

        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        if (
            equity_window is not None
            and equity_window
            < self.thresholds["capital_return_coverage_supported_dividend_initiate_equity_window_min"]
        ):
            return False

        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        if total_debt is not None and total_debt > 0:
            total_debt_vs_mcap = total_debt / market_cap
            if (
                total_debt_vs_mcap
                > self.thresholds["capital_return_coverage_supported_dividend_initiate_total_debt_vs_mcap_max"]
            ):
                return False

        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        if net_debt is not None and net_debt > 0:
            net_debt_vs_mcap = net_debt / market_cap
            if (
                net_debt_vs_mcap
                > self.thresholds["capital_return_coverage_supported_dividend_initiate_net_debt_vs_mcap_max"]
            ):
                return False

        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        if (
            maturity_ratio is not None
            and maturity_ratio
            > self.thresholds["capital_return_coverage_supported_dividend_initiate_maturity_wall_ratio_24m_max"]
        ):
            return False

        debt_due_24m = self._effective_debt_due_24m(features)
        if debt_due_24m is not None and debt_due_24m > 0:
            liquidity_cover = available_for_actions / debt_due_24m
            if (
                liquidity_cover
                < self.thresholds["capital_return_coverage_supported_dividend_initiate_liquidity_cover_24m_min"]
            ):
                return False

        free_cash_flow = _to_float(_feature_value(features, "cash_flow.free_cash_flow_ttm"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        return bool(
            (free_cash_flow is not None and free_cash_flow > 0)
            or (fcf_conversion is not None and fcf_conversion > 0)
        )

    def _debt_bearing_dividend_initiate_override_profile(self, features: Dict[str, Any]) -> bool:
        if not _dividend_initiation_nonpayer_signal(features):
            return False
        if self._rating_pressure(features):
            return False

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        free_cash_flow = _to_float(_feature_value(features, "cash_flow.free_cash_flow_ttm"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        last_action_type = str(_feature_value(features, "strategic.last_action_type") or "").strip().lower()

        if (
            market_cap is None
            or market_cap <= 0
            or available_for_actions is None
            or available_for_actions
            < self.thresholds["capital_return_debt_bearing_dividend_initiate_liquidity_min_usd"]
            or total_debt is None
            or total_debt <= 0
            or net_debt is None
            or net_debt <= 0
            or net_leverage is None
            or net_leverage > self.thresholds["capital_return_debt_bearing_dividend_initiate_net_leverage_max"]
            or interest_coverage is None
            or interest_coverage
            < self.thresholds["capital_return_debt_bearing_dividend_initiate_interest_coverage_min"]
        ):
            return False

        total_debt_vs_mcap = total_debt / market_cap
        net_debt_vs_mcap = net_debt / market_cap
        if (
            total_debt_vs_mcap
            > self.thresholds["capital_return_debt_bearing_dividend_initiate_total_debt_vs_mcap_max"]
            or net_debt_vs_mcap
            > self.thresholds["capital_return_debt_bearing_dividend_initiate_net_debt_vs_mcap_max"]
        ):
            return False

        if (
            equity_window is not None
            and equity_window
            < self.thresholds["capital_return_debt_bearing_dividend_initiate_equity_window_min"]
        ):
            return False

        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        if (
            maturity_ratio is not None
            and maturity_ratio
            > self.thresholds["capital_return_debt_bearing_dividend_initiate_maturity_wall_ratio_24m_max"]
        ):
            return False

        debt_due_24m = self._effective_debt_due_24m(features)
        if debt_due_24m is not None and debt_due_24m > 0:
            debt_due_vs_mcap = debt_due_24m / market_cap
            if (
                debt_due_vs_mcap
                > self.thresholds["capital_return_debt_bearing_dividend_initiate_debt_due_vs_mcap_max"]
            ):
                return False

        if last_action_type != "buyback":
            return False
        if not self._has_buyback_specific_support(features):
            return False
        if (free_cash_flow is None or free_cash_flow <= 0) and (
            fcf_conversion is None
            or fcf_conversion < self.thresholds["capital_return_debt_bearing_dividend_initiate_fcf_conversion_min"]
        ):
            return False

        return True

    def _capital_return_blocked_by_financing_stress(
        self,
        features: Dict[str, Any],
        *,
        allow_supportive_dividend_override: bool = True,
    ) -> bool:
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)

        debt_due_24m = self._effective_debt_due_24m(features)
        rating_pressure = self._rating_pressure(features)

        leverage_stress = net_leverage is not None and net_leverage >= self.thresholds["net_leverage_high"]
        maturity_stress = maturity_ratio is not None and maturity_ratio >= self.thresholds["maturity_wall_ratio_24m"]
        coverage_stress = (
            interest_coverage is not None
            and interest_coverage <= self.thresholds["capital_return_interest_coverage_min"]
        )
        liquidity_cover_stress = False
        if available_for_actions is not None and debt_due_24m is not None and debt_due_24m > 0:
            liquidity_cover_stress = (
                available_for_actions / debt_due_24m
                <= self.thresholds["capital_return_liquidity_cover_24m_min"]
            )

        # Net-cash payers with strong coverage should not be forced into financing stress
        # solely because debt-due bookkeeping makes maturity metrics look artificially tight.
        if self._healthy_dividend_increase_profile(features):
            maturity_stress = False
            liquidity_cover_stress = False
        if self._missing_schedule_dividend_increase_profile(features):
            leverage_stress = False
            coverage_stress = False
            maturity_stress = False
            liquidity_cover_stress = False
        if allow_supportive_dividend_override and self._supportive_regular_payer_dividend_increase_profile(features):
            leverage_stress = False
            coverage_stress = False
            maturity_stress = False
            liquidity_cover_stress = False
        if self._regular_payer_financing_anomaly_override(features):
            maturity_stress = False
            liquidity_cover_stress = False
        if self._net_cash_maturity_override_profile(features):
            leverage_stress = False
            coverage_stress = False
            maturity_stress = False
            liquidity_cover_stress = False
        if self._buyback_supported_net_cash_override_profile(features):
            leverage_stress = False
            coverage_stress = False
            maturity_stress = False
            liquidity_cover_stress = False
        if self._missing_market_cap_dividend_initiate_override_profile(features):
            leverage_stress = False
        coverage_supported_dividend_initiate = self._coverage_supported_dividend_initiate_override_profile(features)
        debt_bearing_dividend_initiate = self._debt_bearing_dividend_initiate_override_profile(features)
        if coverage_supported_dividend_initiate:
            leverage_stress = False
            coverage_stress = False
            maturity_stress = False
            liquidity_cover_stress = False
        if debt_bearing_dividend_initiate:
            leverage_stress = False
            coverage_stress = False
            maturity_stress = False
            liquidity_cover_stress = False
        market_shutdown_stress = self._market_shutdown_regular_payer_recap_profile(features)
        buyback_recap_stress = self._buyback_regular_payer_recap_profile(features)
        sparse_reset_recap_stress = self._sparse_reset_recap_profile(features)
        if coverage_supported_dividend_initiate:
            sparse_reset_recap_stress = False
        if debt_bearing_dividend_initiate:
            sparse_reset_recap_stress = False

        return bool(
            leverage_stress
            or maturity_stress
            or coverage_stress
            or liquidity_cover_stress
            or rating_pressure
            or market_shutdown_stress
            or buyback_recap_stress
            or sparse_reset_recap_stress
        )

    def _effective_debt_due_24m(self, features: Dict[str, Any]) -> Optional[float]:
        debt_due_24m = _to_float(_feature_value(features, "capital_structure.debt_due_next_24m"), None)
        if debt_due_24m is not None:
            return debt_due_24m

        debt_due_0_12m = _to_float(_feature_value(features, "capital_structure.debt_due_0_12m"), 0.0) or 0.0
        debt_due_12_24m = _to_float(_feature_value(features, "capital_structure.debt_due_12_24m"), 0.0) or 0.0
        if debt_due_0_12m > 0 or debt_due_12_24m > 0:
            return debt_due_0_12m + debt_due_12_24m

        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        if maturity_ratio is not None and total_debt is not None and total_debt > 0:
            return maturity_ratio * total_debt
        return None

    def _gen_dividend_policy_actions(self, run: RecommendationRun, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self._healthy_dividend_increase_profile(features):
            return []

        schema = self.registry.get_action("capital_return.dividend_increase")
        if schema is None:
            return []

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        annualized_commitment = []
        if market_cap is not None and market_cap > 0:
            annualized_commitment = [market_cap * x for x in (0.0025, 0.005, 0.01)]

        overrides: Dict[str, Any] = {"percent_change": [0.05, 0.1, 0.15]}
        if annualized_commitment:
            overrides["annualized_cash_commitment_usd"] = annualized_commitment

        out: List[Dict[str, Any]] = []
        for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=18):
            out.append(
                self._build_candidate_raw(
                    run_id=run.run_id,
                    action_schema=schema,
                    parameters=params,
                    generation_source="deterministic_rule",
                    rationale_refs=[
                        {
                            "reference_type": "state_feature",
                            "reference_id": "capital_return.dividend_payer_flag",
                            "explanation": "Existing dividend payer with durable cash generation supports measured dividend increases.",
                        },
                        {
                            "reference_type": "state_feature",
                            "reference_id": "operating.fcf_conversion",
                            "explanation": "Healthy cash-flow conversion supports sustaining and modestly increasing recurring distributions.",
                        },
                    ],
                    trigger_strength=0.72,
                    playbook_relevance=0.7,
                )
            )
        return out

    def _gen_dividend_missing_schedule_actions(
        self, run: RecommendationRun, features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not self._missing_schedule_dividend_increase_profile(features):
            return []

        schema = self.registry.get_action("capital_return.dividend_increase")
        if schema is None:
            return []

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        annualized_commitment = []
        if market_cap is not None and market_cap > 0:
            annualized_commitment = [market_cap * x for x in (0.001, 0.0025, 0.005)]

        overrides: Dict[str, Any] = {"percent_change": [0.02, 0.05, 0.08]}
        if annualized_commitment:
            overrides["annualized_cash_commitment_usd"] = annualized_commitment

        out: List[Dict[str, Any]] = []
        for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=18):
            out.append(
                self._build_candidate_raw(
                    run_id=run.run_id,
                    action_schema=schema,
                    parameters=params,
                    generation_source="deterministic_rule",
                    rationale_refs=[
                        {
                            "reference_type": "state_feature",
                            "reference_id": "capital_return.last_dividend_event_type",
                            "explanation": "Established regular payer with no observed maturity schedule pressure still merits a cautious dividend-increase candidate.",
                        },
                        {
                            "reference_type": "state_feature",
                            "reference_id": "operating.fcf_conversion",
                            "explanation": "Strong cash conversion and long runway argue against treating an isolated leverage spike as the only decision signal.",
                        },
                    ],
                    trigger_strength=0.4,
                    playbook_relevance=0.52,
                )
            )
        return out

    def _gen_dividend_coverage_supported_actions(
        self, run: RecommendationRun, features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not (
            self._supportive_regular_payer_dividend_increase_profile(features)
        ):
            return []

        schema = self.registry.get_action("capital_return.dividend_increase")
        if schema is None:
            return []

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        annualized_commitment = []
        if market_cap is not None and market_cap > 0:
            annualized_commitment = [market_cap * x for x in (0.001, 0.0025, 0.005)]

        overrides: Dict[str, Any] = {"percent_change": [0.02, 0.04, 0.06]}
        if annualized_commitment:
            overrides["annualized_cash_commitment_usd"] = annualized_commitment

        out: List[Dict[str, Any]] = []
        for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=12):
            out.append(
                self._build_candidate_raw(
                    run_id=run.run_id,
                    action_schema=schema,
                    parameters=params,
                    generation_source="deterministic_rule",
                    rationale_refs=[
                        {
                            "reference_type": "state_feature",
                            "reference_id": "capital_structure.interest_coverage",
                            "explanation": "Established regular payer still shows enough coverage to support a continuity-style dividend increase despite leverage pressure.",
                        },
                        {
                            "reference_type": "state_feature",
                            "reference_id": "market.equity_window_proxy",
                            "explanation": "Open market access and positive deployable liquidity reduce the need to default into financing-only playbooks.",
                        },
                    ],
                    trigger_strength=0.48,
                    playbook_relevance=0.58,
                )
            )
        return out

    def _gen_dividend_balance_sheet_light_actions(
        self, run: RecommendationRun, features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not self._balance_sheet_light_regular_payer_dividend_increase_profile(features):
            return []

        schema = self.registry.get_action("capital_return.dividend_increase")
        if schema is None:
            return []

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        annualized_commitment = []
        if market_cap is not None and market_cap > 0:
            annualized_commitment = [market_cap * x for x in (0.001, 0.0025, 0.005)]

        overrides: Dict[str, Any] = {"percent_change": [0.02, 0.04, 0.06]}
        if annualized_commitment:
            overrides["annualized_cash_commitment_usd"] = annualized_commitment

        out: List[Dict[str, Any]] = []
        for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=12):
            out.append(
                self._build_candidate_raw(
                    run_id=run.run_id,
                    action_schema=schema,
                    parameters=params,
                    generation_source="deterministic_rule",
                    rationale_refs=[
                        {
                            "reference_type": "state_feature",
                            "reference_id": "capital_return.last_dividend_event_type",
                            "explanation": "An established regular payer with a light debt burden should still surface a continuity-style dividend option even when some coverage fields are sparse.",
                        },
                        {
                            "reference_type": "state_feature",
                            "reference_id": "market.equity_window_proxy",
                            "explanation": "A still-open market window and modest balance-sheet pressure reduce the need to collapse into financing-only playbooks from incomplete inputs.",
                        },
                    ],
                    trigger_strength=0.46,
                    playbook_relevance=0.56,
                )
            )
        return out

    def _gen_dividend_continuity_actions(self, run: RecommendationRun, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self._continuity_dividend_increase_profile(features):
            return []

        schema = self.registry.get_action("capital_return.dividend_increase")
        if schema is None:
            return []

        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        annualized_commitment = []
        if market_cap is not None and market_cap > 0:
            annualized_commitment = [market_cap * x for x in (0.001, 0.002, 0.0035)]

        overrides: Dict[str, Any] = {"percent_change": [0.02, 0.04, 0.06]}
        if annualized_commitment:
            overrides["annualized_cash_commitment_usd"] = annualized_commitment

        out: List[Dict[str, Any]] = []
        for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=12):
            out.append(
                self._build_candidate_raw(
                    run_id=run.run_id,
                    action_schema=schema,
                    parameters=params,
                    generation_source="deterministic_rule",
                    rationale_refs=[
                        {
                            "reference_type": "state_feature",
                            "reference_id": "capital_return.last_dividend_event_type",
                            "explanation": "An established regular payer with no near-term maturity burden can still support a modest continuity-style dividend increase.",
                        },
                        {
                            "reference_type": "state_feature",
                            "reference_id": "liquidity.runway_months",
                            "explanation": "Long runway and strong coverage reduce the risk of a small recurring dividend step despite market dislocation.",
                        },
                    ],
                    trigger_strength=0.34,
                    playbook_relevance=0.45,
                )
            )
        return out

    def _gen_special_dividend_actions(self, run: RecommendationRun, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self._special_dividend_profile(features):
            return []

        schema = self.registry.get_action("capital_return.special_dividend")
        if schema is None:
            return []

        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        if available_for_actions is None or available_for_actions <= 0 or market_cap is None or market_cap <= 0:
            return []

        size_cap = min(available_for_actions, market_cap * 0.10)
        size_grid = sorted(
            {
                round(available_for_actions * fraction, 2)
                for fraction in (0.25, 0.5, 0.75)
                if available_for_actions * fraction <= size_cap
            }
            | {round(size_cap, 2)}
        )
        if not size_grid:
            return []

        overrides = {
            "size_absolute_usd": size_grid,
            "funding_mix": [{"cash": 1.0, "debt": 0.0, "equity": 0.0}],
        }

        liquidity_vs_mcap = available_for_actions / market_cap
        out: List[Dict[str, Any]] = []
        for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=12):
            out.append(
                self._build_candidate_raw(
                    run_id=run.run_id,
                    action_schema=schema,
                    parameters=params,
                    generation_source="deterministic_rule",
                    rationale_refs=[
                        {
                            "reference_type": "state_feature",
                            "reference_id": "liquidity.available_for_actions",
                            "explanation": "Net-cash regular payer with excess liquidity supports a one-time special dividend despite muted financing windows.",
                        }
                    ],
                    trigger_strength=_clip(
                        (liquidity_vs_mcap - self.thresholds["special_dividend_liquidity_vs_mcap_min"]) / 0.08,
                        0.55,
                        0.9,
                    ),
                    playbook_relevance=0.72,
                )
            )
        return out

    def _gen_dividend_cut_actions(self, run: RecommendationRun, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self._dividend_cut_profile(features):
            return []

        schema = self.registry.get_action("capital_return.dividend_cut")
        if schema is None:
            return []

        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), 0.0) or 0.0
        cash = _to_float(_feature_value(features, "liquidity.cash"), 0.0) or 0.0
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), 0.0) or 0.0
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), 0.0) or 0.0
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), 0.0) or 0.0
        debt_due_24m = self._effective_debt_due_24m(features)
        maturity_ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)

        cash_gap = 0.0
        if minimum_cash > 0:
            cash_gap = _clip((minimum_cash - cash) / minimum_cash, 0.0, 1.0)

        liquidity_cover_signal = 0.0
        if available_for_actions > 0 and debt_due_24m is not None and debt_due_24m > 0:
            liquidity_cover = available_for_actions / debt_due_24m
            liquidity_cover_signal = _clip(
                (self.thresholds["dividend_cut_liquidity_cover_24m_max"] - liquidity_cover)
                / self.thresholds["dividend_cut_liquidity_cover_24m_max"],
                0.0,
                1.0,
            )

        maturity_signal = 0.0
        if maturity_ratio is not None:
            maturity_signal = _clip(
                (maturity_ratio - self.thresholds["dividend_cut_maturity_wall_ratio_24m_min"])
                / self.thresholds["dividend_cut_maturity_wall_ratio_24m_min"],
                0.0,
                1.0,
            )

        trigger_strength = max(
            _clip(
                (net_leverage - self.thresholds["dividend_cut_net_leverage_min"]) / 3.0,
                0.0,
                1.0,
            ),
            _clip(
                (self.thresholds["dividend_cut_interest_coverage_max"] - interest_coverage)
                / self.thresholds["dividend_cut_interest_coverage_max"],
                0.0,
                1.0,
            ),
            cash_gap,
            liquidity_cover_signal,
            maturity_signal,
            1.0 if available_for_actions <= 0 else 0.0,
        )

        overrides: Dict[str, Any] = {
            "percent_change": [-1.0, -0.75, -0.5, -0.25],
            "target_use_of_cash": ["liquidity_buffer", "deleveraging"],
            "effective_quarter": ["Q1", "Q2", "Q3", "Q4"],
        }

        out: List[Dict[str, Any]] = []
        for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=24):
            out.append(
                self._build_candidate_raw(
                    run_id=run.run_id,
                    action_schema=schema,
                    parameters=params,
                    generation_source="deterministic_rule",
                    rationale_refs=[
                        {
                            "reference_type": "state_feature",
                            "reference_id": "capital_return.dividend_payer_flag",
                            "explanation": "Existing dividend commitment can be resized to preserve cash under financing pressure.",
                        },
                        {
                            "reference_type": "state_feature",
                            "reference_id": "liquidity.available_for_actions",
                            "explanation": "Limited deployable liquidity relative to financing needs raises the value of preserving cash internally.",
                        },
                    ],
                    trigger_strength=trigger_strength,
                    playbook_relevance=0.78,
                )
            )
        return out

    def _gen_equity_backstop(self, run: RecommendationRun, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        credit_window = _to_float(_feature_value(features, "market.credit_window_proxy"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        cash = _to_float(_feature_value(features, "liquidity.cash"), None)
        minimum_cash = _to_float(_feature_value(features, "liquidity.minimum_cash_policy_proxy"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        net_debt = _to_float(_feature_value(features, "capital_structure.net_debt"), None)
        total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        drawdown_90d = _to_float(_feature_value(features, "market.drawdown_90d"), None)
        volatility_30d = _to_float(_feature_value(features, "market.volatility_30d"), None)
        volatility_90d = _to_float(_feature_value(features, "market.volatility_90d"), None)
        fcf_yield = _to_float(_feature_value(features, "market.fcf_yield"), None)
        ebitda_margin = _to_float(_feature_value(features, "operating.ebitda_margin_ttm"), None)
        fcf_conversion = _to_float(_feature_value(features, "operating.fcf_conversion"), None)
        revenue_yoy = _to_float(_feature_value(features, "operating.revenue_yoy_last_q"), None)
        market_shutdown_regular_payer_pressure = self._market_shutdown_regular_payer_recap_profile(features)
        strategic_regular_payer_recap_pressure = self._strategic_regular_payer_recap_profile(features)
        buyback_regular_payer_recap_pressure = self._buyback_regular_payer_recap_profile(features)
        strategic_nonpayer_recap_pressure = self._strategic_nonpayer_recap_profile(features)
        nonpayer_recap_preference_pressure = self._nonpayer_recap_preference_profile(features)
        sparse_reset_recap_pressure = self._sparse_reset_recap_profile(features)
        recap_equity_path = bool(
            market_shutdown_regular_payer_pressure
            or strategic_regular_payer_recap_pressure
            or buyback_regular_payer_recap_pressure
            or strategic_nonpayer_recap_pressure
            or nonpayer_recap_preference_pressure
            or sparse_reset_recap_pressure
        )

        if available_for_actions is None or market_cap is None or market_cap <= 0:
            return []
        if (
            not recap_equity_path
            and (
                credit_window is None
                or credit_window > self.thresholds["equity_backstop_credit_window_max"]
            )
        ):
            return []

        public_equity_window = (
            equity_window is not None
            and equity_window >= self.thresholds["equity_backstop_equity_window_min"]
        )
        relaxed_public_equity_window = (
            equity_window is not None
            and equity_window >= self.thresholds["equity_backstop_relaxed_equity_window_min"]
        )
        distressed_private_placement_window = (
            equity_window is not None
            and equity_window <= self.thresholds["equity_private_placement_equity_window_max"]
        )
        if not (
            public_equity_window
            or distressed_private_placement_window
            or relaxed_public_equity_window
            or nonpayer_recap_preference_pressure
            or sparse_reset_recap_pressure
        ):
            return []

        if self._healthy_dividend_increase_profile(features):
            return []
        if self._supportive_regular_payer_dividend_increase_profile(features):
            return []
        if self._dividend_cut_profile(features):
            return []
        if self._missing_schedule_dividend_increase_profile(features):
            return []
        regular_dividend_payer = bool(
            _is_explicit_true(_feature_value(features, "capital_return.dividend_payer_flag"))
            and str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
            in {"dividend_regular", "regular"}
        )

        liquidity_vs_mcap = available_for_actions / market_cap
        liquidity_vs_mcap_excess = liquidity_vs_mcap > self.thresholds["equity_backstop_liquidity_vs_mcap_max"]

        debt_burden = False
        if total_debt is not None and total_debt > 0:
            debt_burden = (
                available_for_actions / total_debt
                <= self.thresholds["equity_backstop_liquidity_to_debt_max"]
            )

        debt_due_pressure = False
        if debt_due_24m is not None and debt_due_24m > 0:
            debt_due_pressure = (
                available_for_actions / debt_due_24m
                <= self.thresholds["equity_backstop_liquidity_cover_24m_max"]
            )

        coverage_pressure = (
            interest_coverage is not None
            and interest_coverage <= self.thresholds["equity_backstop_interest_coverage_max"]
        )
        leverage_pressure = (
            net_leverage is not None
            and net_leverage >= self.thresholds["equity_backstop_net_leverage_min"]
        )
        debt_vs_mcap = None
        if total_debt is not None and total_debt > 0:
            debt_vs_mcap = total_debt / market_cap

        liquidity_squeeze = bool(
            available_for_actions <= 0
            or (
                cash is not None
                and minimum_cash is not None
                and minimum_cash > 0
                and cash < minimum_cash
            )
        )
        operating_distress = bool(
            (ebitda_margin is not None and ebitda_margin <= 0.0)
            or (fcf_conversion is not None and fcf_conversion <= 0.0)
        )
        market_dislocation = bool(
            (drawdown_90d is not None and drawdown_90d <= self.thresholds["equity_private_placement_drawdown_90d_max"])
            or (
                volatility_30d is not None
                and volatility_30d >= self.thresholds["equity_private_placement_volatility_30d_min"]
            )
        )
        severe_market_dislocation = bool(
            drawdown_90d is not None
            and drawdown_90d <= self.thresholds["equity_backstop_relaxed_drawdown_90d_max"]
        )
        distressed_private_placement_pressure = bool(
            distressed_private_placement_window
            and liquidity_squeeze
            and operating_distress
            and market_dislocation
            and (
                (
                    interest_coverage is not None
                    and interest_coverage <= self.thresholds["equity_private_placement_interest_coverage_max"]
                )
                or (
                    debt_vs_mcap is not None
                    and debt_vs_mcap >= self.thresholds["equity_private_placement_debt_vs_mcap_min"]
                )
            )
        )
        distressed_nonpayer_public_pressure = bool(
            not regular_dividend_payer
            and public_equity_window
            and operating_distress
            and severe_market_dislocation
            and interest_coverage is not None
            and interest_coverage <= self.thresholds["equity_backstop_distressed_nonpayer_interest_coverage_max"]
            and debt_vs_mcap is not None
            and debt_vs_mcap >= self.thresholds["equity_backstop_distressed_nonpayer_public_debt_vs_mcap_min"]
        )
        distressed_nonpayer_private_placement_pressure = bool(
            not regular_dividend_payer
            and distressed_private_placement_window
            and operating_distress
            and severe_market_dislocation
            and interest_coverage is not None
            and interest_coverage <= self.thresholds["equity_backstop_distressed_nonpayer_interest_coverage_max"]
        )
        relaxed_public_equity_pressure = bool(
            not regular_dividend_payer
            and not public_equity_window
            and relaxed_public_equity_window
            and debt_burden
            and severe_market_dislocation
            and not operating_distress
            and not liquidity_vs_mcap_excess
        )

        runway_equity_pressure = self._runway_equity_pressure_profile(features)
        if (
            regular_dividend_payer
            and not market_shutdown_regular_payer_pressure
            and not strategic_regular_payer_recap_pressure
            and not buyback_regular_payer_recap_pressure
            and not self._capital_return_blocked_by_financing_stress(features)
            and not runway_equity_pressure
        ):
            return []
        if liquidity_vs_mcap_excess and not (
            distressed_nonpayer_public_pressure
            or distressed_nonpayer_private_placement_pressure
            or market_shutdown_regular_payer_pressure
            or strategic_regular_payer_recap_pressure
            or buyback_regular_payer_recap_pressure
            or strategic_nonpayer_recap_pressure
            or nonpayer_recap_preference_pressure
            or sparse_reset_recap_pressure
        ):
            return []
        public_equity_pressure = bool(
            public_equity_window
            and (
                debt_burden
                or debt_due_pressure
                or coverage_pressure
                or leverage_pressure
                or runway_equity_pressure
            )
        )
        regular_payer_shutdown_public_pressure = bool(
            market_shutdown_regular_payer_pressure and public_equity_window
        )
        regular_payer_shutdown_private_pressure = bool(
            market_shutdown_regular_payer_pressure and distressed_private_placement_window
        )
        strategic_regular_payer_public_pressure = bool(
            strategic_regular_payer_recap_pressure
            and equity_window >= self.thresholds["strategic_regular_payer_recap_equity_window_min"]
        )
        buyback_regular_payer_public_pressure = bool(
            buyback_regular_payer_recap_pressure
            and equity_window >= self.thresholds["buyback_regular_payer_recap_equity_window_min"]
        )
        strategic_nonpayer_public_pressure = bool(
            strategic_nonpayer_recap_pressure
            and equity_window >= self.thresholds["strategic_nonpayer_recap_equity_window_min"]
        )
        nonpayer_recap_public_pressure = bool(
            nonpayer_recap_preference_pressure
            and equity_window >= self.thresholds["nonpayer_recap_preference_equity_window_min"]
        )
        sparse_reset_public_pressure = bool(
            sparse_reset_recap_pressure
            and equity_window >= self.thresholds["sparse_reset_recap_equity_window_min"]
        )
        nonpayer_recap_private_pressure = bool(
            nonpayer_recap_preference_pressure
            and 0.0 < equity_window <= self.thresholds["equity_private_placement_equity_window_max"]
        )
        public_equity_pressure = bool(
            public_equity_pressure
            or distressed_nonpayer_public_pressure
            or relaxed_public_equity_pressure
            or regular_payer_shutdown_public_pressure
            or strategic_regular_payer_public_pressure
            or buyback_regular_payer_public_pressure
            or strategic_nonpayer_public_pressure
            or nonpayer_recap_public_pressure
            or sparse_reset_public_pressure
        )
        distressed_private_placement_pressure = bool(
            distressed_private_placement_pressure
            or distressed_nonpayer_private_placement_pressure
            or regular_payer_shutdown_private_pressure
            or nonpayer_recap_private_pressure
        )

        if not (public_equity_pressure or distressed_private_placement_pressure):
            return []

        base_raise_sizes = (0.02, 0.05, 0.1)
        if runway_equity_pressure and not (debt_burden or debt_due_pressure or leverage_pressure):
            base_raise_sizes = (0.01, 0.02, 0.05)
        elif distressed_private_placement_pressure:
            base_raise_sizes = (0.25, 0.5, 1.0)
        amount_candidates = [market_cap * x for x in base_raise_sizes]
        if total_debt is not None and total_debt > 0:
            amount_candidates.append(total_debt * 0.5)
        if debt_due_24m is not None and debt_due_24m > 0:
            amount_candidates.append(debt_due_24m)
        amount_candidates = sorted({max(1_000_000.0, float(x)) for x in amount_candidates})

        use_of_proceeds = ["general_corporate", "liquidity_buffer"]
        if debt_burden or debt_due_pressure or leverage_pressure:
            use_of_proceeds = ["deleveraging", "liquidity_buffer", "general_corporate"]
        if distressed_private_placement_pressure:
            use_of_proceeds = ["liquidity_buffer", "deleveraging", "general_corporate"]

        schema = self.registry.get_action("capital_structure.equity_issuance")
        if schema is None:
            return []

        trigger_strength = max(
            _clip(
                (self.thresholds["equity_backstop_liquidity_vs_mcap_max"] - liquidity_vs_mcap)
                / self.thresholds["equity_backstop_liquidity_vs_mcap_max"],
                0.0,
                1.0,
            ),
            _clip(
                (
                    (self.thresholds["equity_backstop_credit_window_max"] - credit_window)
                    / max(self.thresholds["equity_backstop_credit_window_max"], 1e-9)
                )
                if credit_window is not None
                else 0.0,
                0.0,
                1.0,
            ),
            _clip(
                (self.thresholds["equity_private_placement_interest_coverage_max"] - interest_coverage)
                / max(self.thresholds["equity_private_placement_interest_coverage_max"], 1e-9),
                0.0,
                1.0,
            )
            if interest_coverage is not None
            else 0.0,
            _clip(
                (abs(drawdown_90d) - abs(self.thresholds["equity_runway_drawdown_90d_max"])) / 0.25,
                0.0,
                1.0,
            )
            if drawdown_90d is not None
            else 0.0,
        )

        out: List[Dict[str, Any]] = []
        overrides = {
            "amount_usd": amount_candidates,
            "use_of_proceeds": use_of_proceeds,
            "offering_type": (
                ["private_placement"]
                if distressed_private_placement_pressure
                else ["follow_on", "at_the_market"]
            ),
        }
        rationale_refs = [
            {
                "reference_type": "state_feature",
                "reference_id": "market.credit_window_proxy",
                "explanation": "Weak debt-market access with a usable equity window supports equity backstop options.",
            },
            {
                "reference_type": "state_feature",
                "reference_id": "liquidity.available_for_actions",
                "explanation": "Limited deployable liquidity versus needs argues for a balance-sheet backstop.",
            },
        ]
        if runway_equity_pressure:
            rationale_refs.append(
                {
                    "reference_type": "state_feature",
                    "reference_id": "market.drawdown_90d",
                    "explanation": "Severe equity volatility with only a modest cash cushion favors preemptive equity financing before optionality narrows further.",
                }
            )
        if distressed_private_placement_pressure:
            rationale_refs.append(
                {
                    "reference_type": "state_feature",
                    "reference_id": "market.equity_window_proxy",
                    "explanation": "With the public equity window effectively shut, any viable raise likely needs a private-placement style recapitalization.",
                }
            )
        elif distressed_nonpayer_public_pressure or relaxed_public_equity_pressure:
            rationale_refs.append(
                {
                    "reference_type": "state_feature",
                    "reference_id": "market.equity_window_proxy",
                    "explanation": "A partially open equity window can still support a follow-on raise when debt markets are shut and financing flexibility is the binding issue.",
                }
            )
        elif nonpayer_recap_public_pressure:
            rationale_refs.append(
                {
                    "reference_type": "state_feature",
                    "reference_id": "capital_structure.debt_due_next_24m",
                    "explanation": "For non-payers with shut credit markets, recapitalization pressure can make public equity the cleaner primary tool than debt-only maturity management.",
                }
            )
        elif sparse_reset_public_pressure:
            rationale_refs.append(
                {
                    "reference_type": "state_feature",
                    "reference_id": "strategic.last_action_type",
                    "explanation": "When recent capital allocation leaned on buybacks, acquisitions, or debt-funded resets while leverage still burdens the balance sheet, an equity reset can be cleaner than continuing shareholder distributions.",
                }
            )
        elif strategic_nonpayer_public_pressure:
            rationale_refs.append(
                {
                    "reference_type": "state_feature",
                    "reference_id": "strategic.last_action_type",
                    "explanation": "For non-payers with recent balance-sheet or buyback activity, a shut credit market and still-usable equity window can make recapitalizing via equity cleaner than extending debt tools alone.",
                }
            )
        elif buyback_regular_payer_public_pressure:
            rationale_refs.append(
                {
                    "reference_type": "state_feature",
                    "reference_id": "strategic.last_action_type",
                    "explanation": "For regular payers that recently leaned on buybacks, shut credit markets and meaningful recap pressure can make equity issuance the cleaner reset than debt-only extensions.",
                }
            )
        elif regular_payer_shutdown_public_pressure:
            rationale_refs.append(
                {
                    "reference_type": "state_feature",
                    "reference_id": "market.equity_window_proxy",
                    "explanation": "When a regular payer faces a shut debt market and a severely impaired equity tape, preserving financing optionality can outweigh incremental capital return.",
                }
            )
        if distressed_nonpayer_public_pressure or distressed_nonpayer_private_placement_pressure:
            rationale_refs.append(
                {
                    "reference_type": "state_feature",
                    "reference_id": "capital_structure.interest_coverage",
                    "explanation": "Deep operating stress with broken coverage can justify equity recapitalization even before cash falls below policy minimums.",
                }
            )
        if nonpayer_recap_private_pressure:
            rationale_refs.append(
                {
                    "reference_type": "state_feature",
                    "reference_id": "market.equity_window_proxy",
                    "explanation": "When only a narrow equity window remains, non-payers under recap pressure may still need a private-placement style raise instead of debt-only extensions.",
                }
            )
        if market_shutdown_regular_payer_pressure:
            rationale_refs.append(
                {
                    "reference_type": "state_feature",
                    "reference_id": "market.drawdown_90d",
                    "explanation": "Severe market dislocation with limited external financing windows argues for protecting balance-sheet flexibility rather than distributing more capital.",
                }
            )
        for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=24):
            out.append(
                self._build_candidate_raw(
                    run_id=run.run_id,
                    action_schema=schema,
                    parameters=params,
                    generation_source="deterministic_rule",
                    rationale_refs=rationale_refs,
                    trigger_strength=trigger_strength,
                    playbook_relevance=0.72,
                )
            )
        return out

    def _has_clear_mna_capacity(self, features: Dict[str, Any]) -> bool:
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)
        net_leverage = _to_float(_feature_value(features, "capital_structure.net_leverage"), None)
        interest_coverage = _to_float(_feature_value(features, "capital_structure.interest_coverage"), None)
        credit_window = _to_float(_feature_value(features, "market.credit_window_proxy"), None)
        equity_window = _to_float(_feature_value(features, "market.equity_window_proxy"), None)

        debt_due_24m = _to_float(_feature_value(features, "capital_structure.debt_due_next_24m"), None)
        if debt_due_24m is None:
            debt_due_0_12m = _to_float(_feature_value(features, "capital_structure.debt_due_0_12m"), 0.0) or 0.0
            debt_due_12_24m = _to_float(_feature_value(features, "capital_structure.debt_due_12_24m"), 0.0) or 0.0
            if debt_due_0_12m > 0 or debt_due_12_24m > 0:
                debt_due_24m = debt_due_0_12m + debt_due_12_24m

        liquidity_ok = False
        if available_for_actions is not None and available_for_actions > 0 and market_cap is not None and market_cap > 0:
            liquidity_ok = (
                available_for_actions / market_cap >= self.thresholds["mna_liquidity_vs_mcap_min"]
            )
        if liquidity_ok and debt_due_24m is not None and debt_due_24m > 0:
            liquidity_ok = (
                available_for_actions is not None
                and available_for_actions / debt_due_24m >= self.thresholds["mna_liquidity_cover_24m_min"]
            )

        leverage_ok = net_leverage is None or net_leverage <= self.thresholds["mna_net_leverage_max"]
        coverage_ok = (
            interest_coverage is None
            or interest_coverage >= self.thresholds["mna_interest_coverage_min"]
        )

        market_windows = [value for value in (credit_window, equity_window) if value is not None]
        market_access_ok = True
        if market_windows:
            market_access_ok = max(market_windows) >= self.thresholds["mna_market_window_min"]

        return bool(liquidity_ok and leverage_ok and coverage_ok and market_access_ok)

    def _mna_blocked_by_financing_or_capacity(self, features: Dict[str, Any]) -> bool:
        return self._capital_return_blocked_by_financing_stress(features) or not self._has_clear_mna_capacity(features)

    def _gen_maturity_wall(self, run: RecommendationRun, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._healthy_dividend_increase_profile(features):
            return []
        if self._supportive_regular_payer_dividend_increase_profile(features):
            return []
        if self._dividend_cut_profile(features):
            return []
        if self._strategic_regular_payer_recap_profile(features):
            return []
        if self._buyback_regular_payer_recap_profile(features):
            return []
        if self._strategic_nonpayer_recap_profile(features):
            return []
        if self._nonpayer_recap_preference_profile(features):
            return []

        ratio = _to_float(_feature_value(features, "capital_structure.maturity_wall_ratio_24m"), None)
        debt_due_24m = self._effective_debt_due_24m(features)
        available_for_actions = _to_float(_feature_value(features, "liquidity.available_for_actions"), None)
        market_cap = _to_float(_feature_value(features, "market.market_cap"), None)

        ratio_trigger = ratio is not None and ratio > self.thresholds["maturity_wall_ratio_24m"]
        absolute_trigger = False
        if debt_due_24m is not None and debt_due_24m > 0:
            liquidity_cover = None
            if available_for_actions is not None:
                liquidity_cover = available_for_actions / debt_due_24m
            due_vs_mcap = None
            if market_cap is not None and market_cap > 0:
                due_vs_mcap = debt_due_24m / market_cap
            absolute_trigger = (
                liquidity_cover is not None
                and liquidity_cover <= self.thresholds["maturity_wall_liquidity_cover_24m_max"]
                and due_vs_mcap is not None
                and due_vs_mcap >= self.thresholds["maturity_wall_vs_mcap_min"]
            )

        if not (ratio_trigger or absolute_trigger):
            return []

        if debt_due_24m is None:
            total_debt = _to_float(_feature_value(features, "capital_structure.total_debt"), 0.0) or 0.0
            debt_due_24m = total_debt * float(ratio or 0.0)
        debt_due_24m = max(float(debt_due_24m or 0.0), 1.0)

        action_specs = {
            "capital_structure.refinancing": {
                "amount_refinanced_usd": [debt_due_24m * x for x in (0.25, 0.5, 0.75, 1.0)],
                "new_tenor_years": [3.0, 5.0, 7.0],
                "secured_flag": [False, True],
            },
            "capital_structure.tender_offer_debt": {
                "target_tranche_id": ["debt_tranche_24m"],
                "amount_usd": [debt_due_24m * x for x in (0.25, 0.5, 0.75, 1.0)],
                "funding_mix": _default_funding_mixes(),
            },
            "capital_structure.exchange_offer": {
                "target_instruments": ["debt_stack_core"],
                "new_tenor_years": [3.0, 5.0, 7.0],
            },
            "capital_structure.liability_management_exercise": {
                "structure_class": ["amend_extend", "distressed_exchange"],
                "targeted_instruments": ["debt_stack_core"],
                "coercive_level": ["low", "medium", "high"],
            },
        }

        ratio_signal = 0.0
        if ratio is not None:
            ratio_signal = _clip((ratio - self.thresholds["maturity_wall_ratio_24m"]) / 0.5, 0.0, 1.0)
        liquidity_signal = 0.0
        if available_for_actions is not None and debt_due_24m > 0:
            liquidity_cover = available_for_actions / debt_due_24m
            liquidity_signal = _clip(
                (self.thresholds["maturity_wall_liquidity_cover_24m_max"] - liquidity_cover)
                / self.thresholds["maturity_wall_liquidity_cover_24m_max"],
                0.0,
                1.0,
            )
        trigger_strength = max(ratio_signal, liquidity_signal)

        out: List[Dict[str, Any]] = []
        for aid, overrides in action_specs.items():
            schema = self.registry.get_action(aid)
            if schema is None:
                continue
            for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=48):
                out.append(
                    self._build_candidate_raw(
                        run_id=run.run_id,
                        action_schema=schema,
                        parameters=params,
                        generation_source="deterministic_rule",
                        rationale_refs=[
                            {
                                "reference_type": "state_feature",
                                "reference_id": "capital_structure.maturity_wall_ratio_24m",
                                "explanation": "Elevated maturity wall suggests refinancing and liability management variants.",
                            }
                        ],
                        trigger_strength=trigger_strength,
                        playbook_relevance=0.7,
                    )
                )
        return out

    def _gen_liquidity_excess(self, run: RecommendationRun, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        liquidity = _to_float(_feature_value(features, "liquidity.available_for_actions"), 0.0) or 0.0
        mcap = _to_float(_feature_value(features, "market.market_cap"), 0.0) or 0.0
        dividend_initiate_allowed = _dividend_initiation_nonpayer_signal(features)
        nonpayer_without_market_cap = bool(
            mcap <= 0
            and liquidity > 0
            and dividend_initiate_allowed
        )
        if mcap <= 0 and not nonpayer_without_market_cap:
            return []
        ratio = liquidity / mcap if mcap > 0 else 0.0
        net_cash_override = self._net_cash_maturity_override_profile(features)
        buyback_supported_override = self._buyback_supported_net_cash_override_profile(features)
        net_cash_fcf_yield = _to_float(_feature_value(features, "market.fcf_yield"), None)
        if mcap > 0 and ratio <= self.thresholds["liquidity_vs_mcap"]:
            if not (
                net_cash_override
                and ratio >= self.thresholds["capital_return_net_cash_liquidity_vs_mcap_min"]
                and net_cash_fcf_yield is not None
                and net_cash_fcf_yield >= self.thresholds["capital_return_net_cash_fcf_yield_min"]
            ) and not buyback_supported_override:
                return []
        if buyback_supported_override and mcap > 0 and ratio < self.thresholds["capital_return_net_cash_liquidity_vs_mcap_min"]:
            return []
        if self._market_shutdown_regular_payer_recap_profile(features):
            return []
        if self._capital_return_blocked_by_financing_stress(features):
            return []

        prefer_dividend = self._prefer_dividend_over_nonrecurring_capital_return(features, ratio)
        buyback_sizes = [0.02, 0.05, 0.10, 0.15]
        funding = _default_funding_mixes()[:2]
        action_specs = {
            "capital_return.open_market_buyback": {
                "size_pct_market_cap": buyback_sizes,
                "funding_mix": funding,
            },
            "capital_return.accelerated_share_repurchase": {
                "size_pct_market_cap": [0.05, 0.10, 0.15],
                "funding_mix": funding,
            },
            "capital_return.special_dividend": {
                "size_absolute_usd": [mcap * x for x in (0.02, 0.05, 0.1)],
                "funding_mix": funding,
            },
        }
        if not self._expectations_allow_capital_return(features):
            action_specs.pop("capital_return.open_market_buyback", None)
            action_specs.pop("capital_return.accelerated_share_repurchase", None)
            action_specs.pop("capital_return.special_dividend", None)
        if prefer_dividend:
            action_specs.pop("capital_return.open_market_buyback", None)
            action_specs.pop("capital_return.accelerated_share_repurchase", None)
            action_specs.pop("capital_return.special_dividend", None)
        if dividend_initiate_allowed:
            action_specs["capital_return.dividend_initiate"] = {
                "initial_yield_pct": [0.01, 0.02, 0.03],
            }
            if mcap <= 0:
                action_specs.pop("capital_return.open_market_buyback", None)
                action_specs.pop("capital_return.accelerated_share_repurchase", None)
                action_specs.pop("capital_return.special_dividend", None)

        out: List[Dict[str, Any]] = []
        for aid, overrides in action_specs.items():
            schema = self.registry.get_action(aid)
            if schema is None:
                continue
            for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=24):
                out.append(
                    self._build_candidate_raw(
                        run_id=run.run_id,
                        action_schema=schema,
                        parameters=params,
                        generation_source="deterministic_rule",
                        rationale_refs=[
                            {
                                "reference_type": "state_feature",
                                "reference_id": "liquidity.available_for_actions",
                                "explanation": "Excess available liquidity supports capital return variants.",
                            }
                        ],
                        trigger_strength=(
                            0.55
                            if mcap <= 0 and aid == "capital_return.dividend_initiate"
                            else _clip((ratio - self.thresholds["liquidity_vs_mcap"]) / 0.2, 0.0, 1.0)
                        ),
                        playbook_relevance=0.65,
                    )
                )
        return out

    def _gen_undervaluation(self, run: RecommendationRun, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        ev_z = _to_float(_feature_value(features, "market.ev_ebitda_vs_peer_z"), 0.0) or 0.0
        fcf_pct = _to_float(_feature_value(features, "market.fcf_yield_percentile_peers"), 0.0) or 0.0
        if not (ev_z < -1.0 and fcf_pct > 0.7):
            return []
        if not self._expectations_allow_capital_return(features):
            return []

        action_specs: Dict[str, Dict[str, Any]] = {}
        if not self._mna_blocked_by_financing_or_capacity(features):
            action_specs["mna.go_private_lbo"] = {
                "target_size_pct_ev": [0.75, 1.0, 1.25],
                "funding_mix": [
                    {"cash": 0.05, "debt": 0.75, "equity": 0.20},
                    {"cash": 0.10, "debt": 0.70, "equity": 0.20},
                ],
                "leverage_post_close": [4.5, 5.5, 6.5],
            }
        if not self._capital_return_blocked_by_financing_stress(features):
            action_specs.update(
                {
                    "capital_return.open_market_buyback": {
                        "size_pct_market_cap": [0.02, 0.05, 0.1],
                        "funding_mix": _default_funding_mixes()[:2],
                    },
                    "capital_return.accelerated_share_repurchase": {
                        "size_pct_market_cap": [0.05, 0.1],
                        "funding_mix": _default_funding_mixes()[:2],
                    },
                    "capital_return.dividend_increase": {"percent_change": [0.05, 0.1, 0.2]},
                }
            )

        trigger = _clip((abs(ev_z) - 1.0) / 2.0 + (fcf_pct - 0.7), 0.0, 1.0)
        if self._expectations_support_capital_return(features):
            trigger = _clip(trigger + 0.05, 0.0, 1.0)
        out: List[Dict[str, Any]] = []
        for aid, overrides in action_specs.items():
            schema = self.registry.get_action(aid)
            if schema is None:
                continue
            for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=24):
                out.append(
                    self._build_candidate_raw(
                        run_id=run.run_id,
                        action_schema=schema,
                        parameters=params,
                        generation_source="deterministic_rule",
                        rationale_refs=[
                            {
                                "reference_type": "state_feature",
                                "reference_id": "market.ev_ebitda_vs_peer_z",
                                "explanation": "Deep relative valuation discount supports value-return actions.",
                            },
                            {
                                "reference_type": "state_feature",
                                "reference_id": "market.fcf_yield_percentile_peers",
                                "explanation": "High FCF yield percentile strengthens undervaluation signal.",
                            },
                        ],
                        trigger_strength=trigger,
                        playbook_relevance=0.7,
                    )
                )
        return out

    def _gen_conglomerate_discount(
        self,
        run: RecommendationRun,
        features: Dict[str, Any],
        known_segments: List[str],
    ) -> List[Dict[str, Any]]:
        segment_count = _to_float(_feature_value(features, "strategic.segment_count"), 0.0) or 0.0
        discount = _to_float(_feature_value(features, "market.conglomerate_discount_signal"), 0.0) or 0.0
        margin_diverge = _to_float(_feature_value(features, "operating.segment_margin_divergence"), 0.0) or 0.0
        if not (segment_count >= 2 and discount > 0 and margin_diverge > 0):
            return []

        segments = known_segments or ["segment_A", "segment_B"]
        action_specs = {
            "portfolio.divestiture_partial": {"segment_reference": segments[:2], "percent_divested": [0.2, 0.4, 0.6]},
            "portfolio.divestiture_full": {"segment_reference": segments[:2], "percent_divested": [1.0]},
            "portfolio.spin_off": {"segment_reference": segments[:2]},
            "portfolio.carve_out_ipo": {"segment_reference": segments[:2], "percent_float": [0.1, 0.2, 0.3]},
        }

        trigger = _clip(0.5 + min(discount, 1.0) * 0.3 + min(margin_diverge, 1.0) * 0.2, 0.0, 1.0)
        out: List[Dict[str, Any]] = []
        for aid, overrides in action_specs.items():
            schema = self.registry.get_action(aid)
            if schema is None:
                continue
            for params in self.generate_parameter_variants(
                schema,
                features,
                overrides=overrides,
                known_segments=segments,
                max_variants=48,
            ):
                out.append(
                    self._build_candidate_raw(
                        run_id=run.run_id,
                        action_schema=schema,
                        parameters=params,
                        generation_source="deterministic_rule",
                        rationale_refs=[
                            {
                                "reference_type": "state_feature",
                                "reference_id": "market.conglomerate_discount_signal",
                                "explanation": "Conglomerate discount signal supports portfolio separation actions.",
                            },
                            {
                                "reference_type": "state_feature",
                                "reference_id": "operating.segment_margin_divergence",
                                "explanation": "Divergent segment economics increase case for simplification.",
                            },
                        ],
                        trigger_strength=trigger,
                        playbook_relevance=0.8,
                    )
                )
        return out

    def _gen_subscale_position(self, run: RecommendationRun, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        market_share_pct = _to_float(_feature_value(features, "peer_context.relative_positioning.market_share_percentile"), 1.0) or 1.0
        consolidation = _to_float(_feature_value(features, "peer_context.consolidation_wave_score"), 0.0) or 0.0
        if not (
            market_share_pct < self.thresholds["market_share_percentile"]
            and consolidation >= self.thresholds["consolidation_wave"]
        ):
            return []
        if self._mna_blocked_by_financing_or_capacity(features):
            return []

        action_specs = {
            "mna.tuck_in_acquisition": {
                "target_size_pct_ev": [0.05, 0.1, 0.2],
                "funding_mix": _default_funding_mixes(),
            },
            "mna.platform_acquisition": {
                "target_size_pct_ev": [0.1, 0.2, 0.35],
                "funding_mix": _default_funding_mixes(),
            },
        }

        trigger = _clip((self.thresholds["market_share_percentile"] - market_share_pct) + consolidation * 0.5, 0.0, 1.0)
        out: List[Dict[str, Any]] = []
        for aid, overrides in action_specs.items():
            schema = self.registry.get_action(aid)
            if schema is None:
                continue
            for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=48):
                out.append(
                    self._build_candidate_raw(
                        run_id=run.run_id,
                        action_schema=schema,
                        parameters=params,
                        generation_source="deterministic_rule",
                        rationale_refs=[
                            {
                                "reference_type": "state_feature",
                                "reference_id": "peer_context.relative_positioning.market_share_percentile",
                                "explanation": "Subscale peer positioning supports inorganic scale actions.",
                            },
                            {
                                "reference_type": "peer_event",
                                "reference_id": "peer_context.consolidation_wave_score",
                                "explanation": "Active consolidation wave increases urgency for strategic M&A response.",
                            },
                        ],
                        trigger_strength=trigger,
                        playbook_relevance=0.75,
                    )
                )
        return out

    def _gen_margin_underperformance(self, run: RecommendationRun, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        margin_pct = _to_float(_feature_value(features, "operating.ebitda_margin_percentile_peers"), 1.0) or 1.0
        if margin_pct >= self.thresholds["margin_percentile_peers"]:
            return []

        mcap = _to_float(_feature_value(features, "market.market_cap"), 0.0) or 0.0
        base_savings = max(1_000_000.0, mcap * 0.01) if mcap > 0 else 10_000_000.0
        action_specs = {
            "restructuring.cost_program": {"annualized_savings_target_usd": [base_savings, base_savings * 1.5]},
            "restructuring.workforce_reduction": {"employee_pct_reduction": [0.03, 0.05, 0.1]},
            "restructuring.footprint_optimization": {"site_count_affected": [1, 3, 5]},
        }

        trigger = _clip((self.thresholds["margin_percentile_peers"] - margin_pct) / self.thresholds["margin_percentile_peers"], 0.0, 1.0)
        out: List[Dict[str, Any]] = []
        for aid, overrides in action_specs.items():
            schema = self.registry.get_action(aid)
            if schema is None:
                continue
            for params in self.generate_parameter_variants(schema, features, overrides=overrides, max_variants=24):
                out.append(
                    self._build_candidate_raw(
                        run_id=run.run_id,
                        action_schema=schema,
                        parameters=params,
                        generation_source="deterministic_rule",
                        rationale_refs=[
                            {
                                "reference_type": "state_feature",
                                "reference_id": "operating.ebitda_margin_percentile_peers",
                                "explanation": "Peer margin underperformance supports restructuring options.",
                            }
                        ],
                        trigger_strength=trigger,
                        playbook_relevance=0.7,
                    )
                )
        return out

    def generate_parameter_variants(
        self,
        action_schema: Dict[str, Any],
        state_snapshot_features: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
        known_segments: Optional[List[str]] = None,
        max_variants: int = 64,
        include_optional: bool = False,
    ) -> List[Dict[str, Any]]:
        overrides = overrides or {}
        params = action_schema.get("parameter_schema", {})
        known_segments = known_segments or []
        mcap = _to_float(_feature_value(state_snapshot_features, "market.market_cap"), 0.0) or 0.0
        liquidity = _to_float(_feature_value(state_snapshot_features, "liquidity.available_for_actions"), 0.0) or 0.0
        max_anchor = max(mcap * 0.1, liquidity if liquidity > 0 else 0.0, 1_000_000.0)

        value_grid: Dict[str, Any] = {}
        for pname, pdef in params.items():
            required = bool(pdef.get("required", False))
            ptype = pdef.get("type")

            if pname in overrides:
                value_grid[pname] = overrides[pname]
                continue

            if not required and not include_optional:
                continue

            if ptype == "percent":
                lo = _to_float(pdef.get("min"), 0.0) or 0.0
                hi = _to_float(pdef.get("max"), 1.0) or 1.0
                span = max(hi - lo, 0.0)
                pts = [lo, lo + 0.25 * span, lo + 0.5 * span, lo + 0.75 * span, hi]
                dedup = sorted(set(_round_num(_clip(x, lo, hi)) for x in pts))
                value_grid[pname] = dedup[1:4] if len(dedup) >= 4 else dedup
            elif ptype == "numeric":
                value_grid[pname] = _numeric_value_grid(str(pname), dict(pdef or {}), max_anchor=max_anchor)
            elif ptype == "boolean":
                value_grid[pname] = [False, True]
            elif ptype == "enum":
                vals = list(pdef.get("values", []))
                value_grid[pname] = vals[:3] if vals else []
            elif ptype == "funding_mix_object":
                value_grid[pname] = _default_funding_mixes()
            elif ptype == "date_window":
                value_grid[pname] = [{"start": None, "end": None}]
            elif ptype == "range":
                value_grid[pname] = [{"min": 2.0, "max": 3.0}]
            elif ptype == "segment_reference":
                segs = known_segments or ["segment_A"]
                value_grid[pname] = segs[:3]
            elif ptype == "entity_reference":
                value_grid[pname] = ["reference_1"]
            else:
                value_grid[pname] = [None]

        variants = _product_params(value_grid, max_variants=max_variants)
        if not variants:
            variants = [{}]
        return variants

    def _build_candidate_raw(
        self,
        run_id: str,
        action_schema: Dict[str, Any],
        parameters: Dict[str, Any],
        generation_source: str,
        rationale_refs: List[Dict[str, Any]],
        trigger_strength: float,
        playbook_relevance: float,
    ) -> Dict[str, Any]:
        action_id = str(action_schema.get("action_id", ""))
        signature = _candidate_signature(action_id, parameters)
        preconditions = self._build_preconditions(action_schema)
        warnings: List[str] = []
        confidence = self._confidence_score(
            generation_source=generation_source,
            trigger_strength=trigger_strength,
            rationale_count=len(rationale_refs),
            validation_warning_count=len(warnings),
            playbook_relevance=playbook_relevance,
        )
        return {
            "run_id": run_id,
            "action_type": str(action_schema.get("action_type", "")),
            "action_subtype": str(action_schema.get("action_subtype", "")),
            "action_id": action_id,
            "parameters": _json_safe(parameters),
            "params": _json_safe(parameters),
            "assumed_preconditions": preconditions,
            "generation_source": generation_source,
            "rationale_refs": rationale_refs,
            "generation_confidence": confidence,
            "created_at": _now_iso(),
            "candidate_signature": signature,
            "_validation_warnings": warnings,
        }

    def _build_preconditions(self, action_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        prereq = action_schema.get("feasibility_prerequisites", {})
        out: List[Dict[str, Any]] = []
        for cond in prereq.get("state_conditions", []) if isinstance(prereq, dict) else []:
            feature = str(cond.get("feature", ""))
            op = str(cond.get("operator", "=="))
            rel = _RELATION_MAP.get(op, "equal")
            val = cond.get("value")
            out.append(
                {
                    "feature_name": feature,
                    "assumed_relation": rel,
                    "value": val,
                    "explanation": f"Assumes {feature} {op} {val}",
                }
            )
        return out

    def _confidence_score(
        self,
        generation_source: str,
        trigger_strength: float,
        rationale_count: int,
        validation_warning_count: int,
        playbook_relevance: float,
    ) -> float:
        trigger = _clip(trigger_strength, 0.0, 1.0)
        evidence = _clip(0.4 + 0.15 * min(rationale_count, 4), 0.0, 1.0)
        param_realism = _clip(1.0 - 0.15 * validation_warning_count, 0.3, 1.0)
        source_boost = {
            "deterministic_rule": 0.85,
            "playbook_template": 0.80,
            "llm_proposal": 0.60,
        }.get(generation_source, 0.60)
        score = (
            0.35 * trigger
            + 0.25 * evidence
            + 0.20 * param_realism
            + 0.10 * _clip(playbook_relevance, 0.0, 1.0)
            + 0.10 * source_boost
        )
        return _round_num(_clip(score, 0.0, 1.0))

    def _to_draft(self, run_id: str, candidate: Dict[str, Any]) -> ActionCandidateDraft:
        signature = str(candidate.get("candidate_signature"))
        try:
            namespace = uuid.UUID(str(run_id))
        except Exception:
            namespace = uuid.NAMESPACE_URL
        cid = str(uuid.uuid5(namespace, signature))

        pre = [
            Precondition(
                feature_name=str(x.get("feature_name", "")),
                assumed_relation=str(x.get("assumed_relation", "equal")),
                value=x.get("value"),
                explanation=str(x.get("explanation", "")),
            )
            for x in candidate.get("assumed_preconditions", [])
        ]
        refs = [
            RationaleReference(
                reference_type=str(x.get("reference_type", "")),
                reference_id=str(x.get("reference_id", "")),
                explanation=str(x.get("explanation", "")),
            )
            for x in candidate.get("rationale_refs", [])
        ]

        return ActionCandidateDraft(
            candidate_id=cid,
            run_id=run_id,
            action_type=str(candidate.get("action_type", "")),
            action_subtype=str(candidate.get("action_subtype", "")),
            action_id=str(candidate.get("action_id", "")),
            parameters=dict(candidate.get("parameters", {}) or {}),
            params=dict(candidate.get("params", {}) or {}),
            assumed_preconditions=pre,
            generation_source=str(candidate.get("generation_source", "deterministic_rule")),
            rationale_refs=refs,
            generation_confidence=float(candidate.get("generation_confidence", 0.0) or 0.0),
            created_at=str(candidate.get("created_at", _now_iso())),
            candidate_signature=signature,
        )

    def _dedupe_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            sig = str(c.get("candidate_signature", ""))
            if not sig:
                sig = _candidate_signature(str(c.get("action_id", "")), dict(c.get("parameters", {}) or {}))
                c["candidate_signature"] = sig
            prev = seen.get(sig)
            if prev is None:
                seen[sig] = c
                continue
            # Keep higher confidence candidate if duplicate.
            if float(c.get("generation_confidence", 0.0)) > float(prev.get("generation_confidence", 0.0)):
                seen[sig] = c
        return list(seen.values())

    def _validate_llm_proposals(
        self,
        run: RecommendationRun,
        llm_proposals: Optional[List[Dict[str, Any]]],
        llm_metadata: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not llm_proposals:
            return [], {}
        llm_metadata = dict(llm_metadata or {})
        accepted: List[Dict[str, Any]] = []
        discarded: List[Dict[str, Any]] = []
        for i, row in enumerate(llm_proposals):
            if not isinstance(row, dict):
                discarded.append({"index": i, "reason": "proposal_not_object"})
                continue
            aid = str(row.get("action_id", ""))
            schema = self.registry.get_action(aid)
            if schema is None:
                discarded.append({"index": i, "reason": f"unknown_action_id:{aid}"})
                continue
            params = row.get("parameters", {})
            if not isinstance(params, dict):
                discarded.append({"index": i, "reason": "parameters_not_object"})
                continue
            refs = row.get("evidence_refs", [])
            if not isinstance(refs, list):
                discarded.append({"index": i, "reason": "evidence_refs_not_list"})
                continue

            rationale_refs: List[Dict[str, Any]] = []
            invalid_ref = False
            for r in refs:
                if not isinstance(r, dict):
                    invalid_ref = True
                    break
                rt = str(r.get("reference_type", ""))
                if rt not in {"state_feature", "extracted_fact", "precedent_signal", "peer_event", "regime_signal"}:
                    invalid_ref = True
                    break
                rationale_refs.append(
                    {
                        "reference_type": rt,
                        "reference_id": str(r.get("reference_id", "")),
                        "explanation": str(r.get("explanation", "")),
                    }
                )
            if invalid_ref:
                discarded.append({"index": i, "reason": "invalid_evidence_refs"})
                continue

            accepted.append(
                self._build_candidate_raw(
                    run_id=run.run_id,
                    action_schema=schema,
                    parameters=params,
                    generation_source="llm_proposal",
                    rationale_refs=rationale_refs
                    or [
                        {
                            "reference_type": "regime_signal",
                            "reference_id": "llm_proposal",
                            "explanation": str(row.get("rationale", "LLM-proposed candidate")),
                        }
                    ],
                    trigger_strength=0.55,
                    playbook_relevance=0.5,
                )
            )

        trace = {
            "prompt": llm_metadata.get("prompt"),
            "response": llm_metadata.get("response"),
            "temperature": llm_metadata.get("temperature"),
            "seed": llm_metadata.get("seed"),
            "proposal_count": len(llm_proposals),
            "accepted_count": len(accepted),
            "discarded": discarded,
        }
        return accepted, trace

    def _infer_evidence_classes(
        self,
        snapshot: Dict[str, Any],
        extracted_facts: Optional[List[Dict[str, Any]]],
        event_store: Optional[List[Dict[str, Any]]],
    ) -> List[str]:
        classes = {"financial_disclosure"}
        prov = snapshot.get("provenance", {}) if isinstance(snapshot, dict) else {}
        inputs = prov.get("inputs_used", {}) if isinstance(prov, dict) else {}
        if inputs.get("facts") or extracted_facts:
            classes.update({"management_statement", "capital_policy_statement", "liquidity_disclosure"})
        if inputs.get("timeseries") or inputs.get("macro"):
            classes.add("market_signal")
        if inputs.get("events") or event_store:
            classes.update({"recent_action_history", "peer_context_signal"})
        if inputs.get("issuer_ratings"):
            classes.add("rating_disclosure")
        if self._extract_segments(snapshot, extracted_facts):
            classes.add("segment_disclosure")
        return sorted(classes)

    def _extract_segments(
        self,
        snapshot: Dict[str, Any],
        extracted_facts: Optional[List[Dict[str, Any]]],
    ) -> List[str]:
        out: List[str] = []
        feats = feature_view_from_snapshot(snapshot, view_name="candidate_generation")
        segs = _feature_value(feats, "strategic.segment_references")
        if isinstance(segs, list):
            out.extend(str(x) for x in segs if x is not None and str(x))
        elif isinstance(segs, dict):
            out.extend(str(k) for k in segs.keys())
        if extracted_facts:
            for f in extracted_facts:
                if not isinstance(f, dict):
                    continue
                for key in ("segment_reference", "segment", "business_segment"):
                    if key in f and f.get(key) is not None:
                        out.append(str(f.get(key)))
        if not out:
            count = int(_to_float(_feature_value(feats, "strategic.segment_count"), 0.0) or 0.0)
            if count > 0:
                out = [f"segment_{i+1}" for i in range(min(count, 6))]
        return list(dict.fromkeys(out))

    def _constraint_tokens(self, run: RecommendationRun, snapshot: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for c in run.constraints.hard_constraints + run.constraints.soft_constraints:
            out.append(str(c.constraint_type))
            out.append(str(c.constraint_id))
        cs = snapshot.get("constraint_set") if isinstance(snapshot, dict) else None
        if isinstance(cs, dict):
            for bucket in ("hard", "soft"):
                vals = cs.get(bucket, []) or []
                for item in vals:
                    if isinstance(item, dict):
                        if item.get("name"):
                            out.append(str(item.get("name")))
                        if item.get("constraint_id"):
                            out.append(str(item.get("constraint_id")))
                    elif item is not None:
                        out.append(str(item))
        return list(dict.fromkeys(out))


def generate_action_candidates(
    run: RecommendationRun,
    state_snapshot: Dict[str, Any],
    action_registry: ActionSchemaRegistry,
    action_ids: Optional[Sequence[str]] = None,
    action_type: Optional[str] = None,
    max_candidates: int = 1500,
    min_candidates_target: int = 0,
    strict_evidence: bool = False,
    extracted_facts: Optional[List[Dict[str, Any]]] = None,
    event_store: Optional[List[Dict[str, Any]]] = None,
    peer_set: Optional[Dict[str, Any]] = None,
    llm_proposals: Optional[List[Dict[str, Any]]] = None,
    llm_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    engine = CandidateGenerationEngine(action_registry)
    return engine.generate_candidate_set(
        run=run,
        state_snapshot=state_snapshot,
        action_ids=action_ids,
        action_type=action_type,
        max_candidates=max_candidates,
        min_candidates_target=min_candidates_target,
        strict_evidence=strict_evidence,
        extracted_facts=extracted_facts,
        event_store=event_store,
        peer_set=peer_set,
        llm_proposals=llm_proposals,
        llm_metadata=llm_metadata,
    )


__all__ = [
    "ActionCandidateDraft",
    "CandidateGenerationEngine",
    "PlaybookRegistry",
    "PlaybookTemplate",
    "Precondition",
    "RationaleReference",
    "generate_action_candidates",
]
