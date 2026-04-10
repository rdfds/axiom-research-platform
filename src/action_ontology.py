"""
Action ontology registry for candidate generation, feasibility gating, and planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ALLOWED_PARAMETER_TYPES = {
    "numeric",
    "percent",
    "boolean",
    "enum",
    "date_window",
    "entity_reference",
    "segment_reference",
    "funding_mix_object",
    "range",
}

ALLOWED_CHANNEL_TYPES = {
    "value_creation",
    "risk_reduction",
    "optionality_preservation",
    "signaling",
    "capital_structure_optimization",
    "portfolio_optimization",
    "cost_efficiency",
    "growth_substitution",
}

ALLOWED_RULE_TYPES = {
    "requires_prior",
    "unlocks",
    "conflicts_with",
    "discouraged_with",
    "preferred_after",
}

ALLOWED_RULE_STRENGTH = {"hard", "soft"}

ALLOWED_EVIDENCE_CLASSES = {
    "financial_disclosure",
    "management_statement",
    "liquidity_disclosure",
    "segment_disclosure",
    "capital_policy_statement",
    "rating_disclosure",
    "market_signal",
    "peer_context_signal",
    "recent_action_history",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percent_field(required: bool = False, minimum: float = 0.0, maximum: float = 1.0) -> Dict[str, Any]:
    return {
        "type": "percent",
        "required": required,
        "min": minimum,
        "max": maximum,
    }


def _numeric_field(required: bool = False, unit: Optional[str] = None, minimum: Optional[float] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "type": "numeric",
        "required": required,
    }
    if unit is not None:
        out["unit"] = unit
    if minimum is not None:
        out["min"] = minimum
    return out


def _enum_field(values: List[str], required: bool = False) -> Dict[str, Any]:
    return {
        "type": "enum",
        "required": required,
        "values": values,
    }


def _funding_mix(required: bool = True) -> Dict[str, Any]:
    return {
        "type": "funding_mix_object",
        "required": required,
        "fields": {
            "cash": {"type": "percent"},
            "debt": {"type": "percent"},
            "equity": {"type": "percent"},
        },
    }


def _date_window(required: bool = False) -> Dict[str, Any]:
    return {
        "type": "date_window",
        "required": required,
    }


def _range_field(required: bool = False, unit: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "type": "range",
        "required": required,
    }
    if unit:
        out["unit"] = unit
    return out


def _channel(
    channel_id: str,
    channel_type: str,
    description: str,
    activation_signals: List[str],
    negative_signals: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "channel_id": channel_id,
        "channel_type": channel_type,
        "description": description,
        "activation_signals": activation_signals,
        "negative_signals": negative_signals or [],
    }


def _rule(
    rule_type: str,
    target_action_id: str,
    condition: Optional[str],
    strength: str,
    explanation: str,
) -> Dict[str, Any]:
    return {
        "rule_type": rule_type,
        "target_action_id": target_action_id,
        "condition": condition,
        "strength": strength,
        "explanation": explanation,
    }


def _lead_time(minimum_days: int, median_days: int, p90_days: int, conditional_adjustments: Optional[List[dict]] = None) -> Dict[str, Any]:
    return {
        "minimum_days": minimum_days,
        "median_days": median_days,
        "p90_days": p90_days,
        "conditional_adjustments": conditional_adjustments or [],
    }


def _complexity(
    base_complexity_score: int,
    drivers: List[str],
    organizational_burden: str,
    cross_functional_dependencies: List[str],
) -> Dict[str, Any]:
    return {
        "base_complexity_score": base_complexity_score,
        "drivers": drivers,
        "organizational_burden": organizational_burden,
        "cross_functional_dependencies": cross_functional_dependencies,
    }


def _prerequisites(
    state_conditions: List[dict],
    required_features: List[str],
    required_evidence: Optional[List[str]] = None,
    required_disclosures: Optional[List[str]] = None,
    forbidden_constraints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "state_conditions": state_conditions,
        "required_features": required_features,
        "required_evidence": required_evidence or [],
        "required_disclosures": required_disclosures or [],
        "forbidden_constraints": forbidden_constraints or [],
    }


def _evidence(
    minimum_classes_required: List[str],
    optional_supporting_classes: Optional[List[str]] = None,
    must_have_features: Optional[List[str]] = None,
    allow_heuristic_if_missing: bool = True,
) -> Dict[str, Any]:
    return {
        "minimum_classes_required": minimum_classes_required,
        "optional_supporting_classes": optional_supporting_classes or [],
        "must_have_features": must_have_features or [],
        "allow_heuristic_if_missing": allow_heuristic_if_missing,
    }


def _action(
    action_type: str,
    action_subtype: str,
    label: str,
    description: str,
    parameter_schema: Dict[str, Any],
    feasibility_prerequisites: Dict[str, Any],
    mechanism_channels: List[Dict[str, Any]],
    lead_time_prior: Dict[str, Any],
    execution_complexity_prior: Dict[str, Any],
    dependency_rules: List[Dict[str, Any]],
    minimum_evidence_requirements: Dict[str, Any],
    validation_rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "action_type": action_type,
        "action_subtype": action_subtype,
        "action_id": f"{action_type}.{action_subtype}",
        "label": label,
        "description": description,
        "parameter_schema": parameter_schema,
        "feasibility_prerequisites": feasibility_prerequisites,
        "mechanism_channels": mechanism_channels,
        "lead_time_prior": lead_time_prior,
        "execution_complexity_prior": execution_complexity_prior,
        "dependency_rules": dependency_rules,
        "minimum_evidence_requirements": minimum_evidence_requirements,
        "validation_rules": validation_rules,
    }


