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


