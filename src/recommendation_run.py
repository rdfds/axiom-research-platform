"""
Recommendation Run orchestration.

A RecommendationRun freezes a reproducible decision problem:
- as-of company state snapshot (with hash)
- model versions
- objective vector / constraints / scenario assumptions
- data cutoff
- lifecycle + audit trail
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
import hashlib
import json
import os
import shutil
import uuid

import pandas as pd

from .causal_impact_model import DEFAULT_CAUSAL_IMPACT_MODEL_ARTIFACT
from .data_paths import resolve_data_path
from .recommendation_runtime_config import (
    DEFAULT_PRECEDENT_RETRIEVAL_VERSION,
    build_create_config,
    capture_runtime_env_config,
    merge_metadata_patch,
)


RUN_STATUSES = {
    "initialized",
    "candidate_generation",
    "feasibility_evaluation",
    "precedent_retrieval",
    "plan_search",
    "completed",
    "failed",
}

AUDIT_EVENT_TYPES = {
    "run_created",
    "snapshot_frozen",
    "candidate_generation_started",
    "candidate_generation_completed",
    "feasibility_eval_started",
    "feasibility_eval_completed",
    "precedent_retrieval_started",
    "precedent_retrieval_completed",
    "planning_started",
    "planning_completed",
    "run_completed",
    "run_failed",
}

CONSTRAINT_TYPES = {
    "leverage_limit",
    "no_equity_issuance",
    "maintain_investment_grade",
    "minimum_cash_reserve",
    "max_acquisition_size",
    "forbidden_action_type",
    "required_action_type",
}

CONSTRAINT_SOURCES = {
    "user_input",
    "extracted_fact",
    "private_overlay",
}

CONSTRAINT_PRIORITIES = {"hard", "soft"}

REGIME_OVERRIDE_FIELDS = {
    "credit_regime_override": {"tight", "neutral", "loose", "none"},
    "risk_regime_override": {"risk_on", "neutral", "risk_off", "none"},
    "vol_regime_override": {"high", "normal", "low", "none"},
    "sector_cycle_override": {"upcycle", "neutral", "downcycle", "none"},
}

PHASE_STARTED_EVENT = {
    "candidate_generation": "candidate_generation_started",
    "feasibility_evaluation": "feasibility_eval_started",
    "precedent_retrieval": "precedent_retrieval_started",
    "plan_search": "planning_started",
}

PHASE_COMPLETED_EVENT = {
    "candidate_generation": "candidate_generation_completed",
    "feasibility_evaluation": "feasibility_eval_completed",
    "precedent_retrieval": "precedent_retrieval_completed",
    "plan_search": "planning_completed",
}

DEFAULT_OBJECTIVES = {
    "value_creation_weight": 0.35,
    "risk_reduction_weight": 0.25,
    "growth_weight": 0.15,
    "rating_preservation_weight": 0.15,
    "optionality_weight": 0.10,
}


@dataclass
class ObjectiveVector:
    value_creation_weight: float
    risk_reduction_weight: float
    growth_weight: float
    rating_preservation_weight: float
    optionality_weight: float

    @classmethod
    def default(cls) -> "ObjectiveVector":
        return cls(**DEFAULT_OBJECTIVES)

    @classmethod
    def from_any(cls, payload: Optional[Dict[str, Any]]) -> "ObjectiveVector":
        if payload is None:
            return cls.default()
        base = dict(DEFAULT_OBJECTIVES)
        base.update(payload)
        out = cls(
            value_creation_weight=float(base["value_creation_weight"]),
            risk_reduction_weight=float(base["risk_reduction_weight"]),
            growth_weight=float(base["growth_weight"]),
            rating_preservation_weight=float(base["rating_preservation_weight"]),
            optionality_weight=float(base["optionality_weight"]),
        )
        out.normalize_in_place()
        out.validate()
        return out

    def normalize_in_place(self) -> None:
        vals = [
            self.value_creation_weight,
            self.risk_reduction_weight,
            self.growth_weight,
            self.rating_preservation_weight,
            self.optionality_weight,
        ]
        total = sum(vals)
        if total <= 0:
            raise ValueError("ObjectiveVector weights must sum to > 0 before normalization")
        self.value_creation_weight = self.value_creation_weight / total
        self.risk_reduction_weight = self.risk_reduction_weight / total
        self.growth_weight = self.growth_weight / total
        self.rating_preservation_weight = self.rating_preservation_weight / total
        self.optionality_weight = self.optionality_weight / total

    def validate(self) -> None:
        vals = [
            self.value_creation_weight,
            self.risk_reduction_weight,
            self.growth_weight,
            self.rating_preservation_weight,
            self.optionality_weight,
        ]
        for v in vals:
            if v < 0 or v > 1:
                raise ValueError(f"Objective weight out of [0,1] range: {v}")
        if abs(sum(vals) - 1.0) > 1e-9:
            raise ValueError("ObjectiveVector must sum to 1 after normalization")


@dataclass
class Constraint:
    constraint_type: str
    parameters: Dict[str, Any]
    source: str
    priority: str
    constraint_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_any(cls, payload: Dict[str, Any], priority_fallback: Optional[str] = None) -> "Constraint":
        out = cls(
            constraint_type=str(payload.get("constraint_type", "")).strip(),
            parameters=dict(payload.get("parameters", {}) or {}),
            source=str(payload.get("source", "user_input") or "user_input"),
            priority=str(payload.get("priority", priority_fallback or "soft") or "soft"),
            constraint_id=str(payload.get("constraint_id", str(uuid.uuid4()))),
        )
        out.validate()
        return out

    def validate(self) -> None:
        if self.constraint_type not in CONSTRAINT_TYPES:
            raise ValueError(f"Unsupported constraint_type: {self.constraint_type}")
        if self.source not in CONSTRAINT_SOURCES:
            raise ValueError(f"Unsupported constraint source: {self.source}")
        if self.priority not in CONSTRAINT_PRIORITIES:
            raise ValueError(f"Unsupported constraint priority: {self.priority}")
        if not isinstance(self.parameters, dict):
            raise ValueError("Constraint parameters must be an object")

        ctype = self.constraint_type
        if ctype in {"forbidden_action_type", "required_action_type"}:
            if not (self.parameters.get("action_type") or self.parameters.get("action_id")):
                raise ValueError(f"{ctype} requires action_type or action_id parameter")
        if ctype == "leverage_limit":
            if self.parameters.get("max_leverage") is None:
                raise ValueError("leverage_limit requires parameters.max_leverage")
        if ctype == "minimum_cash_reserve":
            if self.parameters.get("min_cash_reserve") is None and self.parameters.get("min_cash_reserve_usd") is None:
                raise ValueError("minimum_cash_reserve requires min_cash_reserve or min_cash_reserve_usd")
        if ctype == "max_acquisition_size":
            if self.parameters.get("max_size_pct_ev") is None and self.parameters.get("max_size_usd") is None:
                raise ValueError("max_acquisition_size requires max_size_pct_ev or max_size_usd")


@dataclass
class ConstraintSet:
    hard_constraints: List[Constraint] = field(default_factory=list)
    soft_constraints: List[Constraint] = field(default_factory=list)

    @classmethod
    def from_any(cls, payload: Optional[Dict[str, Any]]) -> "ConstraintSet":
        if payload is None:
            return cls()

        hard_raw = list(payload.get("hard_constraints", []) or [])
        soft_raw = list(payload.get("soft_constraints", []) or [])

        hard = [Constraint.from_any(x, priority_fallback="hard") for x in hard_raw]
        soft = [Constraint.from_any(x, priority_fallback="soft") for x in soft_raw]

        for c in hard:
            if c.priority != "hard":
                raise ValueError(f"Hard constraint has non-hard priority: {c.constraint_id}")
        for c in soft:
            if c.priority != "soft":
                raise ValueError(f"Soft constraint has non-soft priority: {c.constraint_id}")

        return cls(hard_constraints=hard, soft_constraints=soft)


@dataclass
class ScenarioAssumptions:
    credit_regime_override: str = "none"
    risk_regime_override: str = "none"
    vol_regime_override: str = "none"
    sector_cycle_override: str = "none"
    interest_rate_shift_bp: int = 0
    equity_market_drawdown_pct: float = 0.0
    custom_flags: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, payload: Optional[Dict[str, Any]]) -> "ScenarioAssumptions":
        if payload is None:
            out = cls()
            out.validate()
            return out
        out = cls(
            credit_regime_override=str(payload.get("credit_regime_override", "none") or "none"),
            risk_regime_override=str(payload['risk_regime_override'] or "none"),
            vol_regime_override=str(payload.get("vol_regime_override", "none") or "none"),
            sector_cycle_override=str(payload.get("sector_cycle_override", "none") or "none"),
            interest_rate_shift_bp=int(payload.get("interest_rate_shift_bp", 0) or 0),
            equity_market_drawdown_pct=float(payload.get("equity_market_drawdown_pct", 0.0) or 0.0),
            custom_flags=dict(payload.get("custom_flags", {}) or {}),
        )
        out.validate()
        return out

    def validate(self) -> None:
        for field_name, allowed in REGIME_OVERRIDE_FIELDS.items():
            if getattr(self, field_name) not in allowed:
                raise ValueError(f"Invalid scenario {field_name}: {getattr(self, field_name)}")
        if not isinstance(self.interest_rate_shift_bp, int):
            raise ValueError("interest_rate_shift_bp must be integer")
        if not isinstance(self.equity_market_drawdown_pct, float):
            raise ValueError("equity_market_drawdown_pct must be float")
        if not isinstance(self.custom_flags, dict):
            raise ValueError("custom_flags must be object")


