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
            risk_regime_override=str(payload.get("risk_regime_override", "none") or "none"),
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


@dataclass
class FrozenStateReference:
    snapshot_id: str
    snapshot_hash: str
    snapshot_version: str


@dataclass
class ModelVersionBundle:
    candidate_generator_version: str
    feasibility_model_version: str
    mechanism_model_version: str
    precedent_retrieval_version: str
    planner_model_version: str
    regime_model_version: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataCutoffSpec:
    published_at_lte: str
    ingested_at_lte: str


@dataclass
class AuditEvent:
    event_id: str
    timestamp: str
    event_type: str
    details: Dict[str, Any]


@dataclass
class RecommendationRun:
    run_id: str
    company_id: str
    created_at: str
    as_of_time: str
    objectives: ObjectiveVector
    constraints: ConstraintSet
    scenario: ScenarioAssumptions
    frozen_state: FrozenStateReference
    model_versions: ModelVersionBundle
    data_cutoff: DataCutoffSpec
    status: str
    audit_log: List[AuditEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    planner_random_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "company_id": self.company_id,
            "created_at": self.created_at,
            "as_of_time": self.as_of_time,
            "objectives": asdict(self.objectives),
            "constraints": {
                "hard_constraints": [asdict(c) for c in self.constraints.hard_constraints],
                "soft_constraints": [asdict(c) for c in self.constraints.soft_constraints],
            },
            "scenario": asdict(self.scenario),
            "frozen_state": asdict(self.frozen_state),
            "model_versions": asdict(self.model_versions),
            "data_cutoff": asdict(self.data_cutoff),
            "status": self.status,
            "audit_log": [asdict(e) for e in self.audit_log],
            "metadata": self.metadata,
            "planner_random_seed": self.planner_random_seed,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RecommendationRun":
        return cls(
            run_id=str(payload["run_id"]),
            company_id=str(payload["company_id"]),
            created_at=str(payload["created_at"]),
            as_of_time=str(payload["as_of_time"]),
            objectives=ObjectiveVector.from_any(payload.get("objectives", {})),
            constraints=ConstraintSet.from_any(payload.get("constraints", {})),
            scenario=ScenarioAssumptions.from_any(payload.get("scenario", {})),
            frozen_state=FrozenStateReference(**dict(payload.get("frozen_state", {}))),
            model_versions=ModelVersionBundle(**dict(payload.get("model_versions", {}))),
            data_cutoff=DataCutoffSpec(**dict(payload.get("data_cutoff", {}))),
            status=str(payload.get("status", "initialized")),
            audit_log=[AuditEvent(**ev) for ev in payload.get("audit_log", [])],
            metadata=dict(payload.get("metadata", {}) or {}),
            planner_random_seed=payload.get("planner_random_seed"),
        )


class ModelRegistry:
    """Simple immutable model-version provider."""

    def __init__(self, versions: Optional[ModelVersionBundle] = None):
        mechanism_version = os.environ.get("MECHANISM_MODEL_VERSION")
        if not mechanism_version:
            mechanism_version = self._infer_mechanism_version_from_artifact()
        self._versions = versions or ModelVersionBundle(
            candidate_generator_version=os.environ.get("CANDIDATE_GENERATOR_VERSION", "candidate_generator_v1"),
            feasibility_model_version=os.environ.get("FEASIBILITY_MODEL_VERSION", "feasibility_model_v1"),
            mechanism_model_version=str(mechanism_version or "mechanism_model_v2_causal"),
            precedent_retrieval_version=os.environ.get(
                "PRECEDENT_RETRIEVAL_VERSION",
                DEFAULT_PRECEDENT_RETRIEVAL_VERSION,
            ),
            planner_model_version=os.environ.get("PLANNER_MODEL_VERSION", "planner_model_v1"),
            regime_model_version=os.environ.get("REGIME_MODEL_VERSION", "regime_model_v1"),
        )

    def get_current_versions(self) -> ModelVersionBundle:
        # Return a copy to preserve immutability semantics.
        return ModelVersionBundle(**self._versions.to_dict())

    def _infer_mechanism_version_from_artifact(self) -> Optional[str]:
        path = Path(
            str(
                os.environ.get(
                    "CAUSAL_IMPACT_MODEL_PATH",
                    str(DEFAULT_CAUSAL_IMPACT_MODEL_ARTIFACT),
                )
            )
        )
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return None
        model_version = str(payload.get("version", "")).strip()
        trained_at = str(payload.get("trained_at", "")).strip()
        mode = str(os.environ.get("CAUSAL_IMPACT_MODE", "blend")).strip().lower() or "blend"
        if not model_version:
            return None
        suffix = trained_at[:10].replace("-", "") if trained_at else "undated"
        return f"mechanism_model_v2_causal+{model_version}+{suffix}+mode_{mode}"


class RecommendationRunStore:
    """Persistent registry for RecommendationRun objects."""

    def __init__(self, root: str | Path = "data/recommendation_runs", temp_dir: str | Path | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.root / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "run_index.parquet"
        if temp_dir is None:
            temp_dir = os.environ.get("RECOMMENDATION_RUN_TMP_DIR", "/tmp/recommendation_runs")
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"run_id={run_id}.json"

    def _stage_path(self, out: Path) -> Path:
        # Recreate temp_dir defensively in case an external cleanup removed it.
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # Use a unique staged path per write to avoid cross-run races where two
        # workers target the same "<name>.tmp" at once.
        return self.temp_dir / f"{out.name}.{uuid.uuid4().hex}.tmp"

    def _finalize(self, staged: Path, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(staged), str(out))
        except Exception:
            shutil.copy2(str(staged), str(out))
            try:
                staged.unlink()
            except Exception:
                pass

    def get_run(self, run_id: str) -> Optional[RecommendationRun]:
        p = self._run_path(run_id)
        if not p.exists():
            return None
        payload = json.loads(p.read_text())
        return RecommendationRun.from_dict(payload)

    def write_run(self, run: RecommendationRun) -> Path:
        out = self._run_path(run.run_id)
        if out.exists():
            raise ValueError(f"Run already exists: {run.run_id}")
        self._write_run_unchecked(run, sync_index=True)
        return out

    def update_run(self, run: RecommendationRun, sync_index: bool = True) -> Path:
        existing = self.get_run(run.run_id)
        if existing is None:
            raise ValueError(f"Run does not exist: {run.run_id}")
        self._enforce_immutability(existing, run)
        out = self._write_run_unchecked(run, sync_index=sync_index)
        return out

    def _write_run_unchecked(self, run: RecommendationRun, sync_index: bool = True) -> Path:
        out = self._run_path(run.run_id)
        staged = self._stage_path(out)
        staged.write_text(json.dumps(_json_sanitize(run.to_dict()), indent=2))
        self._finalize(staged, out)
        if sync_index:
            self._upsert_index(run)
        return out

    def _upsert_index(self, run: RecommendationRun) -> None:
        row = {
            "run_id": run.run_id,
            "company_id": run.company_id,
            "as_of_time": run.as_of_time,
            "status": run.status,
            "created_at": run.created_at,
        }
        if self.index_path.exists():
            df = pd.read_parquet(self.index_path)
            df = df[df["run_id"] != run.run_id]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        staged = self._stage_path(self.index_path)
        staged.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(staged, index=False)
        self._finalize(staged, self.index_path)

    def list_runs(
        self,
        company_id: Optional[str] = None,
        as_of_time: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[RecommendationRun]:
        if not self.index_path.exists():
            return []
        idx = pd.read_parquet(self.index_path)
        if company_id is not None:
            idx = idx[idx["company_id"].astype(str) == str(company_id)]
        if as_of_time is not None:
            as_of_norm = _parse_ts(as_of_time).isoformat()
            idx = idx[idx["as_of_time"].astype(str) == as_of_norm]
        if status is not None:
            idx = idx[idx["status"].astype(str) == str(status)]

        out: List[RecommendationRun] = []
        for rid in idx["run_id"].tolist():
            run = self.get_run(str(rid))
            if run is not None:
                out.append(run)
        out.sort(key=lambda r: r.created_at)
        return out

    def log_event(self, run_id: str, event_type: str, details: Optional[Dict[str, Any]] = None) -> RecommendationRun:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        event = _new_audit_event(event_type=event_type, details=details or {})
        run.audit_log.append(event)
        self.update_run(run, sync_index=False)
        return run

    def transition_status(self, run_id: str, new_status: str, details: Optional[Dict[str, Any]] = None) -> RecommendationRun:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        current = run.status
        _validate_status_transition(current, new_status)

        if current in PHASE_COMPLETED_EVENT and new_status != current:
            run.audit_log.append(_new_audit_event(PHASE_COMPLETED_EVENT[current], details or {}))

        run.status = new_status

        if new_status in PHASE_STARTED_EVENT:
            run.audit_log.append(_new_audit_event(PHASE_STARTED_EVENT[new_status], details or {}))
        elif new_status == "completed":
            run.audit_log.append(_new_audit_event("run_completed", details or {}))
        elif new_status == "failed":
            run.audit_log.append(_new_audit_event("run_failed", details or {}))

        self.update_run(run, sync_index=(new_status in {"completed", "failed"}))
        return run

    def attach_artifact(self, run_id: str, artifact_name: str, payload: Any) -> Path:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        out = self.artifacts_dir / f"run_id={run_id}" / f"{artifact_name}.json"
        staged = self._stage_path(out)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(json.dumps(_json_sanitize(payload), indent=2))
        self._finalize(staged, out)

        artifacts = dict(run.metadata.get("artifacts", {}) or {})
        artifacts[artifact_name] = str(out)
        run.metadata["artifacts"] = artifacts
        self.update_run(run, sync_index=False)
        return out

    def merge_metadata(self, run_id: str, metadata_patch: Dict[str, Any]) -> RecommendationRun:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        run.metadata = merge_metadata_patch(dict(run.metadata or {}), dict(metadata_patch or {}))
        self.update_run(run, sync_index=False)
        return run

    @staticmethod
    def _enforce_immutability(existing: RecommendationRun, updated: RecommendationRun) -> None:
        immutable_pairs = [
            ("company_id", existing.company_id, updated.company_id),
            ("as_of_time", existing.as_of_time, updated.as_of_time),
            ("created_at", existing.created_at, updated.created_at),
            ("snapshot_id", existing.frozen_state.snapshot_id, updated.frozen_state.snapshot_id),
            ("snapshot_hash", existing.frozen_state.snapshot_hash, updated.frozen_state.snapshot_hash),
            ("snapshot_version", existing.frozen_state.snapshot_version, updated.frozen_state.snapshot_version),
        ]
        for field_name, before, after in immutable_pairs:
            if before != after:
                raise ValueError(f"Immutable field changed for run {existing.run_id}: {field_name}")

        if existing.model_versions.to_dict() != updated.model_versions.to_dict():
            raise ValueError(f"Model versions are immutable for run {existing.run_id}")


def create_recommendation_run(
    company_id: str,
    as_of_time: str | datetime,
    objectives: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    scenario: Optional[Dict[str, Any]] = None,
    run_store: Optional[RecommendationRunStore] = None,
    model_registry: Optional[ModelRegistry] = None,
    model_versions: Optional[ModelVersionBundle] = None,
    snapshot_root: Optional[str | Path] = None,
    snapshot_path: Optional[str | Path] = None,
    snapshot_builder: Optional[Any] = None,
    snapshot_loader: Optional[Callable[[str, datetime], Dict[str, Any]]] = None,
    entity_graph_path: str | Path = "data/inputs_layer/entity_graph.parquet",
    entity_identifier_path: str | Path = "data/inputs_layer/entity_identifier.parquet",
    company_aliases: Optional[Sequence[str]] = None,
    skip_as_of_lower_bound_validation: bool = False,
    planner_random_seed: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a frozen RecommendationRun and persist it.
    """
    as_of_dt = _parse_ts(as_of_time)

    obj = ObjectiveVector.from_any(objectives)
    cset = ConstraintSet.from_any(constraints)
    scen = ScenarioAssumptions.from_any(scenario)
    resolved_entity_graph_path = resolve_data_path(entity_graph_path)
    resolved_entity_identifier_path = resolve_data_path(entity_identifier_path)
    skip_company_validation = _skip_company_validation()
    snapshot_aliases = [str(company_id), *(list(company_aliases or []))]
    prefetched_snapshot: Optional[Dict[str, Any]] = None
    validation_via_snapshot_fallback = False

    if not skip_company_validation:
        try:
            _validate_company_id_exists(
                company_id=company_id,
                entity_graph_path=Path(resolved_entity_graph_path),
                entity_identifier_path=Path(resolved_entity_identifier_path),
                extra_aliases=company_aliases,
            )
        except ValueError:
            if company_aliases:
                prefetched_snapshot = _resolve_snapshot(
                    company_id=company_id,
                    as_of_time=as_of_dt,
                    snapshot_root=Path(snapshot_root) if snapshot_root else None,
                    snapshot_path=Path(snapshot_path) if snapshot_path else None,
                    snapshot_builder=snapshot_builder,
                    snapshot_loader=snapshot_loader,
                    aliases=snapshot_aliases,
                )
                if _snapshot_has_material_features(prefetched_snapshot):
                    validation_via_snapshot_fallback = True
                else:
                    raise
            else:
                raise
    if (
        not skip_company_validation
        and not skip_as_of_lower_bound_validation
        and not validation_via_snapshot_fallback
    ):
        _validate_as_of_lower_bound(
            company_id=company_id,
            as_of_time=as_of_dt,
            entity_graph_path=Path(resolved_entity_graph_path),
            entity_identifier_path=Path(resolved_entity_identifier_path),
            extra_aliases=company_aliases,
        )

    if not (snapshot_path or skip_company_validation or validation_via_snapshot_fallback):
        snapshot_aliases = _validation_aliases(
            company_id,
            Path(resolved_entity_identifier_path),
            company_aliases,
        )

    snapshot = prefetched_snapshot or _resolve_snapshot(
        company_id=company_id,
        as_of_time=as_of_dt,
        snapshot_root=Path(snapshot_root) if snapshot_root else None,
        snapshot_path=Path(snapshot_path) if snapshot_path else None,
        snapshot_builder=snapshot_builder,
        snapshot_loader=snapshot_loader,
        aliases=snapshot_aliases,
    )

    snapshot = _apply_scenario_overrides(snapshot, scen)
    snapshot_hash = _hash_snapshot(snapshot)
    snapshot_ref = FrozenStateReference(
        snapshot_id=str(snapshot.get("snapshot_id") or str(uuid.uuid4())),
        snapshot_hash=snapshot_hash,
        snapshot_version=_snapshot_version(snapshot),
    )

    metadata_payload = merge_metadata_patch(
        dict(metadata or {}),
        {
            "config": {
                "create": build_create_config(
                    snapshot_root=snapshot_root,
                    snapshot_path=snapshot_path,
                    entity_graph_path=resolved_entity_graph_path,
                    entity_identifier_path=resolved_entity_identifier_path,
                    planner_random_seed=planner_random_seed,
                ),
                "runtime_env": capture_runtime_env_config(),
            }
        },
    )

    versions = model_versions or (model_registry or ModelRegistry()).get_current_versions()
    run = RecommendationRun(
        run_id=str(uuid.uuid4()),
        company_id=str(company_id),
        created_at=_now_iso(),
        as_of_time=as_of_dt.isoformat(),
        objectives=obj,
        constraints=cset,
        scenario=scen,
        frozen_state=snapshot_ref,
        model_versions=versions,
        data_cutoff=DataCutoffSpec(
            published_at_lte=as_of_dt.isoformat(),
            ingested_at_lte=as_of_dt.isoformat(),
        ),
        status="initialized",
        audit_log=[],
        metadata=metadata_payload,
        planner_random_seed=planner_random_seed,
    )

    store = run_store or RecommendationRunStore()
    run.audit_log.append(
        _new_audit_event(
            "run_created",
            {
                "company_id": run.company_id,
                "as_of_time": run.as_of_time,
                "planner_random_seed": planner_random_seed,
            },
        )
    )
    run.audit_log.append(
        _new_audit_event(
            "snapshot_frozen",
            {
                "snapshot_id": snapshot_ref.snapshot_id,
                "snapshot_hash": snapshot_ref.snapshot_hash,
                "snapshot_version": snapshot_ref.snapshot_version,
            },
        )
    )
    store.write_run(run)
    return run.run_id


def enforce_data_cutoff(
    df: pd.DataFrame,
    cutoff: DataCutoffSpec,
    published_col: str = "published_at",
    ingested_col: str = "ingested_at",
) -> pd.DataFrame:
    """Filter a dataframe to rows with published/ingested timestamps <= cutoff."""
    out = df.copy()
    pub_cut = _parse_ts(cutoff.published_at_lte)
    ing_cut = _parse_ts(cutoff.ingested_at_lte)

    if published_col in out.columns:
        out[published_col] = pd.to_datetime(out[published_col], utc=True, errors="coerce")
        out = out[(out[published_col].isna()) | (out[published_col] <= pub_cut)]

    if ingested_col in out.columns:
        out[ingested_col] = pd.to_datetime(out[ingested_col], utc=True, errors="coerce")
        out = out[(out[ingested_col].isna()) | (out[ingested_col] <= ing_cut)]

    return out.reset_index(drop=True)


def validate_plan_hard_constraints(
    plan_actions: Sequence[Dict[str, Any]],
    constraints: ConstraintSet,
    projected_state: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return list of hard-constraint violations for a proposed plan."""
    violations: List[str] = []
    projected_state = projected_state or {}

    action_ids = [str(a.get("action_id", "")) for a in plan_actions]
    action_types = [str(a.get("action_type", "")) for a in plan_actions]

    for c in constraints.hard_constraints:
        ctype = c.constraint_type
        params = c.parameters

        if ctype == "no_equity_issuance":
            has_equity_action = any("equity_issuance" in x for x in action_ids)
            if not has_equity_action:
                for a in plan_actions:
                    subtype = str(a.get("action_subtype", "")).strip()
                    atype = str(a.get("action_type", "")).strip()
                    if subtype == "equity_issuance" or atype == "equity_issuance":
                        has_equity_action = True
                        break
            if has_equity_action:
                violations.append(f"{c.constraint_id}:no_equity_issuance")

        elif ctype == "forbidden_action_type":
            target_type = str(params.get("action_type", ""))
            target_id = str(params.get("action_id", ""))
            if target_type and target_type in action_types:
                violations.append(f"{c.constraint_id}:forbidden_action_type:{target_type}")
            if target_id and target_id in action_ids:
                violations.append(f"{c.constraint_id}:forbidden_action_id:{target_id}")

        elif ctype == "required_action_type":
            target_type = str(params.get("action_type", ""))
            target_id = str(params.get("action_id", ""))
            ok = True
            if target_type:
                ok = ok and (target_type in action_types)
            if target_id:
                ok = ok and (target_id in action_ids)
            if not ok:
                violations.append(f"{c.constraint_id}:required_action_missing")

        elif ctype == "leverage_limit":
            max_lev = params.get("max_leverage")
            lev = projected_state.get("capital_structure.net_leverage")
            try:
                if max_lev is not None and lev is not None and float(lev) > float(max_lev):
                    violations.append(f"{c.constraint_id}:leverage_limit")
            except Exception:
                violations.append(f"{c.constraint_id}:invalid_leverage_limit")

        elif ctype == "minimum_cash_reserve":
            min_cash = params.get("min_cash_reserve")
            cash = projected_state.get("liquidity.cash")
            try:
                if min_cash is not None and cash is not None and float(cash) < float(min_cash):
                    violations.append(f"{c.constraint_id}:minimum_cash_reserve")
            except Exception:
                violations.append(f"{c.constraint_id}:invalid_minimum_cash_reserve")

        elif ctype == "max_acquisition_size":
            max_pct = params.get("max_size_pct_ev")
            max_usd = params.get("max_size_usd")
            for a in plan_actions:
                aid = str(a.get("action_id", ""))
                if "acquisition" not in aid:
                    continue
                aparams = dict(a.get("params", {}) or a.get("parameters", {}) or {})
                if max_pct is not None and aparams.get("target_size_pct_ev") is not None:
                    try:
                        if float(aparams["target_size_pct_ev"]) > float(max_pct):
                            violations.append(f"{c.constraint_id}:max_acquisition_size_pct")
                    except Exception:
                        violations.append(f"{c.constraint_id}:invalid_acquisition_size_pct")
                if max_usd is not None and aparams.get("size_absolute_usd") is not None:
                    try:
                        if float(aparams["size_absolute_usd"]) > float(max_usd):
                            violations.append(f"{c.constraint_id}:max_acquisition_size_usd")
                    except Exception:
                        violations.append(f"{c.constraint_id}:invalid_acquisition_size_usd")

        elif ctype == "maintain_investment_grade":
            is_ig = projected_state.get("capital_structure.rating_state.is_investment_grade")
            if is_ig is False:
                violations.append(f"{c.constraint_id}:maintain_investment_grade")

    return violations


# -------------------------------
# Internal helpers
# -------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_audit_event(event_type: str, details: Dict[str, Any]) -> AuditEvent:
    if event_type not in AUDIT_EVENT_TYPES:
        raise ValueError(f"Unsupported audit event type: {event_type}")
    return AuditEvent(
        event_id=str(uuid.uuid4()),
        timestamp=_now_iso(),
        event_type=event_type,
        details=_json_sanitize(details),
    )


def _parse_ts(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = pd.to_datetime(value, utc=True).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _json_sanitize(obj: Any) -> Any:
    if is_dataclass(obj):
        return _json_sanitize(asdict(obj))
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        if obj.tzinfo is None:
            return obj.tz_localize(timezone.utc).isoformat()
        return obj.tz_convert(timezone.utc).isoformat()
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=timezone.utc)
        return obj.astimezone(timezone.utc).isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, complex):
        return float(obj.real) if abs(obj.imag) < 1e-9 else None
    return obj


def _canonical_json_hash(payload: Dict[str, Any]) -> str:
    txt = json.dumps(_json_sanitize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def _hash_snapshot(snapshot: Dict[str, Any]) -> str:
    return _canonical_json_hash(snapshot)


def _snapshot_version(snapshot: Dict[str, Any]) -> str:
    prov = snapshot.get("provenance", {}) if isinstance(snapshot.get("provenance"), dict) else {}
    version = prov.get("computation_version") or snapshot.get("snapshot_version")
    return str(version or "unknown")


def _snapshot_has_material_features(snapshot: Dict[str, Any]) -> bool:
    features = snapshot.get("features")
    if not isinstance(features, dict) or not features:
        return False
    for value in features.values():
        if isinstance(value, dict):
            if value.get("value") is not None:
                return True
        elif value is not None:
            return True
    return False


def _id_aliases(company_id: str) -> List[str]:
    cid = str(company_id).strip()
    if not cid:
        return []
    out = [cid]
    if cid.isdigit():
        out.extend([cid.lstrip("0") or "0", cid.zfill(10), cid.zfill(6)])
    return list(dict.fromkeys([x for x in out if x]))


def _skip_company_validation() -> bool:
    return str(os.environ.get("AXIOM_SKIP_RUN_COMPANY_VALIDATION", "")).strip().lower() in {"1", "true", "yes", "on"}


def _aliases_from_cik_gvkey(company_id: str, cik_gvkey_path: Path = Path("data/wrds/compustat/cik_gvkey.csv.gz")) -> List[str]:
    if not cik_gvkey_path.exists():
        return []
    try:
        df = pd.read_csv(cik_gvkey_path, dtype=str)
    except Exception:
        return []
    if df.empty:
        return []

    df.columns = [c.lower() for c in df.columns]
    if "gvkey" not in df.columns or "cik" not in df.columns:
        return []

    aliases = set(_id_aliases(company_id))
    gv = df["gvkey"].astype(str).str.strip()
    cik = df["cik"].astype(str).str.strip()
    mask = gv.isin(aliases) | gv.str.zfill(6).isin({a.zfill(6) for a in aliases if a.isdigit()})
    mask = mask | cik.isin(aliases) | cik.str.zfill(10).isin({a.zfill(10) for a in aliases if a.isdigit()})
    if not mask.any():
        return []

    out: List[str] = []
    for x in df.loc[mask, "gvkey"].dropna().astype(str).tolist():
        out.extend(_id_aliases(x))
    for x in df.loc[mask, "cik"].dropna().astype(str).tolist():
        out.extend(_id_aliases(x))
    return list(dict.fromkeys(out))


def _snapshot_company_aliases(company_id: str, entity_identifier_path: Path) -> List[str]:
    aliases = list(_id_aliases(company_id))
    aliases.extend(_aliases_from_cik_gvkey(company_id))
    aliases = list(dict.fromkeys(aliases))
    if not entity_identifier_path.exists():
        return aliases

    try:
        df = _read_parquet_columns(entity_identifier_path, ["entity_id", "identifier_value"])
    except Exception:
        return aliases

    id_aliases = set(aliases)
    mask = df["entity_id"].astype(str).isin(id_aliases) | df["identifier_value"].astype(str).isin(id_aliases)
    if not mask.any():
        return aliases

    for x in df.loc[mask, "entity_id"].dropna().astype(str).tolist():
        aliases.extend(_id_aliases(x))
    for x in df.loc[mask, "identifier_value"].dropna().astype(str).tolist():
        aliases.extend(_id_aliases(x))

    return list(dict.fromkeys(aliases))


def _resolve_snapshot(
    company_id: str,
    as_of_time: datetime,
    snapshot_root: Optional[Path],
    snapshot_path: Optional[Path],
    snapshot_builder: Optional[Any],
    snapshot_loader: Optional[Callable[[str, datetime], Dict[str, Any]]],
    aliases: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if snapshot_loader is not None:
        snap = snapshot_loader(company_id, as_of_time)
        if not isinstance(snap, dict):
            raise ValueError("snapshot_loader must return dict snapshot")
        return snap

    alias_values = list(dict.fromkeys(list(aliases or []) + _id_aliases(company_id)))
    as_of_date = as_of_time.strftime("%Y-%m-%d")

    if snapshot_root is not None:
        for alias in alias_values:
            p = snapshot_root / "keyed" / f"as_of_date={as_of_date}" / f"company_id={alias}.json"
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                if line.strip():
                    return json.loads(line)

    if snapshot_path is not None and snapshot_path.exists():
        with snapshot_path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_cid = str(row.get("company_id", ""))
                row_asof = pd.to_datetime(row.get("as_of_time"), utc=True, errors="coerce")
                if pd.isna(row_asof):
                    continue
                if row_cid in alias_values and row_asof.strftime("%Y-%m-%d") == as_of_date:
                    return row

    if snapshot_builder is not None:
        if hasattr(snapshot_builder, "get_snapshot"):
            row = snapshot_builder.get_snapshot(company_id, as_of_time)
        elif hasattr(snapshot_builder, "build"):
            row = snapshot_builder.build(company_id, as_of_time)
        else:
            raise ValueError("snapshot_builder must expose get_snapshot(...) or build(...)")
        if is_dataclass(row):
            return asdict(row)
        if isinstance(row, dict):
            return row
        raise ValueError(f"Unsupported snapshot builder return type: {type(row)}")

    raise ValueError(
        "Could not resolve frozen snapshot. Provide one of: "
        "snapshot_loader, snapshot_root, snapshot_path, snapshot_builder."
    )


def _apply_scenario_overrides(snapshot: Dict[str, Any], scenario: ScenarioAssumptions) -> Dict[str, Any]:
    out = dict(snapshot)
    regime = dict(out.get("regime", {}) or {})

    if scenario.credit_regime_override != "none":
        regime["credit_regime"] = scenario.credit_regime_override
    if scenario.risk_regime_override != "none":
        regime["risk_regime"] = scenario.risk_regime_override
    if scenario.vol_regime_override != "none":
        regime["vol_regime"] = scenario.vol_regime_override
    if scenario.sector_cycle_override != "none":
        regime["sector_cycle"] = scenario.sector_cycle_override

    signals = dict(regime.get("signals", {}) or {})
    if scenario.interest_rate_shift_bp:
        signals["interest_rate_shift_bp"] = int(scenario.interest_rate_shift_bp)
    if scenario.equity_market_drawdown_pct:
        signals["equity_market_drawdown_pct"] = float(scenario.equity_market_drawdown_pct)
    if scenario.custom_flags:
        signals["custom_flags"] = dict(scenario.custom_flags)
    if signals:
        regime["signals"] = signals

    out["regime"] = regime
    return out


def _validate_status_transition(current: str, new_status: str) -> None:
    if current not in RUN_STATUSES:
        raise ValueError(f"Invalid current status: {current}")
    if new_status not in RUN_STATUSES:
        raise ValueError(f"Invalid target status: {new_status}")
    if current == new_status:
        return
    if current in {"completed", "failed"}:
        raise ValueError(f"Cannot transition terminal status {current} -> {new_status}")
    if new_status == "failed":
        return

    allowed = {
        "initialized": {"candidate_generation"},
        "candidate_generation": {"feasibility_evaluation"},
        "feasibility_evaluation": {"precedent_retrieval"},
        "precedent_retrieval": {"plan_search"},
        "plan_search": {"completed"},
    }
    next_allowed = allowed.get(current, set())
    if new_status not in next_allowed:
        raise ValueError(f"Invalid status transition: {current} -> {new_status}")


def _read_parquet_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(columns))
    return table.to_pandas()


def _validation_aliases(
    company_id: str,
    entity_identifier_path: Path,
    extra_aliases: Optional[Sequence[str]] = None,
) -> List[str]:
    aliases = list(_snapshot_company_aliases(company_id, entity_identifier_path))
    for alias in extra_aliases or []:
        if alias is None:
            continue
        text = str(alias).strip()
        if not text:
            continue
        aliases.extend(_id_aliases(text))
    return list(dict.fromkeys(aliases))


def _validate_company_id_exists(
    company_id: str,
    entity_graph_path: Path,
    entity_identifier_path: Path,
    extra_aliases: Optional[Sequence[str]] = None,
) -> None:
    aliases = set(_validation_aliases(company_id, entity_identifier_path, extra_aliases))

    found = False
    if entity_graph_path.exists():
        cols = ["entity_id", "related_id"]
        df = _read_parquet_columns(entity_graph_path, [c for c in cols if c])
        for c in cols:
            if c in df.columns:
                vals = set(df[c].dropna().astype(str).tolist())
                if aliases.intersection(vals):
                    found = True
                    break

    if not found and entity_identifier_path.exists():
        cols = ["entity_id", "identifier_value"]
        df = _read_parquet_columns(entity_identifier_path, cols)
        entity_vals = set(df["entity_id"].dropna().astype(str).tolist())
        ident_vals = set(df["identifier_value"].dropna().astype(str).tolist())
        if aliases.intersection(entity_vals) or aliases.intersection(ident_vals):
            found = True

    if not found:
        raise ValueError(f"company_id not found in entity graph: {company_id}")


def _validate_as_of_lower_bound(
    company_id: str,
    as_of_time: datetime,
    entity_graph_path: Path,
    entity_identifier_path: Path,
    extra_aliases: Optional[Sequence[str]] = None,
) -> None:
    earliest = _earliest_company_data_time(
        company_id,
        entity_graph_path,
        entity_identifier_path,
        extra_aliases=extra_aliases,
    )
    if earliest is not None and as_of_time < earliest:
        raise ValueError(
            f"as_of_time {as_of_time.isoformat()} is earlier than earliest company data {earliest.isoformat()}"
        )


def _earliest_company_data_time(
    company_id: str,
    entity_graph_path: Path,
    entity_identifier_path: Path,
    extra_aliases: Optional[Sequence[str]] = None,
) -> Optional[datetime]:
    aliases = set(_validation_aliases(company_id, entity_identifier_path, extra_aliases))
    entity_ids = set(aliases)

    if entity_identifier_path.exists():
        df_ident = _read_parquet_columns(entity_identifier_path, ["entity_id", "identifier_value"])
        mask = df_ident["identifier_value"].astype(str).isin(aliases) | df_ident["entity_id"].astype(str).isin(aliases)
        if mask.any():
            entity_ids.update(df_ident.loc[mask, "entity_id"].dropna().astype(str).tolist())

    times: List[datetime] = []
    if entity_graph_path.exists():
        cols = [
            "entity_id",
            "related_id",
            "valid_from",
            "effective_at",
            "published_at",
            "ingested_at",
        ]
        df = _read_parquet_columns(entity_graph_path, cols)
        mask = df["entity_id"].astype(str).isin(entity_ids) | df["related_id"].astype(str).isin(entity_ids)
        df = df.loc[mask]
        for c in ["valid_from", "effective_at", "published_at", "ingested_at"]:
            if c not in df.columns:
                continue
            ser = pd.to_datetime(df[c], utc=True, errors="coerce").dropna()
            if not ser.empty:
                times.append(ser.min().to_pydatetime())

    if not times:
        return None
    out = min(times)
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


__all__ = [
    "AUDIT_EVENT_TYPES",
    "CONSTRAINT_PRIORITIES",
    "CONSTRAINT_SOURCES",
    "CONSTRAINT_TYPES",
    "DataCutoffSpec",
    "FrozenStateReference",
    "ModelRegistry",
    "ModelVersionBundle",
    "ObjectiveVector",
    "RecommendationRun",
    "RecommendationRunStore",
    "ScenarioAssumptions",
    "Constraint",
    "ConstraintSet",
    "AuditEvent",
    "RUN_STATUSES",
    "create_recommendation_run",
    "enforce_data_cutoff",
    "validate_plan_hard_constraints",
]
