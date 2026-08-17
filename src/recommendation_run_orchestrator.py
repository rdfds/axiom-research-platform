"""RecommendationRun execution orchestration.

Executes staged lifecycle under a frozen run_id and persists artifacts:
- CandidateSet
- FeasibilityResults
- PrecedentMatches
- PrecedentIndex
- PlanSet
- BoardReadyDossier
- RecommendationPackage
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from typing import TYPE_CHECKING

from .recommendation_runtime_config import (
    build_execution_config,
    capture_runtime_env_config,
)
from .model_feature_bundle import attach_model_feature_bundle, feature_view_from_snapshot
from .runtime_feature_adapter import adapt_snapshot, resolve_feature_value

if TYPE_CHECKING:
    from .action_ontology import ActionSchemaRegistry
    from .pipeline.types import PrecedentPack
    from .recommendation_run import RecommendationRun, RecommendationRunStore


def _candidate_bindings():
    from .action_ontology import build_default_action_schema_registry
    from .candidate_generation import generate_action_candidates
    from .mechanism_brain import MechanismBrain

    return build_default_action_schema_registry, generate_action_candidates, MechanismBrain


def _causal_bindings():
    from .causal_model_risk import build_causal_model_risk_report

    return build_causal_model_risk_report


def _precedent_bindings():
    from .pipeline.precedent_index import build_precedent_index
    from .pipeline.run import run_precedent
    from .pipeline.types import PrecedentPack

    return build_precedent_index, run_precedent, PrecedentPack


def _dossier_bindings():
    from .board_ready_dossier import build_board_ready_dossier

    return build_board_ready_dossier


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _run_store_bindings():
    from .recommendation_run import (
        RecommendationRunStore,
        _apply_scenario_overrides,
        _hash_snapshot,
        _parse_ts,
        _resolve_snapshot,
        _snapshot_company_aliases,
        create_recommendation_run,
        validate_plan_hard_constraints,
    )

    return (
        RecommendationRunStore,
        _apply_scenario_overrides,
        _hash_snapshot,
        _parse_ts,
        _resolve_snapshot,
        _snapshot_company_aliases,
        create_recommendation_run,
        validate_plan_hard_constraints,
    )


