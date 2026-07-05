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


