from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

from .action_ontology import build_default_action_schema_registry


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_RELEVANT_ACTION_SOURCE_FILES: Sequence[Path] = (
    REPO_ROOT / "src/candidate_generation.py",
    REPO_ROOT / "src/mechanism_brain.py",
    REPO_ROOT / "src/planner_brain.py",
)


def coverage_status(count: int) -> str:
    if count >= 1000:
        return "strong"
    if count >= 100:
        return "usable"
    if count >= 1:
        return "thin"
    return "missing"


