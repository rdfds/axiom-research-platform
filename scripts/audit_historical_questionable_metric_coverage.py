#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
import signal
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.company_state_builder import CompanyStateBuilder


DEFAULT_DATES = [
    "2014-12-31",
    "2018-12-31",
    "2020-12-31",
    "2022-12-31",
    "2024-12-31",
]

QUESTIONABLE_METRICS = [
    "liquidity.revolver_undrawn",
    "liquidity.marketable_securities",
    "liquidity.available_for_actions",
    "capital_structure.interest_coverage",
    "capital_structure.current_debt_statement_direct",
    "capital_structure.long_term_debt_statement_direct",
    "capital_structure.interest_expense_statement_direct",
    "capital_structure.net_pension_liability",
    "capital_structure.combined_retirement_liability",
    "capital_structure.retirement_obligation_regime",
]

EXACTISH = {
    "exact",
    "exact_not_applicable",
    "exact_structural_zero",
    "present",
}

DEFAULT_ARTIFACT_CANDIDATES = [
    REPO_ROOT
    / "out"
    / "materialized_feedback_20260405"
    / "company_state_snapshots_asof=2024-12-31.input_layer_v1_smart_normalized_with_sec.feedback_pipeline.jsonl.gz",
    Path(
        "/tmp/consumer_industrial_snapshots_2024_12_31_feedback_20260401/"
        "company_state_snapshots_asof=2024-12-31.input_layer_v1_smart_normalized_with_sec.fix2.debtrepair_v4."
        "feedback_additions_v7_retirement_carryforward_regime.jsonl.gz"
    ),
]


def _default_inputs_root() -> Path:
    tmp_root = Path("/tmp/axiom_v1_inputs")
    return tmp_root if tmp_root.exists() else (REPO_ROOT / "data")


def _default_artifact_path() -> Path:
    for candidate in DEFAULT_ARTIFACT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_ARTIFACT_CANDIDATES[0]


def _load_company_ids(ids_file: Path | None, artifact: Path | None) -> list[str]:
    if ids_file is not None and ids_file.exists():
        ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
        if ids:
            return ids
    if artifact is None or not artifact.exists():
        raise FileNotFoundError("Need either --company-ids-file or --artifact")
    opener = gzip.open if artifact.suffix == ".gz" else open
    ids: list[str] = []
    with opener(artifact, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            company_id = str(row['company_id'] or "").strip()
            if company_id:
                ids.append(company_id)
    return sorted(set(ids))


def _feature_dict(snapshot: Any) :
    features = getattr(snapshot, "features", {}) or {}
    return features


