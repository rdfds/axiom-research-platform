from __future__ import annotations

import gzip
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .latent_regime_model import (
    fit_latent_regime_kmeans,
    latent_regime_memberships,
    latent_regime_similarity,
    raw_feature_matrix_from_compacts,
)
from .precedent_brain import (
    _outcome_aware_reranker_feature_frame,
    _outcome_aware_reranker_feature_names,
    _sector_similarity,
    _second_stage_reranker_feature_matrix,
    _second_stage_reranker_feature_names,
    _STATE_VECTOR_MATCHING_COLS,
    _STATE_VECTOR_V2_DEFAULT_FEATURE_TRANSFORMS,
    _normalize_feature_transform_mode,
    _transform_matching_values,
)


def load_pairwise_supervision(path: str | Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dataset_path = Path(path)
    open_fn = gzip.open if dataset_path.suffix == ".gz" else open
    with open_fn(dataset_path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _clean_scope_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _feature_names() -> Tuple[str, ...]:
    return tuple(_STATE_VECTOR_MATCHING_COLS)


_INTERACTION_FEATURE_PREFIX = "pairwise_interaction::"
_PENALTY_FEATURE_PREFIX = "pairwise_penalty::"
_LATENT_REGIME_FEATURE_PREFIX = "latent_regime::"
_LATENT_REGIME_SIMILARITY_FEATURE = f"{_LATENT_REGIME_FEATURE_PREFIX}similarity"


def _pairwise_group_key(row: Dict[str, Any]) -> str:
    return (
        f"{str(row.get('company_id') or '')}|"
        f"{str(row.get('as_of_time') or '')}|"
        f"{str(row.get('anchor_action_id') or '')}"
    )


def _scope_config_from_payload(payload: Dict[str, Any], scope_key: str) -> Dict[str, Any]:
    scopes = dict(payload.get("scopes", {}) or {})
    scope = _clean_scope_key(scope_key)
    exact = scopes.get(scope)
    if isinstance(exact, dict):
        return exact
    family = scope.split(".", 1)[0] if "." in scope else scope
    family_scope = scopes.get(family)
    if isinstance(family_scope, dict):
        return family_scope
    all_scope = scopes.get("ALL")
    if isinstance(all_scope, dict):
        return all_scope
    return {}


def load_feature_weight_prior(
    base_payload_path: str | Path,
    *,
    scope_key: str,
    feature_names: Sequence[str],
    missing_default: float = 1.0,
) -> np.ndarray:
    payload = json.loads(Path(base_payload_path).read_text())
    scope = _scope_config_from_payload(payload, scope_key)
    weights = dict(scope.get("feature_relative_weights", {}) or {})
    penalties = dict(scope.get("penalties", {}) or {})
    for term in list(scope.get("interaction_terms", []) or []):
        if not isinstance(term, dict):
            continue
        features = list(term.get("features") or [])
        if len(features) != 2:
            continue
        interaction_name = _interaction_feature_name(str(features[0]), str(features[1]))
        try:
            weights[interaction_name] = float(term.get("weight"))
        except Exception:
            continue
    prior_values: List[float] = []
    for name in feature_names:
        if name in weights:
            prior_values.append(float(weights[name]))
            continue
        name_text = str(name)
        if name_text == f"{_PENALTY_FEATURE_PREFIX}size_gap_excess":
            prior_values.append(float(penalties.get("size_penalty_weight") or 0.0))
            continue
        if name_text == f"{_PENALTY_FEATURE_PREFIX}primary_burden_gap_excess":
            prior_values.append(float(penalties.get("burden_penalty_weight") or 0.0))
            continue
        if name_text.startswith(_INTERACTION_FEATURE_PREFIX) or name_text.startswith(_LATENT_REGIME_FEATURE_PREFIX):
            prior_values.append(0.0)
        else:
            prior_values.append(float(missing_default))
    arr = np.array(prior_values, dtype=float)
    positive = arr[arr > 0.0]
    if positive.size:
        arr = arr / float(np.mean(positive))
    elif missing_default > 0.0:
        arr = np.ones(len(feature_names), dtype=float)
    else:
        arr = np.zeros(len(feature_names), dtype=float)
    return arr


