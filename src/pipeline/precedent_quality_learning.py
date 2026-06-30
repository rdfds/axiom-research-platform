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


def load_feature_weight_floor(
    *,
    feature_names: Sequence[str],
    base_feature_floor: float = 0.25,
) -> np.ndarray:
    floors: List[float] = []
    for name in feature_names:
        name_text = str(name)
        if (
            name_text.startswith(_INTERACTION_FEATURE_PREFIX)
            or name_text.startswith(_PENALTY_FEATURE_PREFIX)
            or name_text.startswith(_LATENT_REGIME_FEATURE_PREFIX)
        ):
            floors.append(0.0)
        else:
            floors.append(float(base_feature_floor))
    return np.asarray(floors, dtype=float)


def load_penalty_feature_specs(
    base_payload_path: str | Path,
    *,
    scope_key: str,
) -> List[Dict[str, Any]]:
    payload = json.loads(Path(base_payload_path).read_text())
    scope = _scope_config_from_payload(payload, scope_key)
    gates = dict(scope.get("gates", {}) or {})
    primary_burden_feature = str(scope.get("primary_burden_feature") or "state_vector_v1.net_obligation_burden").strip()
    specs: List[Dict[str, Any]] = [
        {
            "name": "size_gap_excess",
            "source_feature": "state_vector_v1.size_log_revenue",
            "soft_threshold": float(gates.get("soft_size_gap") or 0.35),
        },
    ]
    if primary_burden_feature:
        specs.append(
            {
                "name": "primary_burden_gap_excess",
                "source_feature": primary_burden_feature,
                "soft_threshold": float(gates.get("soft_burden_gap") or 1.25),
            }
        )
    return specs


def _feature_advantage(row: Dict[str, Any], feature_name: str) -> Optional[float]:
    gap_summary = dict(row.get("feature_gap_summary") or {})
    feature_payload = dict(gap_summary.get(feature_name) or {})
    pos = feature_payload.get("positive_abs_diff")
    neg = feature_payload.get("negative_abs_diff")
    try:
        if pos is None or neg is None:
            return None
        pos_f = float(pos)
        neg_f = float(neg)
        if not np.isfinite(pos_f) or not np.isfinite(neg_f):
            return None
        return neg_f - pos_f
    except Exception:
        return None


def _feature_transform_prior(
    base_payload_path: str | Path,
    *,
    scope_key: str,
    feature_names: Sequence[str],
    feature_transform_mode: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(Path(base_payload_path).read_text())
    scope = _scope_config_from_payload(payload, scope_key)
    normalized_mode = _normalize_feature_transform_mode(
        feature_transform_mode if feature_transform_mode is not None else scope.get("feature_transform_mode")
    )
    overrides = dict(scope.get("feature_transforms", {}) or {})
    out: Dict[str, Dict[str, Any]] = {}
    for feature_name in feature_names:
        spec = {}
        if normalized_mode != "identity":
            spec = dict(_STATE_VECTOR_V2_DEFAULT_FEATURE_TRANSFORMS.get(feature_name, {}) or {})
        override = overrides.get(feature_name)
        if isinstance(override, dict):
            spec.update(override)
        out[feature_name] = _normalize_transform_spec(spec)
    return out


def load_feature_transform_prior(
    base_payload_path: str | Path,
    *,
    scope_key: str,
    feature_names: Optional[Sequence[str]] = None,
    feature_transform_mode: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    names = list(feature_names or _feature_names())
    return _feature_transform_prior(
        base_payload_path,
        scope_key=scope_key,
        feature_names=names,
        feature_transform_mode=feature_transform_mode,
    )


def _clean_numeric(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


def _normalize_transform_spec(spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        return {}
    out: Dict[str, Any] = {}
    kind = str(spec.get("kind") or "").strip().lower()
    if kind and kind not in {"identity", "none"}:
        out["kind"] = kind
    cap = _clean_numeric(spec.get("cap"))
    if cap is not None and cap > 0.0:
        out["cap"] = float(cap)
    scale = _clean_numeric(spec.get("scale"))
    if scale is not None and scale > 0.0:
        out["scale"] = float(scale)
    return out


def _normalize_pair_weight_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"target_regime_rarity", "regime_rarity", "target_density", "rare_target"}:
        return "target_regime_rarity"
    if text in {"teacher_confidence", "teacher_margin", "confidence"}:
        return "teacher_confidence"
    return "uniform"


def _transform_spec_key(spec: Dict[str, Any]) -> Tuple[Any, ...]:
    normalized = _normalize_transform_spec(spec)
    return (
        normalized.get("kind"),
        round(float(normalized.get("cap")), 8) if normalized.get("cap") is not None else None,
        round(float(normalized.get("scale")), 8) if normalized.get("scale") is not None else None,
    )


def _pair_teacher_confidence_weight(row: Dict[str, Any], mode: Any) -> float:
    normalized_mode = _normalize_pair_weight_mode(mode)
    if normalized_mode in {"uniform", "target_regime_rarity"}:
        return 1.0
    pos_score = _clean_numeric(row.get("positive_similarity_score"))
    neg_score = _clean_numeric(row.get("negative_similarity_score"))
    if pos_score is not None and neg_score is not None:
        return max(0.0, float(pos_score) - float(neg_score))
    pos_rank = _clean_numeric(row.get("positive_precedent_rank_within_candidate"))
    neg_rank = _clean_numeric(row.get("negative_precedent_rank_within_candidate"))
    if pos_rank is not None and neg_rank is not None:
        return max(0.0, float(neg_rank) - float(pos_rank))
    return 1.0


def _target_regime_rarity_weights(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_names: Sequence[str],
    transform_specs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, float]:
    base_features = [
        str(feature)
        for feature in list(feature_names or [])
        if not str(feature).startswith(_INTERACTION_FEATURE_PREFIX)
        and not str(feature).startswith(_LATENT_REGIME_FEATURE_PREFIX)
    ]
    if not base_features:
        return {}

    group_rows: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        group_key = _pairwise_group_key(row)
        if group_key and group_key not in group_rows:
            group_rows[group_key] = dict(row)
    ordered_groups = list(group_rows.keys())
    if len(ordered_groups) < 3:
        return {group_key: 1.0 for group_key in ordered_groups}

    transform_overrides = {
        str(feature): _normalize_transform_spec(spec)
        for feature, spec in dict(transform_specs or {}).items()
    }
    target_matrix = np.full((len(ordered_groups), len(base_features)), np.nan, dtype=float)
    for row_idx, group_key in enumerate(ordered_groups):
        target_compact = dict(group_rows[group_key].get("target_compact") or {})
        for feature_idx, feature_name in enumerate(base_features):
            value = _clean_numeric(target_compact.get(feature_name))
            if value is None:
                continue
            transformed = _transform_matching_values(
                np.asarray([float(value)], dtype=float),
                transform_overrides.get(feature_name),
            )
            target_matrix[row_idx, feature_idx] = float(transformed[0]) if transformed.size else float(value)

    standardized = np.column_stack(
        [_robust_standardize_vector(target_matrix[:, idx]) for idx in range(target_matrix.shape[1])]
    )
    pairwise_distance = np.full((standardized.shape[0], standardized.shape[0]), np.nan, dtype=float)
    for left_idx in range(standardized.shape[0]):
        pairwise_distance[left_idx, left_idx] = 0.0
        for right_idx in range(left_idx + 1, standardized.shape[0]):
            valid = np.isfinite(standardized[left_idx]) & np.isfinite(standardized[right_idx])
            if not bool(np.any(valid)):
                continue
            distance = float(np.sqrt(np.mean(np.square(standardized[left_idx, valid] - standardized[right_idx, valid]))))
            pairwise_distance[left_idx, right_idx] = distance
            pairwise_distance[right_idx, left_idx] = distance

    local_rarity = np.ones(standardized.shape[0], dtype=float)
    neighbor_count = min(5, standardized.shape[0] - 1)
    if neighbor_count <= 0:
        return {group_key: 1.0 for group_key in ordered_groups}

    for row_idx in range(standardized.shape[0]):
        candidates = pairwise_distance[row_idx]
        valid = np.isfinite(candidates) & (np.arange(candidates.shape[0]) != row_idx)
        if not bool(np.any(valid)):
            continue
        ordered = np.sort(candidates[valid])[:neighbor_count]
        if ordered.size:
            local_rarity[row_idx] = float(np.mean(ordered))

    finite = local_rarity[np.isfinite(local_rarity) & (local_rarity > 0.0)]
    if finite.size == 0:
        return {group_key: 1.0 for group_key in ordered_groups}
    median_rarity = float(np.median(finite))
    if not np.isfinite(median_rarity) or median_rarity <= 1e-9:
        return {group_key: 1.0 for group_key in ordered_groups}

    rarity_weights = np.sqrt(np.maximum(local_rarity, 1e-9) / median_rarity)
    finite_weight_mask = np.isfinite(rarity_weights) & (rarity_weights > 0.0)
    if bool(np.any(finite_weight_mask)):
        rarity_weights = rarity_weights / float(np.mean(rarity_weights[finite_weight_mask]))
    else:
        rarity_weights = np.ones_like(rarity_weights, dtype=float)
    return {
        group_key: float(rarity_weights[idx]) if np.isfinite(rarity_weights[idx]) and rarity_weights[idx] > 0.0 else 1.0
        for idx, group_key in enumerate(ordered_groups)
    }


def _compact_feature_triplet(row: Dict[str, Any], feature_name: str) -> Optional[Tuple[float, float, float]]:
    target = _clean_numeric(dict(row.get("target_compact") or {}).get(feature_name))
    positive = _clean_numeric(dict(row.get("positive_compact") or {}).get(feature_name))
    negative = _clean_numeric(dict(row.get("negative_compact") or {}).get(feature_name))
    if target is None or positive is None or negative is None:
        return None
    return target, positive, negative


def _feature_advantage_from_compacts(
    row: Dict[str, Any],
    feature_name: str,
    transform_spec: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    triplet = _compact_feature_triplet(row, feature_name)
    if triplet is None:
        return None
    target, positive, negative = triplet
    values = np.array([target, positive, negative], dtype=float)
    transformed = _transform_matching_values(values, _normalize_transform_spec(transform_spec))
    return float(abs(transformed[0] - transformed[2]) - abs(transformed[0] - transformed[1]))


