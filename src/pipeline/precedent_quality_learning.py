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


def _feature_value_sample(rows: Sequence[Dict[str, Any]], feature_name: str) -> np.ndarray:
    values: List[float] = []
    for row in rows:
        triplet = _compact_feature_triplet(row, feature_name)
        if triplet is None:
            continue
        values.extend(triplet)
    if not values:
        return np.empty(0, dtype=float)
    return np.asarray(values, dtype=float)


def _penalty_feature_name(name: str) -> str:
    return f"{_PENALTY_FEATURE_PREFIX}{str(name or '').strip()}"


def _parse_penalty_feature_name(name: str) -> Optional[str]:
    text = str(name or "").strip()
    if not text.startswith(_PENALTY_FEATURE_PREFIX):
        return None
    suffix = text[len(_PENALTY_FEATURE_PREFIX) :].strip()
    return suffix or None


def _penalty_feature_advantage(
    row: Dict[str, Any],
    *,
    source_feature: str,
    soft_threshold: float,
) -> Optional[float]:
    triplet = _compact_feature_triplet(row, source_feature)
    if triplet is None:
        return None
    target, positive, negative = triplet
    positive_excess = max(abs(target - positive) - float(soft_threshold), 0.0)
    negative_excess = max(abs(target - negative) - float(soft_threshold), 0.0)
    return float(negative_excess - positive_excess)


def _candidate_abs_diff_triplets(
    rows: Sequence[Dict[str, Any]],
    feature_name: str,
    *,
    transform_spec: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    target_values: List[float] = []
    positive_values: List[float] = []
    negative_values: List[float] = []
    for row in rows:
        triplet = _compact_feature_triplet(row, feature_name)
        if triplet is None:
            target_values.append(np.nan)
            positive_values.append(np.nan)
            negative_values.append(np.nan)
            continue
        target_values.append(triplet[0])
        positive_values.append(triplet[1])
        negative_values.append(triplet[2])

    target_arr = np.asarray(target_values, dtype=float)
    positive_arr = np.asarray(positive_values, dtype=float)
    negative_arr = np.asarray(negative_values, dtype=float)
    combined = np.concatenate([target_arr, positive_arr, negative_arr])
    transformed = _transform_matching_values(combined, _normalize_transform_spec(transform_spec))
    standardized = _robust_standardize_vector(transformed)
    n = len(rows)
    target_std = standardized[:n]
    positive_std = standardized[n : 2 * n]
    negative_std = standardized[2 * n :]
    positive_abs = np.abs(target_std - positive_std)
    negative_abs = np.abs(target_std - negative_std)
    return positive_abs, negative_abs


def _interaction_feature_name(feature_a: str, feature_b: str) -> str:
    left, right = sorted((str(feature_a), str(feature_b)))
    return f"{_INTERACTION_FEATURE_PREFIX}{left}::{right}"


def _parse_interaction_feature_name(name: str) -> Optional[Tuple[str, str]]:
    raw = str(name or "")
    if not raw.startswith(_INTERACTION_FEATURE_PREFIX):
        return None
    body = raw[len(_INTERACTION_FEATURE_PREFIX) :]
    parts = body.split("::")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _latent_feature_name(name: Any = None) -> str:
    raw = str(name or "").strip()
    return raw if raw.startswith(_LATENT_REGIME_FEATURE_PREFIX) else _LATENT_REGIME_SIMILARITY_FEATURE


def _fit_latent_regime_model_from_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_names: Sequence[str],
    n_clusters: int,
    seed: int = 7,
    max_iter: int = 100,
) -> Optional[Dict[str, Any]]:
    compact_rows: List[Dict[str, Any]] = []
    for row in rows:
        for key in ("target_compact", "positive_compact", "negative_compact"):
            compact = dict(row.get(key) or {})
            if compact:
                compact_rows.append(compact)
    if not compact_rows:
        return None
    raw_matrix = raw_feature_matrix_from_compacts(compact_rows, feature_names=feature_names)
    if raw_matrix.ndim != 2 or raw_matrix.shape[0] == 0:
        return None
    return fit_latent_regime_kmeans(
        raw_matrix,
        feature_names=feature_names,
        n_clusters=int(n_clusters),
        seed=int(seed),
        max_iter=int(max_iter),
    )


def _latent_regime_advantage(
    rows: Sequence[Dict[str, Any]],
    *,
    model: Dict[str, Any],
) -> np.ndarray:
    feature_names = list(model.get("feature_names") or [])
    if not feature_names:
        return np.empty(len(rows), dtype=float)
    target_matrix = raw_feature_matrix_from_compacts(
        [dict(row.get("target_compact") or {}) for row in rows],
        feature_names=feature_names,
    )
    positive_matrix = raw_feature_matrix_from_compacts(
        [dict(row.get("positive_compact") or {}) for row in rows],
        feature_names=feature_names,
    )
    negative_matrix = raw_feature_matrix_from_compacts(
        [dict(row.get("negative_compact") or {}) for row in rows],
        feature_names=feature_names,
    )
    positive_similarity = latent_regime_similarity(target_matrix, positive_matrix, model)
    negative_similarity = latent_regime_similarity(target_matrix, negative_matrix, model)
    advantage = positive_similarity - negative_similarity
    invalid = ~np.isfinite(positive_similarity) | ~np.isfinite(negative_similarity)
    if bool(np.any(invalid)):
        advantage = np.array(advantage, dtype=float, copy=True)
        advantage[invalid] = np.nan
    return np.asarray(advantage, dtype=float)


def _fit_target_latent_regime_model_from_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_names: Sequence[str],
    n_clusters: int,
    seed: int = 7,
    max_iter: int = 100,
) -> Optional[Dict[str, Any]]:
    compact_rows = [dict(row.get("target_compact") or {}) for row in rows if dict(row.get("target_compact") or {})]
    if not compact_rows:
        return None
    raw_matrix = raw_feature_matrix_from_compacts(compact_rows, feature_names=feature_names)
    if raw_matrix.ndim != 2 or raw_matrix.shape[0] == 0:
        return None
    return fit_latent_regime_kmeans(
        raw_matrix,
        feature_names=feature_names,
        n_clusters=int(n_clusters),
        seed=int(seed),
        max_iter=int(max_iter),
    )


def _target_regime_memberships_for_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    model: Dict[str, Any],
) -> np.ndarray:
    feature_names = list(model.get("feature_names") or [])
    if not feature_names:
        return np.empty((len(rows), 0), dtype=float)
    target_matrix = raw_feature_matrix_from_compacts(
        [dict(row.get("target_compact") or {}) for row in rows],
        feature_names=feature_names,
    )
    return latent_regime_memberships(target_matrix, model)


def _fit_regime_conditioned_weights(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target_memberships: np.ndarray,
    prior: np.ndarray,
    sample_weights: np.ndarray,
    l2_lambda: float,
    learning_rate: float,
    max_iter: int,
    min_weights: np.ndarray,
) -> Dict[str, Any]:
    n_clusters = int(target_memberships.shape[1]) if target_memberships.ndim == 2 else 0
    if n_clusters <= 0:
        raise ValueError("target_memberships must have at least one cluster column")
    regime_weights: List[np.ndarray] = []
    regime_biases: List[float] = []
    for cluster_idx in range(n_clusters):
        cluster_row_weights = sample_weights * np.asarray(target_memberships[:, cluster_idx], dtype=float)
        fit = fit_nonnegative_pairwise_logistic(
            X,
            y,
            prior=prior,
            sample_weights=cluster_row_weights,
            l2_lambda=float(l2_lambda),
            learning_rate=float(learning_rate),
            max_iter=int(max_iter),
            min_weights=min_weights,
        )
        regime_weights.append(np.asarray(fit["weights"], dtype=float))
        regime_biases.append(float(fit["bias"]))
    return {
        "regime_weights": np.stack(regime_weights, axis=0),
        "regime_biases": np.asarray(regime_biases, dtype=float),
    }


def _evaluate_regime_conditioned_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target_memberships: np.ndarray,
    regime_weights: np.ndarray,
    regime_biases: np.ndarray,
    prior: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    blended_weights = np.asarray(target_memberships, dtype=float) @ np.asarray(regime_weights, dtype=float)
    blended_bias = np.sum(np.asarray(target_memberships, dtype=float) * np.asarray(regime_biases, dtype=float).reshape(1, -1), axis=1)
    logits = blended_bias + np.sum(X * blended_weights, axis=1)
    prob = _sigmoid(logits)
    prior_prob = _sigmoid(X @ prior)
    positive_mask = y == 1.0
    margin_values = logits[positive_mask] if int(np.count_nonzero(positive_mask)) else np.empty(0)
    prior_margin_values = (X[positive_mask] @ prior) if int(np.count_nonzero(positive_mask)) else np.empty(0)
    positive_weights = (
        np.asarray(sample_weights, dtype=float)[positive_mask]
        if sample_weights is not None and int(np.count_nonzero(positive_mask))
        else None
    )

    def _weighted_mean(values: np.ndarray, weights_arr: Optional[np.ndarray]) -> Optional[float]:
        if values.size == 0:
            return None
        if weights_arr is None:
            return float(np.mean(values))
        total = float(np.sum(weights_arr))
        if total <= 0.0:
            return float(np.mean(values))
        return float(np.sum(values * weights_arr) / total)

    return {
        "pair_accuracy": _pair_accuracy(y, prob, sample_weights=sample_weights),
        "pair_log_loss": _log_loss(y, prob, sample_weights=sample_weights),
        "pair_accuracy_prior": _pair_accuracy(y, prior_prob, sample_weights=sample_weights),
        "pair_log_loss_prior": _log_loss(y, prior_prob, sample_weights=sample_weights),
        "positive_margin_mean": _weighted_mean(margin_values, positive_weights),
        "positive_margin_mean_prior": _weighted_mean(prior_margin_values, positive_weights),
        "positive_margin_positive_rate": _weighted_mean((margin_values > 0.0).astype(float), positive_weights) if margin_values.size else None,
        "positive_margin_positive_rate_prior": _weighted_mean((prior_margin_values > 0.0).astype(float), positive_weights) if prior_margin_values.size else None,
        "n_rows": int(X.shape[0]),
        "n_positive_rows": int(np.count_nonzero(positive_mask)),
    }


def _cv_evaluation_sort_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(row.get("mean_log_loss_improvement") or 0.0),
        float(row.get("mean_accuracy_improvement") or 0.0),
        float(row.get("mean_positive_margin_improvement") or 0.0),
    )


def _feature_transform_candidates(
    rows: Sequence[Dict[str, Any]],
    feature_name: str,
    *,
    base_spec: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    values = _feature_value_sample(rows, feature_name)
    candidates: List[Dict[str, Any]] = []

    def _append(spec: Dict[str, Any]) -> None:
        normalized = _normalize_transform_spec(spec)
        if any(_transform_spec_key(existing) == _transform_spec_key(normalized) for existing in candidates):
            return
        candidates.append(normalized)

    _append(base_spec or {})
    if values.size == 0:
        return candidates

    abs_values = np.abs(values[np.isfinite(values)])
    if abs_values.size == 0:
        return candidates

    signed = bool(np.any(values < 0.0))
    quantile_values = {
        "q75": float(np.quantile(abs_values, 0.75)),
        "q90": float(np.quantile(abs_values, 0.90)),
        "q95": float(np.quantile(abs_values, 0.95)),
        "q99": float(np.quantile(abs_values, 0.99)),
    }
    quantile_values = {
        key: max(1e-6, value) for key, value in quantile_values.items() if np.isfinite(value) and value > 0.0
    }
    if not quantile_values:
        return candidates

    if signed:
        if "q95" in quantile_values:
            _append({"kind": "signed_log1p_cap", "cap": quantile_values["q95"]})
        if "q75" in quantile_values:
            _append({"kind": "signed_asinh", "scale": quantile_values["q75"]})
    else:
        if "q95" in quantile_values:
            _append({"kind": "log1p_cap", "cap": quantile_values["q95"]})
        if "q75" in quantile_values:
            _append({"kind": "signed_asinh", "scale": quantile_values["q75"]})
    return candidates


def _robust_standardize_vector(values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=float, copy=True)
    valid = np.isfinite(arr)
    if not bool(np.any(valid)):
        return np.zeros_like(arr)
    sample = arr[valid]
    med = float(np.median(sample))
    q25 = float(np.quantile(sample, 0.25))
    q75 = float(np.quantile(sample, 0.75))
    scale = (q75 - q25) / 1.349
    if (not np.isfinite(scale)) or scale <= 1e-9:
        scale = float(np.std(sample))
    if (not np.isfinite(scale)) or scale <= 1e-9:
        scale = 1.0
    out = np.zeros_like(arr)
    out[valid] = (arr[valid] - med) / scale
    out[~valid] = 0.0
    return out


def build_pairwise_matrix(
    df: pd.DataFrame,
    *,
    feature_names: Optional[Sequence[str]] = None,
    min_feature_coverage_rows: int = 20,
    transform_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    pair_weight_mode: str = "uniform",
    include_interactions: bool = False,
    interaction_feature_names: Optional[Sequence[str]] = None,
    penalty_feature_specs: Optional[Sequence[Dict[str, Any]]] = None,
    include_latent_regime: bool = False,
    latent_regime_model: Optional[Dict[str, Any]] = None,
    latent_feature_names: Optional[Sequence[str]] = None,
    enforce_feature_names: bool = False,
) -> Dict[str, Any]:
    rows = [dict(row) for row in df.to_dict(orient="records")]
    requested_feature_names = [str(name) for name in list(feature_names or _feature_names())]
    candidate_features = [
        name
        for name in requested_feature_names
        if not name.startswith(_INTERACTION_FEATURE_PREFIX) and not name.startswith(_LATENT_REGIME_FEATURE_PREFIX)
    ]
    transform_overrides = {
        str(feature): _normalize_transform_spec(spec)
        for feature, spec in dict(transform_specs or {}).items()
    }
    raw_advantages: Dict[str, List[Optional[float]]] = {
        feature: [
            (
                _feature_advantage_from_compacts(row, feature, transform_overrides.get(feature))
                if transform_overrides
                else _feature_advantage(row, feature)
            )
            for row in rows
        ]
        for feature in candidate_features
    }
    if enforce_feature_names:
        selected_features = list(candidate_features)
    else:
        selected_features = [
            feature
            for feature in candidate_features
            if sum(1 for value in raw_advantages[feature] if value is not None) >= int(min_feature_coverage_rows)
        ]
    base_selected_features = list(selected_features)

    X_pos_cols: List[np.ndarray] = []
    feature_coverage: Dict[str, int] = {}
    candidate_abs_diff_lookup: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    allowed_interaction_names = {
        str(name)
        for name in list(interaction_feature_names or [])
        if str(name or "").startswith(_INTERACTION_FEATURE_PREFIX)
    } or None
    allowed_penalty_names = {
        str(name)
        for name in list(requested_feature_names)
        if str(name or "").startswith(_PENALTY_FEATURE_PREFIX)
    } or None
    allowed_latent_names = {
        _latent_feature_name(name)
        for name in list(latent_feature_names or [])
        if str(name or "").startswith(_LATENT_REGIME_FEATURE_PREFIX)
    } or None
    for feature in selected_features:
        values = np.array(
            [np.nan if value is None else float(value) for value in raw_advantages[feature]],
            dtype=float,
        )
        feature_coverage[feature] = int(np.isfinite(values).sum())
        X_pos_cols.append(_robust_standardize_vector(values))
        if include_interactions:
            candidate_abs_diff_lookup[feature] = _candidate_abs_diff_triplets(
                rows,
                feature,
                transform_spec=transform_overrides.get(feature),
            )

    if include_interactions and len(selected_features) >= 2:
        for idx_left, feature_left in enumerate(base_selected_features):
            pos_left, neg_left = candidate_abs_diff_lookup[feature_left]
            for feature_right in base_selected_features[idx_left + 1 :]:
                pos_right, neg_right = candidate_abs_diff_lookup[feature_right]
                pos_term = pos_left * pos_right
                neg_term = neg_left * neg_right
                valid = np.isfinite(pos_term) & np.isfinite(neg_term)
                coverage = int(np.count_nonzero(valid))
                interaction_name = _interaction_feature_name(feature_left, feature_right)
                if coverage < int(min_feature_coverage_rows) and not (
                    enforce_feature_names
                    and allowed_interaction_names is not None
                    and interaction_name in allowed_interaction_names
                ):
                    continue
                if allowed_interaction_names is not None and interaction_name not in allowed_interaction_names:
                    continue
                advantage = np.full(pos_term.shape[0], np.nan, dtype=float)
                advantage[valid] = neg_term[valid] - pos_term[valid]
                feature_coverage[interaction_name] = coverage
                X_pos_cols.append(_robust_standardize_vector(advantage))
                selected_features.append(interaction_name)

    for spec in list(penalty_feature_specs or []):
        if not isinstance(spec, dict):
            continue
        penalty_name = _penalty_feature_name(str(spec.get("name") or ""))
        source_feature = str(spec.get("source_feature") or "")
        soft_threshold = _clean_numeric(spec.get("soft_threshold"))
        if not penalty_name or not source_feature or soft_threshold is None:
            continue
        if allowed_penalty_names is not None and penalty_name not in allowed_penalty_names:
            continue
        values = np.array(
            [
                _penalty_feature_advantage(
                    row,
                    source_feature=source_feature,
                    soft_threshold=float(soft_threshold),
                )
                for row in rows
            ],
            dtype=float,
        )
        coverage = int(np.count_nonzero(np.isfinite(values)))
        if coverage < int(min_feature_coverage_rows) and not (
            enforce_feature_names and allowed_penalty_names is not None and penalty_name in allowed_penalty_names
        ):
            continue
        feature_coverage[penalty_name] = coverage
        X_pos_cols.append(_robust_standardize_vector(values))
        selected_features.append(penalty_name)

    if include_latent_regime and isinstance(latent_regime_model, dict):
        latent_name = _LATENT_REGIME_SIMILARITY_FEATURE
        if allowed_latent_names is None or latent_name in allowed_latent_names:
            advantage = _latent_regime_advantage(rows, model=latent_regime_model)
            coverage = int(np.count_nonzero(np.isfinite(advantage)))
            if coverage >= int(min_feature_coverage_rows) or (
                enforce_feature_names
                and (allowed_latent_names is None or latent_name in allowed_latent_names)
            ):
                feature_coverage[latent_name] = coverage
                X_pos_cols.append(_robust_standardize_vector(advantage))
                selected_features.append(latent_name)

    if not X_pos_cols:
        raise ValueError("No features met minimum pairwise coverage threshold")
    X_pos = np.column_stack(X_pos_cols).astype(float)
    X_neg = -1.0 * X_pos
    X = np.vstack([X_pos, X_neg]).astype(float)
    y = np.concatenate(
        [
            np.ones(X_pos.shape[0], dtype=float),
            np.zeros(X_neg.shape[0], dtype=float),
        ]
    )
    group_rarity_weights = (
        _target_regime_rarity_weights(
            rows,
            feature_names=base_selected_features,
            transform_specs=transform_overrides,
        )
        if _normalize_pair_weight_mode(pair_weight_mode) == "target_regime_rarity"
        else {}
    )
    groups = [_pairwise_group_key(row) for row in rows]
    groups = groups + groups
    group_counts = Counter(str(group) for group in groups)
    row_weights = np.array([_pair_teacher_confidence_weight(row, pair_weight_mode) for row in rows], dtype=float)
    if group_rarity_weights:
        row_weights = np.asarray(
            [
                float(row_weights[idx]) * float(group_rarity_weights.get(_pairwise_group_key(row), 1.0))
                for idx, row in enumerate(rows)
            ],
            dtype=float,
        )
    if row_weights.size and not bool(np.any(row_weights > 0.0)):
        row_weights = np.ones_like(row_weights, dtype=float)
    duplicated_row_weights = np.concatenate([row_weights, row_weights]).astype(float)
    sample_weights = np.array(
        [
            duplicated_row_weights[idx] * (1.0 / max(1, int(group_counts[str(group)])))
            for idx, group in enumerate(groups)
        ],
        dtype=float,
    )
    if sample_weights.sum() > 0.0:
        sample_weights = sample_weights * (float(sample_weights.size) / float(sample_weights.sum()))
    return {
        "X": X,
        "y": y,
        "groups": np.array(groups, dtype=object),
        "sample_weights": sample_weights,
        "selected_features": tuple(selected_features),
        "feature_coverage": feature_coverage,
        "pair_count": int(len(rows)),
        "group_rarity_weights": dict(group_rarity_weights),
    }


def _compact_feature_vector(
    compact: Dict[str, Any],
    *,
    feature_names: Sequence[str],
) -> np.ndarray:
    return np.array(
        [
            float(compact.get(feature)) if compact.get(feature) is not None else np.nan
            for feature in feature_names
        ],
        dtype=float,
    )


def _parse_precedent_id(precedent_id: Any) -> Tuple[str, str]:
    parts = str(precedent_id or "").split("::")
    if len(parts) < 2:
        return "", ""
    return str(parts[0] or "").strip(), str(parts[1] or "").strip()


def _normalize_timestamp_key(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return ""
    return str(ts.tz_convert(None))


def _normalize_outcomes_lookup_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id_norm"] = out.get("company_id", pd.Series("", index=out.index)).astype(str).str.zfill(10)
    out["action_date_norm"] = out.get("action_date", pd.Series("", index=out.index)).apply(_normalize_timestamp_key)
    return out


def _build_outcome_feature_lookup(outcomes_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    lookup: Dict[str, pd.DataFrame] = {}
    normalized = _normalize_outcomes_lookup_df(outcomes_df)
    if "normalized_action_id" not in normalized.columns:
        return lookup
    for action_id, subset in normalized.groupby(normalized["normalized_action_id"].astype(str), dropna=False):
        feature_frame = _outcome_aware_reranker_feature_frame(subset).copy()
        feature_frame["company_id_norm"] = subset["company_id_norm"].astype(str).tolist()
        feature_frame["action_date_norm"] = subset["action_date_norm"].astype(str).tolist()
        lookup[str(action_id)] = feature_frame
    return lookup


def _lookup_outcome_feature_row(
    feature_lookup: Dict[str, pd.DataFrame],
    *,
    action_id: str,
    precedent_id: Any,
) -> Dict[str, float]:
    company_id, action_date = _parse_precedent_id(precedent_id)
    if not company_id or not action_date:
        return {}
    frame = feature_lookup.get(str(action_id or ""))
    if frame is None or frame.empty:
        return {}
    company_key = str(company_id).zfill(10)
    action_time_key = _normalize_timestamp_key(action_date)
    mask = frame["company_id_norm"].astype(str).eq(company_key)
    mask &= frame["action_date_norm"].astype(str).eq(action_time_key)
    matches = frame.loc[mask]
    if matches.empty:
        return {}
    row = matches.iloc[0]
    return {
        feature_name: float(row.get(feature_name))
        for feature_name in _outcome_aware_reranker_feature_names()
        if pd.notna(row.get(feature_name))
    }


def _outcome_aware_reranker_prior(feature_names: Sequence[str]) -> np.ndarray:
    prior = np.zeros(len(feature_names), dtype=float)
    for idx, feature_name in enumerate(feature_names):
        if str(feature_name) == "current_similarity_score":
            prior[idx] = 1.0
    return prior


def _second_stage_reranker_prior(feature_names: Sequence[str]) -> np.ndarray:
    prior = np.zeros(len(feature_names), dtype=float)
    for idx, feature_name in enumerate(feature_names):
        if str(feature_name) == "base_state_similarity":
            prior[idx] = 1.0
    return prior


def build_second_stage_reranker_matrix(
    df: pd.DataFrame,
    *,
    pair_weight_mode: str = "uniform",
    same_action_only: bool = True,
) -> Dict[str, Any]:
    rows = [dict(row) for row in df.to_dict(orient="records")]
    feature_names = list(_second_stage_reranker_feature_names())
    action_feature_names = list(_STATE_VECTOR_MATCHING_COLS)
    X_pos_rows: List[np.ndarray] = []
    groups: List[str] = []
    selected_rows: List[Dict[str, Any]] = []
    feature_coverage = {feature: 0 for feature in feature_names}

    for row in rows:
        anchor_action_id = str(row.get("anchor_action_id") or "")
        competitor_action_id = str(row.get("competitor_action_id") or "")
        if same_action_only and anchor_action_id and competitor_action_id and competitor_action_id != anchor_action_id:
            continue
        target_compact = dict(row.get("target_compact") or {})
        positive_compact = dict(row.get("positive_compact") or {})
        negative_compact = dict(row.get("negative_compact") or {})
        if not target_compact or not positive_compact or not negative_compact:
            continue
        target_vec = _compact_feature_vector(target_compact, feature_names=action_feature_names)
        candidate_matrix = np.vstack(
            [
                _compact_feature_vector(positive_compact, feature_names=action_feature_names),
                _compact_feature_vector(negative_compact, feature_names=action_feature_names),
            ]
        )
        feature_payload = _second_stage_reranker_feature_matrix(
            emb_raw=candidate_matrix,
            candidate_vec_raw=target_vec,
            embedding_cols=action_feature_names,
            action_id=anchor_action_id,
            action_subtype=str(row.get("anchor_action_subtype") or anchor_action_id),
            profile_version="weighted_distance_v2",
            target_action_scale=_clean_numeric(row.get("target_action_scale")),
            row_action_scales=np.asarray(
                [
                    _clean_numeric(row.get("positive_action_scale")),
                    _clean_numeric(row.get("negative_action_scale")),
                ],
                dtype=float,
            ),
            feature_overrides={
                "parameter_similarity": np.asarray([np.nan, np.nan], dtype=float),
                "sector_similarity": np.asarray([np.nan, np.nan], dtype=float),
                "action_match_score": np.asarray(
                    [
                        0.92 if anchor_action_id and anchor_action_id == competitor_action_id else 0.65,
                        0.92 if anchor_action_id and anchor_action_id == competitor_action_id else 0.65,
                    ],
                    dtype=float,
                ),
            },
        )
        matrix = np.asarray(feature_payload.get("matrix"), dtype=float)
        if matrix.ndim != 2 or matrix.shape != (2, len(feature_names)):
            continue
        feature_idx = {name: idx for idx, name in enumerate(feature_names)}
        target_sector = str(row.get("target_sector") or "").strip()
        target_subsector = str(row.get("target_subsector") or "").strip()
        positive_sector = str(row.get("positive_sector") or "").strip()
        positive_subsector = str(row.get("positive_subsector") or "").strip()
        negative_sector = str(row.get("negative_sector") or "").strip()
        negative_subsector = str(row.get("negative_subsector") or "").strip()
        if "sector_similarity" in feature_idx and target_sector:
            matrix[0, feature_idx["sector_similarity"]] = float(
                _sector_similarity(target_sector, positive_sector, target_subsector, positive_subsector)
            )
            matrix[1, feature_idx["sector_similarity"]] = float(
                _sector_similarity(target_sector, negative_sector, target_subsector, negative_subsector)
            )
        if "parameter_similarity" in feature_idx:
            target_scale = pd.to_numeric(row.get("target_action_scale"), errors="coerce")
            positive_scale = pd.to_numeric(row.get("positive_action_scale"), errors="coerce")
            negative_scale = pd.to_numeric(row.get("negative_action_scale"), errors="coerce")
            eps = 1e-6
            if pd.notna(target_scale) and pd.notna(positive_scale):
                matrix[0, feature_idx["parameter_similarity"]] = float(
                    np.exp(-abs(np.log((float(target_scale) + eps) / (float(positive_scale) + eps))))
                )
            if pd.notna(target_scale) and pd.notna(negative_scale):
                matrix[1, feature_idx["parameter_similarity"]] = float(
                    np.exp(-abs(np.log((float(target_scale) + eps) / (float(negative_scale) + eps))))
                )
        for feature_idx, feature_name in enumerate(feature_names):
            if np.isfinite(matrix[:, feature_idx]).all():
                feature_coverage[feature_name] += 1
        X_pos_rows.append(matrix[0] - matrix[1])
        groups.append(_pairwise_group_key(row))
        selected_rows.append(row)

    if not X_pos_rows:
        raise ValueError("No rows available for second-stage reranker learning")

    X_pos_raw = np.vstack(X_pos_rows).astype(float)
    X_pos = np.column_stack(
        [
            _robust_standardize_vector(X_pos_raw[:, feature_idx])
            for feature_idx in range(X_pos_raw.shape[1])
        ]
    ).astype(float)
    X_neg = -1.0 * X_pos
    X = np.vstack([X_pos, X_neg]).astype(float)
    y = np.concatenate(
        [
            np.ones(X_pos.shape[0], dtype=float),
            np.zeros(X_neg.shape[0], dtype=float),
        ]
    )
    group_rarity_weights = (
        _target_regime_rarity_weights(
            selected_rows,
            feature_names=list(_feature_names()),
            transform_specs={},
        )
        if _normalize_pair_weight_mode(pair_weight_mode) == "target_regime_rarity"
        else {}
    )
    duplicated_groups = groups + groups
    group_counts = Counter(str(group) for group in duplicated_groups)
    row_weights = np.array(
        [_pair_teacher_confidence_weight(row, pair_weight_mode) for row in selected_rows],
        dtype=float,
    )
    if group_rarity_weights:
        row_weights = np.asarray(
            [
                float(row_weights[idx]) * float(group_rarity_weights.get(_pairwise_group_key(row), 1.0))
                for idx, row in enumerate(selected_rows)
            ],
            dtype=float,
        )
    if row_weights.size and not bool(np.any(row_weights > 0.0)):
        row_weights = np.ones_like(row_weights, dtype=float)
    duplicated_row_weights = np.concatenate([row_weights, row_weights]).astype(float)
    sample_weights = np.array(
        [
            duplicated_row_weights[idx] * (1.0 / max(1, int(group_counts[str(group)])))
            for idx, group in enumerate(duplicated_groups)
        ],
        dtype=float,
    )
    if sample_weights.sum() > 0.0:
        sample_weights = sample_weights * (float(sample_weights.size) / float(sample_weights.sum()))
    return {
        "X": X,
        "y": y,
        "groups": np.array(duplicated_groups, dtype=object),
        "sample_weights": sample_weights,
        "selected_features": tuple(feature_names),
        "feature_coverage": feature_coverage,
        "pair_count": int(len(selected_rows)),
        "group_rarity_weights": dict(group_rarity_weights),
    }


def cross_validate_second_stage_reranker(
    dataset_path: str | Path,
    *,
    pair_weight_mode: str = "uniform",
    same_action_only: bool = True,
    l2_grid: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0),
    learning_rate: float = 0.05,
    max_iter: int = 4000,
) -> Dict[str, Any]:
    df = load_pairwise_supervision(dataset_path)
    matrix = build_second_stage_reranker_matrix(
        df,
        pair_weight_mode=pair_weight_mode,
        same_action_only=same_action_only,
    )
    X = np.asarray(matrix["X"], dtype=float)
    y = np.asarray(matrix["y"], dtype=float)
    groups = np.asarray(matrix["groups"], dtype=object)
    sample_weights = np.asarray(matrix["sample_weights"], dtype=float)
    selected_features = list(matrix["selected_features"])
    prior = _second_stage_reranker_prior(selected_features)
    min_weights = np.zeros(len(selected_features), dtype=float)
    unique_groups = np.array(sorted({str(group) for group in groups.tolist()}), dtype=object)
    if unique_groups.size < 2:
        raise ValueError("Need at least two distinct group cases for second-stage reranker cross-validation")

    evaluations: List[Dict[str, Any]] = []
    for l2_lambda in [float(value) for value in l2_grid]:
        fold_metrics: List[Dict[str, Any]] = []
        fold_weights: List[np.ndarray] = []
        fold_biases: List[float] = []
        for held_out_group in unique_groups.tolist():
            holdout_mask = np.array([str(group) == held_out_group for group in groups.tolist()], dtype=bool)
            train_mask = ~holdout_mask
            fit = fit_nonnegative_pairwise_logistic(
                X[train_mask],
                y[train_mask],
                prior=prior,
                sample_weights=sample_weights[train_mask],
                l2_lambda=l2_lambda,
                learning_rate=float(learning_rate),
                max_iter=int(max_iter),
                min_weights=min_weights,
            )
            fold_weights.append(np.asarray(fit["weights"], dtype=float))
            fold_biases.append(float(fit["bias"]))
            metrics = _evaluate_fit(
                X[holdout_mask],
                y[holdout_mask],
                weights=np.asarray(fit["weights"], dtype=float),
                bias=float(fit["bias"]),
                prior=prior,
                sample_weights=sample_weights[holdout_mask],
            )
            metrics["held_out_group"] = str(held_out_group)
            fold_metrics.append(metrics)

        mean_weights = np.mean(np.stack(fold_weights), axis=0)
        mean_bias = float(np.mean(np.asarray(fold_biases, dtype=float))) if fold_biases else 0.0
        avg_log_loss = float(np.mean([float(item["pair_log_loss"]) for item in fold_metrics]))
        avg_prior_log_loss = float(np.mean([float(item["pair_log_loss_prior"]) for item in fold_metrics]))
        avg_accuracy = float(np.mean([float(item["pair_accuracy"]) for item in fold_metrics]))
        avg_prior_accuracy = float(np.mean([float(item["pair_accuracy_prior"]) for item in fold_metrics]))
        avg_margin = float(
            np.mean(
                [
                    float(item["positive_margin_mean"])
                    for item in fold_metrics
                    if item.get("positive_margin_mean") is not None
                ]
            )
        )
        avg_prior_margin = float(
            np.mean(
                [
                    float(item["positive_margin_mean_prior"])
                    for item in fold_metrics
                    if item.get("positive_margin_mean_prior") is not None
                ]
            )
        )
        evaluations.append(
            {
                "l2_lambda": l2_lambda,
                "fold_count": len(fold_metrics),
                "mean_pair_accuracy": avg_accuracy,
                "mean_pair_accuracy_prior": avg_prior_accuracy,
                "mean_pair_log_loss": avg_log_loss,
                "mean_pair_log_loss_prior": avg_prior_log_loss,
                "mean_positive_margin": avg_margin,
                "mean_positive_margin_prior": avg_prior_margin,
                "mean_log_loss_improvement": avg_prior_log_loss - avg_log_loss,
                "mean_accuracy_improvement": avg_accuracy - avg_prior_accuracy,
                "mean_positive_margin_improvement": avg_margin - avg_prior_margin,
                "mean_weights": {
                    feature: float(mean_weights[idx]) for idx, feature in enumerate(selected_features)
                },
                "mean_bias": mean_bias,
                "fold_metrics": fold_metrics,
            }
        )
    best_evaluation = sorted(evaluations, key=_cv_evaluation_sort_key, reverse=True)[0]
    return {
        "dataset_path": str(dataset_path),
        "selected_features": selected_features,
        "feature_coverage": matrix["feature_coverage"],
        "pair_count": int(matrix["pair_count"]),
        "group_count": int(unique_groups.size),
        "pair_weight_mode": _normalize_pair_weight_mode(pair_weight_mode),
        "same_action_only": bool(same_action_only),
        "evaluations": evaluations,
        "best_evaluation": best_evaluation,
    }


def build_outcome_aware_reranker_matrix(
    df: pd.DataFrame,
    *,
    outcomes_df: pd.DataFrame,
    pair_weight_mode: str = "uniform",
    same_action_only: bool = True,
) -> Dict[str, Any]:
    rows = [dict(row) for row in df.to_dict(orient="records")]
    feature_names = list(_outcome_aware_reranker_feature_names())
    X_pos_rows: List[np.ndarray] = []
    groups: List[str] = []
    selected_rows: List[Dict[str, Any]] = []
    feature_coverage = {feature: 0 for feature in feature_names}
    outcome_lookup = _build_outcome_feature_lookup(outcomes_df)

    for row in rows:
        anchor_action_id = str(row.get("anchor_action_id") or "")
        competitor_action_id = str(row.get("competitor_action_id") or "")
        if same_action_only and anchor_action_id and competitor_action_id and competitor_action_id != anchor_action_id:
            continue
        positive_features = _lookup_outcome_feature_row(
            outcome_lookup,
            action_id=anchor_action_id,
            precedent_id=row.get("positive_precedent_id"),
        )
        negative_features = _lookup_outcome_feature_row(
            outcome_lookup,
            action_id=anchor_action_id,
            precedent_id=row.get("negative_precedent_id"),
        )
        positive_vector: List[float] = []
        negative_vector: List[float] = []
        for feature_name in feature_names:
            if feature_name == "current_similarity_score":
                pos_value = _clean_numeric(row.get("positive_similarity_score"))
                neg_value = _clean_numeric(row.get("negative_similarity_score"))
            else:
                pos_value = _clean_numeric(positive_features.get(feature_name))
                neg_value = _clean_numeric(negative_features.get(feature_name))
            if pos_value is not None and neg_value is not None:
                feature_coverage[feature_name] += 1
            positive_vector.append(
                float(pos_value)
                if pos_value is not None
                else (0.0 if feature_name == "outcome_support_score" else 0.5)
            )
            negative_vector.append(
                float(neg_value)
                if neg_value is not None
                else (0.0 if feature_name == "outcome_support_score" else 0.5)
            )
        X_pos_rows.append(np.asarray(positive_vector, dtype=float) - np.asarray(negative_vector, dtype=float))
        groups.append(_pairwise_group_key(row))
        selected_rows.append(row)

    if not X_pos_rows:
        raise ValueError("No rows available for outcome-aware reranker learning")

    X_pos_raw = np.vstack(X_pos_rows).astype(float)
    X_pos = np.column_stack(
        [
            _robust_standardize_vector(X_pos_raw[:, feature_idx])
            for feature_idx in range(X_pos_raw.shape[1])
        ]
    ).astype(float)
    X_neg = -1.0 * X_pos
    X = np.vstack([X_pos, X_neg]).astype(float)
    y = np.concatenate(
        [
            np.ones(X_pos.shape[0], dtype=float),
            np.zeros(X_neg.shape[0], dtype=float),
        ]
    )
    group_rarity_weights = (
        _target_regime_rarity_weights(
            selected_rows,
            feature_names=list(_feature_names()),
            transform_specs={},
        )
        if _normalize_pair_weight_mode(pair_weight_mode) == "target_regime_rarity"
        else {}
    )
    duplicated_groups = groups + groups
    group_counts = Counter(str(group) for group in duplicated_groups)
    row_weights = np.array(
        [_pair_teacher_confidence_weight(row, pair_weight_mode) for row in selected_rows],
        dtype=float,
    )
    if group_rarity_weights:
        row_weights = np.asarray(
            [
                float(row_weights[idx]) * float(group_rarity_weights.get(_pairwise_group_key(row), 1.0))
                for idx, row in enumerate(selected_rows)
            ],
            dtype=float,
        )
    if row_weights.size and not bool(np.any(row_weights > 0.0)):
        row_weights = np.ones_like(row_weights, dtype=float)
    duplicated_row_weights = np.concatenate([row_weights, row_weights]).astype(float)
    sample_weights = np.array(
        [
            duplicated_row_weights[idx] * (1.0 / max(1, int(group_counts[str(group)])))
            for idx, group in enumerate(duplicated_groups)
        ],
        dtype=float,
    )
    if sample_weights.sum() > 0.0:
        sample_weights = sample_weights * (float(sample_weights.size) / float(sample_weights.sum()))
    return {
        "X": X,
        "y": y,
        "groups": np.array(duplicated_groups, dtype=object),
        "sample_weights": sample_weights,
        "selected_features": tuple(feature_names),
        "feature_coverage": feature_coverage,
        "pair_count": int(len(selected_rows)),
        "group_rarity_weights": dict(group_rarity_weights),
    }


def cross_validate_outcome_aware_reranker(
    dataset_path: str | Path,
    *,
    outcomes_path: str | Path,
    pair_weight_mode: str = "uniform",
    same_action_only: bool = True,
    l2_grid: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0),
    learning_rate: float = 0.05,
    max_iter: int = 4000,
) -> Dict[str, Any]:
    df = load_pairwise_supervision(dataset_path)
    outcomes_df = pd.read_parquet(outcomes_path)
    matrix = build_outcome_aware_reranker_matrix(
        df,
        outcomes_df=outcomes_df,
        pair_weight_mode=pair_weight_mode,
        same_action_only=same_action_only,
    )
    X = np.asarray(matrix["X"], dtype=float)
    y = np.asarray(matrix["y"], dtype=float)
    groups = np.asarray(matrix["groups"], dtype=object)
    sample_weights = np.asarray(matrix["sample_weights"], dtype=float)
    selected_features = list(matrix["selected_features"])
    prior = np.ones(len(selected_features), dtype=float)
    min_weights = np.zeros(len(selected_features), dtype=float)
    unique_groups = np.array(sorted({str(group) for group in groups.tolist()}), dtype=object)
    if unique_groups.size < 2:
        raise ValueError("Need at least two distinct group cases for outcome-aware reranker cross-validation")

    evaluations: List[Dict[str, Any]] = []
    for l2_lambda in [float(value) for value in l2_grid]:
        fold_metrics: List[Dict[str, Any]] = []
        fold_weights: List[np.ndarray] = []
        fold_biases: List[float] = []
        for held_out_group in unique_groups.tolist():
            holdout_mask = np.array([str(group) == held_out_group for group in groups.tolist()], dtype=bool)
            train_mask = ~holdout_mask
            fit = fit_nonnegative_pairwise_logistic(
                X[train_mask],
                y[train_mask],
                prior=prior,
                sample_weights=sample_weights[train_mask],
                l2_lambda=l2_lambda,
                learning_rate=float(learning_rate),
                max_iter=int(max_iter),
                min_weights=min_weights,
            )
            fold_weights.append(np.asarray(fit["weights"], dtype=float))
            fold_biases.append(float(fit["bias"]))
            metrics = _evaluate_fit(
                X[holdout_mask],
                y[holdout_mask],
                weights=np.asarray(fit["weights"], dtype=float),
                bias=float(fit["bias"]),
                prior=prior,
                sample_weights=sample_weights[holdout_mask],
            )
            metrics["held_out_group"] = str(held_out_group)
            fold_metrics.append(metrics)

        mean_weights = np.mean(np.stack(fold_weights), axis=0)
        mean_bias = float(np.mean(np.asarray(fold_biases, dtype=float))) if fold_biases else 0.0
        avg_log_loss = float(np.mean([float(item["pair_log_loss"]) for item in fold_metrics]))
        avg_prior_log_loss = float(np.mean([float(item["pair_log_loss_prior"]) for item in fold_metrics]))
        avg_accuracy = float(np.mean([float(item["pair_accuracy"]) for item in fold_metrics]))
        avg_prior_accuracy = float(np.mean([float(item["pair_accuracy_prior"]) for item in fold_metrics]))
        avg_margin = float(
            np.mean(
                [
                    float(item["positive_margin_mean"])
                    for item in fold_metrics
                    if item.get("positive_margin_mean") is not None
                ]
            )
        )
        avg_prior_margin = float(
            np.mean(
                [
                    float(item["positive_margin_mean_prior"])
                    for item in fold_metrics
                    if item.get("positive_margin_mean_prior") is not None
                ]
            )
        )
        evaluations.append(
            {
                "l2_lambda": l2_lambda,
                "fold_count": len(fold_metrics),
                "mean_pair_accuracy": avg_accuracy,
                "mean_pair_accuracy_prior": avg_prior_accuracy,
                "mean_pair_log_loss": avg_log_loss,
                "mean_pair_log_loss_prior": avg_prior_log_loss,
                "mean_positive_margin": avg_margin,
                "mean_positive_margin_prior": avg_prior_margin,
                "mean_log_loss_improvement": avg_prior_log_loss - avg_log_loss,
                "mean_accuracy_improvement": avg_accuracy - avg_prior_accuracy,
                "mean_positive_margin_improvement": avg_margin - avg_prior_margin,
                "mean_weights": {
                    feature: float(mean_weights[idx]) for idx, feature in enumerate(selected_features)
                },
                "mean_bias": mean_bias,
                "fold_metrics": fold_metrics,
            }
        )
    best_evaluation = sorted(evaluations, key=_cv_evaluation_sort_key, reverse=True)[0]
    return {
        "dataset_path": str(dataset_path),
        "outcomes_path": str(outcomes_path),
        "selected_features": selected_features,
        "feature_coverage": matrix["feature_coverage"],
        "pair_count": int(matrix["pair_count"]),
        "group_count": int(unique_groups.size),
        "pair_weight_mode": _normalize_pair_weight_mode(pair_weight_mode),
        "same_action_only": bool(same_action_only),
        "evaluations": evaluations,
        "best_evaluation": best_evaluation,
    }


def learn_feature_transforms_from_pairwise_supervision(
    dataset_path: str | Path,
    *,
    scope_key: str,
    base_payload_path: str | Path,
    feature_names: Optional[Sequence[str]] = None,
    min_feature_coverage_rows: int = 20,
    l2_grid: Sequence[float] = (0.25, 1.0, 4.0),
    learning_rate: float = 0.05,
    max_iter: int = 2000,
    feature_transform_mode: Optional[str] = None,
    pair_weight_mode: str = "uniform",
    penalty_feature_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    df = load_pairwise_supervision(dataset_path)
    rows = [dict(row) for row in df.to_dict(orient="records")]
    candidate_features = list(feature_names or _feature_names())
    default_specs = _feature_transform_prior(
        base_payload_path,
        scope_key=scope_key,
        feature_names=candidate_features,
        feature_transform_mode=feature_transform_mode,
    )
    base_matrix = build_pairwise_matrix(
        df,
        feature_names=candidate_features,
        min_feature_coverage_rows=min_feature_coverage_rows,
        transform_specs=default_specs,
        pair_weight_mode=pair_weight_mode,
        penalty_feature_specs=penalty_feature_specs,
    )
    selected_features = list(base_matrix["selected_features"])
    chosen_specs = {feature: dict(default_specs.get(feature, {})) for feature in selected_features}
    search_results: Dict[str, Any] = {}

    def _sort_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
        return (
            -float(row.get("mean_pair_log_loss") or 0.0),
            float(row.get("mean_pair_accuracy") or 0.0),
            float(row.get("mean_positive_margin") or 0.0),
        )

    for feature in selected_features:
        candidates = _feature_transform_candidates(
            rows,
            feature,
            base_spec=chosen_specs.get(feature, default_specs.get(feature, {})),
        )
        best_spec = dict(chosen_specs.get(feature, {}))
        best_eval: Optional[Dict[str, Any]] = None
        candidate_evals: List[Dict[str, Any]] = []
        for candidate_spec in candidates:
            candidate_transform_specs = {
                name: dict(spec)
                for name, spec in chosen_specs.items()
            }
            candidate_transform_specs[feature] = dict(candidate_spec)
            cv = cross_validate_pairwise_precedent_quality_weights(
                dataset_path,
                scope_key=scope_key,
                base_payload_path=base_payload_path,
                feature_names=selected_features,
                min_feature_coverage_rows=min_feature_coverage_rows,
                l2_grid=l2_grid,
                learning_rate=learning_rate,
                max_iter=max_iter,
                transform_specs=candidate_transform_specs,
                pair_weight_mode=pair_weight_mode,
                penalty_feature_specs=penalty_feature_specs,
            )
            evaluation = dict(cv["best_evaluation"] or {})
            evaluation["transform_spec"] = dict(candidate_spec)
            candidate_evals.append(evaluation)
            if best_eval is None or _sort_key(evaluation) > _sort_key(best_eval):
                best_eval = evaluation
                best_spec = dict(candidate_spec)
        search_results[feature] = {
            "base_spec": dict(default_specs.get(feature, {})),
            "chosen_spec": dict(best_spec),
            "changed": _transform_spec_key(best_spec) != _transform_spec_key(default_specs.get(feature, {})),
            "candidate_evaluations": candidate_evals,
        }
        chosen_specs[feature] = dict(best_spec)

    return {
        "scope_key": _clean_scope_key(scope_key),
        "dataset_path": str(dataset_path),
        "selected_features": selected_features,
        "default_feature_transforms": {feature: dict(default_specs.get(feature, {})) for feature in selected_features},
        "chosen_feature_transforms": {feature: dict(chosen_specs.get(feature, {})) for feature in selected_features},
        "search_results": search_results,
    }


def _sigmoid(z: np.ndarray) -> np.ndarray:
    clipped = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _log_loss(y_true: np.ndarray, prob: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
    p = np.clip(np.asarray(prob, dtype=float), 1e-9, 1.0 - 1e-9)
    y = np.asarray(y_true, dtype=float)
    losses = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    if sample_weights is None:
        return float(np.mean(losses))
    weights = np.asarray(sample_weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(losses))
    return float(np.sum(losses * weights) / total)


def _pair_accuracy(y_true: np.ndarray, prob: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
    pred = (np.asarray(prob, dtype=float) >= 0.5).astype(float)
    y = np.asarray(y_true, dtype=float)
    correct = (pred == y).astype(float)
    if sample_weights is None:
        return float(np.mean(correct))
    weights = np.asarray(sample_weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(correct))
    return float(np.sum(correct * weights) / total)


def _split_groups(groups: np.ndarray, holdout_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    unique_groups = np.array(sorted({str(group) for group in groups.tolist()}), dtype=object)
    if unique_groups.size < 2:
        mask = np.ones(groups.shape[0], dtype=bool)
        return mask, ~mask
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    holdout_n = max(1, int(round(unique_groups.size * float(holdout_frac))))
    holdout_n = min(holdout_n, max(1, unique_groups.size - 1))
    holdout_groups = set(shuffled[:holdout_n].tolist())
    holdout_mask = np.array([str(group) in holdout_groups for group in groups.tolist()], dtype=bool)
    train_mask = ~holdout_mask
    return train_mask, holdout_mask


def fit_nonnegative_pairwise_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    prior: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    l2_lambda: float = 1.0,
    learning_rate: float = 0.05,
    max_iter: int = 4000,
    min_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    weights = np.array(prior, dtype=float, copy=True)
    weights = np.clip(weights, 0.0, None)
    min_weight_arr = (
        np.asarray(min_weights, dtype=float)
        if min_weights is not None
        else np.full(weights.shape[0], 0.25, dtype=float)
    )
    min_weight_arr = np.clip(min_weight_arr, 0.0, None)
    bias = 0.0
    row_weights = np.asarray(sample_weights, dtype=float) if sample_weights is not None else np.ones(X.shape[0], dtype=float)
    total_weight = float(np.sum(row_weights))
    if total_weight <= 0.0:
        row_weights = np.ones(X.shape[0], dtype=float)
        total_weight = float(X.shape[0])
    normalized_weights = row_weights / total_weight
    prev_objective: Optional[float] = None
    stalled_checks = 0
    check_interval = 50
    objective_tol = 1e-7
    max_stalled_checks = 5
    for step in range(max(1, int(max_iter))):
        logits = bias + X @ weights
        prob = _sigmoid(logits)
        error = prob - y
        weighted_error = normalized_weights * error
        grad_w = X.T @ weighted_error + float(l2_lambda) * (weights - prior)
        grad_b = float(np.sum(weighted_error))
        step_lr = float(learning_rate) / (1.0 + 0.0015 * float(step))
        weights = np.clip(weights - step_lr * grad_w, 0.0, None)
        bias -= step_lr * grad_b
        if ((step + 1) % check_interval) == 0:
            clipped_prob = np.clip(prob, 1e-9, 1.0 - 1e-9)
            weighted_log_loss = -float(
                np.sum(
                    normalized_weights
                    * (
                        y * np.log(clipped_prob)
                        + (1.0 - y) * np.log(1.0 - clipped_prob)
                    )
                )
            )
            regularization = 0.5 * float(l2_lambda) * float(np.sum(np.square(weights - prior)))
            objective = weighted_log_loss + regularization
            if prev_objective is not None and (prev_objective - objective) <= objective_tol:
                stalled_checks += 1
                if stalled_checks >= max_stalled_checks:
                    break
            else:
                stalled_checks = 0
            prev_objective = objective
    positive = weights[weights > 0.0]
    if positive.size:
        weights = weights / float(np.mean(positive))
    else:
        weights = np.array(prior, dtype=float, copy=True)
    max_weight_arr = np.full(weights.shape[0], 4.0, dtype=float)
    weights = np.minimum(np.maximum(weights, min_weight_arr), max_weight_arr)
    return {"weights": weights, "bias": float(bias)}


def _evaluate_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    weights: np.ndarray,
    bias: float,
    prior: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    logits = bias + X @ weights
    prob = _sigmoid(logits)
    prior_prob = _sigmoid(X @ prior)
    positive_mask = y == 1.0
    margin_values = (X[positive_mask] @ weights) if int(np.count_nonzero(positive_mask)) else np.empty(0)
    prior_margin_values = (X[positive_mask] @ prior) if int(np.count_nonzero(positive_mask)) else np.empty(0)
    positive_weights = (
        np.asarray(sample_weights, dtype=float)[positive_mask]
        if sample_weights is not None and int(np.count_nonzero(positive_mask))
        else None
    )
    def _weighted_mean(values: np.ndarray, weights_arr: Optional[np.ndarray]) -> Optional[float]:
        if values.size == 0:
            return None
        if weights_arr is None:
            return float(np.mean(values))
        total = float(np.sum(weights_arr))
        if total <= 0.0:
            return float(np.mean(values))
        return float(np.sum(values * weights_arr) / total)
    return {
        "pair_accuracy": _pair_accuracy(y, prob, sample_weights=sample_weights),
        "pair_log_loss": _log_loss(y, prob, sample_weights=sample_weights),
        "pair_accuracy_prior": _pair_accuracy(y, prior_prob, sample_weights=sample_weights),
        "pair_log_loss_prior": _log_loss(y, prior_prob, sample_weights=sample_weights),
        "positive_margin_mean": _weighted_mean(margin_values, positive_weights),
        "positive_margin_mean_prior": _weighted_mean(prior_margin_values, positive_weights),
        "positive_margin_positive_rate": _weighted_mean((margin_values > 0.0).astype(float), positive_weights) if margin_values.size else None,
        "positive_margin_positive_rate_prior": _weighted_mean((prior_margin_values > 0.0).astype(float), positive_weights) if prior_margin_values.size else None,
        "n_rows": int(X.shape[0]),
        "n_positive_rows": int(np.count_nonzero(positive_mask)),
    }


def cross_validate_pairwise_precedent_quality_weights(
    dataset_path: str | Path,
    *,
    scope_key: str,
    base_payload_path: str | Path,
    feature_names: Optional[Sequence[str]] = None,
    min_feature_coverage_rows: int = 20,
    l2_grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    learning_rate: float = 0.05,
    max_iter: int = 4000,
    transform_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    pair_weight_mode: str = "uniform",
    include_interactions: bool = False,
    interaction_feature_names: Optional[Sequence[str]] = None,
    penalty_feature_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    df = load_pairwise_supervision(dataset_path)
    matrix = build_pairwise_matrix(
        df,
        feature_names=feature_names,
        min_feature_coverage_rows=min_feature_coverage_rows,
        transform_specs=transform_specs,
        pair_weight_mode=pair_weight_mode,
        include_interactions=include_interactions,
        interaction_feature_names=interaction_feature_names,
        penalty_feature_specs=penalty_feature_specs,
    )
    X = np.asarray(matrix["X"], dtype=float)
    y = np.asarray(matrix["y"], dtype=float)
    groups = np.asarray(matrix["groups"], dtype=object)
    sample_weights = np.asarray(matrix["sample_weights"], dtype=float)
    selected_features = list(matrix["selected_features"])
    prior = np.zeros(len(selected_features), dtype=float)
    base_feature_names = [
        feature
        for feature in selected_features
        if not str(feature).startswith(_INTERACTION_FEATURE_PREFIX)
        and not str(feature).startswith(_PENALTY_FEATURE_PREFIX)
    ]
    if base_feature_names:
        base_prior = load_feature_weight_prior(
            base_payload_path,
            scope_key=scope_key,
            feature_names=base_feature_names,
        )
        base_prior_map = {feature: float(base_prior[idx]) for idx, feature in enumerate(base_feature_names)}
        for idx, feature in enumerate(selected_features):
            if feature in base_prior_map:
                prior[idx] = base_prior_map[feature]
    min_weights = load_feature_weight_floor(feature_names=selected_features)
    unique_groups = np.array(sorted({str(group) for group in groups.tolist()}), dtype=object)
    if unique_groups.size < 2:
        raise ValueError("Need at least two distinct group cases for cross-validation")

    evaluations: List[Dict[str, Any]] = []
    for l2_lambda in [float(value) for value in l2_grid]:
        fold_metrics: List[Dict[str, Any]] = []
        fold_weights: List[np.ndarray] = []
        for held_out_group in unique_groups.tolist():
            holdout_mask = np.array([str(group) == held_out_group for group in groups.tolist()], dtype=bool)
            train_mask = ~holdout_mask
            fit = fit_nonnegative_pairwise_logistic(
                X[train_mask],
                y[train_mask],
                prior=prior,
                sample_weights=sample_weights[train_mask],
                l2_lambda=l2_lambda,
                learning_rate=float(learning_rate),
                max_iter=int(max_iter),
                min_weights=min_weights,
            )
            fold_weights.append(np.asarray(fit["weights"], dtype=float))
            metrics = _evaluate_fit(
                X[holdout_mask],
                y[holdout_mask],
                weights=np.asarray(fit["weights"], dtype=float),
                bias=float(fit["bias"]),
                prior=prior,
                sample_weights=sample_weights[holdout_mask],
            )
            metrics["held_out_group"] = str(held_out_group)
            fold_metrics.append(metrics)

        mean_weights = np.mean(np.stack(fold_weights), axis=0)
        avg_log_loss = float(np.mean([float(item["pair_log_loss"]) for item in fold_metrics]))
        avg_prior_log_loss = float(np.mean([float(item["pair_log_loss_prior"]) for item in fold_metrics]))
        avg_accuracy = float(np.mean([float(item["pair_accuracy"]) for item in fold_metrics]))
        avg_prior_accuracy = float(np.mean([float(item["pair_accuracy_prior"]) for item in fold_metrics]))
        avg_margin = float(
            np.mean(
                [
                    float(item["positive_margin_mean"])
                    for item in fold_metrics
                    if item.get("positive_margin_mean") is not None
                ]
            )
        )
        avg_prior_margin = float(
            np.mean(
                [
                    float(item["positive_margin_mean_prior"])
                    for item in fold_metrics
                    if item.get("positive_margin_mean_prior") is not None
                ]
            )
        )
        evaluations.append(
            {
                "l2_lambda": l2_lambda,
                "fold_count": len(fold_metrics),
                "mean_pair_accuracy": avg_accuracy,
                "mean_pair_accuracy_prior": avg_prior_accuracy,
                "mean_pair_log_loss": avg_log_loss,
                "mean_pair_log_loss_prior": avg_prior_log_loss,
                "mean_positive_margin": avg_margin,
                "mean_positive_margin_prior": avg_prior_margin,
                "mean_log_loss_improvement": avg_prior_log_loss - avg_log_loss,
                "mean_accuracy_improvement": avg_accuracy - avg_prior_accuracy,
                "mean_positive_margin_improvement": avg_margin - avg_prior_margin,
                "mean_weights": {
                    feature: float(mean_weights[idx]) for idx, feature in enumerate(selected_features)
                },
                "fold_metrics": fold_metrics,
            }
        )

    best_evaluation = sorted(evaluations, key=_cv_evaluation_sort_key, reverse=True)[0]
    return {
        "scope_key": _clean_scope_key(scope_key),
        "dataset_path": str(dataset_path),
        "selected_features": selected_features,
        "feature_coverage": matrix["feature_coverage"],
        "pair_count": int(matrix["pair_count"]),
        "group_count": int(unique_groups.size),
        "evaluations": evaluations,
        "best_evaluation": best_evaluation,
        "feature_transforms": {
            feature: dict(_normalize_transform_spec(dict(transform_specs or {}).get(feature)))
            for feature in selected_features
        },
        "pair_weight_mode": _normalize_pair_weight_mode(pair_weight_mode),
    }


def learn_pairwise_precedent_quality_weights(
    dataset_path: str | Path,
    *,
    scope_key: str,
    base_payload_path: str | Path,
    feature_names: Optional[Sequence[str]] = None,
    min_feature_coverage_rows: int = 20,
    holdout_frac: float = 0.33,
    seed: int = 7,
    l2_lambda: float = 1.0,
    learning_rate: float = 0.05,
    max_iter: int = 4000,
    transform_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    pair_weight_mode: str = "uniform",
    include_interactions: bool = False,
    interaction_feature_names: Optional[Sequence[str]] = None,
    penalty_feature_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    df = load_pairwise_supervision(dataset_path)
    matrix = build_pairwise_matrix(
        df,
        feature_names=feature_names,
        min_feature_coverage_rows=min_feature_coverage_rows,
        transform_specs=transform_specs,
        pair_weight_mode=pair_weight_mode,
        include_interactions=include_interactions,
        interaction_feature_names=interaction_feature_names,
        penalty_feature_specs=penalty_feature_specs,
    )
    X = np.asarray(matrix["X"], dtype=float)
    y = np.asarray(matrix["y"], dtype=float)
    groups = np.asarray(matrix["groups"], dtype=object)
    sample_weights = np.asarray(matrix["sample_weights"], dtype=float)
    selected_features = list(matrix["selected_features"])
    prior = np.zeros(len(selected_features), dtype=float)
    base_feature_names = [
        feature
        for feature in selected_features
        if not str(feature).startswith(_INTERACTION_FEATURE_PREFIX)
        and not str(feature).startswith(_PENALTY_FEATURE_PREFIX)
    ]
    if base_feature_names:
        base_prior = load_feature_weight_prior(
            base_payload_path,
            scope_key=scope_key,
            feature_names=base_feature_names,
        )
        base_prior_map = {feature: float(base_prior[idx]) for idx, feature in enumerate(base_feature_names)}
        for idx, feature in enumerate(selected_features):
            if feature in base_prior_map:
                prior[idx] = base_prior_map[feature]
    min_weights = load_feature_weight_floor(feature_names=selected_features)
    train_mask, holdout_mask = _split_groups(groups, holdout_frac=float(holdout_frac), seed=int(seed))
    fit = fit_nonnegative_pairwise_logistic(
        X[train_mask],
        y[train_mask],
        prior=prior,
        sample_weights=sample_weights[train_mask],
        l2_lambda=float(l2_lambda),
        learning_rate=float(learning_rate),
        max_iter=int(max_iter),
        min_weights=min_weights,
    )
    weights = np.asarray(fit["weights"], dtype=float)
    bias = float(fit["bias"])

    def _metrics(mask: np.ndarray) -> Dict[str, Any]:
        if int(np.count_nonzero(mask)) == 0:
            return {}
        return _evaluate_fit(
            X[mask],
            y[mask],
            weights=weights,
            bias=bias,
            prior=prior,
            sample_weights=sample_weights[mask],
        )

    train_metrics = _metrics(train_mask)
    holdout_metrics = _metrics(holdout_mask)
    feature_signal: Dict[str, Any] = {}
    positive_rows = df.to_dict(orient="records")
    for idx, feature in enumerate(selected_features):
        raw_advantage = np.array(
            [np.nan if _feature_advantage(row, feature) is None else float(_feature_advantage(row, feature)) for row in positive_rows],
            dtype=float,
        )
        valid = np.isfinite(raw_advantage)
        feature_signal[feature] = {
            "coverage_rows": int(np.count_nonzero(valid)),
            "positive_advantage_rate": float(np.mean(raw_advantage[valid] > 0.0)) if bool(np.any(valid)) else None,
            "mean_advantage": float(np.nanmean(raw_advantage[valid])) if bool(np.any(valid)) else None,
            "learned_weight": float(weights[idx]),
            "prior_weight": float(prior[idx]),
        }

    return {
        "scope_key": _clean_scope_key(scope_key),
        "dataset_path": str(dataset_path),
        "selected_features": selected_features,
        "feature_coverage": matrix["feature_coverage"],
        "pair_count": int(matrix["pair_count"]),
        "train_group_count": int(len({str(g) for g in groups[train_mask].tolist()})),
        "holdout_group_count": int(len({str(g) for g in groups[holdout_mask].tolist()})),
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "feature_signal": feature_signal,
        "weights": {feature: float(weights[idx]) for idx, feature in enumerate(selected_features)},
        "prior_weights": {feature: float(prior[idx]) for idx, feature in enumerate(selected_features)},
        "bias": bias,
        "learning_rate": float(learning_rate),
        "l2_lambda": float(l2_lambda),
        "max_iter": int(max_iter),
        "seed": int(seed),
        "feature_transforms": {
            feature: dict(_normalize_transform_spec(dict(transform_specs or {}).get(feature)))
            for feature in selected_features
        },
        "pair_weight_mode": _normalize_pair_weight_mode(pair_weight_mode),
    }


def search_pairwise_interactions_from_supervision(
    dataset_path: str | Path,
    *,
    scope_key: str,
    base_payload_path: str | Path,
    feature_names: Optional[Sequence[str]] = None,
    min_feature_coverage_rows: int = 20,
    l2_grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    learning_rate: float = 0.05,
    max_iter: int = 4000,
    transform_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    pair_weight_mode: str = "uniform",
    max_interaction_terms: int = 6,
    penalty_feature_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    df = load_pairwise_supervision(dataset_path)
    base_cv = cross_validate_pairwise_precedent_quality_weights(
        dataset_path,
        scope_key=scope_key,
        base_payload_path=base_payload_path,
        feature_names=feature_names,
        min_feature_coverage_rows=min_feature_coverage_rows,
        l2_grid=l2_grid,
        learning_rate=float(learning_rate),
        max_iter=int(max_iter),
        transform_specs=transform_specs,
        pair_weight_mode=pair_weight_mode,
        include_interactions=False,
        penalty_feature_specs=penalty_feature_specs,
    )
    base_selected_features = list(base_cv["selected_features"])
    interaction_matrix = build_pairwise_matrix(
        df,
        feature_names=base_selected_features,
        min_feature_coverage_rows=min_feature_coverage_rows,
        transform_specs=transform_specs,
        pair_weight_mode=pair_weight_mode,
        include_interactions=True,
        penalty_feature_specs=penalty_feature_specs,
    )
    candidate_interactions = [
        str(feature)
        for feature in interaction_matrix["selected_features"]
        if str(feature).startswith(_INTERACTION_FEATURE_PREFIX)
    ]
    current_cv = base_cv
    current_selected_interactions: List[str] = []
    remaining_interactions = list(candidate_interactions)
    search_steps: List[Dict[str, Any]] = []
    max_terms = max(0, int(max_interaction_terms))
    for _ in range(min(max_terms, len(remaining_interactions))):
        best_candidate_name: Optional[str] = None
        best_candidate_cv: Optional[Dict[str, Any]] = None
        best_candidate_sort_key: Optional[Tuple[float, float, float]] = None
        for candidate_name in remaining_interactions:
            candidate_cv = cross_validate_pairwise_precedent_quality_weights(
                dataset_path,
                scope_key=scope_key,
                base_payload_path=base_payload_path,
                feature_names=base_selected_features,
                min_feature_coverage_rows=min_feature_coverage_rows,
                l2_grid=l2_grid,
                learning_rate=float(learning_rate),
                max_iter=int(max_iter),
                transform_specs=transform_specs,
                pair_weight_mode=pair_weight_mode,
                include_interactions=True,
                interaction_feature_names=current_selected_interactions + [candidate_name],
                penalty_feature_specs=penalty_feature_specs,
            )
            candidate_sort_key = _cv_evaluation_sort_key(candidate_cv["best_evaluation"])
            if best_candidate_sort_key is None or candidate_sort_key > best_candidate_sort_key:
                best_candidate_name = candidate_name
                best_candidate_cv = candidate_cv
                best_candidate_sort_key = candidate_sort_key
        if best_candidate_name is None or best_candidate_cv is None or best_candidate_sort_key is None:
            break
        current_sort_key = _cv_evaluation_sort_key(current_cv["best_evaluation"])
        if best_candidate_sort_key <= current_sort_key:
            break
        current_eval = dict(current_cv["best_evaluation"] or {})
        best_eval = dict(best_candidate_cv["best_evaluation"] or {})
        current_selected_interactions.append(best_candidate_name)
        remaining_interactions = [
            name for name in remaining_interactions if name != best_candidate_name
        ]
        current_cv = best_candidate_cv
        search_steps.append(
            {
                "added_interaction": best_candidate_name,
                "selected_interactions": list(current_selected_interactions),
                "best_evaluation": best_eval,
                "incremental_log_loss_improvement": float(best_eval.get("mean_log_loss_improvement") or 0.0)
                - float(current_eval.get("mean_log_loss_improvement") or 0.0),
                "incremental_accuracy_improvement": float(best_eval.get("mean_accuracy_improvement") or 0.0)
                - float(current_eval.get("mean_accuracy_improvement") or 0.0),
                "incremental_positive_margin_improvement": float(best_eval.get("mean_positive_margin_improvement") or 0.0)
                - float(current_eval.get("mean_positive_margin_improvement") or 0.0),
            }
        )
    return {
        "scope_key": _clean_scope_key(scope_key),
        "dataset_path": str(dataset_path),
        "base_selected_features": base_selected_features,
        "candidate_interaction_count": int(len(candidate_interactions)),
        "chosen_interactions": list(current_selected_interactions),
        "base_evaluation": dict(base_cv["best_evaluation"] or {}),
        "best_evaluation": dict(current_cv["best_evaluation"] or {}),
        "steps": search_steps,
    }


def search_latent_regime_models_from_supervision(
    dataset_path: str | Path,
    *,
    scope_key: str,
    base_payload_path: str | Path,
    feature_names: Optional[Sequence[str]] = None,
    min_feature_coverage_rows: int = 20,
    l2_grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    learning_rate: float = 0.05,
    max_iter: int = 4000,
    transform_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    pair_weight_mode: str = "uniform",
    include_interactions: bool = False,
    interaction_feature_names: Optional[Sequence[str]] = None,
    penalty_feature_specs: Optional[Sequence[Dict[str, Any]]] = None,
    n_cluster_grid: Sequence[int] = (2, 3, 4, 5, 6),
    seed: int = 7,
    latent_max_iter: int = 100,
) -> Dict[str, Any]:
    df = load_pairwise_supervision(dataset_path)
    rows = [dict(row) for row in df.to_dict(orient="records")]
    if not rows:
        raise ValueError("pairwise supervision dataset is empty")

    base_matrix = build_pairwise_matrix(
        df,
        feature_names=feature_names,
        min_feature_coverage_rows=min_feature_coverage_rows,
        transform_specs=transform_specs,
        pair_weight_mode=pair_weight_mode,
        include_interactions=include_interactions,
        interaction_feature_names=interaction_feature_names,
        penalty_feature_specs=penalty_feature_specs,
    )
    base_selected_features = list(base_matrix["selected_features"])
    base_feature_names = [
        feature
        for feature in base_selected_features
        if not str(feature).startswith(_INTERACTION_FEATURE_PREFIX)
        and not str(feature).startswith(_PENALTY_FEATURE_PREFIX)
        and not str(feature).startswith(_LATENT_REGIME_FEATURE_PREFIX)
    ]
    all_selected_features = list(base_selected_features) + [_LATENT_REGIME_SIMILARITY_FEATURE]

    base_prior = load_feature_weight_prior(
        base_payload_path,
        scope_key=scope_key,
        feature_names=base_feature_names,
    )
    base_prior_map = {feature: float(base_prior[idx]) for idx, feature in enumerate(base_feature_names)}
    prior = np.zeros(len(all_selected_features), dtype=float)
    for idx, feature in enumerate(all_selected_features):
        if feature in base_prior_map:
            prior[idx] = base_prior_map[feature]
    min_weights = load_feature_weight_floor(feature_names=all_selected_features)

    unique_groups = np.array(sorted({_pairwise_group_key(row) for row in rows}), dtype=object)
    if unique_groups.size < 2:
        raise ValueError("Need at least two distinct group cases for latent-regime cross-validation")

    evaluations: List[Dict[str, Any]] = []
    full_models: Dict[int, Dict[str, Any]] = {}
    cluster_values = sorted({max(1, int(value)) for value in list(n_cluster_grid or [])})
    for n_clusters in cluster_values:
        fold_metrics: List[Dict[str, Any]] = []
        fold_weights: List[np.ndarray] = []
        for held_out_group in unique_groups.tolist():
            train_rows = [row for row in rows if _pairwise_group_key(row) != held_out_group]
            latent_model = _fit_latent_regime_model_from_rows(
                train_rows,
                feature_names=base_feature_names,
                n_clusters=int(n_clusters),
                seed=int(seed),
                max_iter=int(latent_max_iter),
            )
            if latent_model is None:
                continue
            matrix = build_pairwise_matrix(
                df,
                feature_names=base_selected_features,
                min_feature_coverage_rows=min_feature_coverage_rows,
                transform_specs=transform_specs,
                pair_weight_mode=pair_weight_mode,
                include_interactions=include_interactions,
                interaction_feature_names=interaction_feature_names,
                penalty_feature_specs=penalty_feature_specs,
                include_latent_regime=True,
                latent_regime_model=latent_model,
                latent_feature_names=[_LATENT_REGIME_SIMILARITY_FEATURE],
                enforce_feature_names=True,
            )
            selected_features = list(matrix["selected_features"])
            if selected_features != all_selected_features:
                raise ValueError("latent regime CV selected_features drifted across folds")
            X = np.asarray(matrix["X"], dtype=float)
            y = np.asarray(matrix["y"], dtype=float)
            groups = np.asarray(matrix["groups"], dtype=object)
            sample_weights = np.asarray(matrix["sample_weights"], dtype=float)
            holdout_mask = np.array([str(group) == held_out_group for group in groups.tolist()], dtype=bool)
            train_mask = ~holdout_mask
            fit = fit_nonnegative_pairwise_logistic(
                X[train_mask],
                y[train_mask],
                prior=prior,
                sample_weights=sample_weights[train_mask],
                l2_lambda=float(1.0),
                learning_rate=float(learning_rate),
                max_iter=int(max_iter),
                min_weights=min_weights,
            )
            best_fit = fit
            best_eval = _evaluate_fit(
                X[holdout_mask],
                y[holdout_mask],
                weights=np.asarray(fit["weights"], dtype=float),
                bias=float(fit["bias"]),
                prior=prior,
                sample_weights=sample_weights[holdout_mask],
            )
            best_l2 = 1.0
            best_sort_key = (
                float(best_eval.get("pair_log_loss_prior") or 0.0) - float(best_eval.get("pair_log_loss") or 0.0),
                float(best_eval.get("pair_accuracy") or 0.0) - float(best_eval.get("pair_accuracy_prior") or 0.0),
                float(best_eval.get("positive_margin_mean") or 0.0) - float(best_eval.get("positive_margin_mean_prior") or 0.0),
            )
            for l2_lambda in [float(value) for value in l2_grid]:
                if abs(l2_lambda - 1.0) < 1e-12:
                    continue
                fit = fit_nonnegative_pairwise_logistic(
                    X[train_mask],
                    y[train_mask],
                    prior=prior,
                    sample_weights=sample_weights[train_mask],
                    l2_lambda=l2_lambda,
                    learning_rate=float(learning_rate),
                    max_iter=int(max_iter),
                    min_weights=min_weights,
                )
                metrics = _evaluate_fit(
                    X[holdout_mask],
                    y[holdout_mask],
                    weights=np.asarray(fit["weights"], dtype=float),
                    bias=float(fit["bias"]),
                    prior=prior,
                    sample_weights=sample_weights[holdout_mask],
                )
                candidate_sort_key = (
                    float(metrics.get("pair_log_loss_prior") or 0.0) - float(metrics.get("pair_log_loss") or 0.0),
                    float(metrics.get("pair_accuracy") or 0.0) - float(metrics.get("pair_accuracy_prior") or 0.0),
                    float(metrics.get("positive_margin_mean") or 0.0) - float(metrics.get("positive_margin_mean_prior") or 0.0),
                )
                if candidate_sort_key > best_sort_key:
                    best_fit = fit
                    best_eval = metrics
                    best_l2 = l2_lambda
                    best_sort_key = candidate_sort_key
            fold_weights.append(np.asarray(best_fit["weights"], dtype=float))
            best_eval["held_out_group"] = str(held_out_group)
            best_eval["l2_lambda"] = float(best_l2)
            fold_metrics.append(best_eval)

        if not fold_metrics:
            continue
        mean_weights = np.mean(np.stack(fold_weights), axis=0)
        avg_log_loss = float(np.mean([float(item["pair_log_loss"]) for item in fold_metrics]))
        avg_prior_log_loss = float(np.mean([float(item["pair_log_loss_prior"]) for item in fold_metrics]))
        avg_accuracy = float(np.mean([float(item["pair_accuracy"]) for item in fold_metrics]))
        avg_prior_accuracy = float(np.mean([float(item["pair_accuracy_prior"]) for item in fold_metrics]))
        avg_margin = float(
            np.mean(
                [
                    float(item["positive_margin_mean"])
                    for item in fold_metrics
                    if item.get("positive_margin_mean") is not None
                ]
            )
        )
        avg_prior_margin = float(
            np.mean(
                [
                    float(item["positive_margin_mean_prior"])
                    for item in fold_metrics
                    if item.get("positive_margin_mean_prior") is not None
                ]
            )
        )
        chosen_l2 = float(
            np.median([float(item.get("l2_lambda") or 1.0) for item in fold_metrics])
        )
        evaluations.append(
            {
                "latent_regime_n_clusters": int(n_clusters),
                "l2_lambda": chosen_l2,
                "fold_count": len(fold_metrics),
                "mean_pair_accuracy": avg_accuracy,
                "mean_pair_accuracy_prior": avg_prior_accuracy,
                "mean_pair_log_loss": avg_log_loss,
                "mean_pair_log_loss_prior": avg_prior_log_loss,
                "mean_positive_margin": avg_margin,
                "mean_positive_margin_prior": avg_prior_margin,
                "mean_log_loss_improvement": avg_prior_log_loss - avg_log_loss,
                "mean_accuracy_improvement": avg_accuracy - avg_prior_accuracy,
                "mean_positive_margin_improvement": avg_margin - avg_prior_margin,
                "mean_weights": {
                    feature: float(mean_weights[idx]) for idx, feature in enumerate(all_selected_features)
                },
                "fold_metrics": fold_metrics,
            }
        )
        full_model = _fit_latent_regime_model_from_rows(
            rows,
            feature_names=base_feature_names,
            n_clusters=int(n_clusters),
            seed=int(seed),
            max_iter=int(latent_max_iter),
        )
        if full_model is not None:
            full_models[int(n_clusters)] = full_model

    if not evaluations:
        raise ValueError("No latent regime evaluations produced")

    best_evaluation = sorted(evaluations, key=_cv_evaluation_sort_key, reverse=True)[0]
    chosen_clusters = int(best_evaluation["latent_regime_n_clusters"])
    chosen_model = full_models.get(chosen_clusters)
    if chosen_model is None:
        raise ValueError("latent regime search did not produce a full model for the chosen cluster count")
    full_matrix = build_pairwise_matrix(
        df,
        feature_names=base_selected_features,
        min_feature_coverage_rows=min_feature_coverage_rows,
        transform_specs=transform_specs,
        pair_weight_mode=pair_weight_mode,
        include_interactions=include_interactions,
        interaction_feature_names=interaction_feature_names,
        penalty_feature_specs=penalty_feature_specs,
        include_latent_regime=True,
        latent_regime_model=chosen_model,
        latent_feature_names=[_LATENT_REGIME_SIMILARITY_FEATURE],
        enforce_feature_names=True,
    )
    return {
        "scope_key": _clean_scope_key(scope_key),
        "dataset_path": str(dataset_path),
        "selected_features": list(full_matrix["selected_features"]),
        "feature_coverage": dict(full_matrix["feature_coverage"]),
        "pair_count": int(full_matrix["pair_count"]),
        "group_count": int(unique_groups.size),
        "evaluations": evaluations,
        "best_evaluation": best_evaluation,
        "chosen_latent_regime_n_clusters": chosen_clusters,
        "chosen_latent_feature_name": _LATENT_REGIME_SIMILARITY_FEATURE,
        "chosen_latent_regime_model": chosen_model,
        "base_selected_features": base_selected_features,
        "latent_regime_seed": int(seed),
        "latent_regime_max_iter": int(latent_max_iter),
    }


def search_target_regime_mixture_from_supervision(
    dataset_path: str | Path,
    *,
    scope_key: str,
    base_payload_path: str | Path,
    feature_names: Optional[Sequence[str]] = None,
    min_feature_coverage_rows: int = 20,
    l2_grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    learning_rate: float = 0.05,
    max_iter: int = 4000,
    transform_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    pair_weight_mode: str = "uniform",
    include_interactions: bool = False,
    interaction_feature_names: Optional[Sequence[str]] = None,
    penalty_feature_specs: Optional[Sequence[Dict[str, Any]]] = None,
    n_cluster_grid: Sequence[int] = (2, 3, 4, 5, 6),
    seed: int = 7,
    latent_max_iter: int = 100,
) -> Dict[str, Any]:
    df = load_pairwise_supervision(dataset_path)
    rows = [dict(row) for row in df.to_dict(orient="records")]
    if not rows:
        raise ValueError("pairwise supervision dataset is empty")

    active_transform_specs = None if transform_specs is None else dict(transform_specs)
    if active_transform_specs is None:
        active_transform_specs = load_feature_transform_prior(
            base_payload_path,
            scope_key=scope_key,
            feature_names=feature_names,
        )

    base_matrix = build_pairwise_matrix(
        df,
        feature_names=feature_names,
        min_feature_coverage_rows=min_feature_coverage_rows,
        transform_specs=active_transform_specs,
        pair_weight_mode=pair_weight_mode,
        include_interactions=include_interactions,
        interaction_feature_names=interaction_feature_names,
        penalty_feature_specs=penalty_feature_specs,
    )
    selected_features = list(base_matrix["selected_features"])
    base_feature_names = [
        feature
        for feature in selected_features
        if not str(feature).startswith(_INTERACTION_FEATURE_PREFIX)
        and not str(feature).startswith(_PENALTY_FEATURE_PREFIX)
        and not str(feature).startswith(_LATENT_REGIME_FEATURE_PREFIX)
    ]
    unique_groups = np.array(sorted({_pairwise_group_key(row) for row in rows}), dtype=object)
    if unique_groups.size < 2:
        raise ValueError("Need at least two distinct group cases for target-regime cross-validation")

    evaluations: List[Dict[str, Any]] = []
    full_models: Dict[int, Dict[str, Any]] = {}
    full_fit_lookup: Dict[int, Dict[str, Any]] = {}
    full_matrix_lookup: Dict[int, Dict[str, Any]] = {}
    cluster_values = sorted({max(1, int(value)) for value in list(n_cluster_grid or [])})
    for n_clusters in cluster_values:
        fold_metrics: List[Dict[str, Any]] = []
        regime_weight_means: List[np.ndarray] = []
        regime_bias_means: List[np.ndarray] = []
        fold_selected_features: Optional[List[str]] = None
        for held_out_group in unique_groups.tolist():
            train_rows = [row for row in rows if _pairwise_group_key(row) != held_out_group]
            latent_model = _fit_target_latent_regime_model_from_rows(
                train_rows,
                feature_names=base_feature_names,
                n_clusters=int(n_clusters),
                seed=int(seed),
                max_iter=int(latent_max_iter),
            )
            if latent_model is None:
                continue
            fold_matrix = build_pairwise_matrix(
                df,
                feature_names=base_feature_names,
                min_feature_coverage_rows=min_feature_coverage_rows,
                transform_specs=active_transform_specs,
                pair_weight_mode=pair_weight_mode,
                include_interactions=include_interactions,
                interaction_feature_names=interaction_feature_names,
                penalty_feature_specs=penalty_feature_specs,
                include_latent_regime=True,
                latent_regime_model=latent_model,
                latent_feature_names=[_LATENT_REGIME_SIMILARITY_FEATURE],
                enforce_feature_names=True,
            )
            fold_selected_features = list(fold_matrix["selected_features"])
            X = np.asarray(fold_matrix["X"], dtype=float)
            y = np.asarray(fold_matrix["y"], dtype=float)
            groups = np.asarray(fold_matrix["groups"], dtype=object)
            sample_weights = np.asarray(fold_matrix["sample_weights"], dtype=float)
            base_prior = load_feature_weight_prior(
                base_payload_path,
                scope_key=scope_key,
                feature_names=fold_selected_features,
            )
            min_weights = load_feature_weight_floor(feature_names=fold_selected_features)
            row_memberships = _target_regime_memberships_for_rows(rows, model=latent_model)
            duplicated_memberships = np.vstack([row_memberships, row_memberships])
            holdout_mask = np.array([str(group) == held_out_group for group in groups.tolist()], dtype=bool)
            train_mask = ~holdout_mask
            best_fit: Optional[Dict[str, Any]] = None
            best_eval: Optional[Dict[str, Any]] = None
            best_l2 = None
            best_sort_key: Optional[Tuple[float, float, float]] = None
            for l2_lambda in [float(value) for value in l2_grid]:
                global_fit = fit_nonnegative_pairwise_logistic(
                    X[train_mask],
                    y[train_mask],
                    prior=base_prior,
                    sample_weights=sample_weights[train_mask],
                    l2_lambda=l2_lambda,
                    learning_rate=float(learning_rate),
                    max_iter=int(max_iter),
                    min_weights=min_weights,
                )
                fit = _fit_regime_conditioned_weights(
                    X[train_mask],
                    y[train_mask],
                    target_memberships=duplicated_memberships[train_mask],
                    prior=np.asarray(global_fit["weights"], dtype=float),
                    sample_weights=sample_weights[train_mask],
                    l2_lambda=l2_lambda,
                    learning_rate=float(learning_rate),
                    max_iter=int(max_iter),
                    min_weights=min_weights,
                )
                metrics = _evaluate_regime_conditioned_fit(
                    X[holdout_mask],
                    y[holdout_mask],
                    target_memberships=duplicated_memberships[holdout_mask],
                    regime_weights=np.asarray(fit["regime_weights"], dtype=float),
                    regime_biases=np.asarray(fit["regime_biases"], dtype=float),
                    prior=np.asarray(global_fit["weights"], dtype=float),
                    sample_weights=sample_weights[holdout_mask],
                )
                candidate_sort_key = (
                    float(metrics.get("pair_log_loss_prior") or 0.0) - float(metrics.get("pair_log_loss") or 0.0),
                    float(metrics.get("pair_accuracy") or 0.0) - float(metrics.get("pair_accuracy_prior") or 0.0),
                    float(metrics.get("positive_margin_mean") or 0.0) - float(metrics.get("positive_margin_mean_prior") or 0.0),
                )
                if best_sort_key is None or candidate_sort_key > best_sort_key:
                    best_fit = fit
                    best_eval = metrics
                    best_l2 = l2_lambda
                    best_sort_key = candidate_sort_key
            if best_fit is None or best_eval is None or best_l2 is None:
                continue
            regime_weight_means.append(np.asarray(best_fit["regime_weights"], dtype=float))
            regime_bias_means.append(np.asarray(best_fit["regime_biases"], dtype=float))
            best_eval["held_out_group"] = str(held_out_group)
            best_eval["l2_lambda"] = float(best_l2)
            fold_metrics.append(best_eval)

        if not fold_metrics:
            continue
        avg_log_loss = float(np.mean([float(item["pair_log_loss"]) for item in fold_metrics]))
        avg_prior_log_loss = float(np.mean([float(item["pair_log_loss_prior"]) for item in fold_metrics]))
        avg_accuracy = float(np.mean([float(item["pair_accuracy"]) for item in fold_metrics]))
        avg_prior_accuracy = float(np.mean([float(item["pair_accuracy_prior"]) for item in fold_metrics]))
        avg_margin = float(
            np.mean(
                [
                    float(item["positive_margin_mean"])
                    for item in fold_metrics
                    if item.get("positive_margin_mean") is not None
                ]
            )
        )
        avg_prior_margin = float(
            np.mean(
                [
                    float(item["positive_margin_mean_prior"])
                    for item in fold_metrics
                    if item.get("positive_margin_mean_prior") is not None
                ]
            )
        )
        chosen_l2 = float(np.median([float(item.get("l2_lambda") or 1.0) for item in fold_metrics]))
        mean_regime_weights = np.mean(np.stack(regime_weight_means, axis=0), axis=0)
        mean_regime_biases = np.mean(np.stack(regime_bias_means, axis=0), axis=0)
        evaluations.append(
            {
                "target_regime_n_clusters": int(n_clusters),
                "l2_lambda": chosen_l2,
                "fold_count": len(fold_metrics),
                "mean_pair_accuracy": avg_accuracy,
                "mean_pair_accuracy_prior": avg_prior_accuracy,
                "mean_pair_log_loss": avg_log_loss,
                "mean_pair_log_loss_prior": avg_prior_log_loss,
                "mean_positive_margin": avg_margin,
                "mean_positive_margin_prior": avg_prior_margin,
                "mean_log_loss_improvement": avg_prior_log_loss - avg_log_loss,
                "mean_accuracy_improvement": avg_accuracy - avg_prior_accuracy,
                "mean_positive_margin_improvement": avg_margin - avg_prior_margin,
                "mean_regime_feature_weights": [
                    {
                        "cluster": int(cluster_idx),
                        "weights": {
                            feature: float(mean_regime_weights[cluster_idx, idx])
                            for idx, feature in enumerate(fold_selected_features or [])
                        },
                        "bias": float(mean_regime_biases[cluster_idx]),
                    }
                    for cluster_idx in range(mean_regime_weights.shape[0])
                ],
                "fold_metrics": fold_metrics,
            }
        )
        full_model = _fit_target_latent_regime_model_from_rows(
            rows,
            feature_names=base_feature_names,
            n_clusters=int(n_clusters),
            seed=int(seed),
            max_iter=int(latent_max_iter),
        )
        if full_model is None:
            continue
        full_matrix = build_pairwise_matrix(
            df,
            feature_names=base_feature_names,
            min_feature_coverage_rows=min_feature_coverage_rows,
            transform_specs=active_transform_specs,
            pair_weight_mode=pair_weight_mode,
            include_interactions=include_interactions,
            interaction_feature_names=interaction_feature_names,
            penalty_feature_specs=penalty_feature_specs,
            include_latent_regime=True,
            latent_regime_model=full_model,
            latent_feature_names=[_LATENT_REGIME_SIMILARITY_FEATURE],
            enforce_feature_names=True,
        )
        X = np.asarray(full_matrix["X"], dtype=float)
        y = np.asarray(full_matrix["y"], dtype=float)
        sample_weights = np.asarray(full_matrix["sample_weights"], dtype=float)
        base_prior = load_feature_weight_prior(
            base_payload_path,
            scope_key=scope_key,
            feature_names=list(full_matrix["selected_features"]),
        )
        min_weights = load_feature_weight_floor(feature_names=list(full_matrix["selected_features"]))
        full_memberships = _target_regime_memberships_for_rows(rows, model=full_model)
        duplicated_full_memberships = np.vstack([full_memberships, full_memberships])
        global_fit = fit_nonnegative_pairwise_logistic(
            X,
            y,
            prior=base_prior,
            sample_weights=sample_weights,
            l2_lambda=chosen_l2,
            learning_rate=float(learning_rate),
            max_iter=int(max_iter),
            min_weights=min_weights,
        )
        full_fit = _fit_regime_conditioned_weights(
            X,
            y,
            target_memberships=duplicated_full_memberships,
            prior=np.asarray(global_fit["weights"], dtype=float),
            sample_weights=sample_weights,
            l2_lambda=chosen_l2,
            learning_rate=float(learning_rate),
            max_iter=int(max_iter),
            min_weights=min_weights,
        )
        full_models[int(n_clusters)] = full_model
        full_fit_lookup[int(n_clusters)] = full_fit
        full_matrix_lookup[int(n_clusters)] = full_matrix

    if not evaluations:
        raise ValueError("No target-regime evaluations produced")

    best_evaluation = sorted(evaluations, key=_cv_evaluation_sort_key, reverse=True)[0]
    chosen_clusters = int(best_evaluation["target_regime_n_clusters"])
    chosen_model = full_models.get(chosen_clusters)
    chosen_fit = full_fit_lookup.get(chosen_clusters)
    chosen_matrix = full_matrix_lookup.get(chosen_clusters)
    if chosen_model is None or chosen_fit is None or chosen_matrix is None:
        raise ValueError("target-regime search did not produce a full model for the chosen cluster count")

    regime_payload = []
    regime_weights = np.asarray(chosen_fit["regime_weights"], dtype=float)
    regime_biases = np.asarray(chosen_fit["regime_biases"], dtype=float)
    chosen_selected_features = list(chosen_matrix["selected_features"])
    for cluster_idx in range(regime_weights.shape[0]):
        feature_relative_weights: Dict[str, float] = {}
        interaction_terms: List[Dict[str, Any]] = []
        latent_regime_penalty_weight = 0.0
        for feature_idx, feature_name in enumerate(chosen_selected_features):
            weight_value = float(regime_weights[cluster_idx, feature_idx])
            feature_text = str(feature_name)
            if feature_text.startswith(_INTERACTION_FEATURE_PREFIX):
                parsed = _parse_interaction_feature_name(feature_text)
                if parsed is None:
                    continue
                interaction_terms.append(
                    {
                        "features": [parsed[0], parsed[1]],
                        "weight": weight_value,
                    }
                )
            elif feature_text == _LATENT_REGIME_SIMILARITY_FEATURE:
                latent_regime_penalty_weight = weight_value
            else:
                feature_relative_weights[feature_text] = weight_value
        regime_entry = {
            "cluster": int(cluster_idx),
            "bias": float(regime_biases[cluster_idx]),
            "feature_relative_weights": feature_relative_weights,
            "interaction_terms": interaction_terms,
        }
        if latent_regime_penalty_weight > 0.0:
            regime_entry["latent_regime_penalty_weight"] = float(latent_regime_penalty_weight)
        regime_payload.append(regime_entry)

    return {
        "scope_key": _clean_scope_key(scope_key),
        "dataset_path": str(dataset_path),
        "selected_features": chosen_selected_features,
        "feature_coverage": dict(chosen_matrix["feature_coverage"]),
        "pair_count": int(chosen_matrix["pair_count"]),
        "group_count": int(unique_groups.size),
        "evaluations": evaluations,
        "best_evaluation": best_evaluation,
        "chosen_target_regime_n_clusters": chosen_clusters,
        "chosen_target_regime_model": chosen_model,
        "chosen_target_regime_payload": regime_payload,
        "latent_regime_seed": int(seed),
        "latent_regime_max_iter": int(latent_max_iter),
    }


def build_scope_payload_with_pairwise_weights(
    base_payload_path: str | Path,
    *,
    scope_key: str,
    learned_weights: Dict[str, float],
    learned_feature_transforms: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_transform_mode: Optional[str] = None,
    latent_regime_model: Optional[Dict[str, Any]] = None,
    target_regime_payload: Optional[Dict[str, Any]] = None,
    second_stage_reranker: Optional[Dict[str, Any]] = None,
    outcome_aware_reranker: Optional[Dict[str, Any]] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = json.loads(Path(base_payload_path).read_text())
    scopes = dict(payload.get("scopes", {}) or {})
    scope = dict(_scope_config_from_payload(payload, scope_key) or {})
    scope["scope_key"] = _clean_scope_key(scope_key)
    normalized_transform_mode = _normalize_feature_transform_mode(
        feature_transform_mode if feature_transform_mode is not None else scope.get("feature_transform_mode")
    )
    scope["feature_transform_mode"] = normalized_transform_mode
    feature_relative_weights = dict(scope.get("feature_relative_weights", {}) or {})
    interaction_weights: Dict[str, float] = {}
    learned_penalties = dict(scope.get("penalties", {}) or {})
    latent_regime_penalty_weight: Optional[float] = None
    for key, value in dict(learned_weights or {}).items():
        key_text = str(key)
        if key_text.startswith(_INTERACTION_FEATURE_PREFIX):
            interaction_weights[key_text] = float(value)
        elif key_text.startswith(_PENALTY_FEATURE_PREFIX):
            penalty_name = _parse_penalty_feature_name(key_text)
            if penalty_name == "size_gap_excess":
                learned_penalties["size_penalty_weight"] = float(value)
            elif penalty_name == "primary_burden_gap_excess":
                learned_penalties["burden_penalty_weight"] = float(value)
        elif key_text.startswith(_LATENT_REGIME_FEATURE_PREFIX):
            if key_text == _LATENT_REGIME_SIMILARITY_FEATURE:
                latent_regime_penalty_weight = float(value)
        else:
            feature_relative_weights[key_text] = float(value)
    scope["feature_relative_weights"] = feature_relative_weights
    feature_transforms = (
        {}
        if feature_transform_mode is not None and normalized_transform_mode == "identity"
        else dict(scope.get("feature_transforms", {}) or {})
    )
    feature_transforms.update(
        {
            str(k): dict(_normalize_transform_spec(v))
            for k, v in dict(learned_feature_transforms or {}).items()
            if _normalize_transform_spec(v)
        }
    )
    if feature_transforms:
        scope["feature_transforms"] = feature_transforms
    elif "feature_transforms" in scope:
        del scope["feature_transforms"]
    if interaction_weights:
        interaction_terms = []
        for feature_name, weight in interaction_weights.items():
            parsed = _parse_interaction_feature_name(feature_name)
            if parsed is None:
                continue
            interaction_terms.append(
                {
                    "features": [parsed[0], parsed[1]],
                    "weight": float(weight),
                }
            )
        scope["interaction_terms"] = interaction_terms
    if learned_penalties:
        scope["penalties"] = learned_penalties
    if isinstance(latent_regime_model, dict) and latent_regime_penalty_weight is not None and latent_regime_penalty_weight > 0.0:
        scope["latent_regime_model"] = dict(latent_regime_model)
        scope["latent_regime_penalty_weight"] = float(latent_regime_penalty_weight)
        scope["latent_regime_feature_name"] = _LATENT_REGIME_SIMILARITY_FEATURE
    if isinstance(target_regime_payload, dict) and isinstance(target_regime_payload.get("model"), dict):
        scope["target_regime_mixture"] = {
            "model": dict(target_regime_payload.get("model") or {}),
            "regimes": list(target_regime_payload.get("regimes") or []),
        }
    if isinstance(second_stage_reranker, dict):
        feature_weights = {
            str(key): float(value)
            for key, value in dict(second_stage_reranker.get("feature_weights", {}) or {}).items()
            if str(key) in set(_second_stage_reranker_feature_names()) and _clean_numeric(value) is not None and float(value) > 0.0
        }
        if feature_weights:
            scope["second_stage_reranker"] = {
                "feature_weights": feature_weights,
                "bias": float(_clean_numeric(second_stage_reranker.get("bias")) or 0.0),
                "shortlist_size": int(_clean_numeric(second_stage_reranker.get("shortlist_size")) or 80),
            }
    if isinstance(outcome_aware_reranker, dict):
        feature_weights = {
            str(key): float(value)
            for key, value in dict(outcome_aware_reranker.get("feature_weights", {}) or {}).items()
            if str(key) in set(_outcome_aware_reranker_feature_names()) and _clean_numeric(value) is not None and float(value) > 0.0
        }
        if feature_weights:
            scope["outcome_aware_reranker"] = {
                "feature_weights": feature_weights,
                "bias": float(_clean_numeric(outcome_aware_reranker.get("bias")) or 0.0),
                "shortlist_size": int(_clean_numeric(outcome_aware_reranker.get("shortlist_size")) or 40),
            }
    # Pairwise-learned scope payloads are produced as runtime candidates by default so
    # previews and promotion checks exercise the same weighted-distance stack they would use live.
    scope["use_in_runtime"] = True
    scope["default_enabled"] = True
    scope["pairwise_precedent_quality_learning"] = dict(notes or {})
    scopes[_clean_scope_key(scope_key)] = scope
    payload["scopes"] = scopes
    payload.setdefault("notes", {})
    payload["notes"]["pairwise_precedent_quality_learning_scope"] = _clean_scope_key(scope_key)
    return payload


def write_json(payload: Dict[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
