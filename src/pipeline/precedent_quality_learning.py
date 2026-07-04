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


