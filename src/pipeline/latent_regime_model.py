from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


_LATENT_REGIME_MODEL_VERSION = "latent_regime_kmeans_soft_v1"


def _clean_numeric(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def raw_feature_matrix_from_dataframe(
    df: pd.DataFrame,
    *,
    feature_names: Sequence[str],
) -> np.ndarray:
    cols = []
    for feature_name in feature_names:
        cols.append(pd.to_numeric(df.get(str(feature_name)), errors="coerce").to_numpy(dtype=float))
    if not cols:
        return np.empty((len(df), 0), dtype=float)
    return np.column_stack(cols).astype(float)


def raw_feature_matrix_from_compacts(
    compact_rows: Iterable[Dict[str, Any]],
    *,
    feature_names: Sequence[str],
) -> np.ndarray:
    rows: List[List[float]] = []
    names = [str(name) for name in feature_names]
    for compact in compact_rows:
        payload = dict(compact or {})
        rows.append(
            [
                np.nan if _clean_numeric(payload.get(name)) is None else float(_clean_numeric(payload.get(name)))
                for name in names
            ]
        )
    if not rows:
        return np.empty((0, len(names)), dtype=float)
    return np.asarray(rows, dtype=float)


def _robust_center_scale(raw_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if raw_matrix.ndim != 2:
        raise ValueError("raw_matrix must be 2D")
    n_features = raw_matrix.shape[1]
    medians = np.zeros(n_features, dtype=float)
    scales = np.ones(n_features, dtype=float)
    for idx in range(n_features):
        sample = raw_matrix[:, idx]
        valid = sample[np.isfinite(sample)]
        if valid.size == 0:
            continue
        med = float(np.median(valid))
        q25 = float(np.quantile(valid, 0.25))
        q75 = float(np.quantile(valid, 0.75))
        scale = (q75 - q25) / 1.349
        if (not np.isfinite(scale)) or scale <= 1e-9:
            scale = float(np.std(valid))
        if (not np.isfinite(scale)) or scale <= 1e-9:
            scale = 1.0
        medians[idx] = med
        scales[idx] = scale
    return medians, scales


def _latent_regime_design_matrix(
    raw_matrix: np.ndarray,
    *,
    medians: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    if raw_matrix.ndim != 2:
        raise ValueError("raw_matrix must be 2D")
    centered = (raw_matrix - medians.reshape(1, -1)) / scales.reshape(1, -1)
    missing = ~np.isfinite(centered)
    centered = np.where(missing, 0.0, centered)
    return np.concatenate([centered, missing.astype(float)], axis=1)


def _kmeans_pp_init(
    X: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n_rows = X.shape[0]
    if n_rows == 0:
        raise ValueError("cannot initialize kmeans with empty matrix")
    first_idx = int(rng.integers(0, n_rows))
    centroids = [X[first_idx].copy()]
    while len(centroids) < int(n_clusters):
        dist_sq = np.min(
            np.stack([np.sum((X - centroid.reshape(1, -1)) ** 2, axis=1) for centroid in centroids], axis=1),
            axis=1,
        )
        total = float(np.sum(dist_sq))
        if total <= 1e-12:
            candidate_idx = int(rng.integers(0, n_rows))
        else:
            probs = dist_sq / total
            candidate_idx = int(rng.choice(n_rows, p=probs))
        centroids.append(X[candidate_idx].copy())
    return np.stack(centroids, axis=0)


def fit_latent_regime_kmeans(
    raw_matrix: np.ndarray,
    *,
    feature_names: Sequence[str],
    n_clusters: int,
    seed: int = 7,
    max_iter: int = 100,
) -> Dict[str, Any]:
    if raw_matrix.ndim != 2 or raw_matrix.shape[0] == 0:
        raise ValueError("raw_matrix must be non-empty 2D")
    feature_list = [str(name) for name in feature_names]
    n_clusters = max(1, min(int(n_clusters), int(raw_matrix.shape[0])))
    medians, scales = _robust_center_scale(raw_matrix)
    X = _latent_regime_design_matrix(raw_matrix, medians=medians, scales=scales)
    centroids = _kmeans_pp_init(X, n_clusters=n_clusters, seed=int(seed))
    assignments = np.zeros(X.shape[0], dtype=int)
    for _ in range(max(1, int(max_iter))):
        dist_sq = np.stack(
            [np.sum((X - centroid.reshape(1, -1)) ** 2, axis=1) for centroid in centroids],
            axis=1,
        )
        new_assignments = np.argmin(dist_sq, axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        new_centroids = centroids.copy()
        for idx in range(n_clusters):
            mask = assignments == idx
            if not bool(np.any(mask)):
                continue
            new_centroids[idx] = np.mean(X[mask], axis=0)
        centroids = new_centroids
    final_dist_sq = np.stack(
        [np.sum((X - centroid.reshape(1, -1)) ** 2, axis=1) for centroid in centroids],
        axis=1,
    )
    nearest_dist_sq = np.min(final_dist_sq, axis=1)
    temperature = float(np.median(nearest_dist_sq[np.isfinite(nearest_dist_sq)])) if np.isfinite(nearest_dist_sq).any() else 1.0
    if (not np.isfinite(temperature)) or temperature <= 1e-9:
        temperature = 1.0
    return {
        "version": _LATENT_REGIME_MODEL_VERSION,
        "feature_names": feature_list,
        "n_clusters": int(n_clusters),
        "medians": [float(x) for x in medians.tolist()],
        "scales": [float(x) for x in scales.tolist()],
        "centroids": [[float(v) for v in row.tolist()] for row in centroids],
        "temperature": float(temperature),
        "seed": int(seed),
        "max_iter": int(max_iter),
    }


def latent_regime_memberships(
    raw_matrix: np.ndarray,
    model: Dict[str, Any],
) -> np.ndarray:
    if raw_matrix.ndim != 2:
        raise ValueError("raw_matrix must be 2D")
    medians = np.asarray(model.get("medians") or [], dtype=float)
    scales = np.asarray(model.get("scales") or [], dtype=float)
    centroids = np.asarray(model.get("centroids") or [], dtype=float)
    if medians.ndim != 1 or scales.ndim != 1 or centroids.ndim != 2:
        raise ValueError("invalid latent regime model")
    X = _latent_regime_design_matrix(raw_matrix, medians=medians, scales=scales)
    dist_sq = np.stack(
        [np.sum((X - centroid.reshape(1, -1)) ** 2, axis=1) for centroid in centroids],
        axis=1,
    )
    temperature = float(model.get("temperature") or 1.0)
    if (not np.isfinite(temperature)) or temperature <= 1e-9:
        temperature = 1.0
    logits = -dist_sq / temperature
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    denom = np.sum(probs, axis=1, keepdims=True)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    return probs / denom


def latent_regime_similarity(
    left_raw_matrix: np.ndarray,
    right_raw_matrix: np.ndarray,
    model: Dict[str, Any],
) -> np.ndarray:
    left = latent_regime_memberships(left_raw_matrix, model)
    right = latent_regime_memberships(right_raw_matrix, model)
    if left.shape != right.shape:
        raise ValueError("left/right membership shapes must match")
    return np.sum(left * right, axis=1)
