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


