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


