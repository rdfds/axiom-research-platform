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


