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


