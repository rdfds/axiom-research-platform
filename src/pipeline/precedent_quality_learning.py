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


