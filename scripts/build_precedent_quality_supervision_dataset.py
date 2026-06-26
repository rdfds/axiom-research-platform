#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import duckdb
import gzip
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RECO_DISABLE_PRECEDENT_NARRATIVE", "1")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_feature_bundle import _STATE_VECTOR_V1_FEATURES, build_model_feature_bundle
from src.pipeline.historical_price_metric_backfill import backfill_historical_price_window_metrics
from src.pipeline.latent_regime_model import fit_latent_regime_kmeans, latent_regime_memberships
from src.pipeline.precedent import _state_vector_baseline_value
from src.pipeline.precedent_brain import (
    _effective_action_subtype,
    _estimate_action_scale,
    _enrich_missing_historical_taxonomy,
    _historical_taxonomy_for_ticker,
    _weighted_state_similarity,
    augment_precedent_state_vector_columns,
)
from src.pipeline.run import _default_precedent_outcomes_path, adapt_snapshot, attach_model_feature_bundle


_PAIRWISE_FEATURE_GAP_SUMMARY_FEATURES = tuple(_STATE_VECTOR_V1_FEATURES)
_REFINITIV_TAXONOMY_REFERENCE_PATH = REPO_ROOT / "data" / "refinitiv" / "fundamentals_all.parquet"
_SEC_TICKER_CIK_PATH = REPO_ROOT / "data" / "mappings" / "sec_ticker_cik.parquet"
_SNAPSHOT_TAXONOMY_LOOKUP_PATH = REPO_ROOT / "data" / "curated" / "snapshot_taxonomy_lookup_2026-02-28.parquet"


def _maybe_backfill_historical_price_window_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if str(os.environ.get("RECO_DISABLE_HISTORICAL_PRICE_BACKFILL") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return frame.copy()
    return backfill_historical_price_window_metrics(frame)


def _prefer_pandas_outcomes_reads() -> bool:
    return str(os.environ.get("RECO_FORCE_PANDAS_OUTCOMES_READ") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


@lru_cache(maxsize=4)
def _cached_pandas_outcomes_frame(outcomes_path_str: str) -> pd.DataFrame:
    return pd.read_parquet(outcomes_path_str)


def _resolve_teacher_recipe(
    *,
    teacher_recipe: str,
    positive_source_mode: str,
    include_within_action_hard_negatives: bool,
    include_same_action_positive_ordering: bool,
    actual_anchor_within_action_negative_source: str,
    always_include_actual_anchor_positive: bool,
    same_family_negatives_only_if_available: bool,
    hard_negative_taxonomy_mode: str,
) -> Dict[str, Any]:
    recipe = str(teacher_recipe or "explicit_flags").strip().lower() or "explicit_flags"
    config = {
        "teacher_recipe": recipe,
        "positive_source_mode": str(positive_source_mode or "include_retrieved").strip().lower() or "include_retrieved",
        "include_within_action_hard_negatives": bool(include_within_action_hard_negatives),
        "include_same_action_positive_ordering": bool(include_same_action_positive_ordering),
        "actual_anchor_within_action_negative_source": str(
            actual_anchor_within_action_negative_source or "retrieved_pool"
        ).strip().lower()
        or "retrieved_pool",
        "always_include_actual_anchor_positive": bool(always_include_actual_anchor_positive),
        "same_family_negatives_only_if_available": bool(same_family_negatives_only_if_available),
        "hard_negative_taxonomy_mode": str(hard_negative_taxonomy_mode or "none").strip().lower() or "none",
    }
    if recipe == "explicit_flags":
        return config
    if recipe == "same_action_best_analog":
        config.update(
            {
                "positive_source_mode": "analog_consensus_same_action_universe",
                "include_within_action_hard_negatives": True,
                "include_same_action_positive_ordering": True,
                "actual_anchor_within_action_negative_source": "same_action_universe",
                "always_include_actual_anchor_positive": False,
                "same_family_negatives_only_if_available": False,
            }
        )
        return config
    if recipe == "same_action_regime_best_analog":
        config.update(
            {
                "positive_source_mode": "analog_regime_consensus_same_action_universe",
                "include_within_action_hard_negatives": True,
                "include_same_action_positive_ordering": True,
                "actual_anchor_within_action_negative_source": "same_action_universe",
                "always_include_actual_anchor_positive": False,
                "same_family_negatives_only_if_available": False,
            }
        )
        return config
    if recipe == "same_action_actual_anchor":
        config.update(
            {
                "positive_source_mode": "actual_anchor_preferred",
                "include_within_action_hard_negatives": True,
                "include_same_action_positive_ordering": False,
                "actual_anchor_within_action_negative_source": "same_action_universe",
                "always_include_actual_anchor_positive": True,
                "same_family_negatives_only_if_available": False,
            }
        )
        return config
    raise ValueError(f"Unsupported teacher_recipe: {teacher_recipe}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a pairwise precedent-quality supervision dataset.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--summary-path", required=False, default="")
    parser.add_argument("--snapshot-catalog-path", required=False, default="")
    parser.add_argument("--snapshot-cache-root", required=False, default="")
    parser.add_argument("--runs-root", required=False, default="")
    parser.add_argument("--eval-prefix", required=False, default="")
    parser.add_argument("--eval-id", required=False, default="001")
    parser.add_argument("--top-k-per-candidate", type=int, default=5)
    parser.add_argument("--outcomes-path", required=False, default="")
    parser.add_argument("--positive-limit-per-source", type=int, default=0)
    parser.add_argument("--negative-limit-per-competitor", type=int, default=0)
    parser.add_argument("--same-family-negatives-only-if-available", action="store_true")
    parser.add_argument("--always-include-actual-anchor-positive", action="store_true")
    parser.add_argument("--include-within-action-hard-negatives", action="store_true")
    parser.add_argument("--include-same-action-positive-ordering", action="store_true")
    parser.add_argument(
        "--teacher-recipe",
        choices=(
            "explicit_flags",
            "same_action_best_analog",
            "same_action_regime_best_analog",
            "same_action_actual_anchor",
        ),
        default="explicit_flags",
    )
    parser.add_argument(
        "--actual-anchor-within-action-negative-source",
        choices=("retrieved_pool", "same_action_universe"),
        default="retrieved_pool",
    )
    parser.add_argument(
        "--positive-source-mode",
        choices=(
            "include_retrieved",
            "actual_anchor_preferred",
            "analog_consensus_same_action_universe",
            "analog_regime_consensus_same_action_universe",
        ),
        default="include_retrieved",
    )
    parser.add_argument(
        "--hard-negative-taxonomy-mode",
        choices=("none", "prefer_same_sector", "prefer_same_subsector_then_sector"),
        default="none",
    )
    parser.add_argument("--analog-regime-cluster-grid", default="2,3,4,5,6")
    parser.add_argument("--analog-regime-seed", type=int, default=7)
    parser.add_argument("--analog-regime-max-iter", type=int, default=100)
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _first(items: Iterable[Any], default: Any = None) -> Any:
    for item in items:
        if item is not None:
            return item
    return default


def _snapshot_cache_root_for_manifest(manifest_path: Path) -> Path:
    if "/configs/" in str(manifest_path):
        return manifest_path.parent.parent / "reports" / "snapshot_cache" / "keyed"
    raise ValueError(f"Could not infer snapshot cache root from {manifest_path}")


def _normalize_as_of_time(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    stamp = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(stamp):
        return text
    return stamp.isoformat()


@lru_cache(maxsize=4)
def _snapshot_catalog_index(snapshot_catalog_path: str) -> Dict[tuple[str, str], Dict[str, Any]]:
    path = Path(snapshot_catalog_path)
    index: Dict[tuple[str, str], Dict[str, Any]] = {}
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (
                str(row.get("company_id") or "").strip(),
                _normalize_as_of_time(str(row.get("as_of_time") or "")),
            )
            if key[0] and key[1]:
                index[key] = row
    return index


def _load_snapshot_row(
    snapshot_cache_root: Path,
    company_id: str,
    as_of_time: str,
    *,
    snapshot_catalog_path: Path | None = None,
) -> Dict[str, Any]:
    if snapshot_catalog_path is not None and snapshot_catalog_path.exists():
        catalog_key = (str(company_id or "").strip(), _normalize_as_of_time(as_of_time))
        catalog_row = _snapshot_catalog_index(str(snapshot_catalog_path)).get(catalog_key)
        if catalog_row is not None:
            return dict(catalog_row)

    as_of_date = str(as_of_time).split("T", 1)[0]
    legacy_snapshot_path = snapshot_cache_root / f"as_of_date={as_of_date}" / f"company_id={company_id}.json"
    if legacy_snapshot_path.exists():
        return _load_json(legacy_snapshot_path)

    as_of_stamp = pd.Timestamp(as_of_time, tz="UTC")
    modern_snapshot_path = (
        snapshot_cache_root
        / f"company_id={company_id}"
        / f"snapshot_as_of={as_of_stamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    if modern_snapshot_path.exists():
        return _load_json(modern_snapshot_path)

    prefix = f"snapshot_as_of={as_of_stamp.strftime('%Y%m%dT%H%M%SZ')}"
    candidates = sorted((snapshot_cache_root / f"company_id={company_id}").glob(f"{prefix}*.json"))
    if candidates:
        return _load_json(candidates[0])

    return _load_json(legacy_snapshot_path)


def _load_anchor_outcomes_lookup(
    outcomes_path: Path,
    *,
    cases: List[Dict[str, Any]],
) -> Dict[tuple[str, str], List[Dict[str, Any]]]:
    company_ids = sorted(
        {
            str(case.get("source_company_id") or case.get("company_id") or "").strip()
            for case in cases
            if str(case.get("source_company_id") or case.get("company_id") or "").strip()
        }
    )
    action_ids = sorted({str(case.get("anchor_action_id") or "").strip() for case in cases if str(case.get("anchor_action_id") or "").strip()})
    if not company_ids or not action_ids:
        return {}

    def _sql_literal(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    company_sql = ", ".join(_sql_literal(value) for value in company_ids)
    action_sql = ", ".join(_sql_literal(value) for value in action_ids)
    if _prefer_pandas_outcomes_reads():
        frame = _cached_pandas_outcomes_frame(str(outcomes_path)).copy()
        frame["company_id"] = frame["company_id"].astype(str)
        frame["normalized_action_id"] = frame["normalized_action_id"].astype(str)
        frame = frame[
            frame["company_id"].isin(company_ids)
            & frame["normalized_action_id"].isin(action_ids)
        ].reset_index(drop=True)
    else:
        query = f"""
            SELECT *
            FROM read_parquet(?)
            WHERE CAST(company_id AS VARCHAR) IN ({company_sql})
              AND CAST(normalized_action_id AS VARCHAR) IN ({action_sql})
        """
        frame = duckdb.execute(query, [str(outcomes_path)]).df()
    if frame.empty:
        return {}
    frame = _maybe_backfill_historical_price_window_metrics(frame)
    frame = _enrich_missing_historical_taxonomy(frame)
    frame = augment_precedent_state_vector_columns(frame)
    frame["company_id"] = frame["company_id"].astype(str)
    frame["normalized_action_id"] = frame["normalized_action_id"].astype(str)
    frame["action_date"] = pd.to_datetime(frame["action_date"], utc=True, errors="coerce")
    lookup: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in frame.to_dict(orient="records"):
        company_id = str(row.get("company_id") or "").strip()
        action_id = str(row.get("normalized_action_id") or "").strip()
        if not company_id or not action_id:
            continue
        lookup.setdefault((company_id, action_id), []).append(row)
    return lookup


def _load_precedent_outcomes_lookup(
    outcomes_path: Path,
    *,
    cases: List[Dict[str, Any]],
    required_keys: Optional[Iterable[tuple[str, str, str]]] = None,
) -> Dict[tuple[str, str, str], Dict[str, Any]]:
    required_key_set = {
        (
            str(company_id or "").strip(),
            str(action_id or "").strip(),
            _normalize_as_of_time(str(action_time or "")),
        )
        for company_id, action_id, action_time in list(required_keys or [])
        if str(company_id or "").strip() and str(action_id or "").strip() and _normalize_as_of_time(str(action_time or ""))
    }
    if required_keys is not None and not required_key_set:
        return {}

    action_families = sorted(
        {
            str(case.get("anchor_action_family") or "").strip()
            for case in cases
            if str(case.get("anchor_action_family") or "").strip()
        }
    )
    action_ids = sorted(
        {
            str(case.get("anchor_action_id") or "").strip()
            for case in cases
            if str(case.get("anchor_action_id") or "").strip()
        }
    )
    filters: List[str] = []

    def _sql_literal(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    if required_key_set:
        company_ids = sorted({company_id for company_id, _, _ in required_key_set})
        exact_action_ids = sorted({action_id for _, action_id, _ in required_key_set})
        if company_ids:
            company_sql = ", ".join(_sql_literal(value) for value in company_ids)
            filters.append(f"CAST(company_id AS VARCHAR) IN ({company_sql})")
        if exact_action_ids:
            action_sql = ", ".join(_sql_literal(value) for value in exact_action_ids)
            filters.append(f"CAST(normalized_action_id AS VARCHAR) IN ({action_sql})")
        where_sql = " AND ".join(filters) if filters else "TRUE"
    else:
        if action_families:
            family_sql = ", ".join(_sql_literal(value) for value in action_families)
            filters.append(f"CAST(normalized_action_family AS VARCHAR) IN ({family_sql})")
        if action_ids:
            action_sql = ", ".join(_sql_literal(value) for value in action_ids)
            filters.append(f"CAST(normalized_action_id AS VARCHAR) IN ({action_sql})")
        where_sql = " OR ".join(filters) if filters else "TRUE"
    if _prefer_pandas_outcomes_reads():
        frame = _cached_pandas_outcomes_frame(str(outcomes_path)).copy()
        if "company_id" in frame.columns:
            frame["company_id"] = frame["company_id"].astype(str)
        if "normalized_action_id" in frame.columns:
            frame["normalized_action_id"] = frame["normalized_action_id"].astype(str)
        if "normalized_action_family" in frame.columns:
            frame["normalized_action_family"] = frame["normalized_action_family"].astype(str)
        mask = pd.Series(True, index=frame.index, dtype=bool)
        if required_key_set:
            company_ids = {company_id for company_id, _, _ in required_key_set}
            action_ids = {action_id for _, action_id, _ in required_key_set}
            if company_ids:
                mask = mask & frame["company_id"].isin(company_ids)
            if action_ids:
                mask = mask & frame["normalized_action_id"].isin(action_ids)
        else:
            family_mask = (
                frame["normalized_action_family"].isin(action_families)
                if action_families and "normalized_action_family" in frame.columns
                else pd.Series(False, index=frame.index, dtype=bool)
            )
            action_mask = (
                frame["normalized_action_id"].isin(action_ids)
                if action_ids and "normalized_action_id" in frame.columns
                else pd.Series(False, index=frame.index, dtype=bool)
            )
            mask = family_mask | action_mask if (action_families or action_ids) else mask
        frame = frame[mask].reset_index(drop=True)
    else:
        query = f"""
            SELECT *
            FROM read_parquet(?)
            WHERE {where_sql}
        """
        frame = duckdb.execute(query, [str(outcomes_path)]).df()
    if frame.empty:
        return {}
    frame = _maybe_backfill_historical_price_window_metrics(frame)
    frame = augment_precedent_state_vector_columns(frame)
    frame["company_id"] = frame["company_id"].astype(str)
    frame["normalized_action_id"] = frame["normalized_action_id"].astype(str)
    frame["action_date"] = pd.to_datetime(frame["action_date"], utc=True, errors="coerce")
    lookup: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        action_date = row.get("action_date")
        if pd.isna(action_date):
            continue
        key = (
            str(row.get("company_id") or "").strip(),
            str(row.get("normalized_action_id") or "").strip(),
            _normalize_as_of_time(str(action_date)),
        )
        if required_key_set and key not in required_key_set:
            continue
        if key[0] and key[1] and key[2] and key not in lookup:
            lookup[key] = row
    return lookup


def _robust_feature_scale_map(frame: pd.DataFrame) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    for feature in _STATE_VECTOR_V1_FEATURES:
        series = pd.to_numeric(frame.get(feature), errors="coerce").dropna()
        if series.empty:
            scales[feature] = 1.0
            continue
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        scale = float(q75 - q25)
        if not pd.notna(scale) or scale <= 1e-9:
            scale = float(series.std(ddof=0))
        if not pd.notna(scale) or scale <= 1e-9:
            median = float(series.median())
            scale = abs(median)
        if not pd.notna(scale) or scale <= 1e-9:
            scale = 1.0
        scales[feature] = float(scale)
    return scales


def _parse_int_grid(text: str, *, default: Iterable[int]) -> List[int]:
    values: List[int] = []
    for chunk in str(text or "").split(","):
        piece = str(chunk or "").strip()
        if not piece:
            continue
        try:
            values.append(int(piece))
        except Exception:
            continue
    if not values:
        values = [int(value) for value in default]
    return sorted({max(1, int(value)) for value in values})


def _latent_regime_design_matrix(raw_matrix: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    medians = np.asarray(model.get("medians") or [], dtype=float)
    scales = np.asarray(model.get("scales") or [], dtype=float)
    if raw_matrix.ndim != 2:
        raise ValueError("raw_matrix must be 2D")
    if medians.ndim != 1 or scales.ndim != 1 or raw_matrix.shape[1] != medians.shape[0]:
        raise ValueError("latent regime model shape mismatch")
    safe_scales = np.where(np.isfinite(scales) & (scales > 1e-9), scales, 1.0)
    centered = (raw_matrix - medians.reshape(1, -1)) / safe_scales.reshape(1, -1)
    missing = ~np.isfinite(centered)
    centered = np.where(missing, 0.0, centered)
    return np.concatenate([centered, missing.astype(float)], axis=1)


def _mean_pairwise_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left/right must be 2D")
    if left.shape[1] != right.shape[1]:
        raise ValueError("left/right feature counts must match")
    if left.shape[0] == 0 or right.shape[0] == 0:
        return np.empty((left.shape[0], right.shape[0]), dtype=float)
    diff = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _silhouette_score_for_assignments(X: np.ndarray, assignments: np.ndarray) -> float:
    if X.ndim != 2 or assignments.ndim != 1 or X.shape[0] != assignments.shape[0]:
        raise ValueError("shape mismatch for silhouette score")
    unique = [int(value) for value in np.unique(assignments).tolist()]
    if len(unique) <= 1 or X.shape[0] <= len(unique):
        return float("-inf")
    distances = _mean_pairwise_distance(X, X)
    scores: List[float] = []
    row_idx = np.arange(X.shape[0], dtype=int)
    for idx in range(X.shape[0]):
        cluster = int(assignments[idx])
        same_mask = assignments == cluster
        same_mask[idx] = False
        if not bool(np.any(same_mask)):
            continue
        a_value = float(np.mean(distances[idx, same_mask]))
        b_value = float("inf")
        for other_cluster in unique:
            if other_cluster == cluster:
                continue
            other_mask = assignments == other_cluster
            if not bool(np.any(other_mask)):
                continue
            b_value = min(b_value, float(np.mean(distances[idx, other_mask])))
        denom = max(a_value, b_value, 1e-9)
        if np.isfinite(b_value):
            scores.append(float((b_value - a_value) / denom))
    if not scores:
        return float("-inf")
    return float(np.mean(scores))


def _select_best_latent_regime_model(
    raw_matrix: np.ndarray,
    *,
    feature_names: Iterable[str],
    cluster_grid: Iterable[int],
    seed: int,
    max_iter: int,
) -> Optional[Dict[str, Any]]:
    if raw_matrix.ndim != 2 or raw_matrix.shape[0] < 4:
        return None
    best_model: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    best_cluster_count = 0
    feature_list = [str(feature) for feature in feature_names]
    for n_clusters in sorted({max(1, int(value)) for value in list(cluster_grid or [])}):
        if n_clusters <= 1 or n_clusters >= raw_matrix.shape[0]:
            continue
        try:
            model = fit_latent_regime_kmeans(
                raw_matrix,
                feature_names=feature_list,
                n_clusters=int(n_clusters),
                seed=int(seed),
                max_iter=int(max_iter),
            )
            memberships = latent_regime_memberships(raw_matrix, model)
            assignments = np.argmax(memberships, axis=1).astype(int)
            design = _latent_regime_design_matrix(raw_matrix, model)
            score = _silhouette_score_for_assignments(design, assignments)
        except Exception:
            continue
        if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and int(n_clusters) < best_cluster_count):
            best_model = dict(model)
            best_score = float(score)
            best_cluster_count = int(n_clusters)
    if best_model is None:
        return None
    best_model["selection_score"] = float(best_score)
    return best_model


def _load_same_action_universe_lookup(
    outcomes_path: Path,
    *,
    cases: List[Dict[str, Any]],
    include_latent_regime_model: bool = False,
    latent_regime_cluster_grid: Optional[Iterable[int]] = None,
    latent_regime_seed: int = 7,
    latent_regime_max_iter: int = 100,
) -> Dict[str, Dict[str, Any]]:
    action_ids = sorted(
        {
            str(case.get("anchor_action_id") or "").strip()
            for case in cases
            if str(case.get("anchor_action_id") or "").strip()
        }
    )
    if not action_ids:
        return {}

    def _sql_literal(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    action_sql = ", ".join(_sql_literal(value) for value in action_ids)
    if _prefer_pandas_outcomes_reads():
        frame = _cached_pandas_outcomes_frame(str(outcomes_path)).copy()
        frame["normalized_action_id"] = frame["normalized_action_id"].astype(str)
        frame = frame[frame["normalized_action_id"].isin(action_ids)].reset_index(drop=True)
    else:
        query = f"""
            SELECT *
            FROM read_parquet(?)
            WHERE CAST(normalized_action_id AS VARCHAR) IN ({action_sql})
        """
        frame = duckdb.execute(query, [str(outcomes_path)]).df()
    if frame.empty:
        return {}
    frame = _maybe_backfill_historical_price_window_metrics(frame)
    frame = _enrich_missing_historical_taxonomy(frame)
    frame = augment_precedent_state_vector_columns(frame)
    frame["company_id"] = frame["company_id"].astype(str)
    frame["normalized_action_id"] = frame["normalized_action_id"].astype(str)
    frame["action_date"] = pd.to_datetime(frame["action_date"], utc=True, errors="coerce")

    lookup: Dict[str, Dict[str, Any]] = {}
    for action_id, group in frame.groupby("normalized_action_id", dropna=False, sort=False):
        action_key = str(action_id or "").strip()
        if not action_key:
            continue
        group = group.reset_index(drop=True)
        sector_arr = group.get("taxonomy.sector", pd.Series("", index=group.index)).fillna("")
        if "sector" in group.columns:
            sector_arr = sector_arr.where(sector_arr.astype(str).str.strip().astype(bool), group["sector"].fillna(""))
        if "base_sector" in group.columns:
            sector_arr = sector_arr.where(sector_arr.astype(str).str.strip().astype(bool), group["base_sector"].fillna(""))
        subsector_arr = group.get("taxonomy.subsector", pd.Series("", index=group.index)).fillna("")
        if "subsector" in group.columns:
            subsector_arr = subsector_arr.where(
                subsector_arr.astype(str).str.strip().astype(bool),
                group["subsector"].fillna(""),
            )
        if "industry" in group.columns:
            subsector_arr = subsector_arr.where(
                subsector_arr.astype(str).str.strip().astype(bool),
                group["industry"].fillna(""),
            )
        feature_matrix = np.column_stack(
            [
                pd.to_numeric(group.get(feature), errors="coerce").to_numpy(dtype=float)
                for feature in _STATE_VECTOR_V1_FEATURES
            ]
        )
        lookup[action_key] = {
            "rows": group.to_dict(orient="records"),
            "feature_scales": _robust_feature_scale_map(group),
            "feature_matrix": feature_matrix,
            "company_id_arr": group["company_id"].astype(str).to_numpy(dtype=object),
            "action_time_arr": pd.DatetimeIndex(
                pd.to_datetime(group["action_date"], utc=True, errors="coerce")
            ).tz_convert(None).to_numpy(dtype="datetime64[ns]"),
            "sector_arr": sector_arr.astype(str).to_numpy(dtype=object),
            "subsector_arr": subsector_arr.astype(str).to_numpy(dtype=object),
        }
        if include_latent_regime_model:
            latent_model = _select_best_latent_regime_model(
                feature_matrix,
                feature_names=_STATE_VECTOR_V1_FEATURES,
                cluster_grid=list(latent_regime_cluster_grid or [2, 3, 4, 5, 6]),
                seed=int(latent_regime_seed),
                max_iter=int(latent_regime_max_iter),
            )
            if isinstance(latent_model, dict):
                try:
                    memberships = latent_regime_memberships(feature_matrix, latent_model)
                    lookup[action_key]["latent_regime_model"] = dict(latent_model)
                    lookup[action_key]["latent_regime_memberships"] = memberships.astype(float)
                    lookup[action_key]["latent_regime_assignments"] = np.argmax(memberships, axis=1).astype(int)
                except Exception:
                    pass
    return lookup


def _collect_precedent_reference_keys(
    precedent_matches: Dict[str, Any],
    *,
    top_k: int,
) -> List[tuple[str, str, str]]:
    keys: List[tuple[str, str, str]] = []
    for result in list(precedent_matches.get("results", []) or []):
        matches = list((result.get("precedent_pack") or {}).get("matches", []) or [])[:top_k]
        for match in matches:
            key = (
                str(match.get("company_id") or "").strip(),
                str(match.get("action_id") or "").strip(),
                _normalize_as_of_time(str(match.get("decision_time") or "")),
            )
            if key[0] and key[1] and key[2]:
                keys.append(key)
    return keys


def _target_compact_values(snapshot_row: Dict[str, Any]) -> Dict[str, Any]:
    adapted_row, _ = adapt_snapshot(snapshot_row)
    adapted_row = attach_model_feature_bundle(adapted_row)
    bundle = build_model_feature_bundle(adapted_row)
    compact = dict(bundle.get("state_vector_v1", {}).get("values", {}) or {})

    feature_payload = dict(snapshot_row.get("features") or {})
    if not feature_payload:
        return compact

    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            out = float(value)
        except Exception:
            return None
        if pd.isna(out):
            return None
        return out

    flattened: Dict[str, Any] = {}
    for key, payload in feature_payload.items():
        if isinstance(payload, dict):
            flattened[str(key)] = payload.get("value")
        else:
            flattened[str(key)] = payload

    alias_map = {
        "ebitda_margin": ("operating.ebitda_margin_ttm",),
        "ev_ebitda": ("market.ev_ebitda",),
        "gross_leverage_including_retirement": (
            "capital_structure.gross_leverage_including_pension",
            "capital_structure.gross_leverage",
        ),
        "gross_obligation_burden": (
            "capital_structure.gross_leverage_including_pension",
            "capital_structure.gross_leverage",
        ),
        "net_leverage_including_retirement": (
            "capital_structure.net_leverage_including_pension",
            "capital_structure.net_leverage",
        ),
        "leverage_net_debt_ebitda": (
            "capital_structure.net_leverage_including_pension",
            "capital_structure.net_leverage",
        ),
        "available_for_actions": (
            "liquidity.available_for_actions",
            "liquidity.liquidity_total",
            "liquidity.usable_cash",
            "liquidity.cash",
        ),
        "available_liquidity_normalized": (
            "liquidity.available_liquidity_normalized",
        ),
        "debt_due_next_24m": (
            "capital_structure.debt_due_0_12m",
            "capital_structure.debt_due_12_24m",
        ),
        "debt_due_0_12m": (
            "capital_structure.debt_due_0_12m",
        ),
        "current_debt": (
            "capital_structure.debt_due_0_12m",
            "capital_structure.total_debt",
        ),
        "interest_coverage": (
            "capital_structure.interest_coverage",
            "capital_structure.interest_coverage_market",
            "capital_structure.interest_coverage_reported",
            "capital_structure.fixed_charge_coverage",
        ),
        "fcf_yield": (
            "market.fcf_yield",
            "capital_return.buyback_capacity_proxy",
        ),
        "volatility_90d": ("market.volatility_90d",),
        "drawdown_90d": ("market.drawdown_90d",),
        "credit_window_proxy": ("market.credit_window_proxy",),
        "equity_window_proxy": ("market.equity_window_proxy",),
        "credit_spread_level": ("market.credit_spread_level",),
        "fed_funds_effective": ("macro.fed_funds_effective",),
        "hy_oas": ("macro.hy_oas",),
        "macro_vix": ("macro.vix",),
        "revenue_yoy_last_q": ("operating.revenue_yoy_last_q",),
        "revenue_yoy": ("operating.revenue_yoy_last_q", "operating.revenue_cagr_3y"),
        "sector": ("taxonomy.sector",),
        "subsector": ("taxonomy.subsector",),
    }
    for target_key, candidates in alias_map.items():
        if flattened.get(target_key) is not None:
            continue
        for candidate in candidates:
            value = flattened.get(candidate)
            if value is not None:
                flattened[target_key] = value
                break

    if flattened.get("revenue_ttm") is None:
        enterprise_value = _safe_float(flattened.get("market.enterprise_value"))
        ev_ebitda = _safe_float(flattened.get("market.ev_ebitda"))
        margin = _safe_float(flattened.get("operating.ebitda_margin_ttm"))
        if (
            enterprise_value is not None
            and enterprise_value > 0.0
            and ev_ebitda is not None
            and ev_ebitda > 0.0
            and margin is not None
            and margin > 0.0
        ):
            flattened["revenue_ttm"] = enterprise_value / ev_ebitda / margin

    if compact.get("state_vector_v1.market_stress") is None:
        macro_vix = _safe_float(flattened.get("macro_vix"))
        if macro_vix is not None:
            compact["state_vector_v1.market_stress"] = max(0.0, min(1.0, macro_vix / 80.0))

    for feature in _STATE_VECTOR_V1_FEATURES:
        if compact.get(feature) is not None:
            continue
        compact[feature] = _state_vector_baseline_value(flattened, feature)

    return compact


def _target_taxonomy(snapshot_row: Dict[str, Any]) -> Dict[str, str]:
    features = dict(snapshot_row.get("features") or {})

    def _feature_value(name: str) -> str:
        raw = features.get(name)
        if isinstance(raw, dict):
            value = raw.get("value")
        else:
            value = raw
        return str(value or "").strip()

    return {
        "sector": _feature_value("taxonomy.sector"),
        "subsector": _feature_value("taxonomy.subsector"),
    }


def _snapshot_market_cap(snapshot_row: Dict[str, Any]) -> Optional[float]:
    features = dict(snapshot_row.get("features") or {})
    raw = features.get("market.market_cap_provider_direct")
    if isinstance(raw, dict):
        raw = raw.get("value")
    try:
        value = float(raw)
    except Exception:
        return None
    if pd.isna(value):
        return None
    return value


def _outcome_row_action_params(row: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key in (
        "amount_usd",
        "absolute_usd",
        "draw_amount_usd",
        "resize_amount_usd",
        "transaction_size_pct_market_cap",
        "transaction_size_pct_ev",
        "action_size",
    ):
        value = row.get(key)
        try:
            numeric = float(value)
        except Exception:
            numeric = None
        if numeric is None or pd.isna(numeric):
            continue
        params[key] = numeric
    if "amount_usd" not in params and "action_size" in params:
        params["amount_usd"] = params["action_size"]
    if "action_size" not in params and "amount_usd" in params:
        params["action_size"] = params["amount_usd"]
    raw_subtype = str(row.get("raw_action_subtype") or row.get("action_subtype") or "").strip()
    if raw_subtype:
        params["source_action_subtype"] = raw_subtype
    return params


def _case_anchor_action_subtype(case: Dict[str, Any]) -> str:
    return str(case.get("anchor_action_subtype") or case.get("source_action_subtype") or "").strip()


def _case_anchor_effective_action_subtype(case: Dict[str, Any]) -> str:
    action_id = str(case.get("anchor_action_id") or "").strip()
    raw_subtype = _case_anchor_action_subtype(case)
    if not action_id or not raw_subtype:
        return ""
    params = {"source_action_subtype": raw_subtype}
    return str(_effective_action_subtype(action_id, raw_subtype, params) or "").strip()


def _row_effective_action_subtype(action_id: str, row: Dict[str, Any]) -> str:
    raw_subtype = str(row.get("raw_action_subtype") or row.get("action_subtype") or "").strip()
    if not action_id or not raw_subtype:
        return ""
    params = _outcome_row_action_params(row)
    params.setdefault("source_action_subtype", raw_subtype)
    return str(_effective_action_subtype(action_id, raw_subtype, params) or "").strip()


def _outcome_row_market_cap(row: Dict[str, Any]) -> Optional[float]:
    for key in ("base_market_cap", "market_cap"):
        try:
            value = float(row.get(key))
        except Exception:
            value = None
        if value is None or pd.isna(value):
            continue
        return value
    return None


def _target_context_from_anchor_outcome(
    case: Dict[str, Any],
    *,
    anchor_outcomes_lookup: Dict[tuple[str, str], List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    actual_row = _select_actual_anchor_outcome(case, anchor_outcomes_lookup=anchor_outcomes_lookup)
    if actual_row is None:
        return None

    target_compact = {
        feature: actual_row.get(feature)
        for feature in _STATE_VECTOR_V1_FEATURES
    }
    target_taxonomy = {
        "sector": str(
            actual_row.get("taxonomy.sector")
            or actual_row.get("sector")
            or actual_row.get("base_sector")
            or ""
        ).strip(),
        "subsector": str(
            actual_row.get("taxonomy.subsector")
            or actual_row.get("subsector")
            or actual_row.get("industry")
            or actual_row.get("base_industry")
            or ""
        ).strip(),
    }
    return {
        "target_compact": target_compact,
        "target_taxonomy": target_taxonomy,
        "target_action_params": _outcome_row_action_params(actual_row),
        "target_market_cap": _outcome_row_market_cap(actual_row),
        "target_source": "anchor_outcome_fallback",
    }


def _target_context_from_same_action_universe(
    case: Dict[str, Any],
    *,
    same_action_universe_lookup: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    action_id = str(case.get("anchor_action_id") or "").strip()
    if not action_id:
        return None
    payload = dict(same_action_universe_lookup.get(action_id) or {})
    rows = list(payload.get("rows") or [])
    if not rows:
        return None

    company_id = str(case.get("source_company_id") or case.get("company_id") or "").strip()
    ticker = str(case.get("ticker") or "").strip().upper()
    exact_rows = [
        row
        for row in rows
        if company_id and str(row.get("company_id") or "").strip() == company_id
    ]
    if not exact_rows and ticker:
        exact_rows = [
            row
            for row in rows
            if str(row.get("ticker") or "").strip().upper() == ticker
        ]
    if not exact_rows:
        return None

    anchor_dt = pd.to_datetime(case.get("anchor_action_date"), utc=True, errors="coerce")
    ranked: List[tuple[float, Dict[str, Any]]] = []
    for row in exact_rows:
        action_dt = pd.to_datetime(row.get("action_date"), utc=True, errors="coerce")
        delta = abs((action_dt - anchor_dt).total_seconds()) if pd.notna(anchor_dt) and pd.notna(action_dt) else float("inf")
        ranked.append((float(delta), row))
    ranked.sort(key=lambda item: item[0])
    actual_row = dict(ranked[0][1]) if ranked else dict(exact_rows[0])

    target_compact = {
        feature: actual_row.get(feature)
        for feature in _STATE_VECTOR_V1_FEATURES
    }
    target_taxonomy = _outcome_row_taxonomy(actual_row)
    return {
        "target_compact": target_compact,
        "target_taxonomy": target_taxonomy,
        "target_action_params": _outcome_row_action_params(actual_row),
        "target_market_cap": _outcome_row_market_cap(actual_row),
        "target_source": "same_action_universe_fallback",
    }


def _target_context_from_exact_outcomes_row(
    case: Dict[str, Any],
    *,
    outcomes_path: Path,
) -> Optional[Dict[str, Any]]:
    company_id = str(case.get("source_company_id") or case.get("company_id") or "").strip()
    action_id = str(case.get("anchor_action_id") or "").strip()
    if not company_id or not action_id or not outcomes_path.exists():
        return None

    if _prefer_pandas_outcomes_reads():
        frame = _cached_pandas_outcomes_frame(str(outcomes_path)).copy()
        frame["company_id"] = frame["company_id"].astype(str)
        frame["normalized_action_id"] = frame["normalized_action_id"].astype(str)
        frame = frame[
            (frame["company_id"] == company_id)
            & (frame["normalized_action_id"] == action_id)
        ].reset_index(drop=True)
    else:
        query = """
            SELECT *
            FROM read_parquet(?)
            WHERE CAST(company_id AS VARCHAR) = ?
              AND CAST(normalized_action_id AS VARCHAR) = ?
        """
        frame = duckdb.execute(query, [str(outcomes_path), company_id, action_id]).df()
    if frame.empty:
        ticker = str(case.get("ticker") or "").strip().upper()
        if not ticker:
            return None
        if _prefer_pandas_outcomes_reads():
            frame = _cached_pandas_outcomes_frame(str(outcomes_path)).copy()
            frame["normalized_action_id"] = frame["normalized_action_id"].astype(str)
            frame["ticker"] = frame.get("ticker", pd.Series("", index=frame.index)).astype(str).str.upper()
            frame = frame[
                (frame["ticker"] == ticker)
                & (frame["normalized_action_id"] == action_id)
            ].reset_index(drop=True)
        else:
            ticker_query = """
                SELECT *
                FROM read_parquet(?)
                WHERE UPPER(CAST(ticker AS VARCHAR)) = ?
                  AND CAST(normalized_action_id AS VARCHAR) = ?
            """
            frame = duckdb.execute(ticker_query, [str(outcomes_path), ticker, action_id]).df()
        if frame.empty:
            return None

    frame = _maybe_backfill_historical_price_window_metrics(frame)
    frame = _enrich_missing_historical_taxonomy(frame)
    frame = augment_precedent_state_vector_columns(frame)
    frame["action_date"] = pd.to_datetime(frame["action_date"], utc=True, errors="coerce")
    anchor_dt = pd.to_datetime(case.get("anchor_action_date"), utc=True, errors="coerce")
    anchor_raw_subtype = _case_anchor_action_subtype(case).lower()
    anchor_effective_subtype = _case_anchor_effective_action_subtype(case)
    row_raw_subtypes = (
        frame.get("raw_action_subtype", pd.Series("", index=frame.index))
        .fillna(frame.get("action_subtype", pd.Series("", index=frame.index)))
        .astype(str)
        .str.strip()
    )
    if anchor_raw_subtype:
        exact_rank = (row_raw_subtypes.str.lower() != anchor_raw_subtype).astype(int)
    else:
        exact_rank = pd.Series(0, index=frame.index, dtype=int)
    if anchor_effective_subtype:
        family_rank = pd.Series(
            [
                0
                if _row_effective_action_subtype(action_id, dict(row)) == anchor_effective_subtype
                else 1
                for row in frame.to_dict(orient="records")
            ],
            index=frame.index,
            dtype=int,
        )
    else:
        family_rank = pd.Series(0, index=frame.index, dtype=int)
    if pd.notna(anchor_dt):
        delta_rank = (frame["action_date"] - anchor_dt).abs()
    else:
        delta_rank = pd.Series(pd.Timedelta(0), index=frame.index)
    action_sizes = pd.to_numeric(frame.get("action_size", pd.Series(np.nan, index=frame.index)), errors="coerce")
    frame = frame.assign(
        _anchor_exact_rank=exact_rank,
        _anchor_family_rank=family_rank,
        _delta=delta_rank,
        _size_rank=-action_sizes.fillna(0.0),
    ).sort_values(
        ["_anchor_exact_rank", "_anchor_family_rank", "_delta", "_size_rank"],
        kind="stable",
    )
    actual_row = dict(frame.iloc[0].to_dict()) if not frame.empty else None
    if not actual_row:
        return None

    target_compact = {
        feature: actual_row.get(feature)
        for feature in _STATE_VECTOR_V1_FEATURES
    }
    target_taxonomy = _outcome_row_taxonomy(actual_row)
    return {
        "target_compact": target_compact,
        "target_taxonomy": target_taxonomy,
        "target_action_params": _outcome_row_action_params(actual_row),
        "target_market_cap": _outcome_row_market_cap(actual_row),
        "target_source": "exact_outcome_row_fallback",
    }


def _artifact_paths_from_runs_root(
    *,
    runs_root: Path,
    eval_prefix: str,
    eval_id: str,
    company_id: str,
    as_of_time: str = "",
) -> Optional[Dict[str, Path]]:
    normalized_as_of_time = _normalize_as_of_time(as_of_time)
    search_roots: List[Dict[str, Path]] = []
    if eval_prefix:
        eval_dir = runs_root / f"{eval_prefix}_eval_{int(eval_id):03d}"
        search_roots.append({"runs_dir": eval_dir / "runs", "artifacts_dir": eval_dir / "artifacts"})
    search_roots.append({"runs_dir": runs_root / "runs", "artifacts_dir": runs_root / "artifacts"})
    for root in search_roots:
        runs_dir = root["runs_dir"]
        artifacts_dir = root["artifacts_dir"]
        if not runs_dir.exists() or not artifacts_dir.exists():
            continue
        for run_json_path in runs_dir.glob("*.json"):
            try:
                payload = _load_json(run_json_path)
            except Exception:
                continue
            if str(payload.get("company_id") or "") != company_id:
                continue
            payload_times = {
                _normalize_as_of_time(str(payload.get("as_of_time") or "")),
                _normalize_as_of_time(str(payload.get("decision_time") or "")),
                _normalize_as_of_time(str(payload.get("snapshot_as_of") or "")),
                _normalize_as_of_time(str(payload.get("target_as_of_time") or "")),
            }
            nonempty_times = {time_value for time_value in payload_times if time_value}
            if normalized_as_of_time and nonempty_times and normalized_as_of_time not in nonempty_times:
                continue
            run_id = str(payload.get("run_id") or "")
            if not run_id:
                continue
            artifact_root = artifacts_dir / f"run_id={run_id}"
            precedent_index_path = artifact_root / "PrecedentIndex.json"
            precedent_matches_path = artifact_root / "PrecedentMatches.json"
            if precedent_index_path.exists() and precedent_matches_path.exists():
                return {
                    "precedent_index_path": precedent_index_path,
                    "precedent_matches_path": precedent_matches_path,
                }
    return None


def _candidate_rows_by_id(precedent_index: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in list(precedent_index.get("candidate_rows", []) or []):
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id:
            out[candidate_id] = dict(row)
    return out


def _candidate_rankings(precedent_index: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in list(precedent_index.get("candidate_rows", []) or [])]
    rows.sort(key=lambda row: float(row.get("precedent_confidence") or 0.0), reverse=True)
    return rows


def _top_candidate_per_action(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        action_id = str(row.get("action_id") or "").strip()
        if not action_id or action_id in seen:
            continue
        seen.add(action_id)
        out.append(row)
    return out


def _result_by_candidate_id(precedent_matches: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in list(precedent_matches.get("results", []) or []):
        candidate = dict(row.get("candidate") or {})
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if candidate_id:
            out[candidate_id] = dict(row)
    return out


def _absdiff(a: Any, b: Any) -> Optional[float]:
    try:
        if a is None or b is None:
            return None
        a_value = float(a)
        b_value = float(b)
        if not np.isfinite(a_value) or not np.isfinite(b_value):
            return None
        return abs(a_value - b_value)
    except Exception:
        return None


_HARD_NEGATIVE_SAFETY_FEATURES = (
    "state_vector_v1.net_obligation_burden",
    "state_vector_v1.liquidity_flexibility",
    "state_vector_v1.interest_coverage",
)

_DEBT_ISSUANCE_BORROWER_FEATURE_WEIGHTS: Dict[str, float] = {
    "state_vector_v1.profitability": 1.10,
    "state_vector_v1.cash_generation": 1.20,
    "state_vector_v1.gross_obligation_burden": 1.25,
    "state_vector_v1.net_obligation_burden": 1.35,
    "state_vector_v1.interest_coverage": 1.25,
    "state_vector_v1.valuation_multiple": 0.90,
    "state_vector_v1.market_access": 1.10,
    "state_vector_v1.market_stress": 0.90,
    "state_vector_v1.rates_level": 0.85,
    "state_vector_v1.credit_spread": 0.95,
}
_DEBT_ISSUANCE_BORROWER_FEATURE_FALLBACK_SCALES: Dict[str, float] = {
    "state_vector_v1.profitability": 0.08,
    "state_vector_v1.cash_generation": 0.05,
    "state_vector_v1.gross_obligation_burden": 1.25,
    "state_vector_v1.net_obligation_burden": 1.00,
    "state_vector_v1.interest_coverage": 4.0,
    "state_vector_v1.valuation_multiple": 12.0,
    "state_vector_v1.market_access": 0.18,
    "state_vector_v1.market_stress": 0.12,
    "state_vector_v1.rates_level": 1.00,
    "state_vector_v1.credit_spread": 1.00,
}
_REVOLVER_SUPPORT_FEATURE_WEIGHTS: Dict[str, float] = {
    "state_vector_v1.profitability": 1.00,
    "state_vector_v1.cash_generation": 1.05,
    "state_vector_v1.gross_obligation_burden": 1.10,
    "state_vector_v1.net_obligation_burden": 1.20,
    "state_vector_v1.liquidity_flexibility": 1.55,
    "state_vector_v1.interest_coverage": 1.25,
    "state_vector_v1.valuation_multiple": 0.45,
    "state_vector_v1.market_access": 1.25,
    "state_vector_v1.market_stress": 1.35,
    "state_vector_v1.rates_level": 0.75,
    "state_vector_v1.credit_spread": 1.15,
}
_REVOLVER_SUPPORT_FEATURE_FALLBACK_SCALES: Dict[str, float] = {
    "state_vector_v1.profitability": 0.08,
    "state_vector_v1.cash_generation": 0.05,
    "state_vector_v1.gross_obligation_burden": 1.20,
    "state_vector_v1.net_obligation_burden": 1.00,
    "state_vector_v1.liquidity_flexibility": 1.20,
    "state_vector_v1.interest_coverage": 4.00,
    "state_vector_v1.valuation_multiple": 12.00,
    "state_vector_v1.market_access": 0.16,
    "state_vector_v1.market_stress": 0.10,
    "state_vector_v1.rates_level": 0.90,
    "state_vector_v1.credit_spread": 0.90,
}
_DEBT_ISSUANCE_ARCHETYPE_LABELS: tuple[str, ...] = (
    "distressed_borrower",
    "refinancing_pressure",
    "opportunistic_issuer",
)


def _bounded_sigmoid(value: float) -> float:
    clipped = max(-12.0, min(12.0, float(value)))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _numeric_feature_value(features: Dict[str, Any], feature_name: str) -> Optional[float]:
    try:
        value = features.get(feature_name)
        if value is None:
            return None
        numeric = float(value)
    except Exception:
        return None
    if not pd.notna(numeric):
        return None
    return float(numeric)


def _is_revolver_draw_or_resize_action(action_id: str) -> bool:
    return str(action_id or "").strip().lower() == "capital_structure.revolver_draw_or_resize"


def _same_action_prefers_cross_company(action_id: str) -> bool:
    return _is_revolver_draw_or_resize_action(action_id)


def _same_action_target_company_cap(action_id: str) -> int:
    return 1 if _is_revolver_draw_or_resize_action(action_id) else 0


def _same_action_company_cap(action_id: str) -> int:
    return 2 if _is_revolver_draw_or_resize_action(action_id) else 0


def _same_action_regime_requires_latent_model(action_id: str) -> bool:
    action_text = str(action_id or "").strip().lower()
    if action_text in {
        "capital_structure.new_debt_issuance",
        "capital_structure.revolver_draw_or_resize",
    }:
        # These actions have explicit regime heuristics, so the heavy latent
        # clustering pass is unnecessary for regime-aware teacher mining.
        return False
    return True


def _limit_same_action_company_repeats(
    matches: Iterable[Dict[str, Any]],
    *,
    per_company_cap: int,
    target_company_id: str = "",
    target_company_cap: int = 0,
) -> List[Dict[str, Any]]:
    if per_company_cap <= 0 and target_company_cap <= 0:
        return list(matches or [])
    counts: Dict[str, int] = {}
    limited: List[Dict[str, Any]] = []
    for match in list(matches or []):
        company_id = str((match or {}).get("company_id") or "").strip()
        effective_cap = int(per_company_cap)
        if target_company_id and company_id and company_id == str(target_company_id or "") and int(target_company_cap) > 0:
            effective_cap = int(target_company_cap)
        if effective_cap > 0 and company_id and counts.get(company_id, 0) >= effective_cap:
            continue
        limited.append(match)
        if company_id:
            counts[company_id] = counts.get(company_id, 0) + 1
    return limited


def _debt_issuance_market_regime_similarity(
    *,
    target_compact: Dict[str, Any],
    candidate_features: Dict[str, Any],
) -> float:
    similarities: List[float] = []
    specs = (
        ("state_vector_v1.rates_level", 0.25, 0.55),
        ("state_vector_v1.credit_spread", 0.35, 0.60),
        ("state_vector_v1.market_access", 0.00, 0.18),
        ("state_vector_v1.market_stress", 0.00, 0.12),
    )
    for feature_name, threshold, scale in specs:
        target_value = _numeric_feature_value(target_compact, feature_name)
        candidate_value = _numeric_feature_value(candidate_features, feature_name)
        if target_value is None or candidate_value is None:
            continue
        gap = abs(float(target_value) - float(candidate_value))
        similarity = np.exp(-max(gap - float(threshold), 0.0) / max(float(scale), 1e-9))
        similarities.append(float(similarity))
    if not similarities:
        return 1.0
    return float(np.exp(np.mean(np.log(np.clip(np.asarray(similarities, dtype=float), 1e-9, 1.0)))))


def _debt_issuance_archetype_profile(
    *,
    compact_features: Dict[str, Any],
    action_id: str = "capital_structure.new_debt_issuance",
    action_scale: Optional[float] = None,
) -> Dict[str, Any]:
    action_text = str(action_id or "").strip().lower()
    profitability = _numeric_feature_value(compact_features, "state_vector_v1.profitability")
    cash_generation = _numeric_feature_value(compact_features, "state_vector_v1.cash_generation")
    growth = _numeric_feature_value(compact_features, "state_vector_v1.growth")
    gross_burden = _numeric_feature_value(compact_features, "state_vector_v1.gross_obligation_burden")
    net_burden = _numeric_feature_value(compact_features, "state_vector_v1.net_obligation_burden")
    interest_coverage = _numeric_feature_value(compact_features, "state_vector_v1.interest_coverage")
    valuation_multiple = _numeric_feature_value(compact_features, "state_vector_v1.valuation_multiple")
    liquidity_flexibility = _numeric_feature_value(compact_features, "state_vector_v1.liquidity_flexibility")
    market_access = _numeric_feature_value(compact_features, "state_vector_v1.market_access")
    market_stress = _numeric_feature_value(compact_features, "state_vector_v1.market_stress")
    credit_spread = _numeric_feature_value(compact_features, "state_vector_v1.credit_spread")
    scale_value = float(action_scale) if action_scale is not None and pd.notna(action_scale) else None

    def _maybe_score(value: Optional[float], *, threshold: float, scale: float, lower_is_worse: bool) -> Optional[float]:
        if value is None:
            return None
        signed = (threshold - float(value)) if lower_is_worse else (float(value) - threshold)
        return _bounded_sigmoid(signed / max(float(scale), 1e-9))

    if action_text == "capital_structure.revolver_draw_or_resize":
        distressed_components = [
            (_maybe_score(profitability, threshold=0.10, scale=0.06, lower_is_worse=True), 1.00),
            (_maybe_score(cash_generation, threshold=0.00, scale=0.04, lower_is_worse=True), 1.10),
            (_maybe_score(interest_coverage, threshold=3.00, scale=1.50, lower_is_worse=True), 1.15),
            (_maybe_score(net_burden, threshold=1.60, scale=0.95, lower_is_worse=False), 1.10),
            (_maybe_score(gross_burden, threshold=2.50, scale=1.05, lower_is_worse=False), 0.95),
            (_maybe_score(liquidity_flexibility, threshold=1.10, scale=0.60, lower_is_worse=True), 1.45),
            (_maybe_score(market_access, threshold=0.66, scale=0.12, lower_is_worse=True), 1.20),
            (_maybe_score(market_stress, threshold=0.22, scale=0.08, lower_is_worse=False), 1.10),
            (_maybe_score(credit_spread, threshold=3.60, scale=0.85, lower_is_worse=False), 1.00),
        ]
        distressed_numer = sum(score * weight for score, weight in distressed_components if score is not None)
        distressed_denom = sum(weight for score, weight in distressed_components if score is not None)
        distressed_score = float(distressed_numer / distressed_denom) if distressed_denom > 0.0 else 0.5

        refinancing_components = [
            (_maybe_score(liquidity_flexibility, threshold=1.55, scale=0.85, lower_is_worse=True), 1.30),
            (_maybe_score(gross_burden, threshold=1.90, scale=0.95, lower_is_worse=False), 1.00),
            (_maybe_score(net_burden, threshold=1.10, scale=0.85, lower_is_worse=False), 1.05),
            (_maybe_score(interest_coverage, threshold=4.00, scale=2.00, lower_is_worse=True), 0.80),
            (_maybe_score(market_access, threshold=0.76, scale=0.15, lower_is_worse=True), 0.90),
            (_maybe_score(market_stress, threshold=0.18, scale=0.08, lower_is_worse=False), 0.75),
        ]
        if scale_value is not None:
            refinancing_components.append(
                (_maybe_score(scale_value, threshold=0.08, scale=0.05, lower_is_worse=False), 1.10)
            )
        refi_numer = sum(score * weight for score, weight in refinancing_components if score is not None)
        refi_denom = sum(weight for score, weight in refinancing_components if score is not None)
        refinancing_pressure_score = float(refi_numer / refi_denom) if refi_denom > 0.0 else 0.5

        opportunistic_components = [
            (_maybe_score(profitability, threshold=0.16, scale=0.07, lower_is_worse=False), 1.15),
            (_maybe_score(cash_generation, threshold=0.01, scale=0.04, lower_is_worse=False), 1.10),
            (_maybe_score(growth, threshold=0.05, scale=0.12, lower_is_worse=False), 0.70),
            (_maybe_score(interest_coverage, threshold=5.50, scale=2.50, lower_is_worse=False), 1.00),
            (_maybe_score(liquidity_flexibility, threshold=1.80, scale=1.00, lower_is_worse=False), 1.00),
            (_maybe_score(market_access, threshold=0.80, scale=0.12, lower_is_worse=False), 1.15),
            (_maybe_score(market_stress, threshold=0.16, scale=0.08, lower_is_worse=True), 1.00),
            (_maybe_score(credit_spread, threshold=3.20, scale=0.75, lower_is_worse=True), 0.90),
            (_maybe_score(net_burden, threshold=2.20, scale=1.20, lower_is_worse=True), 0.75),
        ]
        opp_numer = sum(score * weight for score, weight in opportunistic_components if score is not None)
        opp_denom = sum(weight for score, weight in opportunistic_components if score is not None)
        opportunistic_score = float(opp_numer / opp_denom) if opp_denom > 0.0 else 0.5

        scores = {
            "distressed_borrower": distressed_score,
            "refinancing_pressure": refinancing_pressure_score,
            "opportunistic_issuer": opportunistic_score,
        }
        if distressed_score >= 0.60 and distressed_score >= opportunistic_score + 0.06:
            label = "distressed_borrower"
        elif opportunistic_score >= 0.60 and opportunistic_score >= distressed_score + 0.06:
            label = "opportunistic_issuer"
        else:
            label = max(scores.items(), key=lambda item: item[1])[0]
        return {
            "label": str(label),
            "scores": scores,
        }

    distressed_components = [
        (_maybe_score(profitability, threshold=0.12, scale=0.06, lower_is_worse=True), 1.10),
        (_maybe_score(cash_generation, threshold=0.00, scale=0.04, lower_is_worse=True), 1.20),
        (_maybe_score(interest_coverage, threshold=3.00, scale=1.50, lower_is_worse=True), 1.20),
        (_maybe_score(net_burden, threshold=1.50, scale=1.00, lower_is_worse=False), 1.20),
        (_maybe_score(gross_burden, threshold=2.40, scale=1.10, lower_is_worse=False), 1.05),
        (_maybe_score(market_access, threshold=0.70, scale=0.14, lower_is_worse=True), 1.15),
        (_maybe_score(market_stress, threshold=0.20, scale=0.10, lower_is_worse=False), 0.85),
        (_maybe_score(credit_spread, threshold=3.00, scale=0.90, lower_is_worse=False), 0.85),
        (_maybe_score(valuation_multiple, threshold=7.00, scale=4.00, lower_is_worse=True), 0.55),
    ]
    distressed_numer = sum(score * weight for score, weight in distressed_components if score is not None)
    distressed_denom = sum(weight for score, weight in distressed_components if score is not None)
    distressed_score = float(distressed_numer / distressed_denom) if distressed_denom > 0.0 else 0.5

    refinancing_components = [
        (_maybe_score(liquidity_flexibility, threshold=1.50, scale=0.75, lower_is_worse=True), 1.25),
        (_maybe_score(gross_burden, threshold=2.00, scale=1.00, lower_is_worse=False), 1.05),
        (_maybe_score(net_burden, threshold=1.00, scale=0.90, lower_is_worse=False), 1.10),
        (_maybe_score(interest_coverage, threshold=4.00, scale=2.00, lower_is_worse=True), 0.80),
        (_maybe_score(market_access, threshold=0.78, scale=0.16, lower_is_worse=True), 0.70),
    ]
    if scale_value is not None:
        refinancing_components.append((_maybe_score(scale_value, threshold=0.08, scale=0.05, lower_is_worse=False), 1.20))
    refi_numer = sum(score * weight for score, weight in refinancing_components if score is not None)
    refi_denom = sum(weight for score, weight in refinancing_components if score is not None)
    refinancing_pressure_score = float(refi_numer / refi_denom) if refi_denom > 0.0 else 0.5

    opportunistic_components = [
        (_maybe_score(profitability, threshold=0.18, scale=0.07, lower_is_worse=False), 1.15),
        (_maybe_score(cash_generation, threshold=0.01, scale=0.04, lower_is_worse=False), 1.15),
        (_maybe_score(interest_coverage, threshold=6.00, scale=3.00, lower_is_worse=False), 1.10),
        (_maybe_score(market_access, threshold=0.82, scale=0.12, lower_is_worse=False), 1.20),
        (_maybe_score(market_stress, threshold=0.14, scale=0.10, lower_is_worse=True), 0.85),
        (_maybe_score(credit_spread, threshold=3.00, scale=0.80, lower_is_worse=True), 0.95),
        (_maybe_score(net_burden, threshold=2.50, scale=1.40, lower_is_worse=True), 0.80),
        (_maybe_score(valuation_multiple, threshold=10.0, scale=6.0, lower_is_worse=False), 0.55),
    ]
    opp_numer = sum(score * weight for score, weight in opportunistic_components if score is not None)
    opp_denom = sum(weight for score, weight in opportunistic_components if score is not None)
    opportunistic_score = float(opp_numer / opp_denom) if opp_denom > 0.0 else 0.5

    scores = {
        "distressed_borrower": distressed_score,
        "refinancing_pressure": refinancing_pressure_score,
        "opportunistic_issuer": opportunistic_score,
    }
    label = "refinancing_pressure"
    if distressed_score >= 0.58 and distressed_score >= opportunistic_score + 0.08:
        label = "distressed_borrower"
    elif opportunistic_score >= 0.58 and opportunistic_score >= distressed_score + 0.08:
        label = "opportunistic_issuer"
    else:
        label = max(scores.items(), key=lambda item: item[1])[0]
    return {
        "label": str(label),
        "scores": scores,
    }


def _debt_issuance_archetype_distance(
    *,
    action_id: str,
    target_compact: Dict[str, Any],
    candidate_features: Dict[str, Any],
    target_action_scale: Optional[float] = None,
    candidate_action_scale: Optional[float] = None,
) -> float:
    target_profile = _debt_issuance_archetype_profile(
        compact_features=target_compact,
        action_id=action_id,
        action_scale=target_action_scale,
    )
    candidate_profile = _debt_issuance_archetype_profile(
        compact_features=candidate_features,
        action_id=action_id,
        action_scale=candidate_action_scale,
    )
    target_scores = dict(target_profile.get("scores") or {})
    candidate_scores = dict(candidate_profile.get("scores") or {})
    shared_labels = [label for label in _DEBT_ISSUANCE_ARCHETYPE_LABELS if label in target_scores and label in candidate_scores]
    if not shared_labels:
        return 0.0
    score_distance = float(
        np.mean(
            [
                abs(float(target_scores[label]) - float(candidate_scores[label]))
                for label in shared_labels
            ]
        )
    )
    target_label = str(target_profile.get("label") or "")
    candidate_label = str(candidate_profile.get("label") or "")
    label_penalty = 0.0
    if target_label and candidate_label and target_label != candidate_label:
        label_pair = {target_label, candidate_label}
        if label_pair == {"distressed_borrower", "opportunistic_issuer"}:
            label_penalty = 1.00
        elif "refinancing_pressure" in label_pair:
            label_penalty = 0.55
        else:
            label_penalty = 0.75
    return float(score_distance + label_penalty)


def _mean_abs_diff(target_compact: Dict[str, Any], candidate_features: Dict[str, Any], feature_names: Iterable[str]) -> float:
    diffs: List[float] = []
    for feature in feature_names:
        diff = _absdiff(target_compact.get(feature), candidate_features.get(feature))
        if diff is not None:
            diffs.append(float(diff))
    if not diffs:
        return float("inf")
    return float(sum(diffs) / len(diffs))


def _action_specific_same_action_distance(
    *,
    action_id: str,
    target_compact: Dict[str, Any],
    candidate_features: Dict[str, Any],
    feature_scales: Optional[Dict[str, Any]] = None,
    target_action_scale: Optional[float] = None,
    candidate_action_scale: Optional[float] = None,
) -> float:
    action_text = str(action_id or "").strip().lower()
    if action_text not in {"capital_structure.new_debt_issuance", "capital_structure.revolver_draw_or_resize"}:
        return float("inf")
    if action_text == "capital_structure.revolver_draw_or_resize":
        feature_weights = _REVOLVER_SUPPORT_FEATURE_WEIGHTS
        fallback_scales = _REVOLVER_SUPPORT_FEATURE_FALLBACK_SCALES
        archetype_weight = 1.05
        regime_floor = 0.78
        regime_weight = 0.90
        action_scale_weight = 0.35
    else:
        feature_weights = _DEBT_ISSUANCE_BORROWER_FEATURE_WEIGHTS
        fallback_scales = _DEBT_ISSUANCE_BORROWER_FEATURE_FALLBACK_SCALES
        archetype_weight = 0.90
        regime_floor = 0.72
        regime_weight = 0.60
        action_scale_weight = 0.20
    numer = 0.0
    denom = 0.0
    for feature, weight in feature_weights.items():
        diff = _absdiff(target_compact.get(feature), candidate_features.get(feature))
        if diff is None or not pd.notna(diff):
            continue
        scale_value = None
        if isinstance(feature_scales, dict):
            scale_value = _first([feature_scales.get(feature)], None)
        try:
            scale = float(scale_value) if scale_value is not None else None
        except Exception:
            scale = None
        if scale is not None and pd.isna(scale):
            scale = None
        if scale is None or not pd.notna(scale) or scale <= 1e-9:
            scale = float(fallback_scales.get(feature, 1.0))
        numer += float(weight) * float(diff) / max(float(scale), 1e-9)
        denom += float(weight)
    if denom <= 1e-12:
        return float("inf")
    base_distance = float(numer / denom)
    archetype_distance = _debt_issuance_archetype_distance(
        action_id=action_text,
        target_compact=target_compact,
        candidate_features=candidate_features,
        target_action_scale=target_action_scale,
        candidate_action_scale=candidate_action_scale,
    )
    market_regime_similarity = _debt_issuance_market_regime_similarity(
        target_compact=target_compact,
        candidate_features=candidate_features,
    )
    action_scale_distance = _action_scale_gap(target_action_scale, candidate_action_scale)
    if not pd.notna(action_scale_distance):
        action_scale_distance = 0.0
    return float(
        base_distance
        + archetype_weight * float(archetype_distance)
        + regime_weight * max(0.0, regime_floor - float(market_regime_similarity))
        + action_scale_weight * min(float(action_scale_distance), 2.0)
    )


def _match_taxonomy_rank(
    match_features: Dict[str, Any],
    *,
    target_sector: str,
    target_subsector: str,
    taxonomy_mode: str,
) -> tuple[int, int]:
    match_sector = str(match_features.get("sector") or match_features.get("base_sector") or "").strip()
    match_subsector = str(match_features.get("subsector") or "").strip()
    same_sector = bool(target_sector and match_sector and target_sector == match_sector)
    same_subsector = bool(target_subsector and match_subsector and target_subsector == match_subsector)
    if taxonomy_mode == "prefer_same_subsector_then_sector":
        return (0 if same_subsector else 1, 0 if same_sector else 1)
    if taxonomy_mode == "prefer_same_sector":
        return (0 if same_sector else 1, 0)
    return (0, 0)


def _rank_hard_negative_matches(
    matches: List[Dict[str, Any]],
    *,
    target_compact: Dict[str, Any],
    target_sector: str,
    target_subsector: str,
    taxonomy_mode: str,
) -> List[Dict[str, Any]]:
    ranked = list(matches or [])
    ranked.sort(
        key=lambda match: (
            *_match_taxonomy_rank(
                dict(match.get("key_state_features") or {}),
                target_sector=target_sector,
                target_subsector=target_subsector,
                taxonomy_mode=taxonomy_mode,
            ),
            _mean_abs_diff(
                target_compact,
                dict(match.get("key_state_features") or {}),
                _HARD_NEGATIVE_SAFETY_FEATURES,
            ),
            -float(match.get("similarity_score") or 0.0),
        )
    )
    return ranked


def _action_scale_gap(target_action_scale: Optional[float], candidate_action_scale: Any) -> float:
    try:
        target_value = float(target_action_scale)
        candidate_value = float(candidate_action_scale)
    except Exception:
        return float("inf")
    if not pd.notna(target_value) or not pd.notna(candidate_value):
        return float("inf")
    return float(abs(np.log1p(max(target_value, 0.0)) - np.log1p(max(candidate_value, 0.0))))


def _rank_same_action_hard_confusers(
    matches: List[Dict[str, Any]],
    *,
    action_id: str,
    target_compact: Dict[str, Any],
    target_sector: str,
    target_subsector: str,
    target_action_scale: Optional[float],
) -> List[Dict[str, Any]]:
    ranked = list(matches or [])
    target_profile = _debt_issuance_archetype_profile(
        compact_features=target_compact,
        action_id=action_id,
        action_scale=target_action_scale,
    )
    target_label = str(target_profile.get("label") or "")

    def _match_action_scale(match: Dict[str, Any]) -> Optional[float]:
        try:
            numeric = float(match.get("action_scale"))
        except Exception:
            return None
        if not pd.notna(numeric):
            return None
        return float(numeric)

    def _confuser_priority(match: Dict[str, Any]) -> tuple[int, float]:
        candidate_features = dict(match.get("key_state_features") or {})
        candidate_scale = _match_action_scale(match)
        candidate_profile = _debt_issuance_archetype_profile(
            compact_features=candidate_features,
            action_id=action_id,
            action_scale=candidate_scale,
        )
        candidate_label = str(candidate_profile.get("label") or "")
        regime_similarity = _debt_issuance_market_regime_similarity(
            target_compact=target_compact,
            candidate_features=candidate_features,
        )
        if target_label and candidate_label and candidate_label != target_label and regime_similarity >= 0.64:
            return (0, -regime_similarity)
        if target_label and candidate_label and candidate_label == target_label and regime_similarity < 0.58:
            return (1, regime_similarity)
        if target_label and candidate_label and candidate_label != target_label:
            return (2, -regime_similarity)
        return (3, regime_similarity)

    ranked.sort(
        key=lambda match: (
            *_match_taxonomy_rank(
                dict(match.get("key_state_features") or {}),
                target_sector=target_sector,
                target_subsector=target_subsector,
                taxonomy_mode="prefer_same_subsector_then_sector",
            ),
            *_confuser_priority(match),
            _action_scale_gap(target_action_scale, match.get("action_scale")),
            _action_specific_same_action_distance(
                action_id=action_id,
                target_compact=target_compact,
                candidate_features=dict(match.get("key_state_features") or {}),
                target_action_scale=target_action_scale,
                candidate_action_scale=_match_action_scale(match),
            ),
            float(match.get("analog_distance")) if pd.notna(match.get("analog_distance")) else float("inf"),
            _mean_abs_diff(
                target_compact,
                dict(match.get("key_state_features") or {}),
                _HARD_NEGATIVE_SAFETY_FEATURES,
            ),
            -float(match.get("similarity_score") or 0.0),
        )
    )
    return ranked


def _same_action_negative_pool_limit(top_k: int, anchor_matches: List[Dict[str, Any]]) -> int:
    positive_count = max(1, len(list(anchor_matches or [])))
    return max(int(max(1, top_k)), min(24, positive_count * 4))


def _same_action_ordering_window(top_k: int) -> int:
    return max(2, min(8, max(2, int(top_k // 2) if int(top_k) > 0 else 4)))


def _select_actual_anchor_outcome(
    case: Dict[str, Any],
    *,
    anchor_outcomes_lookup: Dict[tuple[str, str], List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    company_id = str(case.get("source_company_id") or case.get("company_id") or "").strip()
    action_id = str(case.get("anchor_action_id") or "").strip()
    rows = list(anchor_outcomes_lookup.get((company_id, action_id), []) or [])
    if not rows:
        return None
    anchor_dt = pd.to_datetime(case.get("anchor_action_date"), utc=True, errors="coerce")
    anchor_raw_subtype = _case_anchor_action_subtype(case).lower()
    anchor_effective_subtype = _case_anchor_effective_action_subtype(case)
    if pd.isna(anchor_dt) and not anchor_raw_subtype and not anchor_effective_subtype:
        return rows[0]
    ranked = []
    for row in rows:
        row_raw_subtype = str(row.get("raw_action_subtype") or row.get("action_subtype") or "").strip().lower()
        exact_penalty = int(bool(anchor_raw_subtype) and row_raw_subtype != anchor_raw_subtype)
        row_effective_subtype = _row_effective_action_subtype(action_id, row)
        family_penalty = int(bool(anchor_effective_subtype) and row_effective_subtype != anchor_effective_subtype)
        action_dt = pd.to_datetime(row.get("action_date"), utc=True, errors="coerce")
        delta = (
            abs((action_dt - anchor_dt).total_seconds())
            if pd.notna(anchor_dt) and pd.notna(action_dt)
            else float("inf")
        )
        try:
            action_size = float(row.get("action_size"))
        except Exception:
            action_size = 0.0
        if pd.isna(action_size):
            action_size = 0.0
        ranked.append(((exact_penalty, family_penalty, delta, -action_size), row))
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1] if ranked else None


def _actual_anchor_match_payload(case: Dict[str, Any], outcome_row: Dict[str, Any]) -> Dict[str, Any]:
    precedent_id = str(
        outcome_row.get("precedent_id")
        or f"{outcome_row.get('company_id')}::{outcome_row.get('action_date')}::actual_anchor"
    )
    feature_values = {feature: outcome_row.get(feature) for feature in _STATE_VECTOR_V1_FEATURES}
    return {
        "precedent_id": precedent_id,
        "company_id": str(outcome_row.get("company_id") or ""),
        "similarity_score": 1.0,
        "key_state_features": feature_values,
        "sector": outcome_row.get("sector"),
        "subsector": outcome_row.get("subsector"),
    }


def _enrich_match_compact(
    match: Dict[str, Any],
    *,
    precedent_outcomes_lookup: Dict[tuple[str, str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    key_state_features = dict(match.get("key_state_features") or {})
    lookup_key = (
        str(match.get("company_id") or "").strip(),
        str(match.get("action_id") or "").strip(),
        _normalize_as_of_time(str(match.get("decision_time") or "")),
    )
    outcome_row = precedent_outcomes_lookup.get(lookup_key)
    if outcome_row is None:
        return key_state_features
    enriched = dict(key_state_features)
    for feature in _STATE_VECTOR_V1_FEATURES:
        if enriched.get(feature) is None and outcome_row.get(feature) is not None:
            enriched[feature] = outcome_row.get(feature)
    for feature in (
        "base_sector",
        "sector",
        "subsector",
        "base_revenue_ttm",
        "base_revenue_ttm_lag_1y",
        "base_revenue_growth_yoy",
        "base_ebitda_ttm",
        "base_total_debt",
        "base_current_debt",
        "base_cash",
        "base_available_liquidity",
        "base_interest_expense",
        "base_market_cap",
        "base_ev_ebitda",
        "base_fcf_yield",
        "base_volatility_30d",
        "base_volatility_90d",
        "base_drawdown_90d",
        "base_credit_spread_level",
        "base_equity_window_proxy",
        "base_credit_window_proxy",
        "base_net_debt",
        "base_leverage",
        "base_margin",
        "macro_fed_funds_effective",
        "macro_hy_oas",
        "macro_real_gdp_growth_yoy",
        "macro_vix",
    ):
        if enriched.get(feature) is None and outcome_row.get(feature) is not None:
            enriched[feature] = outcome_row.get(feature)
    return enriched


def _match_identity_key(match: Dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(match.get("precedent_id") or "").strip(),
        str(match.get("company_id") or "").strip(),
        str(match.get("action_id") or "").strip(),
        _normalize_as_of_time(str(match.get("decision_time") or "")),
    )


