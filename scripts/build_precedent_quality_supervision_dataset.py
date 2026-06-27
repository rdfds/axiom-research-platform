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


def _dedupe_matches(matches: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for raw_match in matches:
        match = dict(raw_match or {})
        identity = _match_identity_key(match)
        if identity in seen:
            continue
        seen.add(identity)
        ordered.append(match)
    return ordered


def _outcome_row_taxonomy(row: Dict[str, Any]) -> Dict[str, str]:
    return {
        "sector": str(
            row.get("taxonomy.sector")
            or row.get("sector")
            or row.get("base_sector")
            or row.get("gics_sector")
            or ""
        ).strip(),
        "subsector": str(
            row.get("taxonomy.subsector")
            or row.get("subsector")
            or row.get("industry")
            or row.get("base_industry")
            or ""
        ).strip(),
    }


def _normalize_ticker_key(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_instrument_root(value: Any) -> str:
    return _normalize_ticker_key(value).split(".", 1)[0].strip()


@lru_cache(maxsize=1)
def _load_direct_refinitiv_taxonomy_lookup() -> Dict[str, Dict[str, str]]:
    path = _REFINITIV_TAXONOMY_REFERENCE_PATH
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(path, columns=["Instrument", "GICS Sector Name", "GICS Industry Name"])
    except Exception:
        return {}
    lookup: Dict[str, Dict[str, str]] = {}
    for instrument, sector_name, subsector_name in zip(
        frame.get("Instrument", pd.Series("", index=frame.index)),
        frame.get("GICS Sector Name", pd.Series("", index=frame.index)),
        frame.get("GICS Industry Name", pd.Series("", index=frame.index)),
    ):
        ticker_key = _normalize_instrument_root(instrument)
        if not ticker_key:
            continue
        sector_text = str(sector_name or "").strip()
        subsector_text = str(subsector_name or "").strip()
        if not sector_text and not subsector_text:
            continue
        existing = lookup.get(ticker_key)
        if existing and existing.get("sector") and existing.get("subsector"):
            continue
        lookup[ticker_key] = {
            "sector": sector_text,
            "subsector": subsector_text,
        }
    return lookup


@lru_cache(maxsize=1)
def _load_direct_sec_ticker_cik_lookup() -> Dict[str, str]:
    if str(os.environ.get("RECO_DISABLE_DIRECT_SEC_TICKER_CIK_LOOKUP") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return {}
    path = _SEC_TICKER_CIK_PATH
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(path, columns=["ticker", "cik"])
    except Exception:
        return {}
    lookup: Dict[str, str] = {}
    for ticker, cik in zip(
        frame.get("ticker", pd.Series("", index=frame.index)),
        frame.get("cik", pd.Series("", index=frame.index)),
    ):
        ticker_key = _normalize_ticker_key(ticker)
        cik_text = str(cik or "").strip()
        if cik_text.endswith(".0"):
            cik_text = cik_text[:-2]
        if cik_text.isdigit():
            cik_text = cik_text.zfill(10)
        if ticker_key and cik_text and ticker_key not in lookup:
            lookup[ticker_key] = cik_text
    return lookup


@lru_cache(maxsize=1)
def _load_direct_snapshot_taxonomy_lookup() -> Dict[str, Dict[str, str]]:
    path = _SNAPSHOT_TAXONOMY_LOOKUP_PATH
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(path, columns=["company_id", "taxonomy.sector", "taxonomy.subsector"])
    except Exception:
        return {}
    lookup: Dict[str, Dict[str, str]] = {}
    for company_id, sector_name, subsector_name in zip(
        frame.get("company_id", pd.Series("", index=frame.index)),
        frame.get("taxonomy.sector", pd.Series("", index=frame.index)),
        frame.get("taxonomy.subsector", pd.Series("", index=frame.index)),
    ):
        company_key = str(company_id or "").strip()
        if not company_key:
            continue
        sector_text = str(sector_name or "").strip()
        subsector_text = str(subsector_name or "").strip()
        if not sector_text and not subsector_text:
            continue
        lookup[company_key] = {
            "sector": sector_text,
            "subsector": subsector_text,
        }
    return lookup


def _direct_historical_ticker_taxonomy(ticker: str) -> Dict[str, str]:
    if str(os.environ.get("RECO_DISABLE_DIRECT_HISTORICAL_TAXONOMY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return {}
    payload = dict(_historical_taxonomy_for_ticker(ticker, allow_sec_identity_heuristics=True) or {})
    return {
        "sector": str(payload.get("taxonomy.sector") or "").strip(),
        "subsector": str(payload.get("taxonomy.subsector") or "").strip(),
    }


def _allow_knn_target_taxonomy_inference(action_id: str) -> bool:
    action_text = str(action_id or "").strip().lower()
    if action_text == "capital_structure.equity_issuance":
        # Numeric state neighbors are useful for equity-issuance retrieval, but
        # they are too coarse to impute sector labels safely for the weird
        # small-cap raise cases. Only trust exact history or direct ticker
        # taxonomy for this action.
        return False
    return True


def _infer_target_taxonomy_from_same_action_universe(
    case: Dict[str, Any],
    *,
    same_action_universe_lookup: Dict[str, Dict[str, Any]],
    target_compact: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    action_id = str(case.get("anchor_action_id") or "").strip()
    if not action_id:
        return {}
    payload = dict(same_action_universe_lookup.get(action_id) or {})
    rows = list(payload.get("rows") or [])
    if not rows:
        return {}

    company_id = str(case.get("source_company_id") or case.get("company_id") or "").strip()
    ticker = str(case.get("ticker") or "").strip().upper()

    matches = [
        row
        for row in rows
        if company_id and str(row.get("company_id") or "").strip() == company_id
    ]
    if not matches and ticker:
        matches = [
            row
            for row in rows
            if str(row.get("ticker") or "").strip().upper() == ticker
        ]
    if matches:
        pair_votes: Counter[tuple[str, str]] = Counter()
        sector_votes: Counter[str] = Counter()
        subsector_votes: Counter[str] = Counter()
        for row in matches:
            taxonomy = _outcome_row_taxonomy(row)
            sector = str(taxonomy.get("sector") or "").strip()
            subsector = str(taxonomy.get("subsector") or "").strip()
            if sector:
                sector_votes[sector] += 1
            if subsector:
                subsector_votes[subsector] += 1
            if sector or subsector:
                pair_votes[(sector, subsector)] += 1

        best_sector = sector_votes.most_common(1)[0][0] if sector_votes else ""
        best_subsector = subsector_votes.most_common(1)[0][0] if subsector_votes else ""
        if pair_votes:
            best_pair, _ = pair_votes.most_common(1)[0]
            if best_pair[0]:
                best_sector = best_pair[0]
            if best_pair[1]:
                best_subsector = best_pair[1]
        exact_match_taxonomy = {
            "sector": str(best_sector or "").strip(),
            "subsector": str(best_subsector or "").strip(),
        }
        if exact_match_taxonomy["sector"] or exact_match_taxonomy["subsector"]:
            return exact_match_taxonomy
    if ticker:
        direct_payload = dict(_direct_historical_ticker_taxonomy(ticker) or {})
        direct_taxonomy = {
            "sector": str(direct_payload.get("sector") or "").strip(),
            "subsector": str(direct_payload.get("subsector") or "").strip(),
        }
        if direct_taxonomy["sector"] or direct_taxonomy["subsector"]:
            return direct_taxonomy
    if not _allow_knn_target_taxonomy_inference(action_id):
        return {}

    target_compact = dict(target_compact or {})
    if not target_compact:
        return {}

    feature_scales = dict(payload.get("feature_scales") or {})
    weighted_pair_votes: Dict[tuple[str, str], float] = {}
    weighted_sector_votes: Dict[str, float] = {}
    weighted_subsector_votes: Dict[str, float] = {}
    candidate_rows: List[tuple[float, int, str, str]] = []
    for row in rows:
        taxonomy = _outcome_row_taxonomy(row)
        sector = str(taxonomy.get("sector") or "").strip()
        subsector = str(taxonomy.get("subsector") or "").strip()
        if not sector and not subsector:
            continue
        distance, shared_features = _standardized_feature_distance(
            target_compact,
            row,
            feature_scales=feature_scales,
        )
        if not np.isfinite(distance) or int(shared_features) < 4:
            continue
        candidate_rows.append((float(distance), int(shared_features), sector, subsector))

    if not candidate_rows:
        return {}

    candidate_rows.sort(key=lambda item: (item[0], -item[1], item[2], item[3]))
    for distance, shared_features, sector, subsector in candidate_rows[:25]:
        weight = (float(shared_features) ** 2) / (1.0 + float(distance))
        if sector:
            weighted_sector_votes[sector] = weighted_sector_votes.get(sector, 0.0) + weight
        if subsector:
            weighted_subsector_votes[subsector] = weighted_subsector_votes.get(subsector, 0.0) + weight
        if sector or subsector:
            weighted_pair_votes[(sector, subsector)] = weighted_pair_votes.get((sector, subsector), 0.0) + weight

    best_sector = max(weighted_sector_votes.items(), key=lambda item: item[1])[0] if weighted_sector_votes else ""
    best_subsector = max(weighted_subsector_votes.items(), key=lambda item: item[1])[0] if weighted_subsector_votes else ""
    if weighted_pair_votes:
        best_pair = max(weighted_pair_votes.items(), key=lambda item: item[1])[0]
        if best_pair[0]:
            best_sector = best_pair[0]
        if best_pair[1]:
            best_subsector = best_pair[1]
    return {
        "sector": str(best_sector or "").strip(),
        "subsector": str(best_subsector or "").strip(),
    }


def _standardized_feature_distance(
    target_compact: Dict[str, Any],
    candidate_features: Dict[str, Any],
    *,
    feature_scales: Dict[str, float],
) -> tuple[float, int]:
    diffs: List[float] = []
    for feature in _STATE_VECTOR_V1_FEATURES:
        target_value = target_compact.get(feature)
        candidate_value = candidate_features.get(feature)
        diff = _absdiff(target_value, candidate_value)
        if diff is None:
            continue
        scale = float(feature_scales.get(feature) or 1.0)
        if not pd.notna(scale) or scale <= 1e-9:
            scale = 1.0
        diffs.append(float(diff) / scale)
    if not diffs:
        return float("inf"), 0
    return float(sum(diffs) / len(diffs)), int(len(diffs))


def _same_action_neighborhood_rows(
    rows: List[Dict[str, Any]],
    *,
    target_taxonomy: Dict[str, str],
    minimum_rows: int,
) -> tuple[List[Dict[str, Any]], str]:
    target_sector = str(target_taxonomy.get("sector") or "").strip()
    target_subsector = str(target_taxonomy.get("subsector") or "").strip()
    if not rows:
        return [], "same_action"
    same_subsector = [
        row
        for row in rows
        if target_subsector and _outcome_row_taxonomy(row).get("subsector") == target_subsector
    ]
    if len(same_subsector) >= int(max(1, minimum_rows)):
        return same_subsector, "same_subsector"
    same_sector = [
        row
        for row in rows
        if target_sector and _outcome_row_taxonomy(row).get("sector") == target_sector
    ]
    if len(same_sector) >= int(max(1, minimum_rows)):
        return same_sector, "same_sector"
    return list(rows), "same_action"


def _same_action_analog_match_payload(
    outcome_row: Dict[str, Any],
    *,
    distance: float,
    feature_count: int,
    taxonomy_mode: str,
    latent_regime_similarity: Optional[float] = None,
    latent_regime_cluster: Optional[int] = None,
    debt_archetype_label: Optional[str] = None,
    debt_market_regime_similarity: Optional[float] = None,
) -> Dict[str, Any]:
    action_time = _normalize_as_of_time(str(outcome_row.get("action_date") or ""))
    taxonomy = _outcome_row_taxonomy(outcome_row)
    action_params = _outcome_row_action_params(outcome_row)
    market_cap = _outcome_row_market_cap(outcome_row)
    feature_values = {feature: outcome_row.get(feature) for feature in _STATE_VECTOR_V1_FEATURES}
    feature_values["sector"] = taxonomy["sector"]
    feature_values["subsector"] = taxonomy["subsector"]
    precedent_id = str(
        outcome_row.get("precedent_id")
        or f"{outcome_row.get('company_id')}::{action_time}::{outcome_row.get('normalized_action_id')}::analog_consensus"
    )
    similarity_score = 0.0 if not pd.notna(distance) or distance < 0.0 else float(1.0 / (1.0 + distance))
    return {
        "precedent_id": precedent_id,
        "company_id": str(outcome_row.get("company_id") or ""),
        "ticker": str(outcome_row.get("ticker") or ""),
        "action_id": str(outcome_row.get("normalized_action_id") or ""),
        "decision_time": action_time,
        "similarity_score": similarity_score,
        "analog_distance": float(distance) if pd.notna(distance) else None,
        "analog_feature_count": int(feature_count),
        "analog_taxonomy_mode": str(taxonomy_mode or "same_action"),
        "analog_latent_regime_similarity": (
            float(latent_regime_similarity) if latent_regime_similarity is not None and pd.notna(latent_regime_similarity) else None
        ),
        "analog_latent_regime_cluster": int(latent_regime_cluster) if latent_regime_cluster is not None else None,
        "debt_archetype_label": str(debt_archetype_label or "") if debt_archetype_label else None,
        "debt_market_regime_similarity": (
            float(debt_market_regime_similarity)
            if debt_market_regime_similarity is not None and pd.notna(debt_market_regime_similarity)
            else None
        ),
        "action_params": action_params,
        "market_cap": market_cap,
        "action_scale": _estimate_action_scale(action_params, market_cap),
        "key_state_features": feature_values,
        "sector": taxonomy["sector"],
        "subsector": taxonomy["subsector"],
    }


def _same_action_model_confuser_match_payload(
    outcome_row: Dict[str, Any],
    *,
    similarity_score: float,
    taxonomy_mode: str,
    weighted_coverage: Optional[float] = None,
    critical_coverage: Optional[float] = None,
    analog_distance: Optional[float] = None,
    debt_archetype_label: Optional[str] = None,
    debt_market_regime_similarity: Optional[float] = None,
) -> Dict[str, Any]:
    action_time = _normalize_as_of_time(str(outcome_row.get("action_date") or ""))
    taxonomy = _outcome_row_taxonomy(outcome_row)
    action_params = _outcome_row_action_params(outcome_row)
    market_cap = _outcome_row_market_cap(outcome_row)
    feature_values = {feature: outcome_row.get(feature) for feature in _STATE_VECTOR_V1_FEATURES}
    feature_values["sector"] = taxonomy["sector"]
    feature_values["subsector"] = taxonomy["subsector"]
    precedent_id = str(
        outcome_row.get("precedent_id")
        or f"{outcome_row.get('company_id')}::{action_time}::{outcome_row.get('normalized_action_id')}::model_confuser"
    )
    return {
        "precedent_id": precedent_id,
        "company_id": str(outcome_row.get("company_id") or ""),
        "ticker": str(outcome_row.get("ticker") or ""),
        "action_id": str(outcome_row.get("normalized_action_id") or ""),
        "decision_time": action_time,
        "similarity_score": float(similarity_score) if pd.notna(similarity_score) else 0.0,
        "model_confuser_taxonomy_mode": str(taxonomy_mode or "same_action"),
        "model_confuser_weighted_coverage": (
            float(weighted_coverage) if weighted_coverage is not None and pd.notna(weighted_coverage) else None
        ),
        "model_confuser_critical_coverage": (
            float(critical_coverage) if critical_coverage is not None and pd.notna(critical_coverage) else None
        ),
        "analog_distance": float(analog_distance) if analog_distance is not None and pd.notna(analog_distance) else None,
        "debt_archetype_label": str(debt_archetype_label or "") if debt_archetype_label else None,
        "debt_market_regime_similarity": (
            float(debt_market_regime_similarity)
            if debt_market_regime_similarity is not None and pd.notna(debt_market_regime_similarity)
            else None
        ),
        "action_params": action_params,
        "market_cap": market_cap,
        "action_scale": _estimate_action_scale(action_params, market_cap),
        "key_state_features": feature_values,
        "sector": taxonomy["sector"],
        "subsector": taxonomy["subsector"],
    }


def _build_same_action_model_confuser_matches(
    *,
    case: Dict[str, Any],
    target_compact: Dict[str, Any],
    target_taxonomy: Dict[str, str],
    target_action_params: Optional[Dict[str, Any]],
    target_market_cap: Optional[float],
    top_k: int,
    positive_limit_per_source: int,
    negative_limit_per_competitor: int,
    same_action_universe_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    action_id = str(case.get("anchor_action_id") or "").strip()
    if not action_id:
        return []
    payload = dict(same_action_universe_lookup.get(action_id) or {})
    universe_rows = list(payload.get("rows") or [])
    if not universe_rows:
        return []

    feature_matrix = (
        np.asarray(payload.get("feature_matrix"), dtype=float)
        if payload.get("feature_matrix") is not None
        else np.column_stack(
            [
                np.asarray(
                    [pd.to_numeric((row or {}).get(feature), errors="coerce") for row in universe_rows],
                    dtype=float,
                )
                for feature in _STATE_VECTOR_V1_FEATURES
            ]
        )
    )
    n_rows = len(universe_rows)
    if feature_matrix.size == 0 or feature_matrix.shape[0] != n_rows:
        return []

    target_action_scale = _estimate_action_scale(dict(target_action_params or {}), target_market_cap)

    company_id_arr = (
        np.asarray(payload.get("company_id_arr"), dtype=object)
        if payload.get("company_id_arr") is not None
        else np.asarray([str((row or {}).get("company_id") or "") for row in universe_rows], dtype=object)
    )
    action_time_arr = (
        np.asarray(payload.get("action_time_arr"), dtype="datetime64[ns]")
        if payload.get("action_time_arr") is not None
        else pd.to_datetime(
            [str((row or {}).get("action_date") or "") for row in universe_rows],
            utc=True,
            errors="coerce",
        ).tz_convert(None).to_numpy(dtype="datetime64[ns]")
    )
    sector_arr = (
        np.asarray(payload.get("sector_arr"), dtype=object)
        if payload.get("sector_arr") is not None
        else np.asarray([_outcome_row_taxonomy(row).get("sector") for row in universe_rows], dtype=object)
    )
    subsector_arr = (
        np.asarray(payload.get("subsector_arr"), dtype=object)
        if payload.get("subsector_arr") is not None
        else np.asarray([_outcome_row_taxonomy(row).get("subsector") for row in universe_rows], dtype=object)
    )

    target_vector = np.asarray(
        [
            float(target_compact.get(feature)) if target_compact.get(feature) is not None else np.nan
            for feature in _STATE_VECTOR_V1_FEATURES
        ],
        dtype=float,
    )
    target_action_subtype = _effective_action_subtype(
        action_id,
        dict(target_action_params or {}).get("source_action_subtype"),
        dict(target_action_params or {}),
    )
    similarity = _weighted_state_similarity(
        emb_raw=feature_matrix,
        candidate_vec_raw=target_vector,
        embedding_cols=_STATE_VECTOR_V1_FEATURES,
        action_id=action_id,
        action_subtype=target_action_subtype,
    )
    similarity_scores = np.asarray(similarity.get("state_similarity"), dtype=float)
    weighted_coverage = np.asarray(similarity.get("weighted_coverage"), dtype=float)
    critical_coverage = np.asarray(similarity.get("critical_coverage"), dtype=float)
    coverage_gate_mask = np.asarray(similarity.get("coverage_gate_mask"), dtype=bool)
    size_gate_mask = np.asarray(similarity.get("size_gate_mask"), dtype=bool)
    if similarity_scores.shape[0] != n_rows:
        return []

    borrower_profile_distance = np.asarray(
        [
            _action_specific_same_action_distance(
                action_id=action_id,
                target_compact=target_compact,
                candidate_features=dict(universe_rows[int(idx)] or {}),
                feature_scales=dict(payload.get("feature_scales") or {}),
                target_action_scale=target_action_scale,
                candidate_action_scale=_estimate_action_scale(
                    _outcome_row_action_params(universe_rows[int(idx)]),
                    _outcome_row_market_cap(universe_rows[int(idx)]),
                ),
            )
            for idx in range(n_rows)
        ],
        dtype=float,
    )
    candidate_action_scales = np.asarray(
        [
            _estimate_action_scale(
                _outcome_row_action_params(universe_rows[int(idx)]),
                _outcome_row_market_cap(universe_rows[int(idx)]),
            )
            for idx in range(n_rows)
        ],
        dtype=float,
    )
    candidate_archetype_labels = np.asarray(
        [
            _debt_issuance_archetype_profile(
                compact_features=dict(universe_rows[int(idx)] or {}),
                action_id=action_id,
                action_scale=float(candidate_action_scales[int(idx)]) if np.isfinite(candidate_action_scales[int(idx)]) else None,
            ).get("label")
            for idx in range(n_rows)
        ],
        dtype=object,
    )
    candidate_market_regime_similarity = np.asarray(
        [
            _debt_issuance_market_regime_similarity(
                target_compact=target_compact,
                candidate_features=dict(universe_rows[int(idx)] or {}),
            )
            for idx in range(n_rows)
        ],
        dtype=float,
    )

    target_company_id = str(case.get("source_company_id") or case.get("company_id") or "").strip()
    as_of_dt = pd.to_datetime(case.get("as_of_time"), utc=True, errors="coerce")
    anchor_dt = pd.to_datetime(case.get("anchor_action_date") or case.get("as_of_time"), utc=True, errors="coerce")
    normalized_anchor_dt = _normalize_as_of_time(str(anchor_dt)) if pd.notna(anchor_dt) else ""

    positive_count = int(positive_limit_per_source) if int(positive_limit_per_source) > 0 else int(max(1, top_k))
    tail_count = int(negative_limit_per_competitor) if int(negative_limit_per_competitor) > 0 else int(max(1, top_k))
    candidate_limit = max(positive_count + tail_count, int(max(1, top_k)))

    eligible_mask = np.isfinite(similarity_scores)
    if pd.notna(as_of_dt):
        eligible_mask &= np.isnat(action_time_arr) | (action_time_arr <= as_of_dt.to_datetime64())
    if target_company_id and normalized_anchor_dt:
        anchor_time64 = pd.to_datetime(normalized_anchor_dt, utc=True, errors="coerce")
        if pd.notna(anchor_time64):
            eligible_mask &= ~(
                (company_id_arr.astype(str) == target_company_id)
                & (action_time_arr == anchor_time64.to_datetime64())
            )
    if coverage_gate_mask.shape[0] == n_rows:
        eligible_mask &= coverage_gate_mask
    if size_gate_mask.shape[0] == n_rows:
        eligible_mask &= size_gate_mask
    if not bool(np.any(eligible_mask)):
        return []

    target_sector = str(target_taxonomy.get("sector") or "").strip()
    target_subsector = str(target_taxonomy.get("subsector") or "").strip()
    same_subsector_mask = eligible_mask & (subsector_arr.astype(str) == target_subsector) if target_subsector else np.zeros(n_rows, dtype=bool)
    same_sector_mask = eligible_mask & (sector_arr.astype(str) == target_sector) if target_sector else np.zeros(n_rows, dtype=bool)
    if int(np.count_nonzero(same_subsector_mask)) >= int(max(1, candidate_limit)):
        neighborhood_mask = same_subsector_mask
        taxonomy_mode = "same_subsector"
    elif int(np.count_nonzero(same_sector_mask)) >= int(max(1, candidate_limit)):
        neighborhood_mask = same_sector_mask
        taxonomy_mode = "same_sector"
    else:
        neighborhood_mask = eligible_mask
        taxonomy_mode = "same_action"

    candidate_idx = np.flatnonzero(neighborhood_mask)
    if candidate_idx.size == 0:
        return []
    prefer_cross_company = bool(_same_action_prefers_cross_company(action_id) and target_company_id)
    per_company_cap = int(_same_action_company_cap(action_id))
    target_company_cap = int(_same_action_target_company_cap(action_id))
    ranked_pool_limit = int(candidate_limit)
    if prefer_cross_company or per_company_cap > 0 or target_company_cap > 0:
        ranked_pool_limit = max(int(candidate_limit), int(candidate_limit) * 4)
    ranked_idx = np.asarray(
        sorted(
            candidate_idx.tolist(),
            key=lambda idx: (
                1 if (prefer_cross_company and str(company_id_arr[int(idx)] or "") == target_company_id) else 0,
                _action_specific_same_action_distance(
                    action_id=action_id,
                    target_compact=target_compact,
                    candidate_features=dict(universe_rows[int(idx)] or {}),
                    feature_scales=dict(payload.get("feature_scales") or {}),
                ),
                -float(similarity_scores[int(idx)] if np.isfinite(similarity_scores[int(idx)]) else -1.0),
                -float(weighted_coverage[int(idx)] if np.isfinite(weighted_coverage[int(idx)]) else 0.0),
                -float(critical_coverage[int(idx)] if np.isfinite(critical_coverage[int(idx)]) else 0.0),
                int(idx),
            ),
        )[:ranked_pool_limit],
        dtype=int,
    )
    matches = [
        _same_action_model_confuser_match_payload(
            universe_rows[int(idx)],
            similarity_score=float(similarity_scores[int(idx)]),
            taxonomy_mode=taxonomy_mode,
            weighted_coverage=(
                float(weighted_coverage[int(idx)]) if np.isfinite(weighted_coverage[int(idx)]) else None
            ),
            critical_coverage=(
                float(critical_coverage[int(idx)]) if np.isfinite(critical_coverage[int(idx)]) else None
            ),
            analog_distance=(
                float(borrower_profile_distance[int(idx)]) if np.isfinite(borrower_profile_distance[int(idx)]) else None
            ),
            debt_archetype_label=str(candidate_archetype_labels[int(idx)] or "") if candidate_archetype_labels.size else None,
            debt_market_regime_similarity=(
                float(candidate_market_regime_similarity[int(idx)])
                if np.isfinite(candidate_market_regime_similarity[int(idx)])
                else None
            ),
        )
        for idx in ranked_idx.tolist()
    ]
    matches = _limit_same_action_company_repeats(
        matches,
        per_company_cap=per_company_cap,
        target_company_id=target_company_id,
        target_company_cap=target_company_cap,
    )
    return matches[:candidate_limit]


def _build_same_action_analog_positive_source(
    *,
    case: Dict[str, Any],
    target_compact: Dict[str, Any],
    target_taxonomy: Dict[str, str],
    target_action_params: Optional[Dict[str, Any]],
    target_market_cap: Optional[float],
    top_k: int,
    positive_limit_per_source: int,
    negative_limit_per_competitor: int,
    same_action_universe_lookup: Dict[str, Dict[str, Any]],
    regime_aware: bool = False,
) -> Optional[Dict[str, Any]]:
    action_id = str(case.get("anchor_action_id") or "").strip()
    if not action_id:
        return None
    payload = dict(same_action_universe_lookup.get(action_id) or {})
    universe_rows = list(payload.get("rows") or [])
    if not universe_rows:
        return None
    feature_scales = dict(payload.get("feature_scales") or {})
    if payload.get("feature_matrix") is not None:
        feature_matrix = np.asarray(payload.get("feature_matrix"), dtype=float)
    else:
        feature_matrix = np.column_stack(
            [
                np.asarray(
                    [pd.to_numeric((row or {}).get(feature), errors="coerce") for row in universe_rows],
                    dtype=float,
                )
                for feature in _STATE_VECTOR_V1_FEATURES
            ]
        )
    company_id_arr = (
        np.asarray(payload.get("company_id_arr"), dtype=object)
        if payload.get("company_id_arr") is not None
        else np.asarray([str((row or {}).get("company_id") or "") for row in universe_rows], dtype=object)
    )
    action_time_arr = (
        np.asarray(payload.get("action_time_arr"), dtype="datetime64[ns]")
        if payload.get("action_time_arr") is not None
        else pd.to_datetime(
            [str((row or {}).get("action_date") or "") for row in universe_rows],
            utc=True,
            errors="coerce",
        ).tz_convert(None).to_numpy(dtype="datetime64[ns]")
    )
    sector_arr = (
        np.asarray(payload.get("sector_arr"), dtype=object)
        if payload.get("sector_arr") is not None
        else np.asarray([_outcome_row_taxonomy(row).get("sector") for row in universe_rows], dtype=object)
    )
    subsector_arr = (
        np.asarray(payload.get("subsector_arr"), dtype=object)
        if payload.get("subsector_arr") is not None
        else np.asarray([_outcome_row_taxonomy(row).get("subsector") for row in universe_rows], dtype=object)
    )
    target_company_id = str(case.get("source_company_id") or case.get("company_id") or "").strip()
    as_of_dt = pd.to_datetime(case.get("as_of_time"), utc=True, errors="coerce")
    anchor_dt = pd.to_datetime(case.get("anchor_action_date") or case.get("as_of_time"), utc=True, errors="coerce")
    normalized_anchor_dt = _normalize_as_of_time(str(anchor_dt)) if pd.notna(anchor_dt) else ""

    positive_count = int(positive_limit_per_source) if int(positive_limit_per_source) > 0 else int(max(1, top_k))
    tail_count = int(negative_limit_per_competitor) if int(negative_limit_per_competitor) > 0 else int(max(1, top_k))
    candidate_limit = max(positive_count + tail_count, int(max(1, top_k)))

    n_rows = len(universe_rows)
    if feature_matrix.size == 0 or feature_matrix.shape[0] != n_rows:
        return None
    eligible_mask = np.ones(n_rows, dtype=bool)
    if pd.notna(as_of_dt):
        eligible_mask &= np.isnat(action_time_arr) | (action_time_arr <= as_of_dt.to_datetime64())
    if target_company_id and normalized_anchor_dt:
        anchor_time64 = pd.to_datetime(normalized_anchor_dt, utc=True, errors="coerce")
        if pd.notna(anchor_time64):
            eligible_mask &= ~(
                (company_id_arr.astype(str) == target_company_id)
                & (action_time_arr == anchor_time64.to_datetime64())
            )
    if not bool(np.any(eligible_mask)):
        return None

    target_vector = np.asarray(
        [
            float(target_compact.get(feature)) if target_compact.get(feature) is not None else np.nan
            for feature in _STATE_VECTOR_V1_FEATURES
        ],
        dtype=float,
    )
    scale_vector = np.asarray(
        [float(feature_scales.get(feature) or 1.0) for feature in _STATE_VECTOR_V1_FEATURES],
        dtype=float,
    )
    scale_vector = np.where(np.isfinite(scale_vector) & (scale_vector > 1e-9), scale_vector, 1.0)
    valid_matrix = np.isfinite(feature_matrix) & np.isfinite(target_vector[None, :])
    diff_matrix = np.where(
        valid_matrix,
        np.abs(feature_matrix - target_vector[None, :]) / scale_vector[None, :],
        np.nan,
    )
    feature_counts = np.sum(valid_matrix, axis=1)
    with np.errstate(invalid="ignore"):
        analog_distances = np.nanmean(diff_matrix, axis=1)
    eligible_mask &= feature_counts > 0
    if not bool(np.any(eligible_mask)):
        return None

    target_action_scale = _estimate_action_scale(dict(target_action_params or {}), target_market_cap)
    candidate_action_scales = np.asarray(
        [
            _estimate_action_scale(
                _outcome_row_action_params(universe_rows[int(idx)]),
                _outcome_row_market_cap(universe_rows[int(idx)]),
            )
            for idx in range(n_rows)
        ],
        dtype=float,
    )
    if np.isfinite(target_action_scale):
        candidate_log_scales = np.where(
            np.isfinite(candidate_action_scales) & (candidate_action_scales >= 0.0),
            np.log1p(candidate_action_scales),
            np.nan,
        )
        target_log_scale = np.log1p(max(float(target_action_scale), 0.0))
        scale_valid = np.isfinite(candidate_log_scales)
        if bool(np.any(scale_valid)):
            scale_center = float(np.nanmedian(candidate_log_scales[scale_valid]))
            scale_dispersion = float(np.nanmedian(np.abs(candidate_log_scales[scale_valid] - scale_center)))
            if not np.isfinite(scale_dispersion) or scale_dispersion <= 1e-6:
                scale_dispersion = float(np.nanstd(candidate_log_scales[scale_valid]))
            if not np.isfinite(scale_dispersion) or scale_dispersion <= 1e-6:
                scale_dispersion = 0.05
            scale_distances = np.abs(candidate_log_scales - target_log_scale) / max(scale_dispersion, 0.05)
            analog_distances = np.where(
                scale_valid,
                np.where(
                    np.isfinite(analog_distances),
                    (analog_distances * feature_counts + scale_distances) / np.maximum(feature_counts + 1, 1),
                    scale_distances,
                ),
                analog_distances,
            )
            feature_counts = feature_counts + scale_valid.astype(int)

    borrower_profile_distance = np.asarray(
        [
            _action_specific_same_action_distance(
                action_id=action_id,
                target_compact=target_compact,
                candidate_features=dict(universe_rows[int(idx)] or {}),
                feature_scales=feature_scales,
                target_action_scale=target_action_scale,
                candidate_action_scale=float(candidate_action_scales[int(idx)]) if np.isfinite(candidate_action_scales[int(idx)]) else None,
            )
            for idx in range(n_rows)
        ],
        dtype=float,
    )
    target_archetype_profile = _debt_issuance_archetype_profile(
        compact_features=target_compact,
        action_id=action_id,
        action_scale=target_action_scale if np.isfinite(target_action_scale) else None,
    )
    target_archetype_label = str(target_archetype_profile.get("label") or "")
    candidate_archetype_labels = np.asarray(
        [
            _debt_issuance_archetype_profile(
                compact_features=dict(universe_rows[int(idx)] or {}),
                action_id=action_id,
                action_scale=float(candidate_action_scales[int(idx)]) if np.isfinite(candidate_action_scales[int(idx)]) else None,
            ).get("label")
            for idx in range(n_rows)
        ],
        dtype=object,
    )
    candidate_market_regime_similarity = np.asarray(
        [
            _debt_issuance_market_regime_similarity(
                target_compact=target_compact,
                candidate_features=dict(universe_rows[int(idx)] or {}),
            )
            for idx in range(n_rows)
        ],
        dtype=float,
    )

    latent_regime_similarity = np.full(n_rows, np.nan, dtype=float)
    latent_regime_assignments = np.full(n_rows, -1, dtype=int)
    if bool(regime_aware):
        latent_model = payload.get("latent_regime_model")
        latent_memberships = payload.get("latent_regime_memberships")
        if isinstance(latent_model, dict):
            try:
                if latent_memberships is None:
                    latent_memberships = latent_regime_memberships(feature_matrix, latent_model)
                latent_memberships_arr = np.asarray(latent_memberships, dtype=float)
                target_membership = latent_regime_memberships(target_vector.reshape(1, -1), latent_model)
                if latent_memberships_arr.ndim == 2 and latent_memberships_arr.shape[0] == n_rows and target_membership.ndim == 2:
                    latent_regime_similarity = np.sum(
                        latent_memberships_arr * target_membership.reshape(1, -1),
                        axis=1,
                    )
                    latent_regime_assignments = np.argmax(latent_memberships_arr, axis=1).astype(int)
            except Exception:
                latent_regime_similarity = np.full(n_rows, np.nan, dtype=float)
                latent_regime_assignments = np.full(n_rows, -1, dtype=int)

    target_sector = str(target_taxonomy.get("sector") or "").strip()
    target_subsector = str(target_taxonomy.get("subsector") or "").strip()
    same_subsector_mask = eligible_mask & (subsector_arr.astype(str) == target_subsector) if target_subsector else np.zeros(n_rows, dtype=bool)
    same_sector_mask = eligible_mask & (sector_arr.astype(str) == target_sector) if target_sector else np.zeros(n_rows, dtype=bool)
    if int(np.count_nonzero(same_subsector_mask)) >= int(max(1, candidate_limit)):
        neighborhood_mask = same_subsector_mask
        taxonomy_mode = "same_subsector"
    elif int(np.count_nonzero(same_sector_mask)) >= int(max(1, candidate_limit)):
        neighborhood_mask = same_sector_mask
        taxonomy_mode = "same_sector"
    else:
        neighborhood_mask = eligible_mask
        taxonomy_mode = "same_action"

    candidate_idx = np.flatnonzero(neighborhood_mask)
    if candidate_idx.size == 0:
        return None
    same_archetype_mask = (
        neighborhood_mask & (candidate_archetype_labels.astype(str) == target_archetype_label)
        if target_archetype_label
        else np.zeros(n_rows, dtype=bool)
    )
    same_archetype_count = int(np.count_nonzero(same_archetype_mask))
    if same_archetype_count >= int(max(positive_count, min(candidate_limit, 4))):
        candidate_idx = np.flatnonzero(same_archetype_mask)
        taxonomy_mode = f"{taxonomy_mode}_same_archetype"
    prefer_cross_company = bool(_same_action_prefers_cross_company(action_id) and target_company_id)
    per_company_cap = int(_same_action_company_cap(action_id))
    target_company_cap = int(_same_action_target_company_cap(action_id))
    ranked_pool_limit = int(candidate_limit)
    if prefer_cross_company or per_company_cap > 0 or target_company_cap > 0:
        ranked_pool_limit = max(int(candidate_limit), int(candidate_limit) * 4)
    ranked_idx = np.asarray(
        sorted(
            candidate_idx.tolist(),
            key=lambda idx: (
                1 if (prefer_cross_company and str(company_id_arr[int(idx)] or "") == target_company_id) else 0,
                0 if str(candidate_archetype_labels[int(idx)] or "") == target_archetype_label else 1,
                0 if float(candidate_market_regime_similarity[int(idx)]) >= (0.72 if regime_aware else 0.60) else 1,
                _action_specific_same_action_distance(
                    action_id=action_id,
                    target_compact=target_compact,
                    candidate_features=dict(universe_rows[int(idx)] or {}),
                    feature_scales=feature_scales,
                    target_action_scale=target_action_scale,
                    candidate_action_scale=(
                        float(candidate_action_scales[int(idx)]) if np.isfinite(candidate_action_scales[int(idx)]) else None
                    ),
                ),
                float(analog_distances[int(idx)]) if np.isfinite(analog_distances[int(idx)]) else float("inf"),
                -float(candidate_market_regime_similarity[int(idx)] if np.isfinite(candidate_market_regime_similarity[int(idx)]) else 0.0),
                -float(latent_regime_similarity[int(idx)] if np.isfinite(latent_regime_similarity[int(idx)]) else -1.0),
                -int(feature_counts[int(idx)] if np.isfinite(feature_counts[int(idx)]) else 0),
                int(idx),
            ),
        )[:ranked_pool_limit],
        dtype=int,
    )
    if ranked_idx.size == 0:
        return None
    full_matches = [
        _same_action_analog_match_payload(
            universe_rows[int(idx)],
            distance=float(analog_distances[int(idx)]),
            feature_count=int(feature_counts[int(idx)]),
            taxonomy_mode=taxonomy_mode,
            latent_regime_similarity=(
                float(latent_regime_similarity[int(idx)]) if np.isfinite(latent_regime_similarity[int(idx)]) else None
            ),
            latent_regime_cluster=(
                int(latent_regime_assignments[int(idx)]) if int(latent_regime_assignments[int(idx)]) >= 0 else None
            ),
            debt_archetype_label=str(candidate_archetype_labels[int(idx)] or "") if candidate_archetype_labels.size else None,
            debt_market_regime_similarity=(
                float(candidate_market_regime_similarity[int(idx)])
                if np.isfinite(candidate_market_regime_similarity[int(idx)])
                else None
            ),
        )
        for idx in ranked_idx.tolist()
    ]
    full_matches = _limit_same_action_company_repeats(
        full_matches,
        per_company_cap=per_company_cap,
        target_company_id=target_company_id,
        target_company_cap=target_company_cap,
    )[:candidate_limit]
    if not full_matches:
        return None
    matches = full_matches[:positive_count]
    if not matches:
        return None
    source_type = "analog_regime_consensus_same_action_universe" if regime_aware else "analog_consensus_same_action_universe"
    confuser_matches = _build_same_action_model_confuser_matches(
        case=case,
        target_compact=target_compact,
        target_taxonomy=target_taxonomy,
        target_action_params=target_action_params,
        target_market_cap=target_market_cap,
        top_k=top_k,
        positive_limit_per_source=positive_limit_per_source,
        negative_limit_per_competitor=negative_limit_per_competitor,
        same_action_universe_lookup=same_action_universe_lookup,
    )
    confuser_matches = _limit_same_action_company_repeats(
        confuser_matches,
        per_company_cap=per_company_cap,
        target_company_id=target_company_id,
        target_company_cap=target_company_cap,
    )
    return {
        "source_type": source_type,
        "anchor_candidate_id": source_type,
        "anchor_candidate_precedent_confidence": float(matches[0].get("similarity_score") or 0.0),
        "anchor_candidate_rank": 0,
        "matches": matches,
        "full_matches": full_matches,
        "confuser_matches": confuser_matches,
        "taxonomy_mode": taxonomy_mode,
        "regime_aware": bool(regime_aware),
    }


def _pair_rows_for_case(
    *,
    case: Dict[str, Any],
    precedent_index: Dict[str, Any],
    precedent_matches: Dict[str, Any],
    target_compact: Dict[str, Any],
    target_taxonomy: Dict[str, str],
    target_action_params: Optional[Dict[str, Any]] = None,
    target_market_cap: Optional[float] = None,
    target_source: str = "snapshot",
    top_k: int,
    anchor_outcomes_lookup: Optional[Dict[tuple[str, str], List[Dict[str, Any]]]] = None,
    precedent_outcomes_lookup: Optional[Dict[tuple[str, str, str], Dict[str, Any]]] = None,
    positive_limit_per_source: int = 0,
    negative_limit_per_competitor: int = 0,
    same_family_negatives_only_if_available: bool = False,
    always_include_actual_anchor_positive: bool = False,
    include_within_action_hard_negatives: bool = False,
    include_same_action_positive_ordering: bool = False,
    actual_anchor_within_action_negative_source: str = "retrieved_pool",
    positive_source_mode: str = "include_retrieved",
    hard_negative_taxonomy_mode: str = "none",
    same_action_universe_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    candidate_rows = _candidate_rankings(precedent_index)
    result_lookup = _result_by_candidate_id(precedent_matches)
    anchor_action_id = str(case.get("anchor_action_id") or "")
    anchor_action_family = str(case.get("anchor_action_family") or "")

    anchor_candidates = [row for row in candidate_rows if str(row.get("action_id") or "") == anchor_action_id]
    competitor_candidates = _top_candidate_per_action(
        [row for row in candidate_rows if str(row.get("action_id") or "") != anchor_action_id]
    )
    if same_family_negatives_only_if_available and anchor_action_family:
        same_family_competitors = [
            row
            for row in competitor_candidates
            if str(row.get("action_id") or "").split(".", 1)[0] == anchor_action_family
        ]
        if same_family_competitors:
            competitor_candidates = same_family_competitors
    if not competitor_candidates and not include_within_action_hard_negatives and not include_same_action_positive_ordering:
        return []

    positive_sources: List[Dict[str, Any]] = []
    retrieved_anchor_matches_all: List[Dict[str, Any]] = []
    for anchor_candidate in anchor_candidates:
        anchor_result = result_lookup.get(str(anchor_candidate.get("candidate_id") or ""))
        if not anchor_result:
            continue
        anchor_matches = list((anchor_result.get("precedent_pack") or {}).get("matches", []) or [])[:top_k]
        retrieved_anchor_matches_all.extend(anchor_matches)
        if positive_limit_per_source > 0:
            anchor_matches = anchor_matches[: int(positive_limit_per_source)]
        if not anchor_matches:
            continue
        positive_sources.append(
            {
                "source_type": "retrieved_anchor_action",
                "anchor_candidate_id": str(anchor_candidate.get("candidate_id") or ""),
                "anchor_candidate_precedent_confidence": float(anchor_candidate.get("precedent_confidence") or 0.0),
                "anchor_candidate_rank": int(_first((i + 1 for i, row in enumerate(candidate_rows) if row == anchor_candidate), 1)),
                "matches": anchor_matches,
                "full_matches": list((anchor_result.get("precedent_pack") or {}).get("matches", []) or [])[:top_k],
            }
        )

    retrieved_anchor_matches_all = _dedupe_matches(retrieved_anchor_matches_all)
    actual_row = (
        _select_actual_anchor_outcome(case, anchor_outcomes_lookup=anchor_outcomes_lookup)
        if anchor_outcomes_lookup
        else None
    )
    if actual_row and (always_include_actual_anchor_positive or not positive_sources):
        positive_sources.append(
            {
                "source_type": "actual_anchor_outcome",
                "anchor_candidate_id": "actual_anchor_outcome",
                "anchor_candidate_precedent_confidence": 1.0,
                "anchor_candidate_rank": 0,
                "matches": [_actual_anchor_match_payload(case, actual_row)],
                "full_matches": list(retrieved_anchor_matches_all),
            }
        )

    positive_source_mode_text = str(positive_source_mode or "include_retrieved").strip().lower()
    if actual_row and positive_source_mode_text == "actual_anchor_preferred":
        positive_sources = [
            source
            for source in positive_sources
            if str(source.get("source_type") or "") == "actual_anchor_outcome"
        ]
    elif positive_source_mode_text == "analog_consensus_same_action_universe":
        analog_source = _build_same_action_analog_positive_source(
            case=case,
            target_compact=target_compact,
            target_taxonomy=target_taxonomy,
            target_action_params=target_action_params,
            target_market_cap=target_market_cap,
            top_k=int(top_k),
            positive_limit_per_source=int(positive_limit_per_source),
            negative_limit_per_competitor=int(negative_limit_per_competitor),
            same_action_universe_lookup=same_action_universe_lookup or {},
            regime_aware=False,
        )
        if analog_source is not None:
            positive_sources = [analog_source]
    elif positive_source_mode_text == "analog_regime_consensus_same_action_universe":
        analog_source = _build_same_action_analog_positive_source(
            case=case,
            target_compact=target_compact,
            target_taxonomy=target_taxonomy,
            target_action_params=target_action_params,
            target_market_cap=target_market_cap,
            top_k=int(top_k),
            positive_limit_per_source=int(positive_limit_per_source),
            negative_limit_per_competitor=int(negative_limit_per_competitor),
            same_action_universe_lookup=same_action_universe_lookup or {},
            regime_aware=True,
        )
        if analog_source is not None:
            positive_sources = [analog_source]

    if not positive_sources:
        return []

    def _append_pair_row(
        *,
        positive_source: Dict[str, Any],
        pos_rank: int,
        pos: Dict[str, Any],
        pos_features: Dict[str, Any],
        competitor_action_id: str,
        competitor_candidate_id: str,
        competitor_candidate_precedent_confidence: float,
        competitor_candidate_rank: int,
        neg_rank: int,
        neg: Dict[str, Any],
        neg_features: Dict[str, Any],
        pair_source: str,
        negative_source: str,
    ) -> None:
        positive_precedent_id = str(pos.get("precedent_id") or "")
        negative_precedent_id = str(neg.get("precedent_id") or "")
        target_action_scale = _estimate_action_scale(dict(target_action_params or {}), target_market_cap)
        if positive_precedent_id and negative_precedent_id and positive_precedent_id == negative_precedent_id:
            return
        dedupe_key = (
            str(case.get("company_id") or ""),
            anchor_action_id,
            competitor_action_id,
            positive_precedent_id,
            negative_precedent_id,
        )
        if dedupe_key in seen_pairs:
            return
        seen_pairs.add(dedupe_key)
        target_action_subtype = _effective_action_subtype(
            anchor_action_id,
            dict(target_action_params or {}).get("source_action_subtype"),
            dict(target_action_params or {}),
        )
        rows.append(
            {
                "company_id": str(case.get("company_id") or ""),
                "as_of_time": str(case.get("as_of_time") or ""),
                "anchor_action_id": anchor_action_id,
                "anchor_action_subtype": target_action_subtype,
                "anchor_action_family": str(case.get("anchor_action_family") or ""),
                "anchor_candidate_id": str(positive_source.get("anchor_candidate_id") or ""),
                "anchor_candidate_precedent_confidence": float(positive_source.get("anchor_candidate_precedent_confidence") or 0.0),
                "anchor_candidate_rank": int(positive_source.get("anchor_candidate_rank") or 0),
                "competitor_action_id": competitor_action_id,
                "competitor_candidate_id": competitor_candidate_id,
                "competitor_candidate_precedent_confidence": competitor_candidate_precedent_confidence,
                "competitor_candidate_rank": competitor_candidate_rank,
                "label": 1,
                "pair_source": pair_source,
                "positive_source": str(positive_source.get("source_type") or ""),
                "negative_source": negative_source,
                "positive_precedent_id": positive_precedent_id,
                "positive_precedent_company_id": str(pos.get("company_id") or ""),
                "positive_precedent_rank_within_candidate": pos_rank,
                "positive_similarity_score": float(pos.get("similarity_score") or 0.0),
                "negative_precedent_id": negative_precedent_id,
                "negative_precedent_company_id": str(neg.get("company_id") or ""),
                "negative_precedent_rank_within_candidate": neg_rank,
                "negative_similarity_score": float(neg.get("similarity_score") or 0.0),
                "target_compact": {feature: target_compact.get(feature) for feature in _STATE_VECTOR_V1_FEATURES},
                "target_source": str(target_source or "snapshot"),
                "target_sector": str(target_taxonomy.get("sector") or ""),
                "target_subsector": str(target_taxonomy.get("subsector") or ""),
                "target_action_scale": float(target_action_scale) if target_action_scale is not None else None,
                "positive_compact": {feature: pos_features.get(feature) for feature in _STATE_VECTOR_V1_FEATURES},
                "negative_compact": {feature: neg_features.get(feature) for feature in _STATE_VECTOR_V1_FEATURES},
                "positive_sector": pos_features.get("sector") or pos_features.get("base_sector"),
                "positive_subsector": pos_features.get("subsector"),
                "positive_action_scale": pos.get("action_scale"),
                "negative_sector": neg_features.get("sector") or neg_features.get("base_sector"),
                "negative_subsector": neg_features.get("subsector"),
                "negative_action_scale": neg.get("action_scale"),
                "feature_gap_summary": {
                    feature: {
                        "positive_abs_diff": _absdiff(target_compact.get(feature), pos_features.get(feature)),
                        "negative_abs_diff": _absdiff(target_compact.get(feature), neg_features.get(feature)),
                    }
                    for feature in _PAIRWISE_FEATURE_GAP_SUMMARY_FEATURES
                },
            }
        )

    target_action_scale = _estimate_action_scale(dict(target_action_params or {}), target_market_cap)
    rows: List[Dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str, str, str]] = set()
    for positive_source in positive_sources:
        anchor_matches = list(positive_source.get("matches") or [])
        full_anchor_matches = list(positive_source.get("full_matches") or [])
        confuser_matches = list(positive_source.get("confuser_matches") or [])
        if not anchor_matches:
            continue
        if include_within_action_hard_negatives:
            positive_source_type = str(positive_source.get("source_type") or "")
            if positive_source_type == "actual_anchor_outcome":
                negative_source_mode = str(actual_anchor_within_action_negative_source or "retrieved_pool").strip().lower()
                same_action_negative_pool: List[Dict[str, Any]] = []
                same_action_pair_source = "actual_anchor_outcome_vs_same_action_retrieved"
                same_action_negative_source = "same_action_retrieved_pool"
                if negative_source_mode == "same_action_universe":
                    analog_source = _build_same_action_analog_positive_source(
                        case=case,
                        target_compact=target_compact,
                        target_taxonomy=target_taxonomy,
                        top_k=int(top_k),
                        positive_limit_per_source=int(positive_limit_per_source),
                        negative_limit_per_competitor=int(negative_limit_per_competitor),
                        same_action_universe_lookup=same_action_universe_lookup or {},
                        regime_aware=False,
                    )
                    if analog_source is not None:
                        same_action_negative_pool = list(analog_source.get("full_matches") or [])
                        same_action_pair_source = "actual_anchor_outcome_vs_same_action_universe"
                        same_action_negative_source = "same_action_universe"
                if not same_action_negative_pool:
                    same_action_negative_pool = list(retrieved_anchor_matches_all)
                    same_action_pair_source = "actual_anchor_outcome_vs_same_action_retrieved"
                    same_action_negative_source = "same_action_retrieved_pool"
            elif positive_source_type == "analog_consensus_same_action_universe":
                positive_precedent_ids = {
                    str(match.get("precedent_id") or "").strip()
                    for match in anchor_matches
                    if str(match.get("precedent_id") or "").strip()
                }
                same_action_negative_pool = [
                    match
                    for match in _dedupe_matches(
                        list(full_anchor_matches[len(anchor_matches) :])
                        + list(confuser_matches)
                    )
                    if str(match.get("precedent_id") or "").strip() not in positive_precedent_ids
                ]
                same_action_negative_pool = _rank_same_action_hard_confusers(
                    same_action_negative_pool,
                    action_id=anchor_action_id,
                    target_compact=target_compact,
                    target_sector=str(target_taxonomy.get("sector") or ""),
                    target_subsector=str(target_taxonomy.get("subsector") or ""),
                    target_action_scale=target_action_scale,
                )[: _same_action_negative_pool_limit(int(top_k), anchor_matches)]
                same_action_pair_source = "analog_consensus_same_action_universe_vs_same_action_confusers"
                same_action_negative_source = "same_action_confuser_pool"
            elif positive_source_type == "analog_regime_consensus_same_action_universe":
                positive_precedent_ids = {
                    str(match.get("precedent_id") or "").strip()
                    for match in anchor_matches
                    if str(match.get("precedent_id") or "").strip()
                }
                same_action_negative_pool = [
                    match
                    for match in _dedupe_matches(
                        list(full_anchor_matches[len(anchor_matches) :])
                        + list(confuser_matches)
                    )
                    if str(match.get("precedent_id") or "").strip() not in positive_precedent_ids
                ]
                same_action_negative_pool = _rank_same_action_hard_confusers(
                    same_action_negative_pool,
                    action_id=anchor_action_id,
                    target_compact=target_compact,
                    target_sector=str(target_taxonomy.get("sector") or ""),
                    target_subsector=str(target_taxonomy.get("subsector") or ""),
                    target_action_scale=target_action_scale,
                )[: _same_action_negative_pool_limit(int(top_k), anchor_matches)]
                same_action_pair_source = "analog_regime_consensus_same_action_universe_vs_same_action_confusers"
                same_action_negative_source = "same_action_confuser_pool"
            else:
                same_action_negative_pool = list(full_anchor_matches[len(anchor_matches) :])
                same_action_pair_source = "retrieved_anchor_action_vs_same_action_retrieved"
                same_action_negative_source = "same_action_retrieved_pool"
            if hard_negative_taxonomy_mode != "none":
                same_action_negative_pool = _rank_hard_negative_matches(
                    same_action_negative_pool,
                    target_compact=target_compact,
                    target_sector=str(target_taxonomy.get("sector") or ""),
                    target_subsector=str(target_taxonomy.get("subsector") or ""),
                    taxonomy_mode=hard_negative_taxonomy_mode,
                )
            for pos_rank, pos in enumerate(anchor_matches, start=1):
                pos_features = _enrich_match_compact(
                    pos,
                    precedent_outcomes_lookup=precedent_outcomes_lookup or {},
                )
                for neg_rank, neg in enumerate(same_action_negative_pool, start=1):
                    neg_features = _enrich_match_compact(
                        neg,
                        precedent_outcomes_lookup=precedent_outcomes_lookup or {},
                    )
                    _append_pair_row(
                        positive_source=positive_source,
                        pos_rank=pos_rank,
                        pos=pos,
                        pos_features=pos_features,
                        competitor_action_id=anchor_action_id,
                        competitor_candidate_id="same_action_retrieved_pool",
                        competitor_candidate_precedent_confidence=float(
                            positive_source.get("anchor_candidate_precedent_confidence") or 0.0
                        ),
                        competitor_candidate_rank=int(positive_source.get("anchor_candidate_rank") or 0),
                        neg_rank=neg_rank,
                        neg=neg,
                        neg_features=neg_features,
                        pair_source=same_action_pair_source,
                        negative_source=same_action_negative_source,
                    )

        if include_same_action_positive_ordering:
            ordered_same_action_matches = _dedupe_matches(list(full_anchor_matches))
            if hard_negative_taxonomy_mode != "none":
                ordered_same_action_matches = _rank_hard_negative_matches(
                    ordered_same_action_matches,
                    target_compact=target_compact,
                    target_sector=str(target_taxonomy.get("sector") or ""),
                    target_subsector=str(target_taxonomy.get("subsector") or ""),
                    taxonomy_mode=hard_negative_taxonomy_mode,
                )
            comparison_window = _same_action_ordering_window(int(top_k))
            for pos_rank, pos in enumerate(ordered_same_action_matches, start=1):
                pos_features = _enrich_match_compact(
                    pos,
                    precedent_outcomes_lookup=precedent_outcomes_lookup or {},
                )
                next_matches = ordered_same_action_matches[pos_rank : pos_rank + comparison_window]
                for neg_rank, neg in enumerate(next_matches, start=pos_rank + 1):
                    neg_features = _enrich_match_compact(
                        neg,
                        precedent_outcomes_lookup=precedent_outcomes_lookup or {},
                    )
                    _append_pair_row(
                        positive_source=positive_source,
                        pos_rank=pos_rank,
                        pos=pos,
                        pos_features=pos_features,
                        competitor_action_id=anchor_action_id,
                        competitor_candidate_id="same_action_ranked_ordering",
                        competitor_candidate_precedent_confidence=float(
                            positive_source.get("anchor_candidate_precedent_confidence") or 0.0
                        ),
                        competitor_candidate_rank=int(positive_source.get("anchor_candidate_rank") or 0),
                        neg_rank=neg_rank,
                        neg=neg,
                        neg_features=neg_features,
                        pair_source=f"{str(positive_source.get('source_type') or 'same_action')}_rank_ordering",
                        negative_source="same_action_ranked_ordering",
                    )

        if not competitor_candidates:
            continue

        for competitor_candidate in competitor_candidates:
            competitor_result = result_lookup.get(str(competitor_candidate.get("candidate_id") or ""))
            if not competitor_result:
                continue
            competitor_matches = list((competitor_result.get("precedent_pack") or {}).get("matches", []) or [])[:top_k]
            if hard_negative_taxonomy_mode != "none":
                competitor_matches = _rank_hard_negative_matches(
                    competitor_matches,
                    target_compact=target_compact,
                    target_sector=str(target_taxonomy.get("sector") or ""),
                    target_subsector=str(target_taxonomy.get("subsector") or ""),
                    taxonomy_mode=hard_negative_taxonomy_mode,
                )
            if negative_limit_per_competitor > 0:
                competitor_matches = competitor_matches[: int(negative_limit_per_competitor)]
            if not competitor_matches:
                continue
            for pos_rank, pos in enumerate(anchor_matches, start=1):
                pos_features = _enrich_match_compact(
                    pos,
                    precedent_outcomes_lookup=precedent_outcomes_lookup or {},
                )
                for neg_rank, neg in enumerate(competitor_matches, start=1):
                    neg_features = _enrich_match_compact(
                        neg,
                        precedent_outcomes_lookup=precedent_outcomes_lookup or {},
                    )
                    _append_pair_row(
                        positive_source=positive_source,
                        pos_rank=pos_rank,
                        pos=pos,
                        pos_features=pos_features,
                        competitor_action_id=str(competitor_candidate.get("action_id") or ""),
                        competitor_candidate_id=str(competitor_candidate.get("candidate_id") or ""),
                        competitor_candidate_precedent_confidence=float(
                            competitor_candidate.get("precedent_confidence") or 0.0
                        ),
                        competitor_candidate_rank=int(
                            _first((i + 1 for i, row in enumerate(candidate_rows) if row == competitor_candidate), 2)
                        ),
                        neg_rank=neg_rank,
                        neg=neg,
                        neg_features=neg_features,
                        pair_source=str(
                            positive_source.get("source_type") or "anchor_exact_action_vs_competitor_action"
                        ),
                        negative_source="competitor_action_retrieved",
                    )
    return rows


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest_path)
    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path) if args.summary_path else None
    snapshot_catalog_path = Path(args.snapshot_catalog_path) if args.snapshot_catalog_path else None
    snapshot_cache_root = Path(args.snapshot_cache_root) if args.snapshot_cache_root else _snapshot_cache_root_for_manifest(manifest_path)
    runs_root = Path(args.runs_root) if args.runs_root else None
    outcomes_path = Path(args.outcomes_path) if args.outcomes_path else Path(_default_precedent_outcomes_path())
    eval_prefix = str(args.eval_prefix or "").strip()
    eval_id = str(args.eval_id or "001").strip()
    teacher_config = _resolve_teacher_recipe(
        teacher_recipe=str(args.teacher_recipe or "explicit_flags"),
        positive_source_mode=str(args.positive_source_mode or "include_retrieved"),
        include_within_action_hard_negatives=bool(args.include_within_action_hard_negatives),
        include_same_action_positive_ordering=bool(args.include_same_action_positive_ordering),
        actual_anchor_within_action_negative_source=str(
            args.actual_anchor_within_action_negative_source or "retrieved_pool"
        ),
        always_include_actual_anchor_positive=bool(args.always_include_actual_anchor_positive),
        same_family_negatives_only_if_available=bool(args.same_family_negatives_only_if_available),
        hard_negative_taxonomy_mode=str(args.hard_negative_taxonomy_mode or "none"),
    )

    manifest = _load_json(manifest_path)
    positive_source_mode_text = str(teacher_config.get("positive_source_mode") or "").strip().lower()
    allow_cases_only_same_action_teacher = positive_source_mode_text in {
        "analog_consensus_same_action_universe",
        "analog_regime_consensus_same_action_universe",
    }
    selection_rows = list(manifest.get("selection_rankings", []) or [])
    cases = list(manifest.get("cases", []) or [])
    if not selection_rows:
        selection_rows = [
            {
                "company_id": str(case.get("company_id") or ""),
                "as_of_time": str(case.get("as_of_time") or ""),
                "anchor_action_id": str(case.get("anchor_action_id") or ""),
            }
            for case in cases
        ]
    if not selection_rows:
        raise SystemExit("Manifest must include cases or selection_rankings")

    case_lookup = {
        (
            str(case.get("company_id") or ""),
            str(case.get("as_of_time") or ""),
            str(case.get("anchor_action_id") or ""),
        ): dict(case)
        for case in cases
    }
    resolved_cases: List[Dict[str, Any]] = []
    referenced_precedent_keys: List[tuple[str, str, str]] = []
    for row in selection_rows:
        company_id = str(row.get("company_id") or "")
        as_of_time = str(row.get("as_of_time") or "")
        anchor_action_id = str(row.get("anchor_action_id") or "")
        case = case_lookup.get((company_id, as_of_time, anchor_action_id), dict(row))
        raw_precedent_index_path = str(row.get("precedent_index_path") or "").strip()
        precedent_index_path = Path(raw_precedent_index_path) if raw_precedent_index_path else Path("")
        precedent_matches_path = (
            precedent_index_path.with_name("PrecedentMatches.json")
            if raw_precedent_index_path
            else Path("")
        )
        if not (precedent_index_path.is_file() and precedent_matches_path.is_file()):
            if runs_root and eval_prefix:
                resolved = _artifact_paths_from_runs_root(
                    runs_root=runs_root,
                    eval_prefix=eval_prefix,
                    eval_id=eval_id,
                    company_id=company_id,
                    as_of_time=as_of_time,
                )
                if resolved:
                    precedent_index_path = resolved["precedent_index_path"]
                    precedent_matches_path = resolved["precedent_matches_path"]
            elif runs_root:
                resolved = _artifact_paths_from_runs_root(
                    runs_root=runs_root,
                    eval_prefix="",
                    eval_id=eval_id,
                    company_id=company_id,
                    as_of_time=as_of_time,
                )
                if resolved:
                    precedent_index_path = resolved["precedent_index_path"]
                    precedent_matches_path = resolved["precedent_matches_path"]
        if not (precedent_index_path.is_file() and precedent_matches_path.is_file()):
            if not allow_cases_only_same_action_teacher:
                continue
            precedent_index = {"candidate_rows": []}
            precedent_matches = {"results": []}
        else:
            precedent_index = _load_json(precedent_index_path)
            precedent_matches = _load_json(precedent_matches_path)
            referenced_precedent_keys.extend(
                _collect_precedent_reference_keys(
                    precedent_matches,
                    top_k=int(args.top_k_per_candidate),
                )
            )
        resolved_cases.append(
            {
                "case": case,
                "company_id": company_id,
                "as_of_time": as_of_time,
                "precedent_index": precedent_index,
                "precedent_matches": precedent_matches,
            }
        )

    anchor_outcomes_lookup = (
        _load_anchor_outcomes_lookup(outcomes_path, cases=cases)
        if outcomes_path is not None and outcomes_path.exists()
        else {}
    )
    same_action_universe_lookup = (
        _load_same_action_universe_lookup(
            outcomes_path,
            cases=cases,
            include_latent_regime_model=(
                positive_source_mode_text == "analog_regime_consensus_same_action_universe"
                and any(
                    _same_action_regime_requires_latent_model(case.get("anchor_action_id"))
                    for case in cases
                )
            ),
            latent_regime_cluster_grid=_parse_int_grid(
                str(args.analog_regime_cluster_grid or ""),
                default=(2, 3, 4, 5, 6),
            ),
            latent_regime_seed=int(args.analog_regime_seed),
            latent_regime_max_iter=int(args.analog_regime_max_iter),
        )
        if outcomes_path is not None and outcomes_path.exists()
        else {}
    )
    precedent_outcomes_lookup = (
        _load_precedent_outcomes_lookup(
            outcomes_path,
            cases=cases,
            required_keys=referenced_precedent_keys,
        )
        if outcomes_path is not None and outcomes_path.exists()
        else {}
    )

    pair_rows: List[Dict[str, Any]] = []
    cases_with_pairs = 0
    target_source_counts: Counter[str] = Counter()
    for resolved in resolved_cases:
        case = dict(resolved["case"])
        company_id = str(resolved["company_id"] or "")
        as_of_time = str(resolved["as_of_time"] or "")
        target_source = "snapshot"
        target_action_params: Dict[str, Any] = {}
        target_market_cap: Optional[float] = None
        prefer_anchor_outcome_target = str(case.get("mapping_method") or "").strip().lower().startswith(
            "outcomes_parquet"
        )
        fallback_context = (
            _target_context_from_anchor_outcome(
                case,
                anchor_outcomes_lookup=anchor_outcomes_lookup,
            )
            if prefer_anchor_outcome_target
            else None
        )
        if fallback_context is None and prefer_anchor_outcome_target:
            fallback_context = _target_context_from_same_action_universe(
                case,
                same_action_universe_lookup=same_action_universe_lookup,
            )
        if fallback_context is None and prefer_anchor_outcome_target and outcomes_path is not None and outcomes_path.exists():
            fallback_context = _target_context_from_exact_outcomes_row(
                case,
                outcomes_path=outcomes_path,
            )
        if fallback_context is not None:
            target_compact = dict(fallback_context.get("target_compact") or {})
            target_taxonomy = dict(fallback_context.get("target_taxonomy") or {})
            target_action_params = dict(fallback_context.get("target_action_params") or {})
            target_market_cap = fallback_context.get("target_market_cap")
            target_source = str(fallback_context.get("target_source") or "anchor_outcome_fallback")
        else:
            try:
                snapshot_row = _load_snapshot_row(
                    snapshot_cache_root,
                    company_id=company_id,
                    as_of_time=as_of_time,
                    snapshot_catalog_path=snapshot_catalog_path,
                )
            except FileNotFoundError:
                fallback_context = _target_context_from_anchor_outcome(
                    case,
                    anchor_outcomes_lookup=anchor_outcomes_lookup,
                )
                if fallback_context is None:
                    fallback_context = _target_context_from_same_action_universe(
                        case,
                        same_action_universe_lookup=same_action_universe_lookup,
                    )
                if fallback_context is None and outcomes_path is not None and outcomes_path.exists():
                    fallback_context = _target_context_from_exact_outcomes_row(
                        case,
                        outcomes_path=outcomes_path,
                    )
                if fallback_context is None:
                    raise
                target_compact = dict(fallback_context.get("target_compact") or {})
                target_taxonomy = dict(fallback_context.get("target_taxonomy") or {})
                target_action_params = dict(fallback_context.get("target_action_params") or {})
                target_market_cap = fallback_context.get("target_market_cap")
                target_source = str(fallback_context.get("target_source") or "anchor_outcome_fallback")
            else:
                target_compact = _target_compact_values(snapshot_row)
                target_taxonomy = _target_taxonomy(snapshot_row)
                target_action_params = dict(snapshot_row.get("action_params") or {})
                target_market_cap = _snapshot_market_cap(snapshot_row)
        if not str(target_taxonomy.get("sector") or "").strip() or not str(target_taxonomy.get("subsector") or "").strip():
            inferred_target_taxonomy = _infer_target_taxonomy_from_same_action_universe(
                case,
                same_action_universe_lookup=same_action_universe_lookup,
                target_compact=target_compact,
            )
            if inferred_target_taxonomy:
                target_taxonomy = {
                    "sector": str(target_taxonomy.get("sector") or inferred_target_taxonomy.get("sector") or "").strip(),
                    "subsector": str(
                        target_taxonomy.get("subsector") or inferred_target_taxonomy.get("subsector") or ""
                    ).strip(),
                }
        target_source_counts[target_source] += 1
        case_rows = _pair_rows_for_case(
            case=case,
            precedent_index=dict(resolved["precedent_index"]),
            precedent_matches=dict(resolved["precedent_matches"]),
            target_compact=target_compact,
            target_taxonomy=target_taxonomy,
            target_action_params=target_action_params,
            target_market_cap=target_market_cap,
            target_source=target_source,
            top_k=int(args.top_k_per_candidate),
            anchor_outcomes_lookup=anchor_outcomes_lookup,
            precedent_outcomes_lookup=precedent_outcomes_lookup,
            positive_limit_per_source=int(args.positive_limit_per_source),
            negative_limit_per_competitor=int(args.negative_limit_per_competitor),
            same_family_negatives_only_if_available=bool(
                teacher_config.get("same_family_negatives_only_if_available")
            ),
            always_include_actual_anchor_positive=bool(
                teacher_config.get("always_include_actual_anchor_positive")
            ),
            include_within_action_hard_negatives=bool(
                teacher_config.get("include_within_action_hard_negatives")
            ),
            include_same_action_positive_ordering=bool(
                teacher_config.get("include_same_action_positive_ordering")
            ),
            actual_anchor_within_action_negative_source=str(
                teacher_config.get("actual_anchor_within_action_negative_source") or "retrieved_pool"
            ),
            positive_source_mode=str(teacher_config.get("positive_source_mode") or "include_retrieved"),
            hard_negative_taxonomy_mode=str(teacher_config.get("hard_negative_taxonomy_mode") or "none"),
            same_action_universe_lookup=same_action_universe_lookup,
        )
        if case_rows:
            cases_with_pairs += 1
            pair_rows.extend(case_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    open_fn = gzip.open if out_path.suffix == ".gz" else open
    with open_fn(out_path, "wt") as handle:
        for row in pair_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "manifest_path": str(manifest_path),
        "snapshot_cache_root": str(snapshot_cache_root),
        "snapshot_catalog_path": str(snapshot_catalog_path) if snapshot_catalog_path else "",
        "teacher_recipe": str(teacher_config.get("teacher_recipe") or "explicit_flags"),
        "case_count_manifest": len(cases),
        "selection_count": len(selection_rows),
        "cases_with_pairs": cases_with_pairs,
        "pair_row_count": len(pair_rows),
        "top_k_per_candidate": int(args.top_k_per_candidate),
        "positive_limit_per_source": int(args.positive_limit_per_source),
        "negative_limit_per_competitor": int(args.negative_limit_per_competitor),
        "same_family_negatives_only_if_available": bool(
            teacher_config.get("same_family_negatives_only_if_available")
        ),
        "always_include_actual_anchor_positive": bool(
            teacher_config.get("always_include_actual_anchor_positive")
        ),
        "include_within_action_hard_negatives": bool(
            teacher_config.get("include_within_action_hard_negatives")
        ),
        "include_same_action_positive_ordering": bool(
            teacher_config.get("include_same_action_positive_ordering")
        ),
        "actual_anchor_within_action_negative_source": str(
            teacher_config.get("actual_anchor_within_action_negative_source") or "retrieved_pool"
        ),
        "positive_source_mode": str(teacher_config.get("positive_source_mode") or "include_retrieved"),
        "hard_negative_taxonomy_mode": str(teacher_config.get("hard_negative_taxonomy_mode") or "none"),
        "analog_regime_cluster_grid": _parse_int_grid(
            str(args.analog_regime_cluster_grid or ""),
            default=(2, 3, 4, 5, 6),
        ),
        "analog_regime_seed": int(args.analog_regime_seed),
        "analog_regime_max_iter": int(args.analog_regime_max_iter),
        "anchor_outcomes_lookup_size": int(len(anchor_outcomes_lookup)),
        "same_action_universe_lookup_size": int(len(same_action_universe_lookup)),
        "same_action_universe_action_count": int(len(same_action_universe_lookup)),
        "same_action_universe_row_count_total": int(
            sum(len((payload or {}).get("rows") or []) for payload in same_action_universe_lookup.values())
        ),
        "same_action_universe_row_count_by_action": {
            str(action_id): int(len((payload or {}).get("rows") or []))
            for action_id, payload in sorted(same_action_universe_lookup.items())
        },
        "precedent_outcomes_lookup_size": int(len(precedent_outcomes_lookup)),
        "runs_root": str(runs_root) if runs_root else "",
        "eval_prefix": eval_prefix,
        "eval_id": eval_id,
        "out_path": str(out_path),
        "target_source_counts": dict(target_source_counts),
        "pair_source_counts": dict(Counter(str(row.get("pair_source") or "") for row in pair_rows)),
        "positive_source_counts": dict(Counter(str(row.get("positive_source") or "") for row in pair_rows)),
        "negative_source_counts": dict(Counter(str(row.get("negative_source") or "") for row in pair_rows)),
        "within_action_pair_row_count": int(
            sum(1 for row in pair_rows if str(row.get("competitor_action_id") or "") == str(row.get("anchor_action_id") or ""))
        ),
        "competitor_action_pair_row_count": int(
            sum(1 for row in pair_rows if str(row.get("competitor_action_id") or "") != str(row.get("anchor_action_id") or ""))
        ),
        "case_row_counts": {
            f"{company_id}|{as_of_time}|{anchor_action_id}": count
            for (company_id, as_of_time, anchor_action_id), count in Counter(
                (
                    str(pair_row.get("company_id") or ""),
                    str(pair_row.get("as_of_time") or ""),
                    str(pair_row.get("anchor_action_id") or ""),
                )
                for pair_row in pair_rows
            ).items()
        },
    }
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
