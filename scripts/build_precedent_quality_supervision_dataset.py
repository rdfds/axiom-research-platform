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
            flattened[str(key)] = payload['value']
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


