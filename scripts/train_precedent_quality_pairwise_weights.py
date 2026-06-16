#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.precedent_quality_learning import (
    build_pairwise_matrix,
    build_outcome_aware_reranker_matrix,
    build_scope_payload_with_pairwise_weights,
    cross_validate_pairwise_precedent_quality_weights,
    cross_validate_outcome_aware_reranker,
    fit_nonnegative_pairwise_logistic,
    learn_feature_transforms_from_pairwise_supervision,
    load_feature_transform_prior,
    load_feature_weight_prior,
    _outcome_aware_reranker_prior,
    load_penalty_feature_specs,
    load_pairwise_supervision,
    search_latent_regime_models_from_supervision,
    search_pairwise_interactions_from_supervision,
    search_target_regime_mixture_from_supervision,
    write_json,
)
from src.pipeline.run import _default_precedent_outcomes_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pairwise precedent-quality weights for a scope.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--scope-key", required=True)
    parser.add_argument("--base-payload-path", required=True)
    parser.add_argument("--out-payload-path", required=True)
    parser.add_argument("--out-summary-path", default="")
    parser.add_argument("--min-feature-coverage-rows", type=int, default=20)
    parser.add_argument("--l2-grid", default="0.25,0.5,1.0,2.0,4.0,8.0")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--learn-feature-transforms", action="store_true")
    parser.add_argument("--feature-transform-mode", choices=("default", "identity"), default="")
    parser.add_argument(
        "--pair-weight-mode",
        choices=("uniform", "teacher_confidence", "target_regime_rarity"),
        default="uniform",
    )
    parser.add_argument("--include-runtime-penalties", action="store_true")
    parser.add_argument("--transform-search-l2-grid", default="0.25,1.0,4.0")
    parser.add_argument("--transform-search-learning-rate", type=float, default=0.05)
    parser.add_argument("--transform-search-max-iter", type=int, default=2000)
    parser.add_argument("--include-interactions", action="store_true")
    parser.add_argument("--search-interactions", action="store_true")
    parser.add_argument("--max-interaction-terms", type=int, default=6)
    parser.add_argument("--interaction-feature-names", default="")
    parser.add_argument("--search-latent-regimes", action="store_true")
    parser.add_argument("--search-target-regime-mixture", action="store_true")
    parser.add_argument("--latent-regime-cluster-grid", default="2,3,4,5,6")
    parser.add_argument("--latent-regime-seed", type=int, default=7)
    parser.add_argument("--latent-regime-max-iter", type=int, default=100)
    parser.add_argument("--train-outcome-aware-reranker", action="store_true")
    parser.add_argument("--outcomes-path", default="")
    parser.add_argument("--outcome-aware-shortlist-size", type=int, default=40)
    return parser.parse_args()


def _parse_grid(value: str) -> Sequence[float]:
    values = [float(item.strip()) for item in str(value or "").split(",") if item.strip()]
    if not values:
        raise ValueError("l2 grid must include at least one value")
    return values


def _parse_int_grid(value: str) -> Sequence[int]:
    values = [int(item.strip()) for item in str(value or "").split(",") if item.strip()]
    if not values:
        raise ValueError("integer grid must include at least one value")
    return values


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset_path)
    base_payload_path = Path(args.base_payload_path)
    out_payload_path = Path(args.out_payload_path)
    out_summary_path = Path(args.out_summary_path) if args.out_summary_path else None
    l2_grid = _parse_grid(args.l2_grid)
    transform_search_l2_grid = _parse_grid(args.transform_search_l2_grid)
    latent_regime_cluster_grid = _parse_int_grid(args.latent_regime_cluster_grid)
    explicit_interaction_feature_names = [
        str(item.strip())
        for item in str(args.interaction_feature_names or "").split(",")
        if str(item).strip()
    ]
    requested_feature_transform_mode = str(args.feature_transform_mode or "").strip().lower() or None
    pair_weight_mode = str(args.pair_weight_mode or "uniform").strip().lower() or "uniform"
    outcomes_path = Path(args.outcomes_path) if str(args.outcomes_path or "").strip() else Path(_default_precedent_outcomes_path())
    penalty_feature_specs = (
        load_penalty_feature_specs(base_payload_path, scope_key=args.scope_key)
        if bool(args.include_runtime_penalties)
        else None
    )

    learned_feature_transforms = None
    transform_search_summary: Dict[str, Any] = {}
    if args.learn_feature_transforms:
        transform_search = learn_feature_transforms_from_pairwise_supervision(
            dataset_path,
            scope_key=args.scope_key,
            base_payload_path=base_payload_path,
            min_feature_coverage_rows=int(args.min_feature_coverage_rows),
            l2_grid=transform_search_l2_grid,
            learning_rate=float(args.transform_search_learning_rate),
            max_iter=int(args.transform_search_max_iter),
            feature_transform_mode=requested_feature_transform_mode,
            pair_weight_mode=pair_weight_mode,
            penalty_feature_specs=penalty_feature_specs,
        )
        learned_feature_transforms = dict(transform_search["chosen_feature_transforms"] or {})
        transform_search_summary = transform_search
    active_transform_specs = (
        dict(learned_feature_transforms)
        if learned_feature_transforms is not None
        else load_feature_transform_prior(
            base_payload_path,
            scope_key=args.scope_key,
            feature_transform_mode=requested_feature_transform_mode,
        )
    )

    interaction_search_summary: Dict[str, Any] = {}
    chosen_interaction_features = explicit_interaction_feature_names or None
    if args.search_interactions:
        interaction_search = search_pairwise_interactions_from_supervision(
            dataset_path,
            scope_key=args.scope_key,
            base_payload_path=base_payload_path,
            min_feature_coverage_rows=int(args.min_feature_coverage_rows),
            l2_grid=l2_grid,
            learning_rate=float(args.learning_rate),
            max_iter=int(args.max_iter),
            transform_specs=active_transform_specs,
            pair_weight_mode=pair_weight_mode,
            max_interaction_terms=int(args.max_interaction_terms),
            penalty_feature_specs=penalty_feature_specs,
        )
        chosen_interaction_features = list(interaction_search["chosen_interactions"] or [])
        interaction_search_summary = interaction_search

    use_interactions = bool(args.include_interactions or chosen_interaction_features)
    latent_regime_search_summary: Dict[str, Any] = {}
    learned_latent_regime_model = None
    chosen_latent_feature_name = None
    target_regime_search_summary: Dict[str, Any] = {}
    learned_target_regime_payload = None

    if args.search_target_regime_mixture:
        target_regime_search = search_target_regime_mixture_from_supervision(
            dataset_path,
            scope_key=args.scope_key,
            base_payload_path=base_payload_path,
            min_feature_coverage_rows=int(args.min_feature_coverage_rows),
            l2_grid=l2_grid,
            learning_rate=float(args.learning_rate),
            max_iter=int(args.max_iter),
            transform_specs=active_transform_specs,
            pair_weight_mode=pair_weight_mode,
            include_interactions=use_interactions,
            interaction_feature_names=chosen_interaction_features,
            penalty_feature_specs=penalty_feature_specs,
            n_cluster_grid=latent_regime_cluster_grid,
            seed=int(args.latent_regime_seed),
            latent_max_iter=int(args.latent_regime_max_iter),
        )
        target_regime_search_summary = target_regime_search
        learned_target_regime_payload = {
            "model": dict(target_regime_search.get("chosen_target_regime_model") or {}),
            "regimes": list(target_regime_search.get("chosen_target_regime_payload") or []),
        }
        cv = target_regime_search
    elif args.search_latent_regimes:
        latent_regime_search = search_latent_regime_models_from_supervision(
            dataset_path,
            scope_key=args.scope_key,
            base_payload_path=base_payload_path,
            min_feature_coverage_rows=int(args.min_feature_coverage_rows),
            l2_grid=l2_grid,
            learning_rate=float(args.learning_rate),
            max_iter=int(args.max_iter),
            transform_specs=active_transform_specs,
            pair_weight_mode=pair_weight_mode,
            include_interactions=use_interactions,
            interaction_feature_names=chosen_interaction_features,
            penalty_feature_specs=penalty_feature_specs,
            n_cluster_grid=latent_regime_cluster_grid,
            seed=int(args.latent_regime_seed),
            latent_max_iter=int(args.latent_regime_max_iter),
        )
        latent_regime_search_summary = latent_regime_search
        learned_latent_regime_model = dict(latent_regime_search.get("chosen_latent_regime_model") or {})
        chosen_latent_feature_name = str(latent_regime_search.get("chosen_latent_feature_name") or "")
        cv = latent_regime_search
    else:
        cv = cross_validate_pairwise_precedent_quality_weights(
            dataset_path,
            scope_key=args.scope_key,
            base_payload_path=base_payload_path,
            min_feature_coverage_rows=int(args.min_feature_coverage_rows),
            l2_grid=l2_grid,
            learning_rate=float(args.learning_rate),
            max_iter=int(args.max_iter),
            transform_specs=active_transform_specs,
            pair_weight_mode=pair_weight_mode,
            include_interactions=use_interactions,
            interaction_feature_names=chosen_interaction_features,
            penalty_feature_specs=penalty_feature_specs,
        )
    best_eval = dict(cv["best_evaluation"] or {})
    selected_features = list(cv["selected_features"])
    best_lambda = float(best_eval.get("l2_lambda") or 1.0)

    prior = load_feature_weight_prior(
        base_payload_path,
        scope_key=args.scope_key,
        feature_names=selected_features,
    )
    if args.search_target_regime_mixture:
        full_fit = {"weights": np.asarray(prior, dtype=float), "bias": 0.0}
        learned_weights = {
            feature: float(prior[idx]) for idx, feature in enumerate(selected_features)
        }
    else:
        df = load_pairwise_supervision(dataset_path)
        matrix = build_pairwise_matrix(
            df,
            feature_names=selected_features,
            min_feature_coverage_rows=int(args.min_feature_coverage_rows),
            transform_specs=active_transform_specs,
            pair_weight_mode=pair_weight_mode,
            include_interactions=use_interactions,
            interaction_feature_names=chosen_interaction_features,
            penalty_feature_specs=penalty_feature_specs,
            include_latent_regime=bool(learned_latent_regime_model),
            latent_regime_model=learned_latent_regime_model,
            latent_feature_names=[chosen_latent_feature_name] if chosen_latent_feature_name else None,
            enforce_feature_names=bool(learned_latent_regime_model),
        )
        X = np.asarray(matrix["X"], dtype=float)
        y = np.asarray(matrix["y"], dtype=float)
        sample_weights = np.asarray(matrix["sample_weights"], dtype=float)
        full_fit = fit_nonnegative_pairwise_logistic(
            X,
            y,
            prior=prior,
            sample_weights=sample_weights,
            l2_lambda=best_lambda,
            learning_rate=float(args.learning_rate),
            max_iter=int(args.max_iter),
        )
        learned_weights = {
            feature: float(full_fit["weights"][idx]) for idx, feature in enumerate(selected_features)
        }

    outcome_aware_summary: Dict[str, Any] = {}
    learned_outcome_aware_reranker = None
    if args.train_outcome_aware_reranker:
        outcome_aware_cv = cross_validate_outcome_aware_reranker(
            dataset_path,
            outcomes_path=outcomes_path,
            pair_weight_mode=pair_weight_mode,
            same_action_only=True,
            l2_grid=l2_grid,
            learning_rate=float(args.learning_rate),
            max_iter=int(args.max_iter),
        )
        outcome_best_eval = dict(outcome_aware_cv["best_evaluation"] or {})
        outcome_recommend_promote = bool(
            float(outcome_best_eval.get("mean_log_loss_improvement") or 0.0) > 0.005
            and float(outcome_best_eval.get("mean_accuracy_improvement") or 0.0) >= 0.0
        )
        outcome_df = load_pairwise_supervision(dataset_path)
        outcomes_df = pd.read_parquet(outcomes_path)
        outcome_matrix = build_outcome_aware_reranker_matrix(
            outcome_df,
            outcomes_df=outcomes_df,
            pair_weight_mode=pair_weight_mode,
            same_action_only=True,
        )
        outcome_features = list(outcome_matrix["selected_features"])
        outcome_prior = _outcome_aware_reranker_prior(outcome_features)
        outcome_fit = fit_nonnegative_pairwise_logistic(
            np.asarray(outcome_matrix["X"], dtype=float),
            np.asarray(outcome_matrix["y"], dtype=float),
            prior=outcome_prior,
            sample_weights=np.asarray(outcome_matrix["sample_weights"], dtype=float),
            l2_lambda=float(outcome_best_eval.get("l2_lambda") or 1.0),
            learning_rate=float(args.learning_rate),
            max_iter=int(args.max_iter),
            min_weights=np.zeros(len(outcome_features), dtype=float),
        )
        candidate_outcome_aware_reranker = {
            "feature_weights": {
                feature: float(outcome_fit["weights"][idx]) for idx, feature in enumerate(outcome_features)
            },
            "bias": float(outcome_fit["bias"]),
            "shortlist_size": int(args.outcome_aware_shortlist_size),
        }
        learned_outcome_aware_reranker = (
            candidate_outcome_aware_reranker if outcome_recommend_promote else None
        )
        outcome_aware_summary = {
            **outcome_aware_cv,
            "outcomes_path": str(outcomes_path),
            "shortlist_size": int(args.outcome_aware_shortlist_size),
            "full_fit_weights": dict(candidate_outcome_aware_reranker["feature_weights"]),
            "full_fit_bias": float(candidate_outcome_aware_reranker["bias"]),
            "promote_candidate": bool(outcome_recommend_promote),
            "promotion_gate": {
                "min_log_loss_improvement": 0.005,
                "min_accuracy_improvement": 0.0,
            },
        }

    recommend_promote = bool(
        float(best_eval.get("mean_log_loss_improvement") or 0.0) > 0.005
        and float(best_eval.get("mean_accuracy_improvement") or 0.0) >= 0.0
    )
    summary: Dict[str, Any] = {
        "scope_key": str(args.scope_key),
        "dataset_path": str(dataset_path),
        "base_payload_path": str(base_payload_path),
        "selected_features": selected_features,
        "feature_coverage": cv["feature_coverage"],
        "pair_count": int(cv["pair_count"]),
        "group_count": int(cv["group_count"]),
        "cv_best_evaluation": best_eval,
        "cv_evaluation_count": len(cv["evaluations"]),
        "full_fit_weights": learned_weights,
        "full_fit_bias": float(full_fit["bias"]),
        "promote_candidate": recommend_promote,
        "promotion_gate": {
            "min_log_loss_improvement": 0.005,
            "min_accuracy_improvement": 0.0,
        },
        "learn_feature_transforms": bool(args.learn_feature_transforms),
        "include_interactions": bool(use_interactions),
        "search_interactions": bool(args.search_interactions),
        "max_interaction_terms": int(args.max_interaction_terms),
        "chosen_interaction_features": list(chosen_interaction_features or []),
        "explicit_interaction_feature_names": list(explicit_interaction_feature_names),
        "search_latent_regimes": bool(args.search_latent_regimes),
        "search_target_regime_mixture": bool(args.search_target_regime_mixture),
        "latent_regime_cluster_grid": [int(v) for v in latent_regime_cluster_grid],
        "latent_regime_seed": int(args.latent_regime_seed),
        "latent_regime_max_iter": int(args.latent_regime_max_iter),
        "chosen_latent_feature_name": str(chosen_latent_feature_name or ""),
        "chosen_latent_regime_model": dict(learned_latent_regime_model or {}),
        "latent_regime_search_summary": latent_regime_search_summary,
        "chosen_target_regime_payload": dict(learned_target_regime_payload or {}),
        "target_regime_search_summary": target_regime_search_summary,
        "learned_feature_transforms": {
            feature: dict(spec) for feature, spec in dict(learned_feature_transforms or {}).items()
        },
        "requested_feature_transform_mode": str(requested_feature_transform_mode or ""),
        "pair_weight_mode": pair_weight_mode,
        "include_runtime_penalties": bool(args.include_runtime_penalties),
        "penalty_feature_specs": [dict(spec) for spec in list(penalty_feature_specs or [])],
        "active_feature_transforms": {
            feature: dict(spec) for feature, spec in dict(active_transform_specs or {}).items()
        },
        "transform_search_summary": transform_search_summary,
        "interaction_search_summary": interaction_search_summary,
        "transform_search_l2_grid": [float(v) for v in transform_search_l2_grid],
        "transform_search_learning_rate": float(args.transform_search_learning_rate),
        "transform_search_max_iter": int(args.transform_search_max_iter),
        "learning_rate": float(args.learning_rate),
        "max_iter": int(args.max_iter),
        "l2_grid": [float(v) for v in l2_grid],
        "train_outcome_aware_reranker": bool(args.train_outcome_aware_reranker),
        "outcomes_path": str(outcomes_path),
        "outcome_aware_shortlist_size": int(args.outcome_aware_shortlist_size),
        "outcome_aware_summary": outcome_aware_summary,
    }
    payload = build_scope_payload_with_pairwise_weights(
        base_payload_path,
        scope_key=args.scope_key,
        learned_weights=learned_weights,
        learned_feature_transforms=learned_feature_transforms,
        feature_transform_mode=requested_feature_transform_mode,
        latent_regime_model=learned_latent_regime_model,
        target_regime_payload=learned_target_regime_payload,
        outcome_aware_reranker=learned_outcome_aware_reranker,
        notes=summary,
    )
    payload["scopes"][str(args.scope_key)]["use_in_runtime"] = True
    payload["scopes"][str(args.scope_key)]["default_enabled"] = True
    payload.setdefault("notes", {})
    payload["notes"]["pairwise_precedent_quality_learning_summary"] = {
        "scope_key": str(args.scope_key),
        "dataset_path": str(dataset_path),
        "out_summary_path": str(out_summary_path) if out_summary_path else "",
        "promote_candidate": recommend_promote,
    }

    write_json(payload, out_payload_path)
    if out_summary_path:
        out_summary_path.parent.mkdir(parents=True, exist_ok=True)
        out_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
