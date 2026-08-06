from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.pipeline.precedent_quality_learning import (
    _second_stage_reranker_prior,
    build_scope_payload_with_pairwise_weights,
    build_second_stage_reranker_matrix,
    cross_validate_second_stage_reranker,
    fit_nonnegative_pairwise_logistic,
    load_pairwise_supervision,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a second-stage precedent reranker from pairwise supervision.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--scope-key", required=True)
    parser.add_argument("--base-payload-path", required=True)
    parser.add_argument("--out-payload-path", required=True)
    parser.add_argument("--out-summary-path", required=True)
    parser.add_argument("--pair-weight-mode", default="uniform", choices=("uniform", "teacher_confidence", "target_regime_rarity"))
    parser.add_argument("--shortlist-size", type=int, default=120)
    parser.add_argument("--same-action-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=4000)
    args = parser.parse_args()

    cv = cross_validate_second_stage_reranker(
        args.dataset_path,
        pair_weight_mode=args.pair_weight_mode,
        same_action_only=bool(args.same_action_only),
        learning_rate=float(args.learning_rate),
        max_iter=int(args.max_iter),
    )
    best_evaluation = dict(cv.get("best_evaluation") or {})
    best_l2 = float(best_evaluation.get("l2_lambda") or 1.0)

    df = load_pairwise_supervision(args.dataset_path)
    matrix = build_second_stage_reranker_matrix(
        df,
        pair_weight_mode=args.pair_weight_mode,
        same_action_only=bool(args.same_action_only),
    )
    selected_features = list(matrix["selected_features"])
    prior = _second_stage_reranker_prior(selected_features)
    min_weights = np.zeros(len(selected_features), dtype=float)
    fit = fit_nonnegative_pairwise_logistic(
        np.asarray(matrix["X"], dtype=float),
        np.asarray(matrix["y"], dtype=float),
        prior=prior,
        sample_weights=np.asarray(matrix["sample_weights"], dtype=float),
        l2_lambda=best_l2,
        learning_rate=float(args.learning_rate),
        max_iter=int(args.max_iter),
        min_weights=min_weights,
    )
    reranker_payload = {
        "feature_weights": {
            feature: float(np.asarray(fit["weights"], dtype=float)[idx])
            for idx, feature in enumerate(selected_features)
        },
        "bias": float(fit["bias"]),
        "shortlist_size": int(args.shortlist_size),
    }
    payload = build_scope_payload_with_pairwise_weights(
        args.base_payload_path,
        scope_key=args.scope_key,
        learned_weights={},
        second_stage_reranker=reranker_payload,
        notes={
            "second_stage_reranker_dataset_path": str(args.dataset_path),
            "second_stage_reranker_pair_weight_mode": str(args.pair_weight_mode),
            "second_stage_reranker_same_action_only": bool(args.same_action_only),
            "second_stage_reranker_best_l2": best_l2,
        },
    )
    summary = {
        "dataset_path": str(args.dataset_path),
        "scope_key": str(args.scope_key),
        "pair_weight_mode": str(args.pair_weight_mode),
        "same_action_only": bool(args.same_action_only),
        "shortlist_size": int(args.shortlist_size),
        "pair_count": int(cv.get("pair_count") or 0),
        "group_count": int(cv.get("group_count") or 0),
        "selected_features": selected_features,
        "feature_coverage": dict(cv.get("feature_coverage") or {}),
        "best_evaluation": best_evaluation,
        "full_fit_feature_weights": dict(reranker_payload["feature_weights"]),
        "full_fit_bias": float(reranker_payload["bias"]),
        "out_payload_path": str(Path(args.out_payload_path)),
    }
    write_json(payload, args.out_payload_path)
    write_json(summary, args.out_summary_path)


if __name__ == "__main__":
    main()
