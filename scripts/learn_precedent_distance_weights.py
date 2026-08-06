#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.precedent_distance_learning import (  # noqa: E402
    learn_precedent_distance_weights,
    write_precedent_distance_weights,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn diagonal precedent-distance weights from historical outcomes.")
    parser.add_argument(
        "--outcomes-path",
        default=str(
            REPO_ROOT / "data/curated/action_outcomes_with_credit_ratings.normalized_full.rich_contract_v3.parquet"
        ),
    )
    parser.add_argument(
        "--out-path",
        default=str(REPO_ROOT / "data/curated/precedent_distance_weights_v1.json"),
    )
    parser.add_argument("--max-pairs", type=int, default=25000)
    parser.add_argument("--min-rows", type=int, default=1500)
    parser.add_argument("--min-state-coverage", type=float, default=0.60)
    parser.add_argument("--min-outcome-coverage", type=float, default=0.50)
    parser.add_argument("--min-outcome-non-null", type=int, default=800)
    parser.add_argument("--ridge-lambda", type=float, default=30.0)
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = learn_precedent_distance_weights(
        Path(args.outcomes_path),
        max_pairs=args.max_pairs,
        min_rows=args.min_rows,
        min_state_coverage=args.min_state_coverage,
        min_outcome_coverage=args.min_outcome_coverage,
        min_outcome_non_null=args.min_outcome_non_null,
        ridge_lambda=args.ridge_lambda,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
    )
    out_path = Path(args.out_path)
    write_precedent_distance_weights(payload, out_path)
    scopes = payload.get("scopes") or {}
    summary = {
        "ok": True,
        "out_path": str(out_path),
        "scope_count": int(len(scopes)),
        "scopes": {
            key: {
                "n_rows": int((value or {}).get("n_rows", 0) or 0),
                "n_pairs": int((value or {}).get("n_pairs", 0) or 0),
                "holdout_pair_correlation": (value or {}).get("holdout_pair_correlation"),
                "holdout_prior_pair_correlation": (value or {}).get("holdout_prior_pair_correlation"),
                "holdout_pair_correlation_improvement": (value or {}).get("holdout_pair_correlation_improvement"),
                "use_in_runtime": bool((value or {}).get("use_in_runtime")),
            }
            for key, value in scopes.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
