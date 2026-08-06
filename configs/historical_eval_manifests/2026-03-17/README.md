# Historical Eval Manifests

These files freeze the exact `cases` lists used for the March 17, 2026
historical evaluation workflow.

Use them with:

```bash
PYTHONPATH=. \
python ./scripts/evaluate_historical_recommendation_quality.py \
  --fixed-cases-json ./configs/historical_eval_manifests/2026-03-17/capital_return_holdout_25.json \
  ...
```

Reference manifests:

- `capital_return_dev_25.json`
- `capital_return_holdout_25.json`
- `capital_structure_dev_25.json`
- `capital_structure_holdout_25.json`

## Best Frozen-Holdout Checkpoint

As of March 17, 2026, the best balanced frozen-holdout checkpoint is:

- capital return report: `/tmp/fixed_manifest_capreturn_holdout_buyback_recap_v36.json`
- capital structure report: `/tmp/fixed_manifest_capstructure_holdout_buyback_recap_v36.json`

Topline holdout metrics:

- capital return `mean_alignment_score = 0.831481`
- capital return `anchor_primary_exact_rate = 0.62963`
- capital return `anchor_primary_family_rate = 0.814815`
- capital structure `mean_alignment_score = 0.965517`
- capital structure `anchor_primary_exact_rate = 0.931034`
- capital structure `anchor_primary_family_rate = 0.965517`

Notes:

- These results come from replaying the frozen manifests above, not moving slices.
- The remaining capital-structure family leak is still `0000023197`, which
  appears to be a contradictory snapshot rather than a clean policy miss.

## Manual Replay Checkpoint

As of March 25, 2026, the best manual-replay checkpoint for the historical
replay hardening path is:

- capital return manual replay report: `/tmp/fixed_manifest_capreturn_holdout_manual_v42_noplanfix.json`
- capital structure manual replay report: `/tmp/fixed_manifest_capstructure_holdout_manual_v43_recaprank.json`

Topline manual-replay metrics:

- capital return `mean_alignment_score = 0.82963`
- capital return `anchor_primary_exact_rate = 0.259259`
- capital return `anchor_primary_family_rate = 0.814815`
- capital return `unsupported_case_count = 0`
- capital structure `mean_alignment_score = 0.764`
- capital structure `anchor_primary_exact_rate = 0.44`
- capital structure `anchor_primary_family_rate = 0.56`
- capital structure `unsupported_case_count = 4`

Notes:

- These runs use the manual replay harness with the local facts cache, not the
  standard frozen-holdout runner above.
- `v42` remains the capital-return replay checkpoint; `v43` is the improved
  capital-structure replay checkpoint after recap-ranking cleanup.
- The standard frozen-holdout benchmark checkpoint remains the `v36` pair listed
  above.

## Frozen Manual Replay Stack

The manual replay benchmark is now frozen in-repo instead of relying on a
`/tmp` script:

- runner: `./scripts/run_manual_replay_benchmark.py`
- lock config: `./configs/historical_eval_manifests/2026-03-17/manual_replay_benchmark_lock.json`

Example:

```bash
PYTHONPATH=. \
python \
  ./scripts/run_manual_replay_benchmark.py \
  --benchmark capital_return_holdout \
  --runs-root /tmp/manual_replay_capreturn_runs \
  --snapshot-cache-dir /tmp/manual_replay_capreturn_cache \
  --out-json /tmp/manual_replay_capreturn_report.json
```

The lock config pins:

- the outcomes artifact
- the action-support manifest
- the manual replay manifests
- the canonical methodology/config inputs
- the preferred facts-path resolution order

Each output report now records the resolved artifact paths and env overrides so
manual replay comparisons stop drifting silently.
