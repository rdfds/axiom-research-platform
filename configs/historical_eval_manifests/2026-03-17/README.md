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

