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

