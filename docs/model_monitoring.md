# Model Monitoring

Validated on March 14, 2026.

## Scope

This is the current broader monitoring path for the accepted production baseline:

- precedent corpus:
  - `./data/curated/action_outcomes_with_credit_ratings.normalized_full.parquet`
- causal model:
  - `./data/models/causal_impact_model_v5_5_hybrid.json`
- causal blocklist:
  - `./config/causal_action_blocklist_prod_v2.txt`

## Broad Canary

Use the broader 20-company sample as the non-regression canary outside the fixed 5-company set.

Latest accepted gate:

- source runs:
  - `/tmp/recommendation_runs_prod_causal_v2_20`
- source run IDs:
  - `/tmp/recommendation_runs_prod_causal_v2_20_run_ids.txt`
- gate output:
  - `/tmp/recommendation_canary_gate_20.json`

Latest gate metrics:

- `runs_analyzed = 20`
- `causal_rate_mean = 0.848167`
- `strict_all_mean = 0.848167`
- `strict_causal_mean = 1.0`
- `precedent_conf_mean = 0.346832`
- `precedent_oos_mean = 0.308`

Thresholds used:

- `min_causal_rate_mean = 0.82`
- `min_strict_all_mean = 0.82`
- `min_strict_causal_mean = 0.95`
- `min_precedent_conf_mean = 0.34`
- `max_precedent_oos_mean = 0.33`

## Gate Command

```bash
cd .

PYTHONPATH=. \
python ./scripts/gate_recommendation_canary.py \
  --runs-roots /tmp/recommendation_runs_prod_causal_v2_20 \
  --run-ids-file /tmp/recommendation_runs_prod_causal_v2_20_run_ids.txt \
  --out /tmp/recommendation_canary_gate_20.json \
  --min-action-rows 1 \
  --min-causal-rate-mean 0.82 \
  --min-strict-all-mean 0.82 \
  --min-strict-causal-mean 0.95 \
  --min-precedent-conf-mean 0.34 \
  --max-precedent-oos-mean 0.33
```

## Decision Rule

- if the gate passes, the current production baseline is still healthy on the broader canary
- if the gate fails:
  - inspect action-level causal rows with `./scripts/audit_full_ml_status.py`
  - check whether the regression is:
    - precedent quality
    - causal coverage
    - strict-gate quality
    - or run failure / pipeline health
