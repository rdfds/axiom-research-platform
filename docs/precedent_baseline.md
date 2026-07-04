# Precedent Baseline

Validated on March 13, 2026.

## Default Corpus

Use this dataset for precedent retrieval by default:

- `./data/curated/action_outcomes_with_credit_ratings.normalized_full.parquet`

Do not use this file as the precedent default:

- `./data/curated/action_outcomes_with_credit_ratings.parquet`

Reason:

- the rebuilt credit-ratings file does not currently preserve the full historical action universe needed for precedent quality
- the `normalized_full` corpus preserves full coverage and adds lossless normalized action columns

## Baseline Code Paths

Current precedent baseline is defined by:

- `./src/pipeline/precedent_brain.py`
- `./src/pipeline/run.py`
- `./src/action_normalization.py`

Default precedent CLI/API entrypoints now resolve to the normalized full corpus unless overridden:

- `./scripts/run_recommendation_prod.py`
- `./scripts/run_precedent_only.py`
- `./scripts/benchmark_precedent_families.py`
- `./scripts/run_precedent_api.py`
- `./scripts/run_recommendation_run_api.py`
- `./scripts/execute_recommendation_run.py`
- `./scripts/50_run_precedent_pipeline.py`

## Validation

### 5-Company Batch

Current baseline batch:

Source:

- `/tmp/ml_status_precedent_normfull_v2.json`

Metrics:

- `precedent_conf_mean = 0.353771`
- `precedent_oos_mean = 0.216`
- `causal_rate_mean = 0.79`
- `strict_causal_mean = 1.0`

Tier mix:

- `exact = 55`
- `family = 70`
- `sibling_type = 0`

This is the current accepted precedent baseline.

### 20-Company Broad Sample

Source:

- `/tmp/ml_status_precedent_normfull_20.json`

Metrics:

- `precedent_conf_mean = 0.346849`
- `precedent_oos_mean = 0.308`
- `causal_rate_mean = 0.800167`
- `strict_causal_mean = 1.0`

Tier mix:

- `exact = 208`
- `family = 292`
- `sibling_type = 0`

Interpretation:

- the 5-company regression result holds on a broader 20-company sample
- confidence is slightly lower than the fixed regression set, which is expected on broader coverage
- no fallback collapse to `sibling_type` appeared in the broader sample

### Prior Comparison Batch

Source:

- `/tmp/ml_status_precedent_normfull_v1.json`

Metrics:

- `precedent_conf_mean = 0.345856`
- `precedent_oos_mean = 0.328`
- `causal_rate_mean = 0.79`
- `strict_causal_mean = 1.0`

Tier mix from that batch:

- `exact = 55`
- `family = 55`
- `sibling_type = 15`

At that point, every `sibling_type` case was `capital_structure.convertible_issuance`.

### Post-Batch Targeted Fix

Convertible issuance was patched after the 5-company batch and validated separately.

Source:

- `/tmp/precedent_convertible_check_v2.json`

Result:

- `action_id = capital_structure.convertible_issuance`
- `retrieval_tier = family`
- `precedent_conf_mean = 0.46208`
- `oos_rate = 0.0`
- `selected_family_scale_keys = capital_structure.equity_issuance.scale_small`

### Targeted Family Checks

Source:

- `/tmp/precedent_targeted_norm_full_v4_benchmark.json`

Validated families:

- `capital_structure.new_debt_issuance`
- `capital_structure.refinancing`
- `capital_structure.convertible_issuance`
- `mna.platform_acquisition`
- `mna.tuck_in_acquisition`
- `portfolio.divestiture_partial`
- `capital_return.open_market_buyback`
- `capital_return.accelerated_share_repurchase`

All targeted checks are currently `oos = 0.0`.

## Operational Notes

- precedent quality work is separated from runtime work
- remaining runtime slowness is wrapper/orchestration overhead, not the core retriever
- if future runs regress to `sibling_type`, audit the action ids first before changing scoring
- causal production notes and exceptions are documented in `./docs/causal_baseline.md`
- broader non-regression monitoring is documented in `./docs/model_monitoring.md`

## Newly Added Standard Actions

The following actions are now wired into the ontology and normalized precedent corpus:

- `capital_return.dividend_initiate`
- `mna.go_private_lbo`

Supporting code paths:

- `./src/action_ontology.py`
- `./src/action_normalization.py`
- `./src/pipeline/precedent_brain.py`
- `./src/pipeline/run.py`

Corpus status in `./data/curated/action_outcomes_with_credit_ratings.normalized_full.parquet`:

- `capital_return.dividend_initiate`
  - exact normalized rows: `1044`
- `mna.go_private_lbo`
  - exact normalized acquisition-LBO rows: `1680`

Snapshot and candidate-generation support:

- `./src/company_state_builder.py`
  - adds:
    - `capital_return.dividend_payer_flag`
    - `capital_return.last_dividend_event_type`
- `./src/candidate_generation.py`
  - `capital_return.dividend_initiate` is only generated when:
    - `capital_return.dividend_payer_flag == False`
    - liquidity-excess conditions also hold

Validation:

- `0000320193`
  - `dividend_payer_flag = True`
  - `capital_return.dividend_initiate` not generated in natural candidate generation
- `0000794619`
  - `dividend_payer_flag = False`
  - `capital_return.dividend_initiate` generated in natural candidate generation

Status:

- `capital_return.dividend_initiate`
  - supported for ontology, normalized precedent corpus, and gated candidate generation
- `mna.go_private_lbo`
  - supported for ontology and normalized precedent corpus
  - targeted precedent probe:
    - source: `/tmp/precedent_go_private_lbo_probe.json`
    - `precedent_conf_mean = 0.314205`
    - `oos_rate = 0.0`
    - `retrieval_tier = exact`
    - `candidate_pool_size_after_prefilter = 34.0`
  - promoted into the built-in targeted precedent benchmark preset in:
    - `./scripts/benchmark_precedent_families.py`

## Refresh Commands

Targeted benchmark:

```bash
cd .

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=. \
python -u ./scripts/benchmark_precedent_families.py \
  --run-id ba18753a-59fc-4f91-8650-d73f3025adeb \
  --runs-root /tmp/recommendation_runs_v4_clean \
  --precedent-top-k 1 \
  --artifact-prefix precedent_targeted_norm_full_v4 \
  --out /tmp/precedent_targeted_norm_full_v4_benchmark.json
```

5-company validation batch:

```bash
cd .

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
RECO_PRECEDENT_MAX_PER_ACTION=3 \
RECO_PRECEDENT_MAX_PER_DEBT_ACTION=1 \
PYTHONPATH=. \
python -u ./scripts/run_recommendation_prod.py \
  --runs-root /tmp/recommendation_runs_prod_precedent_normfull_v1 \
  --companies 0000320193 0000789019 0001652044 0001018724 0001326801 \
  --causal-model-path ./data/models/causal_impact_model_v5_5_hybrid.json \
  --causal-action-blocklist-path /tmp/causal_blocklist_prod.txt \
  --precedent-workers 6 \
  --run-ids-out /tmp/recommendation_runs_prod_precedent_normfull_v1_run_ids.txt \
  --summary-out /tmp/recommendation_runs_prod_precedent_normfull_v1_summary.json
```
