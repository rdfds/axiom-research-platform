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

