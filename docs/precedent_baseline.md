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

