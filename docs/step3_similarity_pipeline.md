# Step 3 — Similarity / Scoring Pipeline

This step consumes **CompanyState** and produces similarity matches + outcomes.

## Inputs

- `data/company_state/company_state.parquet` (long)
- `data/curated/similarity_*` (features, weights, hyperparams)

## Core Outputs

- Similarity search results (top‑k analogs)
- Predicted outcomes (distance‑weighted)
- Validation summaries (MAE / corr / sign hit)

## Pipeline Overview

1. **Feature engineering**
   - Assemble `similarity_features.parquet`
   - Derived regime features / stability / growth / valuation

2. **Target selection**
   - `scripts/64_select_best_targets.py`

3. **Weight learning**
   - `scripts/62_learn_similarity_weights.py`

4. **Hyperparam tuning**
   - `scripts/66_optimize_similarity_hyperparams.py`

5. **Similarity search**
   - `scripts/61_similarity_search.py`

6. **Validation**
   - `scripts/63_validate_similarity.py`

## Notes

- CompanyState provides the consistent, as‑of snapshot.
- Similarity pipeline should reference CompanyState (not raw inputs).

