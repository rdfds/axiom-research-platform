# Model Monitoring Contract

This document defines the review and promotion contract for Axiom model and policy changes. It separates checks reproducible from committed fixtures from provider-backed checks that require licensed data.

## 1. Data contract

Before scoring or training, validate:

- as-of timestamps and publication dates are compatible with the decision date;
- entity identifiers resolve without ambiguous joins;
- units, currencies, and fiscal periods are normalized;
- required provenance and confidence fields are present;
- missingness and fallback rates stay within the candidate artifact's declared bounds.

A material schema, coverage, or fallback-rate change blocks promotion until the affected slices are reviewed.

## 2. Temporal performance

Temporal claims require time-ordered evaluation. Record:

- training cutoff and fixed test window;
- eligible population and excluded sectors or families;
- baseline and candidate metrics by slice, not only in aggregate;
- candidate count, successful evaluation count, and failure reasons.

The committed forward-gap benchmark uses four training cutoffs and two-year test windows. Its exact result is stored in [`../results/public_benchmark.json`](../results/public_benchmark.json).

## 3. Baseline and placebo gates

A candidate must improve on the current baseline and pass the applicable negative control. For the committed forward-gap policy, the negative control shuffles gap values within each driver/horizon panel before scoring the same validation split.

```bash
python scripts/build_public_benchmark.py --check
```

Review aggregate lift, pass rate, and family/sector slices. A positive aggregate result does not override a severe regression in a high-support slice.

## 4. Retrieval, support, and calibration

For learned precedent ranking and action-evidence components, review:

- ranking quality against the accepted baseline;
- support coverage and out-of-support rate;
- calibration and confidence by action family;
- fallback frequency and mismatch reasons;
- behavior on intentionally sparse, contradictory, or low-quality evidence.

If support is insufficient, the correct behavior is a broader cohort, a lower-confidence result, or no claim—not an extrapolated high-confidence score.

## 5. Product-contract checks

Model output must survive into the final recommendation contract without losing provenance or limitations. CI therefore covers company-state validation, feature bundles, ranking and backtest behavior, learned-quality evaluation, orchestration, planner behavior, dossier evaluation, and end-to-end examples.

These tests establish deterministic implementation behavior on committed fixtures; provider-scale coverage and operational metrics are evaluated in their deployment environments.

## 6. Drift and operational monitoring

Provider-backed deployments should track:

- schema and missingness drift;
- feature-distribution and support-coverage drift;
- ranking, calibration, and baseline-relative performance by slice;
- run failures, latency, and fallback rates;
- changes in recommendation mix or confidence distribution.

Alert thresholds belong with the versioned candidate artifact and evaluation manifest. Provider paths, credentials, and live operational metrics remain in their deployment environments.

