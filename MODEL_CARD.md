# Axiom Model Card

## Summary

Axiom is a point-in-time decision-evidence system for corporate-finance analysis. It combines timestamp-aligned company state, valuation-driver models, learned precedent retrieval, action-evidence scoring, and deterministic decision contracts. The public repository is a sanitized engineering showcase built around committed fixtures; private provider data and credentials are excluded.

The system is deliberately modular. A model score cannot silently become a recommendation: downstream contracts preserve provenance, support, uncertainty, fallbacks, and evidence limitations.

## Intended use

Axiom is designed to help an analyst investigate questions such as:

- Which operating or balance-sheet drivers best explain a peer-relative valuation gap?
- Which historical corporate actions are sufficiently comparable to review?
- How strong is the available evidence for an action, and where is support thin?
- What assumptions, objections, regret cases, and monitoring triggers should accompany a recommendation?

It is not designed to autonomously trade, provide investment advice, or establish causal effects from observational relationships. Human review is required before any real decision.

## System layers

| Layer | Inputs | Output and safeguards |
|---|---|---|
| Company state | Financial, market, filing, macro, and action observations | As-of feature snapshot with timing, units, provenance, confidence, and fallback flags |
| Valuation drivers | Company state and peer context | Driver surfaces and forward-gap policies evaluated on time-ordered splits |
| Precedent retrieval | Candidate actions and historical outcomes | Learned distance/reranking signals with mismatch reasons and cohort support |
| Action evidence | Observed actions, outcomes, and controls | Quality-gated evidence scores; causal language is withheld when identification is insufficient |
| Decision contract | Model outputs and evidence packs | Recommendation, sizing, objections, regret cases, and monitoring triggers |

## Public benchmark

The committed public benchmark evaluates a forward-gap regularization policy. It uses four walk-forward cutoffs, two-year test windows, and three shuffled-placebo runs. The selected global lambda is `0.5`.

| Measure | Result |
|---|---:|
| Successful driver/horizon evaluations | 912 |
| Candidate configurations attempted | 84 |
| Walk-forward training cutoffs | 4 |
| Placebo runs per validation split | 3 |
| Actual mean MAE improvement | 0.007413 |
| Lift over shuffled placebo | 0.007885 |
| Evaluations where actual beat placebo | 63.49% |

These values are parsed from the committed validation report into [`results/public_benchmark.json`](results/public_benchmark.json). CI fails if the report and artifact diverge:

```bash
python scripts/build_public_benchmark.py --check
```

The benchmark supports a narrow claim: the selected forward-gap policy improved the reported error measure more often than its shuffled placebo under this evaluation design. It does not by itself establish economic materiality, future performance, or causal identification.

## Evaluation design

- **Temporal integrity:** training cutoffs precede fixed two-year test windows; company-state contracts reject incompatible timing and provenance.
- **Baselines and placebos:** forward-gap claims are compared with an unregularized baseline and within-panel shuffled placebos.
- **Retrieval evaluation:** public tests cover ranking calibration, learned quality weights, head-to-head evaluation, and fallback behavior.
- **Product contracts:** orchestration, dossier evaluation, and showcase tests check that model evidence survives into deterministic downstream output.
- **Reproducibility:** the public contract runs entirely from committed fixtures without paid data-provider access.

See [`docs/validation/README.md`](docs/validation/README.md) for the broader validation inventory.

## Data boundaries

The original system was developed against licensed and private data sources. This repository contains schemas, transformations, model and evaluation code, and sanitized fixtures, but not the underlying private datasets. Consequently:

- public checks validate implementation and product contracts, not private-source coverage;
- the public benchmark should not be treated as a current live-production score;
- provider-specific ingestion and large-scale backfills require credentials not included here.

## Failure modes and mitigations

| Risk | Mitigation |
|---|---|
| Look-ahead leakage | As-of timestamps, time-ordered splits, and point-in-time contract tests |
| Sparse or mismatched precedents | Support thresholds, mismatch diagnostics, and broader-cohort fallbacks |
| Spurious explanatory relationship | Placebo/baseline comparison and explicit separation of explanatory, predictive, and causal claims |
| Distribution or provider drift | Schema, missingness, support-coverage, and performance gates before promotion |
| False precision in recommendations | Confidence labels, objections, regret cases, monitoring triggers, and human review |
| Public/private reproducibility gap | CI is limited to committed fixtures; non-public metrics are not presented as public CI results |

## Monitoring and promotion

Changes to models or policies should pass data-contract checks, time-based backtests, baseline/placebo comparisons, support and calibration review, and the portable public test contract. Promotion decisions should record the candidate artifact, evaluation window, thresholds, failure analysis, and rollback target. The complete public monitoring contract is documented in [`docs/model_monitoring.md`](docs/model_monitoring.md).

## Limitations

- Historical relationships can change across regimes, sectors, and data vendors.
- A positive aggregate result may conceal weak slices; sector and family/horizon results must be inspected.
- Shuffled placebos test one class of spurious signal and are not a substitute for full causal identification.
- Sanitized fixtures cannot reproduce the scale, coverage, or operational behavior of private provider-backed runs.
- Decision usefulness and financial impact have not been established by the public benchmark.
