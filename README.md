# Axiom

### Decision intelligence for corporate finance

[![CI](https://github.com/rdfds/axiom-research-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/rdfds/axiom-research-platform/actions/workflows/ci.yml)

Axiom turns point-in-time company data, peer context, market pricing, historical transactions, and action-impact evidence into recommendations a CFO, banker, or investment committee can interrogate.

This is not a dashboard wrapper or a single prediction model. It is an evidence system: every important output carries its timing, provenance, confidence, limitations, and supporting historical context.

![Axiom market expectations demo](docs/assets/market_expectations_hd.png)

## Evaluation snapshot

The committed forward-gap benchmark evaluates 84 candidate policies across **912 successful driver/horizon evaluations**, four walk-forward training cutoffs, fixed two-year test windows, and three shuffled placebos. The selected policy (`lambda = 0.5`) beat its placebo in **63.5%** of evaluations and improved mean absolute error by **0.0079** relative to placebo.

| Evidence artifact | What a reviewer can verify |
|---|---|
| [Model card](MODEL_CARD.md) | Intended use, system layers, evaluation design, failure modes, and limitations |
| [Benchmark JSON](results/public_benchmark.json) | Exact metrics, split metadata, selected policy, source path, and source hash |
| [Validation report](examples/hd_market_expectations/forward_gap_placebo_walk_forward_operating_ex_energy.sample.md) | Candidate, slice, sector, and placebo results |

Reproduce the committed artifact from the source report:

```bash
python scripts/build_public_benchmark.py --check
```

Results are from the documented offline evaluation and should be interpreted within its time windows, population, and placebo design.

## What Axiom does

- **Builds an auditable company state** from financial, market, filing, macro, and corporate-action inputs.
- **Explains valuation differences** with peer-relative driver surfaces rather than an unexplained score.
- **Separates priced expectations from residuals** so a premium or discount is not forced into a story the data cannot support.
- **Retrieves historical precedents** using learned distance weights, regime context, outcome cohorts, and mismatch diagnostics.
- **Evaluates action evidence** across market, valuation, credit, and operating outcomes with explicit quality gates.
- **Packages evidence for decisions** through structured evidence packs, recommendation contracts, monitoring triggers, and board-ready dossiers.

## Why the engineering is difficult

Corporate-finance decisions fail quietly when the data is not aligned to the decision date. Axiom is designed around the difficult parts:

1. **Point-in-time correctness**: features are built from what was available at the time, not from a later revised dataset.
2. **Traceability**: every feature records provenance, confidence, units, fallback behavior, and timing.
3. **Evidence separation**: explanatory valuation relationships are kept distinct from forward expectation claims and causal claims.
4. **Honest uncertainty**: thin precedent or action families fall back to broader evidence instead of receiving false precision.
5. **Decision translation**: model output becomes sizing guidance, objections, regret cases, and monitoring triggers rather than a chart with no action.

