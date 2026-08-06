# Showcase Tour

This is the curated path for someone evaluating Axiom as a serious application project. It is intentionally narrower than the full research workspace: each example is a compact sample of one major technical layer.

## 1. Market-Implied Valuation Gap

Start here:

- [Home Depot market expectations example](../examples/hd_market_expectations/README.md)
- [publication package](../paper/README.md)
- sample data: `examples/hd_market_expectations/valuation_driver_data.sample.json`
- rebuild command: `python scripts/build_hd_market_expectations_demo.py`
- paper smoke command: `python scripts/paper/run_market_expectations_experiments.py --preset smoke`

What it proves:

- Axiom can explain what part of a valuation premium/discount is underwritten by validated forward driver expectations.
- The model separates drivers that matter to valuation from drivers the market appears to be pricing differently.
- The residual is explicit: outside measured financial drivers, not silently forced into the model.

Why it is differentiated:

- Most comp tools stop at "premium/discount versus peers."
- This view asks whether the premium historically predicts future changes in specific business drivers.
- The paper package turns that into a reproducible empirical protocol with walk-forward splits, placebo tests, lambda ablations, and case-study reconciliation.

## 2. As-Of Company State

Open:

- [Company state snapshot example](../examples/company_state_snapshot/README.md)
- sample data: `examples/company_state_snapshot/company_state_hd.sample.json`

What it proves:

- Axiom is built on point-in-time data, not a loose pile of current metrics.
- Every feature has a timestamp, confidence score, provenance, fallback field, and unit.
- Downstream models can explain where a number came from.

Why it matters:

- Without as-of semantics, backtests leak future information.
- Without provenance, model output cannot be audited by a CFO, banker, or investment committee.

## 3. Precedent Retrieval Brain

Open:

- [Precedent retrieval example](../examples/precedent_retrieval/README.md)
- sample data: `examples/precedent_retrieval/precedent_retrieval.sample.json`

What it proves:

- Axiom retrieves historical corporate-action analogs using company state, action parameters, market regime, sector context, and learned distance weights.
- It returns confidence, matched cohorts, outcome distributions, top matches, and mismatch diagnostics.
- It can say when precedent support is weak instead of pretending every analogy is equally valid.

Why it matters:

- Bankers use precedents constantly, but usually as static tables.
- This turns precedents into a testable retrieval/risk layer.

## 4. CFO Decision Surface

Open:

- [CFO decision surface example](../examples/cfo_decision_surface/README.md)
- sample data: `examples/cfo_decision_surface/cfo_decision_surface_hd.sample.json`

What it proves:

- Axiom can reconcile multiple model layers into one CFO-facing decision read.
- The sample includes M&A sizing, capital-allocation frontier points, deal-size danger zones, model evidence layers, and a board-ready dossier excerpt.
- It shows how model output becomes action language: recommendation thesis, sizing guidance, regret analysis, scorecard, monitoring triggers, and supporting evidence.

Why it matters:

- This is the difference between a model repo and an application.
- The system is trying to help a decision-maker act, not just inspect a chart.

## One-Command Gallery Check

Run:

```bash
python scripts/inspect_showcase_gallery.py
```

This prints a compact summary of the committed showcase samples and fails if the expected sample contracts are missing.

## What To Show First

For applications or interviews, the best sequence is:

1. Open the README and screenshot.
2. Run the HD rebuild command.
3. Explain the market-implied valuation gap model.
4. Show the company-state sample to prove the data layer is serious.
5. Show precedent retrieval and CFO decision surface to prove this is end-to-end.

The headline should be:

> Axiom builds an auditable company state, explains valuation gaps, retrieves similar historical actions, validates action impact, and packages the result into CFO-grade evidence.
