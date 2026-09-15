# Axiom repository tour

Each example illustrates one technical layer. The final example shows how the layers combine into a decision surface.

## 1. Market-implied expectations

Start with the [Home Depot market expectations example](../examples/hd_market_expectations/README.md).

It demonstrates:

- valuation-gap decomposition into measured drivers and an explicit residual
- forward driver expectations grounded in historical validation
- a static, reproducible HTML view built from committed sample inputs

The question is not simply “why does this company trade at a premium?” It is “which part of the premium is supported by measurable forward expectations, and which part remains outside the model’s driver surface?”

## 2. Point-in-time company state

Open the [company state snapshot example](../examples/company_state_snapshot/README.md).

It demonstrates that every feature carries:

- an as-of timestamp
- source provenance
- confidence and fallback metadata
- units and feature-level interpretation

This is the foundation for leakage-safe backtests and auditable downstream recommendations.

## 3. Precedent retrieval

Open the [precedent retrieval example](../examples/precedent_retrieval/README.md).

It demonstrates retrieval over historical corporate actions using company state, action parameters, market regime, sector context, and learned distance weights. The output includes confidence, outcome cohorts, top matches, and mismatch diagnostics—and can say when precedent support is weak.

## 4. CFO decision surface

Open the [CFO decision surface example](../examples/cfo_decision_surface/README.md).

It demonstrates the product layer: evidence from multiple models becomes action sizing, risk and regret cases, recommendation language, monitoring triggers, and a board-ready dossier.

## One-command gallery check

From the repository root:

```bash
python scripts/inspect_examples.py
```

## Suggested Reading Order

1. Open the README and architecture diagram.
2. Rebuild the market-expectations demo.
3. Inspect the company-state sample's timing, provenance, and confidence fields.
4. Inspect precedent retrieval, match diagnostics, and outcome distributions.
5. Read the CFO decision surface and recommendation contract.

The [architecture guide](architecture.md) distinguishes included implementation from upstream analysis represented by the samples. The [model card](../MODEL_CARD.md) describes evaluation scope and limitations.
