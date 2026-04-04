# Axiom repository tour

This is the fastest path for someone evaluating Axiom as a serious engineering project. Each example isolates one major technical layer, then the final example shows how the layers become a decision surface.

## 1. Market-implied expectations

Start with the [Home Depot market expectations example](../examples/hd_market_expectations/README.md).

It demonstrates:

- valuation-gap decomposition into measured drivers and an explicit residual
- forward driver expectations grounded in historical validation
- a static, reproducible HTML view built from committed sample inputs

The question is not simply “why does this company trade at a premium?” It is “which part of the premium is supported by measurable forward expectations, and which part remains outside the model’s driver surface?”

