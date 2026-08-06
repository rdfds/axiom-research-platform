# Inferring Market-Implied Fundamental Expectations from Valuation Gaps

## Abstract

Draft: We study whether peer-relative valuation premiums and discounts contain predictive information about future operating fundamentals. We introduce an interpretable framework that compares fundamentals-only forecasts against valuation-gap-enhanced forecasts, applies conservative shrinkage, and decomposes current valuation gaps into driver-underwritten and residual components. In walk-forward validation, the valuation-gap signal is evaluated against baselines and placebo assignments. The framework produces company-level explanations while preserving a non-causal interpretation boundary.

## 1. Introduction

- Motivation: valuation gaps are common in corporate finance and investing, but usually described qualitatively.
- Research question: what, if anything, is the market pricing into future drivers?
- Contribution: predictive test plus interpretable decomposition.

## 2. Related Work

- Cross-sectional valuation and peer multiples.
- Fundamental forecasting.
- Market-implied expectations.
- Interpretable ML in finance.

## 3. Method

- Value-surface support model.
- Valuation-gap definition.
- Fundamentals-only forecast.
- Gap-enhanced forecast.
- Shrinkage lambda.
- Driver-underwritten/residual decomposition.

## 4. Empirical Design

- Universe and exclusions.
- Point-in-time data construction.
- Walk-forward splits.
- Baselines.
- Placebos and sanity checks.
- Metrics.

## 5. Results

- Universe coverage.
- Baseline comparison.
- Lambda ablation.
- Placebo results.
- Sector robustness.
- Driver-family/horizon results.

## 6. Case Studies

- Home Depot flagship case.
- Residual-dominant case.
- Multi-driver underwritten case.

## 7. Limitations

- Non-causal interpretation.
- Private data dependency.
- Sector/accounting comparability.
- Residual factors outside measured financial drivers.

## 8. Conclusion

- Summary of valuation-gap signal.
- Practical use as CFO/investor evidence layer.
- Future work: broader public-data replication and richer residual taxonomy.
