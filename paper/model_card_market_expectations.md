# Model Card: Market-Implied Fundamental Expectations

## Intended Use

Estimate whether a company's valuation premium or discount contains forward predictive information about specific operating drivers, and decompose the current gap into driver-underwritten and residual components.

## Non-Claims

- The model is not causal.
- It does not prove that markets are correct.
- It does not explain every premium or discount.
- It does not turn weak or immaterial drivers into decision evidence.

## Inputs

- Peer-relative value-surface model outputs.
- Current driver levels.
- Recent driver momentum.
- Cycle context where available.
- Market/support valuation gap.

## Outputs

- Driver-level forecast lift versus fundamentals-only baseline.
- Shrinkage lambda policy.
- Placebo and sanity-check results.
- Company-level driver contribution and residual decomposition.

## Known Failure Modes

- Thin peer histories can make driver-specific estimates unstable.
- Sector-specific accounting regimes can weaken comparability.
- Residual gaps can dominate when brand, duration, defensiveness, sentiment, or risk premia are outside the measured driver set.
- Strong valuation-driver importance does not imply a driver is currently market-priced.

## Publication Boundary

Private financial data may be used for empirical results, but the public repository should expose only code, methodology, sanitized samples, smoke fixtures, and non-sensitive summaries.
