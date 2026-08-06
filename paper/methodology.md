# Methodology: Market-Implied Fundamental Expectations

## Research Question

Can peer-relative valuation premiums or discounts improve forecasts of future company fundamentals beyond what current fundamentals, recent momentum, and cycle context already predict?

The model is intentionally predictive and interpretive. It does not claim that valuation gaps cause fundamental changes.

## Core Objects

- `support_multiple`: the model-implied valuation multiple supported by current measured drivers.
- `market_multiple`: the observed market valuation multiple.
- `valuation_gap`: log market multiple minus log support multiple.
- `fundamentals_forecast`: a baseline forecast of forward driver change from current driver level, recent driver momentum, and cycle controls.
- `gap_enhanced_forecast`: the same forecast after adding the valuation gap.
- `lambda`: shrinkage weight applied to the incremental gap-enhanced forecast.
- `driver_underwritten_gap`: the portion of the current valuation gap allocated to drivers with validated forward signal.
- `residual_gap`: the part of the valuation gap outside measured financial drivers.

## Forecast Test

For each driver and horizon, compare two out-of-sample forecasts:

1. Fundamentals-only forecast.
2. Gap-enhanced forecast.

A driver is considered market-priced only when the gap-enhanced forecast improves out-of-sample error relative to the fundamentals-only forecast.

## Validation Protocol

- Use point-in-time predictor availability dates.
- Use outcome availability dates for training eligibility.
- Train only on rows whose outcomes are available by the split boundary.
- Test only on predictor rows after the split boundary.
- Report weak, negative, and immaterial results rather than hiding them.

## Placebo And Sanity Checks

- Shuffle valuation gaps within driver/horizon panels.
- Assign wrong-sector gaps.
- Assign random-driver gaps.
- Compare no-shrinkage against shrinkage.
- Verify every outcome date is after the predictor date.

## Interpretation Boundary

Use terms such as "market-implied," "predictive information," and "valuation-gap signal." Avoid causal language. Residual valuation gaps should be described as outside measured drivers, not as model error by default.
