# Validation Overview

This folder summarizes the validation work that should be visible in a public GitHub review. The live workspace contains many generated artifacts under local `out/` folders and `./data/`; this document pulls the highest-signal results into one place.

## Validation Philosophy

Axiom tries to avoid model confidence theater. The validation standard is:

- point-in-time inputs only
- out-of-sample or walk-forward splits where possible
- explicit placebo or baseline comparisons for market-implied claims
- model family gates before a score can become decision evidence
- honest fallbacks when exact samples are thin
- materiality labels so tiny effects are not oversold

## Market-Implied Valuation Gap

Public evidence fixture:

- `examples/hd_market_expectations/forward_gap_placebo_walk_forward_operating_ex_energy.sample.md`

Scope:

- as-of date: 2024-12-31
- operating company universe
- excluded sectors: Energy, Financials, Utilities, Real Estate
- candidates attempted: 84
- successful driver/horizon evaluations: 912
- walk-forward train ends: 2014-12-31, 2016-12-31, 2018-12-31, 2020-12-31
- test window: 2 years

Headline results:

| Lambda | Actual mean MAE improvement | Placebo mean MAE improvement | Actual - placebo | Actual beats placebo |
|---:|---:|---:|---:|---:|
| 0.10 | 0.0026 | -0.0000 | 0.0027 | 69.1% |
| 0.20 | 0.0047 | -0.0001 | 0.0048 | 68.2% |
| 0.35 | 0.0067 | -0.0002 | 0.0070 | 65.6% |
| 0.50 | 0.0074 | -0.0005 | 0.0079 | 63.5% |
| 0.65 | 0.0068 | -0.0008 | 0.0076 | 61.2% |
| 0.80 | 0.0050 | -0.0012 | 0.0062 | 57.6% |
| 1.00 | 0.0007 | -0.0018 | 0.0025 | 52.1% |

Current policy:

```json
{
  "default_lambda": 0.5,
  "fallback_lambda": 0.5,
  "by_family_horizon": {
    "cash_conversion:1Y": 0.1,
    "cash_margin:2Y": 0.35
  },
  "min_group_evaluations": 20,
  "min_mean_lift_vs_default": 0.001,
  "min_pass_rate_lift_vs_default": 0.15
}
```

Interpretation:

- The global lambda of 0.50 is the best broad policy in walk-forward validation.
- The placebo check is important: shuffled gaps do not produce the same MAE lift.
- The model should describe the residual bucket honestly as "outside measured financial drivers," not as a bug.

