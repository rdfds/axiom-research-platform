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

Source artifacts:

- local generated artifact: `forward_gap_walk_forward_operating_ex_energy.md`
- local generated artifact: `forward_gap_placebo_walk_forward_operating_ex_energy.md`
- `scripts/validate_forward_gap_lambda_policy.py`

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

## Valuation Driver Surface

Source artifacts:

- local generated artifact: `value_surface_native_validation_report.md`
- local generated artifact: `valuation_driver_importance_v2_backtest.md`
- `src/valuation_driver_validation.py`
- `scripts/build_curated_company_valuation_drivers.py`

Native display contract:

- P/E display can route driver weights through P/Revenue.
- EV/EBITDA display can route driver weights through EV/Revenue.
- Native fair multiples are translated algebraically from the value surface.
- Driver weights must sum to 100% and be sorted by modeled impact.

Named-company validation snapshot:

| Ticker | Display | Weight surface | Grade | R2 | Rank IC | Top driver |
|---|---|---|---|---:|---:|---|
| NKE | P/E | P/Revenue | strong | 0.482 | 0.748 | EPS growth |
| AAPL | P/E | P/Revenue | strong | 0.490 | 0.673 | FCF margin |
| GOOGL | P/E | P/Revenue | strong | 0.666 | 0.794 | Revenue growth |
| MSFT | P/E | P/Revenue | strong | 0.670 | 0.829 | Revenue growth |
| AMZN | EV/EBITDA | EV/Revenue | strong | 0.704 | 0.755 | EPS growth |
| CAT | EV/EBITDA | EV/Revenue | strong | 0.429 | 0.654 | EBITDA margin |
| HD | P/E | P/Revenue | strong | 0.573 | 0.786 | Revenue growth |
| PG | P/E | P/Revenue | strong | 0.582 | 0.807 | EPS growth |

Historical v2 backtest:

| Target | N | Weighted IC | Equal-weight IC | Weighted hit rate | Equal hit rate |
|---|---:|---:|---:|---:|---:|
| Business value | 745 | 0.137 | 0.095 | 55.6% | 55.4% |
| Equity value | 745 | 0.182 | 0.160 | 59.4% | 59.1% |

Interpretation:

- The driver system is strongest as an explanatory value-surface model.
- It should not be described as a causal recommendation engine.
- The forward gap model is the separate layer that asks whether the market is pricing future driver movement.

## Precedent and Causal Monitoring

Source docs:

- `docs/precedent_baseline.md`
- `docs/causal_baseline.md`
- `docs/model_monitoring.md`

Accepted broad 20-company canary:

| Metric | Value |
|---|---:|
| causal_rate_mean | 0.848 |
| strict_all_mean | 0.848 |
| strict_causal_mean | 1.000 |
| precedent_conf_mean | 0.347 |
| precedent_oos_mean | 0.308 |

Gate thresholds:

| Metric | Threshold |
|---|---:|
| min_causal_rate_mean | 0.82 |
| min_strict_all_mean | 0.82 |
| min_strict_causal_mean | 0.95 |
| min_precedent_conf_mean | 0.34 |
| max_precedent_oos_mean | 0.33 |

Known limitation:

- `capital_structure.revolver_draw_or_resize` remains precedent-driven because targeted causal rescue training did not clear OOS quality gates.

That limitation should stay visible. It makes the public story more credible because the system refuses to label weak cells as strong.

## Stock Impact and Action Surface

Source docs:

- `docs/stock_impact_validation_artifacts.md`
- `src/action_stock_impact_validation.py`
- `src/stock_impact_profile.py`

Refreshed families include:

- stock splits
- debt refinancing
- M&A disclosed, tuck-in, platform, transformational
- dividend increase/initiation/cut/special dividend
- debt issuance
- buyback
- equity issuance

Important modeling choice:

- direct abnormal-return regressors are noisy
- classifier-calibrated return-band evidence often gives a more stable product signal
- exact thin families can fall back to broader family evidence when appropriate

## What To Productize Next

For a GitHub-ready validation package, the next pass should:

- copy or regenerate stable public validation summaries into this folder
- replace local absolute paths with repo-relative model-card references
- add one command that rebuilds the market-implied gap validation from a small sample
- add confidence intervals around current driver contributions
- add a residual taxonomy for gaps mostly outside measured drivers
