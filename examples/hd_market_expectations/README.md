# Home Depot Market Expectations Example

This example is the current best showcase for the valuation driver and market-implied gap model.

It answers:

> Home Depot trades at a premium to the model's driver-supported multiple. How much of that premium is underwritten by validated forward driver expectations, and how much remains outside the measured financial-driver surface?

## Rebuild The Public Sample

From the repository root:

```bash
python scripts/build_hd_market_expectations_demo.py
```

This stages the committed sample files into:

```text
examples/hd_market_expectations/build/
```

and writes:

```text
examples/hd_market_expectations/build/valuation_action_bridge.html
```

Open the generated page directly:

```bash
open examples/hd_market_expectations/build/valuation_action_bridge.html
```

## Optional Full Local Demo

If you are on the original development machine, there may also be a full multi-company generated page under the local materialized artifacts directory. The public sample does not require that private/local artifact.

```bash
open ./data/mna_insights/valuation_action_bridge.html
```

Then select `HD`, or open directly:

```bash
open ./data/mna_insights/valuation_action_bridge.html#HD
```

## Sample Inputs

The public rebuild uses committed, sanitized sample files:

```text
examples/hd_market_expectations/valuation_driver_data.sample.json
examples/hd_market_expectations/expectation_driver_history.sample.json
examples/hd_market_expectations/expectation_evidence_cohort.sample.json
examples/hd_market_expectations/forward_gap_placebo_walk_forward_operating_ex_energy.sample.md
```

The sample includes only the HD company payload plus aggregate cohort evidence needed by the visualization. The broader local artifact contains other company payloads and generated workspace outputs that are intentionally omitted here.

## Full Generated Inputs

The original full workspace demo was built from:

```text
./data/mna_insights/valuation_driver_data.json
```

The current builder is:

```text
scripts/build_valuation_action_bridge.py
```

The upstream company/data builder is:

```text
scripts/build_curated_company_valuation_drivers.py
```

## What The View Shows

The market-expectations section has four jobs:

1. Translate the valuation premium/discount into a money and multiple gap.
2. Allocate the gap between validated driver expectations and residual outside-model factors.
3. Show each priced driver as a two-path forecast: fundamentals-only versus market-gap-enhanced.
4. Expose validation context through evidence-vs-placebo and cohort reads.

## Why This Is Different From A Normal Comp Sheet

A normal comp sheet says:

> HD trades at a premium because the market likes the company.

The Axiom view tries to say:

> Of the premium, this amount is statistically underwritten by specific forward driver expectations, and this amount is outside the measured financial drivers. The underwritten driver claims only appear when the valuation gap historically improved out-of-sample forecasts for that driver.

That distinction matters. It separates:

- drivers that matter to valuation
- drivers where HD is currently strong or weak
- drivers the market appears to be pricing differently
- residual premium/discount that likely reflects brand, defensiveness, risk, sentiment, duration, or factors not captured in the model

## GitHub Packaging TODO

This example should become a one-command public demo by adding:

- a small static HTML fixture committed for GitHub Pages or release previews
- a screenshot or short GIF of the interaction
- a model-card appendix that points to `docs/validation/README.md`
