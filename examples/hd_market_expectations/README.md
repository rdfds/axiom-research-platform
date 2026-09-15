# Home Depot Market Expectations Example

This example provides the most complete walkthrough of the valuation-driver and market-implied gap model.

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

## Optional Multi-Company Artifacts

If you have separately materialized a multi-company page, you can open it at the following path. That page and its upstream datasets are not included or required to rebuild the committed Home Depot sample.

```bash
open ./data/mna_insights/valuation_action_bridge.html
```

Then select `HD`, or open directly:

```bash
open ./data/mna_insights/valuation_action_bridge.html#HD
```

## Sample Inputs

The rebuild uses committed sample files:

```text
examples/hd_market_expectations/valuation_driver_data.sample.json
examples/hd_market_expectations/expectation_driver_history.sample.json
examples/hd_market_expectations/expectation_evidence_cohort.sample.json
examples/hd_market_expectations/forward_gap_placebo_walk_forward_operating_ex_energy.sample.md
```

The sample includes only the HD company payload plus aggregate cohort evidence needed by the visualization. The broader local artifact contains other company payloads and generated workspace outputs that are intentionally omitted here.

## Upstream Inputs and Rebuild Scope

The presentation builder accepts a materialized valuation input at:

```text
./data/mna_insights/valuation_driver_data.json
```

The current builder is:

```text
scripts/build_valuation_action_bridge.py
```

The upstream company/data builder and full validation pipeline are not included. The sample rebuild reproduces the presentation from committed inputs; it does not regenerate those inputs or retrain the model. See the [model card](../../MODEL_CARD.md) for the evaluation scope.

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

## Why this example matters

This is the clearest compact demonstration of Axiom's central design choice: make the model's support and uncertainty visible. The view does not hide the residual, overstate causal language, or require a private data account to inspect the product behavior.
