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

## Optional Full Local Demo

If you are on the original development machine, there may also be a full multi-company generated page under the local materialized artifacts directory. The public sample does not require that private/local artifact.

```bash
open ./data/mna_insights/valuation_action_bridge.html
```

Then select `HD`, or open directly:

```bash
open ./data/mna_insights/valuation_action_bridge.html#HD
```

