# CFO Decision Surface Example

This example shows how Axiom turns model evidence into a CFO-facing decision surface.

The sample file is:

```text
examples/cfo_decision_surface/cfo_decision_surface_hd.sample.json
```

It combines two real materialized artifacts:

- Home Depot CFO decision-surface layers
- a board-ready dossier excerpt from a recommendation run

## What To Look At

The Home Depot section includes:

- `mna_decision_summary`
- `capital_allocation_frontier`
- `deal_size_sensitivity_curve`
- `mna_deal_size_danger_zone`
- `defensible_model_layers`
- `defensible_model_wedge`

The dossier excerpt includes:

- `recommendation_thesis`
- `sizing_guidance`
- `regret_analysis`
- `scorecard`
- `monitoring`
- `ranked_action_views`
- `supporting_evidence`
- `recommendation_contract`

## Why This Matters

This is the application layer. It proves the system is not just producing scores.

A CFO needs to know:

- what action is recommended
- why now
- how large it should be
- what evidence supports it
- what could go wrong
- when to stop or revisit the plan

The decision surface turns model artifacts into that language.

