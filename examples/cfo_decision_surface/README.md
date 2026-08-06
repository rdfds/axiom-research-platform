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

## Example Reads

The sample includes a Home Depot M&A decision summary where the system:

- selects a digestible tuck-in size
- identifies where deal size starts becoming dangerous
- compares M&A against buyback, debt issuance, dividends, and other actions
- flags evidence conflicts when the historical neighborhood challenges the model direction

The dossier excerpt demonstrates a separate board-ready recommendation contract:

- recommendation thesis
- sizing guardrails
- regret framing
- monitoring triggers
- supporting evidence and precedent confidence

## Related Code

- `src/cfo_decision_surface.py`
- `src/board_ready_dossier.py`
- `src/recommendation_run_orchestrator.py`
- `src/evidence_pack.py`
- `tests/test_cfo_decision_surface.py`
- `tests/test_board_ready_dossier.py`
