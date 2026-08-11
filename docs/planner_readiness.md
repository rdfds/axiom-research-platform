# Planner Readiness

## Status

The current planner implementation is not step-9-complete.

`./src/recommendation_run_orchestrator.py` currently builds one-step plans by scoring individual precedent-matched candidates. It does not yet implement:

- multi-step plan search
- dependency graph expansion
- branch generation
- lead-time-aware scheduling
- robustness scoring across regimes
- plan-level risk summaries

That said, the inputs needed to build the Planner Brain are now in place and validated enough to proceed.

## Existing Inputs Ready For Planner

### Mechanism / Feasibility

- `FeasibilityResult.pass_probability`
- `FeasibilityResult.lead_time_prior_days`
- mechanism activation and impact distributions already attached to evaluated candidates

### Precedent

`./src/pipeline/precedent_brain.py` already emits:

- outcome distributions
- tail events
- regime splits
- second-order effects
- mismatch diagnostics
- calibrated precedent confidence

### Structural Priors

`./src/action_ontology.py` already stores:

- dependency rules per action
- lead-time priors per action
- execution complexity priors per action

`./src/candidate_generation.py` already includes playbook templates that are natural seeds for planner search:

- deleveraging
- simplification
- growth substitution

