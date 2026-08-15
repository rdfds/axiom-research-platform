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

## User-Value Expansion Added Before Planner

Two low-risk structural additions were made so step 9 can start from cleaner primitives:

### 1. Planner-Normalized Dependency Edges

Added `ActionSchemaRegistry.fetch_planner_dependency_edges(...)`.

This maps ontology rule types into planner-facing relationships:

- `requires_prior -> requires`
- `unlocks -> unlocks`
- `conflicts_with -> conflicts`
- `discouraged_with -> conflicts`
- `preferred_after -> recommended_after`

The helper preserves:

- condition
- strength
- explanation
- original rule type

### 2. Planner Lead-Time Distribution Helper

Added `ActionSchemaRegistry.fetch_planner_lead_time_distribution(...)`.

The ontology stores `minimum_days`, `median_days`, and `p90_days`. Planner search needs a richer scheduling prior, so this helper deterministically interpolates:

- `p25_days`
- `p75_days`
- `mean_days`

Source is marked as `schema_prior_interpolated` so later historical transition priors can replace it cleanly.

### 3. Planner Type Scaffolding

Added `./src/planner_types.py`.

This defines planner-facing schemas for:

- dependency graph
- lead-time distribution
- plan / plan step / plan trigger / branch / risk / score breakdown

The type scaffolding also includes optional explanation fields so the planner can emit user-facing structural rationales, not just action lists.

## Evaluation Coverage

### Precedent

Validated on:

- fixed 5-company regression set
- broader 20-company canary
- targeted family benchmarks

Accepted broad canary metrics:

- `precedent_conf_mean = 0.346832`
- `precedent_oos_mean = 0.308`

### Causal

Accepted broad canary metrics:

- `causal_rate_mean = 0.848167`
- `strict_causal_mean = 1.0`

Intentional causal exceptions:

- `capital_structure.revolver_draw_or_resize`
- `mna.go_private_lbo`

Both remain precedent-only by policy.

## Runtime / UI Constraint

Planner work can proceed now, but user-facing UX should assume:

- async execution
- warmed service
- cached results

The current system is suitable for a snappy UI only if the UI does not block on full recommendation completion.

## Recommendation

Proceed to step 9 now.

Do not spend more time on precedent or causal before planner implementation unless a regression appears.
