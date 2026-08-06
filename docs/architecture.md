# Architecture

Axiom is organized as a layered corporate-finance decision engine. Each layer has a separate job: preserve point-in-time truth, model the company and market, retrieve historical analogs, validate action effects, then package the output into evidence a CFO can use.

## System Map

```mermaid
flowchart TB
    subgraph Data["1. As-of data plane"]
        Raw["Raw immutable inputs"]
        Warehouse["Bitemporal warehouse"]
        State["CompanyStateSnapshot"]
    end

    subgraph Models["2. Modeling layer"]
        Drivers["Valuation driver surface"]
        Gap["Market-implied gap model"]
        Precedent["Precedent retrieval"]
        Causal["Action impact / causal layer"]
        Planner["Planner and action logic"]
    end

    subgraph Product["3. Product layer"]
        Evidence["EvidencePack"]
        CFO["CFO decision surface"]
        Demo["Static valuation/action bridge"]
    end

    Raw --> Warehouse --> State
    State --> Drivers --> Gap
    State --> Precedent
    State --> Causal
    State --> Planner
    Gap --> Evidence
    Precedent --> Evidence
    Causal --> Evidence
    Planner --> Evidence
    Evidence --> CFO --> Demo
```

## 1. As-of Data Plane

The as-of layer is the foundation. It keeps market, financial, filing, event, estimate, macro, and private overlay data separate from model output.

Important files:

- `docs/data_contract.md`
- `docs/asof_views.md`
- `docs/quality_flags.md`
- `src/company_state_builder.py`
- `src/asof_store.py`

Design principles:

- raw data is append-only
- modeled features must carry provenance
- every feature has an as-of interpretation
- missing values stay explicit
- fallbacks are allowed only when flagged

## 2. Company State

`CompanyStateSnapshot` is the canonical runtime object. It collects the data needed to evaluate a company as of a specific date.

The snapshot layer tracks:

- feature value
- source input references
- confidence
- support mode
- fallback use
- quality flags
- component breakdowns

This is what lets downstream models explain where a value came from instead of only emitting a score.

## 3. Valuation Driver Engine

The valuation driver system answers:

> What business drivers explain valuation differences inside the relevant peer set?

It supports investor-native display lenses such as P/E and EV/EBITDA while routing driver weights through more stable value surfaces like P/Revenue or EV/Revenue when appropriate.

Important files:

- `src/valuation_driver_validation.py`
- `src/valuation_driver_interpretation.py`
- `scripts/build_curated_company_valuation_drivers.py`
- `tests/test_investor_native_valuation_lens.py`
- `tests/test_cfo_native_valuation_basis.py`

The interpretation layer intentionally labels these as conditional peer-set associations, not causal recommendations.

## 4. Market-Implied Gap Model

The market-implied model answers a harder question:

> When the market pays a premium or applies a discount, which future driver changes does that premium or discount historically predict?

It compares two forecasts:

- a fundamentals-only forecast based on current level, recent momentum, and cycle context
- a market-gap enhanced forecast that adds the valuation premium or discount

If the gap-enhanced forecast improves out-of-sample MAE for a driver, the model treats that driver as market-priced. The current company gap is then allocated between validated driver expectations and a residual "outside measured drivers" bucket.

Important files:

- `scripts/validate_forward_gap_lambda_policy.py`
- `scripts/build_valuation_action_bridge.py`
- `tests/test_roic_materialization_valuation_drivers.py`
- `tests/test_valuation_action_bridge_wwntbt.py`

## 5. Precedent Retrieval

The precedent system retrieves similar historical action cases. It is not a text search feature; it uses company state, action type, regime context, outcome distributions, mismatch diagnostics, and calibrated confidence.

Important files:

- `src/pipeline/precedent_brain.py`
- `src/pipeline/precedent_distance_v2_learning.py`
- `src/pipeline/precedent_quality_learning.py`
- `tests/test_precedent_brain.py`
- `tests/test_precedent_distance_v2_learning.py`

The learned-distance layer tunes similarity weights by action family and objective so that analogs are retrieved for the decision being made, not just for superficial similarity.

## 6. Action Impact Layer

The action-impact layer evaluates whether action setups historically predict measurable outcomes.

Examples:

- short-window abnormal stock returns
- positive/material return classifiers
- valuation rerating
- credit spread movement
- leverage or rating changes
- operating metric movement

Important files:

- `src/action_stock_impact_validation.py`
- `src/action_valuation_rerating_validation.py`
- `src/causal_impact_model.py`
- `src/mechanism_brain.py`
- `tests/test_action_stock_impact_validation.py`
- `tests/test_action_valuation_rerating_validation.py`
- `tests/test_mechanism_causal_strict_gate.py`

The system deliberately separates metric-routed decision evidence from generic causal descriptions.

## 7. EvidencePack and CFO Surface

The final product layer packages output into evidence that can be reviewed.

Important files:

- `src/evidence_pack.py`
- `src/cfo_decision_surface.py`
- `src/board_ready_dossier.py`
- `src/recommendation_run_orchestrator.py`

The EvidencePack concept matters because it gives every user-facing claim a bounded data source. It is the bridge between quantitative models and board-ready language.

## Public Packaging Direction

The active repo is still a broad workbench. The GitHub-facing version should emphasize:

- a clean architecture story
- one polished demo case
- stable validation summaries
- sample data rather than private/local artifacts
- a small command-line path that rebuilds the demo from sample inputs

The core modeling work is strong enough. The highest-return next work is packaging and curation.
