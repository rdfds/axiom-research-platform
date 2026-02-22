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

