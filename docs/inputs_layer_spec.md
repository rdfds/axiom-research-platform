# Inputs Layer Specification

This document defines the Inputs Layer contract for the system. It is the gate between raw data ingestion and all downstream modeling.

The Inputs Layer produces:
RawDocumentStore
RawTimeSeriesStore
EventRegistry
ExtractedFactRegistry
EntityGraph
PrivateOverlayRegistry
DataIntegrityLog

All inputs must be time-indexed, reproducible as-of, traceable to a source, and non-leaky.

All objects must carry:
source_id
source_type
entity_id (or equivalent)
published_at
effective_at (if applicable)
ingested_at
confidence_score
raw_pointer

Schemas live in:
schemas/inputs_layer/

Validator script:
scripts/validate_inputs_layer.py

## Public Inputs

**A. Financial Statements + Filings**
Scope includes 10-K, 10-Q, 8-K, annual reports, foreign equivalents, exhibits, MD&A, footnotes, and segment disclosures.
Required extraction targets include revenue, EBITDA or proxy, EBIT, net income, FCF or proxy, cash, debt, leases, pensions, share count, dividend policy, capex, working capital, interest expense, and liquidity policy.
Structural data includes debt maturities, revolver size and usage, call provisions, convertibles, preferred equity, segment and geographic revenue splits.
Qualitative facts include capital allocation priorities, leverage targets, rating intent, constraints, cost programs, synergy expectations, and strategic focus language.

**B. Earnings Call Transcripts + Presentations**
Extract guidance changes, tone shifts, liquidity commentary, capex outlook, margin outlook, growth constraints, competitive commentary, M&A appetite language, cost program updates, and regulatory risk commentary.
Each extracted fact must link to the speaker, include transcript timestamp, and tag Q&A vs prepared remarks.

**C. Corporate Actions Feed**
All events must normalize to a typed registry.
Supported action types include capital structure actions, capital return, M&A, restructuring, and governance changes.
Each event includes event_id, company_id, action_type, action_subtype, announcement_date, effective_date, status, parameters, and evidence_links.

**D. Market Data**
Time series must be point-in-time correct, split-adjusted, and survivorship-bias free.
Required domains include equity price/market cap/volatility/drawdown/total return and credit spreads/yield curves.

**E. Macro / Industry Data**
Include Treasury yields, inflation, GDP, commodities, FX, sector indices, credit indices, and VIX.
Each series must include frequency metadata, revision flags, and release lag.

**F. Estimates / Consensus**
Include EPS, revenue, EBITDA estimates and revisions with as-of correctness.

**G. News / Regulatory / Legal Events**
Extract litigation, DOJ/FTC, approvals, fines, and geopolitical impacts with materiality scores and forward-looking risk flags.

## Optional Private Overlays

Private overlays do not overwrite public truth. They add constraints and scenario assumptions.
Each overlay must include versioning, expiration, author, and timestamps, and remain separable from the base state.

Examples:
Covenant packages, detailed debt schedules, internal projections, non-public segment KPIs, and explicit board constraints.

## Stores

RawDocumentStore
Schema: schemas/inputs_layer/raw_document_store.schema.json
Use for filings, transcripts, presentations, and raw document metadata.

RawTimeSeriesStore
Schema: schemas/inputs_layer/raw_timeseries_store.schema.json
Use for point-in-time time series with as-of correctness.

EventRegistry
Schema: schemas/inputs_layer/event_registry.schema.json
Use for normalized corporate actions with typed parameters.

ExtractedFactRegistry
Schema: schemas/inputs_layer/extracted_fact_registry.schema.json
Use for extracted facts with citations and confidence.

EntityGraph
Schema: schemas/inputs_layer/entity_graph.schema.json
Use for ID mapping and entity relationships.

PrivateOverlayRegistry
Schema: schemas/inputs_layer/private_overlay_registry.schema.json
Use for internal overlays and constraints. Overlays must be removable.

DataIntegrityLog
Schema: schemas/inputs_layer/data_integrity_log.schema.json
Use for validation output and auditability.

## Invariants

As-of integrity is required for all records.
published_at and ingested_at must be populated.
effective_at must be populated when applicable.
Missing data must be explicit.
Units and scaling must be recorded for numeric fields.
No forward-looking leakage is permitted.
Private overlays must be removable without corrupting the base state.

## Validation

The validator checks:
Required columns per schema.
Basic dtype compatibility.
Timestamp parseability.
published_at and effective_at vs ingested_at ordering.
confidence_score bounds.

Example:
python -u scripts/validate_inputs_layer.py --config configs/inputs_layer.json
