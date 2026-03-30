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

