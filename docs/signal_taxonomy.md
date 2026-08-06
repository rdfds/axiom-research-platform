# Signal Taxonomy (MVP)

This taxonomy defines the **initial** unstructured text signals extracted from
documents (press releases, transcripts, research, presentations). Each signal
produces:

- `signal_name`
- `value` (normalized score or numeric)
- `confidence` (0–1)
- `supporting_chunk_ids` (array)

## Conventions

- **Score range** for qualitative signals: `0–100` (higher = stronger presence).
- **Binary** signals: `0` or `100`.
- **Confidence**: model‑reported or heuristic quality, `0–1`.
- **Value type**: numeric; categorical values are encoded to numeric with a
  documented mapping.

## Core Signals (Unstructured)

| signal_name | type | scale | description | cues / examples | primary sources |
|---|---|---|---|---|---|
| `management_risk_posture` | qualitative | 0–100 | Management tone: risk‑seeking vs cautious | “aggressive expansion”, “conservative stance”, “risk mitigation” | transcripts, press releases |
| `capital_allocation_intent` | categorical → numeric | 0–100 | Emphasis on buybacks/dividends vs reinvestment/deleveraging | “returning capital”, “buyback authorization”, “debt reduction” | press releases, presentations |
| `growth_vs_defense` | qualitative | 0–100 | Growth orientation vs defensive posture | “accelerate growth”, “cost containment”, “protect margins” | transcripts, presentations |
| `uncertainty_hedging_intensity` | qualitative | 0–100 | Degree of hedging language and uncertainty | “may”, “could”, “subject to”, “uncertain” | transcripts, press releases |
| `strategic_pressure_indicator` | qualitative | 0–100 | Strategic stress or external pressure | “competitive pressure”, “regulatory headwinds”, “activist” | press releases, research |
| `guidance_change` | categorical → numeric | 0–100 | Guidance raised/maintained/lowered | “raises guidance”, “reaffirms”, “lowers” | press releases, transcripts |
| `restructuring_signal` | binary | 0/100 | Restructuring / layoffs / asset sales | “restructuring”, “cost‑cutting”, “divestiture” | press releases |
| `mna_intent_signal` | qualitative | 0–100 | Intent or openness to M&A | “strategic alternatives”, “acquisition pipeline” | transcripts, press releases |
| `liquidity_concern_signal` | qualitative | 0–100 | Liquidity stress or runway concerns | “liquidity”, “covenant”, “going concern” | filings, transcripts |
| `pricing_power_signal` | qualitative | 0–100 | Pricing power or discounting pressure | “pricing power”, “promotions”, “discounts” | transcripts, presentations |

## Numeric Signal Encodings

- `capital_allocation_intent`
  - 0 = reinvestment / growth capex
  - 50 = balanced / mixed
  - 100 = return of capital / buybacks / dividends

- `guidance_change`
  - 0 = lowered
  - 50 = reaffirmed
  - 100 = raised

## Required Canonical Table

Signals are stored in `warehouse_text_signals` (see `docs/canonical_schemas.sql`)
with the global bitemporal fields.

## Versioning

- `signal_name` is stable. Changes create new `extraction_version`.
- All signal extraction must reference `supporting_chunk_ids`.
- No free‑form summaries are stored upstream.
