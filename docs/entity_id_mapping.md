# Entity ID Mapping (MVP)

This document defines how company and security identifiers are resolved and
versioned. The MVP uses **company-first** identifiers:

- `company_id = gvkey`
- `security_id = permno`

Alternate identifiers (RIC, PermID, CUSIP, ISIN, ticker) are stored in a
canonical mapping table and used for reconciliation and joins.

## Canonical Mapping Table

**File:** `data/mappings/entity_id_map.parquet`

Required fields:

- `company_id` (gvkey)
- `security_id` (permno)
- `ric`
- `permid`
- `cusip`
- `isin`
- `ticker`
- `exchange`
- `valid_from`
- `valid_to`
- `source_system`
- `version_id`

Rules:

- Append-only. Never overwrite; new versions supersede via `version_id`.
- Overlaps are allowed only if `source_system` differs; conflicts must be
  flagged and resolved downstream using priority rules.
- `valid_from` / `valid_to` must be populated; use an open-ended max date
  (e.g., `2099-12-31`) for active mappings.

## Resolution Priority

When mapping a record to `company_id` / `security_id`, apply the following
priority order:

### Security-level (primary)
1. `permno` (if provided)
2. `permid` → `permno`
3. `ric` → `permno`
4. `cusip` → `permno`
5. `isin` → `permno`
6. `ticker + exchange` → `permno`

### Company-level (fallback)
1. `gvkey` (if provided)
2. `permid` → `gvkey`
3. `cusip` → `gvkey` (via permno or issuer mapping)
4. `ticker + exchange` → `gvkey`

## As-Of Matching

All mappings must be time-valid:

- Join with `valid_from <= event_time <= valid_to`
- If multiple matches exist, prefer:
  1. Exact source_system match
  2. Most recent `valid_from`
  3. Highest priority identifier rule

## Conflict Handling

When two mappings disagree:

- Emit `source_conflict` quality flag
- Prefer earliest `available_time` unless overridden by an authoritative source
  (e.g., CRSP for permno mappings)

## Mapping Sources (MVP)

Primary sources (authoritative):

- CRSP `msenames` + `ccmxpf_lnkhist` (permno/gvkey linkage)
- Refinitiv Symbology (RIC ↔ CUSIP/ISIN)

Secondary sources:

- Vendor-specific reference data
- Manual overrides (must include `source_system = "manual_override"`)

## Example Resolution

Given a record with:

- `ric = "AAPL.O"`
- `event_time = 2025-03-31`

Resolve:

1. Lookup `ric` in mapping table with `valid_from <= event_time <= valid_to`
2. If multiple matches, choose highest priority mapping
3. Set `security_id = permno`, `company_id = gvkey`
4. Record `upstream_version_ids` referencing the mapping row(s)
