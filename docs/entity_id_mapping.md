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

