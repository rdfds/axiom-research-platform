# Data Plane Coverage (Current)

This note captures what’s complete vs. missing as of now, based on the current pipeline outputs.

## ✅ Complete (Through 2025)

**RawDocumentStore (with raw text + hashes)**
- 2000–2025 built
- Version stream built (`raw_document_versions`)

**RawTimeSeriesStore**
- Built and validated

**EventRegistry + Canonical Event Store**
- Enriched params for M&A, bond issuance, loan issuance, equity offering proxy
- Lifecycle timestamps populated

**Entity Tables**
- Entity + Identifier mappings built
- M&A relationships built into `entity_relationship`

**Extracted Fact Registry + Validity**
- `extracted_fact_registry_enriched` built through 2025
- `extracted_fact_registry_validity` built through 2025 (contradiction / expiration scaffold)

**CompanyState**
- Rebuilt using enriched facts + time series + events

---

## ⚠️ Missing / Incomplete

### 2026 Text Facts & Citations
`warehouse_doc_chunks/year=2026` is empty (0B).  
As a result:
- Text signals cannot be enriched with citation spans / speaker / paragraph info.
- 2026 `extracted_fact_registry_enriched` is effectively empty.
- 2026 `extracted_fact_registry_validity` is not meaningful.

### Root Cause
The upstream **doc chunking pipeline** for 2026 is not present in this repo (no `doc_chunks` builder found).  
So full 2026 text facts require the external chunking pipeline to be re-run.

---

## Next Steps to Restore 2026

1. Rebuild `warehouse_doc_chunks` for 2026 (from the upstream chunking pipeline).
2. Rebuild `warehouse_text_signals` for 2026.
3. Re-run:
   - `scripts/build_extracted_fact_registry.py --years 2026`
   - `scripts/enrich_extracted_fact_registry.py --years 2026`
   - `scripts/build_financial_facts_registry.py --years 2026`
   - `scripts/build_fact_validity.py --years 2026`

---

## Summary

The Data Plane is **complete and validated through 2025**.  
2026 is missing only because **doc chunks + signals are absent**, not due to pipeline bugs.
