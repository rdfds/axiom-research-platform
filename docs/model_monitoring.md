# Model Monitoring Contract

This document defines the review and promotion contract for Axiom model and policy changes. It separates checks reproducible from committed fixtures from provider-backed checks that require licensed data.

## 1. Data contract

Before scoring or training, validate:

- as-of timestamps and publication dates are compatible with the decision date;
- entity identifiers resolve without ambiguous joins;
- units, currencies, and fiscal periods are normalized;
- required provenance and confidence fields are present;
- missingness and fallback rates stay within the candidate artifact's declared bounds.

A material schema, coverage, or fallback-rate change blocks promotion until the affected slices are reviewed.

