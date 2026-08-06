# Axiom V1 - Architecture Gap Analysis

## Current State vs Target Architecture

### Executive Summary

**V1 covers approximately 40-50% of the minimal build plan** and has the foundational pieces in place. The core "analog retrieval + outcomes" logic works. The main gaps are:
1. Proper bitemporal data handling (as-of enforcement)
2. Feature store formalization
3. EvidencePack + grounded narrative generation
4. Export to PPT
5. Audit logging

---

## Plane-by-Plane Analysis

### 1. DATA PLANE

| Component | Target | V1 Status | Gap |
|-----------|--------|-----------|-----|
| Raw Data Lake | Immutable, timestamped | **Partial** - Parquet files, no formal lake | Need versioning + available_time tracking |
| Normalized Warehouse | Bitemporal tables | **Partial** - Tables exist but no bitemporal logic | Need valid_from/valid_to + available_from/available_to |
| As-of Snapshot Builder | `get_snapshot(company, as_of)` | **YES** - `AsOfSnapshotBuilder` in `snapshot.py` | Works but uses `datadate`, not `rdq` consistently |
| Document Store | Chunks + embeddings | **NO** | Not needed for V1 MVP |
| Source Connectors | Adapters with timestamps | **Partial** - Scripts pull from WRDS | Need formalization |

**V1 Has:**
- `snapshot.py` with `AsOfSnapshotBuilder` class
- `get_snapshot(gvkey, as_of_date)` method
- `get_universe_snapshot()` for company lists
- Parquet storage for fundamentals, prices, deals

**Gap Priority: MEDIUM** - Works for MVP, needs hardening for production

---

### 2. FEATURE PLANE (Feature Store)

| Component | Target | V1 Status | Gap |
|-----------|--------|-----------|-----|
| Feature Store | Keyed by (entity, as_of) | **NO** - Computed on-the-fly | Need persistence layer |
| Base Metrics | Ratios, margins, leverage | **YES** - In `signals.py` | ✓ |
| Trend Metrics | Slopes, deltas | **Partial** - Some in signals | Need more |
| Peer-Relative | Percentiles vs peers | **Partial** - In `insights.py` | Needs expansion |
| Text-Derived | NLP signals | **NO** | Phase 2 |

**V1 Has:**
- Signal computation with base metrics
- On-the-fly feature calculation

**Gap Priority: LOW for V1** - On-the-fly works, formalize later

---

### 3. MODEL PLANE - SIGNALS

| Component | Target | V1 Status | Gap |
|-----------|--------|-----------|-----|
| Signal Engine | 10-20 signals | **YES** - 7 signals in `signals.py` | Could add more |
| Signal Confidence | Per-signal confidence | **Partial** - Some handling | Need explicit confidence scores |
| Signal Drivers | Top feature contributions | **NO** | Need to track which features drove each signal |
| Signal Explanations | 1-sentence per signal | **NO** | Need explanation generation |
| Stability Controls | Smoothing, hysteresis | **NO** | Need anti-whipsaw logic |

**V1 Has (in `signals.py`):**
```
1. balance_sheet_optionality
2. growth_momentum
3. valuation_dislocation
4. margin_trend
5. refinancing_pressure
6. size_factor
7. asset_intensity
```

**Gap Priority: MEDIUM** - Works, but missing drivers/explanations

---

### 4. MODEL PLANE - REGIMES

| Component | Target | V1 Status | Gap |
|-----------|--------|-----------|-----|
| Regime Model | 3-4 interpretable regimes | **YES** - `regimes.py` | ✓ |
| Regime Inference | `infer_regime(features)` | **YES** - `classify_regime()` | ✓ |
| Regime Characteristics | Action implications | **YES** - `get_regime_characteristics()` | ✓ |
| Transition Detection | "Is regime changing?" | **NO** | Nice to have |

**V1 Has:**
- 3 regimes: LOOSE, SELECTIVE, TIGHT
- Based on VIX + credit spreads (mocked data currently)
- Regime characteristics for deal activity, financing

**Gap Priority: LOW** - Works well

---

### 5. MODEL PLANE - CASE STORE & ANALOG RETRIEVAL

| Component | Target | V1 Status | Gap |
|-----------|--------|-----------|-----|
| Case Store | Historical cases with signals + outcomes | **YES** - `clean_action_profiles.parquet` | ✓ |
| Case Definition | (company, time, signals, action, outcomes) | **YES** - In profiles | ✓ |
| Analog Retrieval | Similarity ranking | **YES** - `_find_similar_from_profiles()` | ✓ |
| Sector Filtering | Match within industry | **YES** - `sector_filter` param | ✓ |
| Regime Filtering | Match within regime | **NO** | Should add |
| Sparse Handling | Expand radius when N low | **NO** | Need fallback logic |
| N Thresholds | Quality warnings | **NO** | Need minimum N checks |

**V1 Has:**
- ~40K+ action profiles (buybacks, acquisitions, bankruptcies, dividends)
- Cosine similarity matching
- Sector filtering (2-digit SIC)
- Action weighting for data imbalance

**Gap Priority: MEDIUM** - Core works, needs sparse handling + regime filter

---

### 6. MODEL PLANE - OUTCOMES

| Component | Target | V1 Status | Gap |
|-----------|--------|-----------|-----|
| Outcome Calculator | TSR at horizons | **YES** - `outcomes.py` | ✓ |
| Empirical Quantiles | P25/P50/P75 | **Partial** - Median only | Need full quantiles |
| Confidence Intervals | Widen as N shrinks | **NO** | Need uncertainty handling |
| Ex-ante Framing | Only as-of data | **YES** - By design | ✓ |

**V1 Has:**
- `OutcomeCalculator` class
- TSR at 1m, 3m, 6m, 12m horizons
- Stored in action profiles

**Gap Priority: LOW** - Works, enhance with quantiles later

---

### 7. NARRATIVE PLANE

| Component | Target | V1 Status | Gap |
|-----------|--------|-----------|-----|
| Evidence Pack | Structured grounded data | **Partial** - `insights.py` generates ideas | Need formal EvidencePack object |
| Story Templates | Deterministic templates | **Partial** - `_generate_ideas()` | Need more templates |
| LLM Usage | Bounded phrasing only | **NO** | Not using LLM yet |
| Grounded Validation | Check all claims | **NO** | Need validator |

**V1 Has:**
- `InsightsGenerator` class
- Idea scoring (M&A, Buyback, Debt, Dividend)
- "Why Now" bullet generation
- Decision table generation
- Constraint identification

**Gap Priority: HIGH** - Need formal EvidencePack for audit trail

---

### 8. PRODUCT PLANE

| Component | Target | V1 Status | Gap |
|-----------|--------|-----------|-----|
| Web UI | Company picker, tabs, charts | **YES** - Streamlit app | ✓ |
| State Profile Dashboard | Signals + drivers | **YES** - Radar chart + details | ✓ |
| Action Cards | Per archetype | **YES** - Idea cards | ✓ |
| Analog Explorer | Cohorts + examples | **Partial** - Similar cases list | Need distribution charts |
| PPTX Export | Slide-ready | **NO** | HIGH PRIORITY GAP |
| API Services | REST endpoints | **NO** | Not needed for V1 pilot |

**V1 Has:**
- Streamlit app with Overview/Logic/Evidence tabs
- Company selection with date picker
- Signal radar chart
- Historical precedent cards
- Peer comparison table
- Decision table + constraints

**Gap Priority: HIGH** - Need PPT export

---

### 9. TRUST PLANE

| Component | Target | V1 Status | Gap |
|-----------|--------|-----------|-----|
| Audit Logging | Immutable run records | **NO** | Need for production |
| Version Tracking | All model versions | **NO** | Need for production |
| Evidence Pointers | DB row IDs | **NO** | Partial - could add |
| Security | RBAC, encryption | **NO** | Not needed for pilot |

**Gap Priority: MEDIUM** - Not blocking pilot, critical for production

---

### 10. MLOPS PLANE

| Component | Target | V1 Status | Gap |
|-----------|--------|-----------|-----|
| Training Pipelines | Offline model training | **N/A** - V1 is rule-based | Not needed |
| Evaluation | Calibration metrics | **NO** | Nice to have |
| Drift Detection | Monitor regime changes | **NO** | Nice to have |

**Gap Priority: LOW** - V1 is deterministic/rule-based, no ML training needed

---

## Summary: What V1 Has vs Needs

### ✅ V1 HAS (Working)
1. As-of Snapshot Builder
2. 7 interpretable signals
3. 3 market regimes
4. Case store with 40K+ action profiles
5. Analog retrieval with sector filtering
6. Outcome calculation (TSR)
7. Insights generation (ideas, why now, decision table)
8. Web UI with tabs matching mock

### ⚠️ V1 NEEDS (High Priority for Pilot)
1. **PPT/PDF Export** - Bankers need slides
2. **Formal EvidencePack** - For audit trail
3. **Sparse handling** - What to do when N < 10
4. **Signal drivers** - "What's driving this score?"
5. **Quantile distributions** - P25/P50/P75 not just median

### 📋 V1 CAN DEFER (Post-Pilot)
1. Bitemporal data enforcement
2. Feature store persistence
3. LLM-based phrasing
4. Text/NLP signals
5. Audit logging infrastructure
6. API microservices
7. Security/RBAC

---

## Recommended Next Steps

### Phase 1: Pilot-Ready (1-2 weeks)
1. Add PPT export using `python-pptx`
2. Formalize EvidencePack structure
3. Add signal drivers (top 3 features per signal)
4. Add N-threshold warnings ("Based on N=12 cases")
5. Add quantile displays (P25/P50/P75)

### Phase 2: Production-Ready (2-4 weeks)
1. Bitemporal data layer
2. Audit logging
3. Regime filtering in analog retrieval
4. Sparse fallback logic
5. Stability controls for signals

### Phase 3: Scale (1-2 months)
1. Feature store
2. API services
3. LLM narrative layer
4. Text/NLP signals
5. MLOps infrastructure
