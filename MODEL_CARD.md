# Axiom Model Card

## Summary

Axiom is a point-in-time decision-evidence system for corporate-finance analysis. It combines timestamp-aligned company state, valuation-driver models, learned precedent retrieval, action-evidence scoring, and deterministic decision contracts. The repository includes the runnable modeling and evaluation layers, with committed fixtures for reproducibility; licensed datasets and credentials are excluded.

The system is deliberately modular. A model score cannot silently become a recommendation: downstream contracts preserve provenance, support, uncertainty, fallbacks, and evidence limitations.

## Intended use

Axiom is designed to help an analyst investigate questions such as:

- Which operating or balance-sheet drivers best explain a peer-relative valuation gap?
- Which historical corporate actions are sufficiently comparable to review?
- How strong is the available evidence for an action, and where is support thin?
- What assumptions, objections, regret cases, and monitoring triggers should accompany a recommendation?

It is not designed to autonomously trade, provide investment advice, or establish causal effects from observational relationships. Human review is required before any real decision.

## System layers

| Layer | Inputs | Output and safeguards |
|---|---|---|
| Company state | Financial, market, filing, macro, and action observations | As-of feature snapshot with timing, units, provenance, confidence, and fallback flags |
| Valuation drivers | Company state and peer context | Driver surfaces and forward-gap policies evaluated on time-ordered splits |
| Precedent retrieval | Candidate actions and historical outcomes | Learned distance/reranking signals with mismatch reasons and cohort support |
| Action evidence | Observed actions, outcomes, and controls | Quality-gated evidence scores; causal language is withheld when identification is insufficient |
| Decision contract | Model outputs and evidence packs | Recommendation, sizing, objections, regret cases, and monitoring triggers |

