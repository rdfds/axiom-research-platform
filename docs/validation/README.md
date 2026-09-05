# Validation Overview

This folder summarizes the validation work that should be visible in a public GitHub review. The live workspace contains many generated artifacts under local `out/` folders and `./data/`; this document pulls the highest-signal results into one place.

## Validation Philosophy

Axiom tries to avoid model confidence theater. The validation standard is:

- point-in-time inputs only
- out-of-sample or walk-forward splits where possible
- explicit placebo or baseline comparisons for market-implied claims
- model family gates before a score can become decision evidence
- honest fallbacks when exact samples are thin
- materiality labels so tiny effects are not oversold

