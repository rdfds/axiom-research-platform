# Causal Baseline

Validated on March 14, 2026.

## Production Defaults

Use:

- model: `./data/models/causal_impact_model_v5_5_hybrid.json`
- blocklist: `./config/causal_action_blocklist_prod_v2.txt`

`./scripts/run_recommendation_prod.py` now defaults to that blocklist path unless overridden.

## Current Status

Broad 20-company validation after the M&A causal unblocking:

- source: `/tmp/ml_status_causal_v2_20.json`
- `causal_rate_mean = 0.848167`
- `strict_causal_mean = 1.0`

Action-level status from that validation:

- `mna.platform_acquisition`
  - `causal_rate = 1.0`
  - `strict_pass_rate = 1.0`
- `mna.tuck_in_acquisition`
  - `causal_rate = 1.0`
  - `strict_pass_rate = 1.0`

Post-patch targeted validation:

- `capital_return.special_dividend`
  - source: `/tmp/recommendation_runs_special_dividend_v1`
  - `causal_rate = 1.0` on the targeted 20-company run
  - example causal quality:
    - `causal_model_blend_weight = 0.25047`
    - `causal_model_quality = 0.093651`
    - `causal_model_support_score = 0.857544`
    - `causal_model_min_oos_r2 = 0.091365`

## Intentional Exception

`capital_structure.revolver_draw_or_resize` is intentionally excluded from production causal blending.

Reason:

- targeted rescue training with subtype-aware loan/revolver cells still failed out-of-sample quality gates
- the rescue artifact is:
  - `./data/models/causal_impact_model_v5_6_revolver_rescue.json`
  - `./data/models/causal_impact_model_v5_6_revolver_rescue.model_card.json`
- resulting loan/revolver cells remained disabled with negative OOS R2

Representative rescue results:

- `loan_issuance::all`
  - `n_valid = 219`
  - `oos_r2 = -0.6314524314607071` for `value_creation`
- `loan_issuance::revolver_line_1_yr`
  - `n_valid = 106`
  - `oos_r2 = -1.0` across objectives

Policy:

- keep `capital_structure.revolver_draw_or_resize` precedent-driven for now
- do not re-enable causal support unless a future model clears the existing OOS quality gate

## Notes

- This is a modeling limitation, not a routing bug.
- The current causal stack is strong for:
  - core capital return
  - core capital structure
  - platform/tuck-in M&A
  - special dividend
- Revolver remains the main causal exception in otherwise standard finance actions.
- broader non-regression monitoring is documented in:
  - `./docs/model_monitoring.md`

## Newly Added Actions

The following standard actions were added after the current accepted causal batch:

- `capital_return.dividend_initiate`
- `mna.go_private_lbo`

Current status:

- `capital_return.dividend_initiate`
  - ontology, normalization, precedent, and snapshot-gating support exist
  - targeted causal routing validation now exists against:
    - source: `/tmp/causal_dividend_initiate_probe.json`
    - attached benchmark artifact:
      - `/tmp/recommendation_runs_dividend_initiate_gate_check/artifacts/run_id=e354efc1-6392-456f-ac0f-14731334bf17/CausalBenchmark_causal_dividend_initiate_probe_dividend_initiate.json`
    - snapshot: `./data/company_state_snapshots/final_run_2026-02-28/keyed/as_of_date=2026-02-28/company_id=0000794619.json`
    - model: `./data/models/causal_impact_model_v5_5_hybrid.json`
  - current causal fallback routes to `dividend_regular::regular`
  - enabled objective cells selected:
    - `value_creation`
    - `risk_reduction`
    - `rating_preservation`
    - `optionality`
  - targeted causal routing metrics:
    - `blend_weight = 0.238987`
    - `coverage_score = 0.346039`
    - `model_quality = 0.076831`
    - `min_oos_r2 = 0.059732`
    - `support_score = 0.937545`
    - `out_of_sample_flag = False`
  - production-style end-to-end validation is still noisy on this machine because the full runner intermittently hits an OpenMP shared-memory error, but the routing and model selection are now validated

- `mna.go_private_lbo`
  - ontology, normalization, and precedent support exist
  - causal alias routing exists
  - targeted causal probe:
    - source: `/tmp/causal_go_private_lbo_probe.json`
    - selected model keys:
      - `acquisition::all` on `value_creation`
      - `acquisition::all` on `growth`
    - `coverage_score_mean = 0.362296`
    - `model_quality_mean = 0.089837`
    - `support_score_mean = 0.92564`
    - `blend_weight_mean = 0.247866`
    - `min_oos_r2_mean = 0.087272`
    - `oos_rate = 0.0`
  - formal decision: reject for production causal baseline
  - reason:
    - subtype-specific LBO causal cell `acquisition::acquisition_lbo` remains disabled in `./data/models/causal_impact_model_v5_5_hybrid.model_card.json`
    - current causal support only comes from the broad `acquisition::all` fallback
    - that is not action-specific enough to approve LBO as a production causal-baseline action
  - production policy:
    - `mna.go_private_lbo` is blocklisted in `./config/causal_action_blocklist_prod_v2.txt`
    - treat it as precedent-supported first, not a causal-baseline action

Operationally:

- both actions are available in the action universe
- `capital_return.dividend_initiate` now has targeted causal-routing validation
- `mna.go_private_lbo` is now formally rejected from the production causal baseline and kept precedent-only
- targeted causal routing harness:
  - `./scripts/benchmark_causal_actions.py`
  - built-in preset now includes:
    - `mna.platform_acquisition`
    - `mna.tuck_in_acquisition`
    - `capital_return.special_dividend`
    - `capital_return.dividend_initiate`
