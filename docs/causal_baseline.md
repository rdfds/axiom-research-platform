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

