# Planner Evaluation Rubric

Use `./scripts/evaluate_planner_quality.py` to generate:

- machine-readable planner quality metrics
- a markdown review queue for human scoring

The planner is considered ready only when both heuristic and human gates are met.

## Heuristic Gates

These are automatic checks from the eval harness:

- `positive_top_plan_rate >= 0.90`
  - top plan raw score should be positive on at least 90% of reviewed runs
- `supported_top_plan_rate >= 0.90`
  - every top-plan step should have precedent support or causal support on at least 90% of runs
- `explanation_complete_rate >= 0.95`
  - top plan should have a summary explanation plus complete step explanations
- `heuristic_overall_mean >= 0.75`
  - aggregate planner quality should stay comfortably above the “review required” band

Any case with one or more of these flags should be reviewed manually:

- `top_plan_nonpositive`
- `top_plan_unsupported_step`
- `top_plan_summary_missing`
- `step_explanation_incomplete`
- `top3_contains_nonpositive`
- `top3_duplicate_paths`

