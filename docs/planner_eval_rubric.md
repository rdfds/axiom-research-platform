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

## Human Gates

Review the markdown queue and score each case:

- Top-1 plan is strategically sensible
- Top-3 contains no obvious nonsense
- Explanation is persuasive and numbers-backed
- Risks / triggers / branches are useful

Acceptance thresholds:

- Top-1 sensible: at least 80%
- Top-3 sensible: at least 90%
- Explanation persuasive: at least 80%
- Risks / triggers useful: at least 75%

## Recommended Eval Set

Do not rely only on the fixed 5-company regression set.

Preferred process:

1. run a broader 50-company production batch
2. generate the planner eval report
3. review the markdown queue
4. turn repeated failures into planner regression fixtures

## Command

```bash
cd .

PYTHONPATH=. \
python ./scripts/evaluate_planner_quality.py \
  --runs-roots /tmp/recommendation_runs_prod_causal_v2_20 \
  --out-json /tmp/planner_eval_report.json \
  --out-md /tmp/planner_eval_report.md \
  --review-count 20
```
