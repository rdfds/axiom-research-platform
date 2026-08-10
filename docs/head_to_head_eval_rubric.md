# Head-To-Head Benchmark

This benchmark compares:

- the model dossier
- a public or human-written baseline memo

It is not a pure truth test. It is a structured comparison of:

- idea quality
- justification quality
- timing logic
- alternative analysis
- risk framing

## Baseline Sources

The easiest usable baseline sources are public:

1. activist letters
2. investor letters
3. earnings-call capital-allocation commentary
4. transaction announcement rationale from filings
5. rating or credit-rationale summaries

Do not wait for real banker memos. They are usually private.

## Baseline File Format

Put one file per company in a directory such as:

`./data/head_to_head_baselines`

Supported names:

- `company_id=0000320193.md`
- `company_id=0000320193.json`
- `0000320193.md`
- `0000320193.json`

Add baseline metadata so the report can separate fair comparisons from weak ones:

- `Baseline-Type: activist | investor_letter | investor_campaign | rating_note | management_ir`
- `Task-Match: direct | partial | weak`

### Markdown Template

```md
Baseline-Type: activist
Task-Match: direct

# Problem
The company has excess capital and limited near-term higher-return uses.

# Recommendation
Prefer `capital_return.open_market_buyback` over a special dividend because repurchases are more flexible and better exploit undervaluation.

# Why Now
Undervaluation is still open, leverage is manageable, and waiting mostly leaves capital idle.

# Alternatives
- `capital_return.special_dividend` is less flexible.
- `mna.tuck_in_acquisition` does not clear the return hurdle today.

# Risks
- If operating conditions weaken, the company may need the capital back.
- If valuation rerates immediately, repurchase value capture shrinks.

# Kill Criteria
- Stop if leverage rises materially.
- Stop if a higher-return strategic use for capital appears.

# Evidence
- Net leverage is 1.8x.
- Deployable liquidity is about 10% of market value.
- FCF conversion remains strong.
```

### JSON Template

```json
{
  "source_label": "public_investor_letter",
  "baseline_type": "investor_letter",
  "task_match": "direct",
  "primary_recommendation": "open market buyback",
  "action_path": ["capital_return.open_market_buyback"],
  "problem_statement": "The company has excess capital and limited near-term higher-return uses.",
  "recommendation_thesis": "Prefer repurchases over a special dividend because they are more flexible and better exploit undervaluation.",
  "why_now": "Undervaluation is still open, leverage is manageable, and waiting mostly leaves capital idle.",
  "alternatives": [
    "A special dividend is less flexible.",
    "A tuck-in acquisition does not clear the return hurdle today."
  ],
  "risks": [
    "The business may need the capital if conditions weaken."
  ],
  "kill_criteria": [
    "Stop if leverage rises materially.",
    "Stop if a higher-return strategic use appears."
  ],
  "evidence_points": [
    "Net leverage is 1.8x.",
    "Deployable liquidity is about 10% of market value."
  ]
}
```

