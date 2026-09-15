# Precedent Retrieval Example

This example shows Axiom's historical analog engine.

The sample file is:

```text
examples/precedent_retrieval/precedent_retrieval.sample.json
```

It is a compact extract from the existing precedent retrieval sample bundle.

## What The System Does

For each candidate corporate action, the retrieval layer returns:

- action parameters
- feasibility status
- top historical analogs
- similarity scores
- matched cohort size
- precedent confidence
- outcome distributions
- empirical risk profile
- mismatch diagnostics

This is not meant to be a raw precedent table. It is a retrieval and evidence layer.

## What To Look At

Open the sample and inspect one result:

```json
{
  "action_id": "capital_return.open_market_buyback",
  "precedent_confidence": 0.5912,
  "retrieval_tier": "exact",
  "cohort_size": 30,
  "top_similarity_mean": 0.8747
}
```

Then look at:

- `top_matches`
- `outcome_distribution_12m_or_nearest`
- `mismatch_diagnostics`
- `empirical_risk_profile`

Those fields show whether the analogy is actually credible.

