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

