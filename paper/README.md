# Paper Package

This folder contains the publication-oriented version of the Axiom market-implied valuation gap work.

## Commands

Public smoke run:

```bash
python scripts/paper/run_market_expectations_experiments.py --preset smoke
```

Private-data publication run:

```bash
python scripts/paper/run_market_expectations_experiments.py --preset publication
```

The publication preset uses the private valuation validation engine when it is
available locally. In a sanitized clone, set `AXIOM_MARKET_EXPECTATIONS_PANEL`
to a private panel with the same schema as `fixtures/market_expectations_smoke_panel.csv`
to run the canonical walk-forward/placebo pipeline without committing raw data.

Build case-study artifacts:

```bash
python scripts/paper/build_case_studies.py
```

## Contents

- `methodology.md`: notation and empirical protocol.
- `model_card_market_expectations.md`: intended use, limitations, and non-claims.
- `manuscript_skeleton.md`: paper draft scaffold.
- `fixtures/`: public smoke fixture.
- `results/`: generated result artifacts.
- `case_studies/`: generated decomposition examples.
