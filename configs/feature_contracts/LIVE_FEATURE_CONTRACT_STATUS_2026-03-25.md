# Live Feature Contract Status (2026-03-25)

This memo separates the features referenced by the live policy stack into:

- core live features
- source-limited live features
- unwired / contract-mismatch features

The policy references were inspected in:

- `./src/candidate_generation.py`
- `./src/mechanism_brain.py`
- `./src/recommendation_run.py`

The live builder output was checked in:

- `./src/company_state_builder.py`

Representative live snapshot used for sanity check:

- `/tmp/hd_live_snapshot_2024_12_31/company_state_snapshots_asof=2024-12-31.jsonl`

Representative source checks for `0000354950` (Home Depot):

- `event_store.parquet`: `0` company rows
- `ownership_13f_summary.parquet`: `4409` company rows
- `issuer_rating_history.parquet`: `204` company rows
- `entity_graph.parquet`: `5` related rows

