# Data Storage Plan

## Goal

Keep code in Git, but move large data out of the repo working tree so we do not
depend on Desktop/OneDrive hydration for day-to-day reads.

## Recommended Layout

- Code repo:
  - local checkout plus GitHub backup
- Live data root:
  - set `AXIOM_DATA_ROOT` to an alternate data location
  - examples:
    - `/Volumes/AxiomData/axiom_data`
    - `./data`
- Optional companyfacts override:
  - set `AXIOM_COMPANYFACTS_ROOT`

## First Folders To Move

These are the highest-value folders to relocate first:

- `data/curated/`
- `data/models/`
- `data/inputs_layer/raw_timeseries.parquet`
- `data/inputs_layer/entity_graph.parquet`
- `data/inputs_layer/entity_identifier.parquet`
- `data/inputs_layer/entity.parquet`
- `data/inputs_layer/extracted_fact_registry_validity/`

## Keep Local For Now

These are still more naturally local filesystem reads today:

- `data/sec/companyfacts/`

