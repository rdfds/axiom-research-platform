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

