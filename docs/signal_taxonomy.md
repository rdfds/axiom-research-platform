# Signal Taxonomy (MVP)

This taxonomy defines the **initial** unstructured text signals extracted from
documents (press releases, transcripts, research, presentations). Each signal
produces:

- `signal_name`
- `value` (normalized score or numeric)
- `confidence` (0–1)
- `supporting_chunk_ids` (array)

## Conventions

- **Score range** for qualitative signals: `0–100` (higher = stronger presence).
- **Binary** signals: `0` or `100`.
- **Confidence**: model‑reported or heuristic quality, `0–1`.
- **Value type**: numeric; categorical values are encoded to numeric with a
  documented mapping.

