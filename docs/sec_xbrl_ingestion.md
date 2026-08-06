# SEC XBRL Ingestion (Company Facts)

This pipeline pulls SEC `companyfacts` JSON, writes raw payloads to the lake,
and normalizes facts into `data/warehouse/warehouse_financials.parquet` with
bitemporal enforcement.

## Prereqs

- Set your SEC user agent:

```bash
export SEC_USER_AGENT="Axiom Research (you@example.com)"
```

## Smoke Test (single ticker)

```bash
python -u scripts/26_ingest_sec_xbrl.py --tickers AAPL --start 2020-01-01 --end 2024-12-31 --limit 1
```

## Full Universe (R3000 proxy)

```bash
python -u scripts/26_ingest_sec_xbrl.py --start 2000-01-01 --end 2025-12-31
```

## Incremental (latest only)

```bash
python -u scripts/26_ingest_sec_xbrl.py --incremental
```

By default, `--end` is set to today’s date.

## Notes

- CIKs are resolved from SEC `company_tickers.json` and your CRSP universe.
- Forms default to 10-K and 10-Q (use `--include-8k` to add 8-K).
- Raw payloads are cached under `data/sec/companyfacts/`.
