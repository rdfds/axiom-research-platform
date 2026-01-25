# SEC XBRL Ingestion (Company Facts)

This pipeline pulls SEC `companyfacts` JSON, writes raw payloads to the lake,
and normalizes facts into `data/warehouse/warehouse_financials.parquet` with
bitemporal enforcement.

