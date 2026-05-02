from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.company_state_builder as company_state_builder
from src.company_state_builder import CompanyStateBuilder


def _write_parquet(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _base_builder(
    tmp_path: Path,
    facts_path: Path,
    dealscan_revolver_path: Path | None = None,
    timeseries_path: Path | None = None,
    skip_timeseries: bool = True,
    skip_macro: bool = True,
    events_path: Path | None = None,
    skip_events: bool = True,
    entity_table_path: Path | None = None,
    entity_identifier_path: Path | None = None,
    taxonomy_reference_path: Path | None = None,
    skip_peer_context: bool = True,
    ownership_path: Path | None = None,
    issuer_ratings_path: Path | None = None,
    estimates_path: Path | None = None,
    corporate_actions_path: Path | None = None,
    historical_backfill_mode: bool = False,
    companyfacts_root: Path | None = None,
    enable_market_relevant_smart_normalized_inputs: bool = False,
) -> CompanyStateBuilder:
    entity_path = entity_table_path or (tmp_path / "entity.parquet")
    ident_path = entity_identifier_path or (tmp_path / "entity_identifier.parquet")
    taxonomy_ref = taxonomy_reference_path or (tmp_path / "taxonomy_reference.parquet")
    event_store = events_path or (tmp_path / "events.parquet")
    raw_ts = timeseries_path or (tmp_path / "timeseries.parquet")
    ownership = ownership_path or (tmp_path / "ownership.parquet")
    ratings = issuer_ratings_path or (tmp_path / "issuer_ratings.parquet")
    estimates = estimates_path or (tmp_path / "warehouse_estimates.parquet")
    return CompanyStateBuilder(
        raw_timeseries_path=raw_ts,
        macro_timeseries_path=raw_ts,
        event_store_path=event_store,
        corporate_actions_master_path=corporate_actions_path or (tmp_path / "corporate_actions_master.parquet"),
        facts_path=facts_path,
        dealscan_revolver_path=dealscan_revolver_path or (tmp_path / "dealscan_revolver.parquet"),
        ownership_summary_path=ownership,
        issuer_ratings_path=ratings,
        estimates_path=estimates,
        entity_table_path=entity_path,
        entity_identifier_path=ident_path,
        taxonomy_reference_path=taxonomy_ref,
        skip_timeseries=skip_timeseries,
        skip_macro=skip_macro,
        skip_events=skip_events,
        skip_peer_context=skip_peer_context,
        historical_backfill_mode=historical_backfill_mode,
        companyfacts_root=companyfacts_root,
        enable_market_relevant_smart_normalized_inputs=enable_market_relevant_smart_normalized_inputs,
    )


def _facts_row(
    fact_id: str,
    entity_id: str,
    fact_type: str,
    fact_value,
    published_at: str,
    ingested_at: str,
    valid_from: str,
    valid_to=None,
):
    return {
        "fact_id": fact_id,
        "entity_id": entity_id,
        "fact_type": fact_type,
        "fact_value": fact_value,
        "confidence_score": 0.9,
        "source_type": "SEC",
        "published_at": published_at,
        "ingested_at": ingested_at,
        "valid_from": valid_from,
        "valid_to": valid_to,
    }


def _note_fact_row(
    document_id: str,
    entity_id: str,
    metric_key: str,
    value,
    bucket_label: str | None,
    published_at: str,
):
    return {
        "document_id": document_id,
        "entity_id": entity_id,
        "metric_key": metric_key,
        "value": value,
        "bucket_label": bucket_label,
        "source_type": "sec_edgar_filing",
        "published_at": published_at,
        "ingested_at": published_at,
        "effective_at": published_at,
        "extraction_confidence": 0.86,
    }


