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


def _entity_row(entity_id: str, *, sector: str | None = None, subsector: str | None = None, sic: str | None = None):
    return {
        "entity_id": entity_id,
        "sector": sector,
        "subsector": subsector,
        "gics_sector": sector,
        "gics_sub_industry": subsector,
        "sic": sic,
    }


def test_asof_filters_future_facts(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    rows = [
        _facts_row(
            fact_id="cash_old",
            entity_id="ABC",
            fact_type="financial.cash",
            fact_value=100.0,
            published_at="2026-02-01T00:00:00Z",
            ingested_at="2026-02-01T00:00:00Z",
            valid_from="2026-02-01T00:00:00Z",
        ),
        _facts_row(
            fact_id="cash_future",
            entity_id="ABC",
            fact_type="financial.cash",
            fact_value=999.0,
            published_at="2026-03-01T00:00:00Z",
            ingested_at="2026-03-01T00:00:00Z",
            valid_from="2026-03-01T00:00:00Z",
        ),
    ]
    _write_parquet(facts_path, rows)

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-02-28")
    cash = snap.features["liquidity.cash"]["value"]
    assert cash == 100.0


def test_load_facts_falls_back_to_pandas_when_duckdb_scan_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    facts_dir = tmp_path / "facts"
    _write_parquet(
        facts_dir / "year=2024" / "part.parquet",
        [
            _facts_row(
                fact_id="cash_current",
                entity_id="ABC",
                fact_type="financial.cash",
                fact_value=125.0,
                published_at="2024-05-01T00:00:00Z",
                ingested_at="2024-05-01T00:00:00Z",
                valid_from="2024-05-01T00:00:00Z",
            ),
            _facts_row(
                fact_id="cash_other",
                entity_id="XYZ",
                fact_type="financial.cash",
                fact_value=999.0,
                published_at="2024-05-01T00:00:00Z",
                ingested_at="2024-05-01T00:00:00Z",
                valid_from="2024-05-01T00:00:00Z",
            ),
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_dir, skip_timeseries=True)

    class _BrokenConnection:
        def execute(self, _query: str):
            raise RuntimeError("duckdb parquet scan failed")

    monkeypatch.setattr(company_state_builder.duckdb, "connect", lambda: _BrokenConnection())

    df = builder._load_facts("ABC", pd.Timestamp("2024-06-01T00:00:00Z"))
    assert len(df) == 1
    assert df.iloc[0]["entity_id"] == "ABC"
    assert df.iloc[0]["fact_id"] == "cash_current"


def test_is_readable_file_allows_large_zero_block_files_without_probative_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    candidate = tmp_path / "placeholder.parquet"
    candidate.write_text("placeholder")

    real_stat = candidate.stat()
    original_stat = Path.stat

    class _FakeStat:
        st_size = real_stat.st_size
        st_blocks = 0

    monkeypatch.setattr(Path, "stat", lambda self: _FakeStat() if self == candidate else original_stat(self))

    def _unexpected_open(*_args, **_kwargs):
        raise AssertionError("placeholder probe should not open the file")

    monkeypatch.setattr(company_state_builder, "open", _unexpected_open, raising=False)

    assert company_state_builder._is_readable_file(candidate) is True


def test_historical_backfill_mode_ignores_ingested_cutoff_for_facts(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row(
                fact_id="cash_backfilled",
                entity_id="ABC",
                fact_type="financial.cash",
                fact_value=123.0,
                published_at="2024-08-01T00:00:00Z",
                ingested_at="2026-02-01T00:00:00Z",
                valid_from="2024-08-01T00:00:00Z",
            ),
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        historical_backfill_mode=True,
    )
    snap = builder.build("ABC", "2024-09-01")
    assert snap.features["liquidity.cash"]["value"] == 123.0


def test_null_contradiction_group_does_not_collapse_fact_history(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            {
                **_facts_row("rev_q1", "ABC", "financial.revenue", 90.0, "2025-01-20T00:00:00Z", "2025-01-21T00:00:00Z", "2025-01-20T00:00:00Z"),
                "effective_at": "2024-12-31T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2024-12-31 00:00:00; fiscal_year=2025; fiscal_quarter=1",
                "contradiction_group_id": None,
            },
            {
                **_facts_row("rev_q2", "ABC", "financial.revenue", 120.0, "2025-04-20T00:00:00Z", "2025-04-21T00:00:00Z", "2025-04-20T00:00:00Z"),
                "effective_at": "2025-03-31T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2025-03-31 00:00:00; fiscal_year=2025; fiscal_quarter=2",
                "contradiction_group_id": None,
            },
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    facts = builder._load_facts("ABC", pd.Timestamp("2026-02-28", tz="UTC"))
    revenue_series, _ = builder._dated_fact_series(facts, builder.fact_map["revenue"])
    assert len(revenue_series) == 2


