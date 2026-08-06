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


def test_negative_ebitda_sets_leverage_null(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    rows = [
        _facts_row("cash", "ABC", "financial.cash", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        _facts_row("debt", "ABC", "financial.total_debt", 1000.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        _facts_row("ebitda", "ABC", "financial.ebitda", -25.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
    ]
    _write_parquet(facts_path, rows)

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-02-28")
    feat = snap.features["capital_structure.net_leverage"]
    assert feat["value"] is None
    assert feat["missing_reason"] == "negative_ebitda"


def test_dealscan_revolver_fallback_populates_proxy_value(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    dealscan_path = tmp_path / "dealscan_revolver.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"

    _write_parquet(
        facts_path,
        [
            _facts_row(
                "cash",
                "ABC",
                "financial.cash",
                100.0,
                "2024-12-15T00:00:00Z",
                "2024-12-15T00:00:00Z",
                "2024-12-15T00:00:00Z",
            ),
        ],
    )
    _write_parquet(
        ident_path,
        [
            {"entity_id": "ABC", "identifier_value": "ABC", "identifier_type": "ticker"},
        ],
    )
    _write_parquet(
        dealscan_path,
        [
            {
                "ticker": "ABC",
                "borrower_name_norm": "EXAMPLE CORP",
                "parent_norm": "EXAMPLE CORP",
                "loanconnector_company_id": "101",
                "loanconnector_tranche_id": "7001",
                "wrds_facility_id": "222",
                "tranche_type": "Revolver/Line >= 1 Yr.",
                "tranche_active_date": "2024-01-15",
                "tranche_maturity_date": "2029-01-15",
                "tranche_amount_converted_usd": 2_500_000_000.0,
            }
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        dealscan_revolver_path=dealscan_path,
        entity_identifier_path=ident_path,
        skip_timeseries=True,
    )
    snap = builder.build("ABC", "2024-12-31")
    revolver = snap.features["liquidity.revolver_undrawn"]
    assert revolver["value"] == 2_500_000_000.0
    assert revolver["fallback_used"] == "dealscan_revolver_capacity"
    assert "dealscan_revolver_capacity_proxy" in (revolver.get("quality_flags") or [])
    assert revolver["support_mode"] == "proxy_missing_component"


def test_dealscan_revolver_fallback_rejects_ambiguous_ticker_only_match(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    dealscan_path = tmp_path / "dealscan_revolver.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"

    _write_parquet(facts_path, [])
    _write_parquet(
        ident_path,
        [
            {"entity_id": "COST", "identifier_value": "COST", "identifier_type": "ticker"},
        ],
    )
    _write_parquet(
        dealscan_path,
        [
            {
                "ticker": "COST",
                "borrower_name_norm": "COSTCO WHOLESALE CORP",
                "parent_norm": "COSTCO WHOLESALE CORP",
                "company_name_norm": "COSTCO WHOLESALE CORP",
                "loanconnector_company_id": "1",
                "loanconnector_tranche_id": "10",
                "tranche_active_date": "2024-01-01",
                "tranche_maturity_date": "2029-01-01",
                "tranche_amount_converted_usd": 1_000_000_000.0,
            },
            {
                "ticker": "COST",
                "borrower_name_norm": "COSTAIN GROUP PLC",
                "parent_norm": "COSTAIN GROUP PLC",
                "company_name_norm": "COSTAIN GROUP PLC",
                "loanconnector_company_id": "2",
                "loanconnector_tranche_id": "20",
                "tranche_active_date": "2024-01-01",
                "tranche_maturity_date": "2029-01-01",
                "tranche_amount_converted_usd": 2_000_000_000.0,
            },
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        dealscan_revolver_path=dealscan_path,
        entity_identifier_path=ident_path,
        skip_timeseries=True,
    )
    snap = builder.build("COST", "2024-12-31")
    revolver = snap.features["liquidity.revolver_undrawn"]
    assert revolver["value"] is None
    assert revolver["support_mode"] == "unsupported"
    covenant = snap.features["capital_structure.max_leverage_ratio_covenant_proxy"]
    assert covenant["value"] is None
    assert covenant["support_mode"] == "unsupported"


def test_dealscan_covenant_proxy_emits_restrictive_thresholds(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    dealscan_path = tmp_path / "dealscan_revolver.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"
    taxonomy_path = tmp_path / "taxonomy_reference.parquet"

    _write_parquet(
        facts_path,
        [
            _facts_row(
                "cash",
                "ABC",
                "financial.cash",
                100.0,
                "2024-12-15T00:00:00Z",
                "2024-12-15T00:00:00Z",
                "2024-12-15T00:00:00Z",
            ),
        ],
    )
    _write_parquet(
        ident_path,
        [
            {"entity_id": "ABC", "identifier_value": "ABC", "identifier_type": "ticker"},
        ],
    )
    _write_parquet(
        taxonomy_path,
        [
            {
                "Instrument": "ABC",
                "Company Common Name": "Example Corp",
            }
        ],
    )
    _write_parquet(
        dealscan_path,
        [
            {
                "ticker": "ABC",
                "borrower_name_norm": "EXAMPLE CORP",
                "parent_norm": "EXAMPLE CORP",
                "company_name_norm": "EXAMPLE CORP",
                "loanconnector_company_id": "101",
                "loanconnector_tranche_id": "7001",
                "wrds_facility_id": "222",
                "tranche_type": "Revolver/Line >= 1 Yr.",
                "tranche_active_date": "2024-01-15",
                "tranche_maturity_date": "2029-01-15",
                "tranche_amount_converted_usd": 2_500_000_000.0,
                "max_leverage_ratio": "3.50:1",
                "min_interest_coverage_ratio": "2.00:1",
                "min_fixed_charge_coverage_ratio": "1.25:1",
                "min_current_ratio": "1.10:1",
                "all_covenants_financial": "Max Leverage Ratio: Value is 3.50; Min. Interest Coverage Ratio: Value is 2.00",
            },
            {
                "ticker": "ABC",
                "borrower_name_norm": "EXAMPLE CORP",
                "parent_norm": "EXAMPLE CORP",
                "company_name_norm": "EXAMPLE CORP",
                "loanconnector_company_id": "101",
                "loanconnector_tranche_id": "7002",
                "wrds_facility_id": "223",
                "tranche_type": "Revolver/Line >= 1 Yr.",
                "tranche_active_date": "2024-06-01",
                "tranche_maturity_date": "2030-06-01",
                "tranche_amount_converted_usd": 1_000_000_000.0,
                "max_leverage_ratio": "4.25:1",
                "min_interest_coverage_ratio": "2.50:1",
                "min_fixed_charge_coverage_ratio": "1.40:1",
                "min_current_ratio": "1.00:1",
                "all_covenants_financial": "Max Leverage Ratio: Value is 4.25; Min. Interest Coverage Ratio: Value is 2.50",
            },
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        dealscan_revolver_path=dealscan_path,
        entity_identifier_path=ident_path,
        taxonomy_reference_path=taxonomy_path,
        skip_timeseries=True,
    )
    snap = builder.build("ABC", "2024-12-31")

    max_lev = snap.features["capital_structure.max_leverage_ratio_covenant_proxy"]
    min_int = snap.features["capital_structure.min_interest_coverage_ratio_covenant_proxy"]
    min_fcc = snap.features["capital_structure.min_fixed_charge_coverage_ratio_covenant_proxy"]
    min_curr = snap.features["capital_structure.min_current_ratio_covenant_proxy"]

    assert max_lev["value"] == 3.5
    assert min_int["value"] == 2.5
    assert min_fcc["value"] == 1.4
    assert min_curr["value"] == 1.1
    assert max_lev["support_mode"] == "proxy_missing_component"
    assert max_lev["fallback_used"] == "dealscan_active_revolver_covenants"
    assert "dealscan_covenant_proxy" in (max_lev.get("quality_flags") or [])
    assert "dealscan_multiple_facilities_aggregated" in (max_lev.get("quality_flags") or [])
    assert max_lev["component_breakdown"]["active_revolver_facility_count"] == 2
    assert max_lev["component_breakdown"]["selection_rule"] == "minimum_observed_threshold_across_active_revolver_facilities"
    assert min_int["component_breakdown"]["selection_rule"] == "maximum_observed_threshold_across_active_revolver_facilities"
    hard_constraints = {item["name"]: item for item in snap.constraint_set["hard"]}
    assert hard_constraints["capital_structure.max_leverage_ratio_covenant_proxy"]["value"] == 3.5
    assert hard_constraints["capital_structure.min_interest_coverage_ratio_covenant_proxy"]["value"] == 2.5


def test_market_metric_engine_emits_market_views_and_lineage(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    entity_path = tmp_path / "entity.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 25.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("restricted", "ABC", "financial.restricted_cash", 10.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "ABC", "financial.total_debt", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("lease_current", "ABC", "financial.lease_liability_current", 20.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("lease_long", "ABC", "financial.lease_liability_noncurrent", 80.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "ABC", "financial.ebitda", 50.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebit", "ABC", "financial.ebit", 40.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("interest", "ABC", "financial.interest_expense", 10.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("revenue", "ABC", "financial.revenue", 1_000.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("fcf", "ABC", "financial.free_cash_flow", -60.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [_entity_row("ABC", sector="Consumer Discretionary", subsector="Specialty Retail", sic="5331")],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        entity_table_path=entity_path,
    )
    snap = builder.build("ABC", "2026-02-28")

    assert snap.features["taxonomy.archetype"]["value"] == "lease_heavy"
    assert snap.features["capital_structure.total_debt_reported"]["value"] == 100.0
    assert snap.features["capital_structure.total_debt_market"]["value"] == 200.0
    assert snap.features["capital_structure.total_debt"]["value"] == 200.0
    assert snap.features["capital_structure.net_debt_market"]["value"] == 185.0
    assert snap.features["capital_structure.interest_coverage"]["applicability_status"] == "secondary"
    assert snap.features["capital_structure.fixed_charge_coverage"]["applicability_status"] == "primary"
    assert snap.features["capital_structure.total_debt_market"]["component_breakdown"]["included_lease_liabilities"] == 100.0
    assert snap.provenance["market_metric_context"]["subsector"] == "Specialty Retail"
    assert snap.provenance["market_metric_context"]["methodology_registry_id"] == "consumer_industrials_metric_methodology_registry_v1"
    assert snap.provenance["market_metric_context"]["input_source_registry_id"] == "company_state_input_source_registry_v1"
    assert snap.features["capital_structure.total_debt_market"]["canonical_owner_id"] == "fitch_ratings"
    assert snap.features["capital_structure.total_debt_market"]["canonical_classification"] == "canonical_external"
    assert snap.features["capital_structure.total_debt_market"]["input_source_owner_name"] == "Fitch credit methodology"
    assert snap.features["capital_structure.total_debt_market"]["input_source_classification"] == "canonical_external"
    assert snap.features["capital_structure.total_debt_market"]["definition_requirement"] == "must_have_external_definition"
    assert snap.features["capital_structure.total_debt_market"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert snap.features["liquidity.available_for_actions_market"]["canonical_classification"] == "internal_only"
    assert snap.features["liquidity.available_for_actions_market"]["market_layer_status"] == "rename"
    lineage = snap.provenance["feature_lineage"]["capital_structure.total_debt_market"]["metric_context"]
    assert lineage["metric_policy_id"] == "market_metric_policy_v1"
    assert lineage["methodology_registry_id"] == "consumer_industrials_metric_methodology_registry_v1"
    assert lineage["canonical_owner_id"] == "fitch_ratings"
    assert lineage["canonical_classification"] == "canonical_external"
    assert lineage["input_source_registry_id"] == "company_state_input_source_registry_v1"
    assert lineage["input_source_owner_name"] == "Fitch credit methodology"
    assert lineage["definition_requirement"] == "must_have_external_definition"
    assert lineage["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert lineage["view_type"] == "market"
    assert lineage["archetype"] == "lease_heavy"


def test_market_metric_engine_suppresses_unsupported_financial_leverage(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    entity_path = tmp_path / "entity.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "BANK", "financial.cash", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "BANK", "financial.total_debt", 500.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "BANK", "financial.ebitda", 50.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [_entity_row("BANK", sector="Financials", subsector="Regional Banks", sic="6021")],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        entity_table_path=entity_path,
    )
    snap = builder.build("BANK", "2026-02-28")

    assert snap.features["taxonomy.archetype"]["value"] == "financial_institution"
    assert snap.features["capital_structure.net_leverage"]["value"] is None
    assert snap.features["capital_structure.net_leverage"]["missing_reason"] == "unsupported_for_archetype"
    assert snap.features["capital_structure.net_leverage"]["support_mode"] == "unsupported"
    assert snap.features["capital_structure.net_leverage"]["applicability_status"] == "unsupported"


def test_market_metric_engine_resolves_consumer_staples_policy(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    entity_path = tmp_path / "entity.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "FOOD", "financial.cash", 40.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("restricted", "FOOD", "financial.restricted_cash", 5.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "FOOD", "financial.total_debt", 300.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("pension", "FOOD", "financial.unfunded_pension", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "FOOD", "financial.ebitda", 60.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("revenue", "FOOD", "financial.revenue", 2_000.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("fcf", "FOOD", "financial.free_cash_flow", -24.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [_entity_row("FOOD", sector="Consumer Staples", subsector="Packaged Foods", sic="2090")],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        entity_table_path=entity_path,
    )
    snap = builder.build("FOOD", "2026-02-28")

    assert snap.features["taxonomy.archetype"]["value"] == "consumer_branded_staples"
    assert snap.features["capital_structure.total_debt_market"]["value"] == 300.0
    assert "pension_excluded_from_debt" in snap.features["capital_structure.total_debt_market"]["quality_flags"]
    assert snap.features["liquidity.available_for_actions_market"]["value"] == 35.0
    assert snap.features["liquidity.available_for_actions_market"]["component_breakdown"]["minimum_cash_policy_proxy"] == 40.0


def test_liquidity_structured_support_sums_restricted_cash_components(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("restricted_current", "ABC", "financial.restricted_cash_current", 6.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("restricted_noncurrent", "ABC", "financial.cash_restricted_noncurrent", 4.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-02-28")

    assert snap.features["liquidity.restricted_cash"]["value"] == 10.0
    assert snap.features["liquidity.usable_cash_market"]["value"] == 90.0
    assert snap.features["liquidity.usable_cash_market"]["component_breakdown"]["restricted_cash"] == 10.0


def test_liquidity_structured_support_derives_marketable_securities_from_combined_balance(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("cash_and_investments", "ABC", "financial.cash_and_short_term_investments", 140.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-02-28")

    assert snap.features["liquidity.marketable_securities"]["value"] == 40.0
    assert snap.features["liquidity.liquidity_total"]["value"] == 140.0
    assert snap.features["liquidity.usable_cash_market"]["component_breakdown"]["marketable_securities"] == 40.0


def test_liquidity_structured_support_uses_reference_cash_and_short_term_investments(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    taxonomy_reference_path = tmp_path / "taxonomy_reference.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        taxonomy_reference_path,
        [
            {
                "Instrument": "ABC.N",
                "Company Common Name": "ABC Corp",
                "Cash and Short Term Investments": 165.0,
                "GICS Sector Name": "Industrials",
                "GICS Industry Name": "Industrial Conglomerates",
            }
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        taxonomy_reference_path=taxonomy_reference_path,
    )
    snap = builder.build("ABC", "2026-02-28")

    assert snap.features["liquidity.marketable_securities"]["value"] == 65.0
    assert snap.features["liquidity.marketable_securities"]["fallback_used"] == "reference_cash_and_short_term_investments"
    assert "reference_cash_and_short_term_investments_fallback" in (
        snap.features["liquidity.marketable_securities"]["quality_flags"] or []
    )
    assert snap.features["liquidity.usable_cash_market"]["component_breakdown"]["marketable_securities"] == 65.0


def test_liquidity_structured_support_resolves_revolver_capacity_patterns(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 50.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("revolver", "ABC", "financial.unused_revolving_credit_capacity", 75.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-02-28")

    assert snap.features["liquidity.revolver_undrawn"]["value"] == 75.0
    assert snap.features["liquidity.liquidity_total"]["value"] == 125.0


def test_liquidity_pattern_matching_uses_fact_id_for_restricted_cash(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            {
                **_facts_row("us-gaap_CashAndCashEquivalentsAtCarryingValue", "ABC", "financial.cash", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            },
            {
                **_facts_row(
                    "us-gaap_RestrictedCashAndCashEquivalentsAtCarryingValue",
                    "ABC",
                    "financial.other_balance_sheet_item",
                    12.0,
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                ),
            },
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-02-28")

    assert snap.features["liquidity.restricted_cash"]["value"] == 12.0
    assert snap.features["liquidity.usable_cash_market"]["component_breakdown"]["restricted_cash"] == 12.0


def test_liquidity_pattern_matching_uses_fact_id_for_revolver_undrawn(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 50.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            {
                **_facts_row(
                    "custom_UnusedCommitmentUnderRevolvingCreditFacility",
                    "ABC",
                    "financial.other_liquidity_item",
                    80.0,
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                ),
            },
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-02-28")

    assert snap.features["liquidity.revolver_undrawn"]["value"] == 80.0
    assert snap.features["liquidity.liquidity_total"]["value"] == 130.0


def test_market_metric_engine_resolves_transport_logistics_policy(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    entity_path = tmp_path / "entity.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "TRNS", "financial.cash", 50.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "TRNS", "financial.total_debt", 500.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("lease_current", "TRNS", "financial.lease_liability_current", 40.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("lease_long", "TRNS", "financial.lease_liability_noncurrent", 60.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "TRNS", "financial.ebitda", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebit", "TRNS", "financial.ebit", 70.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("interest", "TRNS", "financial.interest_expense", 20.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [_entity_row("TRNS", sector="Industrials", subsector="Air Freight & Logistics", sic="4213")],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        entity_table_path=entity_path,
    )
    snap = builder.build("TRNS", "2026-02-28")

    assert snap.features["taxonomy.archetype"]["value"] == "transport_logistics"
    assert snap.features["capital_structure.total_debt_market"]["value"] == 600.0
    assert snap.features["capital_structure.fixed_charge_coverage"]["applicability_status"] == "primary"
    assert snap.features["capital_structure.interest_coverage"]["applicability_status"] == "secondary"


def test_market_metric_engine_resolves_aerospace_defense_policy(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    entity_path = tmp_path / "entity.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "AERO", "financial.cash", 80.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "AERO", "financial.total_debt", 400.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("pension", "AERO", "financial.unfunded_pension", 120.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "AERO", "financial.ebitda", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [_entity_row("AERO", sector="Industrials", subsector="Aerospace & Defense", sic="3721")],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        entity_table_path=entity_path,
    )
    snap = builder.build("AERO", "2026-02-28")

    assert snap.features["taxonomy.archetype"]["value"] == "aerospace_defense"
    assert snap.features["capital_structure.total_debt_market"]["value"] == 400.0
    assert "pension_excluded_from_debt" in snap.features["capital_structure.total_debt_market"]["quality_flags"]


def test_market_metric_engine_uses_taxonomy_reference_when_entity_table_is_thin(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    entity_path = tmp_path / "entity.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"
    taxonomy_reference_path = tmp_path / "taxonomy_reference.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "NPO_CIK", "financial.cash", 60.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "NPO_CIK", "financial.total_debt", 300.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "NPO_CIK", "financial.ebitda", 75.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [{"entity_id": "NPO_CIK", "legal_name": "Enpro Inc"}],
    )
    _write_parquet(
        ident_path,
        [
            {"entity_id": "NPO_CIK", "identifier_type": "ticker", "identifier_value": "NPO"},
            {"entity_id": "NPO_CIK", "identifier_type": "cik", "identifier_value": "NPO_CIK"},
        ],
    )
    _write_parquet(
        taxonomy_reference_path,
        [
            {
                "Instrument": "NPO.N",
                "Company Common Name": "Enpro Inc",
                "GICS Sector Name": "Industrials",
                "GICS Industry Name": "Machinery",
            }
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        entity_table_path=entity_path,
        entity_identifier_path=ident_path,
        taxonomy_reference_path=taxonomy_reference_path,
    )
    snap = builder.build("NPO_CIK", "2026-02-28")

    assert snap.features["taxonomy.archetype"]["value"] == "machinery_capital_goods"
    assert snap.features["taxonomy.sector"]["value"] == "Industrials"
    assert snap.features["taxonomy.subsector"]["value"] == "Machinery"


def test_market_metric_engine_resolves_automotive_oem_policy(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    entity_path = tmp_path / "entity.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "AUTO", "financial.cash", 120.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "AUTO", "financial.total_debt", 800.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("pension", "AUTO", "financial.unfunded_pension", 200.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "AUTO", "financial.ebitda", 160.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [_entity_row("AUTO", sector="Consumer Discretionary", subsector="Automobiles", sic="3711")],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        entity_table_path=entity_path,
    )
    snap = builder.build("AUTO", "2026-02-28")

    assert snap.features["taxonomy.archetype"]["value"] == "automotive_oem"
    assert snap.features["capital_structure.total_debt_market"]["value"] == 800.0
    assert "pension_excluded_from_debt" in snap.features["capital_structure.total_debt_market"]["quality_flags"]


def test_arithmetic_identity_for_market_and_net_debt(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"

    facts_rows = [
        _facts_row("cash", "ABC", "financial.cash", 50.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        _facts_row("debt", "ABC", "financial.total_debt", 200.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        _facts_row("ebitda", "ABC", "financial.ebitda", 10.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        _facts_row("shares", "ABC", "financial.shares_out", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
    ]
    _write_parquet(facts_path, facts_rows)

    ts_rows = [
        {
            "entity_id": "ABC",
            "series_type": "price",
            "trade_date": "2026-02-20T00:00:00Z",
            "available_time": "2026-02-20T00:00:00Z",
            "ingestion_time": "2026-02-20T00:00:00Z",
            "close": 10.0,
        }
    ]
    _write_parquet(ts_path, ts_rows)

    builder = _base_builder(tmp_path, facts_path=facts_path, timeseries_path=ts_path, skip_timeseries=False)
    snap = builder.build("ABC", "2026-02-28")

    assert snap.features["capital_structure.net_debt"]["value"] == 150.0
    assert snap.features["market.market_cap"]["value"] == 1000.0
    assert snap.features["market.market_cap"]["component_breakdown"]["formula"] == "close_price * shares_outstanding"
    assert snap.features["market.market_cap"]["component_breakdown"]["shares_source"] == "shares_basic"
    assert snap.features["market.market_cap"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert "price_shares_fallback" in (snap.features["market.market_cap"]["quality_flags"] or [])
    assert snap.features["market.enterprise_value"]["value"] == 1150.0
    assert snap.features["liquidity.liquidity_total"]["value"] >= snap.features["liquidity.cash"]["value"]


def test_market_fcf_yield_prefers_operating_cash_flow_minus_capex(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"

    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 50.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "ABC", "financial.total_debt", 200.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("shares", "ABC", "financial.shares_out", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "ABC", "financial.ebitda", 200.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ocf", "ABC", "financial.operating_cash_flow", 120.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("capex", "ABC", "financial.capex", 20.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("fcf_provider", "ABC", "financial.free_cash_flow", 70.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        ts_path,
        [
            {
                "entity_id": "ABC",
                "series_type": "price",
                "trade_date": "2026-02-20T00:00:00Z",
                "available_time": "2026-02-20T00:00:00Z",
                "ingestion_time": "2026-02-20T00:00:00Z",
                "close": 10.0,
            }
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, timeseries_path=ts_path, skip_timeseries=False)
    snap = builder.build("ABC", "2026-02-28")

    assert snap.features["market.market_cap"]["value"] == 1000.0
    assert snap.features["market.fcf_yield"]["value"] == 0.1
    assert snap.features["market.fcf_yield"]["fallback_used"] is None
    assert snap.features["operating.fcf_conversion"]["value"] == 0.5
    assert snap.features["operating.fcf_conversion"]["fallback_used"] is None
    assert snap.features["operating.fcf_conversion"]["input_source_classification"] == "external_raw_plus_deterministic_formula"
    assert snap.features["operating.fcf_conversion"]["definition_requirement"] == "can_be_externally_anchored"
    assert snap.features["operating.fcf_conversion"]["component_breakdown"]["formula"] == "(operating_cash_flow - capex) / ebitda"


def test_operating_ebitda_margin_uses_matched_reporting_periods(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("rev_older", "ABC", "financial.revenue", 80.0, "2025-01-15T00:00:00Z", "2025-01-16T00:00:00Z", "2025-01-15T00:00:00Z"),
            _facts_row("ebitda_older", "ABC", "financial.ebitda", 12.0, "2025-01-15T00:00:00Z", "2025-01-16T00:00:00Z", "2025-01-15T00:00:00Z"),
            _facts_row("rev_latest", "ABC", "financial.revenue", 100.0, "2026-01-20T00:00:00Z", "2026-01-21T00:00:00Z", "2026-01-20T00:00:00Z"),
            _facts_row("ebitda_latest", "ABC", "financial.ebitda", 20.0, "2026-01-20T00:00:00Z", "2026-01-21T00:00:00Z", "2026-01-20T00:00:00Z"),
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-02-28")

    feature = snap.features["operating.ebitda_margin_ttm"]
    assert feature["value"] == 0.2
    assert feature["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert feature["component_breakdown"]["period_match_type"] == "exact_period_match"
    assert feature["component_breakdown"]["formula"] == "ebitda / revenue"
    assert feature["quality_flags"] is None


def test_operating_metrics_use_reference_fallbacks_in_historical_backfill_mode(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    taxonomy_reference_path = tmp_path / "taxonomy_reference.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("rev_latest", "ABC", "financial.revenue", 100.0, "2026-01-20T00:00:00Z", "2026-01-21T00:00:00Z", "2026-01-20T00:00:00Z"),
        ],
    )
    _write_parquet(
        taxonomy_reference_path,
        [
            {
                "Instrument": "ABC.N",
                "Revenue": 120.0,
                "EBITDA": 24.0,
                "Free Cash Flow": 12.0,
            }
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        taxonomy_reference_path=taxonomy_reference_path,
        skip_timeseries=True,
        historical_backfill_mode=True,
    )
    snap = builder.build("ABC", "2026-02-28")

    margin = snap.features["operating.ebitda_margin_ttm"]
    fcf_conversion = snap.features["operating.fcf_conversion"]

    assert margin["value"] == 0.2
    assert margin["fallback_used"] == "reference_ebitda_margin_fallback"
    assert margin["component_breakdown"]["reference_revenue"] == 120.0
    assert margin["component_breakdown"]["reference_ebitda"] == 24.0
    assert margin["component_breakdown"]["period_match_type"] == "reference_ttm_fallback"
    assert "reference_ebitda_margin_fallback" in (margin["quality_flags"] or [])

    assert fcf_conversion["value"] == 0.5
    assert fcf_conversion["fallback_used"] == "reference_fcf_conversion_fallback"
    assert fcf_conversion["component_breakdown"]["reference_free_cash_flow"] == 12.0
    assert fcf_conversion["component_breakdown"]["reference_ebitda"] == 24.0
    assert "reference_fcf_conversion_fallback" in (fcf_conversion["quality_flags"] or [])


def test_stability_without_new_data(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    rows = [
        _facts_row("cash", "ABC", "financial.cash", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        _facts_row("debt", "ABC", "financial.total_debt", 500.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        _facts_row("ebitda", "ABC", "financial.ebitda", 50.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
    ]
    _write_parquet(facts_path, rows)

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap_t = builder.build("ABC", "2026-02-28")
    snap_t1 = builder.build("ABC", "2026-03-01")

    for key in snap_t.features:
        left = snap_t.features[key]
        right = snap_t1.features[key]
        assert left["value"] == right["value"], key
        assert left["missing_reason"] == right["missing_reason"], key
        assert left["fallback_used"] == right["fallback_used"], key


def test_capital_structure_maturity_and_rating_from_events(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    events_path = tmp_path / "events.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 200.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "ABC", "financial.total_debt", 1000.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "ABC", "financial.ebitda", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebit", "ABC", "financial.ebit", 80.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ie", "ABC", "financial.interest_expense", 20.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        events_path,
        [
            {
                "event_id": "evt_debt_1",
                "company_id": "ABC",
                "event_type": "debt_issuance",
                "event_subtype": None,
                "announced_at": "2026-01-01T00:00:00Z",
                "effective_at": "2026-01-01T00:00:00Z",
                "created_at": "2026-01-01T00:00:00Z",
                "source_type": "fisd",
                "params": {"maturity_date": "2026-10-01T00:00:00Z", "offering_amt_k": 100.0},
            },
            {
                "event_id": "evt_debt_2",
                "company_id": "ABC",
                "event_type": "debt_issuance",
                "event_subtype": None,
                "announced_at": "2026-01-01T00:00:00Z",
                "effective_at": "2026-01-01T00:00:00Z",
                "created_at": "2026-01-01T00:00:00Z",
                "source_type": "fisd",
                "params": {"maturity_date": "2027-08-01T00:00:00Z", "offering_amt_k": 300.0},
            },
            {
                "event_id": "evt_rating_1",
                "company_id": "ABC",
                "event_type": "rating_action",
                "event_subtype": "FCLONG",
                "announced_at": "2026-02-05T00:00:00Z",
                "effective_at": "2026-02-05T00:00:00Z",
                "created_at": "2026-02-05T00:00:00Z",
                "source_type": "ciq_ratings",
                "params": {"current_rating_symbol": "BBB-", "outlook": "stable", "creditwatch": "N"},
            },
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        events_path=events_path,
        skip_events=False,
    )
    snap = builder.build("ABC", "2026-02-28")

    assert snap.features["capital_structure.debt_due_0_12m"]["value"] == 100.0
    assert snap.features["capital_structure.debt_due_12_24m"]["value"] == 300.0
    assert snap.features["capital_structure.debt_schedule_total"]["value"] == 400.0
    assert snap.features["capital_structure.debt_schedule_vs_total_debt"]["value"] == 0.4
    assert snap.features["capital_structure.debt_schedule_inconsistency_flag"]["value"] == 0.0
    assert snap.features["capital_structure.maturity_wall_ratio_24m"]["value"] == 0.4
    assert snap.features["capital_structure.refi_pressure_flag"]["value"] == 1.0
    rating_state = snap.features["capital_structure.rating_state"]["value"]
    assert isinstance(rating_state, dict)
    assert rating_state["rating"] == "BBB-"
    assert rating_state["outlook"] == "stable"


def test_build_flags_inconsistent_debt_schedule(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    events_path = tmp_path / "events.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 150.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "ABC", "financial.total_debt", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "ABC", "financial.ebitda", 50.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebit", "ABC", "financial.ebit", 40.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ie", "ABC", "financial.interest_expense", 5.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        events_path,
        [
            {
                "event_id": "evt_debt_1",
                "company_id": "ABC",
                "event_type": "debt_issuance",
                "event_subtype": None,
                "announced_at": "2026-01-01T00:00:00Z",
                "effective_at": "2026-01-01T00:00:00Z",
                "created_at": "2026-01-01T00:00:00Z",
                "source_type": "fisd",
                "params": {"maturity_date": "2026-10-01T00:00:00Z", "offering_amt_k": 100.0},
            },
            {
                "event_id": "evt_debt_2",
                "company_id": "ABC",
                "event_type": "debt_issuance",
                "event_subtype": None,
                "announced_at": "2026-01-01T00:00:00Z",
                "effective_at": "2026-01-01T00:00:00Z",
                "created_at": "2026-01-01T00:00:00Z",
                "source_type": "fisd",
                "params": {"maturity_date": "2030-01-01T00:00:00Z", "offering_amt_k": 200.0},
            },
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        events_path=events_path,
        skip_events=False,
    )
    snap = builder.build("ABC", "2026-02-28")

    assert snap.features["capital_structure.debt_schedule_total"]["value"] == 300.0
    assert snap.features["capital_structure.debt_schedule_vs_total_debt"]["value"] == 3.0
    assert snap.features["capital_structure.debt_schedule_inconsistency_flag"]["value"] == 1.0
    assert snap.features["capital_structure.debt_due_0_12m"]["value"] is None
    assert snap.features["capital_structure.debt_due_60m_plus"]["value"] is None
    assert snap.features["capital_structure.debt_due_0_12m"]["missing_reason"] == "anomalous_schedule"
    assert snap.features["capital_structure.maturity_wall_ratio_24m"]["value"] is None
    assert snap.features["capital_structure.maturity_wall_ratio_24m"]["missing_reason"] == "anomalous_schedule"
    assert snap.features["capital_structure.refi_pressure_flag"]["value"] is None


def test_build_uses_note_extracted_maturity_schedule_when_events_absent(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 150.0, "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z"),
            _facts_row("debt", "ABC", "financial.total_debt", 1000.0, "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z"),
            _note_fact_row("sec:abc:10k:2024", "ABC", "financial.debt_maturity_bucket", 200.0, "2025", "2024-12-20T00:00:00Z"),
            _note_fact_row("sec:abc:10k:2024", "ABC", "financial.debt_maturity_bucket", 350.0, "2026", "2024-12-20T00:00:00Z"),
            _note_fact_row("sec:abc:10k:2024", "ABC", "financial.debt_maturity_bucket", 1200.0, "Thereafter", "2024-12-20T00:00:00Z"),
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        skip_events=True,
        historical_backfill_mode=True,
    )
    snap = builder.build("ABC", "2024-12-31")

    assert snap.features["capital_structure.debt_due_0_12m"]["value"] == 200.0
    assert snap.features["capital_structure.debt_due_12_24m"]["value"] == 350.0
    assert snap.features["capital_structure.debt_due_60m_plus"]["value"] == 1200.0
    assert snap.features["capital_structure.debt_schedule_total"]["value"] == 1750.0
    assert snap.features["capital_structure.maturity_wall_ratio_24m"]["value"] == 0.55
    assert snap.features["capital_structure.maturity_wall_ratio_24m"]["fallback_used"] == "note_pattern_extract"
    assert "maturity_schedule_note_extract" in (snap.features["capital_structure.maturity_wall_ratio_24m"]["quality_flags"] or [])


def test_build_uses_extra_aliases_for_event_matching(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    events_path = tmp_path / "events.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "0001", "financial.cash", 200.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "0001", "financial.total_debt", 1000.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "0001", "financial.ebitda", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebit", "0001", "financial.ebit", 80.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ie", "0001", "financial.interest_expense", 20.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        events_path,
        [
            {
                "event_id": "evt_rating_alias",
                "company_id": "SRC123",
                "event_type": "rating_action",
                "event_subtype": "FCLONG",
                "announced_at": "2026-02-05T00:00:00Z",
                "effective_at": "2026-02-05T00:00:00Z",
                "created_at": "2026-02-05T00:00:00Z",
                "source_type": "ciq_ratings",
                "params": {"current_rating_symbol": "BBB-", "outlook": "stable", "creditwatch": "N"},
            },
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        events_path=events_path,
        skip_events=False,
    )
    snap = builder.build("0001", "2026-02-28", extra_aliases=["SRC123"])
    rating_state = snap.features["capital_structure.rating_state"]["value"]
    assert isinstance(rating_state, dict)
    assert rating_state["rating"] == "BBB-"


def test_build_uses_extra_aliases_for_fact_matching_even_when_primary_id_has_other_facts(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("div_hist", "0001", "financial.common_dividends_cash", 0.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("cash_alias", "SRC123", "financial.cash", 250.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt_alias", "SRC123", "financial.total_debt", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        skip_events=True,
    )
    snap = builder.build("0001", "2026-02-28", extra_aliases=["SRC123"])

    assert snap.features["liquidity.cash"]["value"] == 250.0
    assert snap.features["capital_structure.total_debt"]["value"] == 100.0
    assert snap.features["capital_structure.net_debt"]["value"] == -150.0
    assert snap.features["liquidity.available_for_actions"]["value"] == 250.0


def test_resolve_entity_aliases_can_canonicalize_from_extra_alias_when_primary_id_is_unknown(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"
    _write_parquet(facts_path, [])
    _write_parquet(
        ident_path,
        [
            {"entity_id": "0001932393", "identifier_value": "GEHC", "identifier_type": "ticker"},
            {"entity_id": "0001932393", "identifier_value": "1932393", "identifier_type": "cik"},
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        entity_identifier_path=ident_path,
        skip_timeseries=True,
        skip_events=True,
    )

    canonical, aliases = builder._resolve_entity_aliases("041818", extra_aliases=["GEHC"])

    assert canonical == "0001932393"
    assert "GEHC" in aliases
    assert "0001932393" in aliases
    assert "1932393" in aliases


def test_dividend_payer_flag_from_recent_dividend_events(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    events_path = tmp_path / "events.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 200.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        events_path,
        [
            {
                "event_id": "evt_div_1",
                "company_id": "ABC",
                "event_type": "dividend_regular",
                "event_subtype": "regular",
                "announced_at": "2025-12-01T00:00:00Z",
                "effective_at": "2025-12-01T00:00:00Z",
                "created_at": "2025-12-01T00:00:00Z",
                "source_type": "event_store",
            },
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        events_path=events_path,
        skip_events=False,
    )
    snap = builder.build("ABC", "2026-02-28")

    feature = snap.features["capital_return.dividend_payer_flag"]
    assert feature["value"] is True
    assert feature["methodology_execution_decision"] == "keep_externally_anchored_house_formula"
    assert feature["input_layer_bucket"] == "secondary_externally_anchored"
    assert feature["strict_market_defined"] is False
    assert feature["input_source_classification"] == "external_raw_plus_deterministic_formula"
    assert feature["fallback_used"] is None
    assert feature["component_breakdown"]["recurring_event_count_450d"] == 1
    last_feature = snap.features["capital_return.last_dividend_event_type"]
    assert last_feature["value"] == "dividend_regular"
    assert last_feature["methodology_execution_decision"] == "keep_externally_anchored_house_formula"
    assert last_feature["input_layer_bucket"] == "secondary_externally_anchored"
    assert last_feature["component_breakdown"]["formula"] == "latest_recurring_dividend_event_type"


def test_dividend_payer_flag_false_without_recurring_dividend_history(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    events_path = tmp_path / "events.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 200.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        events_path,
        [
            {
                "event_id": "evt_special_1",
                "company_id": "ABC",
                "event_type": "dividend_special",
                "event_subtype": "special",
                "announced_at": "2025-12-01T00:00:00Z",
                "effective_at": "2025-12-01T00:00:00Z",
                "created_at": "2025-12-01T00:00:00Z",
                "source_type": "event_store",
            },
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        events_path=events_path,
        skip_events=False,
    )
    snap = builder.build("ABC", "2026-02-28")

    feature = snap.features["capital_return.dividend_payer_flag"]
    assert feature["value"] is False
    assert feature["methodology_execution_decision"] == "keep_externally_anchored_house_formula"
    assert "no_recurring_dividend_events_in_history" in (feature["quality_flags"] or [])


def test_dividend_payer_flag_falls_back_to_dividend_facts(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            {
                **_facts_row(
                    "div_ps_1",
                    "ABC",
                    "financial.dividends_per_share_cash",
                    0.5,
                    "2025-08-15T00:00:00Z",
                    "2025-08-15T00:00:00Z",
                    "2025-08-15T00:00:00Z",
                ),
                "period_end": "2025-06-30T00:00:00Z",
            },
            {
                **_facts_row(
                    "div_ps_2",
                    "ABC",
                    "financial.dividends_per_share_cash",
                    0.5,
                    "2025-11-15T00:00:00Z",
                    "2025-11-15T00:00:00Z",
                    "2025-11-15T00:00:00Z",
                ),
                "period_end": "2025-09-30T00:00:00Z",
            },
        ],
    )
    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        skip_events=True,
    )
    snap = builder.build("ABC", "2026-02-28")

    feature = snap.features["capital_return.dividend_payer_flag"]
    assert feature["value"] is True
    assert "dividend_fact_fallback" in (feature["quality_flags"] or [])
    assert feature["component_breakdown"]["dividend_fact_fallback_recent_count_24m"] == 2

    last_feature = snap.features["capital_return.last_dividend_event_type"]
    assert last_feature["value"] == "dividend_regular"


def test_event_history_falls_back_to_corporate_actions_master(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    corp_actions_path = tmp_path / "corporate_actions_master.parquet"
    _write_parquet(
        facts_path,
        [
            {
                **_facts_row(
                    "cash",
                    "ABC",
                    "financial.cash",
                    200.0,
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                ),
                "period_end": "2025-12-31T00:00:00Z",
            },
        ],
    )
    _write_parquet(
        corp_actions_path,
        [
            {
                "ticker": "ABC",
                "action_type": "dividend_regular",
                "action_subtype": "regular",
                "action_date": "2025-12-10T00:00:00Z",
                "dclrdt": "2025-12-01T00:00:00Z",
                "paydt": "2025-12-20T00:00:00Z",
                "amount": 25.0,
                "source": "crsp",
            },
            {
                "ticker": "ABC",
                "action_type": "buyback",
                "action_subtype": "open_market",
                "action_date": "2025-11-15T00:00:00Z",
                "dclrdt": "2025-11-14T00:00:00Z",
                "amount": 150.0,
                "source": "compustat_prstkcy",
            },
        ],
    )
    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        skip_events=False,
        corporate_actions_path=corp_actions_path,
    )
    snap = builder.build("ABC", "2026-02-28")

    dividend_flag = snap.features["capital_return.dividend_payer_flag"]
    assert dividend_flag["value"] is True
    assert dividend_flag["missing_reason"] is None
    assert dividend_flag["fallback_used"] is None

    last_dividend = snap.features["capital_return.last_dividend_event_type"]
    assert last_dividend["value"] == "dividend_regular"

    recent_actions = snap.features["strategic.recent_actions_count_24m"]
    assert recent_actions["value"] == 1.0
    assert recent_actions["missing_reason"] is None

    last_action = snap.features["strategic.last_action_type"]
    assert last_action["value"] == "buyback"


def test_market_window_and_credit_spread_features(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 50.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "ABC", "financial.total_debt", 200.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "ABC", "financial.ebitda", 20.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("shares", "ABC", "financial.shares_out", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    ts_rows = []
    for i in range(120):
        ts_rows.append(
            {
                "entity_id": "ABC",
                "series_type": "price",
                "trade_date": f"2025-11-{(i % 28) + 1:02d}T00:00:00Z",
                "available_time": f"2025-11-{(i % 28) + 1:02d}T00:00:00Z",
                "ingestion_time": f"2025-11-{(i % 28) + 1:02d}T00:00:00Z",
                "close": 90.0 + i * 0.2,
            }
        )
    for i in range(60):
        ts_rows.append(
            {
                "entity_id": "ABC",
                "series_type": "price",
                "trade_date": f"2025-12-{(i % 28) + 1:02d}T00:00:00Z",
                "available_time": f"2025-12-{(i % 28) + 1:02d}T00:00:00Z",
                "ingestion_time": f"2025-12-{(i % 28) + 1:02d}T00:00:00Z",
                "series_id": "issuer_oas",
                "value": 120.0 + i,
            }
        )
    _write_parquet(ts_path, ts_rows)
    builder = _base_builder(tmp_path, facts_path=facts_path, timeseries_path=ts_path, skip_timeseries=False)
    snap = builder.build("ABC", "2026-02-28")
    assert snap.features["market.credit_spread_level"]["value"] is not None
    assert snap.features["market.equity_window_proxy"]["value"] is not None
    assert snap.features["market.credit_window_proxy"]["value"] is not None
    assert snap.features["market.credit_spread_level"]["input_layer_bucket"] == "strict_market_defined"
    assert snap.features["market.credit_spread_level"]["strict_market_defined"] is True
    assert snap.features["market.credit_spread_level"]["support_mode"] == "exact"
    assert snap.features["market.credit_window_proxy"]["input_layer_bucket"] == "internal_inference"
    assert snap.features["market.credit_window_proxy"]["strict_market_defined"] is False
    assert snap.features["market.credit_window_proxy"]["support_mode"] == "inferred"
    views = snap.provenance["input_layer_views"]
    assert views["strict_market_defined"]["registry_metric_count"] == 41
    assert views["secondary_externally_anchored"]["registry_metric_count"] == 6
    assert views["internal_inference"]["registry_metric_count"] == 33
    assert "market.credit_spread_level" in views["strict_market_defined"]["snapshot_input_metric_ids_present"]
    assert "market.credit_window_proxy" in views["internal_inference"]["snapshot_input_metric_ids_present"]


def test_market_volatility_and_drawdown_have_explicit_formula_metadata(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 10.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )

    dates = pd.date_range("2025-10-01", periods=100, freq="D", tz="UTC")
    prices = pd.Series([100.0 + i for i in range(100)])
    returns = prices.pct_change().dropna()
    expected_vol_30 = float(returns.tail(30).std(ddof=0) * (252 ** 0.5))
    expected_vol_90 = float(returns.tail(90).std(ddof=0) * (252 ** 0.5))
    expected_dd_90 = float((prices.tail(90).min() / prices.tail(90).max()) - 1.0)
    _write_parquet(
        ts_path,
        [
            {
                "entity_id": "ABC",
                "series_type": "price",
                "trade_date": date.isoformat(),
                "available_time": date.isoformat(),
                "ingestion_time": date.isoformat(),
                "close": float(price),
            }
            for date, price in zip(dates, prices)
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, timeseries_path=ts_path, skip_timeseries=False)
    snap = builder.build("ABC", "2026-02-28")

    vol_30 = snap.features["market.volatility_30d"]
    vol_90 = snap.features["market.volatility_90d"]
    dd_90 = snap.features["market.drawdown_90d"]

    assert round(vol_30["value"], 10) == round(expected_vol_30, 10)
    assert round(vol_90["value"], 10) == round(expected_vol_90, 10)
    assert round(dd_90["value"], 10) == round(expected_dd_90, 10)
    assert vol_30["support_mode"] == "exact"
    assert vol_90["support_mode"] == "exact"
    assert dd_90["support_mode"] == "exact"
    assert vol_30["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert vol_90["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert dd_90["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert vol_30["input_layer_bucket"] == "strict_market_defined"
    assert vol_30["strict_market_defined"] is True
    assert vol_30["component_breakdown"]["formula"] == "stddev(daily_returns_30d) * sqrt(252)"
    assert vol_90["component_breakdown"]["formula"] == "stddev(daily_returns_90d) * sqrt(252)"
    assert dd_90["component_breakdown"]["formula"] == "min(price_window_90d) / max(price_window_90d) - 1"


def test_market_volatility_uses_single_equity_price_series_when_multiple_price_candidates_exist(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 10.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )

    dates = pd.date_range("2025-10-01", periods=100, freq="D", tz="UTC")
    equity_prices = pd.Series([100.0 + i for i in range(100)])
    distressed_bond_prices = pd.Series([100.0 if i % 2 == 0 else 10.0 for i in range(100)])
    expected_returns = equity_prices.pct_change().dropna()
    expected_vol_30 = float(expected_returns.tail(30).std(ddof=0) * (252 ** 0.5))
    expected_vol_90 = float(expected_returns.tail(90).std(ddof=0) * (252 ** 0.5))
    expected_dd_90 = float((equity_prices.tail(90).min() / equity_prices.tail(90).max()) - 1.0)

    rows = []
    for date, eq_price, bond_price in zip(dates, equity_prices, distressed_bond_prices):
        rows.append(
            {
                "entity_id": "ABC",
                "series_type": "price",
                "security_id": "EQ1",
                "instrument_type": "equity",
                "trade_date": date.isoformat(),
                "available_time": (date + pd.Timedelta(hours=20)).isoformat(),
                "close": float(eq_price),
                "adjusted_close": float(eq_price),
            }
        )
        rows.append(
            {
                "entity_id": "ABC",
                "series_type": "price",
                "security_id": "BOND1",
                "instrument_type": "bond",
                "trade_date": date.isoformat(),
                "available_time": (date + pd.Timedelta(hours=21)).isoformat(),
                "close": float(bond_price),
                "adjusted_close": float(bond_price),
            }
        )
    _write_parquet(ts_path, rows)

    builder = _base_builder(tmp_path, facts_path=facts_path, timeseries_path=ts_path, skip_timeseries=False)
    snap = builder.build("ABC", "2026-02-28")

    vol_30 = snap.features["market.volatility_30d"]
    vol_90 = snap.features["market.volatility_90d"]
    dd_90 = snap.features["market.drawdown_90d"]

    assert round(vol_30["value"], 10) == round(expected_vol_30, 10)
    assert round(vol_90["value"], 10) == round(expected_vol_90, 10)
    assert round(dd_90["value"], 10) == round(expected_dd_90, 10)
    assert vol_30["component_breakdown"]["selected_price_series"]["group_field"] == "security_id"
    assert vol_30["component_breakdown"]["selected_price_series"]["group_value"] == "EQ1"
    assert "multiple_price_series_candidates" in (vol_30["quality_flags"] or [])


def test_market_volatility_and_drawdown_require_actual_30d_90d_history_not_just_30_90_observations(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 10.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )

    dates = pd.date_range("2018-01-31", periods=120, freq="ME", tz="UTC")
    prices = pd.Series([100.0 + (i * 2.0) for i in range(120)])
    _write_parquet(
        ts_path,
        [
            {
                "entity_id": "ABC",
                "series_type": "price",
                "security_id": "EQ1",
                "trade_date": date.isoformat(),
                "available_time": date.isoformat(),
                "close": float(price),
                "adjusted_close": float(price),
            }
            for date, price in zip(dates, prices)
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, timeseries_path=ts_path, skip_timeseries=False)
    snap = builder.build("ABC", "2028-01-31")

    assert snap.features["market.volatility_30d"]["value"] is None
    assert snap.features["market.volatility_90d"]["value"] is None
    assert snap.features["market.drawdown_90d"]["value"] is None
    assert snap.features["market.volatility_30d"]["missing_reason"] == "unavailable"
    assert "insufficient_return_history" in (snap.features["market.volatility_30d"]["quality_flags"] or [])
    assert "insufficient_price_history" in (snap.features["market.drawdown_90d"]["quality_flags"] or [])


def test_market_volatility_and_drawdown_fail_honestly_for_monthly_only_history_in_historical_backfill_mode(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 10.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )

    dates = pd.date_range("2025-01-31", periods=24, freq="ME", tz="UTC")
    prices = pd.Series([100.0 + (i * 3.0) for i in range(24)], index=dates)
    _write_parquet(
        ts_path,
        [
            {
                "entity_id": "ABC",
                "series_type": "price",
                "security_id": "EQ1",
                "trade_date": date.isoformat(),
                "available_time": date.isoformat(),
                "close": float(price),
                "adjusted_close": float(price),
            }
            for date, price in zip(dates, prices)
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        timeseries_path=ts_path,
        skip_timeseries=False,
        historical_backfill_mode=True,
    )
    snap = builder.build("ABC", "2026-12-31")

    vol_30 = snap.features["market.volatility_30d"]
    vol_90 = snap.features["market.volatility_90d"]
    dd_90 = snap.features["market.drawdown_90d"]

    assert vol_30["value"] is None
    assert vol_90["value"] is None
    assert dd_90["value"] is None
    assert vol_30["support_mode"] == "unsupported"
    assert vol_90["support_mode"] == "unsupported"
    assert dd_90["support_mode"] == "unsupported"
    assert vol_30["fallback_used"] is None
    assert vol_90["fallback_used"] is None
    assert dd_90["fallback_used"] is None
    assert "low_frequency_price_history" in (vol_30["quality_flags"] or [])
    assert "low_frequency_price_history" in (vol_90["quality_flags"] or [])
    assert "low_frequency_price_history" in (dd_90["quality_flags"] or [])


def test_market_pe_metrics_are_strict_and_company_specific(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"
    entity_path = tmp_path / "entity.parquet"

    _write_parquet(
        facts_path,
        [
            _facts_row("shares", "ABC", "financial.shares_diluted", 100.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [
            {
                **_entity_row("ABC", sector="Consumer Staples", subsector="Food Retail"),
                "pe_ratio": 21.0,
            },
            {
                **_entity_row("P1", sector="Consumer Staples", subsector="Food Retail"),
                "pe_ratio": 10.0,
            },
            {
                **_entity_row("P2", sector="Consumer Staples", subsector="Food Retail"),
                "pe_ratio": 12.0,
            },
            {
                **_entity_row("P3", sector="Consumer Staples", subsector="Food Retail"),
                "pe_ratio": 15.0,
            },
            {
                **_entity_row("P4", sector="Consumer Staples", subsector="Food Retail"),
                "pe_ratio": 18.0,
            },
            {
                **_entity_row("P5", sector="Consumer Staples", subsector="Food Retail"),
                "pe_ratio": 24.0,
            },
            {
                **_entity_row("P6", sector="Consumer Staples", subsector="Food Retail"),
                "pe_ratio": 27.0,
            },
            {
                **_entity_row("P7", sector="Consumer Staples", subsector="Food Retail"),
                "pe_ratio": 30.0,
            },
        ],
    )

    pe_dates = pd.date_range("2025-03-31", periods=12, freq="ME", tz="UTC")
    pe_values = [10.0 + idx for idx in range(12)]
    _write_parquet(
        ts_path,
        [
            {
                "entity_id": "ABC",
                "series_type": "price",
                "trade_date": date.isoformat(),
                "available_time": date.isoformat(),
                "ingestion_time": date.isoformat(),
                "series_id": "pe_ratio",
                "value": float(value),
            }
            for date, value in zip(pe_dates, pe_values)
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        timeseries_path=ts_path,
        skip_timeseries=False,
        entity_table_path=entity_path,
    )
    snap = builder.build("ABC", "2026-02-28")

    pe_ratio = snap.features["market.pe_ratio"]
    pe_peer = snap.features["market.pe_percentile_peers"]
    pe_hist = snap.features["market.pe_percentile_history"]

    assert pe_ratio["value"] == 21.0
    assert pe_ratio["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert pe_ratio["input_layer_bucket"] == "strict_market_defined"
    assert pe_ratio["strict_market_defined"] is True
    assert pe_ratio["component_breakdown"]["formula"] == "provider_pe_series"

    assert pe_peer["value"] == 62.5
    assert pe_peer["input_layer_bucket"] == "strict_market_defined"
    assert pe_peer["component_breakdown"]["peer_group_col"] == "gics_sub_industry"
    assert pe_peer["component_breakdown"]["peer_row_count"] == 8

    assert pe_hist["value"] == 100.0
    assert pe_hist["input_layer_bucket"] == "strict_market_defined"
    assert pe_hist["component_breakdown"]["observation_count"] == 12


def test_market_pe_ratio_uses_ttm_net_income_bridge_for_q1(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"

    _write_parquet(
        facts_path,
        [
            {
                **_facts_row("ni_annual", "ABC", "financial.net_income", 800.0, "2025-11-15T00:00:00Z", "2025-11-16T00:00:00Z", "2025-11-15T00:00:00Z"),
                "effective_at": "2025-09-30T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2025-09-30 00:00:00; fiscal_year=2025; fiscal_quarter=",
            },
            {
                **_facts_row("ni_q1_prior", "ABC", "financial.net_income", 180.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
                "effective_at": "2024-12-31T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2024-12-31 00:00:00; fiscal_year=2026; fiscal_quarter=1",
            },
            {
                **_facts_row("ni_q1_current", "ABC", "financial.net_income", 220.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
                "effective_at": "2025-12-31T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2025-12-31 00:00:00; fiscal_year=2026; fiscal_quarter=1",
            },
            {
                **_facts_row("shares", "ABC", "financial.shares_out", 100.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
                "effective_at": "2025-12-31T00:00:00Z",
            },
        ],
    )
    _write_parquet(
        ts_path,
        [
            {
                "entity_id": "ABC",
                "series_type": "price",
                "trade_date": "2025-12-31T00:00:00Z",
                "available_time": "2025-12-31T00:00:00Z",
                "ingestion_time": "2026-01-02T00:00:00Z",
                "adjusted_close": 50.0,
            }
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, timeseries_path=ts_path, skip_timeseries=False)
    snap = builder.build("ABC", "2026-02-28")

    pe_ratio = snap.features["market.pe_ratio"]
    assert round(pe_ratio["value"], 6) == round(5000.0 / (800.0 + 220.0 - 180.0), 6)
    assert pe_ratio["fallback_used"] == "derived_from_market_cap_and_net_income_ttm"
    assert pe_ratio["component_breakdown"]["formula"] == "market_cap / net_income_ttm"
    assert pe_ratio["component_breakdown"]["net_income_ttm"] == 840.0


def test_market_ev_ebitda_uses_ttm_bridge_for_q1(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"

    _write_parquet(
        facts_path,
        [
            {
                **_facts_row("cash", "ABC", "financial.cash", 50.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
                "effective_at": "2025-12-31T00:00:00Z",
            },
            {
                **_facts_row("debt", "ABC", "financial.total_debt", 200.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
                "effective_at": "2025-12-31T00:00:00Z",
            },
            {
                **_facts_row("shares", "ABC", "financial.shares_out", 100.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
                "effective_at": "2025-12-31T00:00:00Z",
            },
            {
                **_facts_row("ebitda_annual", "ABC", "financial.ebitda", 1000.0, "2025-11-15T00:00:00Z", "2025-11-16T00:00:00Z", "2025-11-15T00:00:00Z"),
                "effective_at": "2025-09-30T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2025-09-30 00:00:00; fiscal_year=2025; fiscal_quarter=",
            },
            {
                **_facts_row("ebitda_q1_prior", "ABC", "financial.ebitda", 220.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
                "effective_at": "2024-12-31T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2024-12-31 00:00:00; fiscal_year=2026; fiscal_quarter=1",
            },
            {
                **_facts_row("ebitda_q1_current", "ABC", "financial.ebitda", 260.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
                "effective_at": "2025-12-31T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2025-12-31 00:00:00; fiscal_year=2026; fiscal_quarter=1",
            },
        ],
    )
    _write_parquet(
        ts_path,
        [
            {
                "entity_id": "ABC",
                "series_type": "price",
                "trade_date": "2025-12-31T00:00:00Z",
                "available_time": "2025-12-31T00:00:00Z",
                "ingestion_time": "2026-01-02T00:00:00Z",
                "close": 10.0,
            }
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, timeseries_path=ts_path, skip_timeseries=False)
    snap = builder.build("ABC", "2026-02-28")

    feature = snap.features["market.ev_ebitda"]
    assert round(feature["value"], 6) == round(1150.0 / 1040.0, 6)
    assert feature["component_breakdown"]["ebitda_ttm"] == 1040.0
    assert feature["component_breakdown"]["ebitda_ttm_context"]["formula"] == "latest_annual + current_q1 - prior_year_q1"


def test_market_pe_ratio_uses_reference_market_cap_when_shares_missing(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"
    taxonomy_reference_path = tmp_path / "taxonomy_reference.parquet"

    _write_parquet(
        facts_path,
        [
            {
                **_facts_row("ni_annual", "ABC_CIK", "financial.net_income", 800.0, "2025-11-15T00:00:00Z", "2025-11-16T00:00:00Z", "2025-11-15T00:00:00Z"),
                "effective_at": "2025-09-30T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2025-09-30 00:00:00; fiscal_year=2025; fiscal_quarter=",
            },
        ],
    )
    _write_parquet(
        ts_path,
        [
            {
                "entity_id": "ABC_CIK",
                "series_type": "price",
                "trade_date": "2025-12-31T00:00:00Z",
                "available_time": "2025-12-31T00:00:00Z",
                "ingestion_time": "2026-01-02T00:00:00Z",
                "adjusted_close": 50.0,
            }
        ],
    )
    _write_parquet(
        ident_path,
        [
            {"entity_id": "ABC_CIK", "identifier_type": "ticker", "identifier_value": "ABC"},
            {"entity_id": "ABC_CIK", "identifier_type": "cik", "identifier_value": "ABC_CIK"},
        ],
    )
    _write_parquet(
        taxonomy_reference_path,
        [
            {
                "Instrument": "ABC.N",
                "Company Common Name": "ABC Inc",
                "Company Market Cap": 5000.0,
                "GICS Sector Name": "Industrials",
                "GICS Industry Name": "Machinery",
            }
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        timeseries_path=ts_path,
        skip_timeseries=False,
        entity_identifier_path=ident_path,
        taxonomy_reference_path=taxonomy_reference_path,
    )
    snap = builder.build("ABC_CIK", "2026-02-28")

    market_cap = snap.features["market.market_cap"]
    pe_ratio = snap.features["market.pe_ratio"]
    assert market_cap["value"] == 5000.0
    assert market_cap["fallback_used"] == "reference_company_market_cap"
    assert market_cap["support_mode"] == "exact"
    assert "reference_market_cap_fallback" in (market_cap["quality_flags"] or [])
    assert round(pe_ratio["value"], 6) == round(5000.0 / 800.0, 6)
    assert pe_ratio["fallback_used"] == "derived_from_market_cap_and_net_income_ttm"
    assert pe_ratio["support_mode"] == "exact"


def test_market_ev_ebitda_uses_reference_ebitda_when_statement_metric_missing(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"
    taxonomy_reference_path = tmp_path / "taxonomy_reference.parquet"

    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC_CIK", "financial.cash", 50.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
            _facts_row("debt", "ABC_CIK", "financial.total_debt", 200.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
        ],
    )
    _write_parquet(
        ts_path,
        [
            {
                "entity_id": "ABC_CIK",
                "series_type": "price",
                "trade_date": "2025-12-31T00:00:00Z",
                "available_time": "2025-12-31T00:00:00Z",
                "ingestion_time": "2026-01-02T00:00:00Z",
                "adjusted_close": 50.0,
            }
        ],
    )
    _write_parquet(
        ident_path,
        [
            {"entity_id": "ABC_CIK", "identifier_type": "ticker", "identifier_value": "ABC"},
            {"entity_id": "ABC_CIK", "identifier_type": "cik", "identifier_value": "ABC_CIK"},
        ],
    )
    _write_parquet(
        taxonomy_reference_path,
        [
            {
                "Instrument": "ABC.N",
                "Company Common Name": "ABC Inc",
                "Company Market Cap": 5000.0,
                "EBITDA": 1000.0,
                "Enterprise Value To EBITDA (Daily Time Series Ratio)": 5.15,
                "GICS Sector Name": "Industrials",
                "GICS Industry Name": "Machinery",
            }
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        timeseries_path=ts_path,
        skip_timeseries=False,
        entity_identifier_path=ident_path,
        taxonomy_reference_path=taxonomy_reference_path,
    )
    snap = builder.build("ABC_CIK", "2026-02-28")

    feature = snap.features["market.ev_ebitda"]
    assert round(feature["value"], 6) == round((5000.0 + 150.0) / 1000.0, 6)
    assert feature["fallback_used"] == "reference_ebitda"
    assert feature["support_mode"] == "proxy_missing_component"
    assert feature["component_breakdown"]["ebitda_ttm"] == 1000.0
    assert feature["component_breakdown"]["ebitda_ttm_context"]["formula"] == "provider_ebitda_reference"


def test_market_cap_prefers_reference_over_stale_price_shares(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"
    taxonomy_reference_path = tmp_path / "taxonomy_reference.parquet"

    _write_parquet(
        facts_path,
        [
            _facts_row("shares", "ABC_CIK", "financial.shares_out", 100.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
        ],
    )
    _write_parquet(
        ts_path,
        [
            {
                "entity_id": "ABC_CIK",
                "series_type": "price",
                "trade_date": "2025-01-31T00:00:00Z",
                "available_time": "2025-01-31T00:00:00Z",
                "ingestion_time": "2025-02-01T00:00:00Z",
                "adjusted_close": 10.0,
            }
        ],
    )
    _write_parquet(
        ident_path,
        [
            {"entity_id": "ABC_CIK", "identifier_type": "ticker", "identifier_value": "ABC"},
            {"entity_id": "ABC_CIK", "identifier_type": "cik", "identifier_value": "ABC_CIK"},
        ],
    )
    _write_parquet(
        taxonomy_reference_path,
        [
            {
                "Instrument": "ABC.N",
                "Company Common Name": "ABC Inc",
                "Company Market Cap": 5000.0,
                "GICS Sector Name": "Industrials",
                "GICS Industry Name": "Machinery",
            }
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        timeseries_path=ts_path,
        skip_timeseries=False,
        entity_identifier_path=ident_path,
        taxonomy_reference_path=taxonomy_reference_path,
    )
    snap = builder.build("ABC_CIK", "2026-02-28")

    market_cap = snap.features["market.market_cap"]
    assert market_cap["value"] == 5000.0
    assert market_cap["fallback_used"] == "reference_company_market_cap"
    assert "reference_market_cap_preferred_over_stale_price_shares" in (market_cap["quality_flags"] or [])


def test_gross_leverage_uses_reference_ebitda_and_proxy_denominator_when_lease_charge_missing(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"
    taxonomy_reference_path = tmp_path / "taxonomy_reference.parquet"

    _write_parquet(
        facts_path,
        [
            _facts_row("debt", "ABC_CIK", "financial.total_debt", 300.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
            _facts_row("cash", "ABC_CIK", "financial.cash", 25.0, "2026-02-10T00:00:00Z", "2026-02-11T00:00:00Z", "2026-02-10T00:00:00Z"),
        ],
    )
    _write_parquet(
        ident_path,
        [
            {"entity_id": "ABC_CIK", "identifier_type": "ticker", "identifier_value": "ABC"},
            {"entity_id": "ABC_CIK", "identifier_type": "cik", "identifier_value": "ABC_CIK"},
        ],
    )
    _write_parquet(
        taxonomy_reference_path,
        [
            {
                "Instrument": "ABC.N",
                "Company Common Name": "ABC Retail",
                "Company Market Cap": 5000.0,
                "EBITDA": 100.0,
                "GICS Sector Name": "Consumer Staples",
                "GICS Industry Name": "Consumer Staples Distribution & Retail",
            }
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        entity_identifier_path=ident_path,
        taxonomy_reference_path=taxonomy_reference_path,
    )
    snap = builder.build("ABC_CIK", "2026-02-28")

    gross_leverage = snap.features["capital_structure.gross_leverage"]
    assert gross_leverage["value"] == 3.0
    assert gross_leverage["support_mode"] == "proxy_missing_component"
    assert gross_leverage["component_breakdown"]["ebitda"] == 100.0
    assert gross_leverage["component_breakdown"]["effective_denominator_policy"] == "ebitda_proxy_for_missing_lease_charge"
    assert "reference_ebitda_fallback" in (gross_leverage["quality_flags"] or [])
    assert "lease_adjusted_denominator_fallback_to_ebitda" in (gross_leverage["quality_flags"] or [])


def test_fixed_charge_preference_flag_does_not_downgrade_exact_support(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    entity_path = tmp_path / "entity.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "RETL", "financial.cash", 20.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("debt", "RETL", "financial.total_debt", 120.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("lease_current", "RETL", "financial.lease_liability_current", 10.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("lease_long", "RETL", "financial.lease_liability_noncurrent", 30.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("lease_expense", "RETL", "financial.lease_expense", 12.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebitda", "RETL", "financial.ebitda", 80.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("ebit", "RETL", "financial.ebit", 68.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
            _facts_row("interest", "RETL", "financial.interest_expense", 8.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [_entity_row("RETL", sector="Consumer Discretionary", subsector="Specialty Retail", sic="5331")],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        entity_table_path=entity_path,
    )
    snap = builder.build("RETL", "2026-02-28")

    fixed_charge = snap.features["capital_structure.fixed_charge_coverage"]
    interest_cov = snap.features["capital_structure.interest_coverage"]
    assert fixed_charge["applicability_status"] == "primary"
    assert fixed_charge["support_mode"] == "exact"
    assert "fixed_charge_coverage_preferred" in (fixed_charge["quality_flags"] or [])
    assert interest_cov["applicability_status"] == "secondary"
    assert interest_cov["support_mode"] == "exact"


def test_coverage_metrics_fall_back_to_repaired_statement_direct_interest_expense(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    entity_path = tmp_path / "entity.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"
    companyfacts_root = tmp_path / "companyfacts"
    companyfacts_root.mkdir(parents=True, exist_ok=True)

    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "RETL", "financial.cash", 20.0, "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z"),
            _facts_row("debt", "RETL", "financial.total_debt", 120.0, "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z"),
            _facts_row("lease_current", "RETL", "financial.lease_liability_current", 10.0, "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z"),
            _facts_row("lease_long", "RETL", "financial.lease_liability_noncurrent", 30.0, "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z"),
            _facts_row("lease_expense", "RETL", "financial.lease_expense", 12.0, "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z"),
            _facts_row("ebitda", "RETL", "financial.ebitda", 80.0, "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z"),
            _facts_row("ebit", "RETL", "financial.ebit", 68.0, "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z", "2024-10-31T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [_entity_row("RETL", sector="Consumer Discretionary", subsector="Specialty Retail", sic="5331")],
    )
    _write_parquet(
        ident_path,
        [{"entity_id": "RETL", "identifier_type": "cik", "identifier_value": "1234567890"}],
    )
    (companyfacts_root / "CIK1234567890.json").write_text(
        json.dumps(
            {
                "facts": {
                    "us-gaap": {
                        "InterestExpense": {
                            "units": {
                                "USD": [
                                    {
                                        "start": "2024-01-01",
                                        "end": "2024-09-30",
                                        "filed": "2024-10-31",
                                        "val": 24.0,
                                        "fy": 2024,
                                        "fp": "Q3",
                                        "form": "10-Q",
                                    },
                                    {
                                        "start": "2023-01-01",
                                        "end": "2023-12-31",
                                        "filed": "2024-02-15",
                                        "val": 30.0,
                                        "fy": 2023,
                                        "fp": "FY",
                                        "form": "10-K",
                                    },
                                    {
                                        "start": "2023-01-01",
                                        "end": "2023-09-30",
                                        "filed": "2023-10-31",
                                        "val": 18.0,
                                        "fy": 2023,
                                        "fp": "Q3",
                                        "form": "10-Q",
                                    },
                                ]
                            }
                        }
                    }
                }
            }
        )
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        entity_table_path=entity_path,
        entity_identifier_path=ident_path,
        companyfacts_root=companyfacts_root,
        enable_market_relevant_smart_normalized_inputs=True,
    )
    snap = builder.build("RETL", "2024-12-31")

    fixed_charge = snap.features["capital_structure.fixed_charge_coverage"]
    interest_cov = snap.features["capital_structure.interest_coverage"]
    assert interest_cov["support_mode"] == "exact"
    assert fixed_charge["support_mode"] == "exact"
    assert interest_cov["fallback_used"] == "statement_direct_interest_expense_fallback"
    assert fixed_charge["fallback_used"] == "statement_direct_interest_expense_fallback"


def test_builder_keeps_smart_normalized_market_inputs_as_sidecar_features(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    entity_path = tmp_path / "entity.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"
    companyfacts_root = tmp_path / "companyfacts"
    companyfacts_root.mkdir(parents=True, exist_ok=True)

    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "TEST", "financial.cash", 100.0, "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z"),
            _facts_row(
                "cash_sti",
                "TEST",
                "financial.cash_and_short_term_investments",
                140.0,
                "2024-12-15T00:00:00Z",
                "2024-12-15T00:00:00Z",
                "2024-12-15T00:00:00Z",
            ),
            _facts_row("debt", "TEST", "financial.total_debt", 400.0, "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z"),
            _facts_row("debt_current", "TEST", "financial.debt_current", 50.0, "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z"),
            _facts_row("debt_long", "TEST", "financial.debt_long", 350.0, "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z"),
            _facts_row("ebitda", "TEST", "financial.ebitda", 100.0, "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z"),
            _facts_row("net_income", "TEST", "financial.net_income", 60.0, "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z", "2024-12-15T00:00:00Z"),
            _facts_row(
                "pension",
                "TEST",
                "financial.net_pension_liability",
                30.0,
                "2024-12-15T00:00:00Z",
                "2024-12-15T00:00:00Z",
                "2024-12-15T00:00:00Z",
            ),
            _facts_row(
                "interest_expense",
                "TEST",
                "financial.interest_expense",
                20.0,
                "2024-12-15T00:00:00Z",
                "2024-12-15T00:00:00Z",
                "2024-12-15T00:00:00Z",
            ),
        ],
    )
    _write_parquet(
        entity_path,
        [_entity_row("TEST", sector="Industrials", subsector="Machinery", sic="3530")],
    )
    _write_parquet(
        ident_path,
        [{"entity_id": "TEST", "identifier_type": "cik", "identifier_value": "1234567890"}],
    )
    (companyfacts_root / "CIK1234567890.json").write_text(
        json.dumps(
            {
                "facts": {
                    "us-gaap": {
                        "RestrictedCash": {
                            "units": {
                                "USD": [
                                    {"val": 10.0, "end": "2024-12-31", "filed": "2024-12-31", "fy": 2024, "fp": "FY", "form": "10-K"}
                                ]
                            }
                        },
                        "ShortTermInvestments": {
                            "units": {
                                "USD": [
                                    {"val": 40.0, "end": "2024-12-31", "filed": "2024-12-31", "fy": 2024, "fp": "FY", "form": "10-K"}
                                ]
                            }
                        },
                        "OperatingLeaseLiabilityCurrent": {
                            "units": {
                                "USD": [
                                    {"val": 15.0, "end": "2024-12-31", "filed": "2024-12-31", "fy": 2024, "fp": "FY", "form": "10-K"}
                                ]
                            }
                        },
                        "OperatingLeaseLiabilityNoncurrent": {
                            "units": {
                                "USD": [
                                    {"val": 35.0, "end": "2024-12-31", "filed": "2024-12-31", "fy": 2024, "fp": "FY", "form": "10-K"}
                                ]
                            }
                        },
                    }
                }
            }
        )
    )

    baseline_builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        entity_table_path=entity_path,
        entity_identifier_path=ident_path,
        companyfacts_root=companyfacts_root,
    )
    baseline_snap = baseline_builder.build("TEST", "2024-12-31")

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        entity_table_path=entity_path,
        entity_identifier_path=ident_path,
        companyfacts_root=companyfacts_root,
        enable_market_relevant_smart_normalized_inputs=True,
    )
    snap = builder.build("TEST", "2024-12-31")

    assert snap.features["liquidity.available_liquidity_normalized"]["value"] == 130.0
    assert snap.features["capital_structure.debt_like_obligations_normalized"]["value"] == 450.0
    assert snap.features["capital_structure.net_debt_normalized"]["value"] == 320.0
    assert snap.features["capital_structure.gross_leverage_normalized"]["value"] == 4.5
    assert snap.features["capital_structure.net_leverage_normalized"]["value"] == 3.2
    assert snap.features["capital_structure.net_pension_liability"]["value"] == 30.0
    assert snap.features["capital_structure.debt_like_obligations_including_pension"]["value"] == 450.0
    assert snap.features["capital_structure.debt_like_obligations_including_pension"]["support_mode"] == "proxy_missing_component"
    assert snap.features["capital_structure.net_debt_including_pension"]["value"] == 320.0
    assert snap.features["capital_structure.gross_leverage_including_pension"]["value"] == 4.5
    assert snap.features["capital_structure.net_leverage_including_pension"]["value"] == 3.2
    assert snap.features["liquidity.available_for_actions"]["value"] == baseline_snap.features["liquidity.available_for_actions"]["value"]
    assert snap.features["liquidity.available_for_actions_market"]["value"] == baseline_snap.features["liquidity.available_for_actions_market"]["value"]
    assert snap.features["capital_structure.net_debt"]["value"] == baseline_snap.features["capital_structure.net_debt"]["value"]
    assert snap.features["capital_structure.gross_leverage"]["value"] == baseline_snap.features["capital_structure.gross_leverage"]["value"]
    assert snap.features["capital_structure.net_leverage"]["value"] == baseline_snap.features["capital_structure.net_leverage"]["value"]
    assert snap.features["capital_structure.net_leverage"]["support_mode"] == baseline_snap.features["capital_structure.net_leverage"]["support_mode"]


def test_builder_prefers_fresher_companyfacts_for_raw_cash_and_total_debt(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"
    companyfacts_root = tmp_path / "companyfacts"
    companyfacts_root.mkdir(parents=True, exist_ok=True)

    stale_published = "2025-11-04T00:00:00Z"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash_stale", "ABC", "financial.cash", 80.0, stale_published, stale_published, stale_published),
            _facts_row("debt_stale", "ABC", "financial.total_debt", 90.0, stale_published, stale_published, stale_published),
        ],
    )
    _write_parquet(
        ident_path,
        [{"entity_id": "ABC", "identifier_type": "cik", "identifier_value": "1"}],
    )
    (companyfacts_root / "CIK0000000001.json").write_text(
        json.dumps(
            {
                "facts": {
                    "us-gaap": {
                        "CashAndCashEquivalentsAtCarryingValue": {
                            "units": {
                                "USD": [
                                    {
                                        "val": 70.0,
                                        "end": "2025-12-31",
                                        "filed": "2026-02-13",
                                        "fy": 2025,
                                        "fp": "FY",
                                        "form": "10-K",
                                    }
                                ]
                            }
                        },
                        "DebtLongtermAndShorttermCombinedAmount": {
                            "units": {
                                "USD": [
                                    {
                                        "val": 120.0,
                                        "end": "2025-12-31",
                                        "filed": "2026-02-13",
                                        "fy": 2025,
                                        "fp": "FY",
                                        "form": "10-K",
                                    }
                                ]
                            }
                        },
                    }
                }
            }
        )
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        entity_identifier_path=ident_path,
        companyfacts_root=companyfacts_root,
    )
    snap = builder.build("ABC", "2026-02-28")

    cash_feature = snap.features["liquidity.cash"]
    debt_feature = snap.features["capital_structure.total_debt_reported"]

    assert cash_feature["value"] == 70.0
    assert cash_feature["fallback_used"] == "companyfacts_cash_exact_fresher"
    assert "companyfacts_cash_fresher" in (cash_feature["quality_flags"] or [])

    assert debt_feature["value"] == 120.0
    assert debt_feature["fallback_used"] == "companyfacts_total_debt_exact_fresher"
    assert "companyfacts_total_debt_fresher" in (debt_feature["quality_flags"] or [])


def test_total_debt_completeness_uses_reference_when_latest_local_is_implausibly_low(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ident_path = tmp_path / "entity_identifier.parquet"
    taxonomy_reference_path = tmp_path / "taxonomy_reference.parquet"

    _write_parquet(
        facts_path,
        [
            {
                **_facts_row("debt_annual", "ABC_CIK", "financial.total_debt", 250.0, "2025-02-10T00:00:00Z", "2025-02-11T00:00:00Z", "2025-02-10T00:00:00Z"),
                "effective_at": "2024-12-31T00:00:00Z",
                "context_norm": "statement_type=derived; fiscal_period_end=2024-12-31 00:00:00; fiscal_year=2025; fiscal_quarter=",
            },
            {
                **_facts_row("debt_latest", "ABC_CIK", "financial.total_debt", 10.0, "2025-11-10T00:00:00Z", "2025-11-11T00:00:00Z", "2025-11-10T00:00:00Z"),
                "effective_at": "2025-09-30T00:00:00Z",
                "context_norm": "statement_type=derived; fiscal_period_end=2025-09-30 00:00:00; fiscal_year=2026; fiscal_quarter=3",
            },
            _facts_row("cash", "ABC_CIK", "financial.cash", 50.0, "2025-11-10T00:00:00Z", "2025-11-11T00:00:00Z", "2025-11-10T00:00:00Z"),
        ],
    )
    _write_parquet(
        ident_path,
        [
            {"entity_id": "ABC_CIK", "identifier_type": "ticker", "identifier_value": "ABC"},
            {"entity_id": "ABC_CIK", "identifier_type": "cik", "identifier_value": "ABC_CIK"},
        ],
    )
    _write_parquet(
        taxonomy_reference_path,
        [
            {
                "Instrument": "ABC.N",
                "Company Common Name": "ABC Logistics",
                "Total Debt": 260.0,
                "EBITDA": 100.0,
                "GICS Sector Name": "Industrials",
                "GICS Industry Name": "Air Freight & Logistics",
            }
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        entity_identifier_path=ident_path,
        taxonomy_reference_path=taxonomy_reference_path,
    )
    snap = builder.build("ABC_CIK", "2026-02-28")

    total_debt = snap.features["capital_structure.total_debt"]
    gross_leverage = snap.features["capital_structure.gross_leverage"]
    assert total_debt["value"] == 260.0
    assert total_debt["component_breakdown"]["local_reported_debt"] == 10.0
    assert total_debt["component_breakdown"]["debt_reference_source"] == "reference_total_debt"
    assert "reference_total_debt_used_for_completeness" in (total_debt["quality_flags"] or [])
    assert round(gross_leverage["value"], 6) == 2.6


def test_macro_strict_market_metrics_use_public_series_and_history_percentiles(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"

    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 10.0, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ],
    )

    macro_rows = []
    monthly_dates = pd.date_range("2025-03-31", periods=12, freq="ME", tz="UTC")
    for idx, date in enumerate(monthly_dates):
        macro_rows.extend(
            [
                {
                    "entity_id": "MACRO",
                    "series_type": "macro",
                    "trade_date": date.isoformat(),
                    "available_time": date.isoformat(),
                    "ingestion_time": date.isoformat(),
                    "series_id": "SP500_PE_RATIO",
                    "value": float(18.0 + idx),
                },
                {
                    "entity_id": "MACRO",
                    "series_type": "macro",
                    "trade_date": date.isoformat(),
                    "available_time": date.isoformat(),
                    "ingestion_time": date.isoformat(),
                    "series_id": "DGS10",
                    "value": float(3.0 + (0.1 * idx)),
                },
                {
                    "entity_id": "MACRO",
                    "series_type": "macro",
                    "trade_date": date.isoformat(),
                    "available_time": date.isoformat(),
                    "ingestion_time": date.isoformat(),
                    "series_id": "DGS2",
                    "value": float(2.5 + (0.1 * idx)),
                },
                {
                    "entity_id": "MACRO",
                    "series_type": "macro",
                    "trade_date": date.isoformat(),
                    "available_time": date.isoformat(),
                    "ingestion_time": date.isoformat(),
                    "series_id": "BAMLC0A0CM",
                    "value": float(95.0 + idx),
                },
                {
                    "entity_id": "MACRO",
                    "series_type": "macro",
                    "trade_date": date.isoformat(),
                    "available_time": date.isoformat(),
                    "ingestion_time": date.isoformat(),
                    "series_id": "BAMLH0A0HYM2",
                    "value": float(300.0 + (5.0 * idx)),
                },
                {
                    "entity_id": "MACRO",
                    "series_type": "macro",
                    "trade_date": date.isoformat(),
                    "available_time": date.isoformat(),
                    "ingestion_time": date.isoformat(),
                    "series_id": "BAMLH0A0HYM2EY",
                    "value": float(7.0 + (0.2 * idx)),
                },
                {
                    "entity_id": "MACRO",
                    "series_type": "macro",
                    "trade_date": date.isoformat(),
                    "available_time": date.isoformat(),
                    "ingestion_time": date.isoformat(),
                    "series_id": "DFF",
                    "value": float(4.0 + (0.05 * idx)),
                },
                {
                    "entity_id": "MACRO",
                    "series_type": "macro",
                    "trade_date": date.isoformat(),
                    "available_time": date.isoformat(),
                    "ingestion_time": date.isoformat(),
                    "series_id": "SOFR",
                    "value": float(4.1 + (0.05 * idx)),
                },
                {
                    "entity_id": "MACRO",
                    "series_type": "macro",
                    "trade_date": date.isoformat(),
                    "available_time": date.isoformat(),
                    "ingestion_time": date.isoformat(),
                    "series_id": "VIXCLS",
                    "value": float(15.0 + idx),
                },
            ]
        )

    quarterly_dates = pd.date_range("2022-09-30", periods=14, freq="QE", tz="UTC")
    gdp_values = [
        100.0,
        101.0,
        102.5,
        104.0,
        106.0,
        108.0,
        110.5,
        113.0,
        116.0,
        118.0,
        120.5,
        123.0,
        126.0,
        130.0,
    ]
    for date, value in zip(quarterly_dates, gdp_values):
        macro_rows.append(
            {
                "entity_id": "MACRO",
                "series_type": "macro",
                "trade_date": date.isoformat(),
                "available_time": date.isoformat(),
                "ingestion_time": date.isoformat(),
                "series_id": "GDPC1",
                "value": float(value),
            }
        )

    _write_parquet(ts_path, macro_rows)

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        timeseries_path=ts_path,
        skip_timeseries=True,
        skip_macro=False,
    )
    snap = builder.build("ABC", "2026-02-28")

    sp500_pe = snap.features["macro.sp500_pe_ttm"]
    us2y = snap.features["macro.ust_2y_yield"]
    us10y = snap.features["macro.us10y_treasury_yield"]
    ust10y = snap.features["macro.ust_10y_yield"]
    ig_oas = snap.features["macro.us_ig_oas"]
    ig_oas_alias = snap.features["macro.ig_oas"]
    hy_oas = snap.features["macro.hy_oas"]
    hy_yield = snap.features["macro.us_hy_all_in_yield"]
    fed_funds = snap.features["macro.fed_funds_effective"]
    sofr = snap.features["macro.sofr"]
    vix = snap.features["market.vix"]
    gdp = snap.features["macro.real_gdp_growth_yoy"]

    assert sp500_pe["value"] == 29.0
    assert sp500_pe["input_layer_bucket"] == "strict_market_defined"
    assert sp500_pe["strict_market_defined"] is True
    assert sp500_pe["component_breakdown"]["formula"] == "latest(sp500_pe_ttm_series)"
    assert snap.features["macro.sp500_pe_ttm_percentile_history"]["value"] == 100.0

    assert round(us2y["value"], 10) == 3.6
    assert round(us10y["value"], 10) == 4.1
    assert round(ust10y["value"], 10) == 4.1
    assert snap.features["macro.us10y_treasury_yield_percentile_history"]["value"] == 100.0
    assert ig_oas["value"] == 106.0
    assert ig_oas_alias["value"] == 106.0
    assert snap.features["macro.us_ig_oas_percentile_history"]["value"] == 100.0
    assert hy_oas["value"] == 355.0
    assert hy_yield["value"] == 9.2
    assert snap.features["macro.us_hy_all_in_yield_percentile_history"]["value"] == 100.0
    assert round(fed_funds["value"], 10) == 4.55
    assert round(sofr["value"], 10) == 4.65
    assert vix["value"] == 26.0

    expected_gdp_growth = (130.0 / 118.0) - 1.0
    assert round(gdp["value"], 10) == round(expected_gdp_growth, 10)
    assert snap.features["macro.real_gdp_growth_yoy_percentile_history"]["value"] == 100.0
    assert gdp["component_breakdown"]["formula"] == "(real_gdp_t / real_gdp_t_minus_4) - 1"


def test_macro_history_uses_instrument_id_when_metric_column_is_empty(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"

    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 10.0, "2024-07-28T00:00:00Z", "2024-07-28T00:00:00Z", "2024-07-28T00:00:00Z"),
        ],
    )

    macro_rows = [
        {
            "entity_id": "MACRO",
            "series_type": "macro",
            "trade_date": "2024-07-26T00:00:00+00:00",
            "available_time": "2024-07-27T00:00:00+00:00",
            "ingestion_time": "2024-07-27T00:00:00+00:00",
            "metric": None,
            "instrument_id": "DGS10",
            "value": 4.2,
        },
        {
            "entity_id": "MACRO",
            "series_type": "macro",
            "trade_date": "2024-07-26T00:00:00+00:00",
            "available_time": "2024-07-27T00:00:00+00:00",
            "ingestion_time": "2024-07-27T00:00:00+00:00",
            "metric": None,
            "instrument_id": "DGS2",
            "value": 4.6,
        },
        {
            "entity_id": "MACRO",
            "series_type": "macro",
            "trade_date": "2024-07-26T00:00:00+00:00",
            "available_time": "2024-07-27T00:00:00+00:00",
            "ingestion_time": "2024-07-27T00:00:00+00:00",
            "metric": None,
            "instrument_id": "DFF",
            "value": 5.3,
        },
        {
            "entity_id": "MACRO",
            "series_type": "macro",
            "trade_date": "2024-07-26T00:00:00+00:00",
            "available_time": "2024-07-27T00:00:00+00:00",
            "ingestion_time": "2024-07-27T00:00:00+00:00",
            "metric": None,
            "instrument_id": "SOFR",
            "value": 5.31,
        },
        {
            "entity_id": "MACRO",
            "series_type": "macro",
            "trade_date": "2024-07-26T00:00:00+00:00",
            "available_time": "2024-07-27T00:00:00+00:00",
            "ingestion_time": "2024-07-27T00:00:00+00:00",
            "metric": None,
            "instrument_id": "BAMLC0A0CM",
            "value": 1.12,
        },
        {
            "entity_id": "MACRO",
            "series_type": "macro",
            "trade_date": "2024-07-26T00:00:00+00:00",
            "available_time": "2024-07-27T00:00:00+00:00",
            "ingestion_time": "2024-07-27T00:00:00+00:00",
            "metric": None,
            "instrument_id": "BAMLH0A0HYM2",
            "value": 3.8,
        },
        {
            "entity_id": "MACRO",
            "series_type": "macro",
            "trade_date": "2024-07-26T00:00:00+00:00",
            "available_time": "2024-07-27T00:00:00+00:00",
            "ingestion_time": "2024-07-27T00:00:00+00:00",
            "metric": None,
            "instrument_id": "VIXCLS",
            "value": 16.4,
        },
    ]
    _write_parquet(ts_path, macro_rows)

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        timeseries_path=ts_path,
        skip_timeseries=True,
        skip_macro=False,
        historical_backfill_mode=True,
    )
    snap = builder.build("ABC", "2024-07-28")

    assert snap.features["macro.ust_10y_yield"]["value"] == 4.2
    assert snap.features["macro.ust_2y_yield"]["value"] == 4.6
    assert snap.features["macro.fed_funds_effective"]["value"] == 5.3
    assert snap.features["macro.sofr"]["value"] == 5.31
    assert snap.features["macro.ig_oas"]["value"] == 1.12
    assert snap.features["macro.hy_oas"]["value"] == 3.8
    assert snap.features["market.vix"]["value"] == 16.4


def test_strategic_recent_actions_and_consolidation_score(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    events_path = tmp_path / "events.parquet"
    _write_parquet(facts_path, [_facts_row("cash", "ABC", "financial.cash", 10.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")])
    events = []
    for i in range(6):
        events.append(
            {
                "event_id": f"evt_{i}",
                "company_id": "ABC",
                "event_type": "acquisition" if i % 2 == 0 else "buyback",
                "event_subtype": None,
                "announced_at": f"2025-0{(i % 6) + 1}-01T00:00:00Z",
                "effective_at": f"2025-0{(i % 6) + 1}-02T00:00:00Z",
                "created_at": f"2025-0{(i % 6) + 1}-01T00:00:00Z",
                "source_type": "event_store",
                "params": {"deal_value": 1_000_000_000.0 if i % 2 == 0 else None},
            }
        )
    _write_parquet(events_path, events)
    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        events_path=events_path,
        skip_events=False,
    )
    snap = builder.build("ABC", "2026-02-28")
    assert snap.features["strategic.recent_actions_count_24m"]["value"] == 6.0
    assert snap.features["strategic.recent_actions_count_24m"]["methodology_execution_decision"] == "keep_externally_anchored_house_formula"
    assert snap.features["strategic.recent_actions_count_24m"]["input_layer_bucket"] == "secondary_externally_anchored"
    assert snap.features["strategic.recent_actions_count_24m"]["component_breakdown"]["strategic_event_count_24m"] == 6
    assert snap.features["strategic.last_action_type"]["value"] == "buyback"
    assert snap.features["strategic.last_action_type"]["component_breakdown"]["formula"] == "latest_action_type(strategic_events_24m)"
    assert snap.features["strategic.action_frequency_24m"]["value"] == 0.25
    assert snap.features["strategic.action_frequency_24m"]["input_layer_bucket"] == "secondary_externally_anchored"
    assert snap.features["strategic.action_frequency_24m"]["component_breakdown"]["formula"] == "count(strategic_events_24m) / 24"
    assert snap.features["strategic.action_fatigue_score"]["value"] > 0


def test_peer_consolidation_wave_score(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    events_path = tmp_path / "events.parquet"
    entity_path = tmp_path / "entity.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 10.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            _facts_row("cash2", "PEER1", "financial.cash", 10.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            _facts_row("cash3", "PEER2", "financial.cash", 10.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [
            {"entity_id": "ABC", "sector": "Tech"},
            {"entity_id": "PEER1", "sector": "Tech"},
            {"entity_id": "PEER2", "sector": "Tech"},
            {"entity_id": "OTHER", "sector": "Energy"},
        ],
    )
    _write_parquet(
        events_path,
        [
            {
                "event_id": "evt_abc_1",
                "company_id": "ABC",
                "event_type": "acquisition",
                "event_subtype": None,
                "announced_at": "2025-12-01T00:00:00Z",
                "effective_at": "2025-12-05T00:00:00Z",
                "created_at": "2025-12-01T00:00:00Z",
                "source_type": "event_store",
                "params": {"deal_value": 2_000_000_000.0},
            },
            {
                "event_id": "evt_p1_1",
                "company_id": "PEER1",
                "event_type": "acquisition",
                "event_subtype": None,
                "announced_at": "2025-11-01T00:00:00Z",
                "effective_at": "2025-11-03T00:00:00Z",
                "created_at": "2025-11-01T00:00:00Z",
                "source_type": "event_store",
                "params": {"deal_value": 1_000_000_000.0},
            },
            {
                "event_id": "evt_p2_1",
                "company_id": "PEER2",
                "event_type": "divestiture",
                "event_subtype": None,
                "announced_at": "2025-10-01T00:00:00Z",
                "effective_at": "2025-10-03T00:00:00Z",
                "created_at": "2025-10-01T00:00:00Z",
                "source_type": "event_store",
                "params": {"deal_value": 500_000_000.0},
            },
        ],
    )
    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        events_path=events_path,
        skip_events=False,
        entity_table_path=entity_path,
        skip_peer_context=False,
    )
    snap = builder.build("ABC", "2026-02-28")
    score = snap.features["peer_context.consolidation_wave_score"]["value"]
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_policy_feature_aliases_for_peer_and_activist_context(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    events_path = tmp_path / "events.parquet"
    entity_path = tmp_path / "entity.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash_abc", "ABC", "financial.cash", 10.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            _facts_row("cash_p1", "PEER1", "financial.cash", 10.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            _facts_row("cash_p2", "PEER2", "financial.cash", 10.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [
            {"entity_id": "ABC", "sector": "Retail", "ev_ebitda": 8.0, "fcf_yield": 0.06, "ebitda_margin": 0.10, "revenue": 100.0},
            {"entity_id": "PEER1", "sector": "Retail", "ev_ebitda": 10.0, "fcf_yield": 0.04, "ebitda_margin": 0.20, "revenue": 200.0},
            {"entity_id": "PEER2", "sector": "Retail", "ev_ebitda": 12.0, "fcf_yield": 0.03, "ebitda_margin": 0.30, "revenue": 300.0},
        ],
    )
    _write_parquet(
        events_path,
        [
            {
                "event_id": "evt_activist",
                "company_id": "ABC",
                "event_type": "activist_campaign",
                "event_subtype": "13d",
                "announced_at": "2026-01-10T00:00:00Z",
                "effective_at": "2026-01-10T00:00:00Z",
                "created_at": "2026-01-10T00:00:00Z",
                "source_type": "event_store",
                "params": None,
            }
        ],
    )
    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        events_path=events_path,
        skip_events=False,
        entity_table_path=entity_path,
        skip_peer_context=False,
    )
    snap = builder.build("ABC", "2026-02-28")

    activist_flag = snap.features["ownership_governance.activist_presence_flag"]
    assert activist_flag["value"] is True
    assert activist_flag["primary_source_basis"] == "ownership_governance.activist_signal_alias"
    assert "policy_feature_alias" in (activist_flag["quality_flags"] or [])

    ev_z = snap.features["market.ev_ebitda_vs_peer_z"]
    assert ev_z["value"] == pytest.approx(-1.22474487139)
    assert ev_z["primary_source_basis"] == "peer_relative_ev_ebitda"

    fcf_pct = snap.features["market.fcf_yield_percentile_peers"]
    assert fcf_pct["value"] == pytest.approx(1.0)
    assert fcf_pct["primary_source_basis"] == "peer_relative_fcf_yield"

    margin_pct = snap.features["operating.ebitda_margin_percentile_peers"]
    assert margin_pct["value"] == pytest.approx(1.0 / 3.0)
    assert margin_pct["primary_source_basis"] == "peer_context.margin_percentile_alias"

    market_share_pct = snap.features["peer_context.relative_positioning.market_share_percentile"]
    assert market_share_pct["value"] == pytest.approx(1.0 / 3.0)
    assert market_share_pct["primary_source_basis"] == "peer_relative_revenue_scale_proxy"


def test_segment_portfolio_context_uses_archetype_and_portfolio_events(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    events_path = tmp_path / "events.parquet"
    entity_path = tmp_path / "entity.parquet"
    taxonomy_path = tmp_path / "taxonomy_reference.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash_abc", "ABC", "financial.cash", 20.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            _facts_row("cash_p1", "PEER1", "financial.cash", 20.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            _facts_row("cash_p2", "PEER2", "financial.cash", 20.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        entity_path,
        [
            {"entity_id": "ABC", "sector": "Conglomerates", "ev_ebitda": 8.0, "fcf_yield": 0.05, "ebitda_margin": 0.12, "revenue": 100.0},
            {"entity_id": "PEER1", "sector": "Conglomerates", "ev_ebitda": 11.0, "fcf_yield": 0.04, "ebitda_margin": 0.18, "revenue": 220.0},
            {"entity_id": "PEER2", "sector": "Conglomerates", "ev_ebitda": 13.0, "fcf_yield": 0.03, "ebitda_margin": 0.24, "revenue": 320.0},
        ],
    )
    _write_parquet(
        taxonomy_path,
        [
            {
                "Instrument": "ABC",
                "Company Common Name": "Example Conglomerate",
                "GICS Sector Name": "Conglomerate",
                "GICS Industry Name": "Industrial Conglomerates",
            }
        ],
    )
    _write_parquet(
        events_path,
        [
            {
                "event_id": "evt_div",
                "company_id": "ABC",
                "event_type": "divestiture",
                "event_subtype": None,
                "announced_at": "2025-06-01T00:00:00Z",
                "effective_at": "2025-06-15T00:00:00Z",
                "created_at": "2025-06-01T00:00:00Z",
                "source_type": "event_store",
                "params": {"deal_value": 600_000_000.0},
            }
        ],
    )
    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        events_path=events_path,
        skip_events=False,
        entity_table_path=entity_path,
        taxonomy_reference_path=taxonomy_path,
        skip_peer_context=False,
    )
    snap = builder.build("ABC", "2026-02-28")

    segment_count = snap.features["strategic.segment_count"]
    assert segment_count["value"] == 3.0
    assert segment_count["support_mode"] == "inferred"
    assert "archetype_multisegment_profile" in (segment_count["quality_flags"] or [])

    segment_refs = snap.features["strategic.segment_references"]
    assert segment_refs["value"] == ["segment_1", "segment_2", "segment_3"]

    divergence = snap.features["operating.segment_margin_divergence"]
    assert divergence["value"] is not None
    assert divergence["value"] > 0.0
    assert divergence["primary_source_basis"] == "segment_portfolio_divergence_house_formula"

    discount = snap.features["market.conglomerate_discount_signal"]
    assert discount["value"] is not None
    assert discount["value"] > 0.0
    assert discount["primary_source_basis"] == "conglomerate_discount_house_formula"


def test_segment_portfolio_context_infers_multisegment_from_portfolio_events(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    events_path = tmp_path / "events.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash_abc", "ABC", "financial.cash", 15.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        events_path,
        [
            {
                "event_id": "evt_spin",
                "company_id": "ABC",
                "event_type": "spin_off",
                "event_subtype": None,
                "announced_at": "2024-07-01T00:00:00Z",
                "effective_at": "2024-07-15T00:00:00Z",
                "created_at": "2024-07-01T00:00:00Z",
                "source_type": "event_store",
                "params": None,
            },
            {
                "event_id": "evt_div",
                "company_id": "ABC",
                "event_type": "divestiture",
                "event_subtype": None,
                "announced_at": "2025-08-01T00:00:00Z",
                "effective_at": "2025-08-20T00:00:00Z",
                "created_at": "2025-08-01T00:00:00Z",
                "source_type": "event_store",
                "params": None,
            },
        ],
    )
    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        events_path=events_path,
        skip_events=False,
        skip_peer_context=True,
    )
    snap = builder.build("ABC", "2026-02-28")

    segment_count = snap.features["strategic.segment_count"]
    assert segment_count["value"] == 3.0
    assert "portfolio_event_multisegment_inference" in (segment_count["quality_flags"] or [])

    segment_refs = snap.features["strategic.segment_references"]
    assert segment_refs["value"] == ["segment_1", "segment_2", "segment_3"]

    divergence = snap.features["operating.segment_margin_divergence"]
    assert divergence["value"] == pytest.approx(0.4)
    assert "portfolio_events_support_segment_divergence" in (divergence["quality_flags"] or [])

    discount = snap.features["market.conglomerate_discount_signal"]
    assert discount["value"] is None
    assert discount["missing_reason"] == "unavailable"


def test_expectations_and_revisions_features_from_warehouse_estimates(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    estimates_path = tmp_path / "warehouse_estimates.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash_abc", "ABC", "financial.cash", 10.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        estimates_path,
        [
            {
                "source_system": "refinitiv_estimates",
                "entity_id": "ABC",
                "company_id": "ABC",
                "security_id": "ABC",
                "event_time": "2025-12-01T00:00:00Z",
                "available_time": "2025-12-01T00:00:00Z",
                "ingestion_time": "2025-12-01T00:00:00Z",
                "version_id": "eps_old",
                "raw_payload_hash": "h1",
                "metric": "eps",
                "period": "FY1",
                "consensus_value": 5.0,
                "num_estimates": 12,
                "revision_direction": None,
                "revision_magnitude": None,
                "period_end": "2026-12-31T00:00:00Z",
            },
            {
                "source_system": "refinitiv_estimates",
                "entity_id": "ABC",
                "company_id": "ABC",
                "security_id": "ABC",
                "event_time": "2026-02-01T00:00:00Z",
                "available_time": "2026-02-01T00:00:00Z",
                "ingestion_time": "2026-02-01T00:00:00Z",
                "version_id": "eps_new",
                "raw_payload_hash": "h2",
                "metric": "eps",
                "period": "FY1",
                "consensus_value": 5.5,
                "num_estimates": 14,
                "revision_direction": "up",
                "revision_magnitude": 0.1,
                "period_end": "2026-12-31T00:00:00Z",
            },
            {
                "source_system": "refinitiv_estimates",
                "entity_id": "ABC",
                "company_id": "ABC",
                "security_id": "ABC",
                "event_time": "2025-12-01T00:00:00Z",
                "available_time": "2025-12-01T00:00:00Z",
                "ingestion_time": "2025-12-01T00:00:00Z",
                "version_id": "rev_old",
                "raw_payload_hash": "h3",
                "metric": "revenue",
                "period": "FY1",
                "consensus_value": 1000.0,
                "num_estimates": 10,
                "revision_direction": None,
                "revision_magnitude": None,
                "period_end": "2026-12-31T00:00:00Z",
            },
            {
                "source_system": "refinitiv_estimates",
                "entity_id": "ABC",
                "company_id": "ABC",
                "security_id": "ABC",
                "event_time": "2026-02-01T00:00:00Z",
                "available_time": "2026-02-01T00:00:00Z",
                "ingestion_time": "2026-02-01T00:00:00Z",
                "version_id": "rev_new",
                "raw_payload_hash": "h4",
                "metric": "revenue",
                "period": "FY1",
                "consensus_value": 1050.0,
                "num_estimates": 11,
                "revision_direction": "up",
                "revision_magnitude": 0.05,
                "period_end": "2026-12-31T00:00:00Z",
            },
            {
                "source_system": "refinitiv_estimates",
                "entity_id": "ABC",
                "company_id": "ABC",
                "security_id": "ABC",
                "event_time": "2025-12-15T00:00:00Z",
                "available_time": "2025-12-15T00:00:00Z",
                "ingestion_time": "2025-12-15T00:00:00Z",
                "version_id": "ebitda_old",
                "raw_payload_hash": "h5",
                "metric": "ebitda",
                "period": "FY1",
                "consensus_value": 200.0,
                "num_estimates": 8,
                "revision_direction": None,
                "revision_magnitude": None,
                "period_end": "2026-12-31T00:00:00Z",
            },
            {
                "source_system": "refinitiv_estimates",
                "entity_id": "ABC",
                "company_id": "ABC",
                "security_id": "ABC",
                "event_time": "2026-02-10T00:00:00Z",
                "available_time": "2026-02-10T00:00:00Z",
                "ingestion_time": "2026-02-10T00:00:00Z",
                "version_id": "ebitda_new",
                "raw_payload_hash": "h6",
                "metric": "ebitda",
                "period": "FY1",
                "consensus_value": 190.0,
                "num_estimates": 9,
                "revision_direction": "down",
                "revision_magnitude": 0.05,
                "period_end": "2026-12-31T00:00:00Z",
            },
        ],
    )
    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        estimates_path=estimates_path,
        skip_timeseries=True,
        skip_events=True,
        skip_peer_context=True,
    )
    snap = builder.build("ABC", "2026-02-28")

    assert snap.features["expectations.eps_consensus_fy1"]["value"] == pytest.approx(5.5)
    assert snap.features["expectations.revenue_consensus_fy1"]["value"] == pytest.approx(1050.0)
    assert snap.features["expectations.ebitda_consensus_fy1"]["value"] == pytest.approx(190.0)
    assert snap.features["expectations.analyst_coverage_count"]["value"] == pytest.approx(14.0)
    assert snap.features["expectations.eps_revision_score_90d"]["value"] == pytest.approx(0.1)
    assert snap.features["expectations.revenue_revision_score_90d"]["value"] == pytest.approx(0.05)
    assert snap.features["expectations.ebitda_revision_score_90d"]["value"] == pytest.approx(-0.05)
    assert snap.features["expectations.revision_signal"]["value"] == pytest.approx((0.1 + 0.05 - 0.05) / 3.0)


def test_capital_return_support_features_from_share_history_and_balance_sheet(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ts_path = tmp_path / "timeseries.parquet"
    _write_parquet(
        facts_path,
        [
            {
                **_facts_row(
                    "cash",
                    "ABC",
                    "financial.cash",
                    250.0,
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                ),
                "period_end": "2025-12-31T00:00:00Z",
            },
            {
                **_facts_row(
                    "revenue",
                    "ABC",
                    "financial.revenue",
                    1000.0,
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                ),
                "period_end": "2025-12-31T00:00:00Z",
            },
            {
                **_facts_row(
                    "ebitda",
                    "ABC",
                    "financial.ebitda",
                    200.0,
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                ),
                "period_end": "2025-12-31T00:00:00Z",
            },
            {
                **_facts_row(
                    "fcf",
                    "ABC",
                    "financial.free_cash_flow",
                    160.0,
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                ),
                "period_end": "2025-12-31T00:00:00Z",
            },
            {
                **_facts_row(
                    "debt",
                    "ABC",
                    "financial.total_debt",
                    300.0,
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                ),
                "period_end": "2025-12-31T00:00:00Z",
            },
            {
                **_facts_row(
                    "shares_old",
                    "ABC",
                    "financial.shares_basic",
                    110.0,
                    "2025-02-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                ),
                "period_end": "2024-12-31T00:00:00Z",
            },
            {
                **_facts_row(
                    "shares_new",
                    "ABC",
                    "financial.shares_basic",
                    100.0,
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                    "2026-02-01T00:00:00Z",
                ),
                "period_end": "2025-12-31T00:00:00Z",
            },
        ],
    )
    _write_parquet(
        ts_path,
        [
            {
                "entity_id": "ABC",
                "observation_time": "2026-02-20T00:00:00Z",
                "market_cap": 1000.0,
            }
        ],
    )
    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        timeseries_path=ts_path,
        skip_timeseries=False,
        skip_events=True,
    )
    snap = builder.build("ABC", "2026-02-28")

    share_trend = snap.features["capital_return.share_count_trend"]
    assert share_trend["value"] == pytest.approx((100.0 / 110.0) - 1.0, rel=1e-3)
    assert share_trend["primary_source_basis"] == "share_count_fact_history"

    buyback_capacity = snap.features["capital_return.buyback_capacity_proxy"]
    assert buyback_capacity["value"] == pytest.approx(0.20)
    assert buyback_capacity["primary_source_basis"] == "capital_return_capacity_house_formula"


def test_fact_revision_dedup_keeps_history(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    rows = [
        {
            "fact_id": "rev_old",
            "entity_id": "ABC",
            "fact_type": "financial.revenue",
            "fact_value": 100.0,
            "fact_time": "2025-12-31T00:00:00Z",
            "confidence_score": 0.8,
            "source_type": "SEC",
            "published_at": "2026-01-15T00:00:00Z",
            "ingested_at": "2026-01-16T00:00:00Z",
            "valid_from": "2026-01-15T00:00:00Z",
            "valid_to": None,
        },
        {
            "fact_id": "rev_new",
            "entity_id": "ABC",
            "fact_type": "financial.revenue",
            "fact_value": 120.0,
            "fact_time": "2025-12-31T00:00:00Z",
            "confidence_score": 0.9,
            "source_type": "SEC",
            "published_at": "2026-01-20T00:00:00Z",
            "ingested_at": "2026-01-21T00:00:00Z",
            "valid_from": "2026-01-20T00:00:00Z",
            "valid_to": None,
        },
        {
            "fact_id": "prior_year_same_q",
            "entity_id": "ABC",
            "fact_type": "financial.revenue",
            "fact_value": 90.0,
            "fact_time": "2024-12-31T00:00:00Z",
            "confidence_score": 0.9,
            "source_type": "SEC",
            "published_at": "2025-01-20T00:00:00Z",
            "ingested_at": "2025-01-21T00:00:00Z",
            "valid_from": "2025-01-20T00:00:00Z",
            "valid_to": None,
        },
    ]
    _write_parquet(facts_path, rows)
    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-02-28")
    # Uses the closest same-quarter prior-year revenue, not the immediately preceding quarter.
    feature = snap.features["operating.revenue_yoy_last_q"]
    assert round(feature["value"], 6) == round((120.0 - 90.0) / 90.0, 6)
    assert feature["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert feature["component_breakdown"]["latest_revenue"] == 120.0
    assert feature["component_breakdown"]["prior_revenue"] == 90.0
    assert feature["component_breakdown"]["matching_window_days"] == [270, 460]


def test_revenue_yoy_uses_context_period_end_same_quarter_pair(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            {
                **_facts_row("rev_q1_prior", "ABC", "financial.revenue", 62.151, "2026-01-15T00:00:00Z", "2026-01-16T00:00:00Z", "2026-01-15T00:00:00Z"),
                "effective_at": "2024-11-24T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2024-11-24 00:00:00; fiscal_year=2026; fiscal_quarter=1",
            },
            {
                **_facts_row("rev_q1_current", "ABC", "financial.revenue", 67.307, "2026-01-15T00:00:00Z", "2026-01-16T00:00:00Z", "2026-01-15T00:00:00Z"),
                "effective_at": "2025-11-23T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2025-11-23 00:00:00; fiscal_year=2026; fiscal_quarter=1",
            },
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-02-28")

    feature = snap.features["operating.revenue_yoy_last_q"]
    assert round(feature["value"], 6) == round((67.307 - 62.151) / 62.151, 6)
    assert feature["component_breakdown"]["match_basis"] == "fiscal_quarter_period_end"
    assert feature["component_breakdown"]["latest_period"] == "2025-11-23 00:00:00+00:00"
    assert feature["component_breakdown"]["prior_period"] == "2024-11-24 00:00:00+00:00"


def test_revenue_yoy_normalizes_mixed_ytd_and_quarter_reporting(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            {
                **_facts_row("rev_q1_prior", "ABC", "financial.revenue", 95.0, "2025-05-10T00:00:00Z", "2025-05-11T00:00:00Z", "2025-05-10T00:00:00Z"),
                "effective_at": "2024-03-31T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2024-03-31 00:00:00; fiscal_year=2025; fiscal_quarter=1",
            },
            {
                **_facts_row("rev_q2_prior", "ABC", "financial.revenue", 200.0, "2025-08-10T00:00:00Z", "2025-08-11T00:00:00Z", "2025-08-10T00:00:00Z"),
                "effective_at": "2024-06-30T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2024-06-30 00:00:00; fiscal_year=2025; fiscal_quarter=2",
            },
            {
                **_facts_row("rev_q3_prior", "ABC", "financial.revenue", 290.0, "2025-11-10T00:00:00Z", "2025-11-11T00:00:00Z", "2025-11-10T00:00:00Z"),
                "effective_at": "2024-09-30T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2024-09-30 00:00:00; fiscal_year=2025; fiscal_quarter=3",
            },
            {
                **_facts_row("rev_q1_current", "ABC", "financial.revenue", 100.0, "2026-05-10T00:00:00Z", "2026-05-11T00:00:00Z", "2026-05-10T00:00:00Z"),
                "effective_at": "2025-03-31T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2025-03-31 00:00:00; fiscal_year=2026; fiscal_quarter=1",
            },
            {
                **_facts_row("rev_q2_current", "ABC", "financial.revenue", 210.0, "2026-08-10T00:00:00Z", "2026-08-11T00:00:00Z", "2026-08-10T00:00:00Z"),
                "effective_at": "2025-06-30T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2025-06-30 00:00:00; fiscal_year=2026; fiscal_quarter=2",
            },
            {
                **_facts_row("rev_q3_current", "ABC", "financial.revenue", 110.0, "2026-11-10T00:00:00Z", "2026-11-11T00:00:00Z", "2026-11-10T00:00:00Z"),
                "effective_at": "2025-09-30T00:00:00Z",
                "context_norm": "statement_type=income; fiscal_period_end=2025-09-30 00:00:00; fiscal_year=2026; fiscal_quarter=3",
            },
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-12-01")

    feature = snap.features["operating.revenue_yoy_last_q"]
    assert round(feature["value"], 6) == round((110.0 - (290.0 - 200.0)) / (290.0 - 200.0), 6)
    assert feature["component_breakdown"]["latest_value_basis"] == "as_reported_quarter"
    assert feature["component_breakdown"]["prior_value_basis"] == "derived_from_ytd_delta"
    assert feature["support_mode"] == "proxy_missing_component"
    assert "quarter_value_derived_from_ytd_delta" in (feature["quality_flags"] or [])
    assert "mixed_quarter_value_basis" in (feature["quality_flags"] or [])


def test_revenue_cagr_3y_uses_closest_three_year_observation(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("rev_2022", "ABC", "financial.revenue", 100.0, "2023-01-20T00:00:00Z", "2023-01-21T00:00:00Z", "2023-01-20T00:00:00Z"),
            _facts_row("rev_2023", "ABC", "financial.revenue", 110.0, "2024-01-20T00:00:00Z", "2024-01-21T00:00:00Z", "2024-01-20T00:00:00Z"),
            _facts_row("rev_2025", "ABC", "financial.revenue", 133.1, "2026-01-20T00:00:00Z", "2026-01-21T00:00:00Z", "2026-01-20T00:00:00Z"),
        ],
    )

    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-02-28")

    feature = snap.features["operating.revenue_cagr_3y"]
    assert round(feature["value"], 4) == 0.1000
    assert feature["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert feature["component_breakdown"]["prior_revenue"] == 100.0
    assert feature["component_breakdown"]["latest_revenue"] == 133.1
    assert round(feature["component_breakdown"]["elapsed_years"], 2) == 3.00


def test_ownership_from_13f_summary_fallback(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ownership_path = tmp_path / "ownership.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row(
                "shares",
                "ABC",
                "financial.shares_out",
                100.0,
                "2026-02-01T00:00:00Z",
                "2026-02-01T00:00:00Z",
                "2026-02-01T00:00:00Z",
            ),
        ],
    )
    _write_parquet(
        ownership_path,
        [
            {
                "company_id": "ABC",
                "report_date": "2025-12-31T00:00:00Z",
                "filing_date": "2026-02-14T00:00:00Z",
                "total_13f_shares": 80.0,
                "top5_13f_shares": 40.0,
                "holder_count": 12.0,
                "total_13f_value_usd": 10_000_000.0,
                "published_at": "2026-02-14T00:00:00Z",
                "ingested_at": "2026-02-14T00:00:00Z",
                "effective_at": "2025-12-31T00:00:00Z",
                "source_type": "wrds_13f",
                "artifact_id": "own_abc_2025q4",
            }
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        ownership_path=ownership_path,
    )
    snap = builder.build("ABC", "2026-02-28")
    assert snap.features["ownership_governance.top5_holder_pct"]["value"] == 0.5
    assert snap.features["ownership_governance.institutional_pct"]["value"] == 0.8
    assert snap.features["ownership_governance.holder_count_13f"]["value"] == 12.0
    assert snap.features["ownership_governance.crowding_signal"]["value"] == pytest.approx(
        ((0.5 - 0.35) / 0.35 + (0.8 - 0.5) / 0.4 + (25.0 - 12.0) / 25.0) / 3.0
    )


def test_ownership_summary_prefers_richer_latest_quarter_snapshot(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    ownership_path = tmp_path / "ownership.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row(
                "shares",
                "ABC",
                "financial.shares_out",
                1_000.0,
                "2026-02-01T00:00:00Z",
                "2026-02-01T00:00:00Z",
                "2026-02-01T00:00:00Z",
            ),
        ],
    )
    _write_parquet(
        ownership_path,
        [
            {
                "company_id": "ABC",
                "report_date": "2025-12-31T00:00:00Z",
                "filing_date": "2026-02-14T00:00:00Z",
                "total_13f_shares": 10.0,
                "top5_13f_shares": 10.0,
                "holder_count": 1.0,
                "published_at": "2026-02-14T00:00:00Z",
                "ingested_at": "2026-02-14T00:00:00Z",
                "effective_at": "2025-12-31T00:00:00Z",
                "source_type": "wrds_13f",
                "artifact_id": "own_abc_thin",
            },
            {
                "company_id": "ABC",
                "report_date": "2025-12-31T00:00:00Z",
                "filing_date": "2026-02-13T00:00:00Z",
                "total_13f_shares": 800.0,
                "top5_13f_shares": 400.0,
                "holder_count": 50.0,
                "published_at": "2026-02-13T00:00:00Z",
                "ingested_at": "2026-02-13T00:00:00Z",
                "effective_at": "2025-12-31T00:00:00Z",
                "source_type": "wrds_13f",
                "artifact_id": "own_abc_rich",
            },
        ],
    )

    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        ownership_path=ownership_path,
    )
    snap = builder.build("ABC", "2026-02-28")
    assert snap.features["ownership_governance.top5_holder_pct"]["value"] == 0.5
    assert snap.features["ownership_governance.institutional_pct"]["value"] == 0.8
    assert snap.features["ownership_governance.holder_count_13f"]["value"] == 50.0


def test_rating_state_from_issuer_ratings_fallback(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    events_path = tmp_path / "events.parquet"
    ratings_path = tmp_path / "issuer_ratings.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row("cash", "ABC", "financial.cash", 100.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            _facts_row("debt", "ABC", "financial.total_debt", 500.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            _facts_row("ebitda", "ABC", "financial.ebitda", 50.0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        ],
    )
    _write_parquet(
        events_path,
        [
            {
                "event_id": "evt_other",
                "company_id": "OTHER",
                "event_type": "buyback",
                "event_subtype": None,
                "announced_at": "2026-01-10T00:00:00Z",
                "effective_at": "2026-01-10T00:00:00Z",
                "created_at": "2026-01-10T00:00:00Z",
                "source_type": "event_store",
                "params": None,
            }
        ],
    )
    _write_parquet(
        ratings_path,
        [
            {
                "company_id": "ABC",
                "rating_date": "2026-01-25T00:00:00Z",
                "rating_symbol": "BBB-",
                "current_rating_symbol": "BBB-",
                "rating_type_code": "LT",
                "outlook": "stable",
                "creditwatch": "N",
                "source_type": "moodys_ratings",
                "agency": "Moody's",
                "artifact_id": "rating_abc_moodys",
                "published_at": "2026-01-25T00:00:00Z",
                "ingested_at": "2026-01-25T00:00:00Z",
                "effective_at": "2026-01-25T00:00:00Z",
            },
            {
                "company_id": "ABC",
                "rating_date": "2026-01-20T00:00:00Z",
                "rating_symbol": "BB+",
                "current_rating_symbol": "BB+",
                "rating_type_code": "SPR",
                "outlook": "negative",
                "creditwatch": "Y",
                "source_type": "fitch_ratings",
                "agency": "Fitch",
                "artifact_id": "rating_abc_1",
                "published_at": "2026-01-20T00:00:00Z",
                "ingested_at": "2026-01-20T00:00:00Z",
                "effective_at": "2026-01-20T00:00:00Z",
            }
        ],
    )
    builder = _base_builder(
        tmp_path,
        facts_path=facts_path,
        skip_timeseries=True,
        events_path=events_path,
        skip_events=False,
        issuer_ratings_path=ratings_path,
    )
    snap = builder.build("ABC", "2026-02-28")
    rating_state = snap.features["capital_structure.rating_state"]["value"]
    assert rating_state["rating"] == "BB+"
    assert rating_state["watchlist"] is True


def test_confidence_framework_proxy_penalty(tmp_path: Path):
    facts_path = tmp_path / "facts.parquet"
    _write_parquet(
        facts_path,
        [
            _facts_row(
                "cash",
                "ABC",
                "financial.cash",
                100.0,
                "2026-02-20T00:00:00Z",
                "2026-02-20T00:00:00Z",
                "2026-02-20T00:00:00Z",
            ),
            _facts_row(
                "revenue",
                "ABC",
                "financial.revenue",
                1000.0,
                "2026-02-20T00:00:00Z",
                "2026-02-20T00:00:00Z",
                "2026-02-20T00:00:00Z",
            ),
        ],
    )
    builder = _base_builder(tmp_path, facts_path=facts_path, skip_timeseries=True)
    snap = builder.build("ABC", "2026-02-28")
    cash_conf = snap.features["liquidity.cash"]["confidence"]
    proxy_conf = snap.features["liquidity.minimum_cash_policy_proxy"]["confidence"]
    assert cash_conf is not None and 0.0 <= cash_conf <= 1.0
    assert proxy_conf is not None and 0.0 <= proxy_conf <= 1.0
    assert proxy_conf < cash_conf
