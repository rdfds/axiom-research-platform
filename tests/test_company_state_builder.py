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


