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


