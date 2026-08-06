from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.named_company_metric_benchmarks import generate_named_company_metric_benchmarks


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_parquet(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_named_company_benchmark_marks_dataless_snapshot_blocked(tmp_path: Path):
    targets_path = tmp_path / 'targets.json'
    fundamentals_path = tmp_path / 'fundamentals.parquet'
    snapshot_root = tmp_path / 'snapshots'

    _write_json(
        targets_path,
        {
            'metadata': {},
            'targets': [
                {
                    'case_id': 'wmt',
                    'company_id': '0000104169',
                    'ticker': 'WMT',
                    'display_name': 'Walmart Inc',
                    'as_of_date': '2026-02-28',
                }
            ],
        },
    )
    _write_parquet(
        fundamentals_path,
        [
            {
                'Instrument': 'WMT.N',
                'Company Common Name': 'Walmart Inc',
                'GICS Sector Name': 'Consumer Staples',
                'GICS Industry Name': 'Consumer Staples Distribution & Retail',
            }
        ],
    )
    blocked_path = snapshot_root / 'as_of_date=2026-02-28' / 'company_id=0000104169.json'
    blocked_path.parent.mkdir(parents=True, exist_ok=True)
    blocked_path.touch()

    report = generate_named_company_metric_benchmarks(
        targets_path,
        snapshot_root=snapshot_root,
        fundamentals_path=fundamentals_path,
    )

    assert report['summary']['blocked_dataless_snapshots'] == 1
    result = report['results'][0]
    assert result['benchmark_status'] == 'blocked_dataless_snapshot'
    assert result['expected_archetype'] == 'consumer_grocery_retail'


def test_named_company_benchmark_reads_materialized_snapshot_metrics(tmp_path: Path):
    targets_path = tmp_path / 'targets.json'
    fundamentals_path = tmp_path / 'fundamentals.parquet'
    snapshot_root = tmp_path / 'snapshots'

    _write_json(
        targets_path,
        {
            'metadata': {},
            'targets': [
                {
                    'case_id': 'ge',
                    'company_id': '0000040545',
                    'ticker': 'GE',
                    'display_name': 'General Electric Co',
                    'as_of_date': '2026-02-28',
                }
            ],
        },
    )
    _write_parquet(
        fundamentals_path,
        [
            {
                'Instrument': 'GE.N',
                'Company Common Name': 'General Electric Co',
                'GICS Sector Name': 'Industrials',
                'GICS Industry Name': 'Aerospace & Defense',
            }
        ],
    )

    snapshot_path = snapshot_root / 'as_of_date=2026-02-28' / 'company_id=0000040545.json'
    _write_json(
        snapshot_path,
        {
            'company_id': '0000040545',
            'as_of_time': '2026-02-28T00:00:00+00:00',
            'features': {
                'capital_structure.total_debt_market': {
                    'value': 460.0,
                    'unit': 'usd',
                    'support_mode': 'exact',
                    'applicability_status': 'primary',
                    'canonical_owner_id': 'fitch_ratings',
                    'canonical_classification': 'canonical_external',
                    'market_layer_status': 'keep',
                    'current_alignment_status': 'partial_proxy',
                    'primary_source_document_id': 'fitch_corporate_rating_criteria_2024',
                    'methodology_registry_id': 'consumer_industrials_metric_methodology_registry_v1',
                    'input_source_registry_id': 'company_state_input_source_registry_v1',
                    'input_source_owner_name': 'Fitch credit methodology',
                    'input_source_classification': 'canonical_external',
                    'definition_requirement': 'must_have_external_definition',
                    'definition_requirement_reason': 'This is a first-order balance-sheet or credit input the market underwrites directly, so Axiom should follow a named external definition rather than a house proxy.',
                    'methodology_execution_decision': 'adopt_exact_external_methodology',
                    'methodology_execution_reason': 'A named external owner or official filing regime exists for this metric, so Axiom should follow that methodology directly rather than preserving a house definition.',
                    'input_layer_bucket': 'strict_market_defined',
                    'input_layer_bucket_reason': 'This metric belongs to the strict market-defined input layer because it follows a named trusted external methodology, filing-native definition, or standardized public-market definition.',
                    'strict_market_defined': True,
                    'missing_reason': None,
                    'fallback_used': None,
                    'quality_flags': [],
                    'component_breakdown': {'reported_debt': 400.0, 'pension_weight': 0.5},
                },
                'capital_structure.interest_coverage': {
                    'value': 4.2,
                    'unit': 'x',
                    'support_mode': 'exact',
                    'applicability_status': 'primary',
                    'missing_reason': None,
                    'fallback_used': None,
                    'quality_flags': [],
                    'component_breakdown': {'ebit': 84.0, 'interest_expense': 20.0},
                },
            },
            'provenance': {
                'market_metric_context': {
                    'archetype': 'aerospace_defense',
                    'support_mode': 'exact',
                    'methodology_registry_id': 'consumer_industrials_metric_methodology_registry_v1',
                }
            },
        },
    )

    report = generate_named_company_metric_benchmarks(
        targets_path,
        snapshot_root=snapshot_root,
        fundamentals_path=fundamentals_path,
    )

    assert report['summary']['ready_packets'] == 1
    result = report['results'][0]
    assert result['benchmark_status'] == 'ready'
    assert result['metrics']['capital_structure.total_debt_market']['value'] == 460.0
    assert result['metrics']['capital_structure.total_debt_market']['canonical_owner_id'] == 'fitch_ratings'
    assert result['metrics']['capital_structure.total_debt_market']['canonical_classification'] == 'canonical_external'
    assert result['metrics']['capital_structure.total_debt_market']['input_source_owner_name'] == 'Fitch credit methodology'
    assert result['metrics']['capital_structure.total_debt_market']['definition_requirement'] == 'must_have_external_definition'
    assert result['metrics']['capital_structure.total_debt_market']['methodology_execution_decision'] == 'adopt_exact_external_methodology'
    assert result['metrics']['capital_structure.total_debt_market']['input_layer_bucket'] == 'strict_market_defined'
    assert result['metrics']['capital_structure.total_debt_market']['strict_market_defined'] is True
    assert result['market_metric_context']['archetype'] == 'aerospace_defense'
    assert result['market_metric_context']['methodology_registry_id'] == 'consumer_industrials_metric_methodology_registry_v1'
