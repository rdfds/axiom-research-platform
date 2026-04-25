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


