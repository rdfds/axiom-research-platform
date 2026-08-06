from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import os

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS_PATH = ROOT / 'configs' / 'metric_goldens' / 'consumer_industrials_named_targets.json'
DEFAULT_SNAPSHOT_ROOT = ROOT / 'data' / 'company_state_snapshots' / 'final_run_2026-02-28' / 'keyed'
DEFAULT_ENTITY_IDENTIFIER_PATH = ROOT / 'data' / 'inputs_layer' / 'entity_identifier.parquet'
DEFAULT_FUNDAMENTALS_PATH = ROOT / 'data' / 'refinitiv' / 'fundamentals_all.parquet'


def load_named_company_targets(path: Path | str | None = None) -> Dict[str, Any]:
    targets_path = Path(path) if path is not None else DEFAULT_TARGETS_PATH
    payload = json.loads(targets_path.read_text())
    return {
        'metadata': dict(payload.get('metadata') or {}),
        'targets': list(payload.get('targets') or []),
        'path': str(targets_path),
    }


def _clean_case_ids(case_ids: Optional[Iterable[str]]) -> Optional[set[str]]:
    if case_ids is None:
        return None
    values = {str(case_id).strip() for case_id in case_ids if str(case_id).strip()}
    return values or None


def _snapshot_path(snapshot_root: Path, company_id: str, as_of_date: str) -> Path:
    return snapshot_root / f'as_of_date={as_of_date}' / f'company_id={company_id}.json'


def _snapshot_materialization(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {'exists': False, 'materialized': False, 'size': None, 'blocks': None}
    stat_result = os.stat(path)
    blocks = getattr(stat_result, 'st_blocks', None)
    return {
        'exists': True,
        'materialized': bool(blocks and blocks > 0),
        'size': stat_result.st_size,
        'blocks': blocks,
    }


def _fundamentals_context(ticker: str, fundamentals_path: Path) -> Dict[str, Any]:
    import duckdb

    con = duckdb.connect()
    escaped_path = fundamentals_path.as_posix().replace("'", "''")
    normalized = ''.join(ch for ch in ticker.upper() if ch.isalnum())
    rows = con.execute(
        f'''
        select Instrument, "Company Common Name", "GICS Sector Name", "GICS Industry Name"
        from read_parquet('{escaped_path}', union_by_name=true)
        where upper(regexp_replace(split_part(Instrument,'.',1), '[^A-Za-z0-9]', '', 'g')) = '{normalized}'
        limit 1
        '''
    ).fetchall()
    if not rows:
        return {}
    instrument, company_name, sector, industry = rows[0]
    return {
        'instrument': instrument,
        'company_name': company_name,
        'sector': sector,
        'subsector': industry,
    }


def _entity_context_for_policy(target: Dict[str, Any], fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'sector': target.get('sector') or fundamentals.get('sector'),
        'gics_sector': target.get('sector') or fundamentals.get('sector'),
        'subsector': target.get('subsector') or fundamentals.get('subsector'),
        'gics_sub_industry': target.get('subsector') or fundamentals.get('subsector'),
        'industry': target.get('subsector') or fundamentals.get('subsector'),
        'sic': target.get('sic'),
    }


def _metric_excerpt(features: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    feat = dict(features.get(name) or {})
    if not feat:
        return None
    return {
        'value': feat.get('value'),
        'unit': feat.get('unit'),
        'support_mode': feat.get('support_mode'),
        'applicability_status': feat.get('applicability_status'),
        'canonical_owner_id': feat.get('canonical_owner_id'),
        'canonical_classification': feat.get('canonical_classification'),
        'market_layer_status': feat.get('market_layer_status'),
        'current_alignment_status': feat.get('current_alignment_status'),
        'primary_source_document_id': feat.get('primary_source_document_id'),
        'methodology_registry_id': feat.get('methodology_registry_id'),
        'input_source_registry_id': feat.get('input_source_registry_id'),
        'input_source_owner_id': feat.get('input_source_owner_id'),
        'input_source_owner_name': feat.get('input_source_owner_name'),
        'input_source_classification': feat.get('input_source_classification'),
        'input_source_formula_basis': feat.get('input_source_formula_basis'),
        'input_source_alignment_status': feat.get('input_source_alignment_status'),
        'input_source_document_ids': feat.get('input_source_document_ids'),
        'definition_requirement': feat.get('definition_requirement'),
        'definition_requirement_reason': feat.get('definition_requirement_reason'),
        'methodology_execution_decision': feat.get('methodology_execution_decision'),
        'methodology_execution_reason': feat.get('methodology_execution_reason'),
        'input_layer_bucket': feat.get('input_layer_bucket'),
        'input_layer_bucket_reason': feat.get('input_layer_bucket_reason'),
        'strict_market_defined': feat.get('strict_market_defined'),
        'missing_reason': feat.get('missing_reason'),
        'fallback_used': feat.get('fallback_used'),
        'quality_flags': feat.get('quality_flags'),
        'component_breakdown': feat.get('component_breakdown'),
    }


def _snapshot_metric_packet(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    features = dict(snapshot.get('features') or {})
    metric_names = [
        'capital_structure.total_debt_reported',
        'capital_structure.total_debt_market',
        'capital_structure.total_debt',
        'capital_structure.net_debt_reported',
        'capital_structure.net_debt_market',
        'capital_structure.net_debt',
        'capital_structure.gross_leverage_market',
        'capital_structure.net_leverage_market',
        'capital_structure.interest_coverage',
        'capital_structure.fixed_charge_coverage',
        'market.pe_ratio',
        'market.pe_percentile_peers',
        'market.pe_percentile_history',
        'macro.sp500_pe_ttm',
        'macro.sp500_pe_ttm_percentile_history',
        'macro.us10y_treasury_yield',
        'macro.us10y_treasury_yield_percentile_history',
        'macro.us_ig_oas',
        'macro.us_ig_oas_percentile_history',
        'macro.us_hy_all_in_yield',
        'macro.us_hy_all_in_yield_percentile_history',
        'macro.real_gdp_growth_yoy',
        'macro.real_gdp_growth_yoy_percentile_history',
        'liquidity.available_for_actions_market',
        'liquidity.runway_months',
        'capital_structure.maturity_wall_ratio_24m',
        'capital_structure.refi_pressure_flag',
    ]
    return {
        'market_metric_context': dict((snapshot.get('provenance') or {}).get('market_metric_context') or {}),
        'metrics': {
            name: _metric_excerpt(features, name)
            for name in metric_names
            if _metric_excerpt(features, name) is not None
        },
    }


def generate_named_company_metric_benchmarks(
    targets_path: Path | str | None = None,
    *,
    snapshot_root: Path | str = DEFAULT_SNAPSHOT_ROOT,
    fundamentals_path: Path | str = DEFAULT_FUNDAMENTALS_PATH,
    case_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    payload = load_named_company_targets(targets_path)
    selected_case_ids = _clean_case_ids(case_ids)
    targets = [
        target for target in payload['targets']
        if selected_case_ids is None or str(target.get('case_id')) in selected_case_ids
    ]

    snapshot_root_path = Path(snapshot_root)
    fundamentals_path = Path(fundamentals_path)
    results: List[Dict[str, Any]] = []
    summary = {
        'total_targets': len(targets),
        'ready_packets': 0,
        'blocked_dataless_snapshots': 0,
        'missing_snapshots': 0,
    }

    for target in targets:
        ticker = str(target.get('ticker') or '').strip()
        company_id = str(target.get('company_id') or '').strip()
        as_of_date = str(target.get('as_of_date') or '').strip()
        has_embedded_context = any(
            target.get(field)
            for field in ('sector', 'subsector')
        )
        fundamentals = {}
        if ticker and not has_embedded_context:
            fundamentals = _fundamentals_context(ticker, fundamentals_path)
        expected_archetype = target.get('expected_archetype')
        rules = dict(target.get('policy_rules') or {})
        taxonomy_support_mode = str(target.get('taxonomy_support_mode') or 'exact')
        taxonomy_override_level = str(target.get('taxonomy_override_level') or 'subsector')
        if not expected_archetype or not rules:
            from .metric_policy import MetricPolicyEngine

            policy = MetricPolicyEngine()
            entity_context = _entity_context_for_policy(target, fundamentals)
            taxonomy = policy.resolve_taxonomy(company_id, entity_row=entity_context, fingerprints={})
            expected_archetype = expected_archetype or taxonomy.archetype
            if not rules:
                rules = policy.archetype_rules(str(expected_archetype))
            if 'taxonomy_support_mode' not in target:
                taxonomy_support_mode = taxonomy.support_mode
            if 'taxonomy_override_level' not in target:
                taxonomy_override_level = taxonomy.override_level_applied
        snapshot_path = _snapshot_path(snapshot_root_path, company_id, as_of_date)
        materialization = _snapshot_materialization(snapshot_path)
        result: Dict[str, Any] = {
            'case_id': target.get('case_id'),
            'company_id': company_id,
            'ticker': ticker,
            'display_name': target.get('display_name') or fundamentals.get('company_name'),
            'as_of_date': as_of_date,
            'sector': fundamentals.get('sector') or target.get('sector'),
            'subsector': fundamentals.get('subsector') or target.get('subsector'),
            'expected_archetype': expected_archetype,
            'taxonomy_support_mode': taxonomy_support_mode,
            'taxonomy_override_level': taxonomy_override_level,
            'policy_rules': rules,
            'snapshot_path': str(snapshot_path),
            'snapshot_exists': materialization['exists'],
            'snapshot_materialized': materialization['materialized'],
            'snapshot_blocks': materialization['blocks'],
            'benchmark_status': 'ready' if materialization['materialized'] else ('missing_snapshot' if not materialization['exists'] else 'blocked_dataless_snapshot'),
        }
        if not materialization['exists']:
            summary['missing_snapshots'] += 1
        elif not materialization['materialized']:
            summary['blocked_dataless_snapshots'] += 1
        else:
            snapshot = json.loads(snapshot_path.read_text())
            result.update(_snapshot_metric_packet(snapshot))
            summary['ready_packets'] += 1
        results.append(result)

    return {
        'targets_path': payload['path'],
        'metadata': payload['metadata'],
        'summary': summary,
        'results': results,
    }
