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


