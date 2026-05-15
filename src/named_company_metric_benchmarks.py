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


