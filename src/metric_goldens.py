from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional
import json

import pandas as pd

from .company_state_builder import CompanyStateBuilder
from .company_state_validation import check_invariants
from .data_paths import resolve_companyfacts_root


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDENS_PATH = ROOT / "configs" / "metric_goldens" / "consumer_industrials_wave1.json"


def _default_companyfacts_root() -> Optional[Path]:
    return resolve_companyfacts_root(ROOT / "data" / "sec" / "companyfacts")

def _clean_case_ids(case_ids: Optional[Iterable[str]]) -> Optional[set[str]]:
    if case_ids is None:
        return None
    values = {str(case_id).strip() for case_id in case_ids if str(case_id).strip()}
    return values or None


def load_metric_goldens(path: Path | str | None = None) -> Dict[str, Any]:
    goldens_path = Path(path) if path is not None else DEFAULT_GOLDENS_PATH
    payload = json.loads(goldens_path.read_text())
    if isinstance(payload, list):
        return {"metadata": {}, "cases": payload, "path": str(goldens_path)}
    return {
        "metadata": dict(payload.get("metadata") or {}),
        "cases": list(payload.get("cases") or []),
        "path": str(goldens_path),
    }


def _write_parquet_if_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


