"""
SnapshotStore for CompanyStateSnapshot artifacts.
Provides simple JSONL and Parquet persistence for daily snapshots.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional
import json
import os
import shutil

import numpy as np
import pandas as pd

from .company_state_builder import CompanyStateSnapshot


class SnapshotStore:
    def __init__(self, root: str | Path = "data/company_state_snapshots", temp_dir: str | Path | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if temp_dir is None:
            temp_dir = os.environ.get("SNAPSHOT_TMP_DIR", "/tmp/company_state_snapshots")
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _stage_path(self, out: Path) -> Path:
        return self.temp_dir / out.name

    def _finalize(self, staged: Path, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(staged), str(out))
        except Exception:
            # Fallback: copy then remove
            shutil.copy2(str(staged), str(out))
            try:
                staged.unlink()
            except Exception:
                pass

    def _normalize_as_of_date(self, as_of: str) -> str:
        return pd.to_datetime(as_of).strftime("%Y-%m-%d")

    def _to_snapshot_dict(self, snapshot: Any) -> dict:
        if is_dataclass(snapshot):
            row = asdict(snapshot)
        elif isinstance(snapshot, dict):
            row = snapshot
        else:
            raise TypeError(f"Unsupported snapshot type: {type(snapshot)}")
        return self._json_sanitize(row)

    def _keyed_dir(self, as_of: str) -> Path:
        date = self._normalize_as_of_date(as_of)
        out = self.root / "keyed" / f"as_of_date={date}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _keyed_path(self, company_id: str, as_of: str) -> Path:
        return self._keyed_dir(as_of) / f"company_id={company_id}.json"

    def write_jsonl(
        self,
        snapshots: Iterable[CompanyStateSnapshot],
        as_of: str,
        expected_count: int | None = None,
    ) -> Path:
        out = self.root / f"company_state_snapshots_asof={as_of}.jsonl"
        staged = self._stage_path(out)
        wrote = 0
        with staged.open("w") as f:
            for snap in snapshots:
                f.write(json.dumps(self._to_snapshot_dict(snap)) + "\n")
                wrote += 1
        if expected_count is not None and wrote != expected_count:
            try:
                staged.unlink()
            except Exception:
                pass
            raise RuntimeError(
                f"snapshot row-count mismatch: expected={expected_count} wrote={wrote}. "
                "Aborting to avoid partial overwrite."
            )
        self._finalize(staged, out)
        return out

    def write_parquet(
        self,
        snapshots: Iterable[CompanyStateSnapshot],
        as_of: str,
        expected_count: int | None = None,
    ) -> Path:
        out = self.root / f"company_state_snapshots_asof={as_of}.parquet"
        staged = self._stage_path(out)
        rows = [self._to_snapshot_dict(s) for s in snapshots]
        if expected_count is not None and len(rows) != expected_count:
            raise RuntimeError(
                f"snapshot row-count mismatch: expected={expected_count} wrote={len(rows)}. "
                "Aborting parquet write."
            )
        df = pd.DataFrame(rows)
        df.to_parquet(staged, index=False)
        self._finalize(staged, out)
        return out

    def write_keyed_json(
        self,
        snapshots: Iterable[Any],
        as_of: str,
        expected_count: int | None = None,
    ) -> Path:
        """Write one JSON file per (company_id, as_of_date)."""
        out_dir = self._keyed_dir(as_of)
        wrote = 0
        last_snapshot_time: Optional[str] = None
        for snap in snapshots:
            row = self._to_snapshot_dict(snap)
            cid = row.get("company_id")
            if cid is None:
                raise RuntimeError("snapshot missing company_id for keyed write")
            cid = str(cid)
            if row.get("as_of_time") is not None:
                last_snapshot_time = str(row.get("as_of_time"))
            out = self._keyed_path(cid, as_of)
            staged = self.temp_dir / f"{out.name}.tmp"
            with staged.open("w") as f:
                f.write(json.dumps(row) + "\n")
            self._finalize(staged, out)
            wrote += 1

        if expected_count is not None and wrote != expected_count:
            raise RuntimeError(
                f"keyed snapshot row-count mismatch: expected={expected_count} wrote={wrote}"
            )

        manifest = {
            "as_of_date": self._normalize_as_of_date(as_of),
            "rows": wrote,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "latest_snapshot_time": last_snapshot_time,
        }
        (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
        return out_dir

    def load_keyed_snapshot(self, company_id: str, as_of: str) -> Optional[dict]:
        path = self._keyed_path(str(company_id), as_of)
        if not path.exists():
            return None
        with path.open("r") as f:
            for line in f:
                if line.strip():
                    return json.loads(line)
        return None

    def iter_keyed_snapshots(self, as_of: str) -> Iterator[dict]:
        d = self._keyed_dir(as_of)
        for p in sorted(d.glob("company_id=*.json")):
            with p.open("r") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
                        break

    def upsert_keyed_snapshot(self, snapshot: Any, as_of: str) -> Path:
        row = self._to_snapshot_dict(snapshot)
        cid = row.get("company_id")
        if cid is None:
            raise RuntimeError("snapshot missing company_id for keyed upsert")
        out = self._keyed_path(str(cid), as_of)
        staged = self.temp_dir / f"{out.name}.tmp"
        with staged.open("w") as f:
            f.write(json.dumps(row) + "\n")
        self._finalize(staged, out)
        return out

    def load_jsonl(self, as_of: str) -> List[dict]:
        path = self.root / f"company_state_snapshots_asof={as_of}.jsonl"
        if not path.exists():
            return []
        out: List[dict] = []
        with path.open("r") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out

    def _json_sanitize(self, obj):
        if isinstance(obj, dict):
            return {k: self._json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._json_sanitize(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._json_sanitize(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, complex):
            # Preserve real values; drop if truly complex.
            return float(obj.real) if abs(obj.imag) < 1e-9 else None
        return obj
