#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_CONFIG = ROOT / "configs" / "historical_eval_manifests" / "2026-03-17" / "manual_replay_benchmark_lock.json"
DEFAULT_BUNDLE_ROOT = ROOT / "out" / "manual_replay_bundle_20260405"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a manual replay benchmark bundle into a durable workspace path.")
    parser.add_argument("--lock-config", default=str(DEFAULT_LOCK_CONFIG))
    parser.add_argument("--bundle-root", default=str(DEFAULT_BUNDLE_ROOT))
    parser.add_argument("--include-feedback-model", action="store_true", help="Also freeze the v7 feedback HGB model artifacts.")
    return parser.parse_args()


def _resolve_root_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _resolve_candidate_path(values: Iterable[str | Path]) -> Path:
    candidates = [_resolve_root_path(value) for value in values]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No candidate path exists for {list(values)}")


def _sha256_head(path: Path, limit_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = limit_bytes
        while remaining > 0:
            chunk = handle.read(min(remaining, 1024 * 1024))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _path_metadata(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        info["kind"] = "directory" if path.is_dir() else "file"
        info["size_bytes"] = stat.st_size
        info["modified_at"] = stat.st_mtime
        if path.is_file():
            info["sample_sha256"] = _sha256_head(path)
    return info


