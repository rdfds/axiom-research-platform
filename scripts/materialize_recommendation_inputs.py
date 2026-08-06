#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def _copy_file(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _materialize_keyed_snapshots(snapshot_root: Path, as_of: str, companies: List[str], dest_root: Path) -> Path:
    dest_snapshot_root = dest_root / "snapshot_root"
    dest_keyed_dir = dest_snapshot_root / "keyed" / f"as_of_date={as_of}"
    dest_keyed_dir.mkdir(parents=True, exist_ok=True)
    for company_id in companies:
        src = snapshot_root / "keyed" / f"as_of_date={as_of}" / f"company_id={company_id}.json"
        dst = dest_keyed_dir / f"company_id={company_id}.json"
        _copy_file(src, dst)
    return dest_snapshot_root


def _maybe_copy(path_value: Optional[str], dest_dir: Path) -> Optional[str]:
    if not path_value:
        return None
    src = Path(path_value)
    dst = dest_dir / src.name
    return str(_copy_file(src, dst))


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize recommendation inputs into a local destination such as /tmp.")
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--companies", nargs="+", required=True)
    parser.add_argument("--entity-graph-path", required=True)
    parser.add_argument("--entity-identifier-path", required=True)
    parser.add_argument("--outcomes-path", required=True)
    parser.add_argument("--causal-model-path", required=True)
    parser.add_argument("--causal-action-blocklist-path")
    parser.add_argument("--config-path")
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--out-manifest", required=True)
    args = parser.parse_args()

    dest_root = Path(args.dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    local_snapshot_root = _materialize_keyed_snapshots(
        snapshot_root=Path(args.snapshot_root),
        as_of=str(args.as_of),
        companies=[str(company_id) for company_id in args.companies],
        dest_root=dest_root,
    )
    local_inputs_dir = dest_root / "inputs"
    local_models_dir = dest_root / "models"
    local_config_dir = dest_root / "config"

    manifest: Dict[str, object] = {
        "snapshot_root": str(local_snapshot_root),
        "companies": [str(company_id) for company_id in args.companies],
        "entity_graph_path": _maybe_copy(args.entity_graph_path, local_inputs_dir),
        "entity_identifier_path": _maybe_copy(args.entity_identifier_path, local_inputs_dir),
        "outcomes_path": _maybe_copy(args.outcomes_path, local_inputs_dir),
        "causal_model_path": _maybe_copy(args.causal_model_path, local_models_dir),
        "causal_action_blocklist_path": _maybe_copy(args.causal_action_blocklist_path, local_config_dir),
        "config_path": _maybe_copy(args.config_path, local_config_dir),
    }

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"ok": True, "manifest": str(out_manifest), **manifest}))


if __name__ == "__main__":
    main()
