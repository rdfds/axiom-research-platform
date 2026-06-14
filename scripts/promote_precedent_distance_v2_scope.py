#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_existing_runtime_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = _load_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a fitted precedent distance v2 scope into a runtime payload.")
    parser.add_argument("--source", required=True, help="Fit-output JSON produced by fit_precedent_distance_v2.py")
    parser.add_argument("--scope", required=True, help="Scope key to promote, e.g. capital_structure")
    parser.add_argument("--out-json", required=True, help="Runtime payload path to write")
    parser.add_argument("--default-enable", action="store_true", help="Mark the scope as default-enabled for runtime auto-selection")
    args = parser.parse_args()

    source_path = Path(args.source).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()
    payload = _load_json(source_path)
    scopes = payload.get("scopes") if isinstance(payload, dict) else None
    if not isinstance(scopes, dict):
        raise SystemExit(f"No scopes found in {source_path}")
    scope_key = str(args.scope).strip().lower()
    scope_payload = scopes.get(scope_key)
    if not isinstance(scope_payload, dict):
        raise SystemExit(f"Scope '{scope_key}' not found in {source_path}")

    promoted_scope = dict(scope_payload)
    promoted_scope["scope_key"] = scope_key
    promoted_scope["use_in_runtime"] = True
    if args.default_enable:
        promoted_scope["default_enabled"] = True

    existing_payload = _load_existing_runtime_payload(out_path)
    existing_scopes = dict(existing_payload.get("scopes") or {})
    existing_scopes[scope_key] = promoted_scope

    source_artifacts = []
    for value in [
        existing_payload.get("source_artifact"),
        str(source_path),
        *list(existing_payload.get("source_artifacts") or []),
    ]:
        value_str = str(value or "").strip()
        if not value_str or value_str in source_artifacts:
            continue
        source_artifacts.append(value_str)

    merged_notes = dict(existing_payload.get("notes") or {})
    merged_notes.update(dict(payload.get("notes") or {}))

    runtime_payload = {
        "version": str(existing_payload.get("version") or payload.get("version") or "precedent_distance_weights_v2"),
        "state_distance_version": str(
            existing_payload.get("state_distance_version") or payload.get("state_distance_version") or "weighted_distance_v2"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(source_path),
        "source_artifacts": source_artifacts,
        "notes": merged_notes,
        "objective": dict(existing_payload.get("objective") or payload.get("objective") or {}),
        "scopes": existing_scopes,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True))
    print(str(out_path))


if __name__ == "__main__":
    main()
