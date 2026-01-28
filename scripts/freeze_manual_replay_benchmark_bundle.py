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


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["cp", "-c", str(src), str(dst)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    except Exception:
        pass
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            _copy_tree(child, target)
        else:
            _copy_file(child, target)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _derive_case_company_ids(manifest_paths: Iterable[Path]) -> List[str]:
    ids: Set[str] = set()
    for path in manifest_paths:
        payload = _load_json(path)
        cases = payload["cases"] if isinstance(payload, dict) and "cases" in payload else payload
        for case in cases:
            for key in ("company_id", "source_company_id"):
                value = case.get(key)
                if value:
                    ids.add(str(value).zfill(10))
    return sorted(ids)


def _copy_companyfacts_subset(source_root: Path, dest_root: Path, company_ids: Iterable[str]) -> List[str]:
    copied: List[str] = []
    dest_root.mkdir(parents=True, exist_ok=True)
    for company_id in company_ids:
        source_path = source_root / f"CIK{str(company_id).zfill(10)}.json"
        if not source_path.exists():
            continue
        target_path = dest_root / source_path.name
        # Read/write forces real bytes instead of preserving cloud placeholders.
        target_path.write_text(source_path.read_text())
        copied.append(str(company_id).zfill(10))
    return copied


def _copy_report_if_present(src: Path, dst: Path) -> None:
    if src.exists():
        _copy_file(src, dst)


def main() -> None:
    args = _parse_args()
    lock_config_path = _resolve_root_path(args.lock_config)
    bundle_root = Path(args.bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    bundle_configs = bundle_root / "configs"
    bundle_inputs = bundle_root / "inputs"
    bundle_models = bundle_root / "models"
    bundle_reports = bundle_root / "reports"
    bundle_manifests = bundle_root / "manifests"
    for path in (bundle_configs, bundle_inputs, bundle_models, bundle_reports, bundle_manifests):
        path.mkdir(parents=True, exist_ok=True)

    lock_payload = _load_json(lock_config_path)

    resolved_artifacts = dict(lock_payload.get("artifacts", {}) or {})
    resolved_env_candidates = dict(lock_payload.get("env_override_candidates", {}) or {})

    benchmark_manifest_paths = {
        benchmark_key: _resolve_root_path(benchmark_payload["manifest"])
        for benchmark_key, benchmark_payload in dict(lock_payload.get("benchmarks", {}) or {}).items()
    }
    manifest_dir = lock_config_path.parent

    # Freeze configs/manifests.
    _copy_file(lock_config_path, bundle_configs / lock_config_path.name)
    for path in manifest_dir.glob("*.json"):
        _copy_file(path, bundle_configs / path.name)
    for path in manifest_dir.glob("*.md"):
        _copy_file(path, bundle_configs / path.name)

    # Resolve current inputs exactly as the runner does today.
    source_paths: Dict[str, Path] = {
        "outcomes_path": _resolve_root_path(resolved_artifacts["outcomes_path"]),
        "action_support_manifest": _resolve_root_path(resolved_artifacts["action_support_manifest"]),
        "entity_graph_path": _resolve_root_path(resolved_artifacts["entity_graph_path"]),
        "entity_identifier_path": _resolve_root_path(resolved_artifacts["entity_identifier_path"]),
        "entity_table_path": _resolve_root_path(resolved_artifacts["entity_table_path"]),
        "raw_timeseries_path": _resolve_root_path(resolved_artifacts["raw_timeseries_path"]),
        "event_store_path": _resolve_root_path(resolved_artifacts["event_store_path"]),
        "ownership_summary_path": _resolve_root_path(resolved_artifacts["ownership_summary_path"]),
        "issuer_ratings_path": _resolve_root_path(resolved_artifacts["issuer_ratings_path"]),
        "companyfacts_root": _resolve_root_path(resolved_artifacts["companyfacts_root"]),
        "facts_path": _resolve_candidate_path(resolved_artifacts["facts_path_candidates"]),
    }
    if resolved_artifacts.get("corporate_actions_master_path"):
        source_paths["corporate_actions_master_path"] = _resolve_root_path(resolved_artifacts["corporate_actions_master_path"])

    resolved_env: Dict[str, Path] = {
        name: _resolve_candidate_path(values)
        for name, values in resolved_env_candidates.items()
    }

    # Copy baseline tabular inputs.
    data_destinations = {
        "outcomes_path": bundle_inputs / "action_outcomes_with_credit_ratings.normalized_full.parquet",
        "action_support_manifest": bundle_configs / "action_data_support_manifest.json",
        "entity_graph_path": bundle_inputs / "entity_graph.parquet",
        "entity_identifier_path": bundle_inputs / "entity_identifier.parquet",
        "entity_table_path": bundle_inputs / "entity.parquet",
        "raw_timeseries_path": bundle_inputs / "raw_timeseries.parquet",
        "event_store_path": bundle_inputs / "event_store.parquet",
        "ownership_summary_path": bundle_inputs / "ownership_13f_summary.parquet",
        "issuer_ratings_path": bundle_inputs / "issuer_rating_history.parquet",
    }
    for key, target in data_destinations.items():
        _copy_file(source_paths[key], target)

    # Copy facts cache exactly as used today.
    facts_dest = bundle_inputs / "facts_asof_2026"
    _copy_tree(source_paths["facts_path"], facts_dest)

    # Copy the companyfacts subset touched by the frozen manifests.
    case_company_ids = _derive_case_company_ids(benchmark_manifest_paths.values())
    copied_companyfacts = _copy_companyfacts_subset(
        source_root=source_paths["companyfacts_root"],
        dest_root=bundle_inputs / "companyfacts",
        company_ids=case_company_ids,
    )

    # Copy mutable env/config artifacts.
    env_destinations = {
        "AXIOM_METRIC_POLICY_PATH": bundle_configs / "market_metric_policy_v1.json",
        "AXIOM_METHODOLOGY_REGISTRY_PATH": bundle_configs / "consumer_industrials_canonical_registry_v1.json",
        "AXIOM_INPUT_SOURCE_REGISTRY_PATH": bundle_configs / "company_state_input_source_registry_v1.json",
        "AXIOM_SMART_METRIC_REGISTRY_PATH": bundle_configs / "smart_metric_registry_v1.json",
        "AXIOM_MARKET_AVAILABILITY_OVERRIDES_PATH": bundle_configs / "liquidity_market_availability_overrides.json",
    }
    for env_name, target in env_destinations.items():
        _copy_file(resolved_env[env_name], target)

    # Copy current model artifacts used in the current comparison.
    champion_model_files = [
        ROOT / "data" / "models" / "causal_impact_model_v6_bundle_contract_hgb_actiontype.json",
        ROOT / "data" / "models" / "causal_impact_model_v6_bundle_contract_hgb_actiontype.bundle.pkl",
        ROOT / "data" / "models" / "causal_impact_model_v6_bundle_contract_hgb_actiontype.model_card.json",
    ]
    feedback_model_files = [
        ROOT / "data" / "models" / "causal_impact_model_v7_feedback_hgb_actiontype.json",
        ROOT / "data" / "models" / "causal_impact_model_v7_feedback_hgb_actiontype.bundle.pkl",
        ROOT / "data" / "models" / "causal_impact_model_v7_feedback_hgb_actiontype.model_card.json",
    ]
    for path in champion_model_files:
        _copy_file(path, bundle_models / path.name)
    if args.include_feedback_model:
        for path in feedback_model_files:
            _copy_file(path, bundle_models / path.name)

    # Copy current comparison reports for context.
    report_paths = [
        Path("/tmp/manual_replay_champion_capreturn_full_report_20260405.json"),
        Path("/tmp/manual_replay_champion_capreturn_full_scorecard_20260405.json"),
        Path("/tmp/manual_replay_champion_capreturn_full_scorecard_20260405.md"),
        Path("/tmp/manual_replay_feedback_capreturn_full_report_20260405.json"),
        Path("/tmp/manual_replay_feedback_capreturn_full_scorecard_20260405.json"),
        Path("/tmp/manual_replay_feedback_capreturn_full_scorecard_20260405.md"),
        Path("/tmp/manual_replay_champion_capstructure_full_report_20260405.json"),
        Path("/tmp/manual_replay_champion_capstructure_full_scorecard_20260405.json"),
        Path("/tmp/manual_replay_champion_capstructure_full_scorecard_20260405.md"),
        Path("/tmp/manual_replay_feedback_capstructure_full_report_20260405.json"),
        Path("/tmp/manual_replay_feedback_capstructure_full_scorecard_20260405.json"),
        Path("/tmp/manual_replay_feedback_capstructure_full_scorecard_20260405.md"),
    ]
    for path in report_paths:
        _copy_report_if_present(path, bundle_reports / path.name)

    artifact_manifest_paths = [
        Path("/tmp/manual_replay_champion_capreturn_full_runs_20260405/_backtest_artifacts/capital_return_holdout.artifact_manifest.json"),
        Path("/tmp/manual_replay_champion_capstructure_full_runs_20260405/_backtest_artifacts/capital_structure_holdout.artifact_manifest.json"),
        Path("/tmp/manual_replay_feedback_capreturn_full_runs_20260405/_backtest_artifacts/capital_return_holdout.artifact_manifest.json"),
        Path("/tmp/manual_replay_feedback_capstructure_full_runs_20260405/_backtest_artifacts/capital_structure_holdout.artifact_manifest.json"),
    ]
    for path in artifact_manifest_paths:
        _copy_report_if_present(path, bundle_manifests / path.name.replace(".artifact_manifest", f".{path.parent.parent.parent.name}.artifact_manifest"))

    # Write a bundle-local frozen lock config using absolute paths so the runner can consume it directly.
    frozen_lock = json.loads(json.dumps(lock_payload))
    frozen_lock["artifacts"]["outcomes_path"] = str(data_destinations["outcomes_path"])
    frozen_lock["artifacts"]["action_support_manifest"] = str(data_destinations["action_support_manifest"])
    frozen_lock["artifacts"]["entity_graph_path"] = str(data_destinations["entity_graph_path"])
    frozen_lock["artifacts"]["entity_identifier_path"] = str(data_destinations["entity_identifier_path"])
    frozen_lock["artifacts"]["entity_table_path"] = str(data_destinations["entity_table_path"])
    frozen_lock["artifacts"]["raw_timeseries_path"] = str(data_destinations["raw_timeseries_path"])
    frozen_lock["artifacts"]["event_store_path"] = str(data_destinations["event_store_path"])
    frozen_lock["artifacts"]["ownership_summary_path"] = str(data_destinations["ownership_summary_path"])
    frozen_lock["artifacts"]["issuer_ratings_path"] = str(data_destinations["issuer_ratings_path"])
    frozen_lock["artifacts"]["companyfacts_root"] = str(bundle_inputs / "companyfacts")
    frozen_lock["artifacts"]["facts_path_candidates"] = [str(facts_dest)]
    if "corporate_actions_master_path" in frozen_lock["artifacts"]:
        frozen_lock["artifacts"]["corporate_actions_master_path"] = str(source_paths["corporate_actions_master_path"])
    for benchmark_key, manifest_path in benchmark_manifest_paths.items():
        frozen_lock["benchmarks"][benchmark_key]["manifest"] = str(bundle_configs / manifest_path.name)
    frozen_lock["env_override_candidates"] = {
        "AXIOM_METRIC_POLICY_PATH": [str(env_destinations["AXIOM_METRIC_POLICY_PATH"])],
        "AXIOM_METHODOLOGY_REGISTRY_PATH": [str(env_destinations["AXIOM_METHODOLOGY_REGISTRY_PATH"])],
        "AXIOM_INPUT_SOURCE_REGISTRY_PATH": [str(env_destinations["AXIOM_INPUT_SOURCE_REGISTRY_PATH"])],
        "AXIOM_SMART_METRIC_REGISTRY_PATH": [str(env_destinations["AXIOM_SMART_METRIC_REGISTRY_PATH"])],
        "AXIOM_MARKET_AVAILABILITY_OVERRIDES_PATH": [str(env_destinations["AXIOM_MARKET_AVAILABILITY_OVERRIDES_PATH"])],
    }
    frozen_lock_path = bundle_configs / "manual_replay_benchmark_lock.frozen_20260405.json"
    frozen_lock_path.write_text(json.dumps(frozen_lock, indent=2))

    bundle_manifest = {
        "bundle_root": str(bundle_root),
        "frozen_lock_config": str(frozen_lock_path),
        "source_lock_config": str(lock_config_path),
        "copied_case_companyfacts_count": len(copied_companyfacts),
        "copied_case_companyfacts_ids": copied_companyfacts,
        "benchmark_manifests": {key: str(bundle_configs / path.name) for key, path in benchmark_manifest_paths.items()},
        "champion_model_path": str(bundle_models / champion_model_files[0].name),
        "feedback_model_path": str(bundle_models / feedback_model_files[0].name) if args.include_feedback_model else None,
        "frozen_inputs": {
            "raw_timeseries_path": _path_metadata(data_destinations["raw_timeseries_path"]),
            "event_store_path": _path_metadata(data_destinations["event_store_path"]),
            "ownership_summary_path": _path_metadata(data_destinations["ownership_summary_path"]),
            "facts_path": _path_metadata(facts_dest),
            "companyfacts_root": _path_metadata(bundle_inputs / "companyfacts"),
            "smart_metric_registry": _path_metadata(env_destinations["AXIOM_SMART_METRIC_REGISTRY_PATH"]),
        },
        "reports": sorted(path.name for path in bundle_reports.iterdir()),
        "artifact_manifests": sorted(path.name for path in bundle_manifests.iterdir()),
        "usage": {
            "champion_capreturn": f"CAUSAL_IMPACT_MODEL_PATH={bundle_models / champion_model_files[0].name} PYTHONPATH={ROOT} python {ROOT / 'scripts' / 'run_manual_replay_benchmark.py'} --config {frozen_lock_path} --benchmark capital_return_holdout --runs-root /tmp/manual_replay_bundle_capreturn_runs --snapshot-cache-dir /tmp/manual_replay_bundle_capreturn_cache --out-json /tmp/manual_replay_bundle_capreturn_report.json",
            "champion_capstructure": f"CAUSAL_IMPACT_MODEL_PATH={bundle_models / champion_model_files[0].name} PYTHONPATH={ROOT} python {ROOT / 'scripts' / 'run_manual_replay_benchmark.py'} --config {frozen_lock_path} --benchmark capital_structure_holdout --runs-root /tmp/manual_replay_bundle_capstructure_runs --snapshot-cache-dir /tmp/manual_replay_bundle_capstructure_cache --out-json /tmp/manual_replay_bundle_capstructure_report.json",
        },
    }
    (bundle_root / "bundle_manifest.json").write_text(json.dumps(bundle_manifest, indent=2))

    readme = f"""# Manual Replay Benchmark Bundle (2026-04-05)

This bundle freezes the mutable replay inputs used in the April 5, 2026 investigation.

Main entrypoint:
- `{frozen_lock_path}`

Key frozen inputs:
- `{data_destinations['raw_timeseries_path']}`
- `{facts_dest}`
- `{bundle_inputs / 'companyfacts'}`
- `{env_destinations['AXIOM_SMART_METRIC_REGISTRY_PATH']}`

Included reports:
- champion and feedback rerun reports/scorecards for capital return and capital structure

Notes:
- companyfacts were frozen as a targeted subset for the benchmark case companies
- the champion HGB model artifact is frozen in `{bundle_models}`
"""
    (bundle_root / "README.md").write_text(readme)

    print(json.dumps({
        "bundle_root": str(bundle_root),
        "frozen_lock_config": str(frozen_lock_path),
        "copied_case_companyfacts_count": len(copied_companyfacts),
        "included_feedback_model": bool(args.include_feedback_model),
    }, indent=2))


if __name__ == "__main__":
    main()
