from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .causal_impact_model import (
    DEFAULT_CAUSAL_IMPACT_MODEL_ARTIFACT,
    DEFAULT_CAUSAL_ROUTING_CONFIG_PATH,
)

DEFAULT_PRECEDENT_RETRIEVAL_VERSION = "precedent_retrieval_state_vector_v1"

_RUNTIME_ENV_KEYS = (
    "AXIOM_DATA_ROOT",
    "AXIOM_COMPANYFACTS_ROOT",
    "AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER",
    "AXIOM_RUNTIME_FEATURE_ADAPTER_PROFILE",
    "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
    "CAUSAL_IMPACT_MODEL_PATH",
    "CAUSAL_ROUTING_CONFIG_PATH",
    "CAUSAL_IMPACT_MODE",
    "CAUSAL_ACTION_BLOCKLIST",
    "CAUSAL_ACTION_DENYLIST",
    "CAUSAL_ACTION_BLOCKLIST_PATH",
    "CAUSAL_MIN_OBJECTIVE_OOS_R2",
    "CAUSAL_STRICT_QUALITY_FLOOR",
    "CAUSAL_STRICT_SUPPORT_FLOOR",
    "CAUSAL_STRICT_MIN_TRAIN_ROWS",
    "CAUSAL_STRICT_MIN_OOS_R2",
    "CAUSAL_STRICT_MIN_TREATED_ROWS",
    "CAUSAL_STRICT_MIN_CONTROL_ROWS",
    "MECHANISM_MODEL_VERSION",
    "PRECEDENT_RETRIEVAL_VERSION",
    "RECO_PRECEDENT_WORKERS",
    "RECOMMENDATION_RUN_TMP_DIR",
)


def _normalize_env_value(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if value.lower() in {"none", "null"}:
        return None
    return value


def _safe_float(raw: Any) -> Optional[float]:
    try:
        out = float(raw)
    except Exception:
        return None
    if out != out:
        return None
    return float(out)


def _safe_int(raw: Any) -> Optional[int]:
    val = _safe_float(raw)
    if val is None:
        return None
    return int(val)


def _path_digest(path_value: str) -> Dict[str, Any]:
    raw = str(path_value or "").strip()
    out: Dict[str, Any] = {"path": raw or None, "exists": False, "sha256": None}
    if not raw:
        return out
    p = Path(raw)
    out["exists"] = p.exists()
    if not p.exists() or not p.is_file():
        return out
    try:
        out["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        out["sha256"] = None
    return out


def _parse_action_tokens(text: str) -> list[str]:
    raw = str(text or "").replace("\n", ",")
    out: list[str] = []
    for token in raw.split(","):
        value = str(token or "").strip().lower()
        if not value:
            continue
        if value in {"none", "null"}:
            continue
        if value not in out:
            out.append(value)
    return out


def _load_action_tokens_from_file(path_value: str) -> list[str]:
    raw = str(path_value or "").strip()
    if not raw:
        return []
    p = Path(raw)
    if not p.exists():
        return []
    return _parse_action_tokens(p.read_text())


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for key, value in dict(patch or {}).items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _deep_merge(dict(out.get(key) or {}), dict(value or {}))
        else:
            out[key] = value
    return out


def merge_metadata_patch(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    return _deep_merge(dict(base or {}), dict(patch or {}))


def _runtime_env_raw(keys: Iterable[str] = _RUNTIME_ENV_KEYS) -> Dict[str, Optional[str]]:
    return {str(k): _normalize_env_value(os.environ.get(str(k))) for k in keys}


def capture_runtime_env_config() -> Dict[str, Any]:
    raw = _runtime_env_raw()
    model_path = raw.get("CAUSAL_IMPACT_MODEL_PATH") or str(DEFAULT_CAUSAL_IMPACT_MODEL_ARTIFACT)
    routing_path = raw.get("CAUSAL_ROUTING_CONFIG_PATH") or str(DEFAULT_CAUSAL_ROUTING_CONFIG_PATH)
    blocklist_path = raw.get("CAUSAL_ACTION_BLOCKLIST_PATH") or ""
    inline_tokens = _parse_action_tokens(
        ",".join(
            x
            for x in (
                raw.get("CAUSAL_ACTION_BLOCKLIST") or "",
                raw.get("CAUSAL_ACTION_DENYLIST") or "",
            )
            if x
        )
    )
    file_tokens = _load_action_tokens_from_file(blocklist_path)
    all_block_tokens = sorted(set(inline_tokens + file_tokens))
    return {
        "python_executable": sys.executable,
        "runtime_feature_adapter": {
            "enabled": raw.get("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER"),
            "profile": raw.get("AXIOM_RUNTIME_FEATURE_ADAPTER_PROFILE"),
            "rules": raw.get("AXIOM_RUNTIME_FEATURE_ADAPTER_RULES"),
        },
        "causal": {
            "model": _path_digest(model_path),
            "routing": _path_digest(routing_path),
            "mode": raw.get("CAUSAL_IMPACT_MODE") or "blend",
            "minimum_objective_oos_r2": _safe_float(raw.get("CAUSAL_MIN_OBJECTIVE_OOS_R2")),
            "strict_quality_floor": _safe_float(raw.get("CAUSAL_STRICT_QUALITY_FLOOR")),
            "strict_support_floor": _safe_float(raw.get("CAUSAL_STRICT_SUPPORT_FLOOR")),
            "strict_min_train_rows": _safe_int(raw.get("CAUSAL_STRICT_MIN_TRAIN_ROWS")),
            "strict_min_oos_r2": _safe_float(raw.get("CAUSAL_STRICT_MIN_OOS_R2")),
            "strict_min_treated_rows": _safe_int(raw.get("CAUSAL_STRICT_MIN_TREATED_ROWS")),
            "strict_min_control_rows": _safe_int(raw.get("CAUSAL_STRICT_MIN_CONTROL_ROWS")),
            "blocklist": {
                "file": _path_digest(blocklist_path),
                "inline_entries": inline_tokens,
                "entries": all_block_tokens,
                "entry_count": len(all_block_tokens),
            },
        },
        "precedent": {
            "worker_count": _safe_int(raw.get("RECO_PRECEDENT_WORKERS")),
            "retrieval_version": raw.get("PRECEDENT_RETRIEVAL_VERSION") or DEFAULT_PRECEDENT_RETRIEVAL_VERSION,
        },
        "paths": {
            "recommendation_run_tmp_dir": raw.get("RECOMMENDATION_RUN_TMP_DIR"),
        },
        "raw_env": raw,
    }


