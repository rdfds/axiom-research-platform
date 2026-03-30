#!/usr/bin/env python
"""
Validate Inputs Layer datasets against JSON schemas.

This script performs lightweight checks:
- required columns
- basic dtype compatibility
- timestamp parseability
- timestamp ordering vs ingested_at
- confidence_score bounds

Outputs a DataIntegrityLog parquet.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_schema(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(
    path: Path,
    sample_rows: int | None = None,
    columns: List[str] | None = None,
) -> pd.DataFrame:
    if path.is_dir():
        # Partitioned parquet dataset
        if sample_rows:
            files = sorted(path.rglob("*.parquet"))
            if not files:
                return pd.DataFrame()
            parts = []
            total = 0
            for f in files:
                try:
                    df_part = pd.read_parquet(f, columns=columns)
                except Exception:
                    # skip unreadable/unstable files (e.g., stale NFS handles)
                    continue
                parts.append(df_part)
                total += len(df_part)
                if total >= sample_rows:
                    break
            df = pd.concat(parts, ignore_index=True)
            return df.head(sample_rows)
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".json", ".ndjson"}:
        return pd.read_json(path, lines=path.suffix == ".ndjson")
    raise ValueError(f"Unsupported file type: {path}")


def is_numeric_like(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    # soft check for numeric-ish object columns
    coerced = pd.to_numeric(s, errors="coerce")
    return coerced.notna().mean() >= 0.95


def is_bool_like(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        vals = s.dropna().unique()
        return set(vals).issubset({0, 1})
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        # Try numeric coercion first (handles 0.0/1.0 stored as object)
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().any():
            vals = set(coerced.dropna().unique())
            return vals.issubset({0, 1})
        vals = {str(v).strip().lower() for v in s.dropna().unique()}
        return vals.issubset({"true", "false", "0", "1", "yes", "no", "y", "n"})
    return False


def is_string_like(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s) or s.dtype == object


def is_datetime_like(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def validate_required_columns(
    df: pd.DataFrame,
    required: List[str],
    available_cols: List[str] | None = None,
) -> Tuple[bool, List[str]]:
    cols = available_cols if available_cols is not None else list(df.columns)
    missing = [c for c in required if c not in cols]
    return (len(missing) == 0), missing


def get_dataset_columns(path: Path) -> List[str] | None:
    if path.is_dir() or path.suffix == ".parquet":
        try:
            dataset = ds.dataset(path, format="parquet")
            return list(dataset.schema.names)
        except Exception:
            return None
    return None


def validate_types(df: pd.DataFrame, schema: Dict) -> List[str]:
    issues = []
    props = schema.get("properties", {})
    for col, spec in props.items():
        if col not in df.columns:
            continue
        expected = spec.get("type")
        if expected is None:
            continue
        if isinstance(expected, str):
            expected = [expected]
        expected = [t for t in expected if t != "null"]
        if not expected:
            continue
        s = df[col]
        if s.notna().sum() == 0:
            continue
        ok = True
        if "number" in expected or "integer" in expected:
            ok = is_numeric_like(s)
        elif "boolean" in expected:
            ok = is_bool_like(s)
        elif "string" in expected:
            ok = is_string_like(s) or is_datetime_like(s)
        # object/array checks are intentionally soft
        if not ok:
            issues.append(f"dtype_mismatch:{col}")
    return issues


def parse_datetime_cols(df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], List[str]]:
    parsed = {}
    bad = []
    for col in df.columns:
        if col.endswith("_at") or col.endswith("_date") or col.endswith("_time"):
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            parsed[col] = s
            if df[col].notna().any():
                bad_frac = (df[col].notna() & s.isna()).mean()
                if bad_frac > 0:
                    bad.append(f"bad_datetime:{col}:{bad_frac:.2%}")
    return parsed, bad


def check_timestamp_order(parsed: Dict[str, pd.Series]) -> List[str]:
    issues = []
    if "published_at" in parsed and "ingested_at" in parsed:
        bad = (parsed["published_at"] > parsed["ingested_at"]).mean()
        if bad > 0:
            issues.append(f"published_after_ingested:{bad:.2%}")
    if "effective_at" in parsed and "ingested_at" in parsed:
        bad = (parsed["effective_at"] > parsed["ingested_at"]).mean()
        if bad > 0:
            issues.append(f"effective_after_ingested:{bad:.2%}")
    return issues


def check_confidence(df: pd.DataFrame) -> List[str]:
    if "confidence_score" not in df.columns:
        return []
    s = pd.to_numeric(df["confidence_score"], errors="coerce")
    bad = ((s < 0) | (s > 1)).mean()
    return [f"confidence_out_of_bounds:{bad:.2%}"] if bad > 0 else []


def log_entry(dataset: str, check: str, status: str, message: str | None, rows: int | None = None) -> Dict:
    return {
        "log_id": str(uuid.uuid4()),
        "dataset_name": dataset,
        "check_name": check,
        "status": status,
        "message": message,
        "run_at": utc_now(),
        "rows_checked": rows,
        "failure_count": None,
        "sample_ids": None,
        "source_id": None,
        "source_type": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inputs_layer.json")
    parser.add_argument("--out", default=None, help="Override output log path.")
    parser.add_argument("--sample-rows", type=int, default=20000, help="Row sample for large datasets.")
    parser.add_argument("--include-large-cols", action="store_true", help="Read large blob columns (raw_text).")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = config.get("datasets", [])
    out_path = Path(args.out) if args.out else Path(config.get("output_log_path", "data/inputs_layer/data_integrity_log.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logs: List[Dict] = []
    large_cols_by_dataset = {
        "RawDocumentStore": ["raw_text", "raw_html", "metadata"],
    }

    for ds in datasets:
        name = ds["name"]
        path = Path(ds["path"])
        schema_path = Path(ds["schema"])
        required = bool(ds.get("required", False))

        if not path.exists():
            status = "fail" if required else "warn"
            logs.append(log_entry(name, "file_exists", status, f"Missing file: {path}"))
            continue

        if not schema_path.exists():
            logs.append(log_entry(name, "schema_exists", "fail", f"Missing schema: {schema_path}"))
            continue

        available_cols = get_dataset_columns(path)
        read_cols = None
        if not args.include_large_cols and name in large_cols_by_dataset and available_cols:
            read_cols = [c for c in available_cols if c not in large_cols_by_dataset[name]]

        df = load_dataset(path, sample_rows=args.sample_rows if path.is_dir() else None, columns=read_cols)
        schema = load_schema(schema_path)

        ok, missing = validate_required_columns(df, schema.get("required", []), available_cols=available_cols)
        if not ok:
            logs.append(log_entry(name, "required_columns", "fail", f"Missing columns: {missing}", rows=len(df)))
            continue
        logs.append(log_entry(name, "required_columns", "pass", None, rows=len(df)))

        dtype_issues = validate_types(df, schema)
        if dtype_issues:
            logs.append(log_entry(name, "dtype_check", "warn", ", ".join(dtype_issues), rows=len(df)))
        else:
            logs.append(log_entry(name, "dtype_check", "pass", None, rows=len(df)))

        parsed, bad_dt = parse_datetime_cols(df)
        if bad_dt:
            logs.append(log_entry(name, "datetime_parse", "warn", ", ".join(bad_dt), rows=len(df)))
        else:
            logs.append(log_entry(name, "datetime_parse", "pass", None, rows=len(df)))

        order_issues = check_timestamp_order(parsed)
        if order_issues:
            logs.append(log_entry(name, "timestamp_order", "warn", ", ".join(order_issues), rows=len(df)))
        else:
            logs.append(log_entry(name, "timestamp_order", "pass", None, rows=len(df)))

        conf_issues = check_confidence(df)
        if conf_issues:
            logs.append(log_entry(name, "confidence_bounds", "warn", ", ".join(conf_issues), rows=len(df)))
        else:
            logs.append(log_entry(name, "confidence_bounds", "pass", None, rows=len(df)))

        # Missingness warning for required columns
        req = schema.get("required", [])
        miss_stats = []
        for col in req:
            if col in df.columns:
                miss = df[col].isna().mean()
                if miss > 0:
                    miss_stats.append(f"{col}:{miss:.2%}")
        if miss_stats:
            logs.append(log_entry(name, "missing_required_values", "warn", ", ".join(miss_stats), rows=len(df)))
        else:
            logs.append(log_entry(name, "missing_required_values", "pass", None, rows=len(df)))

    log_df = pd.DataFrame(logs)
    log_df.to_parquet(out_path, index=False)
    print(f"Saved DataIntegrityLog -> {out_path} ({len(log_df):,} rows)")


if __name__ == "__main__":
    main()
