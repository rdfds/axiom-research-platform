#!/usr/bin/env python3
"""Repair credit-market metrics in a materialized company-state artifact.

This pass fills the still-empty credit regime / credit window layer using:
- exact macro IG/HY OAS anchors already present in the artifact
- macro OAS history from the raw timeseries parquet for percentiles
- company risk signals already materialized in the artifact

The repaired company-level spread metrics are intentionally tagged as heuristic
proxy values; they are not direct traded bond/CDS spreads.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import pyarrow.parquet as pq


REPAIR_METRICS = [
    "macro.us_ig_oas",
    "macro.us_ig_oas_percentile_history",
    "market.credit_spread_level",
    "market.credit_spread_percentile_2y",
    "market.credit_window_proxy",
]

IG_OAS_INSTRUMENT = "BAMLC0A0CM"
HY_OAS_INSTRUMENT = "BAMLH0A0HYM2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--macro-timeseries-path", help="Optional raw timeseries parquet path")
    parser.add_argument("--out", required=True, help="Output repaired JSONL artifact")
    parser.add_argument("--summary-out", help="Optional summary JSON")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _node_value(node: Dict[str, Any] | None) -> float | None:
    if not node:
        return None
    value = node.get("value")
    if value is None:
        return None
    return float(value)


def _node_support(node: Dict[str, Any] | None) -> str:
    if not node:
        return "unsupported"
    return str(node.get("support_mode") or "unsupported")


def _union_provenance(*nodes: Dict[str, Any] | None) -> list[Dict[str, Any]]:
    merged: list[Dict[str, Any]] = []
    seen = set()
    for node in nodes:
        for prov in (node or {}).get("provenance") or []:
            key = json.dumps(prov, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(copy.deepcopy(prov))
    return merged


def _base_repaired_node(node: Dict[str, Any], *, computed_at: str) -> Dict[str, Any]:
    repaired = copy.deepcopy(node)
    repaired["computed_at"] = computed_at
    repaired["missing_reason"] = None
    repaired["quality_flags"] = repaired.get("quality_flags") or None
    return repaired


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _infer_macro_timeseries_path(artifact_path: Path) -> Path | None:
    for row in iter_rows(artifact_path):
        features = row.get("features") or {}
        for metric in ("macro.ig_oas", "macro.hy_oas", "macro.us_ig_oas"):
            node = features.get(metric) or {}
            for prov in node.get("provenance") or []:
                source = prov.get("source")
                if source and str(source).endswith(".parquet"):
                    return Path(str(source))
    return None


def _load_spread_histories(path: Path) -> Dict[str, pd.DataFrame]:
    table = pq.read_table(
        path,
        columns=["instrument_id", "event_time", "available_time", "trade_date", "value"],
    )
    df = table.to_pandas()
    df = df[df["instrument_id"].isin([IG_OAS_INSTRUMENT, HY_OAS_INSTRUMENT])].copy()
    if df.empty:
        return {}
    df["time"] = pd.to_datetime(df["available_time"], utc=True, errors="coerce")
    missing = df["time"].isna()
    if missing.any():
        df.loc[missing, "time"] = pd.to_datetime(df.loc[missing, "event_time"], utc=True, errors="coerce")
    missing = df["time"].isna()
    if missing.any():
        df.loc[missing, "time"] = pd.to_datetime(df.loc[missing, "trade_date"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["instrument_id", "time", "value"]).copy()
    df = df.sort_values(["instrument_id", "time"]).drop_duplicates(["instrument_id", "time"], keep="last")
    out: Dict[str, pd.DataFrame] = {}
    for instrument_id, group in df.groupby("instrument_id", sort=False):
        out[str(instrument_id)] = group[["time", "value"]].reset_index(drop=True)
    return out


def _monthly_percentile(history: pd.DataFrame | None, *, as_of: pd.Timestamp, years: int) -> float | None:
    if history is None or history.empty:
        return None
    sub = history[history["time"] <= as_of].copy()
    if sub.empty:
        return None
    monthly = sub.set_index("time")["value"].sort_index().resample("ME").last().dropna().tail(years * 12)
    if len(monthly) < min(24, years * 6):
        return None
    return float(monthly.rank(pct=True).iloc[-1] * 100.0)


def _blended_percentile_2y(
    ig_history: pd.DataFrame | None,
    hy_history: pd.DataFrame | None,
    *,
    as_of: pd.Timestamp,
    risk_score: float,
) -> float | None:
    if ig_history is None or hy_history is None or ig_history.empty or hy_history.empty:
        return None
    cutoff = as_of - pd.Timedelta(days=365 * 2)
    ig_sub = ig_history[(ig_history["time"] <= as_of) & (ig_history["time"] >= cutoff)].copy()
    hy_sub = hy_history[(hy_history["time"] <= as_of) & (hy_history["time"] >= cutoff)].copy()
    if ig_sub.empty or hy_sub.empty:
        return None
    merged = pd.merge(ig_sub, hy_sub, on="time", how="inner", suffixes=("_ig", "_hy"))
    if len(merged) < 20:
        return None
    blended = (merged["value_ig"] + (risk_score * (merged["value_hy"] - merged["value_ig"]))) / 100.0
    return float(blended.rank(pct=True).iloc[-1] * 100.0)


def _company_risk_payload(features: Dict[str, Any]) -> Dict[str, Any] | None:
    components: list[tuple[str, float, float, Dict[str, float]]] = []

    gross_lev_node = features.get("capital_structure.gross_leverage_normalized")
    net_lev_node = features.get("capital_structure.net_leverage_normalized")
    gross_lev = _node_value(gross_lev_node)
    net_lev = _node_value(net_lev_node)
    leverage_value = gross_lev
    leverage_source_metric = "capital_structure.gross_leverage_normalized"
    if leverage_value is None and net_lev is not None:
        leverage_value = max(net_lev, 0.0)
        leverage_source_metric = "capital_structure.net_leverage_normalized"
    if leverage_value is not None:
        score = _clip((leverage_value - 1.5) / 5.0, 0.0, 1.0)
        components.append(("leverage", 0.35, score, {
            "value": leverage_value,
            "source_metric": leverage_source_metric,
            "formula": "clip((leverage - 1.5) / 5.0, 0, 1)",
        }))

    vol_node = features.get("market.volatility_30d")
    vol_30 = _node_value(vol_node)
    if vol_30 is not None:
        score = _clip((vol_30 - 0.15) / 0.45, 0.0, 1.0)
        components.append(("volatility_30d", 0.25, score, {
            "value": vol_30,
            "source_metric": "market.volatility_30d",
            "formula": "clip((volatility_30d - 0.15) / 0.45, 0, 1)",
        }))

    dd_node = features.get("market.drawdown_90d")
    drawdown_90 = _node_value(dd_node)
    if drawdown_90 is not None:
        score = _clip((abs(min(drawdown_90, 0.0)) - 0.10) / 0.40, 0.0, 1.0)
        components.append(("drawdown_90d", 0.15, score, {
            "value": drawdown_90,
            "source_metric": "market.drawdown_90d",
            "formula": "clip((abs(min(drawdown_90d, 0)) - 0.10) / 0.40, 0, 1)",
        }))

    liquidity_node = features.get("liquidity.available_liquidity_normalized")
    debt_node = features.get("capital_structure.debt_like_obligations_normalized") or features.get("capital_structure.total_debt_provider_direct")
    liquidity = _node_value(liquidity_node)
    debt = _node_value(debt_node)
    if liquidity is not None and debt not in (None, 0):
        coverage = liquidity / debt
        score = _clip(1.0 - coverage, 0.0, 1.0)
        components.append(("liquidity_coverage", 0.15, score, {
            "value": coverage,
            "source_metric": f"{liquidity_node.get('name')} / {debt_node.get('name')}",
            "formula": "clip(1 - (available_liquidity / debt_like), 0, 1)",
        }))

    fcf_conv_node = features.get("operating.fcf_conversion")
    fcf_conv = _node_value(fcf_conv_node)
    if fcf_conv is not None:
        score = _clip((0.40 - fcf_conv) / 0.80, 0.0, 1.0)
        components.append(("fcf_conversion", 0.10, score, {
            "value": fcf_conv,
            "source_metric": "operating.fcf_conversion",
            "formula": "clip((0.40 - fcf_conversion) / 0.80, 0, 1)",
        }))

    if not components:
        return None

    total_weight = sum(weight for _, weight, _, _ in components)
    risk_score = sum(weight * score for _, weight, score, _ in components) / total_weight
    payload = {
        "risk_score": float(risk_score),
        "components": {name: {"weight": weight, "score": score, **detail} for name, weight, score, detail in components},
        "supporting_nodes": [
            gross_lev_node,
            net_lev_node,
            vol_node,
            dd_node,
            liquidity_node,
            debt_node,
            fcf_conv_node,
        ],
    }
    return payload


def repair_macro_us_ig_oas(*, features: Dict[str, Any], computed_at: str) -> bool:
    target = features.get("macro.us_ig_oas")
    if not target or target.get("value") is not None:
        return False
    source = features.get("macro.ig_oas")
    value = _node_value(source)
    if value is None:
        return False
    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = value
    repaired["support_mode"] = _node_support(source)
    repaired["fallback_used"] = "macro_ig_oas_alias"
    repaired["provenance"] = _union_provenance(source)
    repaired["component_breakdown"] = {
        "source_metric": "macro.ig_oas",
        "formula": "macro.us_ig_oas := macro.ig_oas",
    }
    repaired["quality_flags"] = None
    features["macro.us_ig_oas"] = repaired
    return True


def repair_macro_us_ig_oas_percentile_history(
    *,
    features: Dict[str, Any],
    ig_history: pd.DataFrame | None,
    as_of: pd.Timestamp,
    computed_at: str,
) -> bool:
    target = features.get("macro.us_ig_oas_percentile_history")
    if not target or target.get("value") is not None:
        return False
    pct = _monthly_percentile(ig_history, as_of=as_of, years=10)
    if pct is None:
        return False
    source = features.get("macro.us_ig_oas") or features.get("macro.ig_oas")
    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = pct
    repaired["support_mode"] = "exact"
    repaired["fallback_used"] = "raw_timeseries_ig_oas_monthly_percentile_10y"
    repaired["provenance"] = _union_provenance(source)
    repaired["component_breakdown"] = {
        "source_instrument_id": IG_OAS_INSTRUMENT,
        "formula": "percentile_rank(current_us_ig_oas, monthly_us_ig_oas_history_10y)",
        "history_frequency": "M",
        "lookback_years": 10,
    }
    repaired["quality_flags"] = None
    features["macro.us_ig_oas_percentile_history"] = repaired
    return True


def repair_credit_spread_level(*, features: Dict[str, Any], computed_at: str) -> bool:
    target = features.get("market.credit_spread_level")
    if not target or target.get("value") is not None:
        return False

    ig_node = features.get("macro.us_ig_oas") or features.get("macro.ig_oas")
    hy_node = features.get("macro.hy_oas")
    ig_pct = _node_value(ig_node)
    hy_pct = _node_value(hy_node)
    risk_payload = _company_risk_payload(features)
    if ig_pct is None or hy_pct is None or risk_payload is None:
        return False

    risk_score = float(risk_payload["risk_score"])
    implied_pct = ig_pct + (risk_score * (hy_pct - ig_pct))
    implied_ratio = implied_pct / 100.0

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = implied_ratio
    repaired["support_mode"] = "proxy_missing_component"
    repaired["fallback_used"] = "macro_oas_plus_company_risk_heuristic"
    repaired["provenance"] = _union_provenance(ig_node, hy_node, *risk_payload["supporting_nodes"])
    repaired["component_breakdown"] = {
        "macro_ig_oas_pct": ig_pct,
        "macro_hy_oas_pct": hy_pct,
        "company_risk_score": risk_score,
        "risk_components": risk_payload["components"],
        "implied_spread_pct": implied_pct,
        "formula": "(macro_ig_oas_pct + company_risk_score * (macro_hy_oas_pct - macro_ig_oas_pct)) / 100",
        "output_unit": "ratio",
    }
    repaired["quality_flags"] = ["heuristic_credit_spread_repair"]
    features["market.credit_spread_level"] = repaired
    return True


def repair_credit_spread_percentile_2y(
    *,
    features: Dict[str, Any],
    ig_history: pd.DataFrame | None,
    hy_history: pd.DataFrame | None,
    as_of: pd.Timestamp,
    computed_at: str,
) -> bool:
    target = features.get("market.credit_spread_percentile_2y")
    if not target or target.get("value") is not None:
        return False

    risk_payload = _company_risk_payload(features)
    credit_spread_node = features.get("market.credit_spread_level")
    if risk_payload is None or _node_value(credit_spread_node) is None:
        return False
    pct = _blended_percentile_2y(
        ig_history,
        hy_history,
        as_of=as_of,
        risk_score=float(risk_payload["risk_score"]),
    )
    if pct is None:
        return False

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = pct
    repaired["support_mode"] = "proxy_missing_component"
    repaired["fallback_used"] = "blended_macro_oas_history_2y"
    repaired["provenance"] = _union_provenance(
        features.get("macro.us_ig_oas") or features.get("macro.ig_oas"),
        features.get("macro.hy_oas"),
        credit_spread_node,
        *risk_payload["supporting_nodes"],
    )
    repaired["component_breakdown"] = {
        "source_instruments": [IG_OAS_INSTRUMENT, HY_OAS_INSTRUMENT],
        "company_risk_score": float(risk_payload["risk_score"]),
        "formula": "percentile_rank(blended_macro_oas_company_proxy, daily_history_2y)",
        "lookback_years": 2,
    }
    repaired["quality_flags"] = ["heuristic_credit_spread_repair"]
    features["market.credit_spread_percentile_2y"] = repaired
    return True


def repair_credit_window_proxy(*, features: Dict[str, Any], computed_at: str) -> bool:
    target = features.get("market.credit_window_proxy")
    if not target or target.get("value") is not None:
        return False

    spread_node = features.get("market.credit_spread_level")
    vol_node = features.get("market.volatility_30d")
    spread = _node_value(spread_node)
    vol_30 = _node_value(vol_node)

    components: list[tuple[str, float]] = []
    if spread is not None:
        components.append(("credit_spread_component", _clip(1.0 - (spread / 0.10), 0.0, 1.0)))
    if vol_30 is not None:
        components.append(("volatility_component", _clip(1.0 - (vol_30 / 1.0), 0.0, 1.0)))
    if not components:
        return False

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = float(sum(value for _, value in components) / len(components))
    repaired["support_mode"] = "proxy_missing_component"
    repaired["fallback_used"] = "credit_spread_plus_price_volatility_heuristic"
    repaired["provenance"] = _union_provenance(spread_node, vol_node)
    repaired["component_breakdown"] = {
        "components": {name: value for name, value in components},
        "credit_spread_level": spread,
        "volatility_30d": vol_30,
        "formula": "mean([clip(1 - credit_spread_level / 0.10, 0, 1), clip(1 - volatility_30d / 1.0, 0, 1)])",
    }
    repaired["quality_flags"] = ["heuristic_credit_window_repair"]
    features["market.credit_window_proxy"] = repaired
    return True


def build_summary(path: Path) -> Dict[str, Dict[str, int]]:
    counters: Dict[str, Counter[str]] = {metric: Counter() for metric in REPAIR_METRICS}
    for row in iter_rows(path):
        features = row.get("features") or {}
        for metric in REPAIR_METRICS:
            node = features.get(metric) or {}
            mode = str(node.get("support_mode") or "unsupported")
            if node.get("value") is None:
                mode = "unsupported"
            counters[metric][mode] += 1
    return {
        metric: {
            "exact": counter["exact"],
            "proxy_missing_component": counter["proxy_missing_component"],
            "unsupported": counter["unsupported"],
        }
        for metric, counter in counters.items()
    }


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    macro_timeseries_path = Path(args.macro_timeseries_path) if args.macro_timeseries_path else _infer_macro_timeseries_path(artifact_path)
    if macro_timeseries_path is None:
        raise SystemExit("Could not infer raw timeseries parquet path from artifact provenance.")

    spread_histories = _load_spread_histories(macro_timeseries_path)
    ig_history = spread_histories.get(IG_OAS_INSTRUMENT)
    hy_history = spread_histories.get(HY_OAS_INSTRUMENT)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    computed_at = _now_iso()

    with out_path.open("w") as out_handle:
        for row in iter_rows(artifact_path):
            features = row.get("features") or {}
            as_of = pd.to_datetime((next(iter(features.values()), {}) or {}).get("as_of_time"), utc=True, errors="coerce")
            if pd.isna(as_of):
                as_of = pd.Timestamp.now(tz="UTC")
            repair_macro_us_ig_oas(features=features, computed_at=computed_at)
            repair_macro_us_ig_oas_percentile_history(
                features=features,
                ig_history=ig_history,
                as_of=as_of,
                computed_at=computed_at,
            )
            repair_credit_spread_level(features=features, computed_at=computed_at)
            repair_credit_spread_percentile_2y(
                features=features,
                ig_history=ig_history,
                hy_history=hy_history,
                as_of=as_of,
                computed_at=computed_at,
            )
            repair_credit_window_proxy(features=features, computed_at=computed_at)
            out_handle.write(json.dumps(row) + "\n")

    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(build_summary(out_path), indent=2))

    print(f"Repaired credit-market metrics -> {out_path}")


if __name__ == "__main__":
    main()
