#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RECO_DISABLE_PRECEDENT_NARRATIVE", "1")

import sys

sys.path.insert(0, str(REPO_ROOT))

from src.model_feature_bundle import _STATE_VECTOR_V1_FEATURES, build_model_feature_bundle
from src.pipeline.precedent_brain import (
    augment_precedent_state_vector_columns,
    build_precedent_pack_v2,
    build_precedent_retrieval_index,
)
from src.pipeline.run import (
    _baseline_from_world_model_features,
    _default_precedent_outcomes_path,
    adapt_snapshot,
    attach_model_feature_bundle,
    feature_view_from_snapshot,
)

SNAPSHOT_PATH = (
    REPO_ROOT
    / "out/materialized_feedback_20260405/company_state_snapshots_asof=2024-12-31.input_layer_v1_smart_normalized_with_sec.feedback_pipeline.full_inputs_v3.jsonl.gz"
)
OUT_DIR = REPO_ROOT / "out/metric_explainers"
SUMMARY_DOC_PATH = OUT_DIR / "Compact_precedent_validation_examples_2026_04_06.md"
DETAIL_DOC_PATH = OUT_DIR / "Compact_precedent_raw_and_compact_validation_2026_04_06.md"
NOTE_DOC_PATH = OUT_DIR / "Compact_precedent_validation_richer_historical_refresh_2026_04_06.md"

TARGETS = [
    {
        "company_id": "0000080424",
        "name": "The Procter & Gamble Company",
        "action_id": "capital_return.dividend_increase",
    },
    {
        "company_id": "0001018724",
        "name": "Amazon.com, Inc.",
        "action_id": "capital_return.open_market_buyback",
    },
    {
        "company_id": "0001318605",
        "name": "Tesla, Inc.",
        "action_id": "capital_return.open_market_buyback",
    },
    {
        "company_id": "0000104169",
        "name": "Walmart Inc.",
        "action_id": "capital_return.dividend_increase",
    },
]

TARGET_RAW_METRIC_ORDER = [
    "operating.revenue_ttm_provider_direct",
    "operating.revenue_ttm_lag_1y",
    "operating.ebitda_ltm_provider_direct",
    "operating.ebitda_margin_ttm",
    "capital_structure.total_debt_provider_direct",
    "capital_structure.net_debt_normalized",
    "capital_structure.lease_liabilities_sec_exact",
    "capital_structure.combined_retirement_liability",
    "liquidity.cash_and_short_term_investments_provider_direct",
    "liquidity.marketable_securities_sec_exact",
    "liquidity.revolver_undrawn",
    "liquidity.available_liquidity_normalized",
    "capital_structure.current_debt_statement_direct",
    "capital_structure.current_debt_provider_direct",
    "capital_structure.interest_expense_statement_direct",
    "capital_structure.interest_coverage",
    "market.market_cap_provider_direct",
    "market.enterprise_value",
    "market.ev_ebitda",
    "market.fcf_yield",
    "market.volatility_90d",
    "market.drawdown_90d",
    "market.credit_window_proxy",
    "market.equity_window_proxy",
    "market.credit_spread_level",
    "macro.fed_funds_effective",
    "macro.hy_oas",
]

HISTORICAL_RAW_FIELD_ORDER = [
    "base_revenue_ttm",
    "base_revenue_ttm_lag_1y",
    "base_revenue_growth_yoy",
    "base_ebitda_ttm",
    "base_total_debt",
    "base_current_debt",
    "base_cash",
    "base_available_liquidity",
    "base_interest_expense",
    "base_market_cap",
    "base_ev_ebitda",
    "base_fcf_yield",
    "base_volatility_30d",
    "base_volatility_90d",
    "base_drawdown_90d",
    "base_momentum_60d",
    "base_credit_spread_level",
    "base_equity_window_proxy",
    "base_credit_window_proxy",
    "base_net_debt",
    "base_leverage",
    "base_margin",
    "macro_fed_funds_effective",
    "macro_hy_oas",
    "macro_real_gdp_growth_yoy",
]


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and not math.isfinite(value))


def _fmt_value(value: Any) -> str:
    if _is_missing(value):
        return "`null`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, int):
        return f"`{value}`"
    if isinstance(value, float):
        return f"`{value:.4f}`"
    return f"`{value}`"


def _load_snapshot_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with gzip.open(SNAPSHOT_PATH, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            company_id = str(row.get("company_id") or "")
            if company_id:
                rows[company_id] = row
    return rows


def _target_context_lines(row: dict[str, Any], bundle: dict[str, Any]) -> list[str]:
    features = row.get("features") or {}
    support = bundle["state_vector_v1"]["support"]
    proxy = [key for key in _STATE_VECTOR_V1_FEATURES if (support.get(key) or {}).get("support_mode") == "proxy_missing_component"]
    missing = [
        key
        for key in _STATE_VECTOR_V1_FEATURES
        if (support.get(key) or {}).get("support_mode") in {None, "unsupported"}
        and _is_missing(bundle["state_vector_v1"]["values"].get(key))
    ]
    sector = (features.get("taxonomy.sector") or {}).get("value")
    subsector = (features.get("taxonomy.subsector") or {}).get("value")
    regime = (features.get("capital_structure.retirement_obligation_regime") or {}).get("value")
    return [
        f"- Sector: `{sector}`",
        f"- Subsector: `{subsector}`",
        f"- Retirement regime: `{regime}`",
        f"- Proxy compact features: {', '.join(f'`{key}`' for key in proxy) if proxy else '`None`'}",
        f"- Missing compact features: {', '.join(f'`{key}`' for key in missing) if missing else '`None`'}",
    ]


def _target_raw_table(row: dict[str, Any]) -> str:
    features = row.get("features") or {}
    lines = ["| Raw metric | Value | Support |", "|---|---:|---|"]
    for metric in TARGET_RAW_METRIC_ORDER:
        record = features.get(metric) or {}
        lines.append(
            f"| `{metric}` | {_fmt_value(record.get('value'))} | `{record.get('support_mode') or 'unsupported'}` |"
        )
    return "\n".join(lines)


def _compact_table(values: dict[str, Any]) -> str:
    lines = ["| Compact feature | Value |", "|---|---:|"]
    for key in _STATE_VECTOR_V1_FEATURES:
        lines.append(f"| `{key}` | {_fmt_value(values.get(key))} |")
    return "\n".join(lines)


def _compact_comparison_table(target_values: dict[str, Any], match_row: pd.Series) -> str:
    lines = ["| Compact feature | Target | Match |", "|---|---:|---:|"]
    for key in _STATE_VECTOR_V1_FEATURES:
        lines.append(f"| `{key}` | {_fmt_value(target_values.get(key))} | {_fmt_value(match_row.get(key))} |")
    return "\n".join(lines)


def _historical_raw_table(match_row: pd.Series) -> str:
    lines = ["| Historical raw field | Match value |", "|---|---:|"]
    for field in HISTORICAL_RAW_FIELD_ORDER:
        lines.append(f"| `{field}` | {_fmt_value(match_row.get(field))} |")
    return "\n".join(lines)


def _nonnull_compact_count(match_row: pd.Series) -> int:
    return sum(0 if _is_missing(match_row.get(key)) else 1 for key in _STATE_VECTOR_V1_FEATURES)


def _locate_match_row(historical_df: pd.DataFrame, case: Any) -> pd.Series:
    action_date = pd.to_datetime(case.decision_time).normalize()
    mask = historical_df["company_id"].astype(str).eq(str(case.company_id))
    mask &= pd.to_datetime(historical_df["action_date"], errors="coerce").dt.normalize().eq(action_date)
    normalized_action_id = historical_df.get("normalized_action_id")
    if normalized_action_id is not None and str(case.action_id or "").strip():
        id_mask = normalized_action_id.fillna("").astype(str).eq(str(case.action_id))
        if bool((mask & id_mask).any()):
            mask &= id_mask
    matches = historical_df.loc[mask]
    if matches.empty:
        raise RuntimeError(
            f"Could not locate historical row for company_id={case.company_id} action_date={action_date} action_id={case.action_id}"
        )
    return matches.iloc[0]


def _build_target_payload(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    adapted_row, _ = adapt_snapshot(row)
    adapted_row = attach_model_feature_bundle(adapted_row)
    bundle = build_model_feature_bundle(adapted_row)
    precedent_features = feature_view_from_snapshot(adapted_row, view_name="precedent")
    baseline_features = _baseline_from_world_model_features(precedent_features)
    regime = adapted_row.get("regime", {}) if isinstance(adapted_row.get("regime"), dict) else {}
    return adapted_row, bundle, {"baseline_features": baseline_features, "regime": regime}


def _render_summary_doc(
    target_rows: dict[str, dict[str, Any]],
    bundles: dict[str, dict[str, Any]],
    match_payloads: dict[str, list[dict[str, Any]]],
    historical_path: Path,
) -> str:
    lines = [
        "# Compact Precedent Validation Examples",
        "",
        f"As of `2026-04-06`. This rerun uses the richer historical outcomes artifact:",
        f"- `{historical_path}`",
        "",
        "The goal here is simple: show the compact state vector for the live target company, then show the top retrieved precedents from the current default precedent path so we can sanity-check whether the matching feels economically sensible.",
        "",
    ]
    for target in TARGETS:
        company_id = target["company_id"]
        lines.extend(
            [
                f"## {target['name']} (`{company_id}`)",
                "",
                f"- Candidate action reviewed: `{target['action_id']}`",
                "",
                "### Target Compact State Vector",
                "",
                _compact_table(bundles[company_id]["state_vector_v1"]["values"]),
                "",
                "### Support Context",
                "",
                *_target_context_lines(target_rows[company_id], bundles[company_id]),
                "",
                "### Top 3 Retrieved Precedents",
                "",
            ]
        )
        for idx, payload in enumerate(match_payloads[company_id], start=1):
            lines.extend(
                [
                    f"#### Match {idx}",
                    "",
                    f"- Precedent id: `{payload['precedent_id']}`",
                    f"- Ticker: `{payload['ticker']}`",
                    f"- Action: `{payload['action_id']}`",
                    f"- Decision time: `{payload['decision_time']}`",
                    f"- Similarity score: `{payload['similarity_score']:.4f}`",
                    f"- Non-null compact features on match row: `{payload['nonnull_compact_features']}/{len(_STATE_VECTOR_V1_FEATURES)}`",
                    "",
                    _compact_comparison_table(
                        bundles[company_id]["state_vector_v1"]["values"],
                        payload["historical_row"],
                    ),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _render_detail_doc(
    target_rows: dict[str, dict[str, Any]],
    bundles: dict[str, dict[str, Any]],
    match_payloads: dict[str, list[dict[str, Any]]],
    historical_path: Path,
) -> str:
    lines = [
        "# Compact Precedent Raw And Compact Validation",
        "",
        f"As of `2026-04-06`. This rerun uses the richer historical outcomes artifact:",
        f"- `{historical_path}`",
        "",
        "This note is meant for manual sanity-checking. It shows both the target compact state vector and the raw inputs behind it, then compares those with the top historical precedent rows currently retrieved by the compact precedent path.",
        "",
        "Important caveat: historical `state_vector_v1.market_stress` now falls back to a macro-VIX proxy when company-level 90-day price windows are unavailable, so the compact feature is denser than the raw `base_volatility_90d` / `base_drawdown_90d` inputs.",
        "Historical raw baseline fields are shown in their source units, which are generally USD millions.",
        "",
    ]
    for target in TARGETS:
        company_id = target["company_id"]
        lines.extend(
            [
                f"## {target['name']} (`{company_id}`)",
                "",
                f"- Candidate action reviewed: `{target['action_id']}`",
                "",
                "### Target Compact State Vector",
                "",
                _compact_table(bundles[company_id]["state_vector_v1"]["values"]),
                "",
                "### Target Raw Inputs Behind The Compact Features",
                "",
                _target_raw_table(target_rows[company_id]),
                "",
                *_target_context_lines(target_rows[company_id], bundles[company_id]),
                "",
                "### Top Historical Precedents",
                "",
            ]
        )
        for idx, payload in enumerate(match_payloads[company_id], start=1):
            lines.extend(
                [
                    f"#### Match {idx}",
                    "",
                    f"- Precedent id: `{payload['precedent_id']}`",
                    f"- Ticker: `{payload['ticker']}`",
                    f"- Action: `{payload['action_id']}`",
                    f"- Decision time: `{payload['decision_time']}`",
                    f"- Similarity score: `{payload['similarity_score']:.4f}`",
                    f"- Non-null compact features on match row: `{payload['nonnull_compact_features']}/{len(_STATE_VECTOR_V1_FEATURES)}`",
                    "",
                    _compact_comparison_table(
                        bundles[company_id]["state_vector_v1"]["values"],
                        payload["historical_row"],
                    ),
                    "",
                    _historical_raw_table(payload["historical_row"]),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _render_note(match_payloads: dict[str, list[dict[str, Any]]], historical_path: Path) -> str:
    lines = [
        "# Compact Precedent Validation Richer Historical Refresh",
        "",
        f"As of `2026-04-06`, I reran the compact precedent validation examples against:",
        f"- `{historical_path}`",
        "",
        "## What Changed",
        "- The matched historical rows now come from the richer historical outcomes artifact rather than the thinner normalized-full artifact.",
        "- That materially densifies the historical compact feature set for `growth`, `gross_obligation_burden`, `liquidity_flexibility`, `interest_coverage`, and `cash_generation`.",
        "- Historical `market_access` is now fed by richer price/macro-derived proxy inputs, and `market_stress` now has a macro-VIX fallback when company-level 90-day price windows are unavailable.",
        "",
        "## Match Density Snapshot",
    ]
    for target in TARGETS:
        company_id = target["company_id"]
        counts = [payload["nonnull_compact_features"] for payload in match_payloads[company_id]]
        avg_count = sum(counts) / max(1, len(counts))
        lines.append(
            f"- `{target['name']}`: top-3 matches average `{avg_count:.2f}/{len(_STATE_VECTOR_V1_FEATURES)}` non-null compact features "
            f"(matches: {', '.join(str(count) for count in counts)})"
        )
    lines.extend(
        [
            "",
            "## Files",
            f"- `{SUMMARY_DOC_PATH}`",
            f"- `{DETAIL_DOC_PATH}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    target_rows = _load_snapshot_rows()
    historical_path = _default_precedent_outcomes_path()
    historical_df = augment_precedent_state_vector_columns(pd.read_parquet(historical_path))
    retrieval_index = build_precedent_retrieval_index(historical_df)

    bundles: dict[str, dict[str, Any]] = {}
    match_payloads: dict[str, list[dict[str, Any]]] = {}

    for target in TARGETS:
        company_id = target["company_id"]
        row = target_rows[company_id]
        adapted_row, bundle, precedent_payload = _build_target_payload(row)
        bundles[company_id] = bundle
        pack = build_precedent_pack_v2(
            candidate_id=f"validation:{company_id}",
            run_id=f"validation:{uuid.uuid4()}",
            company_id=company_id,
            action_id=target["action_id"],
            action_subtype=None,
            action_params={},
            candidate_features=precedent_payload["baseline_features"],
            candidate_regime=precedent_payload["regime"],
            retrieval_index=retrieval_index,
            top_k=3,
            min_k=3,
        )
        payloads: list[dict[str, Any]] = []
        for case in pack.retrieved_cohorts[:3]:
            hist_row = _locate_match_row(historical_df, case)
            payloads.append(
                {
                    "precedent_id": case.precedent_id,
                    "company_id": case.company_id,
                    "ticker": hist_row.get("ticker"),
                    "action_id": case.action_id,
                    "decision_time": case.decision_time,
                    "similarity_score": float(case.similarity_score),
                    "historical_row": hist_row,
                    "nonnull_compact_features": _nonnull_compact_count(hist_row),
                }
            )
        match_payloads[company_id] = payloads

    SUMMARY_DOC_PATH.write_text(_render_summary_doc(target_rows, bundles, match_payloads, historical_path))
    DETAIL_DOC_PATH.write_text(_render_detail_doc(target_rows, bundles, match_payloads, historical_path))
    NOTE_DOC_PATH.write_text(_render_note(match_payloads, historical_path))


if __name__ == "__main__":
    main()
