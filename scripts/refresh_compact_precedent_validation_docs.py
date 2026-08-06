#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_feature_bundle import _STATE_VECTOR_V1_FEATURES, build_model_feature_bundle

ARTIFACT_PATH = REPO_ROOT / "out/materialized_feedback_20260405/company_state_snapshots_asof=2024-12-31.input_layer_v1_smart_normalized_with_sec.feedback_pipeline.full_inputs_v3.jsonl.gz"
VALIDATION_DOC_PATH = REPO_ROOT / "out/metric_explainers/Compact_precedent_validation_examples_2026_04_05.md"
RAW_VALIDATION_DOC_PATH = REPO_ROOT / "out/metric_explainers/Compact_precedent_raw_and_compact_validation_2026_04_05.md"
FOLLOWUP_NOTE_PATH = REPO_ROOT / "out/metric_explainers/State_vector_input_completeness_followup_2026_04_05.md"

COMPANY_IDS = {
    "0000080424": "The Procter & Gamble Company",
    "0001018724": "Amazon.com, Inc.",
    "0001318605": "Tesla, Inc.",
    "0000104169": "Walmart Inc.",
}

RAW_METRIC_ORDER = [
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


def _load_snapshots() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with gzip.open(ARTIFACT_PATH, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            company_id = str(row.get("company_id") or "")
            if company_id in COMPANY_IDS:
                rows[company_id] = row
    missing = set(COMPANY_IDS) - set(rows)
    if missing:
        raise RuntimeError(f"Missing rows in artifact for company ids: {sorted(missing)}")
    return rows


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


def _feature_table(bundle: dict[str, Any]) -> str:
    values = bundle["state_vector_v1"]["values"]
    lines = ["| Feature | Value |", "|---|---:|"]
    for key in _STATE_VECTOR_V1_FEATURES:
        lines.append(f"| `{key}` | {_fmt_value(values.get(key))} |")
    return "\n".join(lines)


def _support_lines(row: dict[str, Any], bundle: dict[str, Any], *, missing_label: str) -> str:
    features = row.get("features") or {}
    support = bundle["state_vector_v1"]["support"]
    proxy = [key for key in _STATE_VECTOR_V1_FEATURES if (support.get(key) or {}).get("support_mode") == "proxy_missing_component"]
    missing = [key for key in _STATE_VECTOR_V1_FEATURES if (support.get(key) or {}).get("support_mode") in {None, "unsupported"} and _is_missing(bundle["state_vector_v1"]["values"].get(key))]
    sector = (features.get("taxonomy.sector") or {}).get("value")
    subsector = (features.get("taxonomy.subsector") or {}).get("value")
    regime = (features.get("capital_structure.retirement_obligation_regime") or {}).get("value")
    proxy_text = ", ".join(f"`{key}`" for key in proxy) if proxy else "`None`"
    missing_text = ", ".join(f"`{key}`" for key in missing) if missing else "`None`"
    return "\n".join(
        [
            f"- Sector: `{sector}`",
            f"- Subsector: `{subsector}`",
            f"- Retirement regime: `{regime}`",
            f"- Proxy features: {proxy_text}",
            f"- {missing_label}: {missing_text}",
        ]
    )


def _raw_metric_table(row: dict[str, Any]) -> str:
    features = row.get("features") or {}
    lines = ["| Raw metric | Value | Support |", "|---|---:|---|"]
    for metric in RAW_METRIC_ORDER:
        record = features.get(metric) or {}
        value = record.get("value")
        support_mode = record.get("support_mode") or "unsupported"
        lines.append(f"| `{metric}` | {_fmt_value(value)} | `{support_mode}` |")
    return "\n".join(lines)


def _replace_block(section: str, start_heading: str, end_heading: str, body: str) -> str:
    pattern = re.compile(
        rf"({re.escape(start_heading)}\n\n)(.*?)(\n{re.escape(end_heading)})",
        flags=re.S,
    )
    match = pattern.search(section)
    if not match:
        raise RuntimeError(f"Could not replace block between {start_heading!r} and {end_heading!r}")
    return section[: match.start()] + match.group(1) + body + match.group(3) + section[match.end() :]


def _refresh_target_rows(section: str, bundle: dict[str, Any]) -> str:
    values = bundle["state_vector_v1"]["values"]
    out_lines: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "`state_vector_v1." not in stripped:
            out_lines.append(line)
            continue
        parts = line.split("|")
        cells = parts[1:-1]
        if len(cells) < 2:
            out_lines.append(line)
            continue
        key = cells[0].strip().strip("`")
        if key not in values:
            out_lines.append(line)
            continue
        cells[1] = f" {_fmt_value(values.get(key))} "
        out_lines.append("|" + "|".join(cells) + "|")
    return "\n".join(out_lines)


def _replace_company_section(doc_text: str, heading_line: str, transform) -> str:
    start = doc_text.find(heading_line)
    if start == -1:
        raise RuntimeError(f"Could not find heading {heading_line!r}")
    next_section = doc_text.find("\n## ", start + len(heading_line))
    end = len(doc_text) if next_section == -1 else next_section + 1
    section = doc_text[start:end]
    updated = transform(section)
    return doc_text[:start] + updated + doc_text[end:]


def _refresh_validation_examples(doc_text: str, company_id: str, bundle: dict[str, Any], row: dict[str, Any]) -> str:
    heading = f"## {COMPANY_IDS[company_id]} (`{company_id}`)"

    def transform(section: str) -> str:
        section = _replace_block(section, "### Target Compact State Vector", "### Support Context", _feature_table(bundle) + "\n")
        section = _replace_block(section, "### Support Context", "### Top 3 Retrieved Precedents", _support_lines(row, bundle, missing_label="Missing features") + "\n")
        section = _refresh_target_rows(section, bundle)
        return section

    return _replace_company_section(doc_text, heading, transform)


def _refresh_raw_validation(doc_text: str, company_id: str, bundle: dict[str, Any], row: dict[str, Any]) -> str:
    heading = f"## {COMPANY_IDS[company_id]} (`{company_id}`)"

    def transform(section: str) -> str:
        section = _replace_block(section, "### Target Compact State Vector", "### Target Raw Inputs Behind The Compact Features", _feature_table(bundle) + "\n")
        raw_plus_support = _raw_metric_table(row) + "\n\n" + _support_lines(row, bundle, missing_label="Missing compact features")
        section = _replace_block(section, "### Target Raw Inputs Behind The Compact Features", "### Top Historical Precedents", raw_plus_support + "\n")
        section = _refresh_target_rows(section, bundle)
        return section

    return _replace_company_section(doc_text, heading, transform)


def _refresh_followup_note(note_text: str) -> str:
    marker = "## Remaining Caveat"
    insert = (
        "## Validation Packs Refreshed\n"
        "- Refreshed `./out/metric_explainers/Compact_precedent_validation_examples_2026_04_05.md` to read target-side compact values from `full_inputs_v3`.\n"
        "- Refreshed `./out/metric_explainers/Compact_precedent_raw_and_compact_validation_2026_04_05.md` so the target raw-input tables now show the lagged revenue metric and the raw liquidity/current-debt components that feed `state_vector_v1.liquidity_flexibility`.\n\n"
    )
    if insert.strip() in note_text:
        return note_text
    index = note_text.find(marker)
    if index == -1:
        return note_text.rstrip() + "\n\n" + insert
    return note_text[:index] + insert + note_text[index:]


def main() -> None:
    rows = _load_snapshots()
    bundles = {company_id: build_model_feature_bundle(row) for company_id, row in rows.items()}

    validation_doc = VALIDATION_DOC_PATH.read_text()
    for company_id in ["0001018724", "0001318605", "0000104169"]:
        validation_doc = _refresh_validation_examples(validation_doc, company_id, bundles[company_id], rows[company_id])
    VALIDATION_DOC_PATH.write_text(validation_doc)

    raw_doc = RAW_VALIDATION_DOC_PATH.read_text()
    for company_id in ["0000080424", "0001018724", "0001318605", "0000104169"]:
        raw_doc = _refresh_raw_validation(raw_doc, company_id, bundles[company_id], rows[company_id])
    RAW_VALIDATION_DOC_PATH.write_text(raw_doc)

    note_text = FOLLOWUP_NOTE_PATH.read_text()
    FOLLOWUP_NOTE_PATH.write_text(_refresh_followup_note(note_text))


if __name__ == "__main__":
    main()
