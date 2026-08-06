#!/usr/bin/env python3
"""Repair operating growth/history metrics in a materialized company-state artifact.

This pass fills a narrow set of growth and trend metrics that are often empty in
the artifact even though SEC companyfacts already contains enough quarterly
history to recover them.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPAIR_METRICS = [
    "operating.revenue_yoy_last_q",
    "operating.revenue_cagr_3y",
    "operating.ebitda_margin_trend_8q",
    "operating.margin_volatility_8q",
]

REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueServicesNet",
    "Revenues",
]
OPERATING_INCOME_TTM_CONCEPTS = ["OperatingIncomeLoss"]
NET_INCOME_TTM_CONCEPTS = ["NetIncomeLoss"]
INTEREST_TTM_CONCEPTS = ["InterestExpense"]
TAX_TTM_CONCEPTS = ["IncomeTaxExpenseBenefit"]
MAX_SEC_FACT_AGE_DAYS = 550
DEPRECIATION_TTM_CONCEPT_GROUPS = [
    ["DepreciationAmortizationAndAccretionNet"],
    ["DepreciationDepletionAndAmortization"],
    ["DepreciationAndAmortization"],
    ["Depreciation"],
    ["Depreciation", "AmortizationOfIntangibleAssets"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--companyfacts-root", required=True, help="SEC companyfacts folder")
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


def _parse_iso_date(value: Any) -> Optional[date]:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:  # noqa: BLE001
        return None


def _node_support(node: Dict[str, Any] | None) -> str:
    if not node:
        return "unsupported"
    return str(node.get("support_mode") or "unsupported")


def _candidate_units_map(companyfacts: dict, concept_name: str) -> dict | None:
    for taxonomy in ("us-gaap", "dei", "ifrs-full"):
        facts = (companyfacts.get("facts") or {}).get(taxonomy) or {}
        if concept_name in facts:
            return facts[concept_name].get("units") or {}
    return None


def _load_companyfacts(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _collect_duration_entries(companyfacts: dict, concept_name: str, as_of_date: str) -> list[dict[str, Any]]:
    units_map = _candidate_units_map(companyfacts, concept_name)
    if not units_map:
        return []
    as_of_dt = _parse_iso_date(as_of_date)
    if as_of_dt is None:
        return []
    rows = []
    for unit, entries in units_map.items():
        if unit.upper() != "USD":
            continue
        for entry in entries:
            start_dt = _parse_iso_date(entry.get("start"))
            end_dt = _parse_iso_date(entry.get("end"))
            filed_dt = _parse_iso_date(entry.get("filed"))
            value = entry.get("val")
            if start_dt is None or end_dt is None or value is None:
                continue
            if end_dt > as_of_dt:
                continue
            if filed_dt is not None and filed_dt > as_of_dt:
                continue
            if (as_of_dt - end_dt).days > MAX_SEC_FACT_AGE_DAYS:
                continue
            rows.append(
                {
                    "concept": concept_name,
                    "start": start_dt,
                    "end": end_dt,
                    "filed": filed_dt or end_dt,
                    "value": float(value),
                    "fy": entry.get("fy"),
                    "fp": entry.get("fp"),
                    "frame": entry.get("frame"),
                    "form": entry.get("form"),
                    "duration_days": max(1, (end_dt - start_dt).days + 1),
                }
            )
    rows.sort(key=lambda item: (item["end"], item["filed"], item["duration_days"]))
    return rows


def _compute_ttm_from_concept(companyfacts: dict, concept_name: str, as_of_date: str) -> tuple[float | None, dict[str, Any] | None]:
    entries = _collect_duration_entries(companyfacts, concept_name, as_of_date)
    if not entries:
        return None, None

    latest = max(entries, key=lambda item: (item["end"], item["filed"], item["duration_days"]))
    latest_fp = str(latest.get("fp") or "").upper()
    if latest_fp == "FY" or latest["duration_days"] >= 300:
        return latest["value"], {
            "concept": concept_name,
            "mode": "latest_fy",
            "end": latest["end"].isoformat(),
            "filed": latest["filed"].isoformat(),
            "fy": latest.get("fy"),
            "fp": latest.get("fp"),
            "frame": latest.get("frame"),
            "form": latest.get("form"),
            "formula": "latest_fiscal_year_value",
        }

    if latest_fp not in {"Q1", "Q2", "Q3"}:
        return None, None

    current_fy = latest.get("fy")
    if current_fy is None:
        return None, None
    try:
        prior_fy = int(current_fy) - 1
    except Exception:  # noqa: BLE001
        return None, None

    annual = None
    prior_same = None
    for entry in entries:
        entry_fp = str(entry.get("fp") or "").upper()
        if entry.get("fy") == prior_fy and entry_fp == "FY":
            if annual is None or (entry["end"], entry["filed"], entry["duration_days"]) > (annual["end"], annual["filed"], annual["duration_days"]):
                annual = entry
        if entry.get("fy") == prior_fy and entry_fp == latest_fp:
            if prior_same is None or (entry["end"], entry["filed"], entry["duration_days"]) > (prior_same["end"], prior_same["filed"], prior_same["duration_days"]):
                prior_same = entry

    if annual is None or prior_same is None:
        return None, None

    return float(latest["value"] + annual["value"] - prior_same["value"]), {
        "concept": concept_name,
        "mode": "ytd_plus_prior_fy_minus_prior_ytd",
        "latest": {
            "end": latest["end"].isoformat(),
            "filed": latest["filed"].isoformat(),
            "fy": latest.get("fy"),
            "fp": latest.get("fp"),
            "frame": latest.get("frame"),
            "form": latest.get("form"),
            "value": latest["value"],
        },
        "prior_fy": {
            "end": annual["end"].isoformat(),
            "filed": annual["filed"].isoformat(),
            "fy": annual.get("fy"),
            "fp": annual.get("fp"),
            "frame": annual.get("frame"),
            "form": annual.get("form"),
            "value": annual["value"],
        },
        "prior_same_period": {
            "end": prior_same["end"].isoformat(),
            "filed": prior_same["filed"].isoformat(),
            "fy": prior_same.get("fy"),
            "fp": prior_same.get("fp"),
            "frame": prior_same.get("frame"),
            "form": prior_same.get("form"),
            "value": prior_same["value"],
        },
        "formula": "latest_ytd + prior_fy - prior_same_period_ytd",
    }


def _collect_duration_entries_all(companyfacts: dict, concept_name: str, as_of_date: str) -> list[dict[str, Any]]:
    units_map = _candidate_units_map(companyfacts, concept_name)
    if not units_map:
        return []
    as_of_dt = _parse_iso_date(as_of_date)
    if as_of_dt is None:
        return []
    rows = []
    for unit, entries in units_map.items():
        if unit.upper() != "USD":
            continue
        for entry in entries:
            start_dt = _parse_iso_date(entry.get("start"))
            end_dt = _parse_iso_date(entry.get("end"))
            filed_dt = _parse_iso_date(entry.get("filed")) or end_dt
            value = entry.get("val")
            if start_dt is None or end_dt is None or value is None:
                continue
            if end_dt > as_of_dt or (filed_dt is not None and filed_dt > as_of_dt):
                continue
            rows.append(
                {
                    "concept": concept_name,
                    "start": start_dt,
                    "end": end_dt,
                    "filed": filed_dt or end_dt,
                    "value": float(value),
                    "fy": entry.get("fy"),
                    "fp": str(entry.get("fp") or "").upper() or None,
                    "frame": entry.get("frame"),
                    "form": entry.get("form"),
                    "duration_days": max(1, (end_dt - start_dt).days + 1),
                }
            )
    rows.sort(key=lambda item: (item["end"], item["filed"], item["duration_days"]))
    return rows


def _companyfacts_priority_ttm(
    companyfacts: Dict[str, Any] | None,
    concepts: list[str],
    *,
    as_of_date: str,
) -> tuple[float | None, Dict[str, Any] | None]:
    if companyfacts is None:
        return None, None
    for concept_name in concepts:
        value, meta = _compute_ttm_from_concept(companyfacts, concept_name, as_of_date)
        if value is not None:
            return value, meta
    return None, None


def _companyfacts_depreciation_ttm(
    companyfacts: Dict[str, Any] | None,
    *,
    as_of_date: str,
) -> tuple[float | None, Dict[str, Any] | None, bool]:
    if companyfacts is None:
        return None, None, False
    for concept_group in DEPRECIATION_TTM_CONCEPT_GROUPS:
        if len(concept_group) == 1:
            value, meta = _compute_ttm_from_concept(companyfacts, concept_group[0], as_of_date)
            if value is not None:
                return value, meta, concept_group[0] != "Depreciation"
        else:
            parts = []
            parts_meta = []
            for concept_name in concept_group:
                part_value, part_meta = _compute_ttm_from_concept(companyfacts, concept_name, as_of_date)
                if part_value is None:
                    parts = []
                    break
                parts.append(part_value)
                parts_meta.append(part_meta)
            if parts:
                return float(sum(parts)), {
                    "mode": "sum_concepts",
                    "components": parts_meta,
                    "formula": "sum_component_ttm_values",
                }, True
    return None, None, False


def _companyfacts_provenance(companyfacts_path: Path, *, as_of_time: str, computed_at: str) -> list[dict[str, Any]]:
    return [
        {
            "artifact_type": "SecCompanyFacts",
            "artifact_id": f"sec_companyfacts:{companyfacts_path.name}",
            "source": str(companyfacts_path),
            "published_at": as_of_time,
            "ingested_at": computed_at,
            "hash": None,
        }
    ]


def _base_repaired_node(node: Dict[str, Any], *, computed_at: str) -> Dict[str, Any]:
    repaired = copy.deepcopy(node)
    repaired["computed_at"] = computed_at
    repaired["missing_reason"] = None
    repaired["quality_flags"] = repaired.get("quality_flags") or None
    return repaired


def _parse_breakdown_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    date_part = text.split(" ", 1)[0]
    try:
        return date.fromisoformat(date_part)
    except ValueError:
        return None


def _should_refresh_existing_metric(
    node: Dict[str, Any] | None,
    *,
    replacement_period: date,
    period_key: str,
    fallback_prefix: str,
) -> bool:
    if not node or node.get("value") is None:
        return True
    breakdown = node.get("component_breakdown") or {}
    existing_period = _parse_breakdown_date(breakdown.get(period_key))
    if existing_period is not None and existing_period < replacement_period:
        return True
    fallback_used = str(node.get("fallback_used") or "")
    if fallback_used.startswith(fallback_prefix):
        return True
    return False


def _choose_revenue_concept(companyfacts: Dict[str, Any] | None, *, as_of_date: str) -> tuple[str | None, list[dict[str, Any]]]:
    if companyfacts is None:
        return None, []
    best_choice: tuple[tuple[Any, ...], str, list[dict[str, Any]]] | None = None
    for preference_idx, concept_name in enumerate(REVENUE_CONCEPTS):
        entries = _collect_duration_entries_all(companyfacts, concept_name, as_of_date)
        if not entries:
            continue
        latest_entry = max(entries, key=lambda item: (item["end"], item["filed"], item["duration_days"]))
        distinct_period_count = len({entry["end"] for entry in entries})
        score = (
            latest_entry["end"],
            latest_entry["filed"],
            distinct_period_count,
            -preference_idx,
        )
        if best_choice is None or score > best_choice[0]:
            best_choice = (score, concept_name, entries)
    if best_choice is None:
        return None, []
    return best_choice[1], best_choice[2]


def _build_revenue_single_quarter_series(
    companyfacts: Dict[str, Any] | None,
    *,
    as_of_date: str,
) -> tuple[list[dict[str, Any]], str | None]:
    concept_name, entries = _choose_revenue_concept(companyfacts, as_of_date=as_of_date)
    if not entries:
        return [], concept_name

    exact_by_end: dict[date, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    ytd_by_key: dict[tuple[Any, Any], tuple[tuple[Any, ...], dict[str, Any]]] = {}

    for entry in entries:
        fp = entry.get("fp")
        duration_days = entry.get("duration_days") or 0
        frame = str(entry.get("frame") or "")
        looks_exact_quarter = duration_days <= 110 or ("Q" in frame and duration_days <= 120)
        if looks_exact_quarter:
            rank = (1 if "Q" in frame else 0, entry["filed"], -abs(duration_days - 91))
            current = exact_by_end.get(entry["end"])
            if current is None or rank > current[0]:
                exact_by_end[entry["end"]] = (rank, entry)
        if fp in {"Q1", "Q2", "Q3", "FY"}:
            rank = (entry["filed"], duration_days)
            current = ytd_by_key.get((entry.get("fy"), fp))
            if current is None or rank > current[0]:
                ytd_by_key[(entry.get("fy"), fp)] = (rank, entry)

    series: list[dict[str, Any]] = []
    for period_end in sorted({entry["end"] for entry in entries}):
        if period_end in exact_by_end:
            exact = exact_by_end[period_end][1]
            series.append(
                {
                    "period_end": period_end,
                    "value": exact["value"],
                    "basis": "as_reported_quarter",
                    "concept": concept_name,
                    "fy": exact.get("fy"),
                    "fp": exact.get("fp"),
                }
            )
            continue

        ending_candidates = [row for _, row in ytd_by_key.values() if row["end"] == period_end]
        if not ending_candidates:
            continue
        current = sorted(ending_candidates, key=lambda item: (item["filed"], item["duration_days"]))[-1]
        fiscal_year = current.get("fy")
        fiscal_period = current.get("fp")
        if fiscal_period == "Q1":
            series.append(
                {
                    "period_end": period_end,
                    "value": current["value"],
                    "basis": "as_reported_without_prior_quarter",
                    "concept": concept_name,
                    "fy": fiscal_year,
                    "fp": fiscal_period,
                }
            )
        elif fiscal_period in {"Q2", "Q3"}:
            prior_fp = "Q1" if fiscal_period == "Q2" else "Q2"
            prior = ytd_by_key.get((fiscal_year, prior_fp))
            if prior is not None:
                series.append(
                    {
                        "period_end": period_end,
                        "value": current["value"] - prior[1]["value"],
                        "basis": "derived_from_ytd_delta",
                        "concept": concept_name,
                        "fy": fiscal_year,
                        "fp": fiscal_period,
                    }
                )
        elif fiscal_period == "FY":
            prior = ytd_by_key.get((fiscal_year, "Q3"))
            if prior is not None:
                series.append(
                    {
                        "period_end": period_end,
                        "value": current["value"] - prior[1]["value"],
                        "basis": "derived_from_ytd_delta",
                        "concept": concept_name,
                        "fy": fiscal_year,
                        "fp": "Q4",
                    }
                )
    deduped = {row["period_end"]: row for row in series}
    return [deduped[key] for key in sorted(deduped)], concept_name


def _build_ttm_revenue_series(
    companyfacts: Dict[str, Any] | None,
    *,
    as_of_date: str,
) -> tuple[list[dict[str, Any]], str | None]:
    concept_name, entries = _choose_revenue_concept(companyfacts, as_of_date=as_of_date)
    if not entries:
        return [], concept_name
    period_ends = sorted(
        {
            entry["end"].isoformat()
            for entry in entries
            if entry.get("fp") in {"Q1", "Q2", "Q3", "FY"} or (entry.get("duration_days") or 0) >= 300
        }
    )
    observations = []
    for period_end in period_ends:
        value, meta = _compute_ttm_from_concept(companyfacts, concept_name, period_end)
        if value is not None:
            observations.append(
                {
                    "period_end": _parse_iso_date(period_end),
                    "value": value,
                    "meta": meta,
                    "concept": concept_name,
                    "exact": True,
                }
            )
    return observations, concept_name


def _operating_earnings_ttm_at(
    companyfacts: Dict[str, Any] | None,
    *,
    as_of_date: str,
) -> tuple[float | None, bool]:
    operating_income, _ = _companyfacts_priority_ttm(
        companyfacts,
        OPERATING_INCOME_TTM_CONCEPTS,
        as_of_date=as_of_date,
    )
    depreciation, _, depreciation_exact = _companyfacts_depreciation_ttm(
        companyfacts,
        as_of_date=as_of_date,
    )
    if operating_income is not None and depreciation is not None:
        return operating_income + depreciation, depreciation_exact

    net_income, _ = _companyfacts_priority_ttm(
        companyfacts,
        NET_INCOME_TTM_CONCEPTS,
        as_of_date=as_of_date,
    )
    interest_expense, _ = _companyfacts_priority_ttm(
        companyfacts,
        INTEREST_TTM_CONCEPTS,
        as_of_date=as_of_date,
    )
    tax, _ = _companyfacts_priority_ttm(
        companyfacts,
        TAX_TTM_CONCEPTS,
        as_of_date=as_of_date,
    )
    depreciation, _, depreciation_exact = _companyfacts_depreciation_ttm(
        companyfacts,
        as_of_date=as_of_date,
    )
    if None in (net_income, interest_expense, tax, depreciation):
        return None, False
    return float(net_income + interest_expense + tax + depreciation), depreciation_exact


def _build_ttm_margin_series(
    companyfacts: Dict[str, Any] | None,
    *,
    as_of_date: str,
) -> tuple[list[dict[str, Any]], str | None]:
    revenue_ttm_series, concept_name = _build_ttm_revenue_series(companyfacts, as_of_date=as_of_date)
    observations = []
    for revenue_row in revenue_ttm_series:
        period_end = revenue_row.get("period_end")
        if period_end is None or revenue_row.get("value") in (None, 0):
            continue
        earnings_value, earnings_exact = _operating_earnings_ttm_at(
            companyfacts,
            as_of_date=period_end.isoformat(),
        )
        if earnings_value is None:
            continue
        observations.append(
            {
                "period_end": period_end,
                "margin": float(earnings_value) / float(revenue_row["value"]),
                "exact": revenue_row.get("exact", False) and earnings_exact,
            }
        )
    return observations, concept_name


def _find_best_prior_match(
    rows: list[dict[str, Any]],
    *,
    latest_end: date,
    min_days: int,
    max_days: int,
    target_days: int,
) -> dict[str, Any] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        period_end = row.get("period_end")
        if period_end is None:
            continue
        delta_days = (latest_end - period_end).days
        if min_days <= delta_days <= max_days:
            candidates.append((abs(delta_days - target_days), row))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]["period_end"]))[0][1]


def _linear_slope(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    n = len(values)
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return None
    numerator = sum((i - x_mean) * (value - y_mean) for i, value in enumerate(values))
    return numerator / denominator


def repair_revenue_yoy_last_q(
    *,
    features: Dict[str, Any],
    companyfacts: Dict[str, Any] | None,
    companyfacts_path: Path,
    computed_at: str,
    as_of_time: str,
) -> bool:
    target = features.get("operating.revenue_yoy_last_q")
    if not target:
        return False

    rows, concept_name = _build_revenue_single_quarter_series(companyfacts, as_of_date=as_of_time[:10])
    if len(rows) < 2:
        return False
    latest = rows[-1]
    latest_end = latest["period_end"]
    if not _should_refresh_existing_metric(
        target,
        replacement_period=latest_end,
        period_key="latest_period",
        fallback_prefix="sec_companyfacts_quarterly_revenue_history",
    ):
        return False
    prior = _find_best_prior_match(
        rows[:-1],
        latest_end=latest_end,
        min_days=270,
        max_days=460,
        target_days=365,
    )
    if prior is None or prior.get("value") in (None, 0) or latest.get("value") is None:
        return False

    latest_basis = str(latest.get("basis") or "")
    prior_basis = str(prior.get("basis") or "")
    quality_flags: list[str] = []
    if latest_basis == "derived_from_ytd_delta" or prior_basis == "derived_from_ytd_delta":
        quality_flags.append("quarter_value_derived_from_ytd_delta")
    if latest_basis != prior_basis:
        quality_flags.append("mixed_quarter_value_basis")

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = (float(latest["value"]) - float(prior["value"])) / float(prior["value"])
    repaired["fallback_used"] = "sec_companyfacts_quarterly_revenue_history"
    repaired["support_mode"] = "exact" if not quality_flags else "proxy_missing_component"
    repaired["provenance"] = _companyfacts_provenance(companyfacts_path, as_of_time=as_of_time, computed_at=computed_at)
    repaired["component_breakdown"] = {
        "formula": "latest_quarter_revenue / prior_year_same_quarter_revenue - 1",
        "latest_revenue": float(latest["value"]),
        "prior_revenue": float(prior["value"]),
        "latest_period": f"{latest_end.isoformat()} 00:00:00+00:00",
        "prior_period": f"{prior['period_end'].isoformat()} 00:00:00+00:00",
        "target_prior_period": f"{(latest_end - timedelta(days=365)).isoformat()} 00:00:00+00:00",
        "matching_window_days": [270, 460],
        "match_basis": "fiscal_quarter_period_end",
        "latest_value_basis": latest_basis or None,
        "prior_value_basis": prior_basis or None,
        "source_concept": concept_name,
    }
    repaired["quality_flags"] = quality_flags or None
    features["operating.revenue_yoy_last_q"] = repaired
    return True


def repair_revenue_cagr_3y(
    *,
    features: Dict[str, Any],
    companyfacts: Dict[str, Any] | None,
    companyfacts_path: Path,
    computed_at: str,
    as_of_time: str,
) -> bool:
    target = features.get("operating.revenue_cagr_3y")
    if not target:
        return False

    rows, concept_name = _build_ttm_revenue_series(companyfacts, as_of_date=as_of_time[:10])
    if len(rows) < 2:
        return False
    latest = rows[-1]
    latest_end = latest["period_end"]
    if latest_end is None or latest.get("value") in (None, 0):
        return False
    if not _should_refresh_existing_metric(
        target,
        replacement_period=latest_end,
        period_key="latest_period",
        fallback_prefix="sec_companyfacts_ttm_revenue_history",
    ):
        return False
    prior = _find_best_prior_match(
        rows[:-1],
        latest_end=latest_end,
        min_days=365 * 2,
        max_days=365 * 4,
        target_days=int(365.25 * 3),
    )
    if prior is None or prior.get("value") in (None, 0):
        return False

    elapsed_years = (latest_end - prior["period_end"]).days / 365.25
    if elapsed_years <= 0:
        return False
    latest_value = float(latest["value"])
    prior_value = float(prior["value"])
    # CAGR is only well-defined for strictly positive revenue anchors.
    if latest_value <= 0 or prior_value <= 0:
        return False
    computed_value = (latest_value / prior_value) ** (1.0 / elapsed_years) - 1.0
    if isinstance(computed_value, complex) or not math.isfinite(float(computed_value)):
        return False

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = float(computed_value)
    repaired["fallback_used"] = "sec_companyfacts_ttm_revenue_history"
    repaired["support_mode"] = "exact" if latest.get("exact") and prior.get("exact") else "proxy_missing_component"
    repaired["provenance"] = _companyfacts_provenance(companyfacts_path, as_of_time=as_of_time, computed_at=computed_at)
    repaired["component_breakdown"] = {
        "formula": "(latest_revenue / prior_revenue) ** (1 / elapsed_years) - 1",
        "latest_revenue": float(latest["value"]),
        "prior_revenue": float(prior["value"]),
        "latest_period": f"{latest_end.isoformat()} 00:00:00+00:00",
        "prior_period": f"{prior['period_end'].isoformat()} 00:00:00+00:00",
        "elapsed_years": elapsed_years,
        "target_prior_period": f"{(latest_end - timedelta(days=int(365.25 * 3))).isoformat()} 00:00:00+00:00",
        "source_concept": concept_name,
        "latest_mode": (latest.get("meta") or {}).get("mode"),
        "prior_mode": (prior.get("meta") or {}).get("mode"),
    }
    repaired["quality_flags"] = None
    features["operating.revenue_cagr_3y"] = repaired
    return True


def repair_margin_history_metrics(
    *,
    features: Dict[str, Any],
    companyfacts: Dict[str, Any] | None,
    companyfacts_path: Path,
    computed_at: str,
    as_of_time: str,
) -> bool:
    trend_target = features.get("operating.ebitda_margin_trend_8q")
    vol_target = features.get("operating.margin_volatility_8q")
    if not trend_target or not vol_target:
        return False

    rows, concept_name = _build_ttm_margin_series(companyfacts, as_of_date=as_of_time[:10])
    rows = rows[-8:]
    if len(rows) < 3:
        return False
    margins = [float(row["margin"]) for row in rows if row.get("margin") is not None]
    if len(margins) < 3:
        return False
    trend_value = _linear_slope(margins)
    if trend_value is None:
        return False
    volatility_value = statistics.pstdev(margins)
    support_mode = "exact" if all(row.get("exact") for row in rows) else "proxy_missing_component"
    latest_period = rows[-1]["period_end"]
    should_refresh_trend = _should_refresh_existing_metric(
        trend_target,
        replacement_period=latest_period,
        period_key="window_end",
        fallback_prefix="sec_companyfacts_ttm_margin_history",
    )
    should_refresh_vol = _should_refresh_existing_metric(
        vol_target,
        replacement_period=latest_period,
        period_key="window_end",
        fallback_prefix="sec_companyfacts_ttm_margin_history",
    )
    if not should_refresh_trend and not should_refresh_vol:
        return False
    breakdown = {
        "formula": "slope(last_8_quarterly_ttm_margins)",
        "volatility_formula": "population_stddev(last_8_quarterly_ttm_margins)",
        "observation_count": len(rows),
        "window_start": f"{rows[0]['period_end'].isoformat()} 00:00:00+00:00",
        "window_end": f"{rows[-1]['period_end'].isoformat()} 00:00:00+00:00",
        "latest_margin": margins[-1],
        "oldest_margin": margins[0],
        "source_concept": concept_name,
        "period_basis": "quarterly_ttm_observations",
    }

    if should_refresh_trend:
        repaired_trend = _base_repaired_node(trend_target, computed_at=computed_at)
        repaired_trend["value"] = trend_value
        repaired_trend["fallback_used"] = "sec_companyfacts_ttm_margin_history"
        repaired_trend["support_mode"] = support_mode
        repaired_trend["provenance"] = _companyfacts_provenance(companyfacts_path, as_of_time=as_of_time, computed_at=computed_at)
        repaired_trend["component_breakdown"] = breakdown
        repaired_trend["quality_flags"] = None
        features["operating.ebitda_margin_trend_8q"] = repaired_trend

    if should_refresh_vol:
        repaired_vol = _base_repaired_node(vol_target, computed_at=computed_at)
        repaired_vol["value"] = volatility_value
        repaired_vol["fallback_used"] = "sec_companyfacts_ttm_margin_history"
        repaired_vol["support_mode"] = support_mode
        repaired_vol["provenance"] = _companyfacts_provenance(companyfacts_path, as_of_time=as_of_time, computed_at=computed_at)
        repaired_vol["component_breakdown"] = breakdown
        repaired_vol["quality_flags"] = None
        features["operating.margin_volatility_8q"] = repaired_vol
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
    summary: Dict[str, Dict[str, int]] = {}
    for metric, counter in counters.items():
        summary[metric] = {
            "exact": counter["exact"],
            "proxy_missing_component": counter["proxy_missing_component"],
            "unsupported": counter["unsupported"],
        }
    return summary


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    companyfacts_root = Path(args.companyfacts_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    computed_at = _now_iso()

    with out_path.open("w") as out_handle:
        for row in iter_rows(artifact_path):
            entity_id = str(row.get("company_id") or "")
            companyfacts_path = companyfacts_root / f"CIK{entity_id}.json"
            companyfacts = _load_companyfacts(companyfacts_path)
            features = row.get("features") or {}
            repair_revenue_yoy_last_q(
                features=features,
                companyfacts=companyfacts,
                companyfacts_path=companyfacts_path,
                computed_at=computed_at,
                as_of_time=str(row.get("as_of_time") or ""),
            )
            repair_revenue_cagr_3y(
                features=features,
                companyfacts=companyfacts,
                companyfacts_path=companyfacts_path,
                computed_at=computed_at,
                as_of_time=str(row.get("as_of_time") or ""),
            )
            repair_margin_history_metrics(
                features=features,
                companyfacts=companyfacts,
                companyfacts_path=companyfacts_path,
                computed_at=computed_at,
                as_of_time=str(row.get("as_of_time") or ""),
            )
            out_handle.write(json.dumps(row) + "\n")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.write_text(json.dumps(build_summary(out_path), indent=2))

    print(f"Repaired operating-history metrics -> {out_path}")


if __name__ == "__main__":
    main()
