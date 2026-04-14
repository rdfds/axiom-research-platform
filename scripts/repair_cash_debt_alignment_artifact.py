#!/usr/bin/env python3
"""Repair exact cash/debt stack alignment in an already-materialized artifact.

This pass refreshes the two instant-stack provider metrics that can drift into
fake `exact` support when their underlying SEC components come from mismatched
statement dates:

1. `liquidity.cash_and_short_term_investments_provider_direct`
2. `capital_structure.total_debt_provider_direct`

It also recomputes the standardized debt / leverage metrics that directly
depend on those core values so downstream stages inherit a consistent base.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_input_layer_v1_metrics as core  # noqa: E402
import backfill_statement_direct_optional_metrics as statement  # noqa: E402
import backfill_smart_normalized_metrics_v1 as smart  # noqa: E402


CORE_METRICS_TO_REPAIR = (
    "liquidity.cash_and_short_term_investments_provider_direct",
    "capital_structure.total_debt_provider_direct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--companyfacts-root", required=True)
    parser.add_argument("--facts-path")
    parser.add_argument("--metric-registry-path")
    parser.add_argument("--component-policy-path")
    parser.add_argument("--source-precedence-path")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _rebuild_core_metric(
    *,
    features: dict,
    metric_name: str,
    companyfacts: dict,
    companyfacts_path: Path,
    as_of_time: str,
    computed_at: str,
) -> None:
    value, support_mode, missing_reason, component_breakdown, quality_flags = core._build_sec_core_metric(
        metric_name,
        companyfacts,
        as_of_time[:10],
    )
    features[metric_name] = core._build_metric_from_value(
        metric_name=metric_name,
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=str(companyfacts_path),
        unit="usd",
        value=value,
        support_mode=support_mode,
        missing_reason=missing_reason,
        component_breakdown=component_breakdown,
        quality_flags=quality_flags,
        primary_source_basis="sec_companyfacts",
        provenance_artifact_type="SecCompanyFacts",
        input_layer_bucket_reason="sec_companyfacts_asof",
    )


def _recompute_standardized_debt_metrics(
    *,
    features: dict,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
) -> None:
    total_debt = core._metric_value(features, "capital_structure.total_debt_provider_direct")
    total_debt_support = core._metric_support(features, "capital_structure.total_debt_provider_direct")
    cash_sti = core._metric_value(features, "liquidity.cash_and_short_term_investments_provider_direct")
    cash_sti_support = core._metric_support(features, "liquidity.cash_and_short_term_investments_provider_direct")
    ebitda = core._metric_value(features, "operating.ebitda_ltm_provider_direct")
    ebitda_support = core._metric_support(features, "operating.ebitda_ltm_provider_direct")

    net_debt = None if total_debt is None or cash_sti is None else total_debt - cash_sti

    features["capital_structure.net_debt_standardized"] = core._build_combo_metric(
        metric_name="capital_structure.net_debt_standardized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="usd",
        numerator=net_debt,
        denominator=None,
        extra_components={
            "total_debt_provider_direct": total_debt,
            "cash_and_short_term_investments_provider_direct": cash_sti,
        },
        component_supports={
            "total_debt_provider_direct": total_debt_support,
            "cash_and_short_term_investments_provider_direct": cash_sti_support,
        },
        formula="total_debt_provider_direct - cash_and_short_term_investments_provider_direct",
        allow_numerator_only=True,
    )
    features["capital_structure.gross_leverage_standardized"] = core._build_combo_metric(
        metric_name="capital_structure.gross_leverage_standardized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="x",
        numerator=total_debt,
        denominator=ebitda,
        extra_components={
            "total_debt_provider_direct": total_debt,
            "ebitda_ltm_provider_direct": ebitda,
        },
        component_supports={
            "total_debt_provider_direct": total_debt_support,
            "ebitda_ltm_provider_direct": ebitda_support,
        },
        formula="total_debt_provider_direct / ebitda_ltm_provider_direct",
    )
    features["capital_structure.net_leverage_standardized"] = core._build_combo_metric(
        metric_name="capital_structure.net_leverage_standardized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="x",
        numerator=net_debt,
        denominator=ebitda,
        extra_components={
            "net_debt_standardized": net_debt,
            "ebitda_ltm_provider_direct": ebitda,
        },
        component_supports={
            "net_debt_standardized": features["capital_structure.net_debt_standardized"]["support_mode"],
            "ebitda_ltm_provider_direct": ebitda_support,
        },
        formula="net_debt_standardized / ebitda_ltm_provider_direct",
    )


def _recompute_smart_metrics(
    *,
    features: dict,
    as_of_time: str,
    computed_at: str,
    registry: dict[str, object],
    provenance_sources: list[str],
) -> None:
    total_debt = smart._node(features, "capital_structure.total_debt_provider_direct")
    cash_grouped = smart._node(features, "liquidity.cash_and_short_term_investments_provider_direct")
    cash_exact = smart._node(features, "liquidity.cash_and_equivalents_statement_direct")
    restricted_cash_sec = smart._node(features, "liquidity.restricted_cash_sec_exact")
    marketable_sec = smart._node(features, "liquidity.marketable_securities_sec_exact")
    revolver_sec = smart._node(features, "liquidity.revolver_undrawn_sec_exact")
    lease_sec = smart._node(features, "capital_structure.lease_liabilities_sec_exact")
    restricted_cash = smart._node(features, "liquidity.restricted_cash")
    marketable = smart._node(features, "liquidity.marketable_securities")
    revolver = smart._node(features, "liquidity.revolver_undrawn")
    ebitda = smart._node(features, "operating.ebitda_ltm_provider_direct")
    current_debt = smart._node(features, "capital_structure.current_debt_statement_direct")
    long_term_debt = smart._node(features, "capital_structure.long_term_debt_statement_direct")

    total_debt_value = smart._value(total_debt)
    total_debt_exact = smart._exact(total_debt) and total_debt_value is not None
    current_debt_value = smart._value(current_debt)
    long_term_debt_value = smart._value(long_term_debt)
    current_debt_exact = smart._exact(current_debt) and current_debt_value is not None
    long_term_debt_exact = smart._exact(long_term_debt) and long_term_debt_value is not None
    lease_sec_value = smart._value(lease_sec) if smart._exact(lease_sec) else None
    lease_exact = smart._exact(lease_sec)
    debt_value = None
    debt_base_formula = "unavailable"
    debt_base_source_metric = None
    if total_debt_exact:
        debt_value = total_debt_value
        debt_base_formula = "baseline_total_debt_provider_direct"
        debt_base_source_metric = "capital_structure.total_debt_provider_direct"
    elif current_debt_exact and long_term_debt_exact:
        debt_value = current_debt_value + long_term_debt_value
        debt_base_formula = "current_debt_statement_direct + long_term_debt_statement_direct"
        debt_base_source_metric = (
            "capital_structure.current_debt_statement_direct + "
            "capital_structure.long_term_debt_statement_direct"
        )
    elif total_debt_value is not None:
        debt_value = total_debt_value
        debt_base_formula = "baseline_total_debt_provider_direct"
        debt_base_source_metric = "capital_structure.total_debt_provider_direct"
    elif current_debt_value is not None or long_term_debt_value is not None:
        debt_value = float((current_debt_value or 0.0) + (long_term_debt_value or 0.0))
        debt_base_formula = "sum_available_statement_debt_components"
        debt_base_source_metric = (
            "capital_structure.current_debt_statement_direct + "
            "capital_structure.long_term_debt_statement_direct"
        )
    debt_exact_ready = lease_exact and (total_debt_exact or (current_debt_exact and long_term_debt_exact))
    debt_components = {
        "baseline_source_metric": debt_base_source_metric,
        "baseline_value": debt_value,
        "total_debt_provider_direct": total_debt_value,
        "current_debt_statement_direct": current_debt_value,
        "long_term_debt_statement_direct": long_term_debt_value,
        "lease_liabilities_sec_exact": lease_sec_value,
        "formula": debt_base_formula + (" + lease_liabilities_sec_exact" if lease_sec_value is not None else ""),
    }
    if debt_value is not None and lease_sec_value is not None:
        debt_value = debt_value + lease_sec_value
    features["capital_structure.debt_like_obligations_normalized"] = smart._smart_value_node(
        metric_name="capital_structure.debt_like_obligations_normalized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=registry["metrics"]["debt_like_obligations_normalized"]["status"],
        promotion_rule=registry["metrics"]["debt_like_obligations_normalized"]["promotion_rule"],
        value=debt_value,
        unit="usd",
        component_breakdown=debt_components,
        provenance_sources=provenance_sources,
        exact_ready=debt_exact_ready,
        missing_reason="component_unavailable" if debt_value is None else None,
    )

    grouped_cash_value = smart._value(cash_grouped)
    cash_exact_value = smart._value(cash_exact)
    restricted_cash_inferred_zero = (
        restricted_cash_sec.get("support_mode") == "unsupported"
        and restricted_cash_sec.get("missing_reason") == "sec_concept_absent"
    )
    marketable_inferred_zero = (
        marketable_sec.get("support_mode") == "unsupported"
        and marketable_sec.get("missing_reason") == "sec_concept_absent"
    )
    restricted_cash_value = (
        smart._value(restricted_cash_sec)
        if smart._exact(restricted_cash_sec)
        else (0.0 if restricted_cash_inferred_zero else (smart._value(restricted_cash) if smart._exact(restricted_cash) else None))
    )
    marketable_value = (
        smart._value(marketable_sec)
        if smart._exact(marketable_sec)
        else (0.0 if marketable_inferred_zero else (smart._value(marketable) if smart._exact(marketable) else None))
    )
    revolver_value = smart._value(revolver_sec) if smart._exact(revolver_sec) else (smart._value(revolver) if smart._exact(revolver) else None)

    grouped_cash_exact = smart._exact(cash_grouped) and grouped_cash_value is not None
    if grouped_cash_exact:
        liquidity_base = grouped_cash_value
        liquidity_formula = "cash_and_short_term_investments_provider_direct"
    elif cash_exact_value is not None and marketable_value is not None:
        liquidity_base = cash_exact_value + marketable_value
        liquidity_formula = (
            "cash_and_equivalents_statement_direct + marketable_securities_sec_exact"
            if not marketable_inferred_zero
            else "cash_and_equivalents_statement_direct + 0_inferred_short_term_investments"
        )
    elif cash_exact_value is not None and grouped_cash_value is not None and abs(grouped_cash_value - cash_exact_value) <= 1.0:
        liquidity_base = cash_exact_value
        liquidity_formula = "cash_and_equivalents_statement_direct"
    elif grouped_cash_value is not None:
        liquidity_base = grouped_cash_value
        liquidity_formula = "cash_and_short_term_investments_provider_direct"
    elif cash_exact_value is not None:
        liquidity_base = cash_exact_value
        liquidity_formula = "cash_and_equivalents_statement_direct"
    else:
        liquidity_base = None
        liquidity_formula = "unavailable"

    available_liquidity_raw = None
    if liquidity_base is not None:
        available_liquidity_raw = liquidity_base
        if restricted_cash_value is not None:
            available_liquidity_raw -= restricted_cash_value
        if revolver_value is not None:
            available_liquidity_raw += revolver_value

    available_liquidity = available_liquidity_raw
    negative_floor_applied = available_liquidity_raw is not None and available_liquidity_raw < 0
    if negative_floor_applied:
        available_liquidity = 0.0

    liquidity_exact_ready = (
        restricted_cash_value is not None
        and (
            grouped_cash_exact
            or (
                cash_exact_value is not None
                and (
                    marketable_value is not None
                    or (grouped_cash_value is not None and abs(grouped_cash_value - cash_exact_value) <= 1.0)
                )
            )
        )
    )
    if negative_floor_applied:
        liquidity_exact_ready = False
    formula_text = (
        liquidity_formula
        + (
            " - restricted_cash_sec_exact"
            if restricted_cash_value is not None and not restricted_cash_inferred_zero
            else (" - 0_inferred_restricted_cash" if restricted_cash_inferred_zero else "")
        )
        + (" + revolver_undrawn_exact" if revolver_value is not None else "")
    )
    if negative_floor_applied:
        formula_text = f"max(0, {formula_text})"
    liquidity_components = {
        "grouped_cash_provider_direct": grouped_cash_value,
        "cash_and_equivalents_statement_direct": cash_exact_value,
        "restricted_cash_sec_exact": restricted_cash_value,
        "marketable_securities_sec_exact": marketable_value,
        "restricted_cash_inferred_zero": restricted_cash_inferred_zero,
        "marketable_securities_inferred_zero": marketable_inferred_zero,
        "revolver_undrawn_sec_exact": smart._value(revolver_sec) if smart._exact(revolver_sec) else None,
        "revolver_undrawn_exact": revolver_value,
        "raw_value_before_floor": available_liquidity_raw,
        "formula": formula_text,
    }
    if negative_floor_applied:
        liquidity_components["exact_guard_reason"] = "negative_available_liquidity"
    features["liquidity.available_liquidity_normalized"] = smart._smart_value_node(
        metric_name="liquidity.available_liquidity_normalized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=registry["metrics"]["available_liquidity_normalized"]["status"],
        promotion_rule=registry["metrics"]["available_liquidity_normalized"]["promotion_rule"],
        value=available_liquidity,
        unit="usd",
        component_breakdown=liquidity_components,
        provenance_sources=provenance_sources,
        exact_ready=liquidity_exact_ready,
        missing_reason="component_unavailable" if available_liquidity is None else None,
    )

    earnings_value = smart._value(ebitda)
    earnings_exact_ready = smart._exact(ebitda)
    features["operating.operating_earnings_normalized"] = smart._smart_value_node(
        metric_name="operating.operating_earnings_normalized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=registry["metrics"]["operating_earnings_normalized"]["status"],
        promotion_rule=registry["metrics"]["operating_earnings_normalized"]["promotion_rule"],
        value=earnings_value,
        unit="usd",
        component_breakdown={
            "baseline_source_metric": "operating.ebitda_ltm_provider_direct",
            "baseline_value": earnings_value,
            "formula": "provider_direct_ebitda_baseline",
        },
        provenance_sources=provenance_sources,
        exact_ready=earnings_exact_ready,
        missing_reason="component_unavailable" if earnings_value is None else None,
    )

    net_debt_value = None if debt_value is None or available_liquidity is None else debt_value - available_liquidity
    net_debt_exact_ready = debt_exact_ready and liquidity_exact_ready
    features["capital_structure.net_debt_normalized"] = smart._smart_value_node(
        metric_name="capital_structure.net_debt_normalized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=registry["metrics"]["net_debt_normalized"]["status"],
        promotion_rule=registry["metrics"]["net_debt_normalized"]["promotion_rule"],
        value=net_debt_value,
        unit="usd",
        component_breakdown={
            "debt_like_obligations_normalized": debt_value,
            "available_liquidity_normalized": available_liquidity,
            "formula": "debt_like_obligations_normalized - available_liquidity_normalized",
        },
        provenance_sources=provenance_sources,
        exact_ready=net_debt_exact_ready,
        missing_reason="component_unavailable" if net_debt_value is None else None,
    )

    if debt_value is None or earnings_value is None:
        gross_lev_value = None
        gross_missing = "component_unavailable"
    elif earnings_value <= 0:
        gross_lev_value = None
        gross_missing = "non_positive_denominator"
    else:
        gross_lev_value = debt_value / earnings_value
        gross_missing = None
    gross_lev_exact_ready = debt_exact_ready and earnings_exact_ready and gross_lev_value is not None
    features["capital_structure.gross_leverage_normalized"] = smart._smart_value_node(
        metric_name="capital_structure.gross_leverage_normalized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=registry["metrics"]["gross_leverage_normalized"]["status"],
        promotion_rule=registry["metrics"]["gross_leverage_normalized"]["promotion_rule"],
        value=gross_lev_value,
        unit="x",
        component_breakdown={
            "debt_like_obligations_normalized": debt_value,
            "operating_earnings_normalized": earnings_value,
            "formula": "debt_like_obligations_normalized / operating_earnings_normalized",
        },
        provenance_sources=provenance_sources,
        exact_ready=gross_lev_exact_ready,
        missing_reason=gross_missing,
    )

    if net_debt_value is None or earnings_value is None:
        net_lev_value = None
        net_missing = "component_unavailable"
    elif earnings_value <= 0:
        net_lev_value = None
        net_missing = "non_positive_denominator"
    else:
        net_lev_value = net_debt_value / earnings_value
        net_missing = None
    net_lev_exact_ready = net_debt_exact_ready and earnings_exact_ready and net_lev_value is not None
    features["capital_structure.net_leverage_normalized"] = smart._smart_value_node(
        metric_name="capital_structure.net_leverage_normalized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=registry["metrics"]["net_leverage_normalized"]["status"],
        promotion_rule=registry["metrics"]["net_leverage_normalized"]["promotion_rule"],
        value=net_lev_value,
        unit="x",
        component_breakdown={
            "net_debt_normalized": net_debt_value,
            "operating_earnings_normalized": earnings_value,
            "formula": "net_debt_normalized / operating_earnings_normalized",
        },
        provenance_sources=provenance_sources,
        exact_ready=net_lev_exact_ready,
        missing_reason=net_missing,
    )


def _repair_row(
    row: dict,
    companyfacts_root: Path,
    companyfacts_cache: dict[str, dict | None],
    statement_current_debt_candidates: dict[str, list[dict[str, object]]] | None,
    statement_long_term_debt_candidates: dict[str, list[dict[str, object]]] | None,
    computed_at: str,
    provenance_source: str,
    registry: dict[str, object] | None,
    provenance_sources: list[str] | None,
) -> dict:
    features = row.setdefault("features", {})
    entity_id = str(row["company_id"])
    as_of_time = row["as_of_time"]
    companyfacts_path = companyfacts_root / f"CIK{entity_id}.json"
    if entity_id not in companyfacts_cache:
        companyfacts_cache[entity_id] = core._load_companyfacts(companyfacts_path)
    companyfacts = companyfacts_cache[entity_id]
    if companyfacts is None:
        return row

    for metric_name in CORE_METRICS_TO_REPAIR:
        _rebuild_core_metric(
            features=features,
            metric_name=metric_name,
            companyfacts=companyfacts,
            companyfacts_path=companyfacts_path,
            as_of_time=as_of_time,
            computed_at=computed_at,
        )

    repaired_total_debt = statement._repair_total_debt_from_statement_split(
        current_node=features.get("capital_structure.total_debt_provider_direct"),
        current_debt_statement_node=features.get("capital_structure.current_debt_statement_direct"),
        long_term_debt_statement_node=features.get("capital_structure.long_term_debt_statement_direct"),
        current_debt_statement_candidates=(
            None if statement_current_debt_candidates is None else statement_current_debt_candidates.get(entity_id)
        ),
        long_term_debt_statement_candidates=(
            None if statement_long_term_debt_candidates is None else statement_long_term_debt_candidates.get(entity_id)
        ),
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=str(companyfacts_path),
    )
    if repaired_total_debt is not None:
        features["capital_structure.total_debt_provider_direct"] = repaired_total_debt

    statement._recompute_standardized_debt_metrics(
        features=features,
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
    )
    if registry is not None and provenance_sources is not None:
        _recompute_smart_metrics(
            features=features,
            as_of_time=as_of_time,
            computed_at=computed_at,
            registry=registry,
            provenance_sources=provenance_sources,
        )
    return row


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    out_path = Path(args.out)
    companyfacts_root = Path(args.companyfacts_root)
    computed_at = core._now_iso()
    provenance_source = f"{artifact_path}:cash_debt_alignment_repair"
    companyfacts_cache: dict[str, dict | None] = {}
    statement_current_debt_candidates: dict[str, list[dict[str, object]]] | None = None
    statement_long_term_debt_candidates: dict[str, list[dict[str, object]]] | None = None
    registry = None
    provenance_sources = None
    if args.metric_registry_path and args.component_policy_path and args.source_precedence_path:
        registry = json.loads(Path(args.metric_registry_path).read_text())
        json.loads(Path(args.component_policy_path).read_text())
        json.loads(Path(args.source_precedence_path).read_text())
        provenance_sources = [
            str(Path(args.metric_registry_path)),
            str(Path(args.component_policy_path)),
            str(Path(args.source_precedence_path)),
        ]

    if args.facts_path:
        entity_ids: list[str] = []
        as_of_time: str | None = None
        with artifact_path.open() as src:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                entity_ids.append(str(row["company_id"]))
                if as_of_time is None:
                    as_of_time = row["as_of_time"]
        if entity_ids and as_of_time:
            facts_path = Path(args.facts_path)
            statement_current_debt_candidates = statement._load_statement_fact_candidates(
                facts_path,
                entity_ids,
                [statement.STATEMENT_FACT_SPECS["current_debt"]["fact_type"]],
                as_of_time,
            )
            statement_long_term_debt_candidates = statement._load_statement_fact_candidates(
                facts_path,
                entity_ids,
                [statement.STATEMENT_FACT_SPECS["long_term_debt"]["fact_type"]],
                as_of_time,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open() as src, out_path.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            row = _repair_row(
                row,
                companyfacts_root,
                companyfacts_cache,
                statement_current_debt_candidates,
                statement_long_term_debt_candidates,
                computed_at,
                provenance_source,
                registry,
                provenance_sources,
            )
            dst.write(json.dumps(row) + "\n")
    print(f"Repaired cash/debt alignment -> {out_path}")


if __name__ == "__main__":
    main()
