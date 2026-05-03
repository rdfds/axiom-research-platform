#!/usr/bin/env python3
"""Repair SEC-selected core metrics in an already-materialized artifact.

This pass refreshes the SEC-backed direct metrics whose selector logic can
change as we tighten concept ranking and freshness rules:

1. `operating.revenue_ttm_provider_direct`
2. `operating.ebitda_ltm_provider_direct`
3. `earnings.net_income_ttm_provider_direct`
4. `liquidity.cash_and_short_term_investments_provider_direct`
5. `capital_structure.total_debt_provider_direct`

It also recomputes the standardized metrics that depend on those values and,
when registry paths are provided, refreshes the smart-normalized metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_input_layer_v1_metrics as core  # noqa: E402
import repair_cash_debt_alignment_artifact as downstream  # noqa: E402
import backfill_smart_normalized_metrics_v1 as smart  # noqa: E402


CORE_METRICS_TO_REPAIR = (
    "operating.revenue_ttm_provider_direct",
    "operating.ebitda_ltm_provider_direct",
    "earnings.net_income_ttm_provider_direct",
    "liquidity.cash_and_short_term_investments_provider_direct",
    "capital_structure.total_debt_provider_direct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--companyfacts-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric-registry-path")
    parser.add_argument("--component-policy-path")
    parser.add_argument("--source-precedence-path")
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
    unit = core.DIRECT_METRIC_SPECS[metric_name]["unit"]
    features[metric_name] = core._build_metric_from_value(
        metric_name=metric_name,
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=str(companyfacts_path),
        unit=unit,
        value=value,
        support_mode=support_mode,
        missing_reason=missing_reason,
        component_breakdown=component_breakdown,
        quality_flags=quality_flags,
        primary_source_basis="sec_companyfacts",
        provenance_artifact_type="SecCompanyFacts",
        input_layer_bucket_reason="sec_companyfacts_asof",
    )


def _recompute_standardized_metrics(
    *,
    features: dict,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
) -> None:
    revenue = core._metric_value(features, "operating.revenue_ttm_provider_direct")
    ebitda = core._metric_value(features, "operating.ebitda_ltm_provider_direct")
    net_income = core._metric_value(features, "earnings.net_income_ttm_provider_direct")
    cash_sti = core._metric_value(features, "liquidity.cash_and_short_term_investments_provider_direct")
    total_debt = core._metric_value(features, "capital_structure.total_debt_provider_direct")

    revenue_support = core._metric_support(features, "operating.revenue_ttm_provider_direct")
    ebitda_support = core._metric_support(features, "operating.ebitda_ltm_provider_direct")
    net_income_support = core._metric_support(features, "earnings.net_income_ttm_provider_direct")
    cash_sti_support = core._metric_support(features, "liquidity.cash_and_short_term_investments_provider_direct")
    total_debt_support = core._metric_support(features, "capital_structure.total_debt_provider_direct")

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

    features["operating.ebitda_margin_standardized"] = core._build_combo_metric(
        metric_name="operating.ebitda_margin_standardized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="ratio",
        numerator=ebitda,
        denominator=revenue,
        extra_components={
            "ebitda_ltm_provider_direct": ebitda,
            "revenue_ttm_provider_direct": revenue,
        },
        component_supports={
            "ebitda_ltm_provider_direct": ebitda_support,
            "revenue_ttm_provider_direct": revenue_support,
        },
        formula="ebitda_ltm_provider_direct / revenue_ttm_provider_direct",
    )

    features["earnings.net_margin_standardized"] = core._build_combo_metric(
        metric_name="earnings.net_margin_standardized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="ratio",
        numerator=net_income,
        denominator=revenue,
        extra_components={
            "net_income_ttm_provider_direct": net_income,
            "revenue_ttm_provider_direct": revenue,
        },
        component_supports={
            "net_income_ttm_provider_direct": net_income_support,
            "revenue_ttm_provider_direct": revenue_support,
        },
        formula="net_income_ttm_provider_direct / revenue_ttm_provider_direct",
    )


