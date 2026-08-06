#!/usr/bin/env python3
"""Repair debt-stack metrics in an already-materialized input-layer artifact.

This is intentionally narrow and fast:

1. Recompute `capital_structure.total_debt_provider_direct` from SEC companyfacts
   using the latest debt-stack logic.
2. Recompute the standardized debt/leverage combos that directly depend on it.

We use this repair pass when the underlying debt builder has improved and we
want to refresh existing artifacts without re-running the whole pipeline.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--companyfacts-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _repair_row(row: dict, companyfacts_root: Path, computed_at: str, provenance_source: str) -> dict:
    features = row.setdefault("features", {})
    entity_id = str(row["company_id"])
    as_of_time = row["as_of_time"]
    as_of_date = as_of_time[:10]
    companyfacts_path = companyfacts_root / f"CIK{entity_id}.json"
    companyfacts = core._load_companyfacts(companyfacts_path)
    if companyfacts is None:
        return row

    value, support_mode, missing_reason, component_breakdown, quality_flags = core._build_sec_core_metric(
        "capital_structure.total_debt_provider_direct",
        companyfacts,
        as_of_date,
    )
    features["capital_structure.total_debt_provider_direct"] = core._build_metric_from_value(
        metric_name="capital_structure.total_debt_provider_direct",
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
    return row


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    out_path = Path(args.out)
    companyfacts_root = Path(args.companyfacts_root)
    computed_at = core._now_iso()
    provenance_source = f"{artifact_path}:debt_stack_repair"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open() as src, out_path.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            row = _repair_row(row, companyfacts_root, computed_at, provenance_source)
            dst.write(json.dumps(row) + "\n")
    print(f"Repaired debt stack -> {out_path}")


if __name__ == "__main__":
    main()
