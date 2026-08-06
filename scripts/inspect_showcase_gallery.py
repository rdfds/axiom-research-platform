#!/usr/bin/env python3
"""Print a compact summary of the committed public showcase samples."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def main() -> int:
    valuation = load("examples/hd_market_expectations/valuation_driver_data.sample.json")
    company_state = load("examples/company_state_snapshot/company_state_hd.sample.json")
    precedents = load("examples/precedent_retrieval/precedent_retrieval.sample.json")
    cfo = load("examples/cfo_decision_surface/cfo_decision_surface_hd.sample.json")

    company = valuation["companies"][0]
    forward = company["forward_expectations"]
    print("Axiom public showcase")
    print("====================")
    print(f"Valuation demo: {company['ticker']} / {company['name']} / forward grade {forward['forward_grade']}")
    print(f"Company state: {company_state['ticker']} / {len(company_state['features'])} provenance-rich features")
    print(
        "Precedents: "
        f"{precedents['retrieved_candidate_rows_in_sample']} candidate rows / "
        f"mean confidence {precedents['evaluation_summary']['mean_precedent_confidence']:.3f}"
    )

    surface = cfo["home_depot_decision_surface"]
    mna = surface["mna_decision_summary"]["authoritative_recommendation"]
    frontier = surface["capital_allocation_frontier"]["points"]
    print(
        "CFO surface: "
        f"{mna['label']} at {mna['deal_size_pct_market_cap_display']} / "
        f"{len(frontier)} capital-allocation frontier points"
    )
    dossier = cfo["board_ready_dossier_excerpt"]
    print(
        "Dossier: "
        f"{dossier['decision_confidence']} confidence / "
        f"{len(dossier['supporting_evidence'])} supporting evidence rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
