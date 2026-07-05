#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _score_actions(candidate_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    action_scores: Dict[str, float] = {}
    for row in candidate_rows:
        action_id = str(row.get("action_id") or "").strip()
        if not action_id:
            continue
        confidence = row.get("precedent_confidence")
        if confidence is None:
            continue
        score = float(confidence)
        if action_id not in action_scores or score > action_scores[action_id]:
            action_scores[action_id] = score
    return action_scores


def _action_family(action_id: str) -> str:
    return action_id.split(".", 1)[0] if "." in action_id else action_id


def _analyze_case(
    *,
    case: Dict[str, Any],
    precedent_index: Dict[str, Any],
) -> Dict[str, Any]:
    action_scores = _score_actions(list(precedent_index.get("candidate_rows", []) or []))
    sorted_actions = sorted(action_scores.items(), key=lambda item: (-item[1], item[0]))
    action_rank_lookup = {action_id: index + 1 for index, (action_id, _) in enumerate(sorted_actions)}

    anchor_action_id = str(case["anchor_action_id"])
    anchor_family = str(case["anchor_action_family"])
    family_scores: Dict[str, float] = {}
    for action_id, score in sorted_actions:
        family = _action_family(action_id)
        if family not in family_scores:
            family_scores[family] = score

    anchor_score = action_scores.get(anchor_action_id)
    anchor_rank = action_rank_lookup.get(anchor_action_id)
    family_present = anchor_family in family_scores
    best_other_action_score = max(
        (score for action_id, score in sorted_actions if action_id != anchor_action_id),
        default=None,
    )
    anchor_margin = (
        None
        if anchor_score is None or best_other_action_score is None
        else float(anchor_score) - float(best_other_action_score)
    )

    return {
        "company_id": case["company_id"],
        "source_company_id": case.get("source_company_id", case["company_id"]),
        "ticker": case.get("ticker", ""),
        "mapping_method": case.get("mapping_method"),
        "anchor_action_id": anchor_action_id,
        "anchor_action_family": anchor_family,
        "anchor_action_date": case.get("anchor_action_date"),
        "as_of_time": case.get("as_of_time"),
        "distinct_action_count": len(sorted_actions),
        "distinct_action_ids": [action_id for action_id, _ in sorted_actions],
        "anchor_action_present": anchor_action_id in action_scores,
        "anchor_family_present": family_present,
        "anchor_action_rank": anchor_rank,
        "anchor_action_score": anchor_score,
        "anchor_action_margin": anchor_margin,
        "top_action_id": sorted_actions[0][0] if sorted_actions else None,
        "top_action_score": sorted_actions[0][1] if sorted_actions else None,
    }


def _select_competitive_cases(
    *,
    cases: List[Dict[str, Any]],
    search_root: Path,
    eval_id: str,
    eval_prefix: str,
    min_distinct_actions: int,
    require_anchor_family_present: bool,
    require_anchor_action_present: bool,
    limit: Optional[int],
) -> Dict[str, Any]:
    base = search_root / f"{eval_prefix}_eval_{eval_id}"
    case_map = {str(case["company_id"]): case for case in cases}
    analyses: List[Dict[str, Any]] = []

    for run_file in sorted((base / "runs").glob("run_id=*.json")):
        run = _load_json(run_file)
        company_id = str(run["company_id"])
        case = case_map.get(company_id)
        if case is None:
            continue
        precedent_index_path = base / "artifacts" / f"run_id={run['run_id']}" / "PrecedentIndex.json"
        precedent_index = _load_json(precedent_index_path)
        analysis = _analyze_case(case=case, precedent_index=precedent_index)
        analysis["precedent_index_path"] = str(precedent_index_path)
        analyses.append(analysis)

    selected: List[Dict[str, Any]] = []
    for analysis in analyses:
        if analysis["distinct_action_count"] < min_distinct_actions:
            continue
        if require_anchor_family_present and not analysis["anchor_family_present"]:
            continue
        if require_anchor_action_present and not analysis["anchor_action_present"]:
            continue
        selected.append(analysis)

    selected.sort(
        key=lambda item: (
            0 if item["anchor_action_present"] else 1,
            -int(item["anchor_family_present"]),
            -int(item["distinct_action_count"]),
            abs(item["anchor_action_margin"]) if item["anchor_action_margin"] is not None else 999.0,
            str(item["company_id"]),
        )
    )
    if limit is not None:
        selected = selected[:limit]

    selected_ids = {item["company_id"] for item in selected}
    selected_cases = [case_map[company_id] for company_id in [item["company_id"] for item in selected] if company_id in selected_ids]
    return {
        "selection_rankings": selected,
        "cases": selected_cases,
        "analysis_count": len(analyses),
        "selected_count": len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a competitive precedent probe manifest.")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--eval-id", default="001")
    parser.add_argument("--eval-prefix", default="capital_structure")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--selection-method", default="competitive_precedent_probe")
    parser.add_argument("--min-distinct-actions", type=int, default=2)
    parser.add_argument("--require-anchor-family-present", action="store_true")
    parser.add_argument("--require-anchor-action-present", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    source_manifest_path = Path(args.source_manifest)
    source_manifest = _load_json(source_manifest_path)
    selection = _select_competitive_cases(
        cases=list(source_manifest.get("cases", []) or []),
        search_root=Path(args.search_root),
        eval_id=args.eval_id,
        eval_prefix=str(args.eval_prefix or "capital_structure").strip(),
        min_distinct_actions=args.min_distinct_actions,
        require_anchor_family_present=args.require_anchor_family_present,
        require_anchor_action_present=args.require_anchor_action_present,
        limit=args.limit,
    )

    output = {
        "manifest_generated_at": source_manifest.get("manifest_generated_at"),
        "label": args.label,
        "selection_method": args.selection_method,
        "selection_source_manifest": str(source_manifest_path),
        "selection_source_search_root": args.search_root,
        "selection_eval_id": args.eval_id,
        "selection_eval_prefix": str(args.eval_prefix or "capital_structure").strip(),
        "selection_filters": {
            "min_distinct_actions": args.min_distinct_actions,
            "require_anchor_family_present": args.require_anchor_family_present,
            "require_anchor_action_present": args.require_anchor_action_present,
            "limit": args.limit,
        },
        "analysis_count": selection["analysis_count"],
        "case_count": selection["selected_count"],
        "selection_rankings": selection["selection_rankings"],
        "cases": selection["cases"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
