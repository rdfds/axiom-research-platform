from src.company_state_validation import (
    check_invariants,
    validate_peer_percentiles,
    validate_peer_zscores,
    validate_peer_bands,
)


def _snapshot(**feature_values):
    feats = {}
    for k, v in feature_values.items():
        feats[k] = {"value": v, "as_of_time": "2026-02-28T00:00:00Z"}
    return {
        "company_id": "001",
        "as_of_time": "2026-02-28T00:00:00Z",
        "features": feats,
    }


def _metric_feature(
    value,
    *,
    view_type: str,
    support_mode: str = "exact",
    applicability_status: str = "primary",
    canonical_owner_id: str = "fitch_ratings",
    canonical_classification: str = "canonical_external",
    market_layer_status: str = "keep",
    current_alignment_status: str = "partial_proxy",
    fallback_used=None,
    quality_flags=None,
    component_breakdown=None,
):
    return {
        "value": value,
        "as_of_time": "2026-02-28T00:00:00Z",
        "metric_policy_id": "market_metric_policy_v1",
        "market_owner": "credit_market",
        "primary_source_basis": "moodys_primary_v1",
        "methodology_registry_id": "consumer_industrials_metric_methodology_registry_v1",
        "methodology_metric_id": "capital_structure.total_debt",
        "canonical_owner_id": canonical_owner_id,
        "canonical_classification": canonical_classification,
        "market_layer_status": market_layer_status,
        "current_alignment_status": current_alignment_status,
        "archetype": "generic_corporate",
        "sector": "Industrials",
        "subsector": "Diversified Industrials",
        "override_level_applied": "sector",
        "support_mode": support_mode,
        "applicability_status": applicability_status,
        "component_breakdown": component_breakdown or {"reported_debt": 200.0},
        "quality_flags": quality_flags or [],
        "view_type": view_type,
        "fallback_used": fallback_used,
    }


