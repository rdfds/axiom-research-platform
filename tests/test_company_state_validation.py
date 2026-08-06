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


def _lineage_for_metric(feat: dict):
    return {
        "metric_context": {
            "metric_policy_id": feat["metric_policy_id"],
            "market_owner": feat["market_owner"],
            "primary_source_basis": feat["primary_source_basis"],
            "methodology_registry_id": feat["methodology_registry_id"],
            "methodology_metric_id": feat["methodology_metric_id"],
            "canonical_owner_id": feat["canonical_owner_id"],
            "canonical_classification": feat["canonical_classification"],
            "market_layer_status": feat["market_layer_status"],
            "current_alignment_status": feat["current_alignment_status"],
            "archetype": feat["archetype"],
            "sector": feat["sector"],
            "subsector": feat["subsector"],
            "override_level_applied": feat["override_level_applied"],
            "support_mode": feat["support_mode"],
            "applicability_status": feat["applicability_status"],
            "view_type": feat["view_type"],
            "component_breakdown": feat["component_breakdown"],
            "quality_flags": feat["quality_flags"],
        }
    }


def test_invariant_liquidity_total_ge_cash():
    snap = _snapshot(**{
        "liquidity.cash": 100.0,
        "liquidity.liquidity_total": 100.0,
    })
    assert "liquidity_total_lt_cash" not in check_invariants(snap)

    snap_bad = _snapshot(**{
        "liquidity.cash": 100.0,
        "liquidity.liquidity_total": 50.0,
    })
    assert "liquidity_total_lt_cash" in check_invariants(snap_bad)


def test_metric_invariants_accept_market_view_triplets():
    total_debt_reported = _metric_feature(200.0, view_type="reported", component_breakdown={"reported_debt": 200.0})
    total_debt_market = _metric_feature(
        240.0,
        view_type="market",
        component_breakdown={
            "reported_debt": 200.0,
            "lease_liabilities": 40.0,
            "lease_weight": 1.0,
            "supplier_finance": 0.0,
            "supplier_finance_weight": 1.0,
            "preferred_equity": 0.0,
            "preferred_weight": 0.5,
            "convertibles": 0.0,
            "convertible_weight": 0.5,
            "unfunded_pension": 0.0,
            "pension_weight": 0.25,
        },
    )
    total_debt_decision = _metric_feature(
        240.0,
        view_type="decision",
        component_breakdown=total_debt_market["component_breakdown"],
    )
    net_debt_market = _metric_feature(
        180.0,
        view_type="market",
        component_breakdown={"economic_debt": 240.0, "usable_cash_market": 60.0},
    )
    snap = {
        "company_id": "001",
        "as_of_time": "2026-02-28T00:00:00Z",
        "features": {
            "liquidity.cash": {"value": 50.0, "as_of_time": "2026-02-28T00:00:00Z"},
            "liquidity.liquidity_total": {"value": 50.0, "as_of_time": "2026-02-28T00:00:00Z"},
            "liquidity.usable_cash_market": _metric_feature(60.0, view_type="market", component_breakdown={"cash": 75.0, "restricted_cash": 15.0}),
            "capital_structure.total_debt_reported": total_debt_reported,
            "capital_structure.total_debt_market": total_debt_market,
            "capital_structure.total_debt": total_debt_decision,
            "capital_structure.net_debt_market": net_debt_market,
            "capital_structure.net_debt_reported": _metric_feature(150.0, view_type="reported", component_breakdown={"economic_debt": 200.0}),
            "capital_structure.net_debt": _metric_feature(180.0, view_type="decision", component_breakdown={"economic_debt": 240.0, "usable_cash_market": 60.0}),
            "market.market_cap": {"value": 500.0, "as_of_time": "2026-02-28T00:00:00Z"},
            "market.enterprise_value": {"value": 650.0, "as_of_time": "2026-02-28T00:00:00Z"},
        },
        "provenance": {
            "feature_lineage": {
                "capital_structure.total_debt_reported": _lineage_for_metric(total_debt_reported),
                "capital_structure.total_debt_market": _lineage_for_metric(total_debt_market),
                "capital_structure.total_debt": _lineage_for_metric(total_debt_decision),
                "capital_structure.net_debt_market": _lineage_for_metric(net_debt_market),
                "capital_structure.net_debt_reported": _lineage_for_metric(_metric_feature(150.0, view_type="reported", component_breakdown={"economic_debt": 200.0})),
                "capital_structure.net_debt": _lineage_for_metric(_metric_feature(180.0, view_type="decision", component_breakdown={"economic_debt": 240.0, "usable_cash_market": 60.0})),
                "liquidity.usable_cash_market": _lineage_for_metric(_metric_feature(60.0, view_type="market", component_breakdown={"cash": 75.0, "restricted_cash": 15.0})),
            }
        },
    }
    assert check_invariants(snap) == []


def test_metric_invariants_flag_unsupported_decision_values():
    unsupported = _metric_feature(
        3.5,
        view_type="decision",
        support_mode="unsupported",
        applicability_status="unsupported",
        quality_flags=["unsupported_metric"],
    )
    snap = {
        "company_id": "001",
        "as_of_time": "2026-02-28T00:00:00Z",
        "features": {
            "capital_structure.net_leverage_reported": _metric_feature(3.5, view_type="reported"),
            "capital_structure.net_leverage_market": _metric_feature(None, view_type="market", support_mode="unsupported", applicability_status="unsupported"),
            "capital_structure.net_leverage": unsupported,
        },
        "provenance": {
            "feature_lineage": {
                "capital_structure.net_leverage_reported": _lineage_for_metric(_metric_feature(3.5, view_type="reported")),
                "capital_structure.net_leverage_market": _lineage_for_metric(_metric_feature(None, view_type="market", support_mode="unsupported", applicability_status="unsupported")),
                "capital_structure.net_leverage": _lineage_for_metric(unsupported),
            }
        },
    }
    errs = check_invariants(snap)
    assert "unsupported_decision_metric_has_value:capital_structure.net_leverage" in errs


def test_peer_percentiles_range():
    snap = _snapshot(**{
        "peer_context.valuation_percentile": 50.0,
        "peer_context.leverage_percentile": 0.0,
        "peer_context.margin_percentile": 100.0,
        "peer_context.action_rate_percentile": 75.0,
    })
    assert validate_peer_percentiles(snap) == []

    snap_bad = _snapshot(**{
        "peer_context.valuation_percentile": 120.0,
    })
    errs = validate_peer_percentiles(snap_bad)
    assert "peer_percentile_out_of_range:peer_context.valuation_percentile" in errs


def test_peer_zscore_numeric():
    snap = _snapshot(**{
        "peer_context.valuation_z": 0.5,
        "peer_context.leverage_z": -1.2,
        "peer_context.margin_z": 2.1,
        "peer_context.action_rate_z": 0.3,
    })
    assert validate_peer_zscores(snap) == []

    snap_bad = _snapshot(**{
        "peer_context.margin_z": "not_a_number",
    })
    errs = validate_peer_zscores(snap_bad)
    assert "peer_zscore_not_numeric:peer_context.margin_z" in errs


def test_peer_bands():
    snap = _snapshot(**{
        "peer_context.valuation_band": "q1",
        "peer_context.leverage_band": "q2",
        "peer_context.margin_band": "q4",
        "peer_context.action_rate_band": "q3",
    })
    assert validate_peer_bands(snap) == []

    snap_bad = _snapshot(**{
        "peer_context.valuation_band": "top",
    })
    errs = validate_peer_bands(snap_bad)
    assert "peer_band_invalid:peer_context.valuation_band" in errs
