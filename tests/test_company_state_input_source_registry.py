import ast
import json
from pathlib import Path

from src.company_state_input_source_registry import CompanyStateInputSourceRegistry


REPO_ROOT = Path(".")
COMPANY_STATE_BUILDER = REPO_ROOT / "src" / "company_state_builder.py"
REGISTRY_PATH = REPO_ROOT / "configs" / "metric_methodologies" / "company_state_input_source_registry_v1.json"


def _static_feature_names() -> set[str]:
    tree = ast.parse(COMPANY_STATE_BUILDER.read_text())
    names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "features":
                    if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                        names.add(target.slice.value)
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "_emit_metric_views":
                for kw in node.keywords:
                    if kw.arg == "base_name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        names.add(kw.value.value)
            self.generic_visit(node)

    Visitor().visit(tree)
    return names


def _dynamic_feature_names() -> set[str]:
    return {
        "macro.sp500_pe_ttm",
        "macro.sp500_pe_ttm_percentile_history",
        "macro.us10y_treasury_yield",
        "macro.us10y_treasury_yield_percentile_history",
        "macro.us_ig_oas",
        "macro.us_ig_oas_percentile_history",
        "macro.us_hy_all_in_yield",
        "macro.us_hy_all_in_yield_percentile_history",
        "macro.real_gdp_growth_yoy",
        "macro.real_gdp_growth_yoy_percentile_history",
        "strategic.intent.return_capital_priority",
        "strategic.intent.deleveraging_priority",
        "strategic.intent.pursue_mna_priority",
        "strategic.intent.focus_on_core",
        "strategic.intent.restructure",
    }


def test_company_state_input_source_registry_covers_every_metric():
    payload = json.loads(REGISTRY_PATH.read_text())
    registry_metrics = set(payload["metrics"])
    expected_metrics = _static_feature_names() | _dynamic_feature_names()

    missing = sorted(expected_metrics - registry_metrics)
    extra = sorted(registry_metrics - expected_metrics)

    assert missing == []
    assert extra == []


def test_company_state_input_source_registry_has_trusted_or_explicit_internal_classification():
    payload = json.loads(REGISTRY_PATH.read_text())
    valid_classifications = {
        "canonical_external",
        "filing_native_external",
        "external_filing_normalized",
        "external_provider_standardized",
        "external_raw_plus_deterministic_formula",
        "external_methodology",
        "external_market_benchmark",
        "internal_derived",
        "internal_heuristic",
        "unsupported_placeholder",
    }
    for metric_id, rec in payload["metrics"].items():
        assert rec["classification"] in valid_classifications, metric_id
        assert rec["canonical_owner_id"] in payload["owners"], metric_id
        assert isinstance(rec["formula_basis"], str) and rec["formula_basis"], metric_id
        assert isinstance(rec["company_tailoring"], str) and rec["company_tailoring"], metric_id


def test_company_state_input_source_registry_has_definition_requirement_split():
    payload = json.loads(REGISTRY_PATH.read_text())
    valid_requirements = {
        "must_have_external_definition",
        "can_be_externally_anchored",
        "must_remain_internal_inference",
    }
    for metric_id, rec in payload["metrics"].items():
        assert rec["definition_requirement"] in valid_requirements, metric_id
        assert isinstance(rec["definition_requirement_reason"], str) and rec["definition_requirement_reason"], metric_id

    metrics = payload["metrics"]
    assert metrics["capital_structure.net_debt"]["definition_requirement"] == "must_have_external_definition"
    assert metrics["operating.fcf_conversion"]["classification"] == "external_raw_plus_deterministic_formula"
    assert metrics["operating.fcf_conversion"]["canonical_owner_id"] == "issuer_filing_lseg_fundamentals"
    assert metrics["peer_context.leverage_percentile"]["definition_requirement"] == "can_be_externally_anchored"
    assert metrics["market.credit_window_proxy"]["definition_requirement"] == "must_remain_internal_inference"
    assert metrics["liquidity.available_for_actions"]["definition_requirement"] == "must_remain_internal_inference"


def test_company_state_input_source_registry_has_methodology_execution_decision():
    payload = json.loads(REGISTRY_PATH.read_text())
    valid_decisions = {
        "adopt_exact_external_methodology",
        "keep_externally_anchored_house_formula",
        "retain_internal_inference",
    }
    for metric_id, rec in payload["metrics"].items():
        assert rec["methodology_execution_decision"] in valid_decisions, metric_id
        assert isinstance(rec["methodology_execution_reason"], str) and rec["methodology_execution_reason"], metric_id

    metrics = payload["metrics"]
    assert metrics["capital_return.dividend_payer_flag"]["classification"] == "external_raw_plus_deterministic_formula"
    assert metrics["capital_structure.net_debt"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["capital_return.dividend_payer_flag"]["methodology_execution_decision"] == "keep_externally_anchored_house_formula"
    assert metrics["capital_return.last_dividend_event_type"]["current_alignment_status"] == "aligned_with_documented_logic"
    assert metrics["strategic.action_frequency_24m"]["current_alignment_status"] == "aligned_with_documented_logic"
    assert metrics["strategic.last_action_type"]["current_alignment_status"] == "aligned_with_documented_logic"
    assert metrics["strategic.recent_actions_count_24m"]["current_alignment_status"] == "aligned_with_documented_logic"
    assert metrics["market.market_cap"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["market.pe_ratio"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["market.pe_percentile_peers"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["market.pe_percentile_history"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["macro.sp500_pe_ttm"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["macro.sp500_pe_ttm_percentile_history"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["macro.us10y_treasury_yield"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["macro.us10y_treasury_yield_percentile_history"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["macro.us_ig_oas"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["macro.us_ig_oas_percentile_history"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["macro.us_hy_all_in_yield"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["macro.us_hy_all_in_yield_percentile_history"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["macro.real_gdp_growth_yoy"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["macro.real_gdp_growth_yoy_percentile_history"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["market.volatility_30d"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["market.volatility_90d"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["market.drawdown_90d"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["operating.ebitda_margin_ttm"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["operating.revenue_yoy_last_q"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["operating.revenue_cagr_3y"]["methodology_execution_decision"] == "adopt_exact_external_methodology"
    assert metrics["operating.fcf_conversion"]["methodology_execution_decision"] == "keep_externally_anchored_house_formula"
    assert metrics["market.credit_window_proxy"]["methodology_execution_decision"] == "retain_internal_inference"


