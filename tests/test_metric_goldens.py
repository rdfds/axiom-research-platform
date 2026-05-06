from __future__ import annotations

from copy import deepcopy

from src.metric_goldens import DEFAULT_GOLDENS_PATH, load_metric_goldens, validate_golden_case, validate_metric_goldens


EXPECTED_CONSUMER_INDUSTRIAL_ARCHETYPES = {
    "lease_heavy",
    "consumer_grocery_retail",
    "consumer_internet_retail",
    "consumer_branded_staples",
    "consumer_apparel_luxury",
    "consumer_durables",
    "diversified_consumer_services",
    "automotive_oem",
    "auto_components",
    "aerospace_defense",
    "machinery_capital_goods",
    "industrial_distributors",
    "packaging_containers",
    "building_products",
    "transport_logistics",
    "professional_services",
    "commercial_services",
    "construction_engineering",
    "industrial_conglomerates",
}


def test_consumer_industrial_metric_goldens_cover_all_explicit_archetypes():
    payload = load_metric_goldens(DEFAULT_GOLDENS_PATH)
    actual = {
        case["expected_taxonomy"]["archetype"]
        for case in payload["cases"]
    }
    assert actual == EXPECTED_CONSUMER_INDUSTRIAL_ARCHETYPES


def test_consumer_industrial_metric_goldens_pass_end_to_end():
    report = validate_metric_goldens(DEFAULT_GOLDENS_PATH)
    assert report["summary"]["failed_cases"] == 0
    assert report["summary"]["passed_cases"] == len(report["results"])


def test_metric_goldens_flag_incorrect_expected_values():
    payload = load_metric_goldens(DEFAULT_GOLDENS_PATH)
    case = deepcopy(payload["cases"][0])
    case["metrics"]["capital_structure.total_debt_market"]["expected_value"] = 999.0
    result = validate_golden_case(case)
    assert not result["passed"]
    assert any(error.startswith("value_mismatch:capital_structure.total_debt_market") for error in result["errors"])
