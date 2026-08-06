#!/usr/bin/env python3
"""Audit the v1 input-layer artifact for impossible values and formula drift."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


MAX_SEC_FACT_AGE_DAYS = 550
LEASE_EXACT_MAX_AGE_DAYS = 220
LEASE_STALE_CARRY_FORWARD_MAX_AGE_DAYS = 420
DEBT_COMPONENT_ALIGNMENT_MAX_GAP_DAYS = 10
LEASE_COMPONENT_ALIGNMENT_MAX_GAP_DAYS = 10
CASH_COMPONENT_ALIGNMENT_MAX_GAP_DAYS = 10
BALANCE_SHEET_EXACT_MAX_AGE_DAYS = 130
GROUPED_CASH_STATEMENT_REPAIR_MAX_AGE_DAYS = 450
APPROVED_MACRO_METRICS = [
    "macro.fed_funds_effective",
    "macro.sofr",
    "macro.sofr_or_fed_funds",
    "macro.ust_2y_yield",
    "macro.ust_10y_yield",
    "macro.curve_2s10s",
    "macro.ig_oas",
    "macro.hy_oas",
    "macro.real_gdp_growth_yoy",
    "macro.cpi_yoy",
    "macro.unemployment_rate",
    "macro.retail_sales_yoy",
    "macro.wti_crude",
]
AUDITED_METRICS = {
    "market.market_cap_provider_direct",
    "operating.revenue_ttm_provider_direct",
    "operating.ebitda_ltm_provider_direct",
    "earnings.net_income_ttm_provider_direct",
    "liquidity.cash_and_short_term_investments_provider_direct",
    "capital_structure.total_debt_provider_direct",
    "capital_structure.net_debt_standardized",
    "capital_structure.gross_leverage_standardized",
    "capital_structure.net_leverage_standardized",
    "operating.ebitda_margin_standardized",
    "earnings.net_margin_standardized",
    "market.price_spot",
    "market.total_return_1m_standardized",
    "market.total_return_3m_standardized",
    "market.total_return_6m_standardized",
    "market.total_return_12m_standardized",
    "liquidity.cash_and_equivalents_statement_direct",
    "capital_structure.current_debt_statement_direct",
    "capital_structure.long_term_debt_statement_direct",
    "operating.ebit_statement_direct",
    "capital_structure.interest_expense_statement_direct",
    "liquidity.restricted_cash_sec_exact",
    "liquidity.marketable_securities_sec_exact",
    "liquidity.revolver_undrawn_sec_exact",
    "capital_structure.lease_liabilities_sec_exact",
    "capital_structure.debt_like_obligations_normalized",
    "capital_structure.net_pension_liability",
    "capital_structure.other_postretirement_benefit_liability",
    "capital_structure.combined_retirement_liability",
    "capital_structure.debt_like_obligations_including_pension",
    "capital_structure.debt_like_obligations_including_retirement",
    "liquidity.available_liquidity_normalized",
    "operating.operating_earnings_normalized",
    "capital_structure.net_debt_normalized",
    "capital_structure.net_debt_including_pension",
    "capital_structure.net_debt_including_retirement",
    "capital_structure.gross_leverage_normalized",
    "capital_structure.gross_leverage_including_pension",
    "capital_structure.gross_leverage_including_retirement",
    "capital_structure.net_leverage_normalized",
    "capital_structure.net_leverage_including_pension",
    "capital_structure.net_leverage_including_retirement",
    *APPROVED_MACRO_METRICS,
}
RESTRICTED_CASH_EXACT_CONCEPTS = {
    "RestrictedCash",
    "RestrictedCashCurrent",
    "RestrictedCashNoncurrent",
    "RestrictedCashAndCashEquivalentsNoncurrent",
}
RESTRICTED_CASH_MIXED_FALLBACK_CONCEPTS = {
    "RestrictedCashAndCashEquivalents",
    "RestrictedCashAndCashEquivalentsAtCarryingValue",
    "RestrictedCashAndInvestmentsCurrent",
}
RESTRICTED_CASH_TOTAL_RECONCILIATION_CONCEPT = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
MARKETABLE_SECURITY_EXACT_CONCEPTS = {
    "ShortTermInvestments",
    "MarketableSecurities",
    "AvailableForSaleSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    "AvailableForSaleDebtSecuritiesCurrent",
    "MarketableSecuritiesCurrent",
}
LEASE_APPROVED_STALE_SUPPORT_OVERRIDES = {
    "stale_internally_consistent_lease_carry_forward",
    "stale_liability_total_corroborated_by_fresh_rou_asset",
}
LEASE_APPROVED_MIXED_SUPPORT_OVERRIDES = {
    "mixed_fresh_and_stale_components_corroborated_by_fresh_rou_asset",
    "mixed_fresh_and_stale_components_rebased_by_rou_basis_delta",
}
GROUPED_CASH_APPROVED_REPAIR_MODES = {
    "cash_and_equivalents_plus_marketable_securities",
    "cash_and_equivalents_plus_inferred_zero_short_term_investments",
}


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--entity-identifier-path")
    parser.add_argument("--raw-timeseries-path")
    parser.add_argument("--crsp-market-cache-path")
    parser.add_argument("--crsp-daily-root")
    return parser.parse_args()


def _value(node: dict[str, Any] | None) -> float | None:
    if not node:
        return None
    value = node.get("value")
    return None if value is None else float(value)


def _support(node: dict[str, Any] | None) -> str:
    if not node:
        return "missing_metric"
    return node.get("support_mode") or "missing_metric"


def _approx_equal(a: float | None, b: float | None, tolerance: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is b
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= tolerance * scale


def _company_label(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("features") or {}
    provider_fields = [
        "market.market_cap_provider_direct",
        "liquidity.cash_and_short_term_investments_provider_direct",
        "capital_structure.total_debt_provider_direct",
    ]
    company_name = None
    reference_instrument = None
    for metric_name in provider_fields:
        breakdown = (features.get(metric_name) or {}).get("component_breakdown") or {}
        company_name = company_name or breakdown.get("provider_company_name")
        reference_instrument = reference_instrument or breakdown.get("reference_instrument")
    return {
        "company_id": row.get("company_id"),
        "company_name": company_name,
        "reference_instrument": reference_instrument,
    }


def _iter_component_ends(component_breakdown: Any) -> list[str]:
    ends: list[str] = []
    if isinstance(component_breakdown, dict):
        if "end" in component_breakdown and component_breakdown["end"]:
            ends.append(component_breakdown["end"])
        for value in component_breakdown.values():
            ends.extend(_iter_component_ends(value))
    elif isinstance(component_breakdown, list):
        for item in component_breakdown:
            ends.extend(_iter_component_ends(item))
    return ends


def _latest_end_age_days(as_of_date: date, component_breakdown: Any) -> int | None:
    ends = []
    for end_text in _iter_component_ends(component_breakdown):
        try:
            ends.append(date.fromisoformat(end_text))
        except ValueError:
            continue
    if not ends:
        return None
    latest_end = max(ends)
    return (as_of_date - latest_end).days


def _iter_component_concepts(component_breakdown: Any) -> list[str]:
    concepts: list[str] = []
    if isinstance(component_breakdown, dict):
        concept = component_breakdown.get("concept")
        if concept:
            concepts.append(str(concept))
        for value in component_breakdown.values():
            concepts.extend(_iter_component_concepts(value))
    elif isinstance(component_breakdown, list):
        for item in component_breakdown:
            concepts.extend(_iter_component_concepts(item))
    return concepts


def _restricted_cash_breakdown_is_semantically_valid(component_breakdown: Any) -> bool:
    if not isinstance(component_breakdown, dict):
        return False
    mode = component_breakdown.get("mode")
    concepts = _iter_component_concepts(component_breakdown)
    if not concepts:
        concept = component_breakdown.get("concept")
        return concept in RESTRICTED_CASH_EXACT_CONCEPTS

    if mode in {
        "cash_plus_restricted_total_minus_cash_equivalents",
        "cash_plus_restricted_total_minus_grouped_cash_cash_only_proxy",
    }:
        return (
            RESTRICTED_CASH_TOTAL_RECONCILIATION_CONCEPT in concepts
            and (
                "cash_and_equivalents_statement_direct" in component_breakdown
                or "cash_and_short_term_investments_provider_direct_cash_only_proxy" in component_breakdown
            )
        )

    if mode == "mixed_total_restricted_cash_fallback":
        return all(
            found in RESTRICTED_CASH_MIXED_FALLBACK_CONCEPTS
            for found in concepts
        )

    return all(found in RESTRICTED_CASH_EXACT_CONCEPTS for found in concepts)


def _selected_lease_component_ends(component_breakdown: Any) -> list[str]:
    ends: list[str] = []
    if not isinstance(component_breakdown, dict):
        return ends
    for key in (
        "components",
        "current_components",
        "noncurrent_components",
        "payments_due_component",
        "undiscounted_excess_component",
        "operating_component",
        "finance_component",
    ):
        value = component_breakdown.get(key)
        if value is not None:
            ends.extend(_iter_component_ends(value))
    return ends


def _selected_lease_support_overrides(component_breakdown: Any) -> set[str]:
    overrides: set[str] = set()
    if not isinstance(component_breakdown, dict):
        return overrides
    for key in ("operating_component", "finance_component"):
        value = component_breakdown.get(key)
        if isinstance(value, dict):
            support_override = value.get("support_override")
            if support_override:
                overrides.add(str(support_override))
    return overrides


def _grouped_cash_age_days(as_of_date: date, component_breakdown: Any) -> int | None:
    if not isinstance(component_breakdown, dict):
        return None
    mode = component_breakdown.get("mode")
    if mode in GROUPED_CASH_APPROVED_REPAIR_MODES:
        statement_component = component_breakdown.get("cash_and_equivalents_statement_direct")
        return _latest_end_age_days(as_of_date, statement_component)
    return _latest_end_age_days(as_of_date, component_breakdown)


def _expected_available_liquidity_from_breakdown(component_breakdown: Any) -> float | None:
    if not isinstance(component_breakdown, dict):
        return None
    source_metric = component_breakdown.get("cash_basis_source_metric")
    grouped_cash = component_breakdown.get("grouped_cash_provider_direct")
    cash_eq = component_breakdown.get("cash_and_equivalents_statement_direct")
    marketable = component_breakdown.get("marketable_securities_sec_exact")
    restricted = component_breakdown.get("restricted_cash_sec_exact")
    restricted_already_excluded = bool(
        component_breakdown.get("restricted_cash_already_excluded_from_cash_basis")
    )
    revolver = component_breakdown.get("revolver_undrawn_exact")
    not_freely_transferable = component_breakdown.get("not_freely_transferable_cash_disclosed")

    if source_metric in {
        "liquidity.cash_and_equivalents_statement_direct",
        "liquidity.cash_and_equivalents_companyfacts_exact",
    }:
        base = cash_eq
        if base is None:
            return None
        base = float(base) + float(marketable or 0.0)
    elif source_metric == "liquidity.cash_and_short_term_investments_provider_direct_cash_component":
        if grouped_cash is None:
            return None
        base = float(grouped_cash) + float(marketable or 0.0)
    elif source_metric == "liquidity.cash_and_short_term_investments_provider_direct":
        if grouped_cash is None:
            return None
        base = float(grouped_cash)
    else:
        return None

    if not restricted_already_excluded and restricted is not None:
        base -= float(restricted)
    if not_freely_transferable is not None:
        base -= float(not_freely_transferable)
    if revolver is not None:
        base += float(revolver)
    return max(0.0, float(base))


def _expected_debt_like_from_breakdown(component_breakdown: Any) -> float | None:
    if not isinstance(component_breakdown, dict):
        return None
    baseline = component_breakdown.get("baseline_value")
    if baseline is None:
        return None
    expected = float(baseline)
    lease_value = component_breakdown.get("lease_liabilities_sec_exact")
    if not component_breakdown.get("lease_liabilities_inferred_zero") and lease_value is not None:
        expected += float(lease_value)
    if component_breakdown.get("debt_like_floored_to_total_debt"):
        expected = max(expected, float(baseline))
    return expected


def _selected_total_debt_breakdown(component_breakdown: Any) -> Any:
    if not isinstance(component_breakdown, dict):
        return component_breakdown
    return {
        key: value
        for key, value in component_breakdown.items()
        if key not in {"repaired_prior_breakdown", "prior_total_debt_breakdown"}
    }


def _lease_selected_age_days(as_of_date: date, component_breakdown: Any) -> int | None:
    ends = []
    for end_text in _selected_lease_component_ends(component_breakdown):
        parsed = _parse_iso_date(end_text)
        if parsed is not None:
            ends.append(parsed)
    if not ends:
        return None
    return (as_of_date - max(ends)).days


def _lease_selected_gap_days(component_breakdown: Any) -> int | None:
    ends = []
    for end_text in _selected_lease_component_ends(component_breakdown):
        parsed = _parse_iso_date(end_text)
        if parsed is not None:
            ends.append(parsed)
    if len(ends) < 2:
        return 0 if ends else None
    return (max(ends) - min(ends)).days


def _selected_component_gap_days(component_breakdown: Any, keys: tuple[str, ...]) -> int | None:
    if not isinstance(component_breakdown, dict):
        return None
    ends = []
    for key in keys:
        value = component_breakdown.get(key)
        if value is not None:
            for end_text in _iter_component_ends(value):
                parsed = _parse_iso_date(end_text)
                if parsed is not None:
                    ends.append(parsed)
    if len(ends) < 2:
        return 0 if ends else None
    return (max(ends) - min(ends)).days


def _record_issue(container: dict[str, Any], category: str, metric_name: str, example: dict[str, Any]) -> None:
    metric_bucket = container.setdefault(category, {}).setdefault(
        metric_name,
        {"count": 0, "examples": []},
    )
    metric_bucket["count"] += 1
    if len(metric_bucket["examples"]) < 5:
        metric_bucket["examples"].append(example)


def _permno_map(entity_identifier_path: Path) -> pd.DataFrame:
    ids = pd.read_parquet(entity_identifier_path)
    ids = ids[ids["identifier_type"].astype(str).str.lower() == "permno"].copy()
    ids["permno"] = ids["identifier_value"].astype(str).str.strip()
    return ids[["entity_id", "permno"]].drop_duplicates()


def _load_price_history(raw_timeseries_path: Path, permnos: list[str]) -> pd.DataFrame:
    permno_sql = ",".join(f"'{permno}'" for permno in sorted(set(permnos)))
    query = f"""
        SELECT
            CAST(entity_id AS VARCHAR) AS permno,
            CAST(trade_date AS DATE) AS trade_date,
            close,
            adjusted_close,
            ret,
            retx
        FROM read_parquet('{raw_timeseries_path}')
        WHERE series_type = 'price'
          AND CAST(entity_id AS VARCHAR) IN ({permno_sql})
    """
    prices = duckdb.sql(query).fetchdf()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], utc=True).dt.normalize()
    prices = prices.sort_values(["permno", "trade_date"]).drop_duplicates(["permno", "trade_date"], keep="last")
    return prices


def _load_crsp_market_cache(crsp_market_cache_path: Path, permnos: list[str]) -> pd.DataFrame:
    permno_sql = ",".join(f"'{permno}'" for permno in sorted(set(permnos)))
    query = f"""
        SELECT
            CAST(permno AS VARCHAR) AS permno,
            CAST(trade_date AS DATE) AS trade_date,
            close_price,
            price_proxy,
            total_return,
            price_return,
            shares_outstanding,
            daily_cap,
            delist_flag
        FROM read_parquet('{crsp_market_cache_path}')
        WHERE CAST(permno AS VARCHAR) IN ({permno_sql})
    """
    prices = duckdb.sql(query).fetchdf()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], utc=True).dt.normalize()
    prices = prices.sort_values(["permno", "trade_date"]).drop_duplicates(["permno", "trade_date"], keep="last")
    return prices


def _load_crsp_daily_from_repo(
    crsp_daily_root: Path,
    permnos: list[str],
    *,
    min_asof_date: pd.Timestamp,
    max_asof_date: pd.Timestamp,
) -> pd.DataFrame:
    if not permnos:
        return pd.DataFrame()
    start_year = int(min_asof_date.year) - 1
    end_year = int(max_asof_date.year)
    files: list[Path] = []
    for year in range(start_year, end_year + 1):
        candidate = crsp_daily_root / f"dsf_{year:04d}-01-01_to_{year:04d}-12-31.parquet"
        if candidate.exists():
            files.append(candidate)
    if not files:
        return pd.DataFrame()

    permno_sql = ",".join(f"'{permno}'" for permno in sorted(set(permnos)))
    min_trade_date = (min_asof_date - pd.Timedelta(days=370)).date().isoformat()
    max_trade_date = max_asof_date.date().isoformat()
    selects = []
    for file_path in files:
        selects.append(
            f"""
            SELECT
                CAST(permno AS VARCHAR) AS permno,
                CAST(date AS DATE) AS trade_date,
                ABS(prc) AS close_price,
                ABS(prc) AS price_proxy,
                ret AS total_return,
                retx AS price_return,
                shrout AS shares_outstanding,
                ABS(prc) * shrout AS daily_cap,
                FALSE AS delist_flag
            FROM read_parquet('{file_path.as_posix()}')
            WHERE CAST(permno AS VARCHAR) IN ({permno_sql})
              AND CAST(date AS DATE) >= DATE '{min_trade_date}'
              AND CAST(date AS DATE) <= DATE '{max_trade_date}'
            """
        )
    prices = duckdb.sql(" UNION ALL ".join(selects)).fetchdf()
    if prices.empty:
        return prices
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], utc=True).dt.normalize()
    prices["date_key"] = prices["trade_date"]
    prices = prices.sort_values(["permno", "trade_date"]).drop_duplicates(["permno", "trade_date"], keep="last")
    return prices


def _latest_row_on_or_before(df: pd.DataFrame, date_key: pd.Timestamp) -> pd.Series | None:
    eligible = df[df["trade_date"] <= date_key]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _compound_trailing_return(price_history: pd.DataFrame, as_of_ts: pd.Timestamp, months: int) -> float | None:
    current_row = _latest_row_on_or_before(price_history, as_of_ts)
    if current_row is None:
        return None
    current_trade_date = current_row["trade_date"]
    target_date = current_trade_date - pd.DateOffset(months=months)
    window = price_history[(price_history["trade_date"] > target_date) & (price_history["trade_date"] <= current_trade_date)].copy()
    if window.empty:
        return None
    if window["ret"].notna().all():
        return float((1.0 + window["ret"].astype(float)).prod() - 1.0)
    if window["retx"].notna().all():
        return float((1.0 + window["retx"].astype(float)).prod() - 1.0)
    return None


def _compound_trailing_crsp_return(price_history: pd.DataFrame, as_of_ts: pd.Timestamp, months: int) -> float | None:
    current_row = _latest_row_on_or_before(price_history, as_of_ts)
    if current_row is None:
        return None
    current_trade_date = current_row["trade_date"]
    target_date = current_trade_date - pd.DateOffset(months=months)
    start_row = _latest_row_on_or_before(price_history, target_date)
    if start_row is None:
        return None
    start_trade_date = start_row["trade_date"]
    window = price_history[(price_history["trade_date"] > start_trade_date) & (price_history["trade_date"] <= current_trade_date)].copy()
    if window.empty:
        return None
    if window["total_return"].notna().all():
        return float((1.0 + window["total_return"].astype(float)).prod() - 1.0)
    if window["price_return"].notna().all():
        return float((1.0 + window["price_return"].astype(float)).prod() - 1.0)
    return None


def _open_artifact_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    rows = []
    with _open_artifact_text(artifact_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        raise SystemExit("Artifact is empty")

    as_of_date = date.fromisoformat(rows[0]["as_of_time"][:10])
    as_of_ts = pd.Timestamp(rows[0]["as_of_time"]).tz_convert("UTC").normalize()
    coverage: dict[str, Counter[str]] = defaultdict(Counter)
    issues: dict[str, Any] = {}
    negative_proxy_liquidity: list[dict[str, Any]] = []
    price_history_by_entity: dict[str, pd.DataFrame] = {}
    permno_by_entity: dict[str, str] = {}

    using_crsp_market_cache = bool(args.crsp_market_cache_path)
    using_crsp_daily_root = bool(args.crsp_daily_root)
    if args.entity_identifier_path and (args.raw_timeseries_path or args.crsp_market_cache_path or args.crsp_daily_root):
        permnos = _permno_map(Path(args.entity_identifier_path))
        permno_by_entity = {
            str(entity_id): permno
            for entity_id, permno in permnos[["entity_id", "permno"]].itertuples(index=False)
        }
        needed_permnos = [permno for row in rows if (permno := permno_by_entity.get(str(row.get("company_id"))))]
        if needed_permnos:
            if args.crsp_market_cache_path:
                prices = _load_crsp_market_cache(Path(args.crsp_market_cache_path), needed_permnos)
            elif args.crsp_daily_root:
                prices = _load_crsp_daily_from_repo(
                    Path(args.crsp_daily_root),
                    needed_permnos,
                    min_asof_date=as_of_ts,
                    max_asof_date=as_of_ts,
                )
            else:
                prices = _load_price_history(Path(args.raw_timeseries_path), needed_permnos)
            for permno, frame in prices.groupby("permno"):
                price_history_by_entity[permno] = frame.reset_index(drop=True)

    for row in rows:
        features = row.get("features") or {}
        label = _company_label(row)
        for metric_name in AUDITED_METRICS:
            coverage[metric_name][_support(features.get(metric_name))] += 1

        def node(metric_name: str) -> dict[str, Any]:
            return features.get(metric_name) or {}

        def value(metric_name: str) -> float | None:
            return _value(node(metric_name))

        def support(metric_name: str) -> str:
            return _support(node(metric_name))

        impossible_nonnegative_metrics = [
            "market.market_cap_provider_direct",
            "market.price_spot",
            "operating.revenue_ttm_provider_direct",
            "liquidity.cash_and_short_term_investments_provider_direct",
            "liquidity.cash_and_equivalents_statement_direct",
            "capital_structure.total_debt_provider_direct",
            "capital_structure.current_debt_statement_direct",
            "capital_structure.long_term_debt_statement_direct",
            "liquidity.restricted_cash_sec_exact",
            "liquidity.marketable_securities_sec_exact",
            "capital_structure.lease_liabilities_sec_exact",
            "capital_structure.debt_like_obligations_normalized",
            "capital_structure.gross_leverage_normalized",
            "capital_structure.gross_leverage_standardized",
        ]
        for metric_name in impossible_nonnegative_metrics:
            metric_value = value(metric_name)
            if metric_value is not None and metric_value < 0:
                _record_issue(
                    issues,
                    "impossible_negative_values",
                    metric_name,
                    {**label, "value": metric_value},
                )

        return_metrics = [
            "market.total_return_1m_standardized",
            "market.total_return_3m_standardized",
            "market.total_return_6m_standardized",
            "market.total_return_12m_standardized",
        ]
        for metric_name in return_metrics:
            metric_value = value(metric_name)
            if metric_value is not None and metric_value < -1.0:
                _record_issue(
                    issues,
                    "impossible_return_values",
                    metric_name,
                    {**label, "value": metric_value},
                )

        permno = permno_by_entity.get(str(row.get("company_id")))
        if permno and permno in price_history_by_entity:
            frame = price_history_by_entity[permno]
            spot = value("market.price_spot")
            latest_row = _latest_row_on_or_before(frame, as_of_ts)
            if using_crsp_market_cache or using_crsp_daily_root:
                if latest_row is None:
                    expected_spot = None
                elif pd.notna(latest_row["close_price"]):
                    expected_spot = float(latest_row["close_price"])
                elif pd.notna(latest_row["price_proxy"]):
                    expected_spot = float(latest_row["price_proxy"])
                else:
                    expected_spot = None
            else:
                expected_spot = None if latest_row is None or pd.isna(latest_row["close"]) else float(latest_row["close"])
            if support("market.price_spot") == "exact" and spot is not None and not _approx_equal(spot, expected_spot):
                _record_issue(
                    issues,
                    "formula_mismatches",
                    "market.price_spot",
                    {**label, "expected": expected_spot, "actual": spot},
                )
            for metric_name, months in [
                ("market.total_return_1m_standardized", 1),
                ("market.total_return_3m_standardized", 3),
                ("market.total_return_6m_standardized", 6),
                ("market.total_return_12m_standardized", 12),
            ]:
                actual = value(metric_name)
                expected = (
                    _compound_trailing_crsp_return(frame, as_of_ts, months)
                    if using_crsp_market_cache or using_crsp_daily_root
                    else _compound_trailing_return(frame, as_of_ts, months)
                )
                if support(metric_name) == "exact" and actual is not None and not _approx_equal(actual, expected):
                    _record_issue(
                        issues,
                        "formula_mismatches",
                        metric_name,
                        {**label, "expected": expected, "actual": actual},
                    )

        for metric_name in APPROVED_MACRO_METRICS:
            if value(metric_name) is None:
                _record_issue(issues, "missing_macro_values", metric_name, label)

        sec_component_metrics = [
            "liquidity.restricted_cash_sec_exact",
            "liquidity.marketable_securities_sec_exact",
            "capital_structure.lease_liabilities_sec_exact",
        ]
        for metric_name in sec_component_metrics:
            metric_node = node(metric_name)
            if support(metric_name) != "exact":
                continue
            breakdown = metric_node.get("component_breakdown")
            concept = (breakdown or {}).get("concept")
            if metric_name == "liquidity.restricted_cash_sec_exact":
                age_days = _latest_end_age_days(as_of_date, breakdown)
                if age_days is not None and age_days > MAX_SEC_FACT_AGE_DAYS:
                    _record_issue(
                        issues,
                        "stale_exact_sec_components",
                        metric_name,
                        {**label, "age_days": age_days, "component_breakdown": breakdown},
                    )
                if not _restricted_cash_breakdown_is_semantically_valid(breakdown):
                    _record_issue(
                        issues,
                        "restricted_cash_semantic_mismatch",
                        metric_name,
                        {**label, "concept": concept, "component_breakdown": breakdown},
                    )
            if metric_name == "liquidity.marketable_securities_sec_exact":
                age_days = _latest_end_age_days(as_of_date, breakdown)
                if age_days is not None and age_days > MAX_SEC_FACT_AGE_DAYS:
                    _record_issue(
                        issues,
                        "stale_exact_sec_components",
                        metric_name,
                        {**label, "age_days": age_days, "component_breakdown": breakdown},
                    )
                if concept not in MARKETABLE_SECURITY_EXACT_CONCEPTS:
                    _record_issue(
                        issues,
                        "marketable_security_semantic_mismatch",
                        metric_name,
                        {**label, "concept": concept, "component_breakdown": breakdown},
                    )
            if metric_name == "capital_structure.lease_liabilities_sec_exact":
                age_days = _lease_selected_age_days(as_of_date, breakdown)
                lease_support_overrides = _selected_lease_support_overrides(breakdown)
                stale_override_ok = (
                    bool(lease_support_overrides & LEASE_APPROVED_STALE_SUPPORT_OVERRIDES)
                    and age_days is not None
                    and age_days <= LEASE_STALE_CARRY_FORWARD_MAX_AGE_DAYS
                )
                if age_days is not None and age_days > LEASE_EXACT_MAX_AGE_DAYS and not stale_override_ok:
                    _record_issue(
                        issues,
                        "stale_exact_lease_components",
                        metric_name,
                        {**label, "age_days": age_days, "component_breakdown": breakdown},
                    )
                gap_days = _lease_selected_gap_days(breakdown)
                mixed_override_ok = bool(lease_support_overrides & LEASE_APPROVED_MIXED_SUPPORT_OVERRIDES)
                if gap_days is not None and gap_days > LEASE_COMPONENT_ALIGNMENT_MAX_GAP_DAYS and not mixed_override_ok:
                    _record_issue(
                        issues,
                        "lease_component_period_mismatch_exact",
                        metric_name,
                        {**label, "gap_days": gap_days, "component_breakdown": breakdown},
                    )

        revenue_breakdown = node("operating.revenue_ttm_provider_direct").get("component_breakdown") or {}
        if (
            support("operating.revenue_ttm_provider_direct") == "exact"
            and revenue_breakdown.get("mode") == "latest_fy"
            and revenue_breakdown.get("frame")
        ):
            _record_issue(
                issues,
                "framed_latest_fy_revenue_exact",
                "operating.revenue_ttm_provider_direct",
                {**label, "component_breakdown": revenue_breakdown},
            )

        grouped_cash_breakdown = node("liquidity.cash_and_short_term_investments_provider_direct").get("component_breakdown") or {}
        if support("liquidity.cash_and_short_term_investments_provider_direct") == "exact":
            grouped_cash_mode = grouped_cash_breakdown.get("mode")
            cash_age_days = _grouped_cash_age_days(as_of_date, grouped_cash_breakdown)
            max_cash_age_days = (
                GROUPED_CASH_STATEMENT_REPAIR_MAX_AGE_DAYS
                if grouped_cash_mode in GROUPED_CASH_APPROVED_REPAIR_MODES
                else BALANCE_SHEET_EXACT_MAX_AGE_DAYS
            )
            if cash_age_days is not None and cash_age_days > max_cash_age_days:
                _record_issue(
                    issues,
                    "stale_exact_grouped_cash",
                    "liquidity.cash_and_short_term_investments_provider_direct",
                    {
                        **label,
                        "age_days": cash_age_days,
                        "component_breakdown": grouped_cash_breakdown,
                    },
                )
            cash_gap_days = _selected_component_gap_days(
                grouped_cash_breakdown,
                (
                    "cash",
                    "short_term_investments",
                    "cash_and_equivalents_statement_direct",
                    "marketable_securities_sec_exact",
                ),
            )
            if (
                cash_gap_days is not None
                and cash_gap_days > CASH_COMPONENT_ALIGNMENT_MAX_GAP_DAYS
                and grouped_cash_mode not in GROUPED_CASH_APPROVED_REPAIR_MODES
            ):
                _record_issue(
                    issues,
                    "cash_component_period_mismatch_exact",
                    "liquidity.cash_and_short_term_investments_provider_direct",
                    {
                        **label,
                        "gap_days": cash_gap_days,
                        "component_breakdown": grouped_cash_breakdown,
                    },
                )

        total_debt_breakdown = node("capital_structure.total_debt_provider_direct").get("component_breakdown") or {}
        total_debt_concepts = _iter_component_concepts(_selected_total_debt_breakdown(total_debt_breakdown))
        if any("CapitalLease" in concept for concept in total_debt_concepts):
            if (
                support("capital_structure.total_debt_provider_direct") == "exact"
                and not total_debt_breakdown.get("finance_lease_adjustment")
            ):
                _record_issue(
                    issues,
                    "capital_lease_overlap_unadjusted",
                    "capital_structure.total_debt_provider_direct",
                    {**label, "component_breakdown": total_debt_breakdown},
                )
        total_debt_gap_days = _selected_component_gap_days(
            total_debt_breakdown,
            (
                "combined_debt",
                "current",
                "noncurrent",
                "short_term_borrowings",
                "current_statement_debt",
                "long_term_statement_debt",
            ),
        )
        if (
            support("capital_structure.total_debt_provider_direct") == "exact"
            and total_debt_gap_days is not None
            and total_debt_gap_days > DEBT_COMPONENT_ALIGNMENT_MAX_GAP_DAYS
        ):
            _record_issue(
                issues,
                "debt_component_period_mismatch_exact",
                "capital_structure.total_debt_provider_direct",
                {
                    **label,
                    "gap_days": total_debt_gap_days,
                    "component_breakdown": total_debt_breakdown,
                },
            )
        if (
            support("capital_structure.total_debt_provider_direct") == "exact"
            and total_debt_breakdown.get("mode") == "statement_direct_current_plus_noncurrent_debt"
        ):
            current_statement = total_debt_breakdown.get("current_statement_debt") or {}
            long_term_statement = total_debt_breakdown.get("long_term_statement_debt") or {}
            statement_age_days = _latest_end_age_days(as_of_date, total_debt_breakdown)
            if statement_age_days is not None and statement_age_days > BALANCE_SHEET_EXACT_MAX_AGE_DAYS:
                _record_issue(
                    issues,
                    "stale_exact_statement_debt",
                    "capital_structure.total_debt_provider_direct",
                    {
                        **label,
                        "age_days": statement_age_days,
                        "component_breakdown": total_debt_breakdown,
                    },
                )
            current_source = current_statement.get("source_type")
            long_term_source = long_term_statement.get("source_type")
            if current_source and long_term_source and current_source != long_term_source:
                _record_issue(
                    issues,
                    "statement_debt_source_mismatch_exact",
                    "capital_structure.total_debt_provider_direct",
                    {
                        **label,
                        "component_breakdown": total_debt_breakdown,
                    },
                )

        # Formula audits
        grouped_cash = value("liquidity.cash_and_short_term_investments_provider_direct")
        grouped_cash_support = support("liquidity.cash_and_short_term_investments_provider_direct")
        cash_eq = value("liquidity.cash_and_equivalents_statement_direct")
        restricted_cash_sec_support = support("liquidity.restricted_cash_sec_exact")
        marketable_sec_support = support("liquidity.marketable_securities_sec_exact")
        restricted_cash_sec_missing = node("liquidity.restricted_cash_sec_exact").get("missing_reason")
        marketable_sec_missing = node("liquidity.marketable_securities_sec_exact").get("missing_reason")
        restricted_cash = (
            value("liquidity.restricted_cash_sec_exact")
            if restricted_cash_sec_support == "exact"
            else (
                0.0
                if restricted_cash_sec_support == "unsupported" and restricted_cash_sec_missing == "sec_concept_absent"
                else (value("liquidity.restricted_cash") if support("liquidity.restricted_cash") == "exact" else None)
            )
        )
        marketable = (
            value("liquidity.marketable_securities_sec_exact")
            if marketable_sec_support == "exact"
            else (
                0.0
                if marketable_sec_support == "unsupported" and marketable_sec_missing == "sec_concept_absent"
                else (value("liquidity.marketable_securities") if support("liquidity.marketable_securities") == "exact" else None)
            )
        )
        revolver = (
            value("liquidity.revolver_undrawn_sec_exact")
            if support("liquidity.revolver_undrawn_sec_exact") == "exact"
            else (
                value("liquidity.revolver_undrawn")
                if support("liquidity.revolver_undrawn") == "exact"
                else None
            )
        )
        available_liquidity = value("liquidity.available_liquidity_normalized")
        expected_avail = _expected_available_liquidity_from_breakdown(
            (node("liquidity.available_liquidity_normalized") or {}).get("component_breakdown")
        )
        if expected_avail is None:
            if grouped_cash is not None and grouped_cash_support == "exact":
                expected_avail = grouped_cash
            elif cash_eq is not None and marketable is not None:
                expected_avail = cash_eq + marketable
            elif cash_eq is not None and grouped_cash is not None and abs(grouped_cash - cash_eq) <= 1.0:
                expected_avail = cash_eq
            elif grouped_cash is not None:
                expected_avail = grouped_cash
            elif cash_eq is not None:
                expected_avail = cash_eq
            else:
                expected_avail = None
            if expected_avail is not None:
                if restricted_cash is not None:
                    expected_avail -= restricted_cash
                if revolver is not None:
                    expected_avail += revolver
                if expected_avail < 0:
                    expected_avail = 0.0
        if available_liquidity is not None and not _approx_equal(available_liquidity, expected_avail):
            _record_issue(
                issues,
                "formula_mismatches",
                "liquidity.available_liquidity_normalized",
                {**label, "expected": expected_avail, "actual": available_liquidity},
            )

        debt_like = value("capital_structure.debt_like_obligations_normalized")
        total_debt = value("capital_structure.total_debt_provider_direct")
        total_debt_support = support("capital_structure.total_debt_provider_direct")
        current_debt = value("capital_structure.current_debt_statement_direct")
        current_debt_support = support("capital_structure.current_debt_statement_direct")
        long_term_debt = value("capital_structure.long_term_debt_statement_direct")
        long_term_debt_support = support("capital_structure.long_term_debt_statement_direct")
        lease = value("capital_structure.lease_liabilities_sec_exact")
        expected_debt_like = _expected_debt_like_from_breakdown(
            (node("capital_structure.debt_like_obligations_normalized") or {}).get("component_breakdown")
        )
        if expected_debt_like is None:
            if total_debt is not None and total_debt_support == "exact":
                expected_debt_base = total_debt
            elif (
                current_debt is not None
                and long_term_debt is not None
                and current_debt_support == "exact"
                and long_term_debt_support == "exact"
            ):
                expected_debt_base = current_debt + long_term_debt
            elif total_debt is not None:
                expected_debt_base = total_debt
            elif current_debt is not None or long_term_debt is not None:
                expected_debt_base = float((current_debt or 0.0) + (long_term_debt or 0.0))
            else:
                expected_debt_base = None
            expected_debt_like = None if expected_debt_base is None else expected_debt_base + (lease or 0.0)
        if debt_like is not None and not _approx_equal(debt_like, expected_debt_like):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.debt_like_obligations_normalized",
                {**label, "expected": expected_debt_like, "actual": debt_like},
            )

        pension_liability = value("capital_structure.net_pension_liability")
        if pension_liability is not None and pension_liability < 0:
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.net_pension_liability",
                {**label, "expected": "non_negative", "actual": pension_liability},
            )

        other_postretirement_liability = value("capital_structure.other_postretirement_benefit_liability")
        if other_postretirement_liability is not None and other_postretirement_liability < 0:
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.other_postretirement_benefit_liability",
                {**label, "expected": "non_negative", "actual": other_postretirement_liability},
            )

        combined_retirement_liability = value("capital_structure.combined_retirement_liability")
        if combined_retirement_liability is not None and combined_retirement_liability < 0:
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.combined_retirement_liability",
                {**label, "expected": "non_negative", "actual": combined_retirement_liability},
            )

        debt_like_including_pension = value("capital_structure.debt_like_obligations_including_pension")
        expected_debt_like_including_pension = (
            None if debt_like is None else debt_like + (pension_liability or 0.0)
        )
        if debt_like_including_pension is not None and not _approx_equal(
            debt_like_including_pension,
            expected_debt_like_including_pension,
        ):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.debt_like_obligations_including_pension",
                {**label, "expected": expected_debt_like_including_pension, "actual": debt_like_including_pension},
            )

        debt_like_including_retirement = value("capital_structure.debt_like_obligations_including_retirement")
        expected_debt_like_including_retirement = (
            None if debt_like is None else debt_like + (combined_retirement_liability or 0.0)
        )
        if debt_like_including_retirement is not None and not _approx_equal(
            debt_like_including_retirement,
            expected_debt_like_including_retirement,
        ):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.debt_like_obligations_including_retirement",
                {
                    **label,
                    "expected": expected_debt_like_including_retirement,
                    "actual": debt_like_including_retirement,
                },
            )

        net_debt_norm = value("capital_structure.net_debt_normalized")
        expected_net_debt = None if debt_like is None or available_liquidity is None else debt_like - available_liquidity
        if net_debt_norm is not None and not _approx_equal(net_debt_norm, expected_net_debt):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.net_debt_normalized",
                {**label, "expected": expected_net_debt, "actual": net_debt_norm},
            )

        net_debt_including_pension = value("capital_structure.net_debt_including_pension")
        expected_net_debt_including_pension = (
            None
            if debt_like_including_pension is None or available_liquidity is None
            else debt_like_including_pension - available_liquidity
        )
        if net_debt_including_pension is not None and not _approx_equal(
            net_debt_including_pension,
            expected_net_debt_including_pension,
        ):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.net_debt_including_pension",
                {**label, "expected": expected_net_debt_including_pension, "actual": net_debt_including_pension},
            )

        net_debt_including_retirement = value("capital_structure.net_debt_including_retirement")
        expected_net_debt_including_retirement = (
            None
            if debt_like_including_retirement is None or available_liquidity is None
            else debt_like_including_retirement - available_liquidity
        )
        if net_debt_including_retirement is not None and not _approx_equal(
            net_debt_including_retirement,
            expected_net_debt_including_retirement,
        ):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.net_debt_including_retirement",
                {
                    **label,
                    "expected": expected_net_debt_including_retirement,
                    "actual": net_debt_including_retirement,
                },
            )

        op_earnings = value("operating.operating_earnings_normalized")
        gross_lev = value("capital_structure.gross_leverage_normalized")
        expected_gross = None
        if debt_like is not None and op_earnings is not None and op_earnings > 0:
            expected_gross = debt_like / op_earnings
        if gross_lev is not None and not _approx_equal(gross_lev, expected_gross):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.gross_leverage_normalized",
                {**label, "expected": expected_gross, "actual": gross_lev},
            )

        gross_lev_including_pension = value("capital_structure.gross_leverage_including_pension")
        expected_gross_including_pension = None
        if debt_like_including_pension is not None and op_earnings is not None and op_earnings > 0:
            expected_gross_including_pension = debt_like_including_pension / op_earnings
        if gross_lev_including_pension is not None and not _approx_equal(
            gross_lev_including_pension,
            expected_gross_including_pension,
        ):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.gross_leverage_including_pension",
                {**label, "expected": expected_gross_including_pension, "actual": gross_lev_including_pension},
            )

        gross_lev_including_retirement = value("capital_structure.gross_leverage_including_retirement")
        expected_gross_including_retirement = None
        if debt_like_including_retirement is not None and op_earnings is not None and op_earnings > 0:
            expected_gross_including_retirement = debt_like_including_retirement / op_earnings
        if gross_lev_including_retirement is not None and not _approx_equal(
            gross_lev_including_retirement,
            expected_gross_including_retirement,
        ):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.gross_leverage_including_retirement",
                {
                    **label,
                    "expected": expected_gross_including_retirement,
                    "actual": gross_lev_including_retirement,
                },
            )

        net_lev = value("capital_structure.net_leverage_normalized")
        expected_net_lev = None
        if net_debt_norm is not None and op_earnings is not None and op_earnings > 0:
            expected_net_lev = net_debt_norm / op_earnings
        if net_lev is not None and not _approx_equal(net_lev, expected_net_lev):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.net_leverage_normalized",
                {**label, "expected": expected_net_lev, "actual": net_lev},
            )

        net_lev_including_pension = value("capital_structure.net_leverage_including_pension")
        expected_net_lev_including_pension = None
        if net_debt_including_pension is not None and op_earnings is not None and op_earnings > 0:
            expected_net_lev_including_pension = net_debt_including_pension / op_earnings
        if net_lev_including_pension is not None and not _approx_equal(
            net_lev_including_pension,
            expected_net_lev_including_pension,
        ):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.net_leverage_including_pension",
                {**label, "expected": expected_net_lev_including_pension, "actual": net_lev_including_pension},
            )

        net_lev_including_retirement = value("capital_structure.net_leverage_including_retirement")
        expected_net_lev_including_retirement = None
        if net_debt_including_retirement is not None and op_earnings is not None and op_earnings > 0:
            expected_net_lev_including_retirement = net_debt_including_retirement / op_earnings
        if net_lev_including_retirement is not None and not _approx_equal(
            net_lev_including_retirement,
            expected_net_lev_including_retirement,
        ):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.net_leverage_including_retirement",
                {
                    **label,
                    "expected": expected_net_lev_including_retirement,
                    "actual": net_lev_including_retirement,
                },
            )

        revenue = value("operating.revenue_ttm_provider_direct")
        ebitda = value("operating.ebitda_ltm_provider_direct")
        ebitda_margin = value("operating.ebitda_margin_standardized")
        expected_ebitda_margin = None if revenue is None or revenue <= 0 or ebitda is None else ebitda / revenue
        if ebitda_margin is not None and not _approx_equal(ebitda_margin, expected_ebitda_margin):
            _record_issue(
                issues,
                "formula_mismatches",
                "operating.ebitda_margin_standardized",
                {**label, "expected": expected_ebitda_margin, "actual": ebitda_margin},
            )

        net_income = value("earnings.net_income_ttm_provider_direct")
        net_margin = value("earnings.net_margin_standardized")
        expected_net_margin = None if revenue is None or revenue <= 0 or net_income is None else net_income / revenue
        if net_margin is not None and not _approx_equal(net_margin, expected_net_margin):
            _record_issue(
                issues,
                "formula_mismatches",
                "earnings.net_margin_standardized",
                {**label, "expected": expected_net_margin, "actual": net_margin},
            )

        std_net_debt = value("capital_structure.net_debt_standardized")
        expected_std_net_debt = None if total_debt is None or grouped_cash is None else total_debt - grouped_cash
        if std_net_debt is not None and not _approx_equal(std_net_debt, expected_std_net_debt):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.net_debt_standardized",
                {**label, "expected": expected_std_net_debt, "actual": std_net_debt},
            )

        std_gross_lev = value("capital_structure.gross_leverage_standardized")
        expected_std_gross = None if total_debt is None or ebitda is None or ebitda <= 0 else total_debt / ebitda
        if std_gross_lev is not None and not _approx_equal(std_gross_lev, expected_std_gross):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.gross_leverage_standardized",
                {**label, "expected": expected_std_gross, "actual": std_gross_lev},
            )

        std_net_lev = value("capital_structure.net_leverage_standardized")
        expected_std_net_lev = None if std_net_debt is None or ebitda is None or ebitda <= 0 else std_net_debt / ebitda
        if std_net_lev is not None and not _approx_equal(std_net_lev, expected_std_net_lev):
            _record_issue(
                issues,
                "formula_mismatches",
                "capital_structure.net_leverage_standardized",
                {**label, "expected": expected_std_net_lev, "actual": std_net_lev},
            )

        if available_liquidity is not None and available_liquidity < 0 and support("liquidity.available_liquidity_normalized") == "proxy_missing_component":
            negative_proxy_liquidity.append(
                {
                    **label,
                    "available_liquidity_normalized": available_liquidity,
                    "component_breakdown": (node("liquidity.available_liquidity_normalized") or {}).get("component_breakdown"),
                    "restricted_cash_breakdown": (node("liquidity.restricted_cash_sec_exact") or {}).get("component_breakdown"),
                    "marketable_breakdown": (node("liquidity.marketable_securities_sec_exact") or {}).get("component_breakdown"),
                }
            )

    macro_variation = {}
    for metric_name in APPROVED_MACRO_METRICS:
        values = sorted({(_value((row.get("features") or {}).get(metric_name))) for row in rows if _value((row.get("features") or {}).get(metric_name)) is not None})
        macro_variation[metric_name] = {"unique_values": len(values), "sample_values": values[:5]}

    summary = {
        "artifact_path": str(artifact_path),
        "row_count": len(rows),
        "coverage": {
            metric_name: dict(sorted(counter.items()))
            for metric_name, counter in sorted(coverage.items())
        },
        "issues": issues,
        "remaining_negative_proxy_liquidity_cases": {
            "count": len(negative_proxy_liquidity),
            "examples": negative_proxy_liquidity[:9],
        },
        "macro_variation": macro_variation,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
