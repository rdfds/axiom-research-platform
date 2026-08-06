from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.company_state_builder import CompanyStateBuilder

DEFAULT_AS_OF = "2026-02-28"
DEFAULT_OUT = ROOT / "out" / "consumer_industrial_metric_audit.json"
SECTORS = ("Consumer Discretionary", "Consumer Staples", "Industrials")
METRIC_KEYS = [
    "taxonomy.archetype",
    "taxonomy.sector",
    "taxonomy.subsector",
    "capital_structure.total_debt_reported",
    "capital_structure.total_debt_market",
    "capital_structure.net_debt_reported",
    "capital_structure.net_debt_market",
    "capital_structure.gross_leverage_reported",
    "capital_structure.gross_leverage_market",
    "capital_structure.net_leverage_reported",
    "capital_structure.net_leverage_market",
    "capital_structure.interest_coverage",
    "capital_structure.fixed_charge_coverage",
    "liquidity.usable_cash_market",
    "liquidity.available_for_actions_market",
]


def _sample_companies(limit_per_sector: int) -> List[Dict[str, Any]]:
    fundamentals = ROOT / "data" / "refinitiv" / "fundamentals_all.parquet"
    identifiers = ROOT / "data" / "inputs_layer" / "entity_identifier.parquet"
    con = duckdb.connect()
    query = f"""
    WITH fundamentals AS (
      SELECT
        split_part(Instrument, '.', 1) AS ticker,
        Instrument AS instrument,
        "Company Common Name" AS company_name,
        "Company Market Cap" AS market_cap,
        "GICS Sector Name" AS sector,
        "GICS Industry Name" AS industry
      FROM read_parquet('{fundamentals.as_posix()}', union_by_name=true)
      WHERE "GICS Sector Name" IN ({", ".join([repr(x) for x in SECTORS])})
    ),
    identifiers AS (
      SELECT
        entity_id AS company_id,
        upper(identifier_value) AS ticker
      FROM read_parquet('{identifiers.as_posix()}', union_by_name=true)
      WHERE lower(identifier_type) = 'ticker'
    ),
    joined AS (
      SELECT
        i.company_id,
        f.ticker,
        f.instrument,
        f.company_name,
        f.market_cap,
        f.sector,
        f.industry,
        row_number() OVER (PARTITION BY f.sector ORDER BY f.market_cap DESC NULLS LAST, f.company_name) AS sector_rank
      FROM fundamentals f
      JOIN identifiers i
        ON upper(f.ticker) = upper(i.ticker)
    )
    SELECT company_id, ticker, instrument, company_name, market_cap, sector, industry
    FROM joined
    WHERE sector_rank <= {int(limit_per_sector)}
    ORDER BY sector, sector_rank
    """
    rows = con.execute(query).fetchall()
    return [
        {
            "company_id": row[0],
            "ticker": row[1],
            "instrument": row[2],
            "company_name": row[3],
            "market_cap": row[4],
            "sector": row[5],
            "industry": row[6],
        }
        for row in rows
    ]


def _metric_view(snapshot: Dict[str, Any], key: str) -> Dict[str, Any]:
    feat = (snapshot.get("features") or {}).get(key) or {}
    return {
        "value": feat.get("value"),
        "missing_reason": feat.get("missing_reason"),
        "support_mode": feat.get("support_mode"),
        "applicability_status": feat.get("applicability_status"),
        "quality_flags": feat.get("quality_flags"),
    }


def build_audit(as_of: str, limit_per_sector: int) -> Dict[str, Any]:
    builder = CompanyStateBuilder()
    companies = _sample_companies(limit_per_sector)
    report: Dict[str, Any] = {
        "as_of": as_of,
        "companies_requested": len(companies),
        "results": [],
    }
    for company in companies:
        result: Dict[str, Any] = dict(company)
        try:
            snapshot = builder.build(company["company_id"], as_of)
            result["market_metric_context"] = snapshot.provenance.get("market_metric_context")
            result["metrics"] = {key: _metric_view(snapshot, key) for key in METRIC_KEYS}
        except Exception as exc:
            result["error"] = str(exc)
        report["results"].append(result)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=DEFAULT_AS_OF)
    parser.add_argument("--limit-per-sector", type=int, default=2)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    report = build_audit(args.as_of, args.limit_per_sector)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
