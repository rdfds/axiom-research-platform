"""
Warehouse Pivot Helpers
=======================
Convert long-form warehouse_financials into wide Compustat-like frames
as-of a given date.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from .asof_store import AsOfWarehouse


LINE_ITEM_MAP = {
    "Revenue": "revtq",
    "COGS": "cogsq",
    "SGA": "xsgaq",
    "EBITDA": "oibdpq",
    "OperatingIncome": "oiadpq",
    "NetIncome": "niq",
    "InterestExpense": "xintq",
    "TotalAssets": "atq",
    "CurrentAssets": "actq",
    "Cash": "cheq",
    "Receivables": "rectq",
    "Inventory": "invtq",
    "PP&E": "ppentq",
    "TotalLiabilities": "ltq",
    "CurrentLiabilities": "lctq",
    "DebtCurrent": "dlcq",
    "DebtLongTerm": "dlttq",
    "CommonEquity": "ceqq",
    "TotalEquity": "seqq",
    "Capex": "capxy",
    "OperatingCashFlow": "oancfy",
    "EPS": "epspxq",
    "SharesOut": "cshoq",
}


def fetch_financials_asof(
    as_of: datetime,
    warehouse: Optional[AsOfWarehouse] = None,
    company_id: Optional[str] = None,
) -> pd.DataFrame:
    warehouse = warehouse or AsOfWarehouse()
    where = f"company_id = '{company_id}'" if company_id else None
    df = warehouse.query("warehouse_financials", as_of=as_of, where=where)
    if df.empty:
        return df

    df = df.copy()
    df["line_item"] = df["line_item"].map(LINE_ITEM_MAP).fillna(df["line_item"])
    df["event_time"] = pd.to_datetime(df["event_time"])

    pivot = (
        df.pivot_table(
            index=["company_id", "event_time"],
            columns="line_item",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )

    pivot = pivot.sort_values(["company_id", "event_time"], ascending=[True, False])
    return pivot


def latest_financials_asof(
    as_of: datetime,
    warehouse: Optional[AsOfWarehouse] = None,
) -> pd.DataFrame:
    df = fetch_financials_asof(as_of, warehouse)
    if df.empty:
        return df
    latest = df.groupby("company_id").head(1).reset_index(drop=True)
    return latest
