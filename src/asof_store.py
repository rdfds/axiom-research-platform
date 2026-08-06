"""
As-Of Warehouse (DuckDB Wrapper)
================================
Lightweight helper to query bitemporal warehouse parquet files with
as-of filtering.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_WAREHOUSE_DIR = Path(os.environ.get("AXIOM_WAREHOUSE_DIR") or (DATA_DIR / "warehouse"))
_FINANCIAL_QUERY_YEARS_BACK = max(2, int(os.environ.get("AXIOM_WAREHOUSE_FINANCIAL_YEARS_BACK", "8")))

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None

try:
    import pyarrow.dataset as ds  # type: ignore
except Exception:  # pragma: no cover
    ds = None


class AsOfWarehouse:
    def __init__(self, warehouse_dir: Optional[Path] = None):
        self.warehouse_dir = Path(warehouse_dir) if warehouse_dir else DEFAULT_WAREHOUSE_DIR
        self.warehouse_dir.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(database=":memory:") if duckdb else None
        self._cik_gvkey_map: Optional[dict] = None

    def table_path(self, table_name: str) -> Path:
        # Prefer directory datasets when available
        dir_path = self.warehouse_dir / table_name
        if dir_path.exists() and dir_path.is_dir():
            return dir_path
        if not table_name.endswith(".parquet"):
            table_name = f"{table_name}.parquet"
        return self.warehouse_dir / table_name

    def query(
        self,
        table_name: str,
        as_of: Optional[datetime] = None,
        columns: Optional[Iterable[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        prefer_gvkey: bool = False,
    ) -> pd.DataFrame:
        path = self.table_path(table_name)
        if not path.exists():
            return pd.DataFrame()
        query_target = self._query_target(table_name, path, as_of)

        if duckdb:
            cols = "*" if columns is None else ", ".join(columns)
            clauses: List[str] = []
            params: List[object] = []
            if as_of is not None:
                clauses.append("available_time <= ?")
                params.append(as_of)
            if where:
                clauses.append(f"({where})")
            query = f"SELECT {cols} FROM read_parquet({query_target}, union_by_name=True)"
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            if limit:
                query += f" LIMIT {int(limit)}"
            df = self._conn.execute(query, params).df()
            if prefer_gvkey and "company_id" in df.columns:
                df = self.annotate_company_ids(df)
            return df

        if path.is_dir() and ds is not None:
            dataset_paths = self._query_paths(table_name, path, as_of)
            dataset = ds.dataset([item.as_posix() for item in dataset_paths], format="parquet")
            filt = None
            if as_of is not None:
                filt = (ds.field("available_time") <= as_of)
            if where:
                # pyarrow.dataset cannot parse SQL; fallback to pandas if custom where
                table = dataset.to_table(filter=filt)
                df = table.to_pandas()
                if where:
                    df = df.query(where)
                if limit:
                    df = df.head(limit)
                return df
            table = dataset.to_table(filter=filt)
            df = table.to_pandas()
            if limit:
                df = df.head(limit)
            return df

        df = pd.read_parquet(path, columns=list(columns) if columns else None)
        if as_of is not None:
            df["available_time"] = pd.to_datetime(df["available_time"])
            df = df[df["available_time"] <= as_of]
        if where:
            df = df.query(where)
        if limit:
            df = df.head(limit)
        if prefer_gvkey and "company_id" in df.columns:
            df = self.annotate_company_ids(df)
        return df

    def _query_paths(self, table_name: str, path: Path, as_of: Optional[datetime]) -> List[Path]:
        if not path.is_dir():
            return [path]
        if table_name != "warehouse_financials" or as_of is None:
            return [path]
        min_year = max(1900, int(as_of.year) - _FINANCIAL_QUERY_YEARS_BACK)
        max_year = int(as_of.year)
        candidates = [
            child
            for child in sorted(path.glob("year=*"))
            if child.is_dir()
            and _partition_year(child) is not None
            and min_year <= int(_partition_year(child)) <= max_year
        ]
        return candidates or [path]

    def _query_target(self, table_name: str, path: Path, as_of: Optional[datetime]) -> str:
        query_paths = self._query_paths(table_name, path, as_of)
        if len(query_paths) == 1:
            return f"'{query_paths[0].as_posix()}'"
        return "[" + ", ".join(f"'{item.as_posix()}'" for item in query_paths) + "]"


def _partition_year(path: Path) -> Optional[int]:
    name = path.name
    if not name.startswith("year="):
        return None
    try:
        return int(name.split("=", 1)[1])
    except Exception:
        return None

    def latest_by_entity(
        self,
        table_name: str,
        as_of: datetime,
        entity_col: str = "entity_id",
        order_cols: Optional[List[str]] = None,
        prefer_gvkey: bool = False,
    ) -> pd.DataFrame:
        path = self.table_path(table_name)
        if not path.exists():
            return pd.DataFrame()

        if order_cols is None:
            order_cols = ["event_time", "available_time", "ingestion_time", "version_id"]

        if duckdb:
            order_expr = ", ".join([f"{col} DESC" for col in order_cols])
            query = f"""
                SELECT * FROM (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY {entity_col}
                            ORDER BY {order_expr}
                        ) AS rn
                    FROM read_parquet('{path.as_posix()}')
                    WHERE available_time <= ?
                ) WHERE rn = 1
            """
            df = self._conn.execute(query, [as_of]).df()
            if prefer_gvkey and "company_id" in df.columns:
                df = self.annotate_company_ids(df)
            return df

        df = pd.read_parquet(path)
        df["available_time"] = pd.to_datetime(df["available_time"])
        df = df[df["available_time"] <= as_of]
        df = df.sort_values(order_cols, ascending=[False] * len(order_cols))
        df = df.groupby(entity_col).head(1).reset_index(drop=True)
        if prefer_gvkey and "company_id" in df.columns:
            df = self.annotate_company_ids(df)
        return df

    @staticmethod
    def _annotate_company_id(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds:
          - company_id_type: gvkey | cik | symbol | unknown
          - company_id_canonical: gvkey if available, else company_id
        """
        out = df.copy()
        out["company_id"] = out["company_id"].astype("string")
        numeric = out["company_id"].str.fullmatch(r"[0-9]+")
        # GVKEYs are typically <= 6 digits; CIKs are often 8-10 digits.
        gvkey_mask = numeric & (out["company_id"].str.len() <= 6)
        cik_mask = numeric & (out["company_id"].str.len() > 6)

        out["company_id_type"] = "symbol"
        out.loc[gvkey_mask, "company_id_type"] = "gvkey"
        out.loc[cik_mask, "company_id_type"] = "cik"
        out.loc[out["company_id"].isna(), "company_id_type"] = "unknown"

        out["company_id_canonical"] = out["company_id"]
        out.loc[gvkey_mask, "company_id_canonical"] = out.loc[gvkey_mask, "company_id"]
        return out

    def _load_cik_gvkey_map(self) -> dict:
        if self._cik_gvkey_map is not None:
            return self._cik_gvkey_map
        path = DATA_DIR / "wrds" / "compustat" / "cik_gvkey.csv.gz"
        if not path.exists():
            self._cik_gvkey_map = {}
            return self._cik_gvkey_map
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.lower() for c in df.columns]
        if "cik" not in df.columns or "gvkey" not in df.columns:
            self._cik_gvkey_map = {}
            return self._cik_gvkey_map
        df["cik"] = df["cik"].astype(str).str.replace(r"^0+", "", regex=True)
        df["gvkey"] = df["gvkey"].astype(str).str.strip()
        self._cik_gvkey_map = dict(zip(df["cik"], df["gvkey"]))
        return self._cik_gvkey_map

    def annotate_company_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds company_id_type + company_id_canonical, and maps CIK -> GVKEY when possible.
        """
        out = self._annotate_company_id(df)
        cik_map = self._load_cik_gvkey_map()
        if not cik_map or "company_id_type" not in out.columns:
            return out
        cik_mask = out["company_id_type"] == "cik"
        if cik_mask.any():
            mapped = out.loc[cik_mask, "company_id"].str.replace(r"^0+", "", regex=True).map(cik_map)
            out.loc[cik_mask & mapped.notna(), "company_id_canonical"] = mapped[mapped.notna()]
        return out

    def query_prices_entity(self, permno: int, as_of: datetime) -> pd.DataFrame:
        """
        Fast path for warehouse_prices entity lookups using DuckDB or PyArrow dataset.
        """
        def _query_path(path: Path) -> pd.DataFrame:
            if not path.exists():
                return pd.DataFrame()
            permno_str = str(permno)
            if duckdb:
                query = """
                    SELECT * FROM read_parquet(?)
                    WHERE entity_id = ?
                      AND available_time <= ?
                      AND event_time <= ?
                """
                return self._conn.execute(query, [path.as_posix(), permno_str, as_of, as_of]).df()
            if ds is not None:
                dataset = ds.dataset(path.as_posix(), format="parquet")
                filt = (
                    (ds.field("entity_id") == permno_str) &
                    (ds.field("available_time") <= as_of) &
                    (ds.field("event_time") <= as_of)
                )
                table = dataset.to_table(filter=filt)
                return table.to_pandas()
            df = pd.read_parquet(path)
            df["available_time"] = pd.to_datetime(df["available_time"])
            df["event_time"] = pd.to_datetime(df["event_time"])
            return df[(df["entity_id"] == permno_str) & (df["available_time"] <= as_of) & (df["event_time"] <= as_of)]

        # Prefer RDP daily dataset, then CRSP daily, then monthly (fallback if empty)
        rdp_daily = self.warehouse_dir / "warehouse_prices_daily_rdp"
        crsp_daily = self.warehouse_dir / "warehouse_prices_daily"
        monthly = self.table_path("warehouse_prices")

        if rdp_daily.exists():
            df = _query_path(rdp_daily)
            if not df.empty:
                return df
        if crsp_daily.exists():
            df = _query_path(crsp_daily)
            if not df.empty:
                return df
        return _query_path(monthly)
