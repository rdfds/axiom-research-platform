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


