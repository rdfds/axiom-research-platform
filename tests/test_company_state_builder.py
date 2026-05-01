from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.company_state_builder as company_state_builder
from src.company_state_builder import CompanyStateBuilder


def _write_parquet(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


