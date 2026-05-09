import argparse
import faulthandler
import json
import os
import sys
import time
import zlib
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Allow trace-hang before heavy imports (set TRACE_HANG=1)
if os.environ.get("TRACE_HANG") == "1":
    faulthandler.dump_traceback_later(30, repeat=True)

import pandas as pd
import duckdb

from src.company_state_builder import CompanyStateBuilder
from src.company_state_store import SnapshotStore


def _parse_company_ids(arg: Optional[str]) :
    if not arg:
        return []
    if "," in arg:
        return [x.strip() for x in arg.split(",") if x.strip()]
    return [arg.strip()]

