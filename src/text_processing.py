"""
Text Processing Utilities (MVP)
===============================
Chunking + heuristic signal extraction for unstructured sources.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"


def ensure_list(value: Optional[object]) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    items: List[object]
    if isinstance(value, list):
        items = value
    elif isinstance(value, (tuple, set)):
        items = list(value)
    else:
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                items = list(value)
            else:
                items = [value]
        except Exception:
            items = [value]
    flat: List[str] = []
    for item in items:
        if item is None or (isinstance(item, float) and pd.isna(item)):
            continue
        try:
            if pd.isna(item):
                continue
        except Exception:
            pass
        if isinstance(item, list):
            for sub in item:
                if sub is None or (isinstance(sub, float) and pd.isna(sub)):
                    continue
                flat.append(str(sub))
        elif isinstance(item, (tuple, set)):
            flat.extend([str(sub) for sub in item])
        else:
            try:
                import numpy as np

                if isinstance(item, np.ndarray):
                    flat.extend([str(sub) for sub in list(item)])
                else:
                    flat.append(str(item))
            except Exception:
                flat.append(str(item))
    return flat


