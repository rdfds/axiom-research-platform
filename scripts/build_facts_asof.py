import argparse
import time
from pathlib import Path
from typing import List, Optional

import duckdb


def _is_readable(path: Path) -> bool:
    try:
        st = path.stat()
        if hasattr(st, "st_blocks") and st.st_blocks == 0:
            try:
                with open(path, "rb") as f:
                    f.read(4)
                return True
            except Exception:
                return False
        with open(path, "rb") as f:
            f.read(4)
        return True
    except Exception:
        return False


def parse_years(arg: Optional[str]) -> List[int]:
    if not arg:
        return []
    out = []
    for part in arg.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


