from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
import zipfile


def normalize_cik(value: str) -> str:
    return str(value).strip().zfill(10)


