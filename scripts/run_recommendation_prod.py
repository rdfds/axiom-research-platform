#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_path(*parts: str) -> str:
    return str(_REPO_ROOT.joinpath(*parts))


