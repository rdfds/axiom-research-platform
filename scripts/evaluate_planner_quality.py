#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.planner_eval import build_planner_eval_report, render_planner_eval_markdown


def _parse_run_ids(path: Path | None) -> List[str] | None:
    if path is None:
        return None
    run_ids: List[str] = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        run_ids.append(text.split()[-1])
    return run_ids


