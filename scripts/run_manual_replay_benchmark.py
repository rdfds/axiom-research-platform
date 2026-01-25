#!/usr/bin/env python
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

def _bootstrap_runtime_threading_defaults() -> None:
    # Historical replay is strictly offline and occasionally trips Intel/OpenMP
    # runtime issues on this machine; keep the harness single-threaded by default.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_AFFINITY", "disabled")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")


_bootstrap_runtime_threading_defaults()

