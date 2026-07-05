from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from scripts.merge_causal_model_overlays import merge_models


def _write_bundle(path: Path, payload: dict) -> None:
    with path.open("wb") as fh:
        pickle.dump(payload, fh)


