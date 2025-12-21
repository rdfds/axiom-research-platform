from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "57_ingest_wrds_dealscan_loanconnector.py"
    spec = importlib.util.spec_from_file_location("ingest_wrds_dealscan_loanconnector", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_active_artifact_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "58_build_wrds_dealscan_active_revolver_artifact.py"
    spec = importlib.util.spec_from_file_location("build_wrds_dealscan_active_revolver_artifact", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


