"""
CompanyState components: RegimeClassifier, PeerSetResolver, ProvenanceTracker.
These are lightweight, auditable building blocks used by CompanyStateBuilder.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class InputReference:
    artifact_type: str
    artifact_id: str
    source: Optional[str] = None
    published_at: Optional[str] = None
    ingested_at: Optional[str] = None
    hash: Optional[str] = None


