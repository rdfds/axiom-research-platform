"""
Base Classes for Data Plane
===========================
Defines the interfaces and canonical formats for all source adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
from enum import Enum
import pandas as pd
import hashlib
import json


class RecordType(Enum):
    """Types of canonical records."""
    FUNDAMENTAL = "fundamental"
    PRICE = "price"
    CORPORATE_ACTION = "corporate_action"
    MA_DEAL = "ma_deal"
    ESTIMATE = "estimate"
    INSIDER = "insider"


