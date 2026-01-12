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


class ActionType(Enum):
    """Standardized corporate action types."""
    # M&A
    ACQUISITION = "acquisition"
    MERGER = "merger"
    DIVESTITURE = "divestiture"
    SPINOFF = "spinoff"
    LBO = "lbo"
    GOING_PRIVATE = "going_private"

    # Capital Returns
    DIVIDEND_REGULAR = "dividend_regular"
    DIVIDEND_SPECIAL = "dividend_special"
    DIVIDEND_INCREASE = "dividend_increase"
    DIVIDEND_CUT = "dividend_cut"
    DIVIDEND_INITIATE = "dividend_initiate"
    DIVIDEND_SUSPEND = "dividend_suspend"
    BUYBACK = "buyback"

    # Equity
    IPO = "ipo"
    SECONDARY_OFFERING = "secondary_offering"
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"

    # Debt
    DEBT_ISSUANCE = "debt_issuance"
    DEBT_REFINANCE = "debt_refinance"

    # Other
    BANKRUPTCY = "bankruptcy"
    OTHER = "other"


