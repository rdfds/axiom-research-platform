#!/usr/bin/env python
"""
Axiom V1 Demo
=============
Demonstrates the complete V1 system:
1. As-of snapshot building (no lookahead bias)
2. State profile computation (7 signals)
3. Market regime classification
4. Analog retrieval from historical M&A transactions

Run: python demo_v1.py
"""

import sys
sys.path.insert(0, '.')

from src.snapshot import AsOfSnapshotBuilder
from src.signals import SignalEngine
from src.regimes import RegimeClassifier
from src.analogs import AnalogRetriever


def print_header(text):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


