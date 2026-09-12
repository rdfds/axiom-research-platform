#!/usr/bin/env python
"""
Axiom V2 Demo
=============
The expanded system that answers:
"For companies in this state, what did they do next, and how did it turn out?"

This demo shows:
1. State profile computation (7 signals)
2. Market regime classification
3. Corporate action analysis - what similar companies did
4. Historical analog retrieval

Run: python demo_v2.py
"""

import sys
sys.path.insert(0, '.')

from src.snapshot import AsOfSnapshotBuilder
from src.signals import SignalEngine
from src.regimes import RegimeClassifier
from src.corporate_actions import CorporateActionsDB, ActionAnalyzer


def print_header(text):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


