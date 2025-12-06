#!/usr/bin/env python3
"""Expand the partial WRDS CDS RED-code map with safe alias rules."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ABBREVIATIONS = {
    "AMERN": "AMERICAN",
    "AIRLS": "AIRLINES",
    "AWYS": "AIRWAYS",
    "BK": "BANK",
    "BKS": "BANKS",
    "FINL": "FINANCIAL",
    "FIN": "FINANCIAL",
    "HLDGS": "HOLDINGS",
    "INTL": "INTERNATIONAL",
    "TECH": "TECHNOLOGY",
    "TECHS": "TECHNOLOGIES",
    "COMMN": "COMMUNICATIONS",
    "COMMS": "COMMUNICATIONS",
    "SVCS": "SERVICES",
    "SVS": "SERVICES",
    "MFG": "MANUFACTURING",
    "WHSL": "WHOLESALE",
    "PRODS": "PRODUCTS",
    "MTG": "MORTGAGE",
    "PWR": "POWER",
    "GEN": "GENERAL",
    "INDS": "INDUSTRIES",
    "INDL": "INDUSTRIAL",
    "GRP": "GROUP",
    "COS": "COMPANIES",
    "CTRL": "CONTROL",
    "ELEC": "ELECTRIC",
    "ELECN": "ELECTRONICS",
    "ELECS": "ELECTRONICS",
    "AMER": "AMERICAN",
}

STOPWORDS = {
    "THE",
    "CORPORATION",
    "CORP",
    "INCORPORATED",
    "INC",
    "COMPANY",
    "CO",
    "GROUP",
    "HOLDINGS",
    "HOLDING",
    "PLC",
    "LTD",
    "LIMITED",
    "NV",
    "N",
    "V",
    "SA",
    "SPA",
    "AG",
    "SE",
    "LLC",
    "LP",
    "NEW",
    "CLASS",
    "A",
    "B",
    "DE",
    "US",
    "COMPANIES",
}

ALLOWED_EXTRA_TOKENS = {
    "FOODS",
    "HOMES",
    "HOLDCO",
    "MOTORS",
    "ENERGY",
    "PETROLEUM",
    "COMMUNICATIONS",
    "HEALTHCARE",
    "TECHNOLOGY",
    "TECHNOLOGIES",
    "PROPERTIES",
    "BRANDS",
    "FINANCIAL",
    "INDUSTRIES",
    "INDUSTRIAL",
    "AIRWAYS",
    "AIRLINES",
    "PRODUCTS",
    "SYSTEMS",
    "SERVICES",
    "GROUP",
    "FOODSVC",
}


def _company_id_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(10)


