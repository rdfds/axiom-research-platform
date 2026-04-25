from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


NORMALIZATION_RULES_VERSION = "v2_lossless_20260317"


def _norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if pd.isna(out):
        return None
    return out


def _size_ratio(row: Dict[str, Any]) -> Optional[float]:
    action_size = _to_float(row.get("action_size"))
    base_market_cap = _to_float(row.get("base_market_cap"))
    if action_size is not None and base_market_cap and base_market_cap > 0:
        ratio = action_size / base_market_cap
        if ratio > 0:
            return ratio

    for key in (
        "target_size_pct_ev",
        "target_size_pct_market_cap",
        "target_size_pct_mc",
        "deal_size_pct_ev",
        "transaction_size_pct_ev",
        "target_ev_pct",
        "percent_divested",
        "percent_sold",
        "stake_pct",
    ):
        ratio = _to_float(row.get(key))
        if ratio is not None and ratio > 0:
            return ratio
    return None


def _scale_bucket(row: Dict[str, Any]) -> Optional[str]:
    ratio = _size_ratio(row)
    if ratio is None:
        return None
    if ratio < 0.05:
        return "small"
    if ratio < 0.25:
        return "medium"
    return "large"


def _exact_result(
    *,
    family: str,
    subfamily: str,
    action_id: str,
    family_scale_bucket: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "normalized_action_family": family,
        "normalized_action_subfamily": subfamily,
        "normalized_action_id": action_id,
        "normalization_level": "exact",
        "normalization_confidence": 0.98,
        "family_scale_bucket": family_scale_bucket,
        "normalization_rules_version": NORMALIZATION_RULES_VERSION,
    }


def _family_result(
    *,
    family: str,
    subfamily: str,
    family_scale_bucket: Optional[str] = None,
    action_id: Optional[str] = None,
) -> Dict[str, Any]:
    level = "family_scale" if family_scale_bucket else "family"
    confidence = 0.85 if family_scale_bucket else 0.65
    return {
        "normalized_action_family": family,
        "normalized_action_subfamily": subfamily,
        "normalized_action_id": action_id,
        "normalization_level": level,
        "normalization_confidence": confidence,
        "family_scale_bucket": family_scale_bucket,
        "normalization_rules_version": NORMALIZATION_RULES_VERSION,
    }


def _mna_acquisition_action_id(action_subtype: str, scale_bucket: Optional[str]) -> str:
    if any(token in action_subtype for token in ("stake", "repurchase", "minority")):
        return "mna.tuck_in_acquisition"
    if scale_bucket == "small":
        return "mna.tuck_in_acquisition"
    if scale_bucket == "large":
        return "mna.transformational_acquisition"
    return "mna.platform_acquisition"


def _portfolio_divestiture_action_id(row: Dict[str, Any]) -> str:
    percent_divested = _to_float(row.get("percent_divested"))
    if percent_divested is None:
        percent_divested = _to_float(row.get("percent_sold"))
    action_subtype = _norm_text(row.get("action_subtype"))
    if percent_divested is not None:
        if percent_divested >= 0.95:
            return "portfolio.divestiture_full"
        if percent_divested > 0:
            return "portfolio.divestiture_partial"
    if action_subtype == "stake purchases deal":
        return "portfolio.divestiture_partial"
    if action_subtype in {
        "acquisition_lbo",
        "acquisition_tender",
        "acquisition_merger",
        "acquisition_exchange",
        "acquisition_reverse",
    }:
        return "portfolio.divestiture_full"
    return "portfolio.asset_sale"


def _refinancing_subfamily(action_subtype: Any) -> str:
    subtype = _norm_text(action_subtype)
    if not subtype:
        return "refinancing"
    if any(
        token in subtype
        for token in (
            "revolver",
            "line",
            "facility",
            "letter of credit",
            "364-day",
        )
    ):
        return "refinancing_revolver_family"
    if any(
        token in subtype
        for token in (
            "term loan",
            "delay draw",
            "bridge loan",
        )
    ):
        return "refinancing_term_loan_family"
    if any(
        token in subtype
        for token in (
            "bond",
            "note",
            "debenture",
            "fixed-rate",
        )
    ):
        return "refinancing_bond_family"
    return "refinancing"


def normalize_action_record(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_action_type = row.get("action_type")
    raw_action_subtype = row.get("action_subtype")
    action_type = _norm_text(raw_action_type)
    action_subtype = _norm_text(raw_action_subtype)
    scale_bucket = _scale_bucket(row)

    if action_type == "dividend_increase" and action_subtype == "dividend_increase":
        return _exact_result(
            family="capital_return",
            subfamily="dividend_increase",
            action_id="capital_return.dividend_increase",
        )
    if action_type == "dividend_cut" and action_subtype == "dividend_cut":
        return _exact_result(
            family="capital_return",
            subfamily="dividend_cut",
            action_id="capital_return.dividend_cut",
        )
    if action_type == "dividend_initiate" and action_subtype == "dividend_initiate":
        return _exact_result(
            family="capital_return",
            subfamily="dividend_initiate",
            action_id="capital_return.dividend_initiate",
        )
    if action_type == "dividend_special" and action_subtype in {"special", "liquidating", "irregular"}:
        return _exact_result(
            family="capital_return",
            subfamily="special_dividend",
            action_id="capital_return.special_dividend",
        )
    if action_type == "stock_split":
        return _exact_result(
            family="governance",
            subfamily="stock_split",
            action_id="governance.stock_split",
        )
    if action_type == "reverse_split":
        return _exact_result(
            family="governance",
            subfamily="reverse_split",
            action_id="governance.reverse_split",
        )
    if action_type == "acquisition" and action_subtype == "acquisition_lbo":
        return _exact_result(
            family="mna",
            subfamily="platform_lbo",
            action_id="mna.go_private_lbo",
            family_scale_bucket=scale_bucket,
        )
    if action_type == "equity_offering_public_proxy" and action_subtype == "share_issuance_proxy":
        return _exact_result(
            family="capital_structure",
            subfamily="equity_issuance",
            action_id="capital_structure.equity_issuance",
            family_scale_bucket=scale_bucket,
        )

    if action_type == "buyback":
        return _family_result(
            family="capital_return",
            subfamily="buyback",
            family_scale_bucket=scale_bucket,
            action_id="capital_return.open_market_buyback",
        )

    if action_type == "bond_issuance":
        return _family_result(
            family="capital_structure",
            subfamily="debt_bond",
            family_scale_bucket=scale_bucket,
            action_id="capital_structure.new_debt_issuance",
        )

    if action_type == "loan_issuance":
        if any(token in action_subtype for token in ("revolver", "line", "facility", "letter of credit")):
            subfamily = "revolver"
            action_id = "capital_structure.revolver_draw_or_resize"
        else:
            subfamily = "debt_loan"
            action_id = "capital_structure.new_debt_issuance"
        return _family_result(
            family="capital_structure",
            subfamily=subfamily,
            family_scale_bucket=scale_bucket,
            action_id=action_id,
        )

    if action_type == "loan_refinancing":
        return _family_result(
            family="capital_structure",
            subfamily=_refinancing_subfamily(raw_action_subtype),
            family_scale_bucket=scale_bucket,
            action_id="capital_structure.refinancing",
        )

    if action_type == "acquisition":
        if action_subtype == "disclosed dollar value deal":
            subfamily = "platform_disclosed"
        elif action_subtype == "undisclosed dollar value deal":
            subfamily = "platform_undisclosed"
        elif action_subtype == "acquisition_merger":
            subfamily = "platform_merger"
        elif action_subtype in {"stake purchases deal", "repurchases deal"}:
            subfamily = "tuck_in_incremental"
        else:
            subfamily = "acquisition_structured"
        return _family_result(
            family="mna",
            subfamily=subfamily,
            family_scale_bucket=scale_bucket,
            action_id=_mna_acquisition_action_id(action_subtype, scale_bucket),
        )

    if action_type == "divestiture":
        return _family_result(
            family="portfolio",
            subfamily="divestiture",
            family_scale_bucket=scale_bucket,
            action_id=_portfolio_divestiture_action_id(row),
        )

    if action_type == "dividend_regular":
        return _family_result(
            family="capital_return",
            subfamily="dividend_regular",
        )

    return {
        "normalized_action_family": None,
        "normalized_action_subfamily": None,
        "normalized_action_id": None,
        "normalization_level": "unknown",
        "normalization_confidence": 0.0,
        "family_scale_bucket": scale_bucket,
        "normalization_rules_version": NORMALIZATION_RULES_VERSION,
    }


def augment_action_outcomes_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        for col in (
            "raw_action_type",
            "raw_action_subtype",
            "normalized_action_family",
            "normalized_action_subfamily",
            "normalized_action_id",
            "normalization_level",
            "normalization_confidence",
            "family_scale_bucket",
            "normalization_rules_version",
        ):
            if col not in out.columns:
                out[col] = pd.Series(dtype="object")
        return out

    out = df.copy()
    out["raw_action_type"] = out.get("action_type")
    out["raw_action_subtype"] = out.get("action_subtype")
    normalized = pd.DataFrame(
        [normalize_action_record(record) for record in out.to_dict(orient="records")],
        index=out.index,
    )
    for col in normalized.columns:
        out[col] = normalized[col]
    return out
