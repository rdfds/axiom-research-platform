from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metric_methodology import MetricMethodologyRegistry


ROOT = Path(__file__).resolve().parents[1]

_FALLBACK_SECTOR_FIELD_CANDIDATES = [
    "gics_sector",
    "sector",
    "Sector",
]

_FALLBACK_SUBSECTOR_FIELD_CANDIDATES = [
    "gics_sub_industry",
    "subsector",
    "Subsector",
    "industry",
    "Industry",
]

_FALLBACK_UNSUPPORTED_FINANCIAL_METRICS = {
    "capital_structure.gross_leverage",
    "capital_structure.net_leverage",
}


def _default_policy_path() -> Path:
    env = str(os.environ.get("AXIOM_METRIC_POLICY_PATH", "") or "").strip()
    if env:
        return Path(env)
    return ROOT / "configs" / "metric_policies" / "market_metric_policy_v1.json"


def _fallback_policy_payload() -> Dict[str, Any]:
    return {
        "policy_id": "market_metric_policy_v1",
        "version": 1,
        "primary_credit_anchor": "moodys_primary_v1",
        "taxonomy": {
            "sector_field_candidates": list(_FALLBACK_SECTOR_FIELD_CANDIDATES),
            "subsector_field_candidates": list(_FALLBACK_SUBSECTOR_FIELD_CANDIDATES),
            "archetypes": {"generic_corporate": {"rules": {}}},
        },
        "metrics": {},
    }


@dataclass
class TaxonomyContext:
    company_id: str
    archetype: str
    sector: Optional[str]
    subsector: Optional[str]
    override_level_applied: str
    confidence: float
    quality_flags: List[str] = field(default_factory=list)
    support_mode: str = "exact"


class MetricPolicyEngine:
    def __init__(
        self,
        policy_path: Path | str | None = None,
        methodology_registry_path: Path | str | None = None,
    ) -> None:
        self.policy_path = Path(policy_path) if policy_path is not None else _default_policy_path()
        self.methodology_registry = MetricMethodologyRegistry(methodology_registry_path)
        self._using_fallback_policy = False
        self.policy = _fallback_policy_payload()
        try:
            if self.policy_path.exists():
                self.policy = json.loads(self.policy_path.read_text())
        except Exception:
            # Synced placeholder files can time out on read; keep snapshot builds alive.
            self._using_fallback_policy = True
            self.policy = _fallback_policy_payload()

    @property
    def policy_id(self) -> str:
        return str(self.policy.get("policy_id") or "market_metric_policy_v1")

    def sector_field_candidates(self) -> List[str]:
        candidates = list((self.policy.get("taxonomy") or {}).get("sector_field_candidates") or [])
        if candidates:
            return candidates
        if self._using_fallback_policy:
            return list(_FALLBACK_SECTOR_FIELD_CANDIDATES)
        return []

    def subsector_field_candidates(self) -> List[str]:
        candidates = list((self.policy.get("taxonomy") or {}).get("subsector_field_candidates") or [])
        if candidates:
            return candidates
        if self._using_fallback_policy:
            return list(_FALLBACK_SUBSECTOR_FIELD_CANDIDATES)
        return []

    def archetypes(self) -> Dict[str, Dict[str, Any]]:
        return dict((self.policy.get("taxonomy") or {}).get("archetypes") or {})

    def issuer_overrides(self) -> Dict[str, Dict[str, Any]]:
        return dict((self.policy.get("taxonomy") or {}).get("issuer_overrides") or {})

    def metric_definition(self, metric_id: str) -> Dict[str, Any]:
        return dict((self.policy.get("metrics") or {}).get(metric_id) or {})

    def metric_methodology(self, metric_id: str) -> Dict[str, Any]:
        return self.methodology_registry.metric(metric_id)

    def archetype_rules(self, archetype: str) -> Dict[str, Any]:
        archetype_defs = self.archetypes()
        selected = archetype_defs.get(archetype) or archetype_defs.get("generic_corporate") or {}
        return dict(selected.get("rules") or {})

    def resolve_applicability(self, metric_id: str, taxonomy: TaxonomyContext) -> str:
        definition = self.metric_definition(metric_id)
        if not definition:
            if self._using_fallback_policy:
                if taxonomy.archetype == "financial_institution" and metric_id in _FALLBACK_UNSUPPORTED_FINANCIAL_METRICS:
                    return "unsupported"
                return "primary"
            return "unsupported"
        status = str(definition.get("default_applicability") or "unsupported")
        overrides = dict(definition.get("archetype_applicability") or {})
        if taxonomy.archetype in overrides:
            status = str(overrides[taxonomy.archetype])
        return status

    def metric_metadata(
        self,
        metric_id: str,
        taxonomy: TaxonomyContext,
        *,
        view_type: str,
        support_mode: Optional[str] = None,
        component_breakdown: Optional[Dict[str, Any]] = None,
        quality_flags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        definition = self.metric_definition(metric_id)
        methodology = self.metric_methodology(metric_id)
        canonical_owners = self.methodology_registry.canonical_owners()
        canonical_owner_id = methodology.get("canonical_owner_id")
        canonical_owner = dict(canonical_owners.get(canonical_owner_id) or {})
        applicability_status = self.resolve_applicability(metric_id, taxonomy)
        resolved_support_mode = support_mode or taxonomy.support_mode or "exact"
        if applicability_status == "unsupported" and view_type != "reported":
            resolved_support_mode = "unsupported"
        return {
            "metric_policy_id": self.policy_id,
            "market_owner": definition.get("market_owner"),
            "primary_source_basis": definition.get("primary_source_basis") or self.policy.get("primary_credit_anchor"),
            "archetype": taxonomy.archetype,
            "sector": taxonomy.sector,
            "subsector": taxonomy.subsector,
            "override_level_applied": taxonomy.override_level_applied,
            "support_mode": resolved_support_mode,
            "applicability_status": applicability_status,
            "component_breakdown": component_breakdown or {},
            "quality_flags": list(dict.fromkeys((taxonomy.quality_flags or []) + (quality_flags or []))),
            "view_type": view_type,
            "methodology_registry_id": self.methodology_registry.registry_id,
            "methodology_metric_id": metric_id,
            "canonical_owner_id": canonical_owner_id,
            "canonical_owner_name": canonical_owner.get("name"),
            "canonical_classification": methodology.get("canonical_classification"),
            "market_layer_status": methodology.get("market_layer_status"),
            "current_alignment_status": methodology.get("current_alignment_status"),
            "primary_source_document_id": methodology.get("primary_source_document_id"),
            "recommended_metric_name": methodology.get("recommended_metric_name"),
        }

    def resolve_taxonomy(
        self,
        company_id: str,
        *,
        entity_row: Optional[Dict[str, Any]] = None,
        fingerprints: Optional[Dict[str, Any]] = None,
    ) -> TaxonomyContext:
        entity_row = entity_row or {}
        fingerprints = fingerprints or {}
        cid = str(company_id)
        quality_flags: List[str] = []

        issuer_override = self.issuer_overrides().get(cid)
        if issuer_override:
            return TaxonomyContext(
                company_id=cid,
                archetype=str(issuer_override.get("archetype") or "generic_corporate"),
                sector=_clean_text(issuer_override.get("sector")),
                subsector=_clean_text(issuer_override.get("subsector")),
                override_level_applied="issuer",
                confidence=0.95,
                quality_flags=[],
                support_mode="exact",
            )

        sector = self._first_entity_value(entity_row, self.sector_field_candidates())
        subsector = self._first_entity_value(entity_row, self.subsector_field_candidates())
        sic_text = _digits_only(entity_row.get("sic"))
        naics_text = _digits_only(entity_row.get("naics"))
        taxonomy_text = self._taxonomy_text(entity_row, sector, subsector)

        chosen_archetype = "generic_corporate"
        override_level = "base"
        confidence = 0.6
        support_mode = "exact"
        best_score = 0.0

        for archetype, config in self.archetypes().items():
            if archetype == "generic_corporate":
                continue
            score, match_level = self._match_taxonomy_rule(
                config,
                sector_text=(sector or "").lower(),
                subsector_text=(subsector or "").lower(),
                taxonomy_text=taxonomy_text,
                sic_text=sic_text,
                naics_text=naics_text,
            )
            if score > best_score:
                best_score = score
                chosen_archetype = archetype
                override_level = match_level
                if match_level == "subsector":
                    confidence = 0.84
                    support_mode = "exact"
                elif match_level == "sector":
                    confidence = 0.76
                    support_mode = "exact"
                else:
                    confidence = 0.68
                    support_mode = "proxy"

        if chosen_archetype == "generic_corporate":
            if self._using_fallback_policy and _looks_like_financial_institution(
                sector=sector,
                subsector=subsector,
                taxonomy_text=taxonomy_text,
                sic_text=sic_text,
                naics_text=naics_text,
            ):
                chosen_archetype = "financial_institution"
                override_level = "fallback_sector" if sector or subsector else "fallback_proxy"
                confidence = 0.72 if sector or subsector else 0.6
                support_mode = "exact" if sector or subsector else "proxy"
            else:
                lease_ratio = _safe_float(fingerprints.get("lease_liability_to_reported_debt"))
                lease_rule = (((self.archetypes().get("lease_heavy") or {}).get("fingerprints") or {}).get("lease_liability_to_reported_debt_min"))
                if lease_ratio is not None and lease_rule is not None and lease_ratio >= float(lease_rule):
                    chosen_archetype = "lease_heavy"
                    override_level = "fingerprint"
                    confidence = 0.78
                    support_mode = "proxy"
                elif not taxonomy_text:
                    quality_flags.append("taxonomy_missing")
                    support_mode = "inferred"
                    confidence = 0.5

        if chosen_archetype in {"financial_institution"} and support_mode == "exact":
            support_mode = "proxy" if sector or subsector else "unsupported"
            if support_mode == "unsupported":
                quality_flags.append("sector_native_metrics_required")

        return TaxonomyContext(
            company_id=cid,
            archetype=chosen_archetype,
            sector=sector,
            subsector=subsector,
            override_level_applied=override_level,
            confidence=confidence,
            quality_flags=quality_flags,
            support_mode=support_mode,
        )

    def _first_entity_value(self, entity_row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
        for key in candidates:
            value = _clean_text(entity_row.get(key))
            if value:
                return value
        return None

    def _taxonomy_text(
        self,
        entity_row: Dict[str, Any],
        sector: Optional[str],
        subsector: Optional[str],
    ) -> str:
        fields = list(dict.fromkeys(
            self.sector_field_candidates()
            + self.subsector_field_candidates()
            + ["industry", "subindustry", "business_description", "company_description"]
        ))
        parts: List[str] = []
        for value in [sector, subsector]:
            text = _clean_text(value)
            if text and text.lower() not in parts:
                parts.append(text.lower())
        for field in fields:
            text = _clean_text(entity_row.get(field))
            if text:
                lowered = text.lower()
                if lowered not in parts:
                    parts.append(lowered)
        return " ".join(parts).strip()

    def _match_taxonomy_rule(
        self,
        config: Dict[str, Any],
        *,
        sector_text: str,
        subsector_text: str,
        taxonomy_text: str,
        sic_text: str,
        naics_text: str,
    ) -> tuple[float, str]:
        best_score = 0.0
        best_level = "base"
        for pattern in config.get("subsector_patterns") or []:
            pattern_text = str(pattern).lower()
            if pattern_text and subsector_text and pattern_text in subsector_text:
                score = 8.0 + min(len(pattern_text) / 100.0, 0.5)
                if score > best_score:
                    best_score = score
                    best_level = "subsector"
            elif pattern_text and pattern_text in taxonomy_text:
                score = 4.0 + min(len(pattern_text) / 100.0, 0.5)
                if score > best_score:
                    best_score = score
                    best_level = "subsector"
        for pattern in config.get("sector_patterns") or []:
            pattern_text = str(pattern).lower()
            if pattern_text and sector_text and pattern_text in sector_text:
                score = 4.0 + min(len(pattern_text) / 100.0, 0.5)
                if score > best_score:
                    best_score = score
                    best_level = "sector"
            elif pattern_text and pattern_text in taxonomy_text:
                score = 2.0 + min(len(pattern_text) / 100.0, 0.5)
                if score > best_score:
                    best_score = score
                    best_level = "sector"
        for prefix in config.get("sic_prefixes") or []:
            prefix = str(prefix)
            if (sic_text and sic_text.startswith(prefix)) or (naics_text and naics_text.startswith(prefix)):
                score = 1.0 + min(len(prefix) / 10.0, 0.5)
                if score > best_score:
                    best_score = score
                    best_level = "sic"
        return best_score, best_level


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _looks_like_financial_institution(
    *,
    sector: Optional[str],
    subsector: Optional[str],
    taxonomy_text: str,
    sic_text: str,
    naics_text: str,
) -> bool:
    combined = " ".join(part for part in [sector, subsector, taxonomy_text] if part).lower()
    if any(token in combined for token in ["financial", "bank", "insurance", "capital markets", "asset management", "consumer finance", "reit"]):
        return True
    return sic_text.startswith(("60", "61", "62", "63", "64", "65", "67")) or naics_text.startswith(("52", "53"))


def _digits_only(value: Any) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in str(value) if ch.isdigit())


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None
