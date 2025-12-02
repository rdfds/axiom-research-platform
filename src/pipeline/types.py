from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class CompanyStateSnapshot:
    company_id: str
    as_of_time: datetime
    features: Dict[str, Any]
    regime: Dict[str, Any] = field(default_factory=dict)
    constraint_set: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_id": self.company_id,
            "as_of_time": self.as_of_time.isoformat(),
            "features": self.features,
            "regime": self.regime,
            "constraint_set": self.constraint_set,
            "provenance": self.provenance,
        }


@dataclass
class ActionCandidate:
    action_type: str
    params: Dict[str, Any]
    action_subtype: Optional[str] = None
    action_id: Optional[str] = None
    assumed_preconditions: List[str] = field(default_factory=list)
    rationale_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "action_subtype": self.action_subtype,
            "action_id": self.action_id,
            "params": self.params,
            "assumed_preconditions": self.assumed_preconditions,
            "rationale_refs": self.rationale_refs,
        }


