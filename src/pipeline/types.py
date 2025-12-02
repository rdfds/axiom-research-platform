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


