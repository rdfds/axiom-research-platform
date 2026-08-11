from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DependencyEdge:
    source_action: str
    target_action: str
    relationship_type: str
    condition: Optional[str] = None
    strength: Optional[str] = None
    explanation: Optional[str] = None
    original_rule_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionDependencyGraph:
    nodes: List[str]
    edges: List[DependencyEdge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [edge.to_dict() for edge in self.edges],
        }


