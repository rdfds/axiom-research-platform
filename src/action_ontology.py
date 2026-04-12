"""
Action ontology registry for candidate generation, feasibility gating, and planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ALLOWED_PARAMETER_TYPES = {
    "numeric",
    "percent",
    "boolean",
    "enum",
    "date_window",
    "entity_reference",
    "segment_reference",
    "funding_mix_object",
    "range",
}

ALLOWED_CHANNEL_TYPES = {
    "value_creation",
    "risk_reduction",
    "optionality_preservation",
    "signaling",
    "capital_structure_optimization",
    "portfolio_optimization",
    "cost_efficiency",
    "growth_substitution",
}

ALLOWED_RULE_TYPES = {
    "requires_prior",
    "unlocks",
    "conflicts_with",
    "discouraged_with",
    "preferred_after",
}

ALLOWED_RULE_STRENGTH = {"hard", "soft"}

ALLOWED_EVIDENCE_CLASSES = {
    "financial_disclosure",
    "management_statement",
    "liquidity_disclosure",
    "segment_disclosure",
    "capital_policy_statement",
    "rating_disclosure",
    "market_signal",
    "peer_context_signal",
    "recent_action_history",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percent_field(required: bool = False, minimum: float = 0.0, maximum: float = 1.0) -> Dict[str, Any]:
    return {
        "type": "percent",
        "required": required,
        "min": minimum,
        "max": maximum,
    }


def _numeric_field(required: bool = False, unit: Optional[str] = None, minimum: Optional[float] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "type": "numeric",
        "required": required,
    }
    if unit is not None:
        out["unit"] = unit
    if minimum is not None:
        out["min"] = minimum
    return out


def _enum_field(values: List[str], required: bool = False) -> Dict[str, Any]:
    return {
        "type": "enum",
        "required": required,
        "values": values,
    }


def _funding_mix(required: bool = True) -> Dict[str, Any]:
    return {
        "type": "funding_mix_object",
        "required": required,
        "fields": {
            "cash": {"type": "percent"},
            "debt": {"type": "percent"},
            "equity": {"type": "percent"},
        },
    }


def _date_window(required: bool = False) -> Dict[str, Any]:
    return {
        "type": "date_window",
        "required": required,
    }


def _range_field(required: bool = False, unit: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "type": "range",
        "required": required,
    }
    if unit:
        out["unit"] = unit
    return out


def _channel(
    channel_id: str,
    channel_type: str,
    description: str,
    activation_signals: List[str],
    negative_signals: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "channel_id": channel_id,
        "channel_type": channel_type,
        "description": description,
        "activation_signals": activation_signals,
        "negative_signals": negative_signals or [],
    }


def _rule(
    rule_type: str,
    target_action_id: str,
    condition: Optional[str],
    strength: str,
    explanation: str,
) -> Dict[str, Any]:
    return {
        "rule_type": rule_type,
        "target_action_id": target_action_id,
        "condition": condition,
        "strength": strength,
        "explanation": explanation,
    }


def _lead_time(minimum_days: int, median_days: int, p90_days: int, conditional_adjustments: Optional[List[dict]] = None) -> Dict[str, Any]:
    return {
        "minimum_days": minimum_days,
        "median_days": median_days,
        "p90_days": p90_days,
        "conditional_adjustments": conditional_adjustments or [],
    }


def _complexity(
    base_complexity_score: int,
    drivers: List[str],
    organizational_burden: str,
    cross_functional_dependencies: List[str],
) -> Dict[str, Any]:
    return {
        "base_complexity_score": base_complexity_score,
        "drivers": drivers,
        "organizational_burden": organizational_burden,
        "cross_functional_dependencies": cross_functional_dependencies,
    }


def _prerequisites(
    state_conditions: List[dict],
    required_features: List[str],
    required_evidence: Optional[List[str]] = None,
    required_disclosures: Optional[List[str]] = None,
    forbidden_constraints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "state_conditions": state_conditions,
        "required_features": required_features,
        "required_evidence": required_evidence or [],
        "required_disclosures": required_disclosures or [],
        "forbidden_constraints": forbidden_constraints or [],
    }


def _evidence(
    minimum_classes_required: List[str],
    optional_supporting_classes: Optional[List[str]] = None,
    must_have_features: Optional[List[str]] = None,
    allow_heuristic_if_missing: bool = True,
) -> Dict[str, Any]:
    return {
        "minimum_classes_required": minimum_classes_required,
        "optional_supporting_classes": optional_supporting_classes or [],
        "must_have_features": must_have_features or [],
        "allow_heuristic_if_missing": allow_heuristic_if_missing,
    }


def _action(
    action_type: str,
    action_subtype: str,
    label: str,
    description: str,
    parameter_schema: Dict[str, Any],
    feasibility_prerequisites: Dict[str, Any],
    mechanism_channels: List[Dict[str, Any]],
    lead_time_prior: Dict[str, Any],
    execution_complexity_prior: Dict[str, Any],
    dependency_rules: List[Dict[str, Any]],
    minimum_evidence_requirements: Dict[str, Any],
    validation_rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "action_type": action_type,
        "action_subtype": action_subtype,
        "action_id": f"{action_type}.{action_subtype}",
        "label": label,
        "description": description,
        "parameter_schema": parameter_schema,
        "feasibility_prerequisites": feasibility_prerequisites,
        "mechanism_channels": mechanism_channels,
        "lead_time_prior": lead_time_prior,
        "execution_complexity_prior": execution_complexity_prior,
        "dependency_rules": dependency_rules,
        "minimum_evidence_requirements": minimum_evidence_requirements,
        "validation_rules": validation_rules,
    }


@dataclass
class CandidateValidationResult:
    action_id: str
    valid: bool
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ActionSchemaRegistry:
    def __init__(self, version: str, actions: List[Dict[str, Any]], last_updated_at: Optional[str] = None) -> None:
        self.version = version
        self.actions = actions
        self.last_updated_at = last_updated_at or _utc_now_iso()

        self._action_by_id: Dict[str, Dict[str, Any]] = {}
        self._action_ids_by_type: Dict[str, List[str]] = {}
        self._action_ids_by_subtype: Dict[str, List[str]] = {}
        for action in actions:
            action_id = str(action.get("action_id"))
            self._action_by_id[action_id] = action
            at = str(action.get("action_type"))
            st = str(action.get("action_subtype"))
            self._action_ids_by_type.setdefault(at, []).append(action_id)
            self._action_ids_by_subtype.setdefault(st, []).append(action_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "last_updated_at": self.last_updated_at,
            "actions": self.actions,
        }

    def write_json(self, out_path: str | Path) -> Path:
        import json

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out

    def get_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        return self._action_by_id.get(str(action_id))

    def get_actions_by_type(self, action_type: str) -> List[Dict[str, Any]]:
        ids = self._action_ids_by_type.get(str(action_type), [])
        return [self._action_by_id[i] for i in ids]

    def get_actions_by_subtype(self, action_subtype: str) -> List[Dict[str, Any]]:
        ids = self._action_ids_by_subtype.get(str(action_subtype), [])
        return [self._action_by_id[i] for i in ids]

    def generate_actions_under_type(self, action_type: str) -> List[Dict[str, Any]]:
        return self.get_actions_by_type(action_type)

    def fetch_prerequisites(self, action_id: str) -> Dict[str, Any]:
        action = self.get_action(action_id) or {}
        return action.get("feasibility_prerequisites", {})

    def fetch_mechanism_channels(self, action_id: str) -> List[Dict[str, Any]]:
        action = self.get_action(action_id) or {}
        return action.get("mechanism_channels", [])

    def fetch_dependency_rules(self, action_id: str) -> List[Dict[str, Any]]:
        action = self.get_action(action_id) or {}
        return action.get("dependency_rules", [])

    def fetch_dependency_graph_edges(self, action_id: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        source_actions = [self.get_action(action_id)] if action_id else self.actions
        for action in source_actions:
            if not action:
                continue
            src_id = str(action.get("action_id"))
            for rule in action.get("dependency_rules", []):
                out.append(
                    {
                        "source_action_id": src_id,
                        "target_action_id": rule.get("target_action_id"),
                        "rule_type": rule.get("rule_type"),
                        "condition": rule.get("condition"),
                        "strength": rule.get("strength"),
                        "explanation": rule.get("explanation"),
                    }
                )
        return out

    def fetch_planner_dependency_edges(self, action_id: Optional[str] = None) -> List[Dict[str, Any]]:
        relationship_map = {
            "requires_prior": "requires",
            "unlocks": "unlocks",
            "conflicts_with": "conflicts",
            "discouraged_with": "conflicts",
            "preferred_after": "recommended_after",
        }
        out: List[Dict[str, Any]] = []
        for edge in self.fetch_dependency_graph_edges(action_id):
            original_rule_type = str(edge.get("rule_type") or "")
            out.append(
                {
                    "source_action": edge.get("source_action_id"),
                    "target_action": edge.get("target_action_id"),
                    "relationship_type": relationship_map.get(original_rule_type, original_rule_type),
                    "condition": edge.get("condition"),
                    "strength": edge.get("strength"),
                    "explanation": edge.get("explanation"),
                    "original_rule_type": original_rule_type,
                }
            )
        return out

    def fetch_planner_lead_time_distribution(self, action_id: str) -> Dict[str, Any]:
        action = self.get_action(action_id) or {}
        lead = action.get("lead_time_prior", {}) or {}
        minimum_days = int(lead.get("minimum_days", 0) or 0)
        median_days = int(lead.get("median_days", minimum_days) or minimum_days)
        p90_days = int(lead.get("p90_days", median_days) or median_days)

        # Planner needs p25/p75/mean-style timing priors, but ontology currently
        # stores minimum/median/p90. Use a deterministic interpolation so plan
        # scheduling has stable defaults before historical transition priors land.
        p25_days = int(round((minimum_days + median_days) / 2.0))
        p75_days = int(round((median_days + p90_days) / 2.0))
        mean_days = round((minimum_days + (2.0 * median_days) + p90_days) / 4.0, 3)

        return {
            "action_id": action_id,
            "minimum_days": minimum_days,
            "mean_days": mean_days,
            "median_days": median_days,
            "p25_days": p25_days,
            "p75_days": p75_days,
            "p90_days": p90_days,
            "conditional_adjustments": list(lead.get("conditional_adjustments", []) or []),
            "source": "schema_prior_interpolated",
        }

    def validate_schema(self) -> List[str]:
        errors: List[str] = []
        required_top_level = {
            "action_type",
            "action_subtype",
            "action_id",
            "label",
            "description",
            "parameter_schema",
            "feasibility_prerequisites",
            "mechanism_channels",
            "lead_time_prior",
            "execution_complexity_prior",
            "dependency_rules",
            "minimum_evidence_requirements",
            "validation_rules",
        }

        seen_action_ids: set[str] = set()
        for action in self.actions:
            aid = str(action.get("action_id"))
            missing = sorted(required_top_level - set(action.keys()))
            if missing:
                errors.append(f"{aid}:missing_top_level:{','.join(missing)}")

            if aid in seen_action_ids:
                errors.append(f"{aid}:duplicate_action_id")
            seen_action_ids.add(aid)

            at = str(action.get("action_type"))
            st = str(action.get("action_subtype"))
            if aid != f"{at}.{st}":
                errors.append(f"{aid}:action_id_mismatch")

            params = action.get("parameter_schema", {})
            if not isinstance(params, dict) or not params:
                errors.append(f"{aid}:invalid_parameter_schema")
            else:
                for pname, pdef in params.items():
                    ptype = pdef.get("type")
                    if ptype not in ALLOWED_PARAMETER_TYPES:
                        errors.append(f"{aid}:invalid_parameter_type:{pname}:{ptype}")
                    if "required" in pdef and not isinstance(pdef.get("required"), bool):
                        errors.append(f"{aid}:param_required_not_bool:{pname}")
                    if ptype == "enum":
                        vals = pdef.get("values")
                        if not isinstance(vals, list) or not vals:
                            errors.append(f"{aid}:enum_missing_values:{pname}")
                    if ptype in {"percent", "numeric"} and "min" in pdef and "max" in pdef:
                        if float(pdef["min"]) > float(pdef["max"]):
                            errors.append(f"{aid}:invalid_param_range:{pname}")
                    if ptype == "funding_mix_object":
                        fields = pdef.get("fields")
                        if not isinstance(fields, dict) or not fields:
                            errors.append(f"{aid}:funding_mix_fields_missing:{pname}")

            prereq = action.get("feasibility_prerequisites", {})
            if not isinstance(prereq, dict):
                errors.append(f"{aid}:invalid_feasibility_prerequisites")
            else:
                required_prereq_keys = {
                    "state_conditions",
                    "required_features",
                    "required_evidence",
                    "required_disclosures",
                    "forbidden_constraints",
                }
                missing_pr = sorted(required_prereq_keys - set(prereq.keys()))
                if missing_pr:
                    errors.append(f"{aid}:missing_prereq_fields:{','.join(missing_pr)}")

            channels = action.get("mechanism_channels", [])
            if not isinstance(channels, list) or not channels:
                errors.append(f"{aid}:missing_mechanism_channels")
            else:
                for idx, channel in enumerate(channels):
                    for key in ("channel_id", "channel_type", "description", "activation_signals", "negative_signals"):
                        if key not in channel:
                            errors.append(f"{aid}:channel_{idx}_missing:{key}")
                    ctype = channel.get("channel_type")
                    if ctype not in ALLOWED_CHANNEL_TYPES:
                        errors.append(f"{aid}:invalid_channel_type:{ctype}")

            lead = action.get("lead_time_prior", {})
            if not isinstance(lead, dict):
                errors.append(f"{aid}:invalid_lead_time_prior")
            else:
                for key in ("minimum_days", "median_days", "p90_days"):
                    if key not in lead:
                        errors.append(f"{aid}:lead_time_missing:{key}")
                if all(k in lead for k in ("minimum_days", "median_days", "p90_days")):
                    if not (int(lead["minimum_days"]) <= int(lead["median_days"]) <= int(lead["p90_days"])):
                        errors.append(f"{aid}:lead_time_order_invalid")

            complexity = action.get("execution_complexity_prior", {})
            if not isinstance(complexity, dict):
                errors.append(f"{aid}:invalid_complexity_prior")
            else:
                score = complexity.get("base_complexity_score")
                if score is None or int(score) < 1 or int(score) > 5:
                    errors.append(f"{aid}:invalid_complexity_score")

            deps = action.get("dependency_rules", [])
            if not isinstance(deps, list):
                errors.append(f"{aid}:invalid_dependency_rules")
            else:
                for idx, rule in enumerate(deps):
                    rt = rule.get("rule_type")
                    tgt = rule.get("target_action_id")
                    strength = rule.get("strength")
                    if rt not in ALLOWED_RULE_TYPES:
                        errors.append(f"{aid}:invalid_rule_type:{idx}:{rt}")
                    if not tgt:
                        errors.append(f"{aid}:missing_dependency_target:{idx}")
                    if strength not in ALLOWED_RULE_STRENGTH:
                        errors.append(f"{aid}:invalid_dependency_strength:{idx}:{strength}")

            ev = action.get("minimum_evidence_requirements", {})
            if not isinstance(ev, dict):
                errors.append(f"{aid}:invalid_evidence_requirements")
            else:
                classes = ev.get("minimum_classes_required", [])
                if not isinstance(classes, list) or not classes:
                    errors.append(f"{aid}:missing_minimum_classes_required")
                else:
                    for c in classes:
                        if c not in ALLOWED_EVIDENCE_CLASSES:
                            errors.append(f"{aid}:invalid_evidence_class:{c}")
                for c in ev.get("optional_supporting_classes", []):
                    if c not in ALLOWED_EVIDENCE_CLASSES:
                        errors.append(f"{aid}:invalid_optional_evidence_class:{c}")

            rules = action.get("validation_rules", [])
            if not isinstance(rules, list):
                errors.append(f"{aid}:validation_rules_not_list")

        for action in self.actions:
            aid = str(action.get("action_id"))
            for rule in action.get("dependency_rules", []):
                tgt = str(rule.get("target_action_id"))
                if tgt not in seen_action_ids:
                    errors.append(f"{aid}:dependency_target_missing:{tgt}")

        return errors

    def validate_registry_integrity(self) -> List[str]:
        errors: List[str] = []
        for action in self.actions:
            aid = str(action.get("action_id"))
            if not action.get("mechanism_channels"):
                errors.append(f"{aid}:missing_mechanism_channels")

            prereq = action.get("feasibility_prerequisites", {})
            has_prereq = False
            if isinstance(prereq, dict):
                for key in ("state_conditions", "required_features", "required_evidence", "required_disclosures", "forbidden_constraints"):
                    v = prereq.get(key, [])
                    if isinstance(v, list) and len(v) > 0:
                        has_prereq = True
                        break
            if not has_prereq:
                errors.append(f"{aid}:missing_prerequisites")

            if not action.get("lead_time_prior"):
                errors.append(f"{aid}:missing_lead_time_prior")

            if not action.get("execution_complexity_prior"):
                errors.append(f"{aid}:missing_execution_complexity_prior")

            if not action.get("minimum_evidence_requirements"):
                errors.append(f"{aid}:missing_minimum_evidence_requirements")
        return errors

    def validate_candidate(
        self,
        candidate: Dict[str, Any],
        strict_evidence: bool = False,
    ) -> CandidateValidationResult:
        action_id = str(candidate.get("action_id", ""))
        schema = self.get_action(action_id)
        errors: List[str] = []
        warnings: List[str] = []

        if schema is None:
            return CandidateValidationResult(
                action_id=action_id,
                valid=False,
                errors=[f"unknown_action_id:{action_id}"],
                warnings=[],
            )

        params = candidate.get("parameters")
        if params is None:
            params = candidate.get("params", {})
        if not isinstance(params, dict):
            errors.append("parameters_not_object")
            params = {}

        known_segments = set(str(x) for x in candidate.get("known_segments", []) if x is not None)
        available_features = set(str(x) for x in candidate.get("available_features", []) if x is not None)
        available_evidence = set(str(x) for x in candidate.get("available_evidence_classes", []) if x is not None)
        constraints = set(str(x) for x in candidate.get("constraints", []) if x is not None)

        for pname, pdef in schema.get("parameter_schema", {}).items():
            required = bool(pdef.get("required", False))
            if required and pname not in params:
                errors.append(f"missing_required_param:{pname}")
                continue
            if pname not in params:
                continue
            value = params.get(pname)
            ptype = pdef.get("type")
            errors.extend(self._validate_param_type(pname, ptype, value, pdef, known_segments))

        prereq = schema.get("feasibility_prerequisites", {})
        for forbidden in prereq.get("forbidden_constraints", []):
            if forbidden in constraints:
                errors.append(f"forbidden_constraint_present:{forbidden}")

        required_features = prereq.get("required_features", [])
        missing_features = sorted([f for f in required_features if f not in available_features])
        if missing_features:
            errors.append("missing_required_features:" + ",".join(missing_features))

        evidence_req = schema.get("minimum_evidence_requirements", {})
        required_classes = set(evidence_req.get("minimum_classes_required", []))
        missing_classes = sorted([c for c in required_classes if c not in available_evidence])
        if missing_classes:
            allow_heur = bool(evidence_req.get("allow_heuristic_if_missing", False))
            tag = "missing_minimum_evidence_classes:" + ",".join(missing_classes)
            if allow_heur and not strict_evidence:
                warnings.append(tag)
            else:
                errors.append(tag)

        for vrule in schema.get("validation_rules", []):
            errors.extend(self._apply_validation_rule(vrule, params, known_segments))

        return CandidateValidationResult(
            action_id=action_id,
            valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
        )

    def _validate_param_type(
        self,
        name: str,
        ptype: str,
        value: Any,
        pdef: Dict[str, Any],
        known_segments: set[str],
    ) -> List[str]:
        errors: List[str] = []
        if ptype in {"numeric", "percent"}:
            if not isinstance(value, (int, float)):
                errors.append(f"param_not_numeric:{name}")
                return errors
            if "min" in pdef and float(value) < float(pdef["min"]):
                errors.append(f"param_below_min:{name}")
            if "max" in pdef and float(value) > float(pdef["max"]):
                errors.append(f"param_above_max:{name}")
        elif ptype == "boolean":
            if not isinstance(value, bool):
                errors.append(f"param_not_boolean:{name}")
        elif ptype == "enum":
            vals = pdef.get("values", [])
            if value not in vals:
                errors.append(f"param_not_in_enum:{name}")
        elif ptype == "date_window":
            if not isinstance(value, dict):
                errors.append(f"param_not_date_window:{name}")
            else:
                if "start" not in value or "end" not in value:
                    errors.append(f"param_date_window_missing_bounds:{name}")
        elif ptype == "entity_reference":
            if not isinstance(value, str) or not value.strip():
                errors.append(f"param_not_entity_reference:{name}")
        elif ptype == "segment_reference":
            if not isinstance(value, str) or not value.strip():
                errors.append(f"param_not_segment_reference:{name}")
            elif known_segments and value not in known_segments:
                errors.append(f"param_unknown_segment_reference:{name}:{value}")
        elif ptype == "funding_mix_object":
            if not isinstance(value, dict):
                errors.append(f"param_not_funding_mix_object:{name}")
            else:
                for key in ("cash", "debt", "equity"):
                    if key not in value:
                        errors.append(f"param_funding_mix_missing:{name}:{key}")
                        continue
                    if not isinstance(value[key], (int, float)):
                        errors.append(f"param_funding_mix_not_percent:{name}:{key}")
        elif ptype == "range":
            if not isinstance(value, dict):
                errors.append(f"param_not_range:{name}")
            else:
                if "min" not in value or "max" not in value:
                    errors.append(f"param_range_missing_bounds:{name}")
                else:
                    try:
                        if float(value["min"]) > float(value["max"]):
                            errors.append(f"param_range_invalid:{name}")
                    except Exception:
                        errors.append(f"param_range_not_numeric:{name}")
        return errors

    def _apply_validation_rule(self, vrule: Dict[str, Any], params: Dict[str, Any], known_segments: set[str]) -> List[str]:
        errors: List[str] = []
        kind = vrule.get("kind")
        field = str(vrule.get("field", ""))

        if kind == "funding_mix_sum_to_one":
            mix = params.get(field)
            if isinstance(mix, dict):
                try:
                    total = float(mix.get("cash", 0.0)) + float(mix.get("debt", 0.0)) + float(mix.get("equity", 0.0))
                    if abs(total - 1.0) > 1e-6:
                        errors.append(f"funding_mix_sum_not_one:{field}")
                except Exception:
                    errors.append(f"funding_mix_invalid:{field}")
        elif kind == "positive":
            if field in params:
                try:
                    if float(params[field]) <= 0:
                        errors.append(f"param_not_positive:{field}")
                except Exception:
                    errors.append(f"param_not_numeric_for_positive_rule:{field}")
        elif kind == "non_positive":
            if field in params:
                try:
                    if float(params[field]) > 0:
                        errors.append(f"param_positive_when_non_positive_required:{field}")
                except Exception:
                    errors.append(f"param_not_numeric_for_non_positive_rule:{field}")
        elif kind == "segment_reference_exists":
            seg = params.get(field)
            if isinstance(seg, str) and known_segments and seg not in known_segments:
                errors.append(f"segment_reference_not_found:{field}:{seg}")

        return errors


def _default_actions() -> List[Dict[str, Any]]:
    common_finance_features = [
        "liquidity.available_for_actions",
        "capital_structure.net_leverage",
        "market.market_cap",
    ]
    actions: List[Dict[str, Any]] = []

    # 4.2.1 Capital Return
    actions.append(
        _action(
            action_type="capital_return",
            action_subtype="open_market_buyback",
            label="Open-Market Share Buyback",
            description="Repurchase common shares over time in open market under authorization and liquidity constraints.",
            parameter_schema={
                "size_pct_market_cap": _percent_field(required=True, minimum=0.0, maximum=0.5),
                "size_absolute_usd": _numeric_field(required=False, unit="USD", minimum=0.0),
                "funding_mix": _funding_mix(required=True),
                "execution_window": _date_window(required=False),
                "pace": _enum_field(["gradual", "front_loaded"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "liquidity.available_for_actions", "operator": ">", "value": 0},
                    {"feature": "capital_structure.net_leverage", "operator": "<=", "value": 4.0},
                ],
                required_features=common_finance_features,
                required_evidence=["strategic.intent_vector", "strategic.constraint_set"],
                forbidden_constraints=["no_capital_return"],
            ),
            mechanism_channels=[
                _channel(
                    "undervaluation_arbitrage",
                    "value_creation",
                    "Retiring shares below intrinsic value can increase per-share value.",
                    [
                        "market.ev_ebitda_vs_peer_z < 0",
                        "market.fcf_yield_percentile_peers > 70",
                    ],
                    [
                        "capital_structure.maturity_wall_ratio_24m > 0.25",
                        "regime.credit_regime == tight",
                    ],
                ),
                _channel(
                    "capital_structure_mix",
                    "capital_structure_optimization",
                    "Buyback can optimize excess equity capital when leverage is conservative.",
                    ["capital_structure.net_leverage <= target_band_high"],
                ),
                _channel(
                    "confidence_signal",
                    "signaling",
                    "Signal confidence in medium-term cash generation.",
                    ["operating.fcf_conversion > 0.4"],
                ),
            ],
            lead_time_prior=_lead_time(1, 30, 90),
            execution_complexity_prior=_complexity(
                2,
                ["requires_board_approval", "requires_treasury_execution", "requires_window_check"],
                "low_to_medium",
                ["CFO", "Treasury", "Board", "IR"],
            ),
            dependency_rules=[
                _rule(
                    "conflicts_with",
                    "mna.transformational_acquisition",
                    "capital_structure.net_leverage > 3.5",
                    "hard",
                    "Transformational M&A and buyback compete for balance-sheet capacity.",
                ),
                _rule(
                    "preferred_after",
                    "capital_structure.refinancing",
                    "capital_structure.maturity_wall_ratio_24m > 0.25",
                    "soft",
                    "Refinance first when near-term debt pressure is elevated.",
                ),
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "market_signal"],
                optional_supporting_classes=["capital_policy_statement", "management_statement", "peer_context_signal"],
                must_have_features=common_finance_features,
                allow_heuristic_if_missing=True,
            ),
            validation_rules=[
                {"kind": "funding_mix_sum_to_one", "field": "funding_mix"},
                {"kind": "positive", "field": "size_pct_market_cap"},
            ],
        )
    )

    actions.append(
        _action(
            action_type="capital_return",
            action_subtype="accelerated_share_repurchase",
            label="Accelerated Share Repurchase",
            description="Front-loaded buyback via bank counterparty requiring stronger funding certainty.",
            parameter_schema={
                "size_pct_market_cap": _percent_field(required=True, minimum=0.0, maximum=0.5),
                "size_absolute_usd": _numeric_field(required=False, unit="USD", minimum=0.0),
                "funding_mix": _funding_mix(required=True),
                "execution_window": _date_window(required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "liquidity.available_for_actions", "operator": ">", "value": 0},
                    {"feature": "market.equity_window_proxy", "operator": ">", "value": 0.4},
                ],
                required_features=common_finance_features + ["market.equity_window_proxy"],
                required_evidence=["strategic.intent_vector"],
                forbidden_constraints=["no_capital_return"],
            ),
            mechanism_channels=[
                _channel(
                    "fast_undervaluation_capture",
                    "value_creation",
                    "ASR accelerates share retirement when management sees near-term undervaluation.",
                    ["market.ev_ebitda_vs_peer_z < 0", "market.equity_window_proxy > 0.5"],
                    ["market.volatility_30d high"],
                ),
                _channel(
                    "confidence_signal",
                    "signaling",
                    "Large ASR signals high conviction in operating trajectory.",
                    ["operating.margin_trend_8q > 0"],
                ),
            ],
            lead_time_prior=_lead_time(3, 21, 60),
            execution_complexity_prior=_complexity(
                3,
                ["requires_counterparty", "requires_cash_or_financing_certainty", "requires_board_approval"],
                "medium",
                ["CFO", "Treasury", "Legal", "Board", "IR"],
            ),
            dependency_rules=[
                _rule(
                    "requires_prior",
                    "capital_structure.refinancing",
                    "capital_structure.maturity_wall_ratio_24m > 0.20",
                    "soft",
                    "Lock refinancing before front-loading capital return under debt pressure.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "market_signal"],
                optional_supporting_classes=["capital_policy_statement"],
                must_have_features=common_finance_features + ["market.equity_window_proxy"],
            ),
            validation_rules=[
                {"kind": "funding_mix_sum_to_one", "field": "funding_mix"},
                {"kind": "positive", "field": "size_pct_market_cap"},
            ],
        )
    )

    actions.append(
        _action(
            action_type="capital_return",
            action_subtype="tender_offer_buyback",
            label="Tender Offer Buyback",
            description="Repurchase shares via fixed-price or Dutch auction tender process.",
            parameter_schema={
                "size_absolute_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "premium_pct": _percent_field(required=False, minimum=0.0, maximum=0.5),
                "funding_mix": _funding_mix(required=True),
                "execution_window": _date_window(required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "liquidity.available_for_actions", "operator": ">", "value": 0},
                    {"feature": "market.equity_window_proxy", "operator": ">", "value": 0.3},
                ],
                required_features=common_finance_features + ["market.equity_window_proxy"],
                required_evidence=["strategic.intent_vector", "strategic.constraint_set"],
                forbidden_constraints=["no_capital_return"],
            ),
            mechanism_channels=[
                _channel(
                    "rapid_share_count_reduction",
                    "value_creation",
                    "Tender can rapidly reduce share count with high execution certainty.",
                    ["market.equity_window_proxy > 0.3"],
                    ["capital_structure.net_leverage above_target"],
                ),
                _channel(
                    "capital_allocation_signal",
                    "signaling",
                    "Tender communicates explicit capital allocation posture.",
                    ["strategic.intent.return_capital_priority > 0.5"],
                ),
            ],
            lead_time_prior=_lead_time(10, 45, 120),
            execution_complexity_prior=_complexity(
                3,
                ["requires_offer_documents", "requires_legal_tender_process", "requires_treasury_execution"],
                "medium",
                ["CFO", "Treasury", "Legal", "Board", "IR"],
            ),
            dependency_rules=[
                _rule(
                    "discouraged_with",
                    "mna.platform_acquisition",
                    "liquidity.available_for_actions constrained",
                    "soft",
                    "Avoid simultaneous large tender and acquisition under limited liquidity.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "capital_policy_statement"],
                optional_supporting_classes=["market_signal", "management_statement"],
                must_have_features=common_finance_features,
            ),
            validation_rules=[
                {"kind": "funding_mix_sum_to_one", "field": "funding_mix"},
                {"kind": "positive", "field": "size_absolute_usd"},
            ],
        )
    )

    actions.append(
        _action(
            action_type="capital_return",
            action_subtype="dividend_increase",
            label="Dividend Increase",
            description="Increase recurring dividend commitment.",
            parameter_schema={
                "percent_change": _percent_field(required=True, minimum=0.0, maximum=1.0),
                "annualized_cash_commitment_usd": _numeric_field(required=False, unit="USD", minimum=0.0),
                "effective_quarter": _enum_field(["Q1", "Q2", "Q3", "Q4"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "operating.fcf_conversion", "operator": ">", "value": 0.2},
                    {"feature": "capital_structure.maturity_wall_ratio_24m", "operator": "<=", "value": 0.25},
                ],
                required_features=[
                    "operating.fcf_conversion",
                    "capital_structure.maturity_wall_ratio_24m",
                    "liquidity.available_for_actions",
                ],
                required_evidence=["strategic.intent_vector"],
                forbidden_constraints=["no_capital_return"],
            ),
            mechanism_channels=[
                _channel(
                    "stability_signal",
                    "signaling",
                    "Dividend increase signals stable cash generation confidence.",
                    ["operating.fcf_conversion stable_or_up"],
                    ["regime.risk_regime == risk_off"],
                ),
                _channel(
                    "shareholder_base_alignment",
                    "value_creation",
                    "Can optimize investor base toward income-oriented holders.",
                    ["ownership_governance.institutional_pct support"],
                ),
            ],
            lead_time_prior=_lead_time(7, 30, 90),
            execution_complexity_prior=_complexity(
                2,
                ["requires_board_approval", "requires_policy_commitment"],
                "low_to_medium",
                ["CFO", "Board", "IR"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "capital_policy_statement"],
                optional_supporting_classes=["management_statement", "market_signal"],
                must_have_features=["operating.fcf_conversion", "liquidity.available_for_actions"],
            ),
            validation_rules=[{"kind": "positive", "field": "percent_change"}],
        )
    )

    actions.append(
        _action(
            action_type="capital_return",
            action_subtype="dividend_initiate",
            label="Dividend Initiation",
            description="Establish a first recurring dividend as a durable capital-return commitment.",
            parameter_schema={
                "initial_yield_pct": _percent_field(required=True, minimum=0.0, maximum=0.08),
                "annualized_cash_commitment_usd": _numeric_field(required=False, unit="USD", minimum=0.0),
                "effective_quarter": _enum_field(["Q1", "Q2", "Q3", "Q4"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "operating.fcf_conversion", "operator": ">", "value": 0.15},
                    {"feature": "capital_structure.maturity_wall_ratio_24m", "operator": "<=", "value": 0.25},
                ],
                required_features=[
                    "operating.fcf_conversion",
                    "capital_structure.maturity_wall_ratio_24m",
                    "liquidity.available_for_actions",
                ],
                required_evidence=["capital_policy_statement", "management_statement"],
                forbidden_constraints=["no_capital_return"],
            ),
            mechanism_channels=[
                _channel(
                    "durable_cash_flow_signal",
                    "signaling",
                    "Initiating a recurring dividend signals confidence in durable free cash flow.",
                    ["operating.fcf_conversion stable_or_up"],
                    ["regime.risk_regime == risk_off"],
                ),
                _channel(
                    "shareholder_base_expansion",
                    "value_creation",
                    "Can broaden the shareholder base toward income-oriented holders.",
                    ["ownership_governance.institutional_pct support"],
                ),
            ],
            lead_time_prior=_lead_time(7, 30, 90),
            execution_complexity_prior=_complexity(
                2,
                ["requires_board_approval", "requires_policy_commitment"],
                "low_to_medium",
                ["CFO", "Board", "IR"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "capital_policy_statement"],
                optional_supporting_classes=["management_statement", "market_signal"],
                must_have_features=["operating.fcf_conversion", "liquidity.available_for_actions"],
            ),
            validation_rules=[{"kind": "positive", "field": "initial_yield_pct"}],
        )
    )

    actions.append(
        _action(
            action_type="capital_return",
            action_subtype="dividend_cut",
            label="Dividend Cut",
            description="Reduce recurring dividend payout to preserve liquidity or support deleveraging.",
            parameter_schema={
                "percent_change": _percent_field(required=True, minimum=-1.0, maximum=0.0),
                "effective_quarter": _enum_field(["Q1", "Q2", "Q3", "Q4"], required=False),
                "target_use_of_cash": _enum_field(["deleveraging", "liquidity_buffer", "reinvestment"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "liquidity.runway_months", "operator": "<", "value": 18},
                ],
                required_features=[
                    "liquidity.runway_months",
                    "capital_structure.maturity_wall_ratio_24m",
                ],
                required_evidence=["strategic.constraint_set"],
            ),
            mechanism_channels=[
                _channel(
                    "liquidity_preservation",
                    "risk_reduction",
                    "Preserves cash under stress or elevated refinancing risk.",
                    ["liquidity.runway_months low", "capital_structure.maturity_wall_ratio_24m high"],
                )
            ],
            lead_time_prior=_lead_time(3, 21, 60),
            execution_complexity_prior=_complexity(
                2,
                ["requires_board_approval", "requires_investor_communication"],
                "medium",
                ["CFO", "Board", "IR"],
            ),
            dependency_rules=[
                _rule(
                    "preferred_after",
                    "capital_structure.refinancing",
                    "market.credit_window_proxy > 0.5",
                    "soft",
                    "Refinancing may reduce the size/severity of a needed cut.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "liquidity_disclosure"],
                optional_supporting_classes=["management_statement", "rating_disclosure"],
                must_have_features=["liquidity.runway_months"],
            ),
            validation_rules=[{"kind": "non_positive", "field": "percent_change"}],
        )
    )

    actions.append(
        _action(
            action_type="capital_return",
            action_subtype="special_dividend",
            label="Special Dividend",
            description="One-time dividend distribution of excess cash.",
            parameter_schema={
                "size_absolute_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "funding_mix": _funding_mix(required=True),
                "payment_date_window": _date_window(required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "liquidity.available_for_actions", "operator": ">", "value": 0},
                    {"feature": "capital_structure.maturity_wall_ratio_24m", "operator": "<=", "value": 0.2},
                ],
                required_features=common_finance_features + ["capital_structure.maturity_wall_ratio_24m"],
                required_evidence=["capital_policy_statement"],
                forbidden_constraints=["no_capital_return"],
            ),
            mechanism_channels=[
                _channel(
                    "excess_cash_distribution",
                    "value_creation",
                    "Returns excess balance-sheet cash to shareholders in one-off format.",
                    ["liquidity.available_for_actions materially_positive"],
                    ["mna.pipeline_high_priority"],
                )
            ],
            lead_time_prior=_lead_time(7, 30, 75),
            execution_complexity_prior=_complexity(
                2,
                ["requires_board_approval", "requires_record_date_and_payment_ops"],
                "low_to_medium",
                ["CFO", "Board", "IR"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "capital_policy_statement"],
                optional_supporting_classes=["management_statement"],
                must_have_features=["liquidity.available_for_actions"],
            ),
            validation_rules=[
                {"kind": "funding_mix_sum_to_one", "field": "funding_mix"},
                {"kind": "positive", "field": "size_absolute_usd"},
            ],
        )
    )

    # 4.2.2 Capital Structure
    actions.append(
        _action(
            action_type="capital_structure",
            action_subtype="new_debt_issuance",
            label="New Debt Issuance",
            description="Issue new debt to fund operations, strategic actions, or liability optimization.",
            parameter_schema={
                "amount_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "tenor_years": _numeric_field(required=True, unit="years", minimum=0.0),
                "secured_flag": {"type": "boolean", "required": False},
                "use_of_proceeds": _enum_field(["general_corporate", "refinancing", "mna", "buyback", "liquidity_buffer"], required=True),
                "fixed_vs_floating": _enum_field(["fixed", "floating", "mixed"], required=False),
                "instrument_type": _enum_field(["bond", "term_loan", "notes"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "market.credit_window_proxy", "operator": ">", "value": 0.25},
                    {"feature": "capital_structure.interest_coverage", "operator": ">", "value": 1.0},
                ],
                required_features=[
                    "market.credit_window_proxy",
                    "capital_structure.interest_coverage",
                    "capital_structure.net_leverage",
                ],
                required_evidence=["strategic.intent_vector"],
                forbidden_constraints=["no_new_debt"],
            ),
            mechanism_channels=[
                _channel(
                    "liquidity_extension",
                    "risk_reduction",
                    "Extends liquidity runway and supports near-term obligations.",
                    ["liquidity.runway_months constrained", "market.credit_window_proxy open"],
                ),
                _channel(
                    "terming_out_maturities",
                    "capital_structure_optimization",
                    "Shifts debt profile to reduce near-term maturity wall.",
                    ["capital_structure.maturity_wall_ratio_24m > 0.2"],
                ),
                _channel(
                    "fragility_risk",
                    "risk_reduction",
                    "Can increase fragility if debt is opportunistic and leverage is stretched.",
                    ["capital_structure.net_leverage elevated"],
                    ["operating.fcf_conversion weak"],
                ),
            ],
            lead_time_prior=_lead_time(3, 21, 60),
            execution_complexity_prior=_complexity(
                3,
                ["requires_market_access", "requires_underwriting_or_lender_process", "requires_disclosure_work"],
                "medium",
                ["CFO", "Treasury", "Legal", "Board"],
            ),
            dependency_rules=[
                _rule(
                    "unlocks",
                    "capital_return.open_market_buyback",
                    "use_of_proceeds in ['buyback','general_corporate']",
                    "soft",
                    "Incremental debt capacity can unlock buyback actions.",
                ),
                _rule(
                    "discouraged_with",
                    "capital_structure.equity_issuance",
                    "use_of_proceeds == 'general_corporate'",
                    "soft",
                    "Simultaneous debt and equity raises can send conflicting capital signals.",
                ),
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "market_signal", "rating_disclosure"],
                optional_supporting_classes=["management_statement", "capital_policy_statement"],
                must_have_features=["market.credit_window_proxy", "capital_structure.interest_coverage"],
            ),
            validation_rules=[{"kind": "positive", "field": "amount_usd"}],
        )
    )

    actions.append(
        _action(
            action_type="capital_structure",
            action_subtype="refinancing",
            label="Refinancing",
            description="Refinance outstanding debt to smooth maturities and improve risk profile.",
            parameter_schema={
                "amount_refinanced_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "maturities_targeted": _range_field(required=False, unit="years"),
                "new_tenor_years": _numeric_field(required=False, unit="years", minimum=0.0),
                "secured_flag": {"type": "boolean", "required": False},
                "rate_structure": _enum_field(["fixed", "floating", "mixed"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "capital_structure.maturity_wall_ratio_24m", "operator": ">", "value": 0.1},
                    {"feature": "market.credit_window_proxy", "operator": ">", "value": 0.2},
                ],
                required_features=[
                    "capital_structure.maturity_wall_ratio_24m",
                    "market.credit_window_proxy",
                    "capital_structure.interest_coverage",
                ],
                required_evidence=["liquidity_disclosure"],
            ),
            mechanism_channels=[
                _channel(
                    "maturity_risk_reduction",
                    "risk_reduction",
                    "Refinancing lowers near-term rollover risk and distress probability.",
                    ["capital_structure.maturity_wall_ratio_24m elevated"],
                ),
                _channel(
                    "optionality_preservation",
                    "optionality_preservation",
                    "Terming out debt preserves strategic flexibility for future actions.",
                    ["liquidity.runway_months pressured"],
                ),
            ],
            lead_time_prior=_lead_time(5, 30, 120),
            execution_complexity_prior=_complexity(
                3,
                ["requires_lender_or_market_process", "requires_documentation", "requires_treasury_coordination"],
                "medium",
                ["CFO", "Treasury", "Legal"],
            ),
            dependency_rules=[
                _rule(
                    "unlocks",
                    "capital_return.open_market_buyback",
                    "capital_structure.maturity_wall_ratio_24m drops below 0.2",
                    "soft",
                    "Refinancing first reduces fragility before returning capital.",
                ),
                _rule(
                    "unlocks",
                    "mna.tuck_in_acquisition",
                    "post_refi liquidity runway improves",
                    "soft",
                    "A cleaner debt profile broadens acquisition capacity.",
                ),
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "liquidity_disclosure", "market_signal"],
                optional_supporting_classes=["rating_disclosure"],
                must_have_features=["capital_structure.maturity_wall_ratio_24m", "market.credit_window_proxy"],
            ),
            validation_rules=[{"kind": "positive", "field": "amount_refinanced_usd"}],
        )
    )

    actions.append(
        _action(
            action_type="capital_structure",
            action_subtype="tender_offer_debt",
            label="Debt Tender Offer",
            description="Repurchase specific debt tranches before maturity via tender.",
            parameter_schema={
                "target_tranche_id": {"type": "entity_reference", "required": True},
                "amount_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "premium_pct": _percent_field(required=False, minimum=0.0, maximum=0.5),
                "funding_mix": _funding_mix(required=True),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "liquidity.available_for_actions", "operator": ">", "value": 0},
                ],
                required_features=["liquidity.available_for_actions", "capital_structure.total_debt"],
                required_evidence=["liquidity_disclosure", "rating_disclosure"],
            ),
            mechanism_channels=[
                _channel(
                    "debt_overhang_reduction",
                    "risk_reduction",
                    "Reduces overhang and smooths liability profile.",
                    ["capital_structure.total_debt elevated"],
                )
            ],
            lead_time_prior=_lead_time(7, 35, 100),
            execution_complexity_prior=_complexity(
                4,
                ["requires_offer_docs", "requires_holder_coordination", "requires_legal_clearance"],
                "high",
                ["CFO", "Treasury", "Legal"],
            ),
            dependency_rules=[
                _rule(
                    "preferred_after",
                    "capital_structure.new_debt_issuance",
                    "funding_mix.debt > 0.5",
                    "soft",
                    "Tender often paired with fresh issuance as a liability management package.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "liquidity_disclosure"],
                optional_supporting_classes=["rating_disclosure", "management_statement"],
                must_have_features=["capital_structure.total_debt"],
            ),
            validation_rules=[
                {"kind": "funding_mix_sum_to_one", "field": "funding_mix"},
                {"kind": "positive", "field": "amount_usd"},
            ],
        )
    )

    actions.append(
        _action(
            action_type="capital_structure",
            action_subtype="exchange_offer",
            label="Debt Exchange Offer",
            description="Exchange existing debt for modified terms to improve maturity/coupon profile.",
            parameter_schema={
                "target_instruments": {"type": "entity_reference", "required": True},
                "new_tenor_years": _numeric_field(required=False, unit="years", minimum=0.0),
                "coupon_step": _numeric_field(required=False, unit="bps"),
                "participation_threshold_pct": _percent_field(required=False, minimum=0.0, maximum=1.0),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "capital_structure.maturity_wall_ratio_24m", "operator": ">", "value": 0.15},
                ],
                required_features=["capital_structure.maturity_wall_ratio_24m", "liquidity.runway_months"],
                required_evidence=["liquidity_disclosure", "rating_disclosure"],
            ),
            mechanism_channels=[
                _channel(
                    "maturity_extension",
                    "risk_reduction",
                    "Extends debt profile and reduces near-term default pressure.",
                    ["capital_structure.maturity_wall_ratio_24m high"],
                )
            ],
            lead_time_prior=_lead_time(14, 60, 180),
            execution_complexity_prior=_complexity(
                4,
                ["requires_creditor_negotiation", "requires_offer_structuring", "legal_complexity"],
                "high",
                ["CFO", "Treasury", "Legal", "Advisors"],
            ),
            dependency_rules=[
                _rule(
                    "preferred_after",
                    "capital_structure.liability_management_exercise",
                    None,
                    "soft",
                    "Exchange may be one component of broader liability management path.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "rating_disclosure"],
                optional_supporting_classes=["management_statement"],
                must_have_features=["capital_structure.maturity_wall_ratio_24m"],
            ),
            validation_rules=[],
        )
    )

    actions.append(
        _action(
            action_type="capital_structure",
            action_subtype="liability_management_exercise",
            label="Liability Management Exercise",
            description="Structured debt management package (amend-and-extend, distressed exchange, covenant relief).",
            parameter_schema={
                "structure_class": _enum_field(["amend_extend", "distressed_exchange", "covenant_relief", "uptier"], required=True),
                "targeted_instruments": {"type": "entity_reference", "required": True},
                "coercive_level": _enum_field(["low", "medium", "high"], required=False),
                "expected_participation_threshold_pct": _percent_field(required=False, minimum=0.0, maximum=1.0),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "capital_structure.maturity_wall_ratio_24m", "operator": ">", "value": 0.2},
                ],
                required_features=[
                    "capital_structure.maturity_wall_ratio_24m",
                    "liquidity.runway_months",
                    "capital_structure.interest_coverage",
                ],
                required_evidence=["rating_disclosure", "management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "distress_mitigation",
                    "risk_reduction",
                    "Mitigates distress risk through negotiated liability profile adjustment.",
                    ["liquidity.runway_months short", "capital_structure.maturity_wall_ratio_24m high"],
                )
            ],
            lead_time_prior=_lead_time(21, 90, 240),
            execution_complexity_prior=_complexity(
                5,
                ["legal_complexity", "multi_creditor_coordination", "high_execution_risk"],
                "very_high",
                ["CFO", "Legal", "Treasury", "Board", "Restructuring Advisors"],
            ),
            dependency_rules=[
                _rule(
                    "unlocks",
                    "capital_structure.exchange_offer",
                    None,
                    "soft",
                    "Broader LME path can precede specific exchange/tender actions.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "rating_disclosure", "liquidity_disclosure"],
                optional_supporting_classes=["management_statement"],
                must_have_features=["capital_structure.maturity_wall_ratio_24m", "liquidity.runway_months"],
                allow_heuristic_if_missing=False,
            ),
            validation_rules=[],
        )
    )

    actions.append(
        _action(
            action_type="capital_structure",
            action_subtype="revolver_draw_or_resize",
            label="Revolver Draw/Resize",
            description="Draw existing revolver or resize commitments to protect short-term liquidity.",
            parameter_schema={
                "draw_amount_usd": _numeric_field(required=False, unit="USD", minimum=0.0),
                "resize_amount_usd": _numeric_field(required=False, unit="USD"),
                "intent": _enum_field(["precautionary_draw", "bridge_funding", "resize"], required=True),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[{"feature": "liquidity.revolver_undrawn", "operator": ">", "value": 0}],
                required_features=["liquidity.revolver_undrawn", "liquidity.runway_months"],
                required_evidence=["liquidity_disclosure"],
            ),
            mechanism_channels=[
                _channel(
                    "liquidity_bridge",
                    "risk_reduction",
                    "Provides immediate liquidity bridge in uncertain market conditions.",
                    ["liquidity.runway_months constrained"],
                )
            ],
            lead_time_prior=_lead_time(1, 5, 20),
            execution_complexity_prior=_complexity(
                2,
                ["bank_coordination", "treasury_execution"],
                "low_to_medium",
                ["Treasury", "CFO"],
            ),
            dependency_rules=[
                _rule(
                    "preferred_after",
                    "capital_structure.refinancing",
                    "intent == 'bridge_funding'",
                    "soft",
                    "Bridge draw should generally transition into longer-term refinancing.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["liquidity_disclosure", "financial_disclosure"],
                optional_supporting_classes=["management_statement"],
                must_have_features=["liquidity.revolver_undrawn"],
            ),
            validation_rules=[],
        )
    )

    actions.append(
        _action(
            action_type="capital_structure",
            action_subtype="equity_issuance",
            label="Equity Issuance",
            description="Issue common equity to de-lever, fund growth, or preserve liquidity.",
            parameter_schema={
                "amount_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "discount_pct": _percent_field(required=False, minimum=0.0, maximum=0.3),
                "use_of_proceeds": _enum_field(["deleveraging", "mna", "liquidity_buffer", "general_corporate"], required=True),
                "offering_type": _enum_field(["follow_on", "at_the_market", "private_placement"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "market.equity_window_proxy", "operator": ">", "value": 0.2},
                ],
                required_features=["market.equity_window_proxy", "market.market_cap"],
                required_evidence=["capital_policy_statement", "management_statement"],
                forbidden_constraints=["no_equity_issuance"],
            ),
            mechanism_channels=[
                _channel(
                    "deleveraging",
                    "risk_reduction",
                    "Injects permanent capital and can reduce leverage/stress.",
                    ["capital_structure.net_leverage elevated"],
                ),
                _channel(
                    "liquidity_preservation",
                    "optionality_preservation",
                    "Preserves optionality by replenishing cash under uncertainty.",
                    ["liquidity.runway_months constrained"],
                ),
            ],
            lead_time_prior=_lead_time(5, 21, 60),
            execution_complexity_prior=_complexity(
                3,
                ["requires_equity_window", "requires_disclosure_and_execution"],
                "medium",
                ["CFO", "Legal", "IR", "Board"],
            ),
            dependency_rules=[
                _rule(
                    "conflicts_with",
                    "capital_return.open_market_buyback",
                    None,
                    "hard",
                    "Issuing and repurchasing equity concurrently is generally incoherent.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["market_signal", "financial_disclosure"],
                optional_supporting_classes=["capital_policy_statement", "management_statement"],
                must_have_features=["market.equity_window_proxy", "market.market_cap"],
            ),
            validation_rules=[{"kind": "positive", "field": "amount_usd"}],
        )
    )

    actions.append(
        _action(
            action_type="capital_structure",
            action_subtype="convertible_issuance",
            label="Convertible Issuance",
            description="Issue convertible securities balancing coupon savings with potential future dilution.",
            parameter_schema={
                "amount_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "tenor_years": _numeric_field(required=False, unit="years", minimum=0.0),
                "conversion_premium_pct": _percent_field(required=False, minimum=0.0, maximum=1.0),
                "use_of_proceeds": _enum_field(["deleveraging", "mna", "general_corporate"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "market.equity_window_proxy", "operator": ">", "value": 0.2},
                    {"feature": "market.volatility_30d", "operator": ">", "value": 0},
                ],
                required_features=["market.equity_window_proxy", "market.volatility_30d", "market.market_cap"],
                required_evidence=["market_signal", "management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "lower_coupon_financing",
                    "capital_structure_optimization",
                    "Convertible can reduce cash coupon versus straight debt.",
                    ["market.equity_window_proxy open", "equity_story credible"],
                ),
                _channel(
                    "delayed_dilution",
                    "optionality_preservation",
                    "Defers dilution versus immediate equity issuance.",
                    ["conversion_premium attractive"],
                ),
            ],
            lead_time_prior=_lead_time(7, 28, 75),
            execution_complexity_prior=_complexity(
                4,
                ["requires_structuring", "hedging_and_convert_math", "investor_marketing"],
                "high",
                ["CFO", "Treasury", "Legal", "IR"],
            ),
            dependency_rules=[
                _rule(
                    "discouraged_with",
                    "capital_return.open_market_buyback",
                    None,
                    "soft",
                    "Concurrent convert issuance and buyback may produce mixed external signaling.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["market_signal", "financial_disclosure"],
                optional_supporting_classes=["management_statement"],
                must_have_features=["market.equity_window_proxy", "market.volatility_30d"],
            ),
            validation_rules=[{"kind": "positive", "field": "amount_usd"}],
        )
    )

    actions.append(
        _action(
            action_type="capital_structure",
            action_subtype="preferred_issuance",
            label="Preferred Issuance",
            description="Issue preferred equity/hybrid capital to support liquidity and leverage objectives.",
            parameter_schema={
                "amount_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "coupon_pct": _percent_field(required=False, minimum=0.0, maximum=1.0),
                "call_protection_years": _numeric_field(required=False, unit="years", minimum=0.0),
                "use_of_proceeds": _enum_field(["deleveraging", "liquidity_buffer", "general_corporate"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[{"feature": "market.equity_window_proxy", "operator": ">", "value": 0.15}],
                required_features=["market.equity_window_proxy", "capital_structure.net_leverage"],
                required_evidence=["financial_disclosure", "capital_policy_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "hybrid_capital_buffer",
                    "optionality_preservation",
                    "Preferred can provide quasi-equity capital without immediate common dilution.",
                    ["capital_structure.net_leverage elevated"],
                )
            ],
            lead_time_prior=_lead_time(10, 35, 90),
            execution_complexity_prior=_complexity(
                4,
                ["structuring_complexity", "investor_placement", "legal_docs"],
                "high",
                ["CFO", "Treasury", "Legal", "IR"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "market_signal"],
                optional_supporting_classes=["capital_policy_statement"],
                must_have_features=["capital_structure.net_leverage"],
            ),
            validation_rules=[{"kind": "positive", "field": "amount_usd"}],
        )
    )

    # 4.2.3 M&A / Portfolio Actions
    actions.append(
        _action(
            action_type="mna",
            action_subtype="tuck_in_acquisition",
            label="Tuck-in Acquisition",
            description="Small adjacency acquisition to add capabilities or cross-sell fit.",
            parameter_schema={
                "target_size_pct_ev": _percent_field(required=True, minimum=0.0, maximum=0.25),
                "target_sector_match": _enum_field(["high", "medium", "low"], required=False),
                "funding_mix": _funding_mix(required=True),
                "leverage_post_close": _numeric_field(required=False, unit="x", minimum=0.0),
                "synergy_case_strength": _enum_field(["low", "medium", "high"], required=False),
                "geography_overlap": _enum_field(["high", "medium", "low"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "liquidity.available_for_actions", "operator": ">", "value": 0},
                    {"feature": "capital_structure.net_leverage", "operator": "<=", "value": 4.5},
                ],
                required_features=common_finance_features + ["strategic.intent.pursue_mna_priority"],
                required_evidence=["management_statement", "strategic.intent_vector"],
                forbidden_constraints=["no_mna"],
            ),
            mechanism_channels=[
                _channel(
                    "capability_extension",
                    "growth_substitution",
                    "Adds product/customer capability without full platform risk.",
                    ["strategic.intent.pursue_mna_priority high"],
                ),
                _channel(
                    "cost_synergy",
                    "value_creation",
                    "Creates operational and procurement synergies.",
                    ["operating.margin_volatility_8q manageable"],
                ),
            ],
            lead_time_prior=_lead_time(30, 120, 270),
            execution_complexity_prior=_complexity(
                3,
                ["due_diligence", "integration_planning", "financing_coordination"],
                "medium",
                ["CorpDev", "CFO", "Legal", "Business Units"],
            ),
            dependency_rules=[
                _rule(
                    "requires_prior",
                    "capital_structure.refinancing",
                    "capital_structure.net_leverage > 3.5",
                    "soft",
                    "Refinancing may be needed first if leverage headroom is tight.",
                ),
                _rule(
                    "conflicts_with",
                    "capital_return.open_market_buyback",
                    "liquidity.available_for_actions constrained",
                    "soft",
                    "Tuck-in and buyback can compete for capital if capacity is limited.",
                ),
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "management_statement"],
                optional_supporting_classes=["peer_context_signal", "market_signal"],
                must_have_features=["liquidity.available_for_actions", "capital_structure.net_leverage"],
            ),
            validation_rules=[
                {"kind": "funding_mix_sum_to_one", "field": "funding_mix"},
                {"kind": "positive", "field": "target_size_pct_ev"},
            ],
        )
    )

    actions.append(
        _action(
            action_type="mna",
            action_subtype="platform_acquisition",
            label="Platform Acquisition",
            description="Mid-size strategic acquisition requiring substantial integration and financing discipline.",
            parameter_schema={
                "target_size_pct_ev": _percent_field(required=True, minimum=0.0, maximum=0.75),
                "funding_mix": _funding_mix(required=True),
                "leverage_post_close": _numeric_field(required=False, unit="x", minimum=0.0),
                "expected_synergy_pct": _percent_field(required=False, minimum=0.0, maximum=1.0),
                "regulatory_risk": _enum_field(["low", "medium", "high"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "capital_structure.net_leverage", "operator": "<=", "value": 4.5},
                ],
                required_features=[
                    "capital_structure.net_leverage",
                    "liquidity.available_for_actions",
                    "market.credit_window_proxy",
                ],
                required_evidence=["management_statement", "strategic.intent_vector", "financial_disclosure"],
                forbidden_constraints=["no_mna"],
            ),
            mechanism_channels=[
                _channel(
                    "scale_and_scope",
                    "growth_substitution",
                    "Accelerates scale or portfolio adjacency versus organic path.",
                    ["strategic.intent.pursue_mna_priority > 0.5"],
                ),
                _channel(
                    "portfolio_upgrade",
                    "portfolio_optimization",
                    "Rebalances business mix toward higher growth/profit pools.",
                    ["peer_context.relative_positioning gap"],
                ),
            ],
            lead_time_prior=_lead_time(45, 180, 365),
            execution_complexity_prior=_complexity(
                4,
                ["deal_execution", "financing", "integration", "regulatory_process"],
                "high",
                ["CorpDev", "CFO", "CEO", "Legal", "Operations"],
            ),
            dependency_rules=[
                _rule(
                    "requires_prior",
                    "capital_structure.refinancing",
                    "capital_structure.maturity_wall_ratio_24m > 0.2",
                    "soft",
                    "Stabilize debt profile before committing to large platform deal.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "management_statement", "peer_context_signal"],
                optional_supporting_classes=["market_signal"],
                must_have_features=["capital_structure.net_leverage", "liquidity.available_for_actions"],
                allow_heuristic_if_missing=False,
            ),
            validation_rules=[
                {"kind": "funding_mix_sum_to_one", "field": "funding_mix"},
                {"kind": "positive", "field": "target_size_pct_ev"},
            ],
        )
    )

    actions.append(
        _action(
            action_type="mna",
            action_subtype="go_private_lbo",
            label="Go-Private / LBO",
            description="Take-private transaction financed primarily with debt and sponsor equity.",
            parameter_schema={
                "target_size_pct_ev": _percent_field(required=True, minimum=0.5, maximum=2.0),
                "funding_mix": _funding_mix(required=True),
                "leverage_post_close": _numeric_field(required=True, unit="x", minimum=0.0),
                "take_private_premium_pct": _percent_field(required=False, minimum=0.0, maximum=0.6),
                "sponsor_type": _enum_field(["financial_sponsor", "consortium", "management_led"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "market.credit_window_proxy", "operator": ">", "value": 0.2},
                ],
                required_features=[
                    "market.credit_window_proxy",
                    "market.equity_window_proxy",
                    "capital_structure.net_leverage",
                ],
                required_evidence=["management_statement", "financial_disclosure", "market_signal"],
                forbidden_constraints=["no_mna", "maintain_low_leverage"],
            ),
            mechanism_channels=[
                _channel(
                    "valuation_reset",
                    "value_creation",
                    "Can crystallize value when public-market valuation is persistently dislocated.",
                    ["market.ev_ebitda_vs_peer_z materially_negative"],
                ),
                _channel(
                    "private_ownership_flexibility",
                    "optionality_preservation",
                    "Private ownership can support restructuring or longer-duration strategic repositioning.",
                    ["strategic.intent_vector privatization_or_strategic_alternatives"],
                ),
            ],
            lead_time_prior=_lead_time(75, 240, 540),
            execution_complexity_prior=_complexity(
                5,
                ["large_financing", "board_process", "shareholder_vote", "regulatory_clearance"],
                "very_high",
                ["Board", "CEO", "CFO", "CorpDev", "Legal", "IR"],
            ),
            dependency_rules=[
                _rule(
                    "conflicts_with",
                    "capital_return.open_market_buyback",
                    None,
                    "hard",
                    "Going private precludes concurrent discretionary share repurchase.",
                ),
                _rule(
                    "conflicts_with",
                    "capital_return.dividend_increase",
                    None,
                    "hard",
                    "A take-private process is incompatible with launching a larger recurring payout commitment.",
                ),
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "management_statement", "market_signal"],
                optional_supporting_classes=["capital_policy_statement"],
                must_have_features=["market.credit_window_proxy", "capital_structure.net_leverage"],
                allow_heuristic_if_missing=False,
            ),
            validation_rules=[
                {"kind": "funding_mix_sum_to_one", "field": "funding_mix"},
                {"kind": "positive", "field": "target_size_pct_ev"},
                {"kind": "positive", "field": "leverage_post_close"},
            ],
        )
    )

    actions.append(
        _action(
            action_type="mna",
            action_subtype="transformational_acquisition",
            label="Transformational Acquisition",
            description="Large strategic transaction with material financing, integration, and regulatory complexity.",
            parameter_schema={
                "target_size_pct_ev": _percent_field(required=True, minimum=0.25, maximum=2.0),
                "funding_mix": _funding_mix(required=True),
                "leverage_post_close": _numeric_field(required=True, unit="x", minimum=0.0),
                "strategic_thesis_strength": _enum_field(["medium", "high"], required=True),
                "regulatory_risk": _enum_field(["medium", "high"], required=True),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "liquidity.available_for_actions", "operator": ">", "value": 0},
                ],
                required_features=[
                    "liquidity.available_for_actions",
                    "capital_structure.net_leverage",
                    "market.credit_window_proxy",
                    "market.equity_window_proxy",
                ],
                required_evidence=["management_statement", "financial_disclosure", "capital_policy_statement"],
                forbidden_constraints=["no_mna", "maintain_low_leverage"],
            ),
            mechanism_channels=[
                _channel(
                    "strategic_repositioning",
                    "growth_substitution",
                    "Repositions company trajectory with significant scale/capability shift.",
                    ["strategic.intent.pursue_mna_priority very_high"],
                    ["regime.risk_regime == risk_off"],
                ),
                _channel(
                    "portfolio_recomposition",
                    "portfolio_optimization",
                    "Can shift mix toward structurally advantaged segments.",
                    ["operating.growth_profile below_target"],
                ),
            ],
            lead_time_prior=_lead_time(60, 240, 540),
            execution_complexity_prior=_complexity(
                5,
                ["large_financing", "regulatory_clearance", "integration_complexity", "organization_change"],
                "very_high",
                ["Board", "CEO", "CFO", "CorpDev", "Legal", "Operations", "HR"],
            ),
            dependency_rules=[
                _rule(
                    "conflicts_with",
                    "capital_return.open_market_buyback",
                    None,
                    "hard",
                    "Transformational M&A usually precludes simultaneous discretionary buyback.",
                ),
                _rule(
                    "preferred_after",
                    "restructuring.cost_program",
                    "integration_capacity constrained",
                    "soft",
                    "Cost/simplification prep can improve integration success odds.",
                ),
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=[
                    "financial_disclosure",
                    "management_statement",
                    "capital_policy_statement",
                    "peer_context_signal",
                ],
                optional_supporting_classes=["market_signal", "segment_disclosure"],
                must_have_features=[
                    "capital_structure.net_leverage",
                    "liquidity.available_for_actions",
                    "market.credit_window_proxy",
                    "market.equity_window_proxy",
                ],
                allow_heuristic_if_missing=False,
            ),
            validation_rules=[
                {"kind": "funding_mix_sum_to_one", "field": "funding_mix"},
                {"kind": "positive", "field": "target_size_pct_ev"},
                {"kind": "positive", "field": "leverage_post_close"},
            ],
        )
    )

    actions.append(
        _action(
            action_type="mna",
            action_subtype="minority_investment",
            label="Minority Investment",
            description="Acquire minority stake for strategic optionality with lower integration risk.",
            parameter_schema={
                "investment_size_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "ownership_pct": _percent_field(required=False, minimum=0.0, maximum=0.5),
                "funding_mix": _funding_mix(required=True),
                "strategic_option": _enum_field(["commercial_access", "technology_option", "future_mna_path"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[{"feature": "liquidity.available_for_actions", "operator": ">", "value": 0}],
                required_features=["liquidity.available_for_actions", "capital_structure.net_leverage"],
                required_evidence=["management_statement", "strategic.intent_vector"],
                forbidden_constraints=["no_mna"],
            ),
            mechanism_channels=[
                _channel(
                    "optionality_preservation",
                    "optionality_preservation",
                    "Creates strategic option value without full control transaction risk.",
                    ["strategic.intent.pursue_mna_priority moderate"],
                )
            ],
            lead_time_prior=_lead_time(14, 75, 180),
            execution_complexity_prior=_complexity(
                3,
                ["diligence", "investment_terms", "governance_rights_negotiation"],
                "medium",
                ["CorpDev", "CFO", "Legal"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "management_statement"],
                optional_supporting_classes=["peer_context_signal"],
                must_have_features=["liquidity.available_for_actions"],
            ),
            validation_rules=[
                {"kind": "funding_mix_sum_to_one", "field": "funding_mix"},
                {"kind": "positive", "field": "investment_size_usd"},
            ],
        )
    )

    actions.append(
        _action(
            action_type="portfolio",
            action_subtype="divestiture_full",
            label="Full Divestiture",
            description="Sell an entire segment/business to generate proceeds and simplify portfolio.",
            parameter_schema={
                "segment_reference": {"type": "segment_reference", "required": True},
                "estimated_ev_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "percent_divested": _percent_field(required=True, minimum=1.0, maximum=1.0),
                "use_of_proceeds": _enum_field(["deleveraging", "buyback", "reinvestment", "liquidity_buffer"], required=True),
                "sale_type": _enum_field(["strategic", "sponsor", "broad_process"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[{"feature": "capital_structure.net_leverage", "operator": ">=", "value": 0}],
                required_features=["capital_structure.net_leverage", "strategic.constraint_set"],
                required_evidence=["segment_disclosure", "management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "deleveraging_via_proceeds",
                    "risk_reduction",
                    "Uses sale proceeds to reduce debt and maturity pressure.",
                    ["capital_structure.net_leverage elevated"],
                ),
                _channel(
                    "portfolio_focus",
                    "portfolio_optimization",
                    "Simplifies portfolio and sharpens management focus.",
                    ["strategic.intent.focus_on_core > 0.5"],
                ),
            ],
            lead_time_prior=_lead_time(45, 150, 360),
            execution_complexity_prior=_complexity(
                4,
                ["separation_planning", "buyer_process", "legal_transaction_execution"],
                "high",
                ["CEO", "CFO", "CorpDev", "Legal", "Operations"],
            ),
            dependency_rules=[
                _rule(
                    "unlocks",
                    "capital_return.open_market_buyback",
                    "use_of_proceeds == 'buyback'",
                    "soft",
                    "Divestiture proceeds can fund return of capital after de-risking.",
                ),
                _rule(
                    "preferred_after",
                    "capital_structure.refinancing",
                    "distress_discount_risk high",
                    "soft",
                    "Stabilize financing before running a sale process under pressure.",
                ),
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["segment_disclosure", "financial_disclosure", "management_statement"],
                optional_supporting_classes=["market_signal", "peer_context_signal"],
                must_have_features=["capital_structure.net_leverage"],
                allow_heuristic_if_missing=False,
            ),
            validation_rules=[{"kind": "segment_reference_exists", "field": "segment_reference"}],
        )
    )

    actions.append(
        _action(
            action_type="portfolio",
            action_subtype="divestiture_partial",
            label="Partial Divestiture",
            description="Sell a partial stake in a segment while retaining some strategic optionality.",
            parameter_schema={
                "segment_reference": {"type": "segment_reference", "required": True},
                "percent_divested": _percent_field(required=True, minimum=0.05, maximum=0.95),
                "estimated_ev_usd": _numeric_field(required=False, unit="USD", minimum=0.0),
                "use_of_proceeds": _enum_field(["deleveraging", "reinvestment", "liquidity_buffer"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["capital_structure.net_leverage"],
                required_evidence=["segment_disclosure", "management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "capital_recycling",
                    "portfolio_optimization",
                    "Recycles capital while preserving strategic exposure.",
                    ["strategic.intent.focus_on_core or strategic.optionality objective"],
                )
            ],
            lead_time_prior=_lead_time(45, 135, 300),
            execution_complexity_prior=_complexity(
                4,
                ["valuation_work", "deal_structuring", "governance_and_control_terms"],
                "high",
                ["CEO", "CFO", "CorpDev", "Legal"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["segment_disclosure", "financial_disclosure"],
                optional_supporting_classes=["management_statement"],
                must_have_features=["capital_structure.net_leverage"],
            ),
            validation_rules=[{"kind": "segment_reference_exists", "field": "segment_reference"}],
        )
    )

    actions.append(
        _action(
            action_type="portfolio",
            action_subtype="asset_sale",
            label="Asset Sale",
            description="Sell non-core asset or business line outside formal segment divestiture.",
            parameter_schema={
                "asset_reference": {"type": "entity_reference", "required": True},
                "estimated_proceeds_usd": _numeric_field(required=False, unit="USD", minimum=0.0),
                "use_of_proceeds": _enum_field(["deleveraging", "liquidity_buffer", "reinvestment"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["capital_structure.net_leverage"],
                required_evidence=["management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "liquidity_generation",
                    "risk_reduction",
                    "Monetizes non-core assets to generate liquidity and simplify footprint.",
                    ["liquidity.runway_months constrained"],
                )
            ],
            lead_time_prior=_lead_time(30, 90, 240),
            execution_complexity_prior=_complexity(
                3,
                ["asset_marketing", "negotiation", "closing_execution"],
                "medium",
                ["CFO", "Legal", "Operations"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["management_statement", "financial_disclosure"],
                optional_supporting_classes=["segment_disclosure"],
                must_have_features=["capital_structure.net_leverage"],
            ),
            validation_rules=[],
        )
    )

    actions.append(
        _action(
            action_type="portfolio",
            action_subtype="spin_off",
            label="Spin-Off",
            description="Separate a business into standalone public entity.",
            parameter_schema={
                "segment_reference": {"type": "segment_reference", "required": True},
                "tax_free_feasibility_flag": {"type": "boolean", "required": False},
                "debt_allocation_pct": _percent_field(required=False, minimum=0.0, maximum=1.0),
                "standalone_readiness_score": _numeric_field(required=False, minimum=0.0),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["strategic.constraint_set", "market.equity_window_proxy"],
                required_evidence=["segment_disclosure", "management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "pure_play_rerating",
                    "portfolio_optimization",
                    "May unlock valuation by creating focused pure-play entities.",
                    ["segment economics separable", "equity window reasonably open"],
                    ["urgent liquidity distress"],
                )
            ],
            lead_time_prior=_lead_time(90, 270, 540),
            execution_complexity_prior=_complexity(
                5,
                ["standalone_build", "tax_structuring", "regulatory_and_listing_work"],
                "very_high",
                ["CEO", "CFO", "Legal", "Tax", "Operations", "IR"],
            ),
            dependency_rules=[
                _rule(
                    "conflicts_with",
                    "restructuring.chapter_pathway",
                    None,
                    "hard",
                    "Spin process is generally infeasible under acute chapter pathway conditions.",
                ),
                _rule(
                    "preferred_after",
                    "restructuring.cost_program",
                    "standalone_readiness_score low",
                    "soft",
                    "Operational prep can improve standalone viability.",
                ),
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["segment_disclosure", "management_statement", "financial_disclosure"],
                optional_supporting_classes=["market_signal"],
                must_have_features=["market.equity_window_proxy"],
                allow_heuristic_if_missing=False,
            ),
            validation_rules=[{"kind": "segment_reference_exists", "field": "segment_reference"}],
        )
    )

    actions.append(
        _action(
            action_type="portfolio",
            action_subtype="carve_out_ipo",
            label="Carve-Out IPO",
            description="IPO minority stake of a business while retaining parent ownership.",
            parameter_schema={
                "segment_reference": {"type": "segment_reference", "required": True},
                "percent_float": _percent_field(required=True, minimum=0.05, maximum=0.49),
                "use_of_proceeds": _enum_field(["deleveraging", "parent_liquidity", "segment_growth"], required=False),
                "readiness_score": _numeric_field(required=False, minimum=0.0),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[{"feature": "market.equity_window_proxy", "operator": ">", "value": 0.3}],
                required_features=["market.equity_window_proxy", "segment_disclosure"],
                required_evidence=["segment_disclosure", "management_statement", "market_signal"],
            ),
            mechanism_channels=[
                _channel(
                    "valuation_unbundling",
                    "portfolio_optimization",
                    "Surfaces standalone valuation signal for embedded segment.",
                    ["market.equity_window_proxy open", "segment story credible"],
                )
            ],
            lead_time_prior=_lead_time(75, 210, 480),
            execution_complexity_prior=_complexity(
                5,
                ["ipo_readiness", "carveout_financials", "market_timing"],
                "very_high",
                ["CEO", "CFO", "IR", "Legal", "Tax", "Operations"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["segment_disclosure", "market_signal", "financial_disclosure"],
                optional_supporting_classes=["management_statement"],
                must_have_features=["market.equity_window_proxy"],
                allow_heuristic_if_missing=False,
            ),
            validation_rules=[{"kind": "segment_reference_exists", "field": "segment_reference"}],
        )
    )

    actions.append(
        _action(
            action_type="portfolio",
            action_subtype="joint_venture",
            label="Joint Venture",
            description="Create joint venture to share investment/risk while accessing partner capabilities.",
            parameter_schema={
                "partner_reference": {"type": "entity_reference", "required": True},
                "ownership_split": _range_field(required=False, unit="percent"),
                "capital_commitment_usd": _numeric_field(required=False, unit="USD", minimum=0.0),
                "scope": _enum_field(["market_entry", "asset_development", "technology", "manufacturing"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["strategic.intent_vector"],
                required_evidence=["management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "risk_sharing",
                    "risk_reduction",
                    "Shares capital intensity and operational risk with partner.",
                    ["capital_constraints or capability_gap"],
                ),
                _channel(
                    "capability_access",
                    "growth_substitution",
                    "Accelerates entry by leveraging partner channels/assets.",
                    ["strategic.intent.pursue_mna_priority moderate_or_low"],
                ),
            ],
            lead_time_prior=_lead_time(45, 150, 365),
            execution_complexity_prior=_complexity(
                4,
                ["partner_selection", "governance_structuring", "contractual_complexity"],
                "high",
                ["CEO", "CFO", "Business Units", "Legal"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["management_statement", "financial_disclosure"],
                optional_supporting_classes=["segment_disclosure", "peer_context_signal"],
                must_have_features=["strategic.intent.pursue_mna_priority"],
            ),
            validation_rules=[],
        )
    )

    # 4.2.4 Restructuring / Operating Actions
    actions.append(
        _action(
            action_type="restructuring",
            action_subtype="cost_program",
            label="Cost Program",
            description="Enterprise cost reduction and efficiency program targeting structural savings.",
            parameter_schema={
                "annualized_savings_target_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "one_time_charge_usd": _numeric_field(required=False, unit="USD", minimum=0.0),
                "implementation_horizon_months": _numeric_field(required=False, unit="months", minimum=0.0),
                "affected_cost_buckets": _enum_field(["sg&a", "cogs", "shared_services", "mixed"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[{"feature": "operating.margin_percentile", "operator": "<", "value": 50}],
                required_features=[
                    "operating.ebitda_margin_ttm",
                    "operating.margin_volatility_8q",
                ],
                required_evidence=["management_statement", "financial_disclosure"],
            ),
            mechanism_channels=[
                _channel(
                    "margin_restoration",
                    "cost_efficiency",
                    "Raises operating margin and free-cash conversion.",
                    ["operating.ebitda_margin_ttm below_peer_band"],
                ),
                _channel(
                    "cashflow_stabilization",
                    "risk_reduction",
                    "Improves cash generation resilience under soft demand.",
                    ["operating.margin_volatility_8q elevated"],
                ),
            ],
            lead_time_prior=_lead_time(14, 90, 240),
            execution_complexity_prior=_complexity(
                3,
                ["cross_function_process_change", "execution_tracking", "stakeholder_management"],
                "medium",
                ["CEO", "CFO", "HR", "Operations"],
            ),
            dependency_rules=[
                _rule(
                    "unlocks",
                    "capital_return.open_market_buyback",
                    "operating.fcf_conversion improves",
                    "soft",
                    "Cost actions can create later capacity for capital return.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "management_statement"],
                optional_supporting_classes=["peer_context_signal"],
                must_have_features=["operating.ebitda_margin_ttm"],
            ),
            validation_rules=[{"kind": "positive", "field": "annualized_savings_target_usd"}],
        )
    )

    actions.append(
        _action(
            action_type="restructuring",
            action_subtype="workforce_reduction",
            label="Workforce Reduction",
            description="Targeted workforce reduction to lower cost base.",
            parameter_schema={
                "employee_pct_reduction": _percent_field(required=True, minimum=0.0, maximum=0.5),
                "severance_cost_usd": _numeric_field(required=False, unit="USD", minimum=0.0),
                "scope": _enum_field(["global", "regional", "function_specific"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["operating.ebitda_margin_ttm"],
                required_evidence=["management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "labor_cost_reset",
                    "cost_efficiency",
                    "Reduces recurring personnel expense.",
                    ["margin pressure"],
                )
            ],
            lead_time_prior=_lead_time(10, 60, 180),
            execution_complexity_prior=_complexity(
                3,
                ["hr_execution", "legal_and_compliance", "change_management"],
                "medium",
                ["CEO", "HR", "Legal", "Business Units"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["management_statement", "financial_disclosure"],
                optional_supporting_classes=["recent_action_history"],
                must_have_features=["operating.ebitda_margin_ttm"],
            ),
            validation_rules=[{"kind": "positive", "field": "employee_pct_reduction"}],
        )
    )

    actions.append(
        _action(
            action_type="restructuring",
            action_subtype="footprint_optimization",
            label="Footprint Optimization",
            description="Optimize plant/office/network footprint to reduce fixed costs and improve utilization.",
            parameter_schema={
                "site_count_affected": _numeric_field(required=False, minimum=0.0),
                "annualized_savings_target_usd": _numeric_field(required=False, unit="USD", minimum=0.0),
                "scope": _enum_field(["plants", "offices", "distribution_network", "mixed"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["operating.margin_volatility_8q"],
                required_evidence=["management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "fixed_cost_rationalization",
                    "cost_efficiency",
                    "Removes underutilized footprint and improves efficiency.",
                    ["utilization_low", "operating.margin_volatility_8q elevated"],
                )
            ],
            lead_time_prior=_lead_time(30, 120, 300),
            execution_complexity_prior=_complexity(
                4,
                ["site_transition", "capex_reconfiguration", "labor_and_regulatory"],
                "high",
                ["Operations", "HR", "Legal", "CFO"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["management_statement", "financial_disclosure"],
                optional_supporting_classes=["segment_disclosure"],
                must_have_features=["operating.margin_volatility_8q"],
            ),
            validation_rules=[],
        )
    )

    actions.append(
        _action(
            action_type="restructuring",
            action_subtype="working_capital_program",
            label="Working Capital Program",
            description="Release cash from receivables/inventory/payables optimization.",
            parameter_schema={
                "cash_release_target_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "horizon_months": _numeric_field(required=False, unit="months", minimum=0.0),
                "focus_area": _enum_field(["receivables", "inventory", "payables", "mixed"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["liquidity.runway_months", "operating.fcf_conversion"],
                required_evidence=["financial_disclosure"],
            ),
            mechanism_channels=[
                _channel(
                    "cash_release",
                    "risk_reduction",
                    "Improves liquidity without external financing.",
                    ["liquidity.runway_months constrained"],
                )
            ],
            lead_time_prior=_lead_time(5, 45, 120),
            execution_complexity_prior=_complexity(
                2,
                ["cross_function_process_discipline", "kpi_tracking"],
                "medium",
                ["CFO", "Treasury", "Operations", "Procurement"],
            ),
            dependency_rules=[
                _rule(
                    "unlocks",
                    "capital_structure.refinancing",
                    "cash_release_target_usd material",
                    "soft",
                    "Near-term liquidity improvement can support cleaner refinancing path.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure"],
                optional_supporting_classes=["management_statement"],
                must_have_features=["liquidity.runway_months"],
            ),
            validation_rules=[{"kind": "positive", "field": "cash_release_target_usd"}],
        )
    )

    actions.append(
        _action(
            action_type="restructuring",
            action_subtype="asset_impairment_or_write_down",
            label="Asset Impairment / Write-down",
            description="Recognize lower asset value to reset balance sheet and strategic baseline.",
            parameter_schema={
                "impairment_amount_usd": _numeric_field(required=True, unit="USD", minimum=0.0),
                "asset_scope": _enum_field(["goodwill", "intangibles", "fixed_assets", "mixed"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["operating.margin_trend_8q"],
                required_evidence=["financial_disclosure", "management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "balance_sheet_reset",
                    "risk_reduction",
                    "Cleans up asset base and may improve strategic transparency.",
                    ["persistent underperformance", "asset carrying value elevated"],
                )
            ],
            lead_time_prior=_lead_time(7, 45, 120),
            execution_complexity_prior=_complexity(
                2,
                ["accounting_assessment", "audit_alignment", "communication"],
                "medium",
                ["CFO", "Controller", "Audit Committee", "IR"],
            ),
            dependency_rules=[
                _rule(
                    "preferred_after",
                    "restructuring.cost_program",
                    None,
                    "soft",
                    "Impairment often accompanies broader restructuring/reset narrative.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure"],
                optional_supporting_classes=["management_statement"],
                must_have_features=["operating.ebitda_margin_ttm"],
            ),
            validation_rules=[{"kind": "positive", "field": "impairment_amount_usd"}],
        )
    )

    actions.append(
        _action(
            action_type="restructuring",
            action_subtype="chapter_pathway",
            label="Chapter Pathway",
            description="Court-supervised restructuring pathway for severe distress scenarios.",
            parameter_schema={
                "filing_window": _date_window(required=False),
                "debtor_in_possession_needed": {"type": "boolean", "required": False},
                "target_exit_horizon_months": _numeric_field(required=False, unit="months", minimum=0.0),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[
                    {"feature": "liquidity.runway_months", "operator": "<", "value": 6},
                    {"feature": "capital_structure.interest_coverage", "operator": "<", "value": 1.0},
                ],
                required_features=["liquidity.runway_months", "capital_structure.interest_coverage"],
                required_evidence=["financial_disclosure", "rating_disclosure"],
                forbidden_constraints=["out_of_court_only"],
            ),
            mechanism_channels=[
                _channel(
                    "comprehensive_balance_sheet_reset",
                    "risk_reduction",
                    "Provides legal framework for broad liability restructuring under severe distress.",
                    ["liquidity.runway_months very_low", "capital_structure.interest_coverage weak"],
                )
            ],
            lead_time_prior=_lead_time(1, 30, 120),
            execution_complexity_prior=_complexity(
                5,
                ["court_process", "stakeholder_litigation", "operational_disruption_risk"],
                "very_high",
                ["Board", "CEO", "CFO", "Legal", "Restructuring Advisors"],
            ),
            dependency_rules=[
                _rule(
                    "conflicts_with",
                    "capital_return.open_market_buyback",
                    None,
                    "hard",
                    "Capital return is incompatible with chapter pathway context.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "rating_disclosure", "liquidity_disclosure"],
                optional_supporting_classes=["management_statement"],
                must_have_features=["liquidity.runway_months", "capital_structure.interest_coverage"],
                allow_heuristic_if_missing=False,
            ),
            validation_rules=[],
        )
    )

    actions.append(
        _action(
            action_type="restructuring",
            action_subtype="out_of_court_restructuring",
            label="Out-of-Court Restructuring",
            description="Consensual restructuring path outside court process.",
            parameter_schema={
                "target_instruments": {"type": "entity_reference", "required": False},
                "expected_participation_pct": _percent_field(required=False, minimum=0.0, maximum=1.0),
                "liquidity_bridge_needed": {"type": "boolean", "required": False},
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[{"feature": "liquidity.runway_months", "operator": "<", "value": 12}],
                required_features=["liquidity.runway_months", "capital_structure.maturity_wall_ratio_24m"],
                required_evidence=["financial_disclosure", "management_statement", "rating_disclosure"],
            ),
            mechanism_channels=[
                _channel(
                    "distress_avoidance",
                    "risk_reduction",
                    "Can de-risk capital structure while avoiding full court process.",
                    ["stakeholder_alignment achievable"],
                    ["participation risk high"],
                )
            ],
            lead_time_prior=_lead_time(14, 75, 210),
            execution_complexity_prior=_complexity(
                5,
                ["creditor_coordination", "legal_negotiation", "execution_timing_risk"],
                "very_high",
                ["CFO", "Legal", "Treasury", "Advisors"],
            ),
            dependency_rules=[
                _rule(
                    "discouraged_with",
                    "capital_return.dividend_increase",
                    None,
                    "hard",
                    "Distress workout is incompatible with upward recurring payout commitment.",
                )
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["financial_disclosure", "rating_disclosure", "liquidity_disclosure"],
                optional_supporting_classes=["management_statement"],
                must_have_features=["liquidity.runway_months"],
                allow_heuristic_if_missing=False,
            ),
            validation_rules=[],
        )
    )

    # 4.2.5 Governance / Defensive Actions
    actions.append(
        _action(
            action_type="governance",
            action_subtype="board_refresh",
            label="Board Refresh",
            description="Refresh board composition to strengthen oversight and strategic credibility.",
            parameter_schema={
                "seats_to_refresh": _numeric_field(required=True, minimum=1.0),
                "skills_focus": _enum_field(["capital_allocation", "operations", "technology", "mixed"], required=False),
                "timeline_months": _numeric_field(required=False, unit="months", minimum=0.0),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["ownership_governance.activist_signal"],
                required_evidence=["management_statement", "recent_action_history"],
            ),
            mechanism_channels=[
                _channel(
                    "credibility_reset",
                    "signaling",
                    "Signals accountability and strategic reset to investors.",
                    ["ownership_governance.activist_signal elevated"],
                )
            ],
            lead_time_prior=_lead_time(14, 90, 240),
            execution_complexity_prior=_complexity(
                3,
                ["nomination_process", "governance_design", "stakeholder_management"],
                "medium",
                ["Board", "Nominating Committee", "CEO", "Legal"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["management_statement", "recent_action_history"],
                optional_supporting_classes=["peer_context_signal"],
                must_have_features=["ownership_governance.activist_signal"],
            ),
            validation_rules=[{"kind": "positive", "field": "seats_to_refresh"}],
        )
    )

    actions.append(
        _action(
            action_type="governance",
            action_subtype="activist_settlement",
            label="Activist Settlement",
            description="Negotiate settlement with activist to reduce proxy uncertainty and align strategic path.",
            parameter_schema={
                "board_seats_granted": _numeric_field(required=False, minimum=0.0),
                "policy_commitments": _enum_field(["capital_return", "cost_program", "portfolio_review", "mixed"], required=False),
                "standstill_months": _numeric_field(required=False, unit="months", minimum=0.0),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[{"feature": "ownership_governance.activist_signal", "operator": ">", "value": 0.5}],
                required_features=["ownership_governance.activist_signal"],
                required_evidence=["recent_action_history", "management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "uncertainty_reduction",
                    "risk_reduction",
                    "Reduces proxy fight uncertainty and governance overhang.",
                    ["ownership_governance.activist_signal elevated"],
                )
            ],
            lead_time_prior=_lead_time(7, 45, 150),
            execution_complexity_prior=_complexity(
                4,
                ["multi_party_negotiation", "governance_commitments", "public_communication"],
                "high",
                ["Board", "CEO", "Legal", "IR"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["recent_action_history", "management_statement"],
                optional_supporting_classes=["financial_disclosure"],
                must_have_features=["ownership_governance.activist_signal"],
            ),
            validation_rules=[],
        )
    )

    actions.append(
        _action(
            action_type="governance",
            action_subtype="poison_pill_or_defensive_action",
            label="Poison Pill / Defensive Action",
            description="Adopt takeover defense mechanism under hostile bid or control threat context.",
            parameter_schema={
                "trigger_threshold_pct": _percent_field(required=True, minimum=0.05, maximum=0.3),
                "duration_months": _numeric_field(required=False, unit="months", minimum=0.0),
                "defense_type": _enum_field(["poison_pill", "staggered_board", "other"], required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["ownership_governance.activist_signal"],
                required_evidence=["recent_action_history", "management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "control_optionality_protection",
                    "optionality_preservation",
                    "Preserves board negotiating leverage under hostile pressure.",
                    ["takeover_or_control_threat present"],
                )
            ],
            lead_time_prior=_lead_time(1, 14, 45),
            execution_complexity_prior=_complexity(
                4,
                ["legal_review", "board_vote", "high_investor_sensitivity"],
                "high",
                ["Board", "Legal", "IR"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["recent_action_history", "management_statement"],
                optional_supporting_classes=["market_signal"],
                must_have_features=[],
            ),
            validation_rules=[{"kind": "positive", "field": "trigger_threshold_pct"}],
        )
    )

    actions.append(
        _action(
            action_type="governance",
            action_subtype="ceo_transition",
            label="CEO Transition",
            description="Transition CEO to reset execution and strategic credibility.",
            parameter_schema={
                "transition_type": _enum_field(["planned", "accelerated", "interim"], required=True),
                "effective_date_window": _date_window(required=False),
                "internal_successor_flag": {"type": "boolean", "required": False},
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["ownership_governance.activist_signal", "operating.margin_trend_8q"],
                required_evidence=["management_statement", "recent_action_history"],
            ),
            mechanism_channels=[
                _channel(
                    "leadership_reset",
                    "signaling",
                    "Signals accountability and potential strategic reset.",
                    ["persistent underperformance", "activism risk elevated"],
                )
            ],
            lead_time_prior=_lead_time(14, 90, 240),
            execution_complexity_prior=_complexity(
                4,
                ["succession_planning", "organizational_stability", "stakeholder_communication"],
                "high",
                ["Board", "CEO Office", "HR", "IR"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["management_statement", "recent_action_history"],
                optional_supporting_classes=["financial_disclosure"],
                must_have_features=["ownership_governance.activist_signal"],
            ),
            validation_rules=[],
        )
    )

    actions.append(
        _action(
            action_type="governance",
            action_subtype="capital_allocation_policy_reset",
            label="Capital Allocation Policy Reset",
            description="Reset policy framework across leverage, returns, and M&A hurdle discipline.",
            parameter_schema={
                "target_leverage_band": _range_field(required=True, unit="x"),
                "capital_return_policy": _enum_field(["opportunistic_buyback", "dividend_focus", "balanced"], required=False),
                "mna_hurdle_policy": _enum_field(["strict", "moderate", "flexible"], required=False),
                "dividend_floor_flag": {"type": "boolean", "required": False},
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["capital_structure.net_leverage", "strategic.constraint_set"],
                required_evidence=["capital_policy_statement", "management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "policy_clarity",
                    "signaling",
                    "Improves investor understanding of allocation priorities and guardrails.",
                    ["strategic.intent_vector available"],
                ),
                _channel(
                    "discipline_framework",
                    "optionality_preservation",
                    "Creates internal discipline to sequence competing actions coherently.",
                    ["constraint_set present"],
                ),
            ],
            lead_time_prior=_lead_time(14, 60, 180),
            execution_complexity_prior=_complexity(
                3,
                ["board_alignment", "policy_authoring", "investor_communication"],
                "medium",
                ["Board", "CFO", "IR"],
            ),
            dependency_rules=[
                _rule(
                    "unlocks",
                    "capital_return.open_market_buyback",
                    "capital_return_policy == opportunistic_buyback",
                    "soft",
                    "Policy reset can enable explicit buyback path under defined guardrails.",
                ),
                _rule(
                    "unlocks",
                    "mna.tuck_in_acquisition",
                    "mna_hurdle_policy in ['strict','moderate']",
                    "soft",
                    "Policy reset clarifies acceptable M&A discipline.",
                ),
            ],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["capital_policy_statement", "management_statement", "financial_disclosure"],
                optional_supporting_classes=["market_signal", "peer_context_signal"],
                must_have_features=["capital_structure.net_leverage"],
            ),
            validation_rules=[],
        )
    )

    actions.append(
        _action(
            action_type="governance",
            action_subtype="stock_split",
            label="Forward Stock Split",
            description="Increase share count via a forward split to improve trading range, liquidity, and retail accessibility without changing enterprise value.",
            parameter_schema={
                "split_ratio": _numeric_field(required=True, minimum=2.0),
                "stated_goal": _enum_field(
                    ["trading_liquidity", "retail_access", "employee_equity_access", "mixed"],
                    required=False,
                ),
                "effective_date_window": _date_window(required=False),
            },
            feasibility_prerequisites=_prerequisites(
                state_conditions=[],
                required_features=["market.market_cap"],
                required_evidence=["management_statement"],
            ),
            mechanism_channels=[
                _channel(
                    "trading_range_optimization",
                    "value_creation",
                    "A lower absolute share price can improve trading accessibility and retail participation.",
                    ["market.market_cap sizable_enough_for_split_signal"],
                ),
                _channel(
                    "confidence_signal",
                    "signaling",
                    "Management can use a forward split to reinforce confidence in sustained share-price strength.",
                    ["recent_price_appreciation strong"],
                ),
            ],
            lead_time_prior=_lead_time(7, 30, 90),
            execution_complexity_prior=_complexity(
                2,
                ["requires_board_approval", "requires_exchange_notice", "requires_record_and_effective_dates"],
                "low_to_medium",
                ["Board", "Legal", "IR"],
            ),
            dependency_rules=[],
            minimum_evidence_requirements=_evidence(
                minimum_classes_required=["management_statement"],
                optional_supporting_classes=["market_signal", "financial_disclosure"],
                must_have_features=["market.market_cap"],
                allow_heuristic_if_missing=True,
            ),
            validation_rules=[{"kind": "positive", "field": "split_ratio"}],
        )
    )

    return actions


def build_default_action_schema_registry(version: str = "v1.0") -> ActionSchemaRegistry:
    return ActionSchemaRegistry(version=version, actions=_default_actions(), last_updated_at=_utc_now_iso())
