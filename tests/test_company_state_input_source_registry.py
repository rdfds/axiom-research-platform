import ast
import json
from pathlib import Path

from src.company_state_input_source_registry import CompanyStateInputSourceRegistry


REPO_ROOT = Path(".")
COMPANY_STATE_BUILDER = REPO_ROOT / "src" / "company_state_builder.py"
REGISTRY_PATH = REPO_ROOT / "configs" / "metric_methodologies" / "company_state_input_source_registry_v1.json"


def _static_feature_names() -> set[str]:
    tree = ast.parse(COMPANY_STATE_BUILDER.read_text())
    names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "features":
                    if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                        names.add(target.slice.value)
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "_emit_metric_views":
                for kw in node.keywords:
                    if kw.arg == "base_name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        names.add(kw.value.value)
            self.generic_visit(node)

    Visitor().visit(tree)
    return names


