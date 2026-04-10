"""Compiler-side preparation of runtime autodiff facts."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

from ...ir.nodes import (
    BindingIR,
    BinaryOpIR,
    DifferentialIR,
    ExpressionIR,
    IdentifierIR,
    IRNode,
)
from ...shared.defid import DefId
from ...shared.types import BinaryOp


AutodiffCompiledFacts = Dict[str, Any]


def collect_runtime_defids(node: Any) -> Set[DefId]:
    out: Set[DefId] = set()
    seen: Set[int] = set()
    stack: list[Any] = [node]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        did = getattr(cur, "defid", None)
        if did is not None and isinstance(did, DefId):
            out.add(did)
        if isinstance(cur, dict):
            stack.extend(cur.keys())
            stack.extend(cur.values())
            continue
        if isinstance(cur, (list, tuple)):
            stack.extend(cur)
            continue
        if isinstance(cur, IRNode):
            for cls in type(cur).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    stack.append(getattr(cur, slot, None))
    return out


def runtime_differential_target(expr: ExpressionIR) -> Optional[DefId]:
    if isinstance(expr, DifferentialIR):
        operand = expr.operand
        if isinstance(operand, IdentifierIR) and operand.defid is not None:
            return operand.defid
    return None


def runtime_source_quotient_pair(expr: ExpressionIR) -> Optional[Tuple[DefId, DefId]]:
    if not isinstance(expr, BinaryOpIR) or expr.operator != BinaryOp.DIV:
        return None
    left = runtime_differential_target(expr.left)
    right = runtime_differential_target(expr.right)
    if left is None or right is None:
        return None
    return left, right


def binding_for_defid(compiled_facts: AutodiffCompiledFacts, defid: DefId) -> Optional[BindingIR]:
    bindings = compiled_facts.get("bindings") or {}
    functions = compiled_facts.get("functions") or {}
    return bindings.get(defid) or functions.get(defid)


def compile_autodiff_graph(analysis: Dict[str, Any]) -> AutodiffCompiledFacts:
    return {
        "bindings": dict(analysis.get("graph_binding_by_defid") or {}),
        "functions": dict(analysis.get("graph_function_ir_map") or {}),
        "leaf_defids": set(analysis.get("graph_leaf_defids") or set()),
    }
