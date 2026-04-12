"""Compiler-side preparation of runtime autodiff facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...ir.nodes import BindingIR, BuiltinCallIR, IdentifierIR, IRNode
from ...shared.autodiff_intrinsics import AutodiffBuiltinKind, autodiff_builtin_kind
from ...shared.defid import DefId


AutodiffCompiledFacts = Dict[str, Any]


@dataclass(frozen=True)
class AutodiffBuiltinRequest:
    kind: AutodiffBuiltinKind
    target_defids: tuple[DefId, ...]
    target_names: tuple[str, ...]


def collect_autodiff_builtin_requests(node: Any) -> Dict[int, AutodiffBuiltinRequest]:
    out: Dict[int, AutodiffBuiltinRequest] = {}
    seen: set[int] = set()
    stack: list[Any] = [node]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(cur, BuiltinCallIR):
            kind = autodiff_builtin_kind(getattr(cur, "defid", None))
            if kind is not None:
                target_defids = []
                target_names = []
                for arg in cur.args or ():
                    if not isinstance(arg, IdentifierIR) or arg.defid is None:
                        continue
                    target_defids.append(arg.defid)
                    target_names.append(arg.name or "?")
                out[oid] = AutodiffBuiltinRequest(
                    kind=kind,
                    target_defids=tuple(target_defids),
                    target_names=tuple(target_names),
                )
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


def binding_for_defid(compiled_facts: AutodiffCompiledFacts, defid: DefId) -> Optional[BindingIR]:
    bindings = compiled_facts.get("bindings") or {}
    functions = compiled_facts.get("functions") or {}
    return bindings.get(defid) or functions.get(defid)


def autodiff_builtin_request(
    compiled_facts: AutodiffCompiledFacts,
    expr: BuiltinCallIR,
) -> Optional[AutodiffBuiltinRequest]:
    requests = compiled_facts.get("builtin_requests_by_expr_id") or {}
    return requests.get(id(expr))


def compile_autodiff_graph(analysis: Dict[str, Any]) -> AutodiffCompiledFacts:
    bindings = dict(analysis.get("graph_binding_by_defid") or {})
    functions = dict(analysis.get("graph_function_ir_map") or {})
    local_contexts_by_defid = dict(analysis.get("graph_local_contexts_by_defid") or {})
    runtime_jvp_templates_by_pair = dict(analysis.get("runtime_jvp_templates_by_pair") or {})
    runtime_vjp_templates_by_target = dict(analysis.get("runtime_vjp_templates_by_target") or {})
    builtin_requests_by_expr_id = dict(analysis.get("graph_builtin_requests_by_expr_id") or {})
    return {
        "bindings": bindings,
        "functions": functions,
        "local_contexts_by_defid": local_contexts_by_defid,
        "runtime_jvp_templates_by_pair": runtime_jvp_templates_by_pair,
        "runtime_vjp_templates_by_target": runtime_vjp_templates_by_target,
        "leaf_defids": set(analysis.get("graph_leaf_defids") or set()),
        "self_recursive_defids": set(analysis.get("graph_self_recursive_defids") or set()),
        "builtin_requests_by_expr_id": builtin_requests_by_expr_id,
    }
