"""Autodiff pass: collect requests and rewrite to runtime JVP/VJP builtins.

This is the phase-1 Einlang implementation of ``docs/AUTODIFF_VJP_JVP_REWRITE.md``.
The pass no longer synthesizes derivative IR. Instead it:

- snapshots the typed high-level binding graph for runtime autodiff
- rewrites ``@x`` and ``@y / @x`` to internal builtin calls
- rewrites direct ``print(@...)`` to symbolic JVP/Jacobian print builtins
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from ..base import BasePass, TyCtxt
from ..extremum_selection_canonicalization import ExtremumSelectionCanonicalizationPass
from ..pre_autodiff_pruning import PreAutodiffPruningPass
from ..shape_analysis import UnifiedShapeAnalysisPass
from ..type_inference import TypeInferencePass
from ...autodiff import (
    AutodiffBuiltinKind,
    autodiff_builtin_defid,
    autodiff_builtin_name,
)
from ...ir.nodes import (
    BindingIR,
    BinaryOpIR,
    BlockExpressionIR,
    BuiltinCallIR,
    DiffRuleIR,
    DifferentialIR,
    ExpressionIR,
    FunctionValueIR,
    IRNode,
    IdentifierIR,
    IndexRestIR,
    IndexVarIR,
    ProgramIR,
    is_function_binding,
)
from ...shared.defid import DefId, fixed_builtin_defid
from ...shared.source_location import SourceLocation
from ...shared.types import BinaryOp, STR, Type, strip_differential_types_deep


DIFF_PREFIX = "_@"
USER_DIFF_PREFIX = "@"
_LOC0 = SourceLocation("", 0, 0)
_PRINT_BUILTIN_DEFID = fixed_builtin_defid("print")


def _ti(node: Any) -> Any:
    return getattr(node, "type_info", None)


def _si(node: Any) -> Any:
    return getattr(node, "shape_info", None)


def _builtin_defid(node: BuiltinCallIR) -> Optional[DefId]:
    did = getattr(node, "defid", None)
    if isinstance(did, DefId):
        return did
    return fixed_builtin_defid(node.builtin_name)


@dataclass
class _AutodiffTargets:
    diff_targets: List[Tuple[DefId, str]] = field(default_factory=list)
    standalone_diff_target_defids: Set[DefId] = field(default_factory=set)
    quotient_pairs: List[Tuple[DefId, DefId]] = field(default_factory=list)
    source_quotient_slots: Dict[DefId, Tuple[DefId, DefId]] = field(default_factory=dict)


class _DependencyQueryCache:
    def __init__(self, bindings: Dict[DefId, BindingIR]) -> None:
        self.bindings = bindings
        self._expr_cache: Dict[int, Set[DefId]] = {}

    def collect_defids(self, expr: Optional[ExpressionIR]) -> Set[DefId]:
        if expr is None:
            return set()
        key = id(expr)
        cached = self._expr_cache.get(key)
        if cached is not None:
            return set(cached)
        out = _collect_defids(expr)
        self._expr_cache[key] = set(out)
        return out


@dataclass
class _AutodiffBindingContext:
    bindings: Dict[DefId, BindingIR]
    dep_cache: _DependencyQueryCache
    binding_deps: Dict[DefId, Set[DefId]] = field(default_factory=dict)

    def deps_for(self, did: DefId) -> Set[DefId]:
        cached = self.binding_deps.get(did)
        if cached is not None:
            return cached
        binding = self.bindings.get(did)
        if binding is None or is_function_binding(binding):
            deps: Set[DefId] = set()
        else:
            deps = _autodiff_primal_data_defids(binding.expr, self.bindings, self.dep_cache)
        self.binding_deps[did] = deps
        return deps


@lru_cache(maxsize=None)
def _slot_names(cls: type) -> Tuple[str, ...]:
    out: List[str] = []
    for base in cls.__mro__:
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            out.append(slots)
        else:
            out.extend(slots)
    seen = set()
    ordered: List[str] = []
    for name in out:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


_SHARED_CLONE_SLOTS = frozenset(
    {
        "location",
        "type_info",
        "shape_info",
        "defid",
        "name",
        "member",
        "builtin_name",
        "operator",
        "module_path",
        "return_type",
        "_is_partially_specialized",
        "_generic_defid",
        "use_argmin",
        "inclusive",
    }
)


def _clone_ir_value(value: Any, memo: Dict[int, Any]) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    oid = id(value)
    if oid in memo:
        return memo[oid]
    if isinstance(value, tuple):
        cloned = tuple(_clone_ir_value(v, memo) for v in value)
        memo[oid] = cloned
        return cloned
    if isinstance(value, list):
        cloned = [_clone_ir_value(v, memo) for v in value]
        memo[oid] = cloned
        return cloned
    if isinstance(value, dict):
        cloned = {_clone_ir_value(k, memo): _clone_ir_value(v, memo) for k, v in value.items()}
        memo[oid] = cloned
        return cloned
    if hasattr(value, "__class__") and any(hasattr(c, "__slots__") for c in value.__class__.__mro__):
        cloned = value.__class__.__new__(value.__class__)
        memo[oid] = cloned
        for slot in _slot_names(value.__class__):
            current = getattr(value, slot, None)
            setattr(
                cloned,
                slot,
                current if slot in _SHARED_CLONE_SLOTS else _clone_ir_value(current, memo),
            )
        return cloned
    return value


def clear_custom_diff_body_everywhere(program: ProgramIR) -> None:
    seen: Set[int] = set()
    stack: List[Any] = [program]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(cur, FunctionValueIR):
            cur.custom_diff_body = None
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


def clear_autodiff_only_ir(obj: Any) -> None:
    if obj is None:
        return
    seen: Set[int] = set()
    stack: List[Any] = [obj]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(cur, ProgramIR):
            new_statements = [stmt for stmt in (cur.statements or []) if not isinstance(stmt, DiffRuleIR)]
            cur.statements = new_statements
            cur.bindings = [stmt for stmt in new_statements if isinstance(stmt, BindingIR)]
        if hasattr(cur, "type_info"):
            try:
                ty = getattr(cur, "type_info", None)
                if isinstance(ty, Type):
                    setattr(cur, "type_info", strip_differential_types_deep(ty))
            except Exception:
                pass
        if isinstance(cur, FunctionValueIR):
            cur.custom_diff_body = None
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
                    if slot == "location":
                        continue
                    stack.append(getattr(cur, slot, None))


def _collect_all_defids_ir(node: object) -> Set[DefId]:
    out: Set[DefId] = set()
    seen: Set[int] = set()
    stack: List[object] = [node]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(cur, (IdentifierIR, IndexVarIR, IndexRestIR)):
            did = getattr(cur, "defid", None)
            if did is not None:
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


def _collect_defids(expr: Optional[ExpressionIR]) -> Set[DefId]:
    return _collect_all_defids_ir(expr)


def _bindings_in(node: Any, _program: Optional[ProgramIR] = None) -> List[BindingIR]:
    out: List[BindingIR] = []
    seen: Set[int] = set()
    stack: List[Any] = [node]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(cur, BindingIR):
            out.append(cur)
        if isinstance(cur, dict):
            stack.extend(cur.keys())
            stack.extend(cur.values())
            continue
        if isinstance(cur, (list, tuple)):
            stack.extend(reversed(cur))
            continue
        if isinstance(cur, IRNode):
            for cls in reversed(type(cur).__mro__):
                for slot in getattr(cls, "__slots__", ()):
                    stack.append(getattr(cur, slot, None))
    return out


def _collect_targets_expr(expr: Optional[ExpressionIR]) -> Tuple[List[Tuple[DefId, str]], List[Tuple[DefId, DefId]]]:
    diff_targets: List[Tuple[DefId, str]] = []
    quotient_pairs: List[Tuple[DefId, DefId]] = []
    seen: Set[int] = set()
    stack: List[Any] = [expr]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(cur, DifferentialIR):
            operand = cur.operand
            if isinstance(operand, IdentifierIR) and operand.defid is not None:
                diff_targets.append((operand.defid, operand.name or ""))
            stack.append(operand)
            continue
        if isinstance(cur, BinaryOpIR):
            pair = _source_requested_quotient_pair(cur)
            if pair is not None:
                quotient_pairs.append(pair)
            stack.append(cur.left)
            stack.append(cur.right)
            continue
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
    return diff_targets, quotient_pairs


def _collect_targets(node: Any) -> Tuple[List[Tuple[DefId, str]], List[Tuple[DefId, DefId]]]:
    if isinstance(node, ProgramIR):
        diff_targets: List[Tuple[DefId, str]] = []
        quotient_pairs: List[Tuple[DefId, DefId]] = []
        for stmt in node.statements or []:
            if isinstance(stmt, BindingIR) and stmt.expr is not None:
                dt, qp = _collect_targets_expr(stmt.expr)
                diff_targets.extend(dt)
                quotient_pairs.extend(qp)
            elif isinstance(stmt, ExpressionIR):
                dt, qp = _collect_targets_expr(stmt)
                diff_targets.extend(dt)
                quotient_pairs.extend(qp)
        return diff_targets, quotient_pairs
    return _collect_targets_expr(node if isinstance(node, ExpressionIR) else None)


def _autodiff_primal_data_defids(
    expr: Optional[ExpressionIR],
    bindings: Dict[DefId, BindingIR],
    dep_cache: Optional[_DependencyQueryCache] = None,
) -> Set[DefId]:
    if expr is None:
        return set()
    defs = dep_cache.collect_defids(expr) if dep_cache is not None else _collect_defids(expr)
    return {did for did in defs if did in bindings}


def _pending_differential_target(expr: ExpressionIR) -> Optional[DefId]:
    if isinstance(expr, DifferentialIR):
        operand = expr.operand
        if isinstance(operand, IdentifierIR) and operand.defid is not None:
            return operand.defid
    return None


def _source_requested_quotient_pair(expr: ExpressionIR) -> Optional[Tuple[DefId, DefId]]:
    if not isinstance(expr, BinaryOpIR) or expr.operator != BinaryOp.DIV:
        return None
    left_target = _pending_differential_target(expr.left)
    right_target = _pending_differential_target(expr.right)
    if left_target is None or right_target is None:
        return None
    return left_target, right_target


def _collect_standalone_diff_targets_expr(expr: ExpressionIR) -> Set[DefId]:
    out: Set[DefId] = set()
    seen: Set[int] = set()
    stack: List[Any] = [expr]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if (
            isinstance(cur, BinaryOpIR)
            and cur.operator == BinaryOp.DIV
            and isinstance(cur.left, DifferentialIR)
            and isinstance(cur.right, DifferentialIR)
        ):
            continue
        if isinstance(cur, DifferentialIR):
            operand = cur.operand
            if isinstance(operand, IdentifierIR) and operand.defid is not None:
                out.add(operand.defid)
            stack.append(operand)
            continue
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


class AutodiffPass(BasePass):
    requires = [
        TypeInferencePass,
        UnifiedShapeAnalysisPass,
        ExtremumSelectionCanonicalizationPass,
        PreAutodiffPruningPass,
    ]

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        try:
            return self._core(ir, tcx)
        finally:
            clear_autodiff_only_ir(ir)
            clear_autodiff_only_ir(getattr(tcx, "function_ir_map", None))
            try:
                analysis = tcx.get_analysis(AutodiffPass)
            except RuntimeError:
                analysis = None
            if isinstance(analysis, dict):
                clear_autodiff_only_ir(analysis.get("diff_block"))

    def _core(self, program: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        bindings = _bindings_in(program, program) or []
        if not bindings:
            self._record_empty_analysis(tcx, [])
            return program

        targets = self._collect_requested_targets(program, bindings)
        binding_ctx = self._build_binding_context(bindings, tcx)
        graph_program = _clone_ir_value(program, {})
        graph_function_ir_map = _clone_ir_value(getattr(tcx, "function_ir_map", None) or {}, {})
        graph_bindings = _bindings_in(graph_program, graph_program) or []
        graph_binding_by_defid = {
            b.defid: b
            for b in graph_bindings
            if isinstance(b, BindingIR) and b.defid is not None
        }
        graph_leaf_defids = {
            did
            for did, binding in binding_ctx.bindings.items()
            if did is not None
            and isinstance(binding, BindingIR)
            and not is_function_binding(binding)
            and not binding_ctx.deps_for(did)
        }
        self._rewrite_program_for_runtime(program)
        tcx.set_analysis(
            AutodiffPass,
            {
                "diff_block": None,
                "differential_targets": set(targets.diff_targets),
                "differential_buffer_by_defid": {},
                "autodiff_differential_map": {},
                "differential_leaves": set(),
                "pending_differential_slot_by_defid": {},
                "pending_quotient_slot_by_defid": {},
                "source_quotient_slot_by_defid": targets.source_quotient_slots,
                "graph_program": graph_program,
                "graph_binding_by_defid": graph_binding_by_defid,
                "graph_function_ir_map": graph_function_ir_map,
                "graph_leaf_defids": graph_leaf_defids,
            },
        )
        return program

    def _record_empty_analysis(self, tcx: TyCtxt, diff_targets: List[Tuple[DefId, str]]) -> None:
        tcx.set_analysis(
            AutodiffPass,
            {
                "diff_block": None,
                "differential_targets": set(diff_targets),
                "differential_buffer_by_defid": {},
                "autodiff_differential_map": {},
                "differential_leaves": set(),
                "pending_differential_slot_by_defid": {},
                "pending_quotient_slot_by_defid": {},
                "source_quotient_slot_by_defid": {},
                "graph_program": None,
                "graph_binding_by_defid": {},
                "graph_function_ir_map": {},
                "graph_leaf_defids": set(),
            },
        )

    def _rewrite_program_for_runtime(self, program: ProgramIR) -> None:
        for stmt in list(program.statements or []):
            self._rewrite_runtime_node(stmt)
        program.bindings = [s for s in (program.statements or []) if isinstance(s, BindingIR)]

    def _rewrite_runtime_node(self, obj: object) -> object:
        if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
            return obj
        if isinstance(obj, DifferentialIR):
            return self._make_runtime_tangent_call(obj)
        if isinstance(obj, BinaryOpIR):
            direct_q = _source_requested_quotient_pair(obj)
            if direct_q is not None:
                return self._make_runtime_jacobian_call(obj, direct_q)
            obj.left = self._rewrite_runtime_node(obj.left)
            obj.right = self._rewrite_runtime_node(obj.right)
            return obj
        if isinstance(obj, BuiltinCallIR) and _builtin_defid(obj) == _PRINT_BUILTIN_DEFID and len(obj.args or []) == 1:
            arg0 = obj.args[0]
            if isinstance(arg0, DifferentialIR):
                obj.args = (self._make_runtime_symbolic_tangent_call(arg0),)
                return obj
            direct_q = _source_requested_quotient_pair(arg0) if isinstance(arg0, BinaryOpIR) else None
            if direct_q is not None:
                obj.args = (self._make_runtime_symbolic_jacobian_call(arg0, direct_q),)
                return obj
        if isinstance(obj, IRNode):
            for cls in type(obj).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    if slot == "location":
                        continue
                    setattr(obj, slot, self._rewrite_runtime_node(getattr(obj, slot, None)))
            return obj
        if isinstance(obj, list):
            return [self._rewrite_runtime_node(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._rewrite_runtime_node(item) for item in obj)
        if isinstance(obj, dict):
            return {self._rewrite_runtime_node(k): self._rewrite_runtime_node(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def _make_runtime_tangent_call(expr: DifferentialIR) -> BuiltinCallIR:
        operand = expr.operand
        if not isinstance(operand, IdentifierIR) or operand.defid is None:
            raise RuntimeError("Phase-1 autodiff only supports identifier differentials")
        kind = AutodiffBuiltinKind.TANGENT
        return BuiltinCallIR(
            autodiff_builtin_name(kind),
            [IdentifierIR(operand.name, operand.location, operand.defid, type_info=_ti(operand), shape_info=_si(operand))],
            expr.location or operand.location,
            defid=autodiff_builtin_defid(kind),
            type_info=_ti(expr),
            shape_info=_si(expr),
        )

    @staticmethod
    def _make_runtime_jacobian_call(expr: BinaryOpIR, pair: Tuple[DefId, DefId]) -> BuiltinCallIR:
        num_defid, den_defid = pair
        num_expr = expr.left.operand if isinstance(expr.left, DifferentialIR) else expr.left
        den_expr = expr.right.operand if isinstance(expr.right, DifferentialIR) else expr.right
        kind = AutodiffBuiltinKind.JACOBIAN
        return BuiltinCallIR(
            autodiff_builtin_name(kind),
            [
                IdentifierIR(getattr(num_expr, "name", None) or "?", getattr(num_expr, "location", expr.location), num_defid, type_info=_ti(num_expr), shape_info=_si(num_expr)),
                IdentifierIR(getattr(den_expr, "name", None) or "?", getattr(den_expr, "location", expr.location), den_defid, type_info=_ti(den_expr), shape_info=_si(den_expr)),
            ],
            expr.location,
            defid=autodiff_builtin_defid(kind),
            type_info=_ti(expr),
            shape_info=_si(expr),
        )

    @staticmethod
    def _make_runtime_symbolic_tangent_call(expr: DifferentialIR) -> BuiltinCallIR:
        operand = expr.operand
        if not isinstance(operand, IdentifierIR) or operand.defid is None:
            raise RuntimeError("Phase-1 autodiff symbolic print only supports identifier differentials")
        kind = AutodiffBuiltinKind.SYMBOLIC_TANGENT
        return BuiltinCallIR(
            autodiff_builtin_name(kind),
            [IdentifierIR(operand.name, operand.location, operand.defid, type_info=_ti(operand), shape_info=_si(operand))],
            expr.location or operand.location,
            defid=autodiff_builtin_defid(kind),
            type_info=STR,
            shape_info=None,
        )

    @staticmethod
    def _make_runtime_symbolic_jacobian_call(expr: BinaryOpIR, pair: Tuple[DefId, DefId]) -> BuiltinCallIR:
        num_defid, den_defid = pair
        num_expr = expr.left.operand if isinstance(expr.left, DifferentialIR) else expr.left
        den_expr = expr.right.operand if isinstance(expr.right, DifferentialIR) else expr.right
        kind = AutodiffBuiltinKind.SYMBOLIC_JACOBIAN
        return BuiltinCallIR(
            autodiff_builtin_name(kind),
            [
                IdentifierIR(getattr(num_expr, "name", None) or "?", getattr(num_expr, "location", expr.location), num_defid, type_info=_ti(num_expr), shape_info=_si(num_expr)),
                IdentifierIR(getattr(den_expr, "name", None) or "?", getattr(den_expr, "location", expr.location), den_defid, type_info=_ti(den_expr), shape_info=_si(den_expr)),
            ],
            expr.location,
            defid=autodiff_builtin_defid(kind),
            type_info=STR,
            shape_info=None,
        )

    def _collect_requested_targets(self, program: ProgramIR, bindings: List[BindingIR]) -> _AutodiffTargets:
        targets = _AutodiffTargets()
        for binding in bindings:
            if binding.expr is None:
                continue
            diff_targets, quotient_pairs = _collect_targets_expr(binding.expr)
            direct_q = _source_requested_quotient_pair(binding.expr)
            targets.standalone_diff_target_defids.update(_collect_standalone_diff_targets_expr(binding.expr))
            targets.diff_targets.extend(diff_targets)
            targets.quotient_pairs.extend(quotient_pairs)
            if direct_q is not None and binding.defid is not None:
                targets.source_quotient_slots[binding.defid] = direct_q
        for stmt in program.statements or []:
            if isinstance(stmt, BindingIR) or not isinstance(stmt, ExpressionIR):
                continue
            diff_targets, quotient_pairs = _collect_targets_expr(stmt)
            targets.standalone_diff_target_defids.update(_collect_standalone_diff_targets_expr(stmt))
            targets.diff_targets.extend(diff_targets)
            targets.quotient_pairs.extend(quotient_pairs)
        return targets

    def _build_binding_context(self, bindings: List[BindingIR], tcx: TyCtxt) -> _AutodiffBindingContext:
        binding_map: Dict[DefId, BindingIR] = {}
        for binding in bindings:
            if binding.defid is not None:
                binding_map[binding.defid] = binding
        function_ir_map = getattr(tcx, "function_ir_map", None) or {}
        for did, fn in function_ir_map.items():
            if did is not None and did not in binding_map and isinstance(fn, BindingIR) and is_function_binding(fn):
                binding_map[did] = fn
        return _AutodiffBindingContext(bindings=binding_map, dep_cache=_DependencyQueryCache(binding_map))


_collect_autodiff_targets = _collect_targets
