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
from .compiler import collect_autodiff_builtin_requests
from ...shared.autodiff_intrinsics import (
    AutodiffBuiltinKind,
    autodiff_builtin_defid,
    autodiff_builtin_name,
)
from ...ir.nodes import (
    ArrayLiteralIR,
    BindingIR,
    BinaryOpIR,
    BlockExpressionIR,
    BuiltinCallIR,
    CastExpressionIR,
    DiffRuleIR,
    DifferentialIR,
    EinsteinIR,
    ExpressionIR,
    FunctionCallIR,
    FunctionValueIR,
    IRNode,
    IdentifierIR,
    IfExpressionIR,
    IndexRestIR,
    IndexVarIR,
    LiteralIR,
    MemberAccessIR,
    ProgramIR,
    RangeIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    UnaryOpIR,
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


@dataclass
class _RuntimeRewriteState:
    pending_differential_slots: Dict[DefId, DefId] = field(default_factory=dict)
    pending_quotient_slots: Dict[DefId, Tuple[DefId, DefId]] = field(default_factory=dict)
    temp_counter: int = 0


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
        if isinstance(cur, FunctionValueIR):
            stack.extend(cur.parameters or ())
            stack.append(cur.body)
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
        if isinstance(cur, FunctionValueIR):
            stack.extend(cur.parameters or ())
            stack.append(cur.body)
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
        if (
            not targets.diff_targets
            and not targets.standalone_diff_target_defids
            and not targets.quotient_pairs
            and not targets.source_quotient_slots
        ):
            self._record_empty_analysis(tcx, [])
            return program
        binding_ctx = self._build_binding_context(bindings, tcx)
        graph_program = _clone_ir_value(program, {})
        graph_function_ir_map = _clone_ir_value(getattr(tcx, "function_ir_map", None) or {}, {})
        self._rewrite_graph_for_runtime(graph_program)
        graph_function_ir_map = self._rewrite_graph_node(graph_function_ir_map)
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
        graph_self_recursive_defids = {
            did
            for did, binding in graph_binding_by_defid.items()
            if binding.expr is not None and did in _collect_defids(binding.expr)
        }
        graph_builtin_requests_by_expr_id = collect_autodiff_builtin_requests(
            {
                "program": graph_program,
                "functions": graph_function_ir_map,
            }
        )
        runtime_rewrite = _RuntimeRewriteState()
        self._rewrite_program_for_runtime(program, tcx, runtime_rewrite)
        function_ir_map_live = getattr(tcx, "function_ir_map", None) or {}
        for fn in function_ir_map_live.values():
            if isinstance(fn, BindingIR) and is_function_binding(fn):
                self._rewrite_statement(fn, getattr(tcx, "resolver", None), runtime_rewrite)
        tcx.set_analysis(
            AutodiffPass,
            {
                "diff_block": None,
                "differential_targets": set(targets.diff_targets),
                "differential_buffer_by_defid": {},
                "autodiff_differential_map": {},
                "differential_leaves": set(),
                "pending_differential_slot_by_defid": dict(runtime_rewrite.pending_differential_slots),
                "pending_quotient_slot_by_defid": dict(runtime_rewrite.pending_quotient_slots),
                "source_quotient_slot_by_defid": targets.source_quotient_slots,
                "graph_program": graph_program,
                "graph_binding_by_defid": graph_binding_by_defid,
                "graph_function_ir_map": graph_function_ir_map,
                "graph_leaf_defids": graph_leaf_defids,
                "graph_self_recursive_defids": graph_self_recursive_defids,
                "graph_builtin_requests_by_expr_id": graph_builtin_requests_by_expr_id,
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
                "graph_self_recursive_defids": set(),
                "graph_builtin_requests_by_expr_id": {},
            },
        )

    def _rewrite_graph_for_runtime(self, program: ProgramIR) -> None:
        for stmt in list(program.statements or []):
            self._rewrite_graph_node(stmt)
        program.bindings = [s for s in (program.statements or []) if isinstance(s, BindingIR)]

    def _rewrite_graph_node(self, obj: object) -> object:
        if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
            return obj
        if isinstance(obj, DifferentialIR):
            return self._make_runtime_tangent_call(obj)
        if isinstance(obj, BinaryOpIR):
            direct_q = _source_requested_quotient_pair(obj)
            if direct_q is not None:
                return self._make_runtime_jacobian_call(obj, direct_q)
            obj.left = self._rewrite_graph_node(obj.left)
            obj.right = self._rewrite_graph_node(obj.right)
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
                    setattr(obj, slot, self._rewrite_graph_node(getattr(obj, slot, None)))
            return obj
        if isinstance(obj, list):
            return [self._rewrite_graph_node(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._rewrite_graph_node(item) for item in obj)
        if isinstance(obj, dict):
            return {self._rewrite_graph_node(k): self._rewrite_graph_node(v) for k, v in obj.items()}
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

    def _rewrite_program_for_runtime(
        self,
        program: ProgramIR,
        tcx: TyCtxt,
        state: _RuntimeRewriteState,
    ) -> None:
        resolver = getattr(tcx, "resolver", None)
        if resolver is None:
            raise RuntimeError("AutodiffPass requires resolver for runtime rewrite")
        program.statements = self._rewrite_statement_list(program.statements or [], resolver, state)
        program.bindings = [s for s in (program.statements or []) if isinstance(s, BindingIR)]

    def _rewrite_statement_list(
        self,
        statements: List[Any],
        resolver: Any,
        state: _RuntimeRewriteState,
    ) -> List[Any]:
        out: List[Any] = []
        for stmt in statements:
            out.extend(self._rewrite_statement(stmt, resolver, state))
        return out

    def _rewrite_statement(
        self,
        stmt: Any,
        resolver: Any,
        state: _RuntimeRewriteState,
    ) -> List[Any]:
        if isinstance(stmt, BindingIR):
            direct_diff = _pending_differential_target(stmt.expr) if stmt.expr is not None else None
            direct_q = _source_requested_quotient_pair(stmt.expr) if stmt.expr is not None else None
            if stmt.defid is not None and direct_diff is not None:
                state.pending_differential_slots[stmt.defid] = direct_diff
                stmt.expr = LiteralIR(
                    0.0,
                    stmt.location or _LOC0,
                    type_info=_ti(stmt.expr),
                    shape_info=_si(stmt.expr),
                )
                return [stmt]
            if stmt.defid is not None and direct_q is not None:
                state.pending_quotient_slots[stmt.defid] = direct_q
                stmt.expr = LiteralIR(
                    0.0,
                    stmt.location or _LOC0,
                    type_info=_ti(stmt.expr),
                    shape_info=_si(stmt.expr),
                )
                return [stmt]
            prefixes, rewritten = self._rewrite_expr_for_program(stmt.expr, resolver, state)
            stmt.expr = rewritten
            return prefixes + [stmt]
        if isinstance(stmt, ExpressionIR):
            prefixes, rewritten = self._rewrite_expr_for_program(stmt, resolver, state)
            return prefixes + [rewritten]
        return [stmt]

    def _fresh_pending_binding(
        self,
        resolver: Any,
        state: _RuntimeRewriteState,
        *,
        name_hint: str,
        expr: ExpressionIR,
        pending_target: Optional[DefId] = None,
        pending_pair: Optional[Tuple[DefId, DefId]] = None,
    ) -> Tuple[BindingIR, IdentifierIR]:
        state.temp_counter += 1
        did = resolver.allocate_for_local()
        name = f"{DIFF_PREFIX}{name_hint}_{state.temp_counter}"
        binding = BindingIR(
            name=name,
            expr=LiteralIR(0.0, expr.location or _LOC0, type_info=_ti(expr), shape_info=_si(expr)),
            location=expr.location or _LOC0,
            defid=did,
            type_info=_ti(expr),
        )
        if pending_target is not None:
            state.pending_differential_slots[did] = pending_target
        if pending_pair is not None:
            state.pending_quotient_slots[did] = pending_pair
        ident = IdentifierIR(name, expr.location or _LOC0, did, type_info=_ti(expr), shape_info=_si(expr))
        return binding, ident

    def _symbolic_print_literal(self, expr: ExpressionIR) -> Optional[LiteralIR]:
        if isinstance(expr, DifferentialIR):
            operand = expr.operand
            if isinstance(operand, IdentifierIR):
                return LiteralIR(
                    f"@{operand.name or '?'}",
                    expr.location or operand.location or _LOC0,
                    type_info=STR,
                )
        pair = _source_requested_quotient_pair(expr)
        if pair is not None and isinstance(expr, BinaryOpIR):
            num_expr = expr.left.operand if isinstance(expr.left, DifferentialIR) else expr.left
            den_expr = expr.right.operand if isinstance(expr.right, DifferentialIR) else expr.right
            num_name = getattr(num_expr, "name", None) or "y"
            den_name = getattr(den_expr, "name", None) or "x"
            return LiteralIR(
                f"(@{num_name} / @{den_name}) · @{den_name}",
                expr.location or _LOC0,
                type_info=STR,
            )
        return None

    def _rewrite_expr_for_program(
        self,
        expr: Optional[ExpressionIR],
        resolver: Any,
        state: _RuntimeRewriteState,
    ) -> Tuple[List[BindingIR], Optional[ExpressionIR]]:
        if expr is None:
            return [], None
        if isinstance(expr, DifferentialIR):
            target = _pending_differential_target(expr)
            if target is None:
                raise RuntimeError("AutodiffPass currently only supports identifier differentials in executable IR")
            operand = expr.operand
            name_hint = getattr(operand, "name", None) or "d"
            binding, ident = self._fresh_pending_binding(
                resolver,
                state,
                name_hint=name_hint,
                expr=expr,
                pending_target=target,
            )
            return [binding], ident
        if isinstance(expr, BinaryOpIR):
            pair = _source_requested_quotient_pair(expr)
            if pair is not None:
                left_name = getattr(expr.left.operand if isinstance(expr.left, DifferentialIR) else expr.left, "name", None) or "j"
                binding, ident = self._fresh_pending_binding(
                    resolver,
                    state,
                    name_hint=left_name,
                    expr=expr,
                    pending_pair=pair,
                )
                return [binding], ident
            lp, left = self._rewrite_expr_for_program(expr.left, resolver, state)
            rp, right = self._rewrite_expr_for_program(expr.right, resolver, state)
            expr.left = left
            expr.right = right
            return lp + rp, expr
        if isinstance(expr, UnaryOpIR):
            pp, operand = self._rewrite_expr_for_program(expr.operand, resolver, state)
            expr.operand = operand
            return pp, expr
        if isinstance(expr, BuiltinCallIR):
            if _builtin_defid(expr) == _PRINT_BUILTIN_DEFID and len(expr.args or []) == 1:
                literal = self._symbolic_print_literal(expr.args[0])
                if literal is not None:
                    expr.args = (literal,)
                    return [], expr
            prefixes: List[BindingIR] = []
            new_args: List[ExpressionIR] = []
            for arg in expr.args or []:
                pp, new_arg = self._rewrite_expr_for_program(arg, resolver, state)
                prefixes.extend(pp)
                if new_arg is not None:
                    new_args.append(new_arg)
            expr.args = tuple(new_args)
            return prefixes, expr
        if isinstance(expr, FunctionCallIR):
            prefixes: List[BindingIR] = []
            if expr.callee_expr is not None:
                pp, callee_expr = self._rewrite_expr_for_program(expr.callee_expr, resolver, state)
                prefixes.extend(pp)
                expr.callee_expr = callee_expr
            new_args: List[ExpressionIR] = []
            for arg in expr.arguments or []:
                pp, new_arg = self._rewrite_expr_for_program(arg, resolver, state)
                prefixes.extend(pp)
                if new_arg is not None:
                    new_args.append(new_arg)
            expr.arguments = tuple(new_args)
            return prefixes, expr
        if isinstance(expr, RectangularAccessIR):
            prefixes: List[BindingIR] = []
            pp, array = self._rewrite_expr_for_program(expr.array, resolver, state)
            prefixes.extend(pp)
            expr.array = array
            new_indices: List[ExpressionIR] = []
            for idx in expr.indices or []:
                pp, new_idx = self._rewrite_expr_for_program(idx, resolver, state)
                prefixes.extend(pp)
                if new_idx is not None:
                    new_indices.append(new_idx)
            expr.indices = tuple(new_indices)
            return prefixes, expr
        if isinstance(expr, CastExpressionIR):
            pp, inner = self._rewrite_expr_for_program(expr.expr, resolver, state)
            expr.expr = inner
            return pp, expr
        if isinstance(expr, MemberAccessIR):
            pp, obj = self._rewrite_expr_for_program(expr.object, resolver, state)
            expr.object = obj
            return pp, expr
        if isinstance(expr, ArrayLiteralIR):
            prefixes: List[BindingIR] = []
            elems: List[ExpressionIR] = []
            for elem in expr.elements or []:
                pp, new_elem = self._rewrite_expr_for_program(elem, resolver, state)
                prefixes.extend(pp)
                if new_elem is not None:
                    elems.append(new_elem)
            expr.elements = tuple(elems)
            return prefixes, expr
        if isinstance(expr, BlockExpressionIR):
            expr.statements = tuple(self._rewrite_statement_list(list(expr.statements or []), resolver, state))
            pp, final_expr = self._rewrite_expr_for_program(expr.final_expr, resolver, state)
            if pp:
                expr.statements = tuple(list(expr.statements or []) + pp)
            expr.final_expr = final_expr
            return [], expr
        if isinstance(expr, IfExpressionIR):
            cond_prefix, cond = self._rewrite_expr_for_program(expr.condition, resolver, state)
            then_prefix, then_expr = self._rewrite_expr_for_program(expr.then_expr, resolver, state)
            else_prefix, else_expr = self._rewrite_expr_for_program(expr.else_expr, resolver, state)
            expr.condition = cond
            expr.then_expr = self._wrap_prefix_block(then_prefix, then_expr)
            expr.else_expr = self._wrap_prefix_block(else_prefix, else_expr)
            return cond_prefix, expr
        if isinstance(expr, RangeIR):
            sp, start = self._rewrite_expr_for_program(expr.start, resolver, state)
            ep, end = self._rewrite_expr_for_program(expr.end, resolver, state)
            expr.start = start
            expr.end = end
            return sp + ep, expr
        if isinstance(expr, ReductionExpressionIR):
            body_prefix, body = self._rewrite_expr_for_program(expr.body, resolver, state)
            expr.body = self._wrap_prefix_block(body_prefix, body)
            return [], expr
        if isinstance(expr, EinsteinIR):
            for clause in expr.clauses or []:
                value_prefix, value = self._rewrite_expr_for_program(clause.value, resolver, state)
                clause.value = self._wrap_prefix_block(value_prefix, value)
            return [], expr
        if isinstance(expr, FunctionValueIR):
            if expr.body is not None:
                body_prefix, body = self._rewrite_expr_for_program(expr.body, resolver, state)
                expr.body = self._wrap_prefix_block(body_prefix, body)
            return [], expr
        return [], expr

    @staticmethod
    def _wrap_prefix_block(prefixes: List[BindingIR], expr: Optional[ExpressionIR]) -> Optional[ExpressionIR]:
        if expr is None:
            return None
        if not prefixes:
            return expr
        return BlockExpressionIR(
            statements=tuple(prefixes),
            final_expr=expr,
            location=getattr(expr, "location", None) or _LOC0,
            type_info=_ti(expr),
            shape_info=_si(expr),
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
