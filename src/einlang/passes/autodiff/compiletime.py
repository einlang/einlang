"""Compile-time autodiff expansion.

This pass eliminates ``DifferentialIR`` and source quotient requests by
synthesizing ordinary Einlang IR at compile time. No runtime autodiff graph or
runtime autodiff builtins are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..base import BasePass, TyCtxt
from ..extremum_selection_canonicalization import ExtremumSelectionCanonicalizationPass
from ..pre_autodiff_pruning import PreAutodiffPruningPass
from ..shape_analysis import UnifiedShapeAnalysisPass
from ..type_inference import TypeInferencePass
from ...ir.nodes import (
    ArrayLiteralIR,
    BinaryOpIR,
    BindingIR,
    BlockExpressionIR,
    BuiltinCallIR,
    CastExpressionIR,
    DifferentialIR,
    DiffRuleIR,
    EinsteinClauseIR,
    EinsteinIR,
    ExpressionIR,
    FunctionCallIR,
    FunctionValueIR,
    IdentifierIR,
    IfExpressionIR,
    IRNode,
    IndexRestIR,
    IndexVarIR,
    LazyJacobianIR,
    LiteralIR,
    MemberAccessIR,
    ProgramIR,
    RangeIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    SelectAtArgmaxIR,
    TupleAccessIR,
    TupleExpressionIR,
    UnaryOpIR,
    WhereClauseIR,
    WhereExpressionIR,
    clear_autodiff_only_fields,
    is_function_binding,
)
from ...shared.defid import DefId
from ...shared.source_location import SourceLocation
from ...shared.types import (
    F32,
    STR,
    Type,
    BinaryOp,
    PrimitiveType,
    ReductionOp,
    RectangularType,
    UnaryOp,
    strip_differential_types_deep,
)


DIFF_PREFIX = "_@"
USER_DIFF_PREFIX = "@"
_LOC0 = SourceLocation("", 0, 0)
_ZERO_TANGENT_BUILTINS = frozenset({"assert", "len", "typeof", "shape"})
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
_TEMPLATE_CRATE = -98


def _ti(node: Any) -> Any:
    return getattr(node, "type_info", None)


def _si(node: Any) -> Any:
    return getattr(node, "shape_info", None)


def _fl(value: float, loc: Optional[SourceLocation] = None) -> LiteralIR:
    return LiteralIR(float(value), loc or _LOC0, type_info=F32)


def _lit_int(value: int, loc: Optional[SourceLocation] = None) -> LiteralIR:
    return LiteralIR(int(value), loc or _LOC0, type_info=PrimitiveType("i32"))


def _z(loc: Optional[SourceLocation] = None) -> LiteralIR:
    return _fl(0.0, loc)


def _one(loc: Optional[SourceLocation] = None) -> LiteralIR:
    return _fl(1.0, loc)


def _is_zero(expr: Optional[ExpressionIR]) -> bool:
    return isinstance(expr, LiteralIR) and bool(expr.value == 0 or expr.value == 0.0)


def _is_one(expr: Optional[ExpressionIR]) -> bool:
    return isinstance(expr, LiteralIR) and bool(expr.value == 1 or expr.value == 1.0)


@lru_cache(maxsize=None)
def _slot_names(cls: type) -> Tuple[str, ...]:
    out: List[str] = []
    for base in cls.__mro__:
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            out.append(slots)
        else:
            out.extend(slots)
    seen: Set[str] = set()
    ordered: List[str] = []
    for name in out:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


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


def _clone_expr(expr: ExpressionIR) -> ExpressionIR:
    return _clone_ir_value(expr, {})


def _expr_eq(left: Optional[ExpressionIR], right: Optional[ExpressionIR]) -> bool:
    if left is None or right is None:
        return left is right
    if type(left) is not type(right):
        return False
    if isinstance(left, LiteralIR):
        return left.value == right.value
    if isinstance(left, IdentifierIR):
        if left.defid is not None or right.defid is not None:
            return left.defid == right.defid
        return left.name == right.name
    if isinstance(left, IndexVarIR):
        return left.defid == right.defid and left.name == right.name
    if isinstance(left, IndexRestIR):
        return left.defid == right.defid and left.name == right.name
    if isinstance(left, UnaryOpIR):
        return left.operator == right.operator and _expr_eq(left.operand, right.operand)
    if isinstance(left, BinaryOpIR):
        return (
            left.operator == right.operator
            and _expr_eq(left.left, right.left)
            and _expr_eq(left.right, right.right)
        )
    if isinstance(left, RectangularAccessIR):
        return _expr_eq(left.array, right.array) and len(left.indices or ()) == len(right.indices or ()) and all(
            _expr_eq(a, b) for a, b in zip(left.indices or (), right.indices or ())
        )
    if isinstance(left, FunctionCallIR):
        return (
            left.function_defid == right.function_defid
            and len(left.arguments or ()) == len(right.arguments or ())
            and all(_expr_eq(a, b) for a, b in zip(left.arguments or (), right.arguments or ()))
        )
    if isinstance(left, BuiltinCallIR):
        return (
            left.builtin_name == right.builtin_name
            and len(left.args or ()) == len(right.args or ())
            and all(_expr_eq(a, b) for a, b in zip(left.args or (), right.args or ()))
        )
    if isinstance(left, CastExpressionIR):
        return left.target_type == right.target_type and _expr_eq(left.expr, right.expr)
    if isinstance(left, IfExpressionIR):
        return (
            _expr_eq(left.condition, right.condition)
            and _expr_eq(left.then_expr, right.then_expr)
            and _expr_eq(left.else_expr, right.else_expr)
        )
    if isinstance(left, ArrayLiteralIR):
        return len(left.elements or ()) == len(right.elements or ()) and all(
            _expr_eq(a, b) for a, b in zip(left.elements or (), right.elements or ())
        )
    if isinstance(left, TupleExpressionIR):
        return len(left.elements or ()) == len(right.elements or ()) and all(
            _expr_eq(a, b) for a, b in zip(left.elements or (), right.elements or ())
        )
    return False


def _simplify(expr: ExpressionIR) -> ExpressionIR:
    if isinstance(expr, UnaryOpIR):
        operand = _simplify(expr.operand)
        if expr.operator == UnaryOp.NEG and isinstance(operand, LiteralIR) and isinstance(operand.value, (int, float)):
            return _fl(-float(operand.value), expr.location)
        if expr.operator == UnaryOp.POS:
            return operand
        expr.operand = operand
        return expr
    if isinstance(expr, CastExpressionIR):
        expr.expr = _simplify(expr.expr)
        return expr
    if isinstance(expr, IfExpressionIR):
        expr.condition = _simplify(expr.condition)
        expr.then_expr = _simplify(expr.then_expr)
        if expr.else_expr is not None:
            expr.else_expr = _simplify(expr.else_expr)
        if isinstance(expr.condition, LiteralIR) and isinstance(expr.condition.value, bool):
            return expr.then_expr if expr.condition.value else (expr.else_expr or _z(expr.location))
        return expr
    if isinstance(expr, BlockExpressionIR):
        new_statements: List[Any] = []
        for stmt in expr.statements or ():
            if isinstance(stmt, BindingIR) and stmt.expr is not None:
                stmt.expr = _simplify(stmt.expr)
            elif isinstance(stmt, ExpressionIR):
                stmt = _simplify(stmt)
            new_statements.append(stmt)
        expr.statements = tuple(new_statements)
        if expr.final_expr is not None:
            expr.final_expr = _simplify(expr.final_expr)
            if not expr.statements:
                return expr.final_expr
        return expr
    if isinstance(expr, BinaryOpIR):
        expr.left = _simplify(expr.left)
        expr.right = _simplify(expr.right)
        left = expr.left
        right = expr.right
        op = expr.operator
        if isinstance(left, LiteralIR) and isinstance(right, LiteralIR):
            if isinstance(left.value, (int, float)) and isinstance(right.value, (int, float)):
                lv = float(left.value)
                rv = float(right.value)
                if op == BinaryOp.ADD:
                    return _fl(lv + rv, expr.location)
                if op == BinaryOp.SUB:
                    return _fl(lv - rv, expr.location)
                if op == BinaryOp.MUL:
                    return _fl(lv * rv, expr.location)
                if op == BinaryOp.DIV and rv != 0:
                    return _fl(lv / rv, expr.location)
                if op == BinaryOp.POW:
                    return _fl(lv ** rv, expr.location)
        if op == BinaryOp.ADD:
            if _is_zero(left):
                return right
            if _is_zero(right):
                return left
            if _expr_eq(left, right):
                return _simplify(
                    BinaryOpIR(
                        BinaryOp.MUL,
                        _fl(2.0, expr.location),
                        left,
                        expr.location,
                        type_info=_ti(expr),
                        shape_info=_si(expr),
                    )
                )
        if op == BinaryOp.SUB:
            if _is_zero(right):
                return left
            if _is_zero(left):
                return UnaryOpIR(UnaryOp.NEG, right, expr.location, type_info=_ti(expr), shape_info=_si(expr))
        if op == BinaryOp.MUL:
            if _is_zero(left) or _is_zero(right):
                return _z(expr.location)
            if _is_one(left):
                return right
            if _is_one(right):
                return left
        if op == BinaryOp.DIV:
            if _is_zero(left):
                return _z(expr.location)
            if _is_one(right):
                return left
        if op == BinaryOp.POW:
            if _is_one(right):
                return left
            if _is_zero(right):
                return _one(expr.location)
        return expr
    if isinstance(expr, RectangularAccessIR):
        expr.array = _simplify(expr.array)
        expr.indices = tuple(_simplify(idx) for idx in (expr.indices or ()))
        if isinstance(expr.array, LiteralIR):
            return expr.array
        if isinstance(expr.array, EinsteinIR):
            clauses = tuple(expr.array.clauses or ())
            if len(clauses) == 1:
                only = clauses[0]
                if (
                    isinstance(only.value, LiteralIR)
                    and not (only.where_clause and only.where_clause.constraints)
                ):
                    return only.value
        return expr
    if isinstance(expr, ReductionExpressionIR):
        expr.body = _simplify(expr.body)
        return expr
    if isinstance(expr, EinsteinIR):
        clauses: List[EinsteinClauseIR] = []
        for clause in expr.clauses or ():
            clause.value = _simplify(clause.value)
            clauses.append(clause)
        expr.clauses = tuple(clauses)
        if len(expr.clauses or ()) == 1:
            only = expr.clauses[0]
            if not (only.indices or ()) and not (only.where_clause and only.where_clause.constraints):
                return only.value
        return expr
    if isinstance(expr, ArrayLiteralIR):
        expr.elements = tuple(_simplify(elem) for elem in (expr.elements or ()))
        return expr
    if isinstance(expr, TupleExpressionIR):
        expr.elements = tuple(_simplify(elem) for elem in (expr.elements or ()))
        return expr
    if isinstance(expr, WhereExpressionIR):
        expr.expr = _simplify(expr.expr)
        expr.constraints = tuple(_simplify(c) for c in (expr.constraints or ()))
        return expr
    return expr


def clear_custom_diff_body_everywhere(obj: Any) -> None:
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
        if isinstance(cur, FunctionValueIR):
            cur.custom_diff_body = None
        if hasattr(cur, "type_info"):
            ty = getattr(cur, "type_info", None)
            if isinstance(ty, Type):
                try:
                    setattr(cur, "type_info", strip_differential_types_deep(ty))
                except Exception:
                    pass
        if isinstance(cur, CastExpressionIR):
            tt = getattr(cur, "target_type", None)
            if isinstance(tt, Type):
                try:
                    cur.target_type = strip_differential_types_deep(tt)
                except Exception:
                    pass
        if isinstance(cur, FunctionValueIR):
            rt = getattr(cur, "return_type", None)
            if isinstance(rt, Type):
                try:
                    cur.return_type = strip_differential_types_deep(rt)
                except Exception:
                    pass
            for param in cur.parameters or ():
                pt = getattr(param, "param_type", None)
                if isinstance(pt, Type):
                    try:
                        param.param_type = strip_differential_types_deep(pt)
                    except Exception:
                        pass
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


def _bindings_in(node: Any) -> List[BindingIR]:
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
            stack.extend(cur)
            continue
        if isinstance(cur, IRNode):
            for cls in type(cur).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    stack.append(getattr(cur, slot, None))
    return out


def _local_binding_contexts(node: Any) -> Dict[DefId, Dict[DefId, BindingIR]]:
    """Map each local binding DefId to the earlier local bindings in its block."""

    contexts: Dict[DefId, Dict[DefId, BindingIR]] = {}
    seen: Set[int] = set()

    def walk(cur: Any) -> None:
        if cur is None:
            return
        oid = id(cur)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(cur, BlockExpressionIR):
            local_seen: Dict[DefId, BindingIR] = {}
            for stmt in cur.statements or ():
                if isinstance(stmt, BindingIR) and stmt.defid is not None and not is_function_binding(stmt):
                    contexts[stmt.defid] = dict(local_seen)
                    local_seen[stmt.defid] = stmt
                walk(stmt)
            walk(cur.final_expr)
            return
        if isinstance(cur, dict):
            for key, value in cur.items():
                walk(key)
                walk(value)
            return
        if isinstance(cur, (list, tuple)):
            for item in cur:
                walk(item)
            return
        if isinstance(cur, IRNode):
            for cls in type(cur).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    walk(getattr(cur, slot, None))

    walk(node)
    return contexts


def _pending_differential_target(expr: ExpressionIR) -> Optional[DefId]:
    if isinstance(expr, DifferentialIR):
        operand = expr.operand
        if isinstance(operand, IdentifierIR) and operand.defid is not None:
            return operand.defid
    return None


def _source_requested_quotient_pair(expr: ExpressionIR) -> Optional[Tuple[DefId, DefId]]:
    if not isinstance(expr, BinaryOpIR) or expr.operator != BinaryOp.DIV:
        return None
    num_defid = _pending_differential_target(expr.left)
    den_defid = _pending_differential_target(expr.right)
    if num_defid is None or den_defid is None:
        return None
    return num_defid, den_defid


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
class _AutodiffTargets:
    diff_targets: List[Tuple[DefId, str]] = field(default_factory=list)
    quotient_pairs: List[Tuple[DefId, DefId]] = field(default_factory=list)


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
            deps = {dep for dep in self.dep_cache.collect_defids(binding.expr) if dep in self.bindings and dep != did}
        self.binding_deps[did] = deps
        return deps


def _binding_map(program: ProgramIR, tcx: TyCtxt) -> Dict[DefId, BindingIR]:
    binding_map: Dict[DefId, BindingIR] = {}
    for binding in _bindings_in(program):
        if binding.defid is not None:
            binding_map[binding.defid] = binding
    fim = getattr(tcx, "function_ir_map", None) or {}
    for did, binding in fim.items():
        if did is not None and isinstance(binding, BindingIR):
            binding_map[did] = binding
    return binding_map


def _shape_dim_expr(dim: Any, loc: SourceLocation) -> ExpressionIR:
    if isinstance(dim, ExpressionIR):
        return _clone_expr(dim)
    if isinstance(dim, bool):
        return LiteralIR(dim, loc)
    if isinstance(dim, int):
        return _lit_int(dim, loc)
    if isinstance(dim, float):
        return _fl(dim, loc)
    return _lit_int(0, loc)


def _infer_shape_from_einstein(binding: BindingIR) -> Tuple[ExpressionIR, ...]:
    expr = getattr(binding, "expr", None)
    if not isinstance(expr, EinsteinIR):
        return ()
    inferred: List[Optional[ExpressionIR]] = []
    for clause in expr.clauses or ():
        for axis, idx in enumerate(clause.indices or ()):
            dim_expr: Optional[ExpressionIR] = None
            if isinstance(idx, IndexVarIR):
                rng = clause.variable_ranges.get(idx.defid) or getattr(idx, "range_ir", None)
                if isinstance(rng, RangeIR):
                    start = _clone_expr(rng.start)
                    end = _clone_expr(rng.end)
                    loc = rng.location or binding.location or _LOC0
                    dim_expr = BinaryOpIR(BinaryOp.SUB, end, start, loc, type_info=PrimitiveType("i32"))
                    if getattr(rng, "inclusive", False):
                        dim_expr = BinaryOpIR(BinaryOp.ADD, dim_expr, _lit_int(1, loc), loc, type_info=PrimitiveType("i32"))
                elif isinstance(rng, ExpressionIR):
                    dim_expr = _clone_expr(rng)
            if dim_expr is None:
                continue
            while len(inferred) <= axis:
                inferred.append(None)
            if inferred[axis] is None:
                inferred[axis] = dim_expr
    return tuple(dim for dim in inferred if dim is not None)


def _binding_shape(binding: Optional[BindingIR]) -> Tuple[ExpressionIR, ...]:
    if binding is None:
        return ()
    shape = _si(binding.expr) if getattr(binding, "expr", None) is not None else None
    if not shape and isinstance(_ti(binding.expr), RectangularType):
        shape = getattr(_ti(binding.expr), "shape", None)
    if not shape and isinstance(_ti(binding), RectangularType):
        shape = getattr(_ti(binding), "shape", None)
    if isinstance(shape, tuple):
        return tuple(_shape_dim_expr(dim, binding.location or _LOC0) for dim in shape)
    if isinstance(shape, list):
        return tuple(_shape_dim_expr(dim, binding.location or _LOC0) for dim in shape)
    inferred = _infer_shape_from_einstein(binding)
    if inferred:
        return inferred
    return ()


def _template_defid(counter: List[int]) -> DefId:
    idx = counter[0]
    counter[0] += 1
    return DefId(_TEMPLATE_CRATE, idx)


def _binding_identifier(binding: BindingIR) -> IdentifierIR:
    expr = getattr(binding, "expr", None)
    return IdentifierIR(
        binding.name or "?",
        binding.location or _LOC0,
        defid=binding.defid,
        type_info=_ti(expr) if expr is not None else binding.type_info,
        shape_info=_si(expr) if expr is not None else None,
    )


def _make_cotangent_objective(binding: BindingIR, seed_ident: IdentifierIR, counter: List[int]) -> ExpressionIR:
    loc = binding.location or _LOC0
    target_ident = _binding_identifier(binding)
    shape = _binding_shape(binding)
    if not shape:
        return BinaryOpIR(
            BinaryOp.MUL,
            target_ident,
            seed_ident,
            loc,
            type_info=_ti(getattr(binding, "expr", None)),
            shape_info=_si(getattr(binding, "expr", None)),
        )

    loop_vars: List[IndexVarIR] = []
    loop_var_ranges: Dict[DefId, RangeIR] = {}
    indices: List[IndexVarIR] = []
    for axis, dim in enumerate(shape):
        iv_defid = _template_defid(counter)
        iv = IndexVarIR(
            f"_ad_idx_{axis}",
            loc,
            iv_defid,
            type_info=PrimitiveType("i32"),
        )
        rng = RangeIR(_lit_int(0, loc), _clone_expr(dim), loc, type_info=PrimitiveType("range"))
        iv.range_ir = rng
        loop_vars.append(iv)
        loop_var_ranges[iv_defid] = rng
        indices.append(iv)

    body = BinaryOpIR(
        BinaryOp.MUL,
        RectangularAccessIR(target_ident, list(indices), loc),
        RectangularAccessIR(seed_ident, list(indices), loc),
        loc,
    )
    return ReductionExpressionIR(
        ReductionOp.SUM,
        list(loop_vars),
        body,
        loc,
        loop_var_ranges=loop_var_ranges,
    )


def _tensor_constant_like(
    binding: Optional[BindingIR],
    resolver: Any,
    loc: SourceLocation,
    value: float,
) -> ExpressionIR:
    shape = _binding_shape(binding)
    if not shape:
        return _fl(value, loc)
    indices: List[IndexVarIR] = []
    variable_ranges: Dict[DefId, RangeIR] = {}
    for axis, dim in enumerate(shape):
        did = resolver.allocate_for_local() if resolver is not None else None
        iv = IndexVarIR(f"_ad{axis}", loc, did, type_info=PrimitiveType("i32"))
        rng = RangeIR(_lit_int(0, loc), _clone_expr(dim), loc, type_info=PrimitiveType("range"))
        iv.range_ir = rng
        indices.append(iv)
        if did is not None:
            variable_ranges[did] = rng
    clause = EinsteinClauseIR(indices=indices, value=_fl(value, loc), location=loc, variable_ranges=variable_ranges)
    return EinsteinIR(
        clauses=[clause],
        shape=list(shape),
        element_type=getattr(_ti(binding.expr), "element_type", None) if binding is not None and binding.expr is not None else None,
        location=loc,
        type_info=_ti(binding.expr) if binding is not None and binding.expr is not None else None,
        shape_info=tuple(shape),
    )


def _seed_expr(
    binding: Optional[BindingIR],
    *,
    symbolic: bool,
    resolver: Any,
    loc: SourceLocation,
    value: float,
) -> ExpressionIR:
    if symbolic:
        name = getattr(binding, "name", None) or "x"
        return IdentifierIR(f"{USER_DIFF_PREFIX}{name}", loc, type_info=_ti(binding.expr) if binding is not None and binding.expr is not None else _ti(binding), shape_info=_si(binding.expr) if binding is not None and binding.expr is not None else _si(binding))
    return _tensor_constant_like(binding, resolver, loc, value)


def _substitute_identifiers(expr: ExpressionIR, repl: Dict[DefId, ExpressionIR]) -> ExpressionIR:
    def walk(node: Any) -> Any:
        if node is None or isinstance(node, (str, int, float, bool, bytes)):
            return node
        if isinstance(node, IdentifierIR) and node.defid in repl:
            return _clone_expr(repl[node.defid])
        if isinstance(node, dict):
            return {walk(k): walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, tuple):
            return tuple(walk(item) for item in node)
        if isinstance(node, IRNode):
            cloned = node.__class__.__new__(node.__class__)
            for slot in _slot_names(node.__class__):
                setattr(cloned, slot, walk(getattr(node, slot, None)))
            return cloned
        return node

    return walk(expr)


def _rewrite_custom_diff_body(
    expr: ExpressionIR,
    primal_subst: Dict[DefId, ExpressionIR],
    tangent_subst: Dict[DefId, ExpressionIR],
) -> ExpressionIR:
    def walk(node: Any) -> Any:
        if node is None or isinstance(node, (str, int, float, bool, bytes)):
            return node
        if isinstance(node, DifferentialIR):
            operand = node.operand
            if isinstance(operand, IdentifierIR) and operand.defid in tangent_subst:
                return _clone_expr(tangent_subst[operand.defid])
            return walk(operand)
        if isinstance(node, IdentifierIR) and node.defid in primal_subst:
            return _clone_expr(primal_subst[node.defid])
        if isinstance(node, dict):
            return {walk(k): walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, tuple):
            return tuple(walk(item) for item in node)
        if isinstance(node, IRNode):
            cloned = node.__class__.__new__(node.__class__)
            for slot in _slot_names(node.__class__):
                setattr(cloned, slot, walk(getattr(node, slot, None)))
            return cloned
        return node

    return walk(expr)


class _Differentiator:
    def __init__(
        self,
        binding_ctx: _AutodiffBindingContext,
        resolver: Any,
        local_contexts: Optional[Dict[DefId, Dict[DefId, BindingIR]]] = None,
    ) -> None:
        self._ctx = binding_ctx
        self._resolver = resolver
        self._local_contexts = local_contexts or {}
        self._self_recursive_bindings = {
            did
            for did, binding in binding_ctx.bindings.items()
            if (
                did is not None
                and binding is not None
                and not is_function_binding(binding)
                and getattr(binding, "expr", None) is not None
                and did in binding_ctx.dep_cache.collect_defids(binding.expr)
            )
        }
        self._leaf_bindings = {
            did: binding
            for did, binding in binding_ctx.bindings.items()
            if (
                not is_function_binding(binding)
                and did not in self._self_recursive_bindings
                and not binding_ctx.deps_for(did)
            )
        }
        self._cache: Dict[Tuple[Optional[DefId], bool, DefId], ExpressionIR] = {}
        self._binding_diff_cache: Optional[Dict[DefId, ExpressionIR]] = None

    @staticmethod
    def _local_depends_on(
        defid: DefId,
        wrt: DefId,
        local_bindings: Dict[DefId, BindingIR],
        memo: Dict[Tuple[DefId, DefId], bool],
    ) -> bool:
        key = (defid, wrt)
        if key in memo:
            return memo[key]
        if defid == wrt:
            memo[key] = True
            return True
        binding = local_bindings.get(defid)
        if binding is None or binding.expr is None:
            memo[key] = False
            return False
        deps = [dep for dep in _collect_defids(binding.expr) if dep in local_bindings]
        memo[key] = any(
            _Differentiator._local_depends_on(dep, wrt, local_bindings, memo)
            for dep in deps
        )
        return memo[key]

    def _seed_map(
        self,
        wrt: Optional[DefId],
        symbolic: bool,
        loc: SourceLocation,
        *,
        local_bindings: Optional[Dict[DefId, BindingIR]] = None,
    ) -> Dict[DefId, ExpressionIR]:
        out: Dict[DefId, ExpressionIR] = {}
        if wrt is None:
            for did, binding in self._leaf_bindings.items():
                out[did] = _seed_expr(binding, symbolic=symbolic, resolver=self._resolver, loc=loc, value=1.0)
            return out
        binding = self._ctx.bindings.get(wrt)
        out[wrt] = _seed_expr(binding, symbolic=symbolic, resolver=self._resolver, loc=loc, value=1.0)
        for did, leaf in self._leaf_bindings.items():
            if did == wrt:
                continue
            out[did] = _seed_expr(leaf, symbolic=symbolic, resolver=self._resolver, loc=loc, value=0.0)
        if local_bindings:
            memo: Dict[Tuple[DefId, DefId], bool] = {}
            for did, local_binding in local_bindings.items():
                if did == wrt or did in out:
                    continue
                if not self._local_depends_on(did, wrt, local_bindings, memo):
                    out[did] = _seed_expr(
                        local_binding,
                        symbolic=symbolic,
                        resolver=self._resolver,
                        loc=loc,
                        value=0.0,
                    )
        return out

    def standalone(self, expr: ExpressionIR, loc: SourceLocation, *, symbolic: bool) -> ExpressionIR:
        seed_map = self._seed_map(None, symbolic, loc)
        prev_cache = self._binding_diff_cache
        self._binding_diff_cache = {}
        try:
            return self._diff_expr(expr, seed_map, {}, {}, symbolic, loc)
        finally:
            self._binding_diff_cache = prev_cache

    def differentiate_expr(
        self,
        expr: ExpressionIR,
        wrt_defid: DefId,
        loc: SourceLocation,
        *,
        symbolic: bool,
        local_bindings: Optional[Dict[DefId, BindingIR]] = None,
        seed_override: Optional[ExpressionIR] = None,
    ) -> ExpressionIR:
        scoped_locals = dict(local_bindings or {})
        seed_map = self._seed_map(wrt_defid, symbolic, loc, local_bindings=scoped_locals)
        if seed_override is not None:
            seed_map[wrt_defid] = _clone_expr(seed_override)
        prev_cache = self._binding_diff_cache
        self._binding_diff_cache = {}
        try:
            expr_out = self._diff_expr(expr, seed_map, scoped_locals, {}, symbolic, loc)
            return _simplify(expr_out)
        finally:
            self._binding_diff_cache = prev_cache

    def wrt(self, target_defid: DefId, wrt_defid: DefId, loc: SourceLocation, *, symbolic: bool) -> ExpressionIR:
        key = (wrt_defid, symbolic, target_defid)
        cached = self._cache.get(key)
        if cached is not None:
            return _clone_expr(cached)
        binding = self._ctx.bindings.get(target_defid)
        if binding is None or binding.expr is None:
            return _z(loc)
        local_bindings = self._local_contexts.get(target_defid) or {}
        expr = self.differentiate_expr(
            binding.expr,
            wrt_defid,
            loc,
            symbolic=symbolic,
            local_bindings=local_bindings,
        )
        self._cache[key] = expr
        return _clone_expr(expr)

    def _diff_binding_ref(
        self,
        defid: DefId,
        seed_map: Dict[DefId, ExpressionIR],
        local_bindings: Dict[DefId, BindingIR],
        local_diffs: Dict[DefId, ExpressionIR],
        symbolic: bool,
        loc: SourceLocation,
    ) -> ExpressionIR:
        if defid in local_diffs:
            return _clone_expr(local_diffs[defid])
        if defid in seed_map:
            return _clone_expr(seed_map[defid])
        binding_cache = self._binding_diff_cache
        if binding_cache is not None and defid in binding_cache:
            return _clone_expr(binding_cache[defid])
        binding = local_bindings.get(defid) or self._ctx.bindings.get(defid)
        if binding is None or binding.expr is None or is_function_binding(binding):
            return _z(loc)
        diff_expr = self._diff_expr(binding.expr, seed_map, local_bindings, local_diffs, symbolic, loc)
        if (
            binding_cache is not None
            and defid not in local_bindings
            and defid not in self._self_recursive_bindings
        ):
            binding_cache[defid] = diff_expr
        return _clone_expr(diff_expr)

    def _diff_expr(
        self,
        expr: ExpressionIR,
        seed_map: Dict[DefId, ExpressionIR],
        local_bindings: Dict[DefId, BindingIR],
        local_diffs: Dict[DefId, ExpressionIR],
        symbolic: bool,
        loc: SourceLocation,
    ) -> ExpressionIR:
        if isinstance(expr, LiteralIR):
            return _z(expr.location or loc)
        if isinstance(expr, IdentifierIR):
            if expr.defid is None:
                return _z(expr.location or loc)
            return self._diff_binding_ref(expr.defid, seed_map, local_bindings, local_diffs, symbolic, expr.location or loc)
        if isinstance(expr, DifferentialIR):
            return self.standalone(expr.operand, expr.location or loc, symbolic=symbolic)
        if isinstance(expr, UnaryOpIR):
            inner = self._diff_expr(expr.operand, seed_map, local_bindings, local_diffs, symbolic, expr.location or loc)
            if expr.operator == UnaryOp.NEG:
                return _simplify(UnaryOpIR(UnaryOp.NEG, inner, expr.location or loc, type_info=_ti(expr), shape_info=_si(expr)))
            if expr.operator == UnaryOp.POS:
                return inner
            return _z(expr.location or loc)
        if isinstance(expr, BinaryOpIR):
            loc0 = expr.location or loc
            left = _clone_expr(expr.left)
            right = _clone_expr(expr.right)
            dleft = self._diff_expr(expr.left, seed_map, local_bindings, local_diffs, symbolic, loc0)
            dright = self._diff_expr(expr.right, seed_map, local_bindings, local_diffs, symbolic, loc0)
            if expr.operator == BinaryOp.ADD:
                return _simplify(BinaryOpIR(BinaryOp.ADD, dleft, dright, loc0, type_info=_ti(expr), shape_info=_si(expr)))
            if expr.operator == BinaryOp.SUB:
                return _simplify(BinaryOpIR(BinaryOp.SUB, dleft, dright, loc0, type_info=_ti(expr), shape_info=_si(expr)))
            if expr.operator == BinaryOp.MUL:
                return _simplify(
                    BinaryOpIR(
                        BinaryOp.ADD,
                        BinaryOpIR(BinaryOp.MUL, left, dright, loc0, type_info=_ti(expr), shape_info=_si(expr)),
                        BinaryOpIR(BinaryOp.MUL, right, dleft, loc0, type_info=_ti(expr), shape_info=_si(expr)),
                        loc0,
                        type_info=_ti(expr),
                        shape_info=_si(expr),
                    )
                )
            if expr.operator == BinaryOp.DIV:
                numerator = BinaryOpIR(
                    BinaryOp.SUB,
                    BinaryOpIR(BinaryOp.MUL, right, dleft, loc0, type_info=_ti(expr), shape_info=_si(expr)),
                    BinaryOpIR(BinaryOp.MUL, left, dright, loc0, type_info=_ti(expr), shape_info=_si(expr)),
                    loc0,
                    type_info=_ti(expr),
                    shape_info=_si(expr),
                )
                denominator = BinaryOpIR(
                    BinaryOp.POW,
                    right,
                    _fl(2.0, loc0),
                    loc0,
                    type_info=_ti(expr),
                    shape_info=_si(expr),
                )
                return _simplify(BinaryOpIR(BinaryOp.DIV, numerator, denominator, loc0, type_info=_ti(expr), shape_info=_si(expr)))
            if expr.operator == BinaryOp.POW:
                if isinstance(right, LiteralIR) and isinstance(right.value, (int, float)):
                    exponent = float(right.value)
                    coeff = _fl(exponent, loc0)
                    lowered_power = BinaryOpIR(
                        BinaryOp.POW,
                        left,
                        _fl(exponent - 1.0, loc0),
                        loc0,
                        type_info=_ti(expr),
                        shape_info=_si(expr),
                    )
                    return _simplify(
                        BinaryOpIR(
                            BinaryOp.MUL,
                            coeff,
                            BinaryOpIR(BinaryOp.MUL, lowered_power, dleft, loc0, type_info=_ti(expr), shape_info=_si(expr)),
                            loc0,
                            type_info=_ti(expr),
                            shape_info=_si(expr),
                        )
                    )
                return _z(loc0)
            if expr.operator == BinaryOp.MOD:
                return dleft
            return _z(loc0)
        if isinstance(expr, RectangularAccessIR):
            loc0 = expr.location or loc
            darr = self._diff_expr(expr.array, seed_map, local_bindings, local_diffs, symbolic, loc0)
            return _simplify(
                RectangularAccessIR(
                    darr,
                    [_clone_expr(idx) for idx in (expr.indices or ())],
                    loc0,
                    type_info=_ti(expr),
                    shape_info=_si(expr),
                )
            )
        if isinstance(expr, CastExpressionIR):
            target = expr.target_type
            if isinstance(target, PrimitiveType) and target.name in {"i8", "i32", "i64", "bool", "str", "range", "unit"}:
                return _z(expr.location or loc)
            return self._diff_expr(expr.expr, seed_map, local_bindings, local_diffs, symbolic, expr.location or loc)
        if isinstance(expr, MemberAccessIR):
            return _z(expr.location or loc)
        if isinstance(expr, ArrayLiteralIR):
            return ArrayLiteralIR(
                [self._diff_expr(elem, seed_map, local_bindings, local_diffs, symbolic, expr.location or loc) for elem in (expr.elements or ())],
                expr.location or loc,
                type_info=_ti(expr),
                shape_info=_si(expr),
            )
        if isinstance(expr, TupleExpressionIR):
            return TupleExpressionIR(
                [self._diff_expr(elem, seed_map, local_bindings, local_diffs, symbolic, expr.location or loc) for elem in (expr.elements or ())],
                expr.location or loc,
                type_info=_ti(expr),
                shape_info=_si(expr),
            )
        if isinstance(expr, TupleAccessIR):
            return TupleAccessIR(
                self._diff_expr(expr.tuple_expr, seed_map, local_bindings, local_diffs, symbolic, expr.location or loc),
                expr.index,
                expr.location or loc,
                type_info=_ti(expr),
                shape_info=_si(expr),
            )
        if isinstance(expr, IfExpressionIR):
            return _simplify(
                IfExpressionIR(
                    _clone_expr(expr.condition),
                    self._diff_expr(expr.then_expr, seed_map, local_bindings, local_diffs, symbolic, expr.location or loc),
                    expr.location or loc,
                    else_expr=self._diff_expr(expr.else_expr, seed_map, local_bindings, local_diffs, symbolic, expr.location or loc) if expr.else_expr is not None else _z(expr.location or loc),
                    type_info=_ti(expr),
                    shape_info=_si(expr),
                )
            )
        if isinstance(expr, BlockExpressionIR):
            loc0 = expr.location or loc
            nested_bindings = dict(local_bindings)
            nested_diffs = dict(local_diffs)
            out_statements: List[Any] = []
            for stmt in expr.statements or ():
                if isinstance(stmt, BindingIR) and stmt.defid is not None and stmt.expr is not None and not is_function_binding(stmt):
                    primal_stmt = BindingIR(
                        stmt.name,
                        _clone_expr(stmt.expr),
                        location=stmt.location or loc0,
                        defid=stmt.defid,
                        type_info=stmt.type_info,
                    )
                    nested_bindings[stmt.defid] = primal_stmt
                    out_statements.append(primal_stmt)
                    diff_defid = self._resolver.allocate_for_local() if self._resolver is not None else None
                    diff_ident = IdentifierIR(
                        f"{DIFF_PREFIX}{stmt.name or 'tmp'}",
                        stmt.location or loc0,
                        diff_defid,
                        type_info=_ti(stmt.expr) or stmt.type_info,
                        shape_info=_si(stmt.expr),
                    )
                    if diff_defid is not None:
                        # Register the tangent placeholder before differentiating the
                        # RHS so self-recursive bindings (for example recurrences like
                        # u[t] = f(u[t-1])) differentiate to _@u[t-1] instead of
                        # recursing forever or falling back to a zero-shaped seed.
                        nested_diffs[stmt.defid] = diff_ident
                    diff_expr = _simplify(self._diff_expr(stmt.expr, seed_map, nested_bindings, nested_diffs, symbolic, stmt.location or loc0))
                    nested_diffs[stmt.defid] = diff_ident if diff_defid is not None else diff_expr
                    if diff_defid is not None:
                        out_statements.append(
                            BindingIR(
                                diff_ident.name,
                                diff_expr,
                                location=stmt.location or loc0,
                                defid=diff_defid,
                                type_info=_ti(stmt.expr) or stmt.type_info,
                            )
                        )
                elif isinstance(stmt, ExpressionIR):
                    out_statements.append(_clone_expr(stmt))
                else:
                    out_statements.append(stmt)
            final_expr = self._diff_expr(expr.final_expr, seed_map, nested_bindings, nested_diffs, symbolic, loc0) if expr.final_expr is not None else _z(loc0)
            return _simplify(BlockExpressionIR(out_statements, loc0, final_expr, type_info=_ti(expr), shape_info=_si(expr)))
        if isinstance(expr, ReductionExpressionIR):
            loc0 = expr.location or loc
            body_diff = self._diff_expr(expr.body, seed_map, local_bindings, local_diffs, symbolic, loc0)
            if expr.operation == ReductionOp.SUM:
                return _simplify(
                    ReductionExpressionIR(
                        ReductionOp.SUM,
                        list(expr.loop_vars or ()),
                        body_diff,
                        loc0,
                        where_clause=expr.where_clause,
                        loop_var_ranges=dict(expr.loop_var_ranges or {}),
                        type_info=_ti(expr),
                        shape_info=_si(expr),
                    )
                )
            if expr.operation == ReductionOp.MAX:
                return SelectAtArgmaxIR(
                    _clone_expr(expr.body),
                    body_diff,
                    list(expr.loop_vars or ()),
                    loop_var_ranges=dict(expr.loop_var_ranges or {}),
                    location=loc0,
                    type_info=_ti(expr),
                    shape_info=_si(expr),
                )
            if expr.operation == ReductionOp.MIN:
                return SelectAtArgmaxIR(
                    _clone_expr(expr.body),
                    body_diff,
                    list(expr.loop_vars or ()),
                    loop_var_ranges=dict(expr.loop_var_ranges or {}),
                    location=loc0,
                    type_info=_ti(expr),
                    shape_info=_si(expr),
                    use_argmin=True,
                )
            if expr.operation == ReductionOp.PROD:
                quotient = BinaryOpIR(
                    BinaryOp.DIV,
                    _clone_expr(expr),
                    _clone_expr(expr.body),
                    loc0,
                    type_info=_ti(expr),
                    shape_info=_si(expr.body),
                )
                return _simplify(
                    ReductionExpressionIR(
                        ReductionOp.SUM,
                        list(expr.loop_vars or ()),
                        BinaryOpIR(BinaryOp.MUL, quotient, body_diff, loc0, type_info=_ti(expr), shape_info=_si(expr.body)),
                        loc0,
                        where_clause=expr.where_clause,
                        loop_var_ranges=dict(expr.loop_var_ranges or {}),
                        type_info=_ti(expr),
                        shape_info=_si(expr),
                    )
                )
            return _z(loc0)
        if isinstance(expr, EinsteinIR):
            clauses: List[EinsteinClauseIR] = []
            for clause in expr.clauses or ():
                value = _simplify(self._diff_expr(clause.value, seed_map, local_bindings, local_diffs, symbolic, clause.location or loc))
                if _is_zero(value):
                    continue
                clauses.append(
                    EinsteinClauseIR(
                        indices=[_clone_expr(idx) for idx in (clause.indices or ())],
                        value=value,
                        location=clause.location,
                        where_clause=clause.where_clause,
                        variable_ranges=dict(clause.variable_ranges or {}),
                    )
                )
            if not clauses:
                return _z(expr.location or loc)
            return _simplify(
                EinsteinIR(
                    clauses=clauses,
                    shape=list(expr.shape or ()) if expr.shape is not None else None,
                    element_type=expr.element_type,
                    location=expr.location or loc,
                    type_info=_ti(expr),
                    shape_info=_si(expr),
                )
            )
        if isinstance(expr, FunctionCallIR):
            loc0 = expr.location or loc
            callee = self._ctx.bindings.get(expr.function_defid) if expr.function_defid is not None else None
            if callee is None or not isinstance(callee.expr, FunctionValueIR):
                return _z(loc0)
            fv = callee.expr
            args = [_clone_expr(arg) for arg in (expr.arguments or ())]
            params = list(fv.parameters or ())
            primal_subst: Dict[DefId, ExpressionIR] = {}
            tangent_subst: Dict[DefId, ExpressionIR] = {}
            for param, arg in zip(params, args):
                if param.defid is None:
                    continue
                primal_subst[param.defid] = arg
                tangent_subst[param.defid] = self._diff_expr(arg, seed_map, local_bindings, local_diffs, symbolic, loc0)
            if fv.custom_diff_body is not None:
                rewritten = _rewrite_custom_diff_body(fv.custom_diff_body, primal_subst, tangent_subst)
                return _simplify(rewritten)
            if fv.body is None:
                return _z(loc0)
            substituted = _substitute_identifiers(fv.body, primal_subst)
            return _simplify(self._diff_expr(substituted, seed_map, local_bindings, local_diffs, symbolic, loc0))
        if isinstance(expr, BuiltinCallIR):
            if expr.builtin_name in _ZERO_TANGENT_BUILTINS:
                return _z(expr.location or loc)
            return _z(expr.location or loc)
        if isinstance(expr, WhereExpressionIR):
            return _simplify(
                WhereExpressionIR(
                    self._diff_expr(expr.expr, seed_map, local_bindings, local_diffs, symbolic, expr.location or loc),
                    [_clone_expr(c) for c in (expr.constraints or ())],
                    expr.location or loc,
                    type_info=_ti(expr),
                    shape_info=_si(expr),
                    binding_constraints=list(expr.binding_constraints or ()) if expr.binding_constraints is not None else None,
                    guard_constraints=list(expr.guard_constraints or ()) if expr.guard_constraints is not None else None,
                )
            )
        raise RuntimeError(f"Autodiff compile-time diff does not support {type(expr).__name__}")


class _ProgramRewriter:
    def __init__(self, differentiator: _Differentiator) -> None:
        self._differentiator = differentiator

    def rewrite_statement_list(self, statements: Iterable[Any]) -> List[Any]:
        out: List[Any] = []
        for stmt in statements:
            out.extend(self.rewrite_statement(stmt))
        return out

    def rewrite_statement(self, stmt: Any) -> List[Any]:
        if isinstance(stmt, BindingIR):
            if stmt.expr is not None:
                stmt.expr = self.rewrite_expr(stmt.expr)
            return [stmt]
        if isinstance(stmt, ExpressionIR):
            return [self.rewrite_expr(stmt)]
        return [stmt]

    def _symbolic_print_literal(self, expr: ExpressionIR) -> Optional[LiteralIR]:
        loc = getattr(expr, "location", None) or _LOC0
        if isinstance(expr, DifferentialIR):
            body = _simplify(self._differentiator.standalone(expr.operand, loc, symbolic=True))
            lhs_name = getattr(expr.operand, "name", None) or "?"
            return LiteralIR(f"let @{lhs_name} = {body};", loc, type_info=STR)
        pair = _source_requested_quotient_pair(expr)
        if pair is not None:
            return LiteralIR(f"(@{getattr(expr.left.operand, 'name', 'y')} / @{getattr(expr.right.operand, 'name', 'x')}) · @{getattr(expr.right.operand, 'name', 'x')}", loc, type_info=STR)
        return None

    def rewrite_expr(self, expr: Optional[ExpressionIR]) -> Optional[ExpressionIR]:
        if expr is None:
            return None
        loc = getattr(expr, "location", None) or _LOC0
        pair = _source_requested_quotient_pair(expr)
        if pair is not None:
            num_defid, den_defid = pair
            numerator = expr.left.operand if isinstance(expr.left, DifferentialIR) else None
            denominator = expr.right.operand if isinstance(expr.right, DifferentialIR) else None
            if isinstance(numerator, IdentifierIR) and isinstance(denominator, IdentifierIR):
                target_binding = self._differentiator._ctx.bindings.get(num_defid)
                wrt_binding = self._differentiator._ctx.bindings.get(den_defid)
                type_info = getattr(target_binding, "expr", None).type_info if target_binding is not None and getattr(target_binding, "expr", None) is not None else _ti(expr)
                shape_info = getattr(target_binding, "expr", None).shape_info if target_binding is not None and getattr(target_binding, "expr", None) is not None else _si(expr)
                if wrt_binding is not None and _binding_shape(wrt_binding):
                    type_info = getattr(wrt_binding, "expr", None).type_info if getattr(wrt_binding, "expr", None) is not None else type_info
                return LazyJacobianIR(
                    target=_clone_expr(numerator),
                    wrt=_clone_expr(denominator),
                    location=loc,
                    type_info=type_info,
                    shape_info=shape_info,
                )
            return _simplify(self._differentiator.wrt(num_defid, den_defid, loc, symbolic=False))
        if isinstance(expr, DifferentialIR):
            return _simplify(self._differentiator.standalone(expr.operand, loc, symbolic=False))
        if isinstance(expr, BuiltinCallIR):
            if expr.builtin_name == "print" and len(expr.args or ()) == 1:
                literal = self._symbolic_print_literal(expr.args[0])
                if literal is not None:
                    expr.args = (literal,)
                    return expr
            expr.args = tuple(
                rewritten
                for arg in (expr.args or ())
                for rewritten in [self.rewrite_expr(arg)]
                if rewritten is not None
            )
            return expr
        if isinstance(expr, BinaryOpIR):
            expr.left = self.rewrite_expr(expr.left)
            expr.right = self.rewrite_expr(expr.right)
            return expr
        if isinstance(expr, UnaryOpIR):
            expr.operand = self.rewrite_expr(expr.operand)
            return expr
        if isinstance(expr, RectangularAccessIR):
            expr.array = self.rewrite_expr(expr.array)
            expr.indices = tuple(self.rewrite_expr(idx) for idx in (expr.indices or ()))
            return expr
        if isinstance(expr, CastExpressionIR):
            expr.expr = self.rewrite_expr(expr.expr)
            return expr
        if isinstance(expr, MemberAccessIR):
            expr.object = self.rewrite_expr(expr.object)
            return expr
        if isinstance(expr, ArrayLiteralIR):
            expr.elements = tuple(self.rewrite_expr(elem) for elem in (expr.elements or ()))
            return expr
        if isinstance(expr, TupleExpressionIR):
            expr.elements = tuple(self.rewrite_expr(elem) for elem in (expr.elements or ()))
            return expr
        if isinstance(expr, TupleAccessIR):
            expr.tuple_expr = self.rewrite_expr(expr.tuple_expr)
            return expr
        if isinstance(expr, BlockExpressionIR):
            expr.statements = tuple(self.rewrite_statement_list(expr.statements or ()))
            expr.final_expr = self.rewrite_expr(expr.final_expr)
            return expr
        if isinstance(expr, IfExpressionIR):
            expr.condition = self.rewrite_expr(expr.condition)
            expr.then_expr = self.rewrite_expr(expr.then_expr)
            expr.else_expr = self.rewrite_expr(expr.else_expr)
            return expr
        if isinstance(expr, RangeIR):
            expr.start = self.rewrite_expr(expr.start)
            expr.end = self.rewrite_expr(expr.end)
            return expr
        if isinstance(expr, ReductionExpressionIR):
            expr.body = self.rewrite_expr(expr.body)
            return expr
        if isinstance(expr, EinsteinIR):
            new_clauses: List[EinsteinClauseIR] = []
            for clause in expr.clauses or ():
                clause.value = self.rewrite_expr(clause.value)
                if clause.where_clause is not None:
                    clause.where_clause.constraints = tuple(
                        self.rewrite_expr(constraint) for constraint in (clause.where_clause.constraints or ())
                    )
                new_clauses.append(clause)
            expr.clauses = tuple(new_clauses)
            expr.shape = tuple(self.rewrite_expr(dim) for dim in (expr.shape or ())) if expr.shape is not None else None
            return expr
        if isinstance(expr, FunctionCallIR):
            expr.callee_expr = self.rewrite_expr(expr.callee_expr)
            expr.arguments = tuple(self.rewrite_expr(arg) for arg in (expr.arguments or ()))
            return expr
        if isinstance(expr, FunctionValueIR):
            if expr.body is not None:
                expr.body = self.rewrite_expr(expr.body)
            return expr
        if isinstance(expr, WhereExpressionIR):
            expr.expr = self.rewrite_expr(expr.expr)
            expr.constraints = tuple(self.rewrite_expr(c) for c in (expr.constraints or ()))
            if expr.guard_constraints is not None:
                expr.guard_constraints = tuple(self.rewrite_expr(c) for c in (expr.guard_constraints or ()))
            if expr.binding_constraints is not None:
                expr.binding_constraints = tuple(
                    binding if not isinstance(binding, BindingIR) or binding.expr is None else BindingIR(
                        binding.name,
                        self.rewrite_expr(binding.expr),
                        location=binding.location,
                        defid=binding.defid,
                        type_info=binding.type_info,
                    )
                    for binding in expr.binding_constraints
                )
            return expr
        return expr


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
            clear_autodiff_only_fields(ir)
            clear_custom_diff_body_everywhere(getattr(tcx, "function_ir_map", None))

    def _core(self, program: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        bindings = _binding_map(program, tcx)
        source_bindings = {
            did: _clone_ir_value(binding, {})
            for did, binding in (bindings or {}).items()
            if did is not None and isinstance(binding, BindingIR)
        }
        dep_cache = _DependencyQueryCache(source_bindings)
        binding_ctx = _AutodiffBindingContext(bindings=source_bindings, dep_cache=dep_cache)
        local_contexts = _local_binding_contexts(program)
        function_ir_map = getattr(tcx, "function_ir_map", None) or {}
        for binding in function_ir_map.values():
            if isinstance(binding, BindingIR):
                local_contexts.update(_local_binding_contexts(binding))
        source_local_contexts = {
            did: {
                local_did: _clone_ir_value(binding, {})
                for local_did, binding in (context or {}).items()
            }
            for did, context in (local_contexts or {}).items()
        }
        source_function_ir_map = {
            did: _clone_ir_value(binding, {})
            for did, binding in (function_ir_map or {}).items()
            if did is not None and isinstance(binding, BindingIR)
        }
        targets = self._collect_targets(program, tcx)
        if not targets.diff_targets and not targets.quotient_pairs:
            self._record_analysis(
                tcx,
                targets,
                binding_ctx=binding_ctx,
                differentiator=None,
                local_contexts=source_local_contexts,
                function_ir_map=source_function_ir_map,
            )
            return program

        resolver = getattr(tcx, "resolver", None)
        differentiator = _Differentiator(binding_ctx, resolver, local_contexts=source_local_contexts)
        rewriter = _ProgramRewriter(differentiator)
        program.statements = rewriter.rewrite_statement_list(program.statements or ())
        program.bindings = [stmt for stmt in (program.statements or []) if isinstance(stmt, BindingIR)]

        for binding in function_ir_map.values():
            if isinstance(binding, BindingIR) and isinstance(binding.expr, FunctionValueIR):
                binding.expr = rewriter.rewrite_expr(binding.expr)

        self._record_analysis(
            tcx,
            targets,
            binding_ctx=binding_ctx,
            differentiator=differentiator,
            local_contexts=source_local_contexts,
            function_ir_map=source_function_ir_map,
        )
        return program

    def _collect_targets(self, program: ProgramIR, tcx: TyCtxt) -> _AutodiffTargets:
        targets = _AutodiffTargets()
        for stmt in program.statements or ():
            if isinstance(stmt, BindingIR) and stmt.expr is not None:
                diff_targets, quotient_pairs = _collect_targets_expr(stmt.expr)
                targets.diff_targets.extend(diff_targets)
                targets.quotient_pairs.extend(quotient_pairs)
            elif isinstance(stmt, ExpressionIR):
                diff_targets, quotient_pairs = _collect_targets_expr(stmt)
                targets.diff_targets.extend(diff_targets)
                targets.quotient_pairs.extend(quotient_pairs)
        for binding in (getattr(tcx, "function_ir_map", None) or {}).values():
            if isinstance(binding, BindingIR) and isinstance(binding.expr, FunctionValueIR):
                if binding.expr.body is not None:
                    diff_targets, quotient_pairs = _collect_targets_expr(binding.expr.body)
                    targets.diff_targets.extend(diff_targets)
                    targets.quotient_pairs.extend(quotient_pairs)
                if binding.expr.custom_diff_body is not None:
                    diff_targets, quotient_pairs = _collect_targets_expr(binding.expr.custom_diff_body)
                    targets.diff_targets.extend(diff_targets)
                    targets.quotient_pairs.extend(quotient_pairs)
        return targets

    def _record_analysis(
        self,
        tcx: TyCtxt,
        targets: _AutodiffTargets,
        *,
        binding_ctx: _AutodiffBindingContext,
        differentiator: Optional[_Differentiator],
        local_contexts: Dict[DefId, Dict[DefId, BindingIR]],
        function_ir_map: Dict[Any, Any],
    ) -> None:
        graph_binding_by_defid = {
            did: _clone_ir_value(binding, {})
            for did, binding in (binding_ctx.bindings or {}).items()
            if did is not None
        }
        graph_function_ir_map = {
            did: _clone_ir_value(binding, {})
            for did, binding in (function_ir_map or {}).items()
            if did is not None and isinstance(binding, BindingIR)
        }
        graph_local_contexts = {
            did: {
                local_did: _clone_ir_value(binding, {})
                for local_did, binding in (context or {}).items()
            }
            for did, context in (local_contexts or {}).items()
        }
        runtime_jvp_templates_by_pair: Dict[Tuple[DefId, DefId], Dict[str, Any]] = {}
        runtime_vjp_templates_by_target: Dict[DefId, Dict[str, Any]] = {}
        # Plain-IR lowering removes autodiff request nodes before backend execution,
        # so the old runtime JVP/VJP template analysis is intentionally left empty.
        tcx.set_analysis(
            AutodiffPass,
            {
                "compiled_graph": None,
                "differential_targets": set(targets.diff_targets),
                "pending_differential_slot_by_defid": {},
                "pending_quotient_slot_by_defid": {},
                "source_quotient_slot_by_defid": {},
                "graph_program": None,
                "graph_binding_by_defid": graph_binding_by_defid,
                "graph_function_ir_map": graph_function_ir_map,
                "graph_leaf_defids": set(getattr(binding_ctx, "_leaf_bindings", {}) or {}),
                "graph_self_recursive_defids": set(getattr(binding_ctx, "_self_recursive_bindings", set()) or set()),
                "graph_local_contexts_by_defid": graph_local_contexts,
                "runtime_jvp_templates_by_pair": runtime_jvp_templates_by_pair,
                "runtime_vjp_templates_by_target": runtime_vjp_templates_by_target,
                "graph_builtin_requests_by_expr_id": {},
            },
        )


__all__ = [
    "AutodiffPass",
    "DIFF_PREFIX",
    "USER_DIFF_PREFIX",
    "clear_custom_diff_body_everywhere",
]
