"""Autodiff pass — expand ``@expr`` and ``@y/@x`` into plain IR.

Design: ``docs/AUTODIFF_DESIGN.md`` · overview: ``docs/AUTODIFF_HIGHLIGHTS.md``

Forward mode propagates differentials (tangents).  For ``y = f(x₁, x₂, …)``
we emit ``d(y) = Σᵢ (∂f/∂xᵢ) · d(xᵢ)`` in execution order.

Float-only constraint
---------------------
Autodiff is only defined for **floating-point** types (``f32``, ``f64``, or
tensors thereof).  Differentiating an integer-typed expression is
mathematically undefined — integers form a discrete set with no smooth
structure.  ``print(@y)`` and ``@y/@x`` are therefore a type error if ``y``
does not have a float type.  This mirrors Julia's ForwardDiff.jl, which
requires ``AbstractFloat`` inputs and rejects integer arguments.

Practically, all literal coefficients produced by the pass (e.g. the ``2.0``
in ``d(x²) = 2.0 * x * _@x``) are float-typed ``LiteralIR`` nodes.  The
``_@*`` tangent bindings carry the same type as their primal.

Concepts
--------
- ``@expr``   — differential of *expr* (same shape/type as *expr*).
- ``@y / @x`` — derivative (Jacobian coefficient), extracted from ``d(y)``
                 by setting ``d(x)=1`` and all other leaf tangents to ``0``.
- ``_@*``     — internal prefix for differential bindings (user-visible in
                 ``print(@y)`` output as ``let _@name: type = expr;``).

Phases
------
1. **Analysis** — collect targets, build dep graph, topo-sort.
2. **Forward differentiation** — ``DiffVisitor`` computes ``d(expr)`` via
   ``DiffContext``; ``_@*`` bindings inserted after their primals.
3. **Expansion** — replace ``DifferentialIR`` with ``_@*`` refs;
   ``JacobianVisitor`` for ``@y/@x``; format ``print(@y)``.
4. **Cleanup** — strip ``DifferentialType``, remove ``DiffRuleIR``.

Pass order / IR shape
---------------------
``AutodiffPass`` runs **before** ``EinsteinLoweringPass``.  Visitors reject
``LoweredEinsteinIR``, ``LoweredRecurrenceIR``, and other lowered-* nodes;
differentiation is defined on ``EinsteinIR`` (and ordinary expressions) only.

Public API (imported by other modules)
--------------------------------------
- ``AutodiffPass``
- ``DIFF_PREFIX``
- ``clear_custom_diff_body_everywhere``
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, NoReturn, Optional, Set, Tuple, Union, cast

from .base import BasePass, TyCtxt
from .type_inference import TypeInferencePass
from .shape_analysis import UnifiedShapeAnalysisPass
from ..ir.nodes import (
    ArrayLiteralIR,
    ProgramIR,
    BindingIR,
    BlockExpressionIR,
    CastExpressionIR,
    DifferentialIR,
    BinaryOpIR,
    UnaryOpIR,
    BuiltinCallIR,
    LiteralIR,
    IdentifierIR,
    ExpressionIR,
    IRVisitor,
    IfExpressionIR,
    InterpolatedStringIR,
    LambdaIR,
    RectangularAccessIR,
    IndexVarIR,
    IndexRestIR,
    RangeIR,
    MemberAccessIR,
    FunctionCallIR,
    FunctionValueIR,
    EinsteinIR,
    EinsteinClauseIR,
    ReductionExpressionIR,
    SelectAtArgmaxIR,
    WhereExpressionIR,
    WhereClauseIR,
    DiffRuleIR,
    TupleExpressionIR,
    TupleAccessIR,
    MatchExpressionIR,
    PipelineExpressionIR,
    ArrayComprehensionIR,
    TryExpressionIR,
    JaggedAccessIR,
    is_function_binding,
)
from ..shared.types import (
    BinaryOp, UnaryOp, PrimitiveType, UNKNOWN, F32, I32, STR, BOOL, Type,
    RectangularType,
    strip_differential_types_deep, ReductionOp,
)
from ..shared.defid import DefId, DefType
from ..shared.source_location import SourceLocation

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

DIFF_PREFIX = "_@"       # compiler-introduced intermediates (callee-locals, call tangents)
USER_DIFF_PREFIX = "@"  # tangents of user-written bindings (visible in print(@y))

def _is_diff_name(name: str) -> bool:
    """Return True for any differential binding name (either prefix)."""
    return name.startswith(DIFF_PREFIX) or name.startswith(USER_DIFF_PREFIX)

_LOC0 = SourceLocation("", 0, 0)

# ═══════════════════════════════════════════════════════════════════════════
# Tiny helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fl(v: float, loc: SourceLocation) -> LiteralIR:
    return LiteralIR(float(v), loc, type_info=F32)

def _z(loc: SourceLocation) -> LiteralIR:
    """Literal scalar zero (float).

    Allowed uses (do not use ``_z`` to mean "unsupported"):
    - Derivative of a float literal; constant array literal tangents in DiffVisitor.
    - JacobianVisitor: identifier independent of ``wrt`` (true zero partial).
    - Algebraic simplification in ``_simplify`` when a factor is literally zero.
    - Missing ``else`` branch / empty sum in structured IR builders where zero is correct.
    - ``stmt_partial`` when the simplified partial is ``_is_zero`` (literal 0).
    - Quotient expansion when the numerator does not depend on the denominator variable.
    - AutodiffPass: seed tangents for quotient leaves and missing deps in the dep map.
    - Einstein forward/Jacobian helpers when no clauses contribute (all-zero tensor).
    - Binding RHS absent or function value (no scalar/tensor tangent binding).

    Unsupported IR nodes must call ``_unsupported_autodiff_ir`` (raise) instead of ``_z``.
    """
    return _fl(0, loc)


def _unsupported_autodiff_ir(visitor: str, node: object) -> NoReturn:
    """Fail fast: unsupported IR must not silently yield a zero derivative."""
    raise ValueError(
        f"Autodiff: {visitor} does not support IR node {type(node).__name__} "
        f"(do not substitute literal zero; add a visitor or lowering rule)"
    )


def _reject_lowered_ir(visitor: str, node: object) -> NoReturn:
    """Autodiff is defined on EinsteinIR; lowered-* nodes appear only after EinsteinLoweringPass."""
    raise ValueError(
        f"Autodiff: {visitor} does not support {type(node).__name__}. "
        f"Run AutodiffPass before EinsteinLoweringPass; differentiate EinsteinIR only, not lowered-* IR."
    )


# Builtins that do not contribute a float tangent w.r.t. tensor inputs (metadata, shape, asserts).
_AD_ZERO_TANGENT_BUILTINS = frozenset({"assert", "len", "typeof", "shape"})


def _is_zero(e: ExpressionIR) -> bool:
    return isinstance(e, LiteralIR) and e.value == 0


def _is_sum_reduction_of_zero(e: ExpressionIR) -> bool:
    return (
        isinstance(e, ReductionExpressionIR)
        and e.operation == ReductionOp.SUM
        and _is_zero(e.body)
    )

def _ti(n: Any) -> Any:
    return getattr(n, "type_info", None)

def _si(n: Any) -> Any:
    return getattr(n, "shape_info", None)


def _cast_target_has_zero_tangent(target_type: Any) -> bool:
    """Casts to discrete/metadata targets are treated as zero tangent."""
    return isinstance(target_type, PrimitiveType) and target_type.name in {
        "i8",
        "i32",
        "i64",
        "bool",
        "str",
        "range",
        "unit",
    }

# ═══════════════════════════════════════════════════════════════════════════
# Structural equality (for simplification)
# ═══════════════════════════════════════════════════════════════════════════

def _seq(a: ExpressionIR, b: ExpressionIR) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, IdentifierIR):
        if a.defid is not None and b.defid is not None:
            return a.defid == b.defid
        if a.defid is None and b.defid is None:
            return a.name == b.name
        return False
    if isinstance(a, LiteralIR):
        return a.value == b.value
    if isinstance(a, IndexVarIR):
        return a.defid is not None and a.defid == b.defid
    if isinstance(a, BinaryOpIR):
        return a.operator == b.operator and _seq(a.left, b.left) and _seq(a.right, b.right)
    if isinstance(a, UnaryOpIR):
        return a.operator == b.operator and _seq(a.operand, b.operand)
    if isinstance(a, RectangularAccessIR):
        ai, bi = a.indices or [], b.indices or []
        return (_seq(a.array, b.array) and len(ai) == len(bi)
                and all(_seq(x, y) for x, y in zip(ai, bi)))
    if isinstance(a, FunctionCallIR):
        aa, ba = a.arguments or [], b.arguments or []
        return (getattr(a, "function_defid", None) == getattr(b, "function_defid", None)
                and len(aa) == len(ba) and all(_seq(x, y) for x, y in zip(aa, ba)))
    if isinstance(a, BuiltinCallIR):
        aa, ba = a.args or [], b.args or []
        return (a.builtin_name == b.builtin_name
                and len(aa) == len(ba) and all(_seq(x, y) for x, y in zip(aa, ba)))
    if isinstance(a, CastExpressionIR):
        return getattr(a, "target_type", None) == getattr(b, "target_type", None) and _seq(a.expr, b.expr)
    return False


def _rc_index_lists_equivalent(
    outer: List[ExpressionIR], inner: List[ExpressionIR]
) -> bool:
    """True when outer and inner index tuples refer to the same slots.

    Replay/substitution can clone ``IndexVarIR`` with fresh DefIds; ``_seq`` then
    fails and ``RectangularAccessIR`` would double-wrap.  Same-type index vars
    with the same name are treated as equivalent for this collapse only.
    """
    oi, ii = list(outer or []), list(inner or [])
    if len(oi) != len(ii):
        return False
    for a, b in zip(oi, ii):
        if _seq(a, b):
            continue
        if isinstance(a, IndexVarIR) and isinstance(b, IndexVarIR):
            if a.name == b.name:
                continue
            return False
        if isinstance(a, IndexRestIR) and isinstance(b, IndexRestIR):
            if a.name == b.name:
                continue
            return False
        return False
    return True

# keep public name used in old code
_ir_structurally_equal = _seq

# ═══════════════════════════════════════════════════════════════════════════
# Algebraic simplification
# ═══════════════════════════════════════════════════════════════════════════

def _simplify(expr: ExpressionIR, loc: SourceLocation) -> ExpressionIR:
    if isinstance(expr, BinaryOpIR):
        L = _simplify(expr.left, loc)
        R = _simplify(expr.right, loc)
        op = expr.operator
        lz, rz = _is_zero(L), _is_zero(R)
        l1 = isinstance(L, LiteralIR) and L.value == 1
        r1 = isinstance(R, LiteralIR) and R.value == 1
        ll, rl = isinstance(L, LiteralIR), isinstance(R, LiteralIR)

        if op == BinaryOp.ADD:
            if lz: return R
            if rz: return L
            if isinstance(L, ReductionExpressionIR) and L.operation == ReductionOp.SUM and _is_zero(L.body):
                return R
            if isinstance(R, ReductionExpressionIR) and R.operation == ReductionOp.SUM and _is_zero(R.body):
                return L
            if ll and rl: return _fl(L.value + R.value, loc)
            if _seq(L, R): return BinaryOpIR(BinaryOp.MUL, _fl(2, loc), L, loc)
        elif op == BinaryOp.SUB:
            if rz: return L
            if ll and rl: return _fl(L.value - R.value, loc)
        elif op == BinaryOp.MUL:
            if lz or rz: return _z(loc)
            if l1: return R
            if r1: return L
            if ll and rl: return _fl(L.value * R.value, loc)
        elif op == BinaryOp.DIV:
            if lz: return _z(loc)
            if (isinstance(R, BinaryOpIR) and R.operator == BinaryOp.POW
                    and isinstance(R.right, LiteralIR) and R.right.value == 2):
                base = R.left
                if isinstance(L, BinaryOpIR) and L.operator == BinaryOp.MUL:
                    if _seq(L.left, base):
                        return BinaryOpIR(BinaryOp.DIV, L.right, base, loc)
                    if _seq(L.right, base):
                        return BinaryOpIR(BinaryOp.DIV, L.left, base, loc)
        elif op == BinaryOp.POW:
            if r1: return L

        return BinaryOpIR(op, L, R, loc)

    if isinstance(expr, UnaryOpIR):
        inner = _simplify(expr.operand, loc)
        if expr.operator == UnaryOp.NEG and isinstance(inner, UnaryOpIR) and inner.operator == UnaryOp.NEG:
            return inner.operand
        return UnaryOpIR(expr.operator, inner, expr.location or loc)

    if isinstance(expr, EinsteinIR):
        nc = []
        for c in expr.clauses or []:
            sv = _unwrap_trivial_einstein_rhs(_simplify(c.value, c.location or loc))
            nc.append(EinsteinClauseIR(indices=c.indices, value=sv, location=c.location,
                      where_clause=c.where_clause, variable_ranges=c.variable_ranges))
        return EinsteinIR(clauses=nc, shape=expr.shape, element_type=expr.element_type,
                          location=expr.location, type_info=_ti(expr), shape_info=_si(expr))

    if isinstance(expr, ReductionExpressionIR):
        sb = _simplify(expr.body, loc)
        return ReductionExpressionIR(expr.operation, expr.loop_vars, sb, expr.location,
               where_clause=expr.where_clause, loop_var_ranges=expr.loop_var_ranges,
               type_info=_ti(expr), shape_info=_si(expr))

    if isinstance(expr, IfExpressionIR):
        nc = _simplify(expr.condition, loc)
        nt = _simplify(expr.then_expr, loc)
        ne = _simplify(expr.else_expr, loc) if expr.else_expr is not None else None
        return IfExpressionIR(
            nc, nt, expr.location or loc, else_expr=ne, type_info=_ti(expr), shape_info=_si(expr)
        )

    if isinstance(expr, BlockExpressionIR):
        nss: List[Any] = []
        for s in expr.statements or []:
            if isinstance(s, BindingIR) and s.expr is not None:
                nss.append(
                    BindingIR(
                        name=s.name,
                        expr=_simplify(s.expr, loc),
                        location=s.location,
                        defid=s.defid,
                        type_info=_ti(s),
                    )
                )
            elif isinstance(s, ExpressionIR):
                nss.append(_simplify(s, loc))
            else:
                nss.append(s)
        nf = _simplify(expr.final_expr, loc) if expr.final_expr is not None else None
        return BlockExpressionIR(
            nss, expr.location or loc, nf, type_info=_ti(expr), shape_info=_si(expr)
        )

    if isinstance(expr, CastExpressionIR):
        ie = _simplify(expr.expr, loc)
        return CastExpressionIR(
            ie, expr.target_type, expr.location or loc, type_info=_ti(expr), shape_info=_si(expr)
        )

    if isinstance(expr, RectangularAccessIR):
        aa = _simplify(expr.array, loc)
        nidx = [_simplify(i, loc) for i in (expr.indices or [])]
        return RectangularAccessIR(
            aa, nidx, expr.location or loc, type_info=_ti(expr), shape_info=_si(expr)
        )

    return expr


def _unwrap_trivial_einstein_rhs(expr: ExpressionIR) -> ExpressionIR:
    """Einstein clause RHS is a single expression unless it has real ``let`` bindings."""
    while isinstance(expr, BlockExpressionIR) and expr.final_expr is not None:
        bindings = [
            s for s in (expr.statements or [])
            if isinstance(s, BindingIR) and s.defid is not None and s.expr is not None
        ]
        if bindings:
            break
        expr = expr.final_expr
    return expr


# ═══════════════════════════════════════════════════════════════════════════
# log() lookup for power chain rule
# ═══════════════════════════════════════════════════════════════════════════

def _log_call(arg: ExpressionIR, bindings: Dict[DefId, Any], resolver: Any, loc: SourceLocation) -> ExpressionIR:
    out_ti = _ti(arg) or F32
    out_si = _si(arg)
    for did, b in bindings.items():
        if not isinstance(b, BindingIR) or not isinstance(b.expr, FunctionValueIR):
            continue
        nm = b.name or ""
        if nm == "log" or nm.endswith("::log") or nm.startswith("log_") or "::log_" in nm:
            callee = IdentifierIR(
                nm,
                loc,
                did,
                type_info=_ti(b) or (_ti(b.expr) if b.expr is not None else None) or UNKNOWN,
            )
            return FunctionCallIR(
                callee_expr=callee,
                location=loc,
                arguments=[arg],
                type_info=out_ti,
                shape_info=out_si,
            )
    return FunctionCallIR(
        callee_expr=IdentifierIR("log", loc, None, type_info=UNKNOWN),
        location=loc,
        arguments=[arg],
        type_info=out_ti,
        shape_info=out_si,
    )


def _pow_chain(node: BinaryOpIR, dL: ExpressionIR, dR: ExpressionIR,
               bindings: Dict[DefId, Any], resolver: Any, loc: SourceLocation) -> ExpressionIR:
    a, b = node.left, node.right
    if isinstance(b, LiteralIR):
        n = b.value
        return BinaryOpIR(BinaryOp.MUL,
                   BinaryOpIR(BinaryOp.MUL, _fl(n, loc),
                              BinaryOpIR(BinaryOp.POW, a, _fl(n - 1, loc), loc), loc),
                   dL, loc)
    if isinstance(a, LiteralIR):
        c = a.value
        return BinaryOpIR(BinaryOp.MUL,
                   BinaryOpIR(BinaryOp.MUL, node, _log_call(_fl(c, loc), bindings, resolver, loc), loc),
                   dR, loc)
    log_a = _log_call(a, bindings, resolver, loc)
    t1 = BinaryOpIR(BinaryOp.MUL, BinaryOpIR(BinaryOp.DIV, b, a, loc), dL, loc)
    t2 = BinaryOpIR(BinaryOp.MUL, log_a, dR, loc)
    return BinaryOpIR(BinaryOp.MUL, node, BinaryOpIR(BinaryOp.ADD, t1, t2, loc), loc)

# ═══════════════════════════════════════════════════════════════════════════
# Substitution
# ═══════════════════════════════════════════════════════════════════════════


class _Rewriter(IRVisitor[ExpressionIR]):
    """Default tree rewrite: rebuild compound nodes via ``child.accept(self)``."""

    def __init__(self, loc: SourceLocation) -> None:
        self._loc = loc

    def _rw_wc(self, wc: Any) -> Any:
        if wc is None:
            return None
        cc = getattr(wc, "constraints", None)
        if not cc:
            return wc
        return WhereClauseIR(constraints=[c.accept(self) for c in cc], location=self._loc)

    def _rw_lr(self, lr: Optional[Dict]) -> Optional[Dict]:
        if not lr:
            return lr
        out: Dict = {}
        for k, v in lr.items():
            if isinstance(v, RangeIR):
                out[k] = RangeIR(
                    start=v.start.accept(self),
                    end=v.end.accept(self),
                    location=v.location or self._loc,
                    type_info=v.type_info,
                )
            elif isinstance(v, ExpressionIR):
                out[k] = v.accept(self)
            else:
                out[k] = v
        return out

    def visit_literal(self, n: LiteralIR) -> ExpressionIR:
        return n

    def visit_identifier(self, n: IdentifierIR) -> ExpressionIR:
        return n

    def visit_index_var(self, n: IndexVarIR) -> ExpressionIR:
        return n

    def visit_index_rest(self, n: IndexRestIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_binary_op(self, n: BinaryOpIR) -> ExpressionIR:
        loc = n.location or self._loc
        return BinaryOpIR(
            n.operator,
            n.left.accept(self),
            n.right.accept(self),
            loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_unary_op(self, n: UnaryOpIR) -> ExpressionIR:
        return UnaryOpIR(
            n.operator,
            n.operand.accept(self),
            n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_function_call(self, n: FunctionCallIR) -> ExpressionIR:
        return FunctionCallIR(
            callee_expr=n.callee_expr,
            location=n.location or self._loc,
            arguments=[a.accept(self) for a in (n.arguments or [])],
            module_path=getattr(n, "module_path", None),
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_rectangular_access(self, n: RectangularAccessIR) -> ExpressionIR:
        return RectangularAccessIR(
            n.array.accept(self),
            [i.accept(self) for i in (n.indices or [])],
            n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_jagged_access(self, n: JaggedAccessIR) -> ExpressionIR:
        return n

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        ns: List[Any] = []
        for s in n.statements or []:
            if isinstance(s, BindingIR) and s.expr is not None:
                ns.append(
                    BindingIR(
                        name=s.name,
                        expr=s.expr.accept(self),
                        location=s.location,
                        defid=s.defid,
                        type_info=_ti(s),
                    )
                )
            else:
                ns.append(s)
        nf = n.final_expr.accept(self) if n.final_expr is not None else None
        return BlockExpressionIR(ns, loc, nf, type_info=_ti(n), shape_info=_si(n))

    def visit_if_expression(self, n: IfExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        return IfExpressionIR(
            condition=n.condition.accept(self),
            then_expr=n.then_expr.accept(self),
            location=loc,
            else_expr=n.else_expr.accept(self) if n.else_expr is not None else None,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_lambda(self, n: LambdaIR) -> ExpressionIR:
        return n

    def visit_differential(self, n: DifferentialIR) -> ExpressionIR:
        return DifferentialIR(
            n.operand.accept(self),
            n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_range(self, n: RangeIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_array_comprehension(self, n: ArrayComprehensionIR) -> ExpressionIR:
        return n

    def visit_module(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_array_literal(self, n: ArrayLiteralIR) -> ExpressionIR:
        return n

    def visit_tuple_expression(self, n: TupleExpressionIR) -> ExpressionIR:
        return n

    def visit_tuple_access(self, n: TupleAccessIR) -> ExpressionIR:
        return n

    def visit_interpolated_string(self, n: InterpolatedStringIR) -> ExpressionIR:
        return n

    def visit_cast_expression(self, n: CastExpressionIR) -> ExpressionIR:
        return CastExpressionIR(
            n.expr.accept(self),
            n.target_type,
            n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_member_access(self, n: MemberAccessIR) -> ExpressionIR:
        return MemberAccessIR(
            n.object.accept(self),
            n.member,
            n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_try_expression(self, n: TryExpressionIR) -> ExpressionIR:
        return n

    def visit_match_expression(self, n: MatchExpressionIR) -> ExpressionIR:
        return n

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> ExpressionIR:
        return ReductionExpressionIR(
            n.operation,
            n.loop_vars,
            n.body.accept(self),
            n.location or self._loc,
            where_clause=self._rw_wc(n.where_clause),
            loop_var_ranges=self._rw_lr(n.loop_var_ranges),
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_builtin_call(self, n: BuiltinCallIR) -> ExpressionIR:
        return BuiltinCallIR(
            n.builtin_name,
            [a.accept(self) for a in (n.args or [])],
            n.location or self._loc,
            defid=getattr(n, "defid", None),
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_where_expression(self, n: WhereExpressionIR) -> ExpressionIR:
        return n

    def visit_pipeline_expression(self, n: PipelineExpressionIR) -> ExpressionIR:
        return n

    def visit_literal_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_identifier_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_wildcard_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_tuple_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_array_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_rest_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_guard_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_or_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_constructor_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_binding_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_range_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_function_value(self, n: FunctionValueIR) -> ExpressionIR:
        return n

    def visit_einstein(self, n: EinsteinIR) -> ExpressionIR:
        nc: List[EinsteinClauseIR] = []
        for c in n.clauses or []:
            cv = c.value.accept(self) if c.value is not None else None
            if cv is not None:
                cv = _unwrap_trivial_einstein_rhs(cv)
            nc.append(
                EinsteinClauseIR(
                    indices=[i.accept(self) for i in (c.indices or [])],
                    value=cv,
                    location=c.location,
                    where_clause=self._rw_wc(c.where_clause),
                    variable_ranges=dict(self._rw_lr(c.variable_ranges) or {}),
                )
            )
        new_shape = tuple(s.accept(self) for s in n.shape) if n.shape else n.shape
        return EinsteinIR(
            clauses=nc,
            shape=new_shape,
            element_type=n.element_type,
            location=n.location,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_einstein_clause(self, n: EinsteinClauseIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_binding(self, n: BindingIR) -> ExpressionIR:
        if n.expr is None:
            return cast(ExpressionIR, n)
        return n.expr.accept(self)

    def visit_program(self, n: ProgramIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> ExpressionIR:
        loc = n.location or self._loc
        pb = n.primal_body.accept(self) if n.primal_body is not None else None
        db = n.diff_body.accept(self) if n.diff_body is not None else None
        return SelectAtArgmaxIR(
            pb,
            db,
            n.loop_vars,
            loop_var_ranges=self._rw_lr(n.loop_var_ranges),
            location=loc,
            type_info=_ti(n),
            shape_info=_si(n),
            use_argmin=n.use_argmin,
        )

    def visit_lowered_reduction(self, n: Any) -> ExpressionIR:
        return n

    def visit_lowered_select_at_argmax(self, n: Any) -> ExpressionIR:
        return n

    def visit_lowered_comprehension(self, n: Any) -> ExpressionIR:
        return n

    def visit_lowered_einstein_clause(self, n: Any) -> ExpressionIR:
        return n

    def visit_lowered_einstein(self, n: Any) -> ExpressionIR:
        return n

    def visit_lowered_recurrence(self, n: Any) -> ExpressionIR:
        return n


class _SubstVisitor(_Rewriter):
    def __init__(self, m: Dict[DefId, ExpressionIR], loc: SourceLocation) -> None:
        super().__init__(loc)
        self._m = m

    def _clone_subst_expr(self, expr: ExpressionIR) -> ExpressionIR:
        """Cheap clone for common leaf substitutions; fall back to deepcopy for structured IR."""
        if isinstance(expr, LiteralIR):
            return LiteralIR(
                expr.value,
                expr.location or self._loc,
                type_info=_ti(expr),
                shape_info=_si(expr),
            )
        if isinstance(expr, IdentifierIR):
            return IdentifierIR(
                expr.name,
                expr.location or self._loc,
                expr.defid,
                type_info=_ti(expr),
                shape_info=_si(expr),
            )
        if isinstance(expr, IndexVarIR):
            return IndexVarIR(
                expr.name,
                expr.location or self._loc,
                expr.defid,
                range_ir=expr.range_ir,
                type_info=_ti(expr),
                shape_info=_si(expr),
            )
        if isinstance(expr, IndexRestIR):
            return IndexRestIR(
                expr.name,
                expr.location or self._loc,
                expr.defid,
                type_info=_ti(expr),
                shape_info=_si(expr),
            )
        return copy.deepcopy(expr)

    def visit_identifier(self, n: IdentifierIR) -> ExpressionIR:
        if n.defid is not None and n.defid in self._m:
            return self._clone_subst_expr(self._m[n.defid])
        return n

    def visit_index_var(self, n: IndexVarIR) -> ExpressionIR:
        if n.defid is not None and n.defid in self._m:
            return self._clone_subst_expr(self._m[n.defid])
        return n


class _SubstDiffsVisitor(_SubstVisitor):
    def __init__(
        self,
        pm: Dict[DefId, ExpressionIR],
        dm: Dict[DefId, ExpressionIR],
        loc: SourceLocation,
    ) -> None:
        super().__init__(pm, loc)
        self._dm = dm

    def visit_differential(self, n: DifferentialIR) -> ExpressionIR:
        op = n.operand
        if isinstance(op, IdentifierIR) and op.defid is not None and op.defid in self._dm:
            return self._clone_subst_expr(self._dm[op.defid])
        return DifferentialIR(
            op.accept(self),
            n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )


def _sub(expr: ExpressionIR, m: Dict[DefId, ExpressionIR], loc: SourceLocation) -> ExpressionIR:
    """Replace IdentifierIR whose defid ∈ m."""
    if expr is None:
        return expr  # type: ignore[return-value]
    return expr.accept(_SubstVisitor(m, loc))


_substitute = _sub


def _sub_wd(
    expr: ExpressionIR,
    pm: Dict[DefId, ExpressionIR],
    dm: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
) -> ExpressionIR:
    """Substitute identifiers AND DifferentialIR(@param) nodes."""
    if expr is None:
        return expr  # type: ignore[return-value]
    return expr.accept(_SubstDiffsVisitor(pm, dm, loc))


_substitute_with_diffs = _sub_wd


def _fresh_prod_interior_index_names(loop_names: Tuple[str, ...]) -> List[str]:
    """Names for cloned ``prod`` binders so they do not reuse the partial-index name in ``/ body``."""
    if len(loop_names) == 1 and loop_names[0] == "j":
        return ["k"]
    if len(loop_names) == 1 and loop_names[0] == "i":
        return ["k"]
    return [f"{n}_p{i}" for i, n in enumerate(loop_names)]


def _prod_reduction_clone_interior_indices(
    n: ReductionExpressionIR,
    loc: SourceLocation,
    resolver: Any,
) -> ReductionExpressionIR:
    """Clone ``prod`` primal: same value, fresh loop DefIds and distinct names from ``n.body`` indices."""
    if resolver is None:
        raise ValueError("Autodiff: prod interior clone requires resolver for fresh DefIds")
    lv = list(n.loop_vars or ())
    if not lv:
        return n
    new_names = _fresh_prod_interior_index_names(tuple(v.name for v in lv))
    if len(new_names) != len(lv):
        raise ValueError("Autodiff: prod interior name list mismatch")
    did_sub: Dict[DefId, ExpressionIR] = {}
    new_lvs: List[Union[IndexVarIR, IdentifierIR]] = []
    new_ranges: Dict[DefId, RangeIR] = {}
    old_ranges = n.loop_var_ranges or {}
    for v, interior_name in zip(lv, new_names):
        if v.defid is None:
            raise ValueError("Autodiff: prod reduction loop var missing defid")
        new_did = resolver.allocate_for_local()
        vl = v.location or loc
        if isinstance(v, IndexVarIR):
            nw = IndexVarIR(
                interior_name,
                vl,
                defid=new_did,
                range_ir=v.range_ir,
                type_info=_ti(v),
                shape_info=_si(v),
            )
        else:
            nw = IdentifierIR(interior_name, vl, defid=new_did, type_info=_ti(v), shape_info=_si(v))
        new_lvs.append(nw)
        did_sub[v.defid] = nw
        r = old_ranges.get(v.defid)
        if r is not None:
            new_ranges[new_did] = r
    new_body = _sub(n.body, did_sub, loc)
    return ReductionExpressionIR(
        n.operation,
        new_lvs,
        new_body,
        n.location or loc,
        where_clause=n.where_clause,
        loop_var_ranges=new_ranges,
        type_info=_ti(n),
        shape_info=_si(n),
    )


def _prod_pullback_via_sum(
    n: ReductionExpressionIR,
    d_body: ExpressionIR,
    loc: SourceLocation,
    resolver: Any,
) -> ExpressionIR:
    """d(∏_j f_j) = ∑_j ((∏ f) / f_j · df_j). SUM over j ensures DIV's lhs (cloned full prod) is evaluated in its own reduction before f_j uses the outer j."""
    if resolver is None:
        raise ValueError("Autodiff: prod pullback requires resolver for fresh DefIds")
    num = _prod_reduction_clone_interior_indices(n, loc, resolver)
    inner = BinaryOpIR(
        BinaryOp.MUL,
        BinaryOpIR(BinaryOp.DIV, num, n.body, loc),
        d_body,
        loc,
    )
    return ReductionExpressionIR(
        ReductionOp.SUM,
        list(n.loop_vars or ()),
        inner,
        loc,
        where_clause=n.where_clause,
        loop_var_ranges=n.loop_var_ranges,
        type_info=_ti(n),
        shape_info=_si(n),
    )


# ═══════════════════════════════════════════════════════════════════════════
# DefId collector
# ═══════════════════════════════════════════════════════════════════════════

# We define a single _ZERO_VISITOR_STUBS string technique is not available,
# so all pattern stubs are written out explicitly.

class _DefIdCollector(IRVisitor[None]):
    """Collect all DefIds referenced in an expression tree."""
    def __init__(self) -> None:
        self.defids: Set[DefId] = set()

    # -- meaningful visits ---------------------------------------------------
    def visit_identifier(self, n: IdentifierIR) -> None:
        if n.defid is not None: self.defids.add(n.defid)
    def visit_literal(self, n: LiteralIR) -> None: pass
    def visit_binary_op(self, n: BinaryOpIR) -> None:
        n.left.accept(self); n.right.accept(self)
    def visit_unary_op(self, n: UnaryOpIR) -> None: n.operand.accept(self)
    def visit_builtin_call(self, n: BuiltinCallIR) -> None:
        for a in n.args or []: a.accept(self)
    def visit_function_call(self, n: FunctionCallIR) -> None:
        if n.callee_expr is not None: n.callee_expr.accept(self)
        for a in n.arguments or []: a.accept(self)
    def visit_rectangular_access(self, n: RectangularAccessIR) -> None:
        n.array.accept(self)
        for i in n.indices or []: i.accept(self)
    def visit_jagged_access(self, n: JaggedAccessIR) -> None:
        if n.base is not None: n.base.accept(self)
    def visit_block_expression(self, n: BlockExpressionIR) -> None:
        for s in n.statements or []:
            if isinstance(s, (BindingIR, ExpressionIR)): s.accept(self)
        if n.final_expr is not None: n.final_expr.accept(self)
    def visit_if_expression(self, n: IfExpressionIR) -> None:
        n.condition.accept(self); n.then_expr.accept(self)
        if n.else_expr is not None: n.else_expr.accept(self)
    def visit_cast_expression(self, n: CastExpressionIR) -> None: n.expr.accept(self)
    def visit_differential(self, n: DifferentialIR) -> None: n.operand.accept(self)
    def visit_lambda(self, n: LambdaIR) -> None: n.body.accept(self)
    def visit_range(self, n: RangeIR) -> None: n.start.accept(self); n.end.accept(self)
    def visit_reduction_expression(self, n: ReductionExpressionIR) -> None: n.body.accept(self)
    def visit_where_expression(self, n: Any) -> None:
        n.expr.accept(self)
        for c in n.constraints or []: c.accept(self)
    def visit_pipeline_expression(self, n: Any) -> None: n.left.accept(self); n.right.accept(self)
    def visit_array_comprehension(self, n: Any) -> None: n.body.accept(self)
    def visit_array_literal(self, n: ArrayLiteralIR) -> None:
        for e in n.elements or []: e.accept(self)
    def visit_tuple_expression(self, n: Any) -> None:
        for e in n.elements or []: e.accept(self)
    def visit_tuple_access(self, n: Any) -> None: n.tuple_expr.accept(self)
    def visit_member_access(self, n: MemberAccessIR) -> None: n.object.accept(self)
    def visit_function_value(self, n: FunctionValueIR) -> None:
        if n.body is not None: n.body.accept(self)
    def visit_try_expression(self, n: Any) -> None: n.operand.accept(self)
    def visit_match_expression(self, n: Any) -> None:
        n.scrutinee.accept(self)
        for arm in n.arms or []:
            if getattr(arm, "body", None) is not None: arm.body.accept(self)
    def visit_interpolated_string(self, n: Any) -> None: pass
    def visit_binding(self, n: BindingIR) -> None:
        if n.expr is not None: n.expr.accept(self)
    def visit_program(self, n: ProgramIR) -> None:
        for b in n.bindings or []: b.accept(self)
    def visit_einstein(self, n: EinsteinIR) -> None:
        for c in n.clauses or []:
            if isinstance(c, EinsteinClauseIR): c.accept(self)
    def visit_einstein_clause(self, n: EinsteinClauseIR) -> None:
        if n.value is not None: n.value.accept(self)
    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> None:
        if n.primal_body is not None: n.primal_body.accept(self)
        if n.diff_body is not None: n.diff_body.accept(self)
    def visit_index_rest(self, n: Any) -> None:
        if getattr(n, "defid", None) is not None: self.defids.add(n.defid)
    # -- no-op stubs --------------------------------------------------------
    def visit_module(self, n: Any) -> None: pass
    def visit_index_var(self, n: Any) -> None: pass
    def visit_identifier_pattern(self, n: Any) -> None: pass
    def visit_wildcard_pattern(self, n: Any) -> None: pass
    def visit_literal_pattern(self, n: Any) -> None: pass
    def visit_tuple_pattern(self, n: Any) -> None: pass
    def visit_array_pattern(self, n: Any) -> None: pass
    def visit_rest_pattern(self, n: Any) -> None: pass
    def visit_guard_pattern(self, n: Any) -> None: pass
    def visit_or_pattern(self, n: Any) -> None: pass
    def visit_constructor_pattern(self, n: Any) -> None: pass
    def visit_binding_pattern(self, n: Any) -> None: pass
    def visit_range_pattern(self, n: Any) -> None: pass


class _RectReadsRootDefIdVisitor(_DefIdCollector):
    """True after walk if any ``RectangularAccess`` uses ``IdentifierIR(tensor_did)`` as array root."""

    def __init__(self, tensor_did: DefId) -> None:
        super().__init__()
        self._tensor_did = tensor_did
        self.found = False

    def visit_rectangular_access(self, n: RectangularAccessIR) -> None:
        a = n.array
        if isinstance(a, IdentifierIR) and a.defid == self._tensor_did:
            self.found = True
        if a is not None:
            a.accept(self)
        for i in n.indices or []:
            i.accept(self)


def _einstein_clause_values_rect_read_tensor(ein: EinsteinIR, tensor_did: DefId) -> bool:
    for c in ein.clauses or []:
        if c.value is None:
            continue
        vis = _RectReadsRootDefIdVisitor(tensor_did)
        c.value.accept(vis)
        if vis.found:
            return True
    return False


class _DependencyQueryCache:
    """Memoized dependency queries scoped to one binding map."""

    def __init__(self, bindings: Dict[DefId, BindingIR]) -> None:
        self._B = bindings
        self._expr_defids: Dict[int, frozenset[DefId]] = {}
        self._binding_expr_defids: Dict[DefId, frozenset[DefId]] = {}
        self._rhs_closure_by_expr: Dict[int, frozenset[DefId]] = {}
        self._reachable_by_src: Dict[DefId, frozenset[DefId]] = {}

    @property
    def bindings(self) -> Dict[DefId, BindingIR]:
        return self._B

    def collect_defids(self, expr: Optional[ExpressionIR]) -> frozenset[DefId]:
        if expr is None:
            return frozenset()
        key = id(expr)
        cached = self._expr_defids.get(key)
        if cached is not None:
            return cached
        collector = _DefIdCollector()
        expr.accept(collector)
        out = frozenset(collector.defids)
        self._expr_defids[key] = out
        return out

    def binding_expr_defids(self, did: DefId) -> frozenset[DefId]:
        cached = self._binding_expr_defids.get(did)
        if cached is not None:
            return cached
        binding = self._B.get(did)
        out = self.collect_defids(binding.expr) if binding is not None else frozenset()
        self._binding_expr_defids[did] = out
        return out

    def jacobian_rhs_closure(self, expr: Optional[ExpressionIR]) -> frozenset[DefId]:
        if expr is None:
            return frozenset()
        key = id(expr)
        cached = self._rhs_closure_by_expr.get(key)
        if cached is not None:
            return cached
        work = list(self.collect_defids(expr))
        closure: Set[DefId] = set()
        while work:
            did = work.pop()
            if did in closure:
                continue
            closure.add(did)
            binding = self._B.get(did)
            if binding is None or binding.expr is None or _is_diff_name(binding.name or ""):
                continue
            for dep in self.binding_expr_defids(did):
                if dep not in closure:
                    work.append(dep)
        out = frozenset(closure)
        self._rhs_closure_by_expr[key] = out
        return out

    def reachable_from(self, src: DefId) -> frozenset[DefId]:
        cached = self._reachable_by_src.get(src)
        if cached is not None:
            return cached
        seen: Set[DefId] = set()
        queue = [src]
        while queue:
            cur = queue.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for dep in self.binding_expr_defids(cur):
                if dep not in seen:
                    queue.append(dep)
        out = frozenset(seen)
        self._reachable_by_src[src] = out
        return out

    def fork(self, bindings: Dict[DefId, BindingIR]) -> "_DependencyQueryCache":
        """Reuse expression-local defid walks for a derived binding map."""
        if bindings is self._B:
            return self
        child = _DependencyQueryCache(bindings)
        child._expr_defids = self._expr_defids
        return child


def _collect_defids(expr: Optional[ExpressionIR]) -> Set[DefId]:
    if expr is None: return set()
    c = _DefIdCollector(); expr.accept(c); return c.defids


def _function_call_ir_label(n: FunctionCallIR) -> str:
    name = n.function_name or "<non-identifier callee>"
    mp = n.module_path
    if mp:
        return "::".join(mp) + "::" + name
    return name


def _autodiff_primal_data_defids(
    expr: Optional[ExpressionIR],
    B: Dict[DefId, Any],
    dep_cache: Optional[_DependencyQueryCache] = None,
) -> Set[DefId]:
    if expr is None:
        return set()
    out: Set[DefId] = set()
    defs = dep_cache.collect_defids(expr) if dep_cache is not None else _collect_defids(expr)
    for d in defs:
        if d is None:
            continue
        bb = B.get(d)
        if bb is not None and is_function_binding(bb):
            continue
        out.add(d)
    return out


def _rectangular_read_root_defid(expr: Optional[ExpressionIR]) -> Optional[DefId]:
    """If ``expr`` is (possibly nested) ``RectangularAccessIR`` ending in ``IdentifierIR``, return that array root's DefId."""
    if expr is None:
        return None
    cur: ExpressionIR = expr
    while isinstance(cur, RectangularAccessIR):
        arr = cur.array
        if isinstance(arr, IdentifierIR) and arr.defid is not None:
            return arr.defid
        cur = arr
    return None


def _binding_is_rect_slice_of_tensor(wrt: DefId, tensor_did: DefId, B: Dict[DefId, BindingIR]) -> bool:
    """True if the binding for ``wrt`` is a rectangular read whose storage root is ``tensor_did``."""
    wb = B.get(wrt)
    if wb is None or wb.expr is None:
        return False
    return _rectangular_read_root_defid(wb.expr) == tensor_did


def _jacobian_rhs_depends_on_wrt(
    expr: Optional[ExpressionIR],
    wrt: DefId,
    B: Dict[DefId, BindingIR],
    dep_cache: Optional[_DependencyQueryCache] = None,
) -> bool:
    """True if ``expr`` references ``wrt`` transitively through non-diff ``let`` bindings (surface DefIds + binding RHS closure).

    Also true when ``wrt`` is a slice alias ``let w_ij = T[…]`` and ``expr`` depends on the same storage tensor ``T``
    (e.g. ``logit_j = sum[a](… T[a, j] …)`` and ``w_ij = T[i, j]``).
    """
    if expr is None:
        return False
    if dep_cache is not None:
        closure = dep_cache.jacobian_rhs_closure(expr)
    else:
        work = [d for d in _collect_defids(expr) if d is not None]
        closure: Set[DefId] = set()
        while work:
            d = work.pop()
            if d in closure:
                continue
            closure.add(d)
            b = B.get(d)
            if b is None or b.expr is None or _is_diff_name(b.name or ""):
                continue
            for e in _collect_defids(b.expr):
                if e is not None and e not in closure:
                    work.append(e)
    if wrt in closure:
        return True
    wb = B.get(wrt)
    if wb is not None and wb.expr is not None:
        root = _rectangular_read_root_defid(wb.expr)
        if root is not None and root in closure:
            return True
    return False


class _TargetCollector(_DefIdCollector):
    """Single walk: differential targets + ``@num/@den`` quotient pairs."""

    def __init__(self) -> None:
        super().__init__()
        self.targets: List[Tuple[DefId, str]] = []
        self.quotient_pairs: List[Tuple[DefId, DefId]] = []

    @staticmethod
    def _tgt_op(op: Any) -> Optional[Tuple[DefId, str]]:
        if isinstance(op, IdentifierIR) and op.defid is not None:
            return (op.defid, op.name or "")
        return None

    @staticmethod
    def _diff_defid(e: Any) -> Optional[DefId]:
        if isinstance(e, DifferentialIR) and isinstance(e.operand, IdentifierIR):
            return e.operand.defid
        if isinstance(e, IdentifierIR):
            return e.defid
        return None

    def visit_differential(self, n: DifferentialIR) -> None:
        t = self._tgt_op(n.operand)
        if t is not None:
            self.targets.append(t)
        n.operand.accept(self)

    def visit_binary_op(self, n: BinaryOpIR) -> None:
        if (n.operator == BinaryOp.DIV and isinstance(n.left, DifferentialIR)
                and isinstance(n.right, DifferentialIR)):
            for op in (n.left.operand, n.right.operand):
                tt = self._tgt_op(op)
                if tt is not None:
                    self.targets.append(tt)
            num, den = self._diff_defid(n.left), self._diff_defid(n.right)
            if num is not None and den is not None:
                self.quotient_pairs.append((num, den))
        n.left.accept(self)
        n.right.accept(self)

    def visit_program(self, n: ProgramIR) -> None:
        for b in n.bindings or []:
            b.accept(self)
        for s in n.statements or []:
            if not isinstance(s, BindingIR) and isinstance(s, ExpressionIR):
                s.accept(self)


# ═══════════════════════════════════════════════════════════════════════════
# Cleanup visitor  (Phase 4)
# ═══════════════════════════════════════════════════════════════════════════

class _TypeStripper(_DefIdCollector):
    """Walk IR in-place: strip DifferentialType from type_info."""
    @staticmethod
    def _st(ty: Any) -> Any:
        return strip_differential_types_deep(ty)
    def _se(self, n: ExpressionIR) -> None:
        if n.type_info is not None: n.type_info = self._st(n.type_info)
    # override visits that also strip
    def visit_literal(self, n: LiteralIR) -> None: self._se(n)
    def visit_identifier(self, n: IdentifierIR) -> None:
        self._se(n)
        if n.defid is not None: self.defids.add(n.defid)
    def visit_binary_op(self, n: BinaryOpIR) -> None:
        self._se(n); n.left.accept(self); n.right.accept(self)
    def visit_unary_op(self, n: UnaryOpIR) -> None:
        self._se(n); n.operand.accept(self)
    def visit_builtin_call(self, n: BuiltinCallIR) -> None:
        self._se(n)
        for a in n.args or []: a.accept(self)
    def visit_function_call(self, n: FunctionCallIR) -> None:
        self._se(n)
        if n.callee_expr is not None: n.callee_expr.accept(self)
        for a in n.arguments or []: a.accept(self)
    def visit_rectangular_access(self, n: RectangularAccessIR) -> None:
        self._se(n); n.array.accept(self)
        for i in n.indices or []: i.accept(self)
    def visit_jagged_access(self, n: JaggedAccessIR) -> None:
        self._se(n)
        if n.base is not None: n.base.accept(self)
        for idx in n.index_chain or []: idx.accept(self)
    def visit_block_expression(self, n: BlockExpressionIR) -> None:
        self._se(n)
        for s in n.statements or []:
            if isinstance(s, (BindingIR, ExpressionIR)): s.accept(self)
        if n.final_expr is not None: n.final_expr.accept(self)
    def visit_if_expression(self, n: IfExpressionIR) -> None:
        self._se(n); n.condition.accept(self); n.then_expr.accept(self)
        if n.else_expr is not None: n.else_expr.accept(self)
    def visit_cast_expression(self, n: CastExpressionIR) -> None:
        self._se(n)
        tt = n.target_type
        if tt is not None and isinstance(tt, Type): n.target_type = self._st(tt)
        n.expr.accept(self)
    def visit_differential(self, n: DifferentialIR) -> None:
        self._se(n); n.operand.accept(self)
    def visit_lambda(self, n: LambdaIR) -> None:
        self._se(n)
        for p in n.parameters or []:
            if p.param_type is not None: p.param_type = self._st(p.param_type)
        n.body.accept(self)
    def visit_range(self, n: RangeIR) -> None:
        self._se(n); n.start.accept(self); n.end.accept(self)
    def visit_reduction_expression(self, n: ReductionExpressionIR) -> None:
        self._se(n)
        for lv in n.loop_vars or []: lv.accept(self)
        n.body.accept(self)
        if n.where_clause is not None:
            for c in n.where_clause.constraints or []: c.accept(self)
    def visit_where_expression(self, n: WhereExpressionIR) -> None:
        self._se(n); n.expr.accept(self)
        for c in n.constraints or []: c.accept(self)
    def visit_pipeline_expression(self, n: PipelineExpressionIR) -> None:
        self._se(n); n.left.accept(self); n.right.accept(self)
    def visit_array_comprehension(self, n: ArrayComprehensionIR) -> None:
        self._se(n)
        for v in n.loop_vars or []: v.accept(self)
        for r in n.ranges or []: r.accept(self)
        for c in n.constraints or []: c.accept(self)
        n.body.accept(self)
    def visit_array_literal(self, n: ArrayLiteralIR) -> None:
        self._se(n)
        for e in n.elements or []: e.accept(self)
    def visit_tuple_expression(self, n: TupleExpressionIR) -> None:
        self._se(n)
        for e in n.elements or []: e.accept(self)
    def visit_tuple_access(self, n: TupleAccessIR) -> None:
        self._se(n); n.tuple_expr.accept(self)
    def visit_member_access(self, n: MemberAccessIR) -> None:
        self._se(n); n.object.accept(self)
    def visit_function_value(self, n: FunctionValueIR) -> None:
        self._se(n)
        if n.return_type is not None: object.__setattr__(n, "return_type", self._st(n.return_type))
        for p in n.parameters or []:
            if p.param_type is not None: p.param_type = self._st(p.param_type)
        if n.body is not None: n.body.accept(self)
    def visit_try_expression(self, n: TryExpressionIR) -> None:
        self._se(n); n.operand.accept(self)
    def visit_match_expression(self, n: MatchExpressionIR) -> None:
        self._se(n); n.scrutinee.accept(self)
        for arm in n.arms or []:
            if getattr(arm, "body", None) is not None: arm.body.accept(self)
    def visit_interpolated_string(self, n: InterpolatedStringIR) -> None:
        self._se(n)
        for p in n.parts or []:
            if isinstance(p, ExpressionIR): p.accept(self)
    def visit_binding(self, n: BindingIR) -> None:
        if n.type_info is not None: n.type_info = self._st(n.type_info)
        if n.expr is not None: n.expr.accept(self)
    def visit_einstein(self, n: EinsteinIR) -> None:
        self._se(n)
        et = n.element_type
        if et is not None and isinstance(et, Type): n.element_type = self._st(et)
        for c in n.clauses or []:
            if isinstance(c, EinsteinClauseIR): c.accept(self)
    def visit_einstein_clause(self, n: EinsteinClauseIR) -> None:
        for idx in n.indices or []:
            if isinstance(idx, ExpressionIR): idx.accept(self)
        if n.value is not None: n.value.accept(self)
        if n.where_clause is not None:
            for c in n.where_clause.constraints or []: c.accept(self)
    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> None:
        self._se(n)
        if n.primal_body is not None: n.primal_body.accept(self)
        if n.diff_body is not None: n.diff_body.accept(self)
    def visit_index_var(self, n: IndexVarIR) -> None:
        self._se(n)
        if n.range_ir is not None: n.range_ir.accept(self)
    def visit_index_rest(self, n: IndexRestIR) -> None:
        self._se(n)
        if getattr(n, "defid", None) is not None: self.defids.add(n.defid)


class _CleanupVisitor(_TypeStripper):
    """Post-pass: clear custom_diff_body, drop DiffRuleIR, strip DifferentialType."""
    def visit_function_value(self, n: FunctionValueIR) -> None:
        cdb = getattr(n, "custom_diff_body", None)
        if cdb is not None:
            cdb.accept(self)
            object.__setattr__(n, "custom_diff_body", None)
        super().visit_function_value(n)
    def visit_program(self, n: ProgramIR) -> None:
        n.statements = [s for s in (n.statements or []) if not isinstance(s, DiffRuleIR)]
        n.bindings = [s for s in n.statements if isinstance(s, BindingIR)]
        for s in n.statements or []:
            if isinstance(s, (BindingIR, ExpressionIR)): s.accept(self)
        for mod in n.modules or []: mod.accept(self)
    def visit_module(self, n: Any) -> None:
        for b in n.functions or []: b.accept(self)
        for b in n.constants or []: b.accept(self)
        for sub in n.submodules or []: sub.accept(self)
    def visit_diff_rule(self, n: DiffRuleIR) -> None:
        if n.body is not None: n.body.accept(self)


def clear_custom_diff_body_everywhere(program: ProgramIR) -> None:
    """Public API: clear autodiff-only IR artefacts."""
    program.accept(_CleanupVisitor())


def _tensor_rank_from_literal_array(expr: Optional[ExpressionIR]) -> int:
    if expr is None or not isinstance(expr, ArrayLiteralIR):
        return 0
    el = expr.elements or []
    if not el:
        return 0
    if isinstance(el[0], ArrayLiteralIR):
        return 1 + _tensor_rank_from_literal_array(el[0])
    return 1


def _tensor_rank_from_einstein_expr(expr: Optional[ExpressionIR]) -> int:
    if expr is None or not isinstance(expr, EinsteinIR):
        return 0
    if expr.shape is not None and len(expr.shape) > 0:
        return len(expr.shape)
    clauses = expr.clauses or []
    if clauses and isinstance(clauses[0], EinsteinClauseIR):
        return len(clauses[0].indices or [])
    return 0


def _tensor_rank_from_expr(expr: Optional[ExpressionIR], bindings: Dict[DefId, BindingIR]) -> int:
    if expr is None:
        return 0
    si = getattr(expr, "shape_info", None)
    if isinstance(si, tuple) and len(si) > 0:
        return len(si)
    if isinstance(expr, IdentifierIR) and expr.defid is not None:
        return _tensor_rank_from_binding(bindings.get(expr.defid))
    if isinstance(expr, EinsteinIR):
        return _tensor_rank_from_einstein_expr(expr)
    if isinstance(expr, ArrayLiteralIR):
        return _tensor_rank_from_literal_array(expr)
    if isinstance(expr, RectangularAccessIR):
        base = _tensor_rank_from_expr(expr.array, bindings)
        return max(0, base - len(expr.indices or []))
    ti = getattr(expr, "type_info", None)
    if isinstance(ti, RectangularType):
        if getattr(ti, "is_dynamic_rank", False):
            return 1
        if ti.shape is not None:
            return len(ti.shape)
        return 1
    return 0


def _tensor_rank_from_binding(b: Optional[BindingIR]) -> int:
    if b is None:
        return 0
    si = getattr(b, "shape_info", None)
    if isinstance(si, tuple) and len(si) > 0:
        return len(si)
    rk_ein = _tensor_rank_from_einstein_expr(getattr(b, "expr", None))
    if rk_ein > 0:
        return rk_ein
    rk_lit = _tensor_rank_from_literal_array(getattr(b, "expr", None))
    if rk_lit > 0:
        return rk_lit
    ex = getattr(b, "expr", None)
    ex_si = getattr(ex, "shape_info", None)
    if isinstance(ex_si, tuple) and len(ex_si) > 0:
        return len(ex_si)
    ex_ti = getattr(ex, "type_info", None)
    if isinstance(ex_ti, RectangularType):
        if getattr(ex_ti, "is_dynamic_rank", False):
            pass
        elif ex_ti.shape is not None:
            return len(ex_ti.shape)
        else:
            return 1
    ti = getattr(b, "type_info", None)
    if isinstance(ti, RectangularType):
        if getattr(ti, "is_dynamic_rank", False):
            pass
        elif ti.shape is not None:
            return len(ti.shape)
        else:
            return 1
    return 0


def _alloc_wrt_gradient_axes(
    wb: BindingIR, wrt: DefId, resolver: Any, loc: SourceLocation
) -> List[IndexVarIR]:
    wid = IdentifierIR(
        wb.name or "_wrt",
        wb.location or loc,
        wrt,
        type_info=_ti(wb) or UNKNOWN,
        shape_info=_si(wb),
    )
    rank = _tensor_rank_from_binding(wb)
    if rank <= 0:
        return []
    out: List[IndexVarIR] = []
    for p in range(rank):
        nd = resolver.allocate_for_local()
        sh = MemberAccessIR(object=wid, member="shape", location=loc, type_info=UNKNOWN)
        dim = LiteralIR(p, loc, type_info=PrimitiveType("i32"))
        sd = RectangularAccessIR(array=sh, indices=[dim], location=loc, type_info=UNKNOWN)
        rng = RangeIR(
            start=LiteralIR(0, loc, type_info=PrimitiveType("i32")),
            end=sd,
            location=loc,
            type_info=UNKNOWN,
        )
        iv = IndexVarIR("_jcot_%d" % p, loc, nd, range_ir=rng)
        out.append(iv)
    return out


def _jacobian_rect_read_wrt_slice_binding(
    read_indices: List[ExpressionIR],
    wrt: DefId,
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    array_root: Optional[DefId] = None,
) -> Optional[ExpressionIR]:
    """∂(T[r₀,…])/∂w when ``let w = T[s₀,…]``: product of (rₖ == sₖ) indicators; ``None`` if not a rect slice."""
    wb = B.get(wrt)
    if wb is None or wb.expr is None:
        return None
    ex = wb.expr
    if isinstance(ex, CastExpressionIR) and ex.expr is not None:
        ex = ex.expr
    if not isinstance(ex, RectangularAccessIR):
        return None
    root = _rectangular_read_root_defid(ex)
    if root is None:
        return None
    if array_root is not None and root != array_root:
        return None
    slice_idxs: List[ExpressionIR] = []
    cur: Optional[ExpressionIR] = ex
    while isinstance(cur, RectangularAccessIR):
        slice_idxs = list(cur.indices or []) + slice_idxs
        cur = cur.array
    read = list(read_indices)
    if len(slice_idxs) != len(read):
        return None

    def chain(k: int) -> ExpressionIR:
        if k >= len(read):
            return _fl(1, loc)
        eq = BinaryOpIR(BinaryOp.EQ, read[k], slice_idxs[k], loc, type_info=BOOL)
        rest = chain(k + 1)
        return IfExpressionIR(eq, rest, loc, else_expr=_z(loc), type_info=F32)

    return chain(0)


def _jacobian_tensor_id_wrt_slice_binding(
    tensor_id: IdentifierIR,
    wrt: DefId,
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    resolver: Any,
) -> Optional[ExpressionIR]:
    """∂(T)/∂w for ``let w = T[s...]`` as a one-hot Einstein tensor over T's shape."""
    wb = B.get(wrt)
    if wb is None or wb.expr is None:
        return None
    ex = wb.expr
    if isinstance(ex, CastExpressionIR) and ex.expr is not None:
        ex = ex.expr
    if not isinstance(ex, RectangularAccessIR):
        return None
    root = _rectangular_read_root_defid(ex)
    if root is None or root != tensor_id.defid:
        return None
    if resolver is None:
        return None

    slice_idxs: List[ExpressionIR] = []
    cur: Optional[ExpressionIR] = ex
    while isinstance(cur, RectangularAccessIR):
        slice_idxs = list(cur.indices or []) + slice_idxs
        cur = cur.array
    if not slice_idxs:
        return None

    dyn_idxs: List[IndexVarIR] = []
    var_ranges: Dict[DefId, RangeIR] = {}
    shape_ref = MemberAccessIR(
        object=tensor_id,
        member="shape",
        location=loc,
        type_info=UNKNOWN,
    )
    for p in range(len(slice_idxs)):
        did = resolver.allocate_for_local()
        dim = LiteralIR(p, loc, type_info=PrimitiveType("i32"))
        bound = RectangularAccessIR(array=shape_ref, indices=[dim], location=loc, type_info=UNKNOWN)
        rng = RangeIR(
            start=LiteralIR(0, loc, type_info=PrimitiveType("i32")),
            end=bound,
            location=loc,
            type_info=UNKNOWN,
        )
        iv = IndexVarIR("_jslice_%d" % p, loc, did, range_ir=rng)
        dyn_idxs.append(iv)
        var_ranges[did] = rng

    body: ExpressionIR = _fl(1, loc)
    for d, s in zip(dyn_idxs, slice_idxs):
        eq = BinaryOpIR(BinaryOp.EQ, d, s, loc, type_info=BOOL)
        body = IfExpressionIR(eq, body, loc, else_expr=_z(loc), type_info=F32)

    return EinsteinIR(
        clauses=[
            EinsteinClauseIR(
                indices=dyn_idxs,
                value=body,
                location=loc,
                variable_ranges=var_ranges,
            )
        ],
        shape=None,
        element_type=None,
        location=loc,
        type_info=_ti(tensor_id),
        shape_info=_si(tensor_id),
    )


def _flatten_rect_access(expr: RectangularAccessIR) -> Tuple[ExpressionIR, List[ExpressionIR]]:
    """Flatten nested rectangular indexing into (root, full_index_list)."""
    idxs: List[ExpressionIR] = list(expr.indices or [])
    root: ExpressionIR = expr.array
    while isinstance(root, RectangularAccessIR):
        idxs = list(root.indices or []) + idxs
        root = root.array
    return root, idxs


def _kronecker_delta_indices(
    accessed: List[ExpressionIR],
    wrt_axes: List[IndexVarIR],
    loc: SourceLocation,
) -> ExpressionIR:
    if len(accessed) != len(wrt_axes):
        raise ValueError("Autodiff: Kronecker index list length mismatch for tensor `wrt` Jacobian")
    if not accessed:
        return _fl(1, loc)
    tail_a = accessed[1:]
    tail_x = wrt_axes[1:]
    if tail_a:
        inner = _kronecker_delta_indices(tail_a, tail_x, loc)
    else:
        inner = _fl(1, loc)
    eq = BinaryOpIR(BinaryOp.EQ, accessed[0], wrt_axes[0], loc, type_info=BOOL)
    return IfExpressionIR(eq, inner, loc, else_expr=_z(loc), type_info=F32)


def _is_bare_wrt_tensor_deriv(
    e: ExpressionIR, wrt: DefId, wrt_axes: Optional[List[IndexVarIR]]
) -> bool:
    return bool(wrt_axes) and isinstance(e, IdentifierIR) and e.defid == wrt


def _reject_bare_wrt_tensor_jacobian(
    e: ExpressionIR, wrt: DefId, wrt_axes: Optional[List[IndexVarIR]], ctx: str
) -> None:
    if _is_bare_wrt_tensor_deriv(e, wrt, wrt_axes):
        raise ValueError(
            "Autodiff: JacobianVisitor (%s) hit an unindexed use of a tensor `wrt` binding; "
            "index it (e.g. x[i]) or use Einstein reductions."
            % (ctx,)
        )


def _unwrap_jacobian_add_zero(expr: ExpressionIR) -> ExpressionIR:
    if isinstance(expr, BinaryOpIR) and expr.operator == BinaryOp.ADD and _is_zero(expr.right):
        return _unwrap_jacobian_add_zero(expr.left)
    return expr


def _merge_primal_clause_with_cotangent_einstein(
    clause: EinsteinClauseIR, d_val: ExpressionIR
) -> Tuple[List, ExpressionIR, Dict]:
    dv = _unwrap_jacobian_add_zero(d_val)
    if isinstance(dv, EinsteinIR) and len(dv.clauses or []) == 1:
        ic = dv.clauses[0]
        ix = list(ic.indices or [])
        if ix and all(getattr(iv, "name", "").startswith("_jcot_") for iv in ix):
            merged_idx = list(clause.indices or []) + ix
            nvr = dict(clause.variable_ranges or {})
            nvr.update(ic.variable_ranges or {})
            return merged_idx, ic.value, nvr
    return list(clause.indices or []), dv, dict(clause.variable_ranges or {})


def _expr_uses_index_defids(expr: Optional[ExpressionIR], defids: Set[DefId]) -> bool:
    if expr is None or not defids:
        return False
    if isinstance(expr, IndexVarIR):
        return expr.defid is not None and expr.defid in defids
    if isinstance(expr, (LiteralIR, IdentifierIR)):
        return False
    if isinstance(expr, UnaryOpIR):
        return _expr_uses_index_defids(expr.operand, defids)
    if isinstance(expr, BinaryOpIR):
        return _expr_uses_index_defids(expr.left, defids) or _expr_uses_index_defids(expr.right, defids)
    if isinstance(expr, IfExpressionIR):
        if _expr_uses_index_defids(expr.condition, defids):
            return True
        if _expr_uses_index_defids(expr.then_expr, defids):
            return True
        return _expr_uses_index_defids(expr.else_expr, defids)
    if isinstance(expr, RectangularAccessIR):
        if _expr_uses_index_defids(expr.array, defids):
            return True
        for ix in expr.indices or []:
            if _expr_uses_index_defids(ix, defids):
                return True
        return False
    if isinstance(expr, ReductionExpressionIR):
        ow = getattr(expr, "where_clause", None)
        if ow is not None:
            for c in getattr(ow, "constraints", None) or []:
                if _expr_uses_index_defids(c, defids):
                    return True
        return _expr_uses_index_defids(expr.body, defids)
    if isinstance(expr, CastExpressionIR):
        return _expr_uses_index_defids(expr.expr, defids)
    if isinstance(expr, EinsteinIR):
        for c in expr.clauses or []:
            for ix in c.indices or []:
                did = getattr(ix, "defid", None)
                if did is not None and did in defids:
                    return True
            cw = getattr(c, "where_clause", None)
            if cw is not None:
                for cn in getattr(cw, "constraints", None) or []:
                    if _expr_uses_index_defids(cn, defids):
                        return True
            if _expr_uses_index_defids(c.value, defids):
                return True
        return False
    if isinstance(expr, SelectAtArgmaxIR):
        return _expr_uses_index_defids(expr.primal_body, defids) or _expr_uses_index_defids(
            expr.diff_body, defids
        )
    if isinstance(expr, FunctionCallIR):
        for a in expr.arguments or []:
            if _expr_uses_index_defids(a, defids):
                return True
        return False
    if isinstance(expr, BlockExpressionIR):
        if expr.final_expr is not None and _expr_uses_index_defids(expr.final_expr, defids):
            return True
        for s in expr.statements or []:
            if isinstance(s, BindingIR) and s.expr is not None and _expr_uses_index_defids(s.expr, defids):
                return True
        return False
    return False


def _eval_const_expr(
    expr: Optional[ExpressionIR],
    B: Dict[DefId, BindingIR],
    max_depth: int = 32,
    _vis: Optional[Set[DefId]] = None,
    subst: Optional[Dict[DefId, ExpressionIR]] = None,
) -> Optional[Union[int, float, bool]]:
    """Best-effort constant evaluator for Jacobian branch pruning.

    Supports literals, simple numeric/bool binary ops, casts, builtin len on array literals,
    and identifier resolution through binding expressions.
    """
    if expr is None or max_depth <= 0:
        return None
    if isinstance(expr, LiteralIR):
        return expr.value
    if isinstance(expr, CastExpressionIR):
        return _eval_const_expr(expr.expr, B, max_depth - 1, _vis, subst)
    if isinstance(expr, IdentifierIR) and expr.defid is not None:
        vis = _vis if _vis is not None else set()
        if expr.defid in vis:
            return None
        vis.add(expr.defid)
        if subst is not None:
            sub = subst.get(expr.defid)
            if sub is not None:
                sv = _eval_const_expr(sub, B, max_depth - 1, vis, subst)
                if sv is not None:
                    return sv
        b = B.get(expr.defid)
        if b is None or b.expr is None:
            return None
        return _eval_const_expr(b.expr, B, max_depth - 1, vis, subst)
    if isinstance(expr, BuiltinCallIR):
        if expr.builtin_name == "len" and len(expr.args or []) == 1:
            cur: Optional[ExpressionIR] = expr.args[0]
            hops = 0
            while (
                isinstance(cur, IdentifierIR)
                and cur.defid is not None
                and hops < 16
            ):
                if subst is not None:
                    sub = subst.get(cur.defid)
                    if sub is not None:
                        cur_try = sub
                        if isinstance(cur_try, ArrayLiteralIR):
                            cur = cur_try
                            break
                        cur = cur_try
                        hops += 1
                        if not isinstance(cur, IdentifierIR):
                            break
                b = B.get(cur.defid)
                if b is None or b.expr is None:
                    break
                cur = b.expr
                hops += 1
            if isinstance(cur, ArrayLiteralIR):
                return len(cur.elements or [])
        return None
    if isinstance(expr, UnaryOpIR):
        v = _eval_const_expr(expr.operand, B, max_depth - 1, _vis, subst)
        if v is None:
            return None
        if expr.operator == UnaryOp.NEG:
            return -float(v)
        if expr.operator == UnaryOp.POS:
            return +float(v)
        return None
    if isinstance(expr, BinaryOpIR):
        lv = _eval_const_expr(expr.left, B, max_depth - 1, _vis, subst)
        rv = _eval_const_expr(expr.right, B, max_depth - 1, _vis, subst)
        if lv is None or rv is None:
            return None
        op = expr.operator
        if op == BinaryOp.ADD:
            return float(lv) + float(rv)
        if op == BinaryOp.SUB:
            return float(lv) - float(rv)
        if op == BinaryOp.MUL:
            return float(lv) * float(rv)
        if op == BinaryOp.DIV:
            if float(rv) == 0.0:
                return None
            return float(lv) / float(rv)
        if op == BinaryOp.MOD:
            if float(rv) == 0.0:
                return None
            return int(float(lv)) % int(float(rv))
        if op == BinaryOp.POW:
            return float(lv) ** float(rv)
        if op == BinaryOp.EQ:
            return lv == rv
        if op == BinaryOp.NE:
            return lv != rv
        if op == BinaryOp.LT:
            return float(lv) < float(rv)
        if op == BinaryOp.LE:
            return float(lv) <= float(rv)
        if op == BinaryOp.GT:
            return float(lv) > float(rv)
        if op == BinaryOp.GE:
            return float(lv) >= float(rv)
        if op == BinaryOp.AND:
            return bool(lv) and bool(rv)
        if op == BinaryOp.OR:
            return bool(lv) or bool(rv)
        return None
    return None


def _clause_index_defids(indices: Optional[List]) -> Set[DefId]:
    out: Set[DefId] = set()
    for ix in indices or []:
        did = getattr(ix, "defid", None)
        if did is not None:
            out.add(did)
    return out


def _ensure_cotangent_axes_on_clause_indices(
    mi: List,
    nvr: Dict[DefId, RangeIR],
    d_val: ExpressionIR,
    axes: Optional[List[IndexVarIR]],
) -> Tuple[List, Dict[DefId, RangeIR]]:
    if not axes:
        return mi, nvr
    axis_ids = {iv.defid for iv in axes if getattr(iv, "defid", None) is not None}
    if not axis_ids or not _expr_uses_index_defids(d_val, axis_ids):
        return mi, nvr
    have = _clause_index_defids(mi)
    out_i = list(mi)
    out_v = dict(nvr)
    for iv in axes:
        did = getattr(iv, "defid", None)
        if did is not None and did not in have:
            out_i.append(iv)
            have.add(did)
        if did is not None and iv.range_ir is not None:
            out_v[did] = iv.range_ir
    return out_i, out_v


def _append_cotangent_axes_to_clause_indices(
    mi: List,
    nvr: Dict[DefId, RangeIR],
    axes: Optional[List[IndexVarIR]],
) -> Tuple[List, Dict[DefId, RangeIR]]:
    if not axes:
        return mi, nvr
    have = _clause_index_defids(mi)
    out_i = list(mi)
    out_v = dict(nvr)
    for iv in axes:
        did = getattr(iv, "defid", None)
        if did is not None and did not in have:
            out_i.append(iv)
            have.add(did)
        if did is not None and iv.range_ir is not None:
            out_v[did] = iv.range_ir
    return out_i, out_v


# ═══════════════════════════════════════════════════════════════════════════
# JacobianVisitor  — ∂(expr)/∂(wrt)
# ═══════════════════════════════════════════════════════════════════════════

class JacobianVisitor(IRVisitor[ExpressionIR]):
    """Symbolic partial derivative ∂(expr)/∂(wrt) as IR.

    Used for ``@y/@x`` quotients and callee-body differentiation.
    """
    def __init__(self, wrt: DefId, loc: SourceLocation,
                 bindings: Dict[DefId, BindingIR], resolver: Any,
                 stmt_partial: Optional[Dict[DefId, ExpressionIR]] = None,
                 wrt_tangent: Optional[ExpressionIR] = None,
                 primal_subst: Optional[Dict[DefId, ExpressionIR]] = None,
                 shared_cotangent_axes: Optional[List[IndexVarIR]] = None,
                 legacy_directional: bool = False,
                 dependency_cache: Optional[_DependencyQueryCache] = None) -> None:
        self._wrt = wrt; self._loc = loc; self._B = bindings
        self._R = resolver; self._sp = stmt_partial; self._wt = wrt_tangent
        self._ps = primal_subst
        self._legacy_directional = legacy_directional
        self._dep_cache = (
            dependency_cache
            if dependency_cache is not None and dependency_cache.bindings is bindings
            else _DependencyQueryCache(bindings)
        )
        self._wrt_axes: Optional[List[IndexVarIR]] = None
        wb = bindings.get(wrt)
        if wb is not None and _tensor_rank_from_binding(wb) > 0:
            self._wrt_axes = (
                list(shared_cotangent_axes)
                if shared_cotangent_axes is not None
                else _alloc_wrt_gradient_axes(wb, wrt, resolver, loc)
            )

    def _wrap_if_cotangent_indices(self, inner: ExpressionIR, n: ReductionExpressionIR) -> ExpressionIR:
        ax = self._wrt_axes
        if not ax:
            return inner
        loc = n.location or self._loc
        axis_ids = {iv.defid for iv in ax if getattr(iv, "defid", None) is not None}
        if not axis_ids or not _expr_uses_index_defids(inner, axis_ids):
            return inner
        mlr: Dict[DefId, RangeIR] = dict(n.loop_var_ranges or {})
        for iv in ax:
            did = getattr(iv, "defid", None)
            if did is not None and iv.range_ir is not None:
                mlr[did] = iv.range_ir
        return EinsteinIR(
            clauses=[
                EinsteinClauseIR(
                    indices=list(ax),
                    value=inner,
                    location=loc,
                    variable_ranges=mlr,
                )
            ],
            shape=None,
            element_type=None,
            location=loc,
            type_info=_ti(n),
            shape_info=None,
        )

    # -- atoms ---------------------------------------------------------------
    def visit_identifier(self, n: IdentifierIR) -> ExpressionIR:
        if n.defid == self._wrt:
            if self._wrt_axes:
                wb = self._B.get(self._wrt)
                return IdentifierIR(
                    n.name or "?",
                    n.location or self._loc,
                    n.defid,
                    type_info=_ti(n) or ( _ti(wb) if wb is not None else None ) or UNKNOWN,
                    shape_info=_si(n) or ( _si(wb) if wb is not None else None ),
                )
            return self._wt if self._wt is not None else _fl(1, self._loc)
        if n.defid is not None and self._sp is not None:
            pre = self._sp.get(n.defid)
            if pre is not None: return pre
        if n.defid is not None and self._ps is not None:
            sub = self._ps.get(n.defid)
            if sub is not None:
                return sub.accept(self)
        if n.defid is not None:
            b = self._B.get(n.defid)
            if b is not None and b.expr is not None:
                if _is_diff_name(b.name or ""): return n
                w_b = self._B.get(self._wrt)
                slice_root = (
                    _rectangular_read_root_defid(w_b.expr)
                    if w_b is not None and w_b.expr is not None
                    else None
                )
                tensor_alias = slice_root is not None and slice_root == n.defid
                if not tensor_alias and not _jacobian_rhs_depends_on_wrt(
                    b.expr, self._wrt, self._B, self._dep_cache
                ):
                    return _z(self._loc)
                if _binding_is_rect_slice_of_tensor(self._wrt, n.defid, self._B):
                    return IdentifierIR(
                        n.name or "?",
                        n.location or self._loc,
                        n.defid,
                        type_info=_ti(n),
                        shape_info=_si(n),
                    )
                if (
                    isinstance(b.expr, EinsteinIR)
                    and n.defid != self._wrt
                    and self._wrt_axes is None
                    and _einstein_clause_values_rect_read_tensor(b.expr, n.defid)
                ):
                    return _z(self._loc)
                return b.expr.accept(self)
        return _z(self._loc)

    def visit_literal(self, n: LiteralIR) -> ExpressionIR: return _z(self._loc)
    def visit_array_literal(self, n: ArrayLiteralIR) -> ExpressionIR: return _z(self._loc)

    # -- binary / unary ------------------------------------------------------
    def visit_binary_op(self, n: BinaryOpIR) -> ExpressionIR:
        L = n.left; R = n.right; loc = n.location or self._loc
        dL = L.accept(self); dR = R.accept(self)
        _reject_bare_wrt_tensor_jacobian(dL, self._wrt, self._wrt_axes, "binary left")
        _reject_bare_wrt_tensor_jacobian(dR, self._wrt, self._wrt_axes, "binary right")
        op = n.operator
        if op == BinaryOp.ADD:
            if _is_zero(dR) or _is_sum_reduction_of_zero(dR):
                return dL
            if _is_zero(dL) or _is_sum_reduction_of_zero(dL):
                return dR
            return BinaryOpIR(BinaryOp.ADD, dL, dR, loc)
        if op == BinaryOp.SUB: return BinaryOpIR(BinaryOp.SUB, dL, dR, loc)
        if op == BinaryOp.MUL:
            return BinaryOpIR(BinaryOp.ADD,
                              BinaryOpIR(BinaryOp.MUL, L, dR, loc),
                              BinaryOpIR(BinaryOp.MUL, R, dL, loc), loc)
        if op == BinaryOp.DIV:
            num = BinaryOpIR(BinaryOp.SUB,
                             BinaryOpIR(BinaryOp.MUL, R, dL, loc),
                             BinaryOpIR(BinaryOp.MUL, L, dR, loc), loc)
            den = BinaryOpIR(BinaryOp.POW, R, _fl(2, loc), loc)
            return BinaryOpIR(BinaryOp.DIV, num, den, loc)
        if op == BinaryOp.POW: return _pow_chain(n, dL, dR, self._B, self._R, loc)
        if op == BinaryOp.MOD: return dL
        if op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.LE, BinaryOp.GT, BinaryOp.GE, BinaryOp.AND, BinaryOp.OR):
            return _z(loc)
        raise ValueError(f"Autodiff: unsupported binary op: {op}")

    def visit_unary_op(self, n: UnaryOpIR) -> ExpressionIR:
        d = n.operand.accept(self)
        _reject_bare_wrt_tensor_jacobian(d, self._wrt, self._wrt_axes, "unary")
        if n.operator == UnaryOp.NEG: return UnaryOpIR(UnaryOp.NEG, d, n.location or self._loc)
        if n.operator == UnaryOp.POS: return d
        raise ValueError(f"Autodiff: unsupported unary op: {n.operator}")

    # -- reductions ----------------------------------------------------------
    def visit_reduction_expression(self, n: ReductionExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc; d_body = n.body.accept(self); op = n.operation
        if op == ReductionOp.SUM:
            inner = ReductionExpressionIR(ReductionOp.SUM, n.loop_vars, d_body, loc,
                   where_clause=n.where_clause, loop_var_ranges=n.loop_var_ranges,
                   type_info=_ti(n), shape_info=_si(n))
            return self._wrap_if_cotangent_indices(inner, n)
        if op == ReductionOp.MAX:
            inner = SelectAtArgmaxIR(n.body, d_body, n.loop_vars, loop_var_ranges=n.loop_var_ranges,
                   location=loc, type_info=_ti(n), shape_info=_si(n))
            return self._wrap_if_cotangent_indices(inner, n)
        if op == ReductionOp.MIN:
            inner = SelectAtArgmaxIR(n.body, d_body, n.loop_vars, loop_var_ranges=n.loop_var_ranges,
                   location=loc, type_info=_ti(n), shape_info=_si(n), use_argmin=True)
            return self._wrap_if_cotangent_indices(inner, n)
        if op == ReductionOp.PROD:
            inner = _prod_pullback_via_sum(n, d_body, loc, self._R)
            return self._wrap_if_cotangent_indices(inner, n)
        raise ValueError(f"Autodiff: unsupported reduction: {op}")

    # -- indexing / cast / control flow --------------------------------------
    def visit_rectangular_access(self, n: RectangularAccessIR) -> ExpressionIR:
        loc = n.location or self._loc
        root_expr, full_indices = _flatten_rect_access(n)
        if isinstance(root_expr, IdentifierIR) and root_expr.defid is not None and self._wrt_axes is None:
            full_sl = _jacobian_rect_read_wrt_slice_binding(
                full_indices, self._wrt, loc, self._B, array_root=root_expr.defid
            )
            if full_sl is not None:
                return full_sl
        indices = list(n.indices or [])
        da = n.array.accept(self)
        if _is_zero(da):
            return _z(loc)
        if (
            isinstance(da, IdentifierIR)
            and da.defid is not None
            and self._wrt_axes is None
        ):
            sl = _jacobian_rect_read_wrt_slice_binding(
                indices, self._wrt, loc, self._B, array_root=da.defid
            )
            if sl is not None:
                return sl
        if isinstance(da, IdentifierIR) and da.defid == self._wrt and self._wrt_axes:
            if len(indices) != len(self._wrt_axes):
                # Branch replay through rank-dispatch functions can transiently produce accesses
                # whose rank does not match ``wrt`` (e.g. visiting 1D/3D conv branches while
                # differentiating a 2D call). Those paths are non-contributing and should be zero.
                return _z(loc)
            return _kronecker_delta_indices(indices, self._wrt_axes, loc)
        if isinstance(da, RectangularAccessIR):
            dai = list(da.indices or [])
            if _rc_index_lists_equivalent(indices, dai):
                return da
        if self._wrt_axes and isinstance(da, EinsteinIR):
            axis_ids = {
                iv.defid for iv in self._wrt_axes if getattr(iv, "defid", None) is not None
            }
            if axis_ids:
                ein_has_axes = False
                for c in da.clauses or []:
                    for ix in c.indices or []:
                        did = getattr(ix, "defid", None)
                        if did is not None and did in axis_ids:
                            ein_has_axes = True
                            break
                    if ein_has_axes:
                        break
                if ein_has_axes:
                    have = {
                        getattr(ix, "defid", None)
                        for ix in indices
                        if getattr(ix, "defid", None) is not None
                    }
                    ext = list(indices)
                    for iv in self._wrt_axes:
                        did = getattr(iv, "defid", None)
                        if did is not None and did not in have:
                            ext.append(iv)
                            have.add(did)
                    indices = ext
        return RectangularAccessIR(da, indices, loc, type_info=_ti(n), shape_info=_si(n))
    def visit_cast_expression(self, n: CastExpressionIR) -> ExpressionIR:
        if _cast_target_has_zero_tangent(n.target_type):
            return _z(self._loc)
        return n.expr.accept(self)
    def visit_if_expression(self, n: IfExpressionIR) -> ExpressionIR:
        cv = _eval_const_expr(n.condition, self._B, subst=self._ps)
        if cv is True:
            return n.then_expr.accept(self)
        if cv is False:
            return n.else_expr.accept(self) if n.else_expr is not None else _z(self._loc)
        dt = n.then_expr.accept(self)
        de = n.else_expr.accept(self) if n.else_expr is not None else _z(self._loc)
        _reject_bare_wrt_tensor_jacobian(dt, self._wrt, self._wrt_axes, "if then")
        _reject_bare_wrt_tensor_jacobian(de, self._wrt, self._wrt_axes, "if else")
        return IfExpressionIR(condition=n.condition, then_expr=dt, location=n.location or self._loc,
                              else_expr=de, type_info=_ti(n), shape_info=_si(n))
    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        if n.final_expr is None:
            raise ValueError("Autodiff: JacobianVisitor block has no final expression")
        loc = n.location or self._loc
        stmts = [
            s
            for s in (n.statements or [])
            if isinstance(s, BindingIR) and s.defid is not None and s.expr is not None
        ]
        if not stmts:
            fp = _simplify(n.final_expr.accept(self), loc)
            return BlockExpressionIR(
                [],
                loc,
                fp,
                type_info=_ti(n),
                shape_info=_si(n),
            )
        if (
            len(stmts) == 1
            and isinstance(n.final_expr, IdentifierIR)
            and n.final_expr.defid == stmts[0].defid
        ):
            return stmts[0].expr.accept(self)
        sp_ext: Dict[DefId, ExpressionIR] = dict(self._sp or {})
        child = JacobianVisitor(
            self._wrt,
            loc,
            self._B,
            self._R,
            stmt_partial=sp_ext,
            wrt_tangent=self._wt,
            primal_subst=self._ps,
            shared_cotangent_axes=self._wrt_axes,
            legacy_directional=self._legacy_directional,
            dependency_cache=self._dep_cache,
        )
        for s in stmts:
            pv = _simplify(s.expr.accept(child), loc)
            sp_ext[s.defid] = _z(loc) if _is_zero(pv) else pv
        der = n.final_expr.accept(child)
        return BlockExpressionIR(list(stmts), loc, der, type_info=_ti(n), shape_info=_si(n))

    # -- function call -------------------------------------------------------
    def visit_function_call(self, n: FunctionCallIR) -> ExpressionIR:
        loc = n.location or self._loc; args = n.arguments or []
        cdid = n.function_defid
        lab = _function_call_ir_label(n)
        if cdid is None or cdid not in self._B:
            detail = "missing function_defid" if cdid is None else "function_defid not in binding map"
            raise ValueError(f"Autodiff: JacobianVisitor cannot differentiate unresolved call {lab!r} ({detail})")
        b = self._B[cdid]
        if not isinstance(b.expr, FunctionValueIR):
            bname = b.name or "?"
            raise ValueError(
                f"Autodiff: JacobianVisitor call {lab!r} resolves to non-function binding {bname!r}"
            )
        fv = b.expr; ps = fv.parameters or []; body = fv.body
        rm = {
            p.defid: _callee_arg_with_binding_metadata(args[j], self._B)
            for j, p in enumerate(ps)
            if p.defid is not None and j < len(args)
        }

        if self._wrt_axes is None and ps:
            tangent_by_param: Dict[DefId, ExpressionIR] = {}
            any_nonzero = False
            for i, p in enumerate(ps):
                if p.defid is None or i >= len(args):
                    continue
                da = args[i].accept(self)
                tangent_by_param[p.defid] = da
                if not _is_zero(da):
                    any_nonzero = True
            if not any_nonzero:
                return _z(loc)
            if self._R is not None:
                try:
                    return _callee_forward_jvp(
                        fv, args, tangent_by_param, loc, self._B, self._R
                    )
                except Exception:
                    # Fall back to Jacobian chain path below for edge cases where the
                    # replayed forward sweep cannot be built.
                    pass

        rule_body = getattr(fv, "custom_diff_body", None)
        if rule_body is not None and len(ps) == len(args):
            if len(ps) == 1 and ps[0].defid is not None:
                # ∂(arg)/∂wrt is wrong for forward JVP when arg is already call-site IR with no
                # wrt identifier (e.g. ln(1+exp(x)) via ln → log @fn: need (1/x)*d_arg with d_arg=wt).
                d_arg = self._wt if self._wt is not None else args[0].accept(self)
                _reject_bare_wrt_tensor_jacobian(d_arg, self._wrt, self._wrt_axes, "custom_diff d_arg")
                return _sub_callee(
                    _sub_wd(rule_body, rm, {ps[0].defid: d_arg}, loc),
                    fv,
                    rm,
                    loc,
                    fold_body_bindings=False,
                )
            terms: List[ExpressionIR] = []
            for i, p in enumerate(ps):
                if p.defid is None: continue
                ud = {ps[j].defid: (_fl(1, loc) if j == i else _z(loc)) for j in range(len(ps)) if ps[j].defid is not None}
                coef = _simplify(
                    _sub_callee(
                        _sub_wd(rule_body, rm, ud, loc),
                        fv,
                        rm,
                        loc,
                        fold_body_bindings=False,
                    ),
                    loc,
                )
                av = args[i].accept(self)
                _reject_bare_wrt_tensor_jacobian(av, self._wrt, self._wrt_axes, "custom_diff chain")
                terms.append(BinaryOpIR(BinaryOp.MUL, coef, av, loc))
            return _sum_terms(terms, loc)
        if body is None:
            raise ValueError("Autodiff: JacobianVisitor cannot differentiate function with no body")
        if isinstance(body, BlockExpressionIR) and self._R is not None:
            # Same as ``_callee_forward_jvp``: inline callee locals into call-site IR. Without
            # ``_sub_callee``, replayed primal bindings keep fresh DefIds (e.g. recurrence ``u``
            # inside ``euler_decay``) that never exist at runtime → "Variable not found".
            return _sub_callee(
                _diff_callee_block(body, self._wrt, loc, self._B, self._R, rm, self._wt),
                fv, rm, loc,
            )
        terms = []
        for i, p in enumerate(ps):
            if p.defid is None or i >= len(args): continue
            iv = JacobianVisitor(p.defid, loc, self._B, self._R, dependency_cache=self._dep_cache)
            av = args[i].accept(self)
            if (
                isinstance(args[i], IdentifierIR)
                and isinstance(av, IdentifierIR)
                and args[i].defid is not None
                and _binding_is_rect_slice_of_tensor(self._wrt, args[i].defid, self._B)
            ):
                one_hot = _jacobian_tensor_id_wrt_slice_binding(
                    IdentifierIR(
                        args[i].name or "?",
                        args[i].location or loc,
                        args[i].defid,
                        type_info=_ti(args[i]),
                        shape_info=_si(args[i]),
                    ),
                    self._wrt,
                    loc,
                    self._B,
                    self._R,
                )
                if one_hot is not None:
                    av = one_hot
            _reject_bare_wrt_tensor_jacobian(av, self._wrt, self._wrt_axes, "call arg partial")
            terms.append(BinaryOpIR(BinaryOp.MUL, _sub(body.accept(iv), rm, loc), av, loc))
        return _sum_terms(terms, loc)

    # -- einstein (index expansion for jacobians) ----------------------------
    def visit_einstein(self, n: EinsteinIR) -> ExpressionIR:
        return _diff_einstein_wrt(n, self._wrt, self._loc, self._B, self._R, self._wt,
                                  sp=self._sp, ps=self._ps,
                                  cotangent_axes=self._wrt_axes,
                                  legacy_directional=self._legacy_directional)
    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> ExpressionIR: return n

    def visit_lowered_einstein(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("JacobianVisitor", n)

    # -- unsupported IR (do not return literal zero) -------------------------
    def visit_differential(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_jagged_access(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_lambda(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_range(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_array_comprehension(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_tuple_expression(self, n: TupleExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        return TupleExpressionIR(
            [elem.accept(self) for elem in (n.elements or [])],
            loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )
    def visit_tuple_access(self, n: TupleAccessIR) -> ExpressionIR:
        loc = n.location or self._loc
        dt = n.tuple_expr.accept(self)
        if _is_zero(dt):
            return _z(loc)
        return TupleAccessIR(dt, n.index, loc, type_info=_ti(n), shape_info=_si(n))
    def visit_interpolated_string(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_member_access(self, n: MemberAccessIR) -> ExpressionIR:
        loc = n.location or self._loc
        if n.member == "shape":
            return _z(loc)
        dobj = n.object.accept(self)
        if _is_zero(dobj):
            return _z(loc)
        return MemberAccessIR(dobj, n.member, loc, type_info=_ti(n), shape_info=_si(n))
    def visit_function_value(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_try_expression(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_match_expression(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_where_expression(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_pipeline_expression(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_builtin_call(self, n: BuiltinCallIR) -> ExpressionIR:
        loc = n.location or self._loc
        if n.builtin_name in _AD_ZERO_TANGENT_BUILTINS:
            return _z(loc)
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_module(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_program(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_binding(self, n: BindingIR) -> ExpressionIR:
        if n.expr is None:
            raise ValueError("Autodiff: JacobianVisitor binding has no expression")
        return n.expr.accept(self)
    def visit_index_var(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_index_rest(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_einstein_clause(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_lowered_reduction(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("JacobianVisitor", n)
    def visit_lowered_select_at_argmax(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("JacobianVisitor", n)
    def visit_lowered_comprehension(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("JacobianVisitor", n)
    def visit_lowered_einstein_clause(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("JacobianVisitor", n)
    def visit_lowered_recurrence(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("JacobianVisitor", n)
    def visit_literal_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_identifier_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_wildcard_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_tuple_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_array_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_rest_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_guard_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_or_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_constructor_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_binding_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_range_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)

# ═══════════════════════════════════════════════════════════════════════════
# Einstein Jacobian helpers
# ═══════════════════════════════════════════════════════════════════════════

def _flatten_product(expr: ExpressionIR) -> Optional[List[Tuple[ExpressionIR, List]]]:
    if isinstance(expr, RectangularAccessIR):
        arr = expr.array
        if isinstance(arr, IdentifierIR) and arr.defid is not None:
            return [(arr, list(expr.indices or []))]
        return None
    if isinstance(expr, BinaryOpIR) and expr.operator == BinaryOp.MUL:
        l = _flatten_product(expr.left); r = _flatten_product(expr.right)
        if l is not None and r is not None: return l + r
    return None

def _build_deriv_idx(clause_indices: List, wrt_indices: List, wrt_id: IdentifierIR,
                     resolver: Any, loc: SourceLocation, allow_reuse: bool
                     ) -> Tuple[List, Set[DefId], Dict]:
    ci_by_did: Dict[DefId, Any] = {}
    for c in clause_indices:
        did = getattr(c, "defid", None)
        if did is not None: ci_by_did[did] = c
    dvars: List = []; new_dids: Set[DefId] = set(); new_vr: Dict = {}
    for p in range(len(wrt_indices)):
        ip = wrt_indices[p]; did_p = getattr(ip, "defid", None)
        if allow_reuse and did_p is not None and did_p in ci_by_did:
            dvars.append(ci_by_did[did_p])
        else:
            nd = resolver.allocate_for_local()
            sh = MemberAccessIR(object=wrt_id, member="shape", location=loc, type_info=UNKNOWN)
            dim = LiteralIR(p, loc, type_info=PrimitiveType("i32"))
            sd = RectangularAccessIR(array=sh, indices=[dim], location=loc, type_info=UNKNOWN)
            rng = RangeIR(start=LiteralIR(0, loc, type_info=PrimitiveType("i32")),
                          end=sd, location=loc, type_info=UNKNOWN)
            iv = IndexVarIR("_ad_%d" % p, loc, nd, range_ir=rng)
            dvars.append(iv); new_dids.add(nd)
            if iv.defid is not None and iv.range_ir is not None: new_vr[iv.defid] = iv.range_ir
    return dvars, new_dids, new_vr

def _merged_lr(val: ReductionExpressionIR, clause: EinsteinClauseIR) -> Dict:
    out: Dict = dict(val.loop_var_ranges or {})
    vr = clause.variable_ranges or {}
    for lv in val.loop_vars or []:
        did = getattr(lv, "defid", None)
        if did is not None and did not in out and did in vr: out[did] = vr[did]
    return out


def _diff_einstein_wrt(expr: EinsteinIR, wrt: DefId, loc: SourceLocation,
                       B: Dict[DefId, BindingIR], R: Any,
                       wt: Optional[ExpressionIR] = None,
                       sp: Optional[Dict[DefId, ExpressionIR]] = None,
                       ps: Optional[Dict[DefId, ExpressionIR]] = None,
                       cotangent_axes: Optional[List[IndexVarIR]] = None,
                       legacy_directional: bool = False) -> ExpressionIR:
    dep_cache = _DependencyQueryCache(B)
    dc: List[EinsteinClauseIR] = []
    # Legacy directional mode is only for direct tensor-over-tensor quotients.
    # For scalar-loss backprop chains, keep cotangent axes through Einstein nodes.
    shared_axes: Optional[List[IndexVarIR]] = (
        [] if legacy_directional else (list(cotangent_axes) if cotangent_axes is not None else [])
    )
    for clause in expr.clauses or []:
        val = clause.value
        if not isinstance(val, ReductionExpressionIR):
            jv = JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps,
                                 shared_cotangent_axes=shared_axes, dependency_cache=dep_cache)
            d_val = _simplify(val.accept(jv), loc)
            mi, merged_v, nvr = _merge_primal_clause_with_cotangent_einstein(clause, d_val)
            mi, nvr = _ensure_cotangent_axes_on_clause_indices(mi, nvr, merged_v, shared_axes)
            mi, nvr = _append_cotangent_axes_to_clause_indices(mi, nvr, shared_axes)
            dc.append(EinsteinClauseIR(indices=mi, value=merged_v, location=clause.location,
                      where_clause=clause.where_clause, variable_ranges=nvr))
            continue
        inner = val.body; factors = _flatten_product(inner) if inner else None
        if not factors:
            if val.operation == ReductionOp.SUM:
                d_inner = inner.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps,
                                                     shared_cotangent_axes=shared_axes, dependency_cache=dep_cache))
                sum_val = ReductionExpressionIR(ReductionOp.SUM, list(val.loop_vars or []), d_inner, loc,
                          where_clause=val.where_clause, loop_var_ranges=_merged_lr(val, clause),
                          type_info=_ti(val), shape_info=_si(val))
                mi = list(clause.indices or [])
                nvr = dict(clause.variable_ranges or {})
                mi, nvr = _ensure_cotangent_axes_on_clause_indices(mi, nvr, sum_val, shared_axes)
                mi, nvr = _append_cotangent_axes_to_clause_indices(mi, nvr, shared_axes)
                dc.append(EinsteinClauseIR(indices=mi,
                    value=sum_val,
                    location=clause.location, where_clause=clause.where_clause,
                    variable_ranges=nvr))
            else:
                d_val = val.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps,
                                                   shared_cotangent_axes=shared_axes, dependency_cache=dep_cache))
                mi = list(clause.indices or [])
                nvr = dict(clause.variable_ranges or {})
                mi, nvr = _ensure_cotangent_axes_on_clause_indices(mi, nvr, d_val, shared_axes)
                mi, nvr = _append_cotangent_axes_to_clause_indices(mi, nvr, shared_axes)
                dc.append(EinsteinClauseIR(indices=mi, value=d_val, location=clause.location,
                          where_clause=clause.where_clause, variable_ranges=nvr))
            continue
        wrt_pos: List[int] = []
        for i, (a, _) in enumerate(factors):
            if not isinstance(a, IdentifierIR) or a.defid is None:
                continue
            if a.defid == wrt:
                wrt_pos.append(i)
                continue
            # Quotient Jacobians through function calls can differentiate callee bodies with
            # ``wrt`` = caller DefId while factors reference callee parameter DefIds.
            # Treat direct parameter aliasing (param -> caller identifier) as the same leaf.
            if ps is not None:
                sub = ps.get(a.defid)
                if isinstance(sub, IdentifierIR) and sub.defid == wrt:
                    wrt_pos.append(i)
        if not wrt_pos:
            if val.operation == ReductionOp.SUM:
                d_inner = inner.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps,
                                                     shared_cotangent_axes=shared_axes, dependency_cache=dep_cache))
                sum_val = ReductionExpressionIR(ReductionOp.SUM, list(val.loop_vars or []), d_inner, loc,
                          where_clause=val.where_clause, loop_var_ranges=_merged_lr(val, clause),
                          type_info=_ti(val), shape_info=_si(val))
                mi = list(clause.indices or [])
                nvr = dict(clause.variable_ranges or {})
                mi, nvr = _ensure_cotangent_axes_on_clause_indices(mi, nvr, sum_val, shared_axes)
                mi, nvr = _append_cotangent_axes_to_clause_indices(mi, nvr, shared_axes)
                dc.append(EinsteinClauseIR(indices=mi,
                    value=sum_val,
                    location=clause.location, where_clause=clause.where_clause,
                    variable_ranges=nvr))
            continue

        first_wi = factors[wrt_pos[0]][1]
        wb = B.get(wrt); wid = IdentifierIR((wb.name if wb else "") or "?", loc, wrt, type_info=UNKNOWN)
        ci = list(clause.indices or [])
        allow_reuse = len(ci) < len(first_wi)
        dvars, new_dids, new_vr = _build_deriv_idx(ci, first_wi, wid, R, loc, allow_reuse)
        lvs = list(val.loop_vars or []); mlr = _merged_lr(val, clause)

        if val.operation in (ReductionOp.MAX, ReductionOp.MIN):
            db: ExpressionIR = _fl(1, loc)
            for p in range(len(first_wi)):
                if getattr(dvars[p], "defid", None) in new_dids:
                    eq = BinaryOpIR(BinaryOp.EQ, first_wi[p], dvars[p], loc, type_info=BOOL)
                    db = IfExpressionIR(eq, db, loc, else_expr=_z(loc), type_info=_ti(val) or F32)
            sel = SelectAtArgmaxIR(val.body, db, lvs, loop_var_ranges=mlr, location=loc,
                                   type_info=_ti(val), shape_info=_si(val),
                                   use_argmin=(val.operation == ReductionOp.MIN))
            nvr = dict(clause.variable_ranges or {}); nvr.update(new_vr)
            out_val: ExpressionIR = sel
            if not allow_reuse:
                # When primal clause indices cannot be reused as cotangent axes (e.g. pooled output
                # indices vs input indices in max_pool), accumulate over those primal output axes so
                # the quotient result keeps wrt-shape (Julia-style pullback).
                sum_loops: List[IndexVarIR] = [ix for ix in ci if isinstance(ix, IndexVarIR)]
                if sum_loops:
                    sum_ranges: Dict[DefId, RangeIR] = {}
                    cvr = dict(clause.variable_ranges or {})
                    for lv in sum_loops:
                        did = getattr(lv, "defid", None)
                        if did is not None and did in cvr:
                            sum_ranges[did] = cvr[did]
                    out_val = ReductionExpressionIR(
                        ReductionOp.SUM,
                        sum_loops,
                        sel,
                        loc,
                        loop_var_ranges=sum_ranges if sum_ranges else None,
                        type_info=_ti(val),
                        shape_info=None,
                    )
            dc.append(EinsteinClauseIR(indices=dvars, value=out_val, location=clause.location,
                      where_clause=clause.where_clause, variable_ranges=nvr))
            continue

        if val.operation == ReductionOp.PROD and allow_reuse:
            exc = [BinaryOpIR(BinaryOp.NE, first_wi[p], dvars[p], loc, type_info=BOOL)
                   for p in range(len(first_wi)) if getattr(dvars[p], "defid", None) in new_dids]
            ow = getattr(val, "where_clause", None)
            oc = list(getattr(ow, "constraints", None) or []) if ow else []
            pw = WhereClauseIR(constraints=oc + exc, location=loc) if (oc or exc) else None
            pr = ReductionExpressionIR(ReductionOp.PROD, lvs, val.body, loc,
                                       where_clause=pw, loop_var_ranges=mlr,
                                       type_info=_ti(val), shape_info=_si(val))
            nvr = dict(clause.variable_ranges or {}); nvr.update(new_vr)
            dc.append(EinsteinClauseIR(indices=dvars, value=pr, location=clause.location,
                      where_clause=clause.where_clause, variable_ranges=nvr))
            continue

        if val.operation != ReductionOp.SUM:
            d_val = val.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps,
                                               shared_cotangent_axes=shared_axes, dependency_cache=dep_cache))
            mi = list(clause.indices or [])
            nvr = dict(clause.variable_ranges or {})
            mi, nvr = _ensure_cotangent_axes_on_clause_indices(mi, nvr, d_val, shared_axes)
            mi, nvr = _append_cotangent_axes_to_clause_indices(mi, nvr, shared_axes)
            dc.append(EinsteinClauseIR(indices=mi, value=d_val, location=clause.location,
                          where_clause=clause.where_clause, variable_ranges=nvr))
            continue

        orig_rc: List[ExpressionIR] = list(getattr(val.where_clause, "constraints", None) or []) if val.where_clause else []
        rterms: List[ExpressionIR] = []
        for pos in wrt_pos:
            _, wi = factors[pos]
            others = [factors[i] for i in range(len(factors)) if i != pos]
            deltas = [BinaryOpIR(BinaryOp.EQ, wi[p], dvars[p], loc)
                      for p in range(len(wi)) if getattr(dvars[p], "defid", None) in new_dids]
            wh = WhereClauseIR(constraints=orig_rc + deltas, location=loc) if (orig_rc or deltas) else None
            if not others:
                body: ExpressionIR = _fl(1, loc)
            else:
                def _mkref(a: ExpressionIR, idxs: List) -> ExpressionIR:
                    bf = B.get(a.defid) if isinstance(a, IdentifierIR) else None
                    nm = a.name or (bf.name if bf else "") or ""
                    ref = (
                        IdentifierIR(nm, loc, a.defid, type_info=_ti(a), shape_info=_si(a))
                        if isinstance(a, IdentifierIR)
                        else a
                    )
                    return RectangularAccessIR(ref, list(idxs), loc)
                body = _mkref(*others[0])
                for ae, il in others[1:]:
                    body = BinaryOpIR(BinaryOp.MUL, body, _mkref(ae, il), loc)
            rterms.append(ReductionExpressionIR(val.operation, lvs, body, loc,
                          where_clause=wh, loop_var_ranges=mlr, type_info=_ti(val), shape_info=_si(val)))
        cv: ExpressionIR = rterms[0]
        for r in rterms[1:]: cv = BinaryOpIR(BinaryOp.ADD, cv, r, loc)
        ni = dvars if allow_reuse else (ci + dvars)
        nvr = dict(clause.variable_ranges or {}); nvr.update(new_vr)
        dc.append(EinsteinClauseIR(indices=ni, value=cv, location=clause.location,
                  where_clause=clause.where_clause, variable_ranges=nvr))

    if not dc: return _z(loc)
    dc = [
        EinsteinClauseIR(
            indices=c.indices,
            value=_unwrap_trivial_einstein_rhs(c.value),
            location=c.location,
            where_clause=c.where_clause,
            variable_ranges=dict(c.variable_ranges or {}),
        )
        for c in dc
    ]
    return EinsteinIR(clauses=dc, shape=None, element_type=expr.element_type,
                      location=expr.location, type_info=expr.type_info, shape_info=None)

# ═══════════════════════════════════════════════════════════════════════════
# Callee-body differentiation (zero-inlining)
# ═══════════════════════════════════════════════════════════════════════════

def _callee_block_build_primal(
    block: BlockExpressionIR,
    pm: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    R: Any,
) -> Tuple[List[BindingIR], Dict[DefId, ExpressionIR]]:
    """Replay callee locals once: primal bindings + map callee DefIds to fresh IdentifierIRs."""
    primal_map: Dict[DefId, ExpressionIR] = dict(pm)
    primal_stmts: List[BindingIR] = []
    for s in block.statements or []:
        if not isinstance(s, BindingIR) or s.defid is None or s.expr is None:
            continue
        subst_expr = _sub(s.expr, primal_map, loc)
        bti = _ti(s) or _ti(subst_expr)
        bsi = _si(s) or _si(subst_expr)
        pd = R.allocate_for_local()
        pn = s.name or "_p"
        pr = IdentifierIR(pn, s.location or loc, pd, type_info=bti, shape_info=bsi)
        primal_map[s.defid] = pr
        primal_stmts.append(
            BindingIR(
                name=pn,
                expr=subst_expr,
                location=s.location or loc,
                defid=pd,
                type_info=bti,
            )
        )
    return primal_stmts, primal_map


def _callee_replay_block(
    block: BlockExpressionIR,
    pm: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    R: Any,
) -> BlockExpressionIR:
    """Replay callee ``let`` bindings inside a block; mutates ``pm`` (caller passes a branch copy for if arms)."""
    bl = block.location or loc
    stmts_out: List[BindingIR] = []
    for s in block.statements or []:
        if not isinstance(s, BindingIR) or s.defid is None or s.expr is None:
            continue
        sub_e = _sub(s.expr, pm, loc)
        bti = _ti(s) or _ti(sub_e)
        bsi = _si(s) or _si(sub_e)
        pd = R.allocate_for_local()
        pn = s.name or "_p"
        pr = IdentifierIR(pn, s.location or bl, pd, type_info=bti, shape_info=bsi)
        pm[s.defid] = pr
        stmts_out.append(
            BindingIR(
                name=pn,
                expr=sub_e,
                location=s.location or bl,
                defid=pd,
                type_info=bti,
            )
        )
    fe = block.final_expr
    fe_out: Optional[ExpressionIR] = None
    if fe is not None:
        fe_out = _callee_replay_expression(fe, pm, loc, R)
    return BlockExpressionIR(
        stmts_out, bl, fe_out, type_info=_ti(block), shape_info=_si(block)
    )


def _callee_replay_expression(
    expr: ExpressionIR,
    pm: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    R: Any,
) -> ExpressionIR:
    """Substitute callee locals and allocate call-site DefIds for nested blocks / ``if`` arms."""
    if isinstance(expr, BlockExpressionIR):
        return _callee_replay_block(expr, pm, loc, R)
    if isinstance(expr, IfExpressionIR):
        el = expr.location or loc
        cond_s = _sub(expr.condition, pm, loc)
        pm_then = dict(pm)
        then_r = _callee_replay_expression(expr.then_expr, pm_then, loc, R)
        else_r: Optional[ExpressionIR] = None
        if expr.else_expr is not None:
            pm_else = dict(pm)
            else_r = _callee_replay_expression(expr.else_expr, pm_else, loc, R)
        return IfExpressionIR(
            cond_s,
            then_r,
            el,
            else_expr=else_r,
            type_info=_ti(expr),
            shape_info=_si(expr),
        )
    return _sub(expr, pm, loc)


def _stmt_partial_for_replayed_callee_final(
    sp: Dict[DefId, ExpressionIR],
    primal_map: Dict[DefId, ExpressionIR],
    block: BlockExpressionIR,
) -> Dict[DefId, ExpressionIR]:
    """Map call-site primal DefIds to stmt partials; replayed ``final_expr`` uses those ids."""
    out: Dict[DefId, ExpressionIR] = dict(sp)
    for s in block.statements or []:
        if not isinstance(s, BindingIR) or s.defid is None:
            continue
        if s.defid not in sp:
            continue
        prim = primal_map.get(s.defid)
        if isinstance(prim, IdentifierIR) and prim.defid is not None:
            out[prim.defid] = sp[s.defid]
    return out


def _tensor_root_defid_for_jacobian_fe(expr: Optional[ExpressionIR]) -> Optional[DefId]:
    """DefId of the tensor leaf for replayed IR when a parameter is bound to a slice/Einstein."""
    if expr is None:
        return None
    if isinstance(expr, IdentifierIR) and expr.defid is not None:
        return expr.defid
    if isinstance(expr, RectangularAccessIR):
        return _tensor_root_defid_for_jacobian_fe(expr.array)
    if isinstance(expr, CastExpressionIR):
        return _tensor_root_defid_for_jacobian_fe(expr.expr)
    if isinstance(expr, UnaryOpIR):
        return _tensor_root_defid_for_jacobian_fe(expr.operand)
    if isinstance(expr, BinaryOpIR):
        r = _tensor_root_defid_for_jacobian_fe(expr.left)
        if r is not None:
            return r
        return _tensor_root_defid_for_jacobian_fe(expr.right)
    if isinstance(expr, EinsteinIR):
        for c in expr.clauses or []:
            if c.value is not None:
                r = _tensor_root_defid_for_jacobian_fe(c.value)
                if r is not None:
                    return r
    if isinstance(expr, IfExpressionIR):
        r = _tensor_root_defid_for_jacobian_fe(expr.then_expr)
        if r is not None:
            return r
        return _tensor_root_defid_for_jacobian_fe(expr.else_expr)
    return None


def _diff_callee_block_tangent(
    block: BlockExpressionIR,
    wrt: DefId,
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    R: Any,
    primal_map: Dict[DefId, ExpressionIR],
    param_subst: Dict[DefId, ExpressionIR],
    wt: Optional[ExpressionIR] = None,
) -> Tuple[List[BindingIR], ExpressionIR]:
    """∂/∂wrt for callee body given an already-built ``primal_map`` (no primal statements)."""
    if block.final_expr is None:
        raise ValueError("Autodiff: callee block has no final expression")
    dep_cache = _DependencyQueryCache(B)
    sp: Dict[DefId, ExpressionIR] = {}
    # Quotient Jacobians use wrt = a call-site DefId (denominator of @y/@x); parameters are
    # keys of ``param_subst``.  Forward-mode uses wrt = callee parameter DefId ∈ param_subst.
    # primal_subst must only apply to the quotient case (see tests using std::ml).
    ps_map = primal_map if wrt not in param_subst else None
    vis = JacobianVisitor(
        wrt,
        loc,
        B,
        R,
        stmt_partial=sp,
        wrt_tangent=wt,
        primal_subst=ps_map,
        dependency_cache=dep_cache,
    )
    diff_stmts: List[BindingIR] = []
    for s in block.statements or []:
        if not isinstance(s, BindingIR) or s.defid is None or s.expr is None:
            continue
        pv = _simplify(_sub(s.expr.accept(vis), primal_map, loc), loc)
        bti = _ti(s) or _ti(pv)
        bsi = _si(s) or _si(pv)
        if _is_zero(pv):
            sp[s.defid] = _z(loc)
        else:
            dd = R.allocate_for_local()
            dn = DIFF_PREFIX + (s.name or "")
            dr = IdentifierIR(dn, s.location or loc, dd, type_info=bti, shape_info=bsi)
            axis_ids = {
                iv.defid
                for iv in (vis._wrt_axes or [])
                if getattr(iv, "defid", None) is not None
            }
            if axis_ids and _expr_uses_index_defids(pv, axis_ids):
                # Preserve explicit cotangent-axis structure for downstream indexed uses
                # (e.g. conv_sum reading _@Xp[...] inside callee-body Jacobians).
                sp[s.defid] = pv
            else:
                sp[s.defid] = dr
            diff_stmts.append(
                BindingIR(
                    name=dn,
                    expr=pv,
                    location=s.location or loc,
                    defid=dd,
                    type_info=bti,
                )
            )
    bl = block.location or loc
    pm_fp = dict(primal_map)
    fe_rep = _callee_replay_expression(block.final_expr, pm_fp, bl, R)
    w_fe = wrt
    if wrt in param_subst:
        pw = primal_map.get(wrt)
        if isinstance(pw, IdentifierIR) and pw.defid is not None:
            w_fe = pw.defid
        else:
            tr = _tensor_root_defid_for_jacobian_fe(pw)
            if tr is not None:
                w_fe = tr
    sp_fe = _stmt_partial_for_replayed_callee_final(sp, primal_map, block)
    vis_fe = JacobianVisitor(
        w_fe,
        loc,
        B,
        R,
        stmt_partial=sp_fe,
        wrt_tangent=wt,
        primal_subst=ps_map,
        dependency_cache=dep_cache,
    )
    fp = _simplify(_sub(fe_rep.accept(vis_fe), pm_fp, loc), loc)
    return diff_stmts, fp


def _callee_args_support_combined_forward(
    rm: Dict[DefId, ExpressionIR], ps: List[Any]
) -> bool:
    """True if callee JVP can use a single ``DiffVisitor`` sweep (identifiers + constant args).

    Literal (or other non-identifier) actuals are inlined in ``primal_map``; they need no
    ``dre`` entry.  Per-parameter ``JacobianVisitor`` sweeps would duplicate tangent lets.
    Array literals (e.g. ``[2, 2]`` for ``std::ml::max_pool`` kernel/strides/pads) are
    constant like scalars: zero tangent and must not force the multi-parameter Jacobian sum.
    """
    for p in ps:
        if p.defid is None:
            return False
        arg = rm.get(p.defid)
        if isinstance(arg, IdentifierIR) and arg.defid is not None:
            continue
        if isinstance(arg, LiteralIR):
            continue
        if isinstance(arg, ArrayLiteralIR):
            continue
        return False
    return True


def _diff_callee_block_combined_forward(
    block: BlockExpressionIR,
    primal_map: Dict[DefId, ExpressionIR],
    rm: Dict[DefId, ExpressionIR],
    tangent_by_param: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    R: Any,
    ps: List[Any],
) -> Tuple[List[BindingIR], ExpressionIR]:
    """Single forward-mode sweep: one ``_@v`` per callee local, full JVP Σᵢ ∂/∂pᵢ·dpᵢ.

    Call-site tangents attach to ``IdentifierIR`` arguments' ``DefId``s. Constant actuals are
    already substituted in ``primal_map`` and are omitted from ``dre``.
    """
    if block.final_expr is None:
        raise ValueError("Autodiff: callee block has no final expression")
    dre: Dict[DefId, ExpressionIR] = {}
    for p in ps:
        if p.defid is None:
            raise ValueError("Autodiff: parameter has no defid")
        da = tangent_by_param.get(p.defid)
        if da is None:
            raise ValueError("Autodiff: missing tangent for callee parameter")
        arg = rm[p.defid]
        if isinstance(arg, LiteralIR):
            continue
        if isinstance(arg, ArrayLiteralIR):
            continue
        if not isinstance(arg, IdentifierIR) or arg.defid is None:
            raise ValueError("Autodiff: combined callee forward requires identifier or literal arguments")
        dre[arg.defid] = da

    diff_stmts: List[BindingIR] = []
    bl = block.location or loc
    for s in block.statements or []:
        if not isinstance(s, BindingIR) or s.defid is None or s.expr is None:
            continue
        e_sub = _sub(s.expr, primal_map, loc)
        pv = _simplify(e_sub.accept(DiffVisitor(dre, loc, B, R)), loc)
        bti = _ti(s) or _ti(pv)
        bsi = _si(s) or _si(pv)
        prim = primal_map.get(s.defid)
        if not isinstance(prim, IdentifierIR) or prim.defid is None:
            raise ValueError("Autodiff: primal_map missing identifier for callee local")
        pd = prim.defid
        if _is_zero(pv):
            dre[pd] = _z(bl)
            continue
        dd = R.allocate_for_local()
        dn = DIFF_PREFIX + (s.name or "")
        dr = IdentifierIR(dn, s.location or bl, dd, type_info=bti, shape_info=bsi)
        dre[pd] = dr
        dre[dd] = dr
        diff_stmts.append(
            BindingIR(
                name=dn,
                expr=pv,
                location=s.location or bl,
                defid=dd,
                type_info=bti,
            )
        )
    pm_fe = dict(primal_map)
    fe_rep = _callee_replay_expression(block.final_expr, pm_fe, bl, R)
    # Final expr may still reference callee parameter DefIds while tangents were keyed by
    # call-site argument DefIds only; alias parameter -> same tangent for this pass only
    # (do not inject into the per-statement loop — multi-let bodies reuse DefId slots).
    dre_final: Dict[DefId, ExpressionIR] = dict(dre)
    for p in ps:
        if p.defid is None:
            continue
        arg = rm[p.defid]
        if not isinstance(arg, IdentifierIR) or arg.defid is None:
            continue
        if p.defid == arg.defid:
            continue
        da0 = tangent_by_param.get(p.defid)
        if da0 is None:
            continue
        dre_final[p.defid] = dre_final.get(arg.defid, da0)
    fp = _simplify(
        fe_rep.accept(
            DiffVisitor(dre_final, loc, B, R, pretty=False, keep_primal_lets=True)
        ),
        loc,
    )
    return diff_stmts, fp


def _diff_callee_block(block: BlockExpressionIR, wrt: DefId, loc: SourceLocation,
                       B: Dict[DefId, BindingIR], R: Any,
                       pm: Dict[DefId, ExpressionIR],
                       wt: Optional[ExpressionIR] = None) -> ExpressionIR:
    if block.final_expr is None:
        raise ValueError("Autodiff: callee block has no final expression")
    # Build primal_map mapping each callee-local DefId to a NEW IdentifierIR so differential
    # expressions reference named bindings rather than inlined expressions.  (Autodiff runs
    # before Einstein lowering; callee bodies must be EinsteinIR here, not lowered-* IR.)
    primal_stmts, primal_map = _callee_block_build_primal(block, pm, loc, R)
    diff_stmts, fp = _diff_callee_block_tangent(block, wrt, loc, B, R, primal_map, pm, wt)
    out_stmts = primal_stmts + diff_stmts
    if out_stmts:
        return BlockExpressionIR(out_stmts, block.location or loc, fp,
                                 type_info=_ti(block), shape_info=_si(block))
    return fp

def _sub_callee(
    expr: ExpressionIR,
    fv: FunctionValueIR,
    rm: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    fold_body_bindings: bool = True,
) -> ExpressionIR:
    cm: Dict[DefId, ExpressionIR] = dict(rm)
    body = fv.body
    if fold_body_bindings and isinstance(body, BlockExpressionIR):
        for s in body.statements or []:
            if isinstance(s, BindingIR) and s.defid is not None and s.expr is not None:
                cm[s.defid] = _sub(s.expr, cm, loc)
    return _sub(expr, cm, loc)

def _sum_terms(terms: List[ExpressionIR], loc: SourceLocation) -> ExpressionIR:
    if not terms: return _z(loc)
    out = terms[0]
    for t in terms[1:]: out = BinaryOpIR(BinaryOp.ADD, out, t, loc)
    return out


def _callee_arg_with_binding_metadata(
    arg: ExpressionIR,
    bindings: Dict[DefId, BindingIR],
) -> ExpressionIR:
    if isinstance(arg, LiteralIR):
        cloned = LiteralIR(
            arg.value,
            arg.location,
            type_info=_ti(arg),
            shape_info=_si(arg),
        )
    elif isinstance(arg, IdentifierIR):
        cloned = IdentifierIR(
            arg.name,
            arg.location,
            arg.defid,
            type_info=_ti(arg),
            shape_info=_si(arg),
        )
    else:
        cloned = copy.deepcopy(arg)
    if isinstance(cloned, IdentifierIR) and cloned.defid is not None:
        b = bindings.get(cloned.defid)
        if b is not None:
            if cloned.type_info is None:
                cloned.type_info = _ti(b) or (_ti(b.expr) if b.expr is not None else None)
            if cloned.shape_info is None:
                cloned.shape_info = _si(b) or (_si(b.expr) if b.expr is not None else None)
    return cloned


def _callee_primal_subst_map(
    fv: FunctionValueIR,
    args: List[ExpressionIR],
    bindings: Dict[DefId, BindingIR],
) -> Dict[DefId, ExpressionIR]:
    """Map each parameter defid to the corresponding call-site argument expression."""
    ps = fv.parameters or []
    return {
        p.defid: _callee_arg_with_binding_metadata(args[j], bindings)
        for j, p in enumerate(ps)
        if p.defid is not None and j < len(args)
    }


def _callee_primal_let_binding_count(fv: FunctionValueIR) -> int:
    """Number of top-level ``let`` bindings in the callee's primal body (not tangents)."""
    body = fv.body
    if not isinstance(body, BlockExpressionIR):
        return 0
    return sum(1 for s in (body.statements or []) if isinstance(s, BindingIR))


def _pretty_use_callee_tangent_block_direct(
    fv: FunctionValueIR,
    args: List[ExpressionIR],
    peeled: ExpressionIR,
) -> bool:
    """``print(@y)`` pretty: skip ``_@f_x`` wrapper when ``y = g(x)`` is unary id, no ``@fn``, ≥2 primal lets."""
    if getattr(fv, "custom_diff_body", None) is not None:
        return False
    if len(args) != 1 or not isinstance(args[0], IdentifierIR):
        return False
    if not isinstance(peeled, BlockExpressionIR):
        return False
    if not any(isinstance(s, BindingIR) for s in (peeled.statements or [])):
        return False
    return _callee_primal_let_binding_count(fv) >= 2


def _callee_forward_jvp(
    fv: FunctionValueIR,
    args: List[ExpressionIR],
    tangent_by_param: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    R: Any,
) -> ExpressionIR:
    """Forward JVP for a user function call: ``Σᵢ ∂f/∂pᵢ · dpᵢ`` in IR.

    **Unified with ``custom_diff_body``:** When ``fv.custom_diff_body`` is set and arity
    matches, the JVP is exactly ``_sub_callee(_sub_wd(rule, rm, tangent_by_param, loc), …)`` —
    the same substitution pipeline as an ``@fn`` rule written in terms of
    ``DifferentialIR(IdentifierIR(param))``. When there is no custom rule and each argument
    is a bare identifier or a literal (inlined in the primal substitution), one ``DiffVisitor``
    sweep (``_diff_callee_block_combined_forward``) emits a single ``_@v`` per callee local.
    Otherwise per-parameter ``JacobianVisitor`` sweeps are summed
    (``_diff_callee_block_tangent`` in a loop).
    """
    ps = fv.parameters or []
    rule_body = getattr(fv, "custom_diff_body", None)
    if rule_body is not None and len(ps) == len(args):
        rm = _callee_primal_subst_map(fv, args, B)
        return _sub_callee(
            _sub_wd(rule_body, rm, tangent_by_param, loc),
            fv,
            rm,
            loc,
            fold_body_bindings=False,
        )

    body = fv.body
    if body is None:
        raise ValueError("Autodiff: user function has no body")
    if len(ps) != len(args):
        raise ValueError("Autodiff: arity mismatch")
    rm = _callee_primal_subst_map(fv, args, B)
    dep_cache = _DependencyQueryCache(B)
    if isinstance(body, BlockExpressionIR) and R is not None:
        primal_stmts, primal_map = _callee_block_build_primal(body, rm, loc, R)
        if _callee_args_support_combined_forward(rm, ps):
            all_diff, acc = _diff_callee_block_combined_forward(
                body, primal_map, rm, tangent_by_param, loc, B, R, ps
            )
        else:
            all_diff = []
            fps: List[ExpressionIR] = []
            for p in ps:
                if p.defid is None:
                    raise ValueError("Autodiff: parameter has no defid")
                da = tangent_by_param.get(p.defid)
                if da is None:
                    raise ValueError("Autodiff: missing tangent for callee parameter")
                ds, fp = _diff_callee_block_tangent(body, p.defid, loc, B, R, primal_map, rm, wt=da)
                all_diff.extend(ds)
                fps.append(fp)
            if not fps:
                raise ValueError("Autodiff: callee forward JVP produced no tangent terms")
            acc = fps[0]
            for fe in fps[1:]:
                acc = BinaryOpIR(BinaryOp.ADD, acc, fe, loc)
        out: ExpressionIR
        if primal_stmts or all_diff:
            out = BlockExpressionIR(
                primal_stmts + all_diff,
                body.location or loc,
                acc,
                type_info=_ti(body),
                shape_info=_si(body),
            )
        else:
            out = acc
        return _sub_callee(out, fv, rm, loc)

    terms: List[ExpressionIR] = []
    for p in ps:
        if p.defid is None:
            raise ValueError("Autodiff: parameter has no defid")
        da = tangent_by_param.get(p.defid)
        if da is None:
            raise ValueError("Autodiff: missing tangent for callee parameter")
        iv = JacobianVisitor(p.defid, loc, B, R, wrt_tangent=da, dependency_cache=dep_cache)
        terms.append(_sub(body.accept(iv), rm, loc))
    if not terms:
        raise ValueError("Autodiff: callee forward JVP produced no tangent terms")
    out2 = _flatten_add_terms(terms, loc)
    return _sub_callee(out2, fv, rm, loc)


# ═══════════════════════════════════════════════════════════════════════════
# Block lifting
# ═══════════════════════════════════════════════════════════════════════════

def _inline_block_lets(block: BlockExpressionIR) -> ExpressionIR:
    if block.final_expr is None: return block
    stmts = [s for s in (block.statements or []) if isinstance(s, BindingIR)]
    if not stmts:
        return block.final_expr
    if len(stmts) == 1 and isinstance(block.final_expr, IdentifierIR):
        if block.final_expr.defid == stmts[0].defid and stmts[0].expr is not None:
            return stmts[0].expr
    return block

def _lift_block_binop(op: BinaryOp, L: ExpressionIR, R: ExpressionIR, loc: SourceLocation) -> ExpressionIR:
    sl: List = []; sr: List = []; l, r = L, R
    if isinstance(L, BlockExpressionIR) and L.final_expr is not None:
        sl = list(L.statements or []); l = L.final_expr
    if isinstance(R, BlockExpressionIR) and R.final_expr is not None:
        sr = list(R.statements or []); r = R.final_expr
    if not sl and not sr: return BinaryOpIR(op, L, R, loc)
    ti = _ti(L) or _ti(R); si = _si(L) or _si(R)
    return BlockExpressionIR(sl + sr, loc, BinaryOpIR(op, l, r, loc, type_info=ti, shape_info=si),
                             type_info=ti, shape_info=si)

def _flatten_add_terms(terms: List[ExpressionIR], loc: SourceLocation) -> ExpressionIR:
    if not terms: return _z(loc)
    if len(terms) == 1: return terms[0]
    ms: List[BindingIR] = []; fs: List[ExpressionIR] = []
    for t in terms:
        if isinstance(t, BlockExpressionIR) and t.final_expr is not None:
            for s in t.statements or []:
                if isinstance(s, BindingIR): ms.append(s)
            fs.append(t.final_expr)
        else:
            fs.append(t)
    acc = fs[0]
    for f in fs[1:]: acc = BinaryOpIR(BinaryOp.ADD, acc, f, loc)
    return BlockExpressionIR(ms, loc, acc) if ms else acc

# ═══════════════════════════════════════════════════════════════════════════
# DiffVisitor  — forward-mode d(expr)
# ═══════════════════════════════════════════════════════════════════════════

class DiffVisitor(IRVisitor[ExpressionIR]):
    """Forward-mode differential: d(expr) via symbolic tangent refs."""
    def __init__(self, d_map: Dict[DefId, ExpressionIR], loc: SourceLocation,
                 bindings: Optional[Dict[DefId, Any]] = None,
                 resolver: Any = None, pretty: bool = False,
                 keep_primal_lets: bool = False) -> None:
        self._d = d_map; self._loc = loc
        self._B: Dict[DefId, Any] = dict(bindings) if bindings else {}
        self._R = resolver; self._pretty = pretty
        self._keep_primal_lets = keep_primal_lets

    def visit_identifier(self, n: IdentifierIR) -> ExpressionIR:
        if n.defid is not None:
            ref = self._d.get(n.defid)
            if ref is not None:
                if isinstance(ref, IdentifierIR):
                    return IdentifierIR(
                        ref.name,
                        n.location or self._loc,
                        ref.defid,
                        type_info=_ti(ref),
                        shape_info=_si(ref),
                    )
                return ref
        raise ValueError("Autodiff: identifier not in differential map")

    def visit_literal(self, n: LiteralIR) -> ExpressionIR: return _z(self._loc)

    def visit_binary_op(self, n: BinaryOpIR) -> ExpressionIR:
        L = n.left; R = n.right; loc = n.location or self._loc
        dL = L.accept(self); dR = R.accept(self)
        op = n.operator
        if op == BinaryOp.ADD:
            return _lift_block_binop(BinaryOp.ADD, dL, dR, loc) if self._pretty else BinaryOpIR(BinaryOp.ADD, dL, dR, loc)
        if op == BinaryOp.SUB:
            return _lift_block_binop(BinaryOp.SUB, dL, dR, loc) if self._pretty else BinaryOpIR(BinaryOp.SUB, dL, dR, loc)
        if op == BinaryOp.MUL:
            return BinaryOpIR(BinaryOp.ADD,
                              BinaryOpIR(BinaryOp.MUL, L, dR, loc),
                              BinaryOpIR(BinaryOp.MUL, R, dL, loc), loc)
        if op == BinaryOp.DIV:
            num = BinaryOpIR(BinaryOp.SUB,
                             BinaryOpIR(BinaryOp.MUL, R, dL, loc),
                             BinaryOpIR(BinaryOp.MUL, L, dR, loc), loc)
            den = BinaryOpIR(BinaryOp.POW, R, _fl(2, loc), loc)
            return BinaryOpIR(BinaryOp.DIV, num, den, loc)
        if op == BinaryOp.POW: return _pow_chain(n, dL, dR, self._B, self._R, loc)
        if op == BinaryOp.MOD: return dL
        if op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.LE, BinaryOp.GT, BinaryOp.GE, BinaryOp.AND, BinaryOp.OR):
            return _z(loc)
        raise ValueError(f"Autodiff: unsupported binary op: {op}")

    def visit_unary_op(self, n: UnaryOpIR) -> ExpressionIR:
        d = n.operand.accept(self)
        if n.operator == UnaryOp.NEG: return UnaryOpIR(UnaryOp.NEG, d, n.location or self._loc)
        if n.operator == UnaryOp.POS: return d
        raise ValueError(f"Autodiff: unsupported unary op: {n.operator}")

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc; db = n.body.accept(self); op = n.operation
        if op == ReductionOp.SUM:
            return ReductionExpressionIR(ReductionOp.SUM, n.loop_vars, db, loc,
                   where_clause=n.where_clause, loop_var_ranges=n.loop_var_ranges,
                   type_info=_ti(n), shape_info=_si(n))
        if op == ReductionOp.MAX:
            return SelectAtArgmaxIR(n.body, db, n.loop_vars, loop_var_ranges=n.loop_var_ranges,
                   location=loc, type_info=_ti(n), shape_info=_si(n))
        if op == ReductionOp.MIN:
            return SelectAtArgmaxIR(n.body, db, n.loop_vars, loop_var_ranges=n.loop_var_ranges,
                   location=loc, type_info=_ti(n), shape_info=_si(n), use_argmin=True)
        if op == ReductionOp.PROD:
            return _prod_pullback_via_sum(n, db, loc, self._R)
        raise ValueError(f"Autodiff: unsupported reduction: {op}")

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        if n.final_expr is None:
            raise ValueError("Autodiff: DiffVisitor block has no final expression")
        loc = n.location or self._loc
        stmts = [
            s
            for s in (n.statements or [])
            if isinstance(s, BindingIR) and s.defid is not None and s.expr is not None
        ]
        if not stmts:
            fp = _simplify(n.final_expr.accept(self), loc)
            return BlockExpressionIR(
                [],
                loc,
                fp,
                type_info=_ti(n),
                shape_info=_si(n),
            )
        if (
            len(stmts) == 1
            and isinstance(n.final_expr, IdentifierIR)
            and n.final_expr.defid == stmts[0].defid
        ):
            return stmts[0].expr.accept(self)
        d_ext: Dict[DefId, ExpressionIR] = dict(self._d)
        child = DiffVisitor(
            d_ext, loc, self._B, self._R, self._pretty, self._keep_primal_lets
        )
        out_stmts: List[BindingIR] = []
        for s in stmts:
            nm = s.name or ""
            if self._keep_primal_lets and not _is_diff_name(nm):
                out_stmts.append(s)
            pv = _simplify(s.expr.accept(child), loc)
            if _is_zero(pv):
                d_ext[s.defid] = _z(loc)
                continue
            if self._R is not None:
                dd = self._R.allocate_for_local()
                dn = DIFF_PREFIX + (s.name or "")
                dr = IdentifierIR(
                    dn,
                    s.location or loc,
                    dd,
                    type_info=_ti(s),
                    shape_info=_si(s),
                )
                d_ext[s.defid] = dr
                out_stmts.append(
                    BindingIR(
                        name=dn,
                        expr=pv,
                        location=s.location or loc,
                        defid=dd,
                        type_info=_ti(s),
                    )
                )
            else:
                d_ext[s.defid] = pv
        fp = _simplify(n.final_expr.accept(child), loc)
        if not out_stmts:
            return fp
        return BlockExpressionIR(
            out_stmts,
            loc,
            fp,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> ExpressionIR: return n

    def visit_function_call(self, n: FunctionCallIR) -> ExpressionIR:
        loc = n.location or self._loc
        args = n.arguments or []
        cdid = n.function_defid
        lab = _function_call_ir_label(n)
        if cdid is None or cdid not in self._B:
            detail = "missing function_defid" if cdid is None else "function_defid not in autodiff binding map"
            raise ValueError(f"Autodiff: cannot differentiate unresolved function call {lab!r} ({detail})")
        binding = self._B[cdid]
        if not isinstance(binding.expr, FunctionValueIR):
            bname = binding.name or "?"
            raise ValueError(
                f"Autodiff: call {lab!r} resolves to non-function binding {bname!r}; expected a function value"
            )
        fv = binding.expr
        ps = fv.parameters or []
        tangent_by_param = {
            p.defid: args[i].accept(self) for i, p in enumerate(ps) if p.defid is not None
        }
        tang = _callee_forward_jvp(fv, args, tangent_by_param, loc, self._B, self._R)
        if self._pretty:
            peeled = _peel_inlineable_tangent_blocks(tang)
            if _pretty_callee_tangent_inlineable(peeled):
                return peeled
            if _pretty_use_callee_tangent_block_direct(fv, list(args), peeled):
                return peeled
            return _wrap_tangent_binding(
                peeled, cdid, fv, ps, list(args), n.callee_expr,
                self._B, self._R, loc, _ti(n), _si(n),
            )
        return tang

    def visit_rectangular_access(self, n: RectangularAccessIR) -> ExpressionIR:
        loc = n.location or self._loc; da = n.array.accept(self)
        if _is_zero(da): return _z(loc)
        indices = list(n.indices or [])
        if isinstance(da, EinsteinIR) and da.clauses and len(da.clauses) == 1:
            c = da.clauses[0]; ci = c.indices or []
            if len(ci) == len(indices):
                rm: Dict[DefId, ExpressionIR] = {}
                for j, cidx in enumerate(ci):
                    if isinstance(cidx, (IndexVarIR, IdentifierIR)) and cidx.defid is not None and j < len(indices):
                        rm[cidx.defid] = indices[j]
                inl = _sub(c.value, rm, loc)
                ti = _ti(n); si = _si(n)
                if ti is not None or si is not None: _propagate_ti(inl, ti, si)
                return inl
        if isinstance(da, RectangularAccessIR):
            dai = list(da.indices or [])
            if _rc_index_lists_equivalent(indices, dai):
                return da
        return RectangularAccessIR(da, indices, loc, type_info=_ti(n), shape_info=_si(n))

    def visit_if_expression(self, n: IfExpressionIR) -> ExpressionIR:
        dt = n.then_expr.accept(self)
        de = n.else_expr.accept(self) if n.else_expr is not None else _z(self._loc)
        return IfExpressionIR(condition=n.condition, then_expr=dt, location=n.location or self._loc,
                              else_expr=de, type_info=_ti(n), shape_info=_si(n))

    def visit_cast_expression(self, n: CastExpressionIR) -> ExpressionIR:
        if _cast_target_has_zero_tangent(n.target_type):
            return _z(self._loc)
        return n.expr.accept(self)

    def visit_einstein(self, n: EinsteinIR) -> ExpressionIR:
        nc: List[EinsteinClauseIR] = []
        for c in n.clauses or []:
            try: dv = _unwrap_trivial_einstein_rhs(_simplify(c.value.accept(self), c.location or self._loc))
            except (ValueError, KeyError): continue
            if _is_zero(dv): continue
            nc.append(EinsteinClauseIR(indices=list(c.indices or []), value=dv, location=c.location,
                      where_clause=c.where_clause, variable_ranges=dict(c.variable_ranges or {})))
        if not nc: return _z(self._loc)
        return EinsteinIR(clauses=nc, shape=n.shape, element_type=n.element_type,
                          location=n.location, type_info=_ti(n), shape_info=_si(n))

    def visit_lowered_einstein(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("DiffVisitor", n)

    def visit_differential(self, n: DifferentialIR) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)

    # -- unsupported IR (do not return literal zero) -----------------------
    def visit_jagged_access(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_lambda(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_range(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_array_comprehension(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_array_literal(self, n: ArrayLiteralIR) -> ExpressionIR: return _z(self._loc)
    def visit_tuple_expression(self, n: TupleExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        return TupleExpressionIR(
            [elem.accept(self) for elem in (n.elements or [])],
            loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )
    def visit_tuple_access(self, n: TupleAccessIR) -> ExpressionIR:
        loc = n.location or self._loc
        dt = n.tuple_expr.accept(self)
        if _is_zero(dt):
            return _z(loc)
        return TupleAccessIR(dt, n.index, loc, type_info=_ti(n), shape_info=_si(n))
    def visit_interpolated_string(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_member_access(self, n: MemberAccessIR) -> ExpressionIR:
        loc = n.location or self._loc
        if n.member == "shape":
            return _z(loc)
        dobj = n.object.accept(self)
        if _is_zero(dobj):
            return _z(loc)
        return MemberAccessIR(dobj, n.member, loc, type_info=_ti(n), shape_info=_si(n))
    def visit_function_value(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_try_expression(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_match_expression(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_where_expression(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_pipeline_expression(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_builtin_call(self, n: BuiltinCallIR) -> ExpressionIR:
        loc = n.location or self._loc
        if n.builtin_name in _AD_ZERO_TANGENT_BUILTINS:
            return _z(loc)
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_module(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_program(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_binding(self, n: BindingIR) -> ExpressionIR:
        if n.expr is None:
            raise ValueError("Autodiff: DiffVisitor binding has no expression")
        return n.expr.accept(self)
    def visit_index_var(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_index_rest(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_einstein_clause(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_lowered_reduction(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("DiffVisitor", n)
    def visit_lowered_select_at_argmax(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("DiffVisitor", n)
    def visit_lowered_comprehension(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("DiffVisitor", n)
    def visit_lowered_einstein_clause(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("DiffVisitor", n)
    def visit_lowered_recurrence(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("DiffVisitor", n)
    def visit_literal_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_identifier_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_wildcard_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_tuple_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_array_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_rest_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_guard_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_or_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_constructor_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_binding_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_range_pattern(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)


def _peel_inlineable_tangent_blocks(expr: ExpressionIR) -> ExpressionIR:
    """Peel ``let _@x = rhs; _@x``-style blocks until fixed point (bounded)."""
    cur: ExpressionIR = expr
    for _ in range(64):
        if not isinstance(cur, BlockExpressionIR):
            return cur
        nxt = _inline_block_lets(cur)
        if nxt is cur:
            return cur
        cur = nxt
    return cur


def _pretty_callee_tangent_inlineable(e: ExpressionIR) -> bool:
    """True if ``print(@y)`` can show callee JVP without an extra ``_@…`` wrapper binding."""
    if isinstance(e, BlockExpressionIR):
        stmts = [s for s in (e.statements or []) if isinstance(s, BindingIR)]
        if stmts:
            return False
        if e.final_expr is None:
            return False
        return _pretty_callee_tangent_inlineable(e.final_expr)
    if isinstance(e, (LiteralIR, IdentifierIR, IndexVarIR, IndexRestIR)):
        return True
    if isinstance(e, UnaryOpIR):
        return _pretty_callee_tangent_inlineable(e.operand)
    if isinstance(e, BinaryOpIR):
        return _pretty_callee_tangent_inlineable(e.left) and _pretty_callee_tangent_inlineable(e.right)
    if isinstance(e, CastExpressionIR):
        return _pretty_callee_tangent_inlineable(e.expr)
    if isinstance(e, FunctionCallIR):
        if e.callee_expr is not None and not _pretty_callee_tangent_inlineable(e.callee_expr):
            return False
        return all(_pretty_callee_tangent_inlineable(a) for a in (e.arguments or []))
    if isinstance(e, BuiltinCallIR):
        return all(_pretty_callee_tangent_inlineable(a) for a in (e.args or []))
    if isinstance(e, RectangularAccessIR):
        if not _pretty_callee_tangent_inlineable(e.array):
            return False
        return all(_pretty_callee_tangent_inlineable(i) for i in (e.indices or []))
    return False


def _wrap_tangent_binding(tang: ExpressionIR, cdid: DefId, fv: FunctionValueIR,
                          ps: List, args: List[ExpressionIR], callee_expr: ExpressionIR,
                          B: Dict, R: Any, loc: SourceLocation, ti: Any, si: Any) -> ExpressionIR:
    cn = getattr(callee_expr, "name", None) or "f"
    dd = R.allocate_for_local()
    # Internal binding name; print(@y) maps defid → "@" + tail (see PRINT_DIFFERENTIAL.md).
    if len(args) == 1 and isinstance(args[0], IdentifierIR):
        an = args[0].name or ""
        if len(cn) == 1:
            dn = DIFF_PREFIX + cn + an
        else:
            dn = DIFF_PREFIX + cn + "_" + an
    elif len(args) > 1:
        dn = DIFF_PREFIX + cn + "_call"
    else:
        dn = DIFF_PREFIX + cn
    db = BindingIR(name=dn, expr=tang, location=loc, defid=dd, type_info=ti)
    dr = IdentifierIR(dn, loc, dd, type_info=ti, shape_info=si)
    return BlockExpressionIR([db], loc, dr, type_info=ti, shape_info=si)

# ═══════════════════════════════════════════════════════════════════════════
# Formatting  — print(@y) support
# ═══════════════════════════════════════════════════════════════════════════

def _idx_str(idx: Any) -> str:
    if isinstance(idx, IndexVarIR): return idx.name or "?"
    if isinstance(idx, IndexRestIR): return f"..{idx.name}" if idx.name else ".."
    if isinstance(idx, IdentifierIR): return idx.name or "?"
    return "?"

def _str_ir(expr: ExpressionIR) -> str:
    """Render an IR expression as pseudo-einlang code via each node's ``__str__``."""
    return str(expr)


def _live_primal_defids_for_print_display(
    stmts: List[Any],
    final_expr: Optional[ExpressionIR],
    dep_cache: Optional[_DependencyQueryCache] = None,
) -> Set[DefId]:
    """Primal (non-``_@``) binding DefIds referenced from tangent lines or the block final."""
    def _deps(expr: Optional[ExpressionIR]) -> Set[DefId]:
        if dep_cache is not None:
            return set(dep_cache.collect_defids(expr))
        return _collect_defids(expr)

    need: Set[DefId] = set()
    if final_expr is not None:
        need |= _deps(final_expr)
    for s in stmts:
        if isinstance(s, BindingIR) and s.expr is not None and _is_diff_name(s.name or ""):
            need |= _deps(s.expr)
    primal_by_did: Dict[DefId, BindingIR] = {}
    for s in stmts:
        if isinstance(s, BindingIR) and s.defid is not None and not _is_diff_name(s.name or ""):
            primal_by_did[s.defid] = s
    while True:
        prev = len(need)
        for pd in list(need):
            pb = primal_by_did.get(pd)
            if pb is not None and pb.expr is not None:
                need |= _deps(pb.expr)
        if len(need) == prev:
            break
    return set(primal_by_did.keys()) & need


class _TrimDeadPrimalPrintRewriter(_Rewriter):
    """Drop callee primal replay lines unused by ``_@*`` / final (``print(@y)`` display only)."""

    def __init__(
        self,
        loc: SourceLocation,
        dep_cache: Optional[_DependencyQueryCache] = None,
    ) -> None:
        super().__init__(loc)
        self._dep_cache = dep_cache

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        if n.final_expr is None:
            return n
        ns: List[Any] = []
        for s in n.statements or []:
            if isinstance(s, BindingIR) and s.expr is not None:
                ns.append(
                    BindingIR(
                        name=s.name,
                        expr=s.expr.accept(self),
                        location=s.location,
                        defid=s.defid,
                        type_info=_ti(s),
                    )
                )
            elif isinstance(s, ExpressionIR):
                ns.append(s.accept(self))
            else:
                ns.append(s)
        nf = n.final_expr.accept(self)
        live = _live_primal_defids_for_print_display(ns, nf, self._dep_cache)
        out: List[Any] = []
        for s in ns:
            if isinstance(s, BindingIR):
                nm = s.name or ""
                if _is_diff_name(nm) or s.defid is None or s.defid in live:
                    out.append(s)
            else:
                out.append(s)
        return BlockExpressionIR(out, loc, nf, type_info=_ti(n), shape_info=_si(n))


def _str_ir_print_differential_rhs(
    expr: ExpressionIR,
    loc: SourceLocation,
    dep_cache: Optional[_DependencyQueryCache] = None,
) -> str:
    """``_str_ir`` for ``print(@…)`` after eliding dead callee primals (does not change executed IR)."""
    trimmed = expr.accept(_TrimDeadPrimalPrintRewriter(loc, dep_cache))
    return str(trimmed)


_expr_to_diff_source = _str_ir

def _fmt_print_msg(lhs: str, rhs: str) -> str:
    return "let " + lhs + " = " + rhs.rstrip("\n") + ";"

_format_print_differential_message = _fmt_print_msg



# ═══════════════════════════════════════════════════════════════════════════
# Target collection  (unified: one walk → targets + quotient pairs)
# ═══════════════════════════════════════════════════════════════════════════

def _collect_targets(node: Any) -> Tuple[List[Tuple[DefId, str]], List[Tuple[DefId, DefId]]]:
    c = _TargetCollector()
    node.accept(c)
    return c.targets, c.quotient_pairs

_collect_autodiff_targets = _collect_targets

def _collect_targets_expr(expr: Any) -> Tuple[List[Tuple[DefId, str]], List[Tuple[DefId, DefId]]]:
    return _collect_targets(expr)

# ═══════════════════════════════════════════════════════════════════════════
# Graph utilities
# ═══════════════════════════════════════════════════════════════════════════

def _is_reachable(src: DefId, tgt: DefId, B: Dict[DefId, BindingIR]) -> bool:
    return _is_reachable_with_cache(src, tgt, B, None)


def _is_reachable_with_cache(
    src: DefId,
    tgt: DefId,
    B: Dict[DefId, BindingIR],
    dep_cache: Optional[_DependencyQueryCache],
) -> bool:
    if dep_cache is not None:
        return tgt in dep_cache.reachable_from(src)
    vis: Set[DefId] = set(); q = [src]
    while q:
        cur = q.pop()
        if cur == tgt: return True
        if cur in vis: continue
        vis.add(cur)
        b = B.get(cur)
        if b is not None and b.expr is not None:
            for d in _collect_defids(b.expr):
                if d not in vis: q.append(d)
    return False

class _TypePropagator(_DefIdCollector):
    """Walk-and-mutate: fill missing ``type_info`` / ``shape_info`` like legacy ``_propagate_ti``."""

    def __init__(self, ti: Any, si: Any) -> None:
        super().__init__()
        self._ti = ti
        self._si = si

    def _stamp(self, n: Any) -> None:
        if self._ti is None:
            ti_apply = None
        elif isinstance(n, LiteralIR):
            ti_apply = F32 if isinstance(self._ti, RectangularType) else self._ti
        elif isinstance(n, IndexVarIR):
            ti_apply = I32 if isinstance(self._ti, RectangularType) else self._ti
        else:
            ti_apply = self._ti
        if ti_apply is not None and hasattr(n, "type_info") and n.type_info is None:
            n.type_info = ti_apply
        if isinstance(n, (LiteralIR, IndexVarIR)):
            return
        if hasattr(n, "shape_info") and n.shape_info is None and self._si is not None:
            n.shape_info = self._si

    def visit_literal(self, n: LiteralIR) -> None:
        self._stamp(n)

    def visit_identifier(self, n: IdentifierIR) -> None:
        self._stamp(n)
        if n.defid is not None:
            self.defids.add(n.defid)

    def visit_binary_op(self, n: BinaryOpIR) -> None:
        self._stamp(n)
        n.left.accept(self)
        n.right.accept(self)

    def visit_unary_op(self, n: UnaryOpIR) -> None:
        self._stamp(n)
        n.operand.accept(self)

    def visit_block_expression(self, n: BlockExpressionIR) -> None:
        self._stamp(n)
        for s in n.statements or []:
            if isinstance(s, BindingIR):
                bti = s.type_info or self._ti
                if s.type_info is None and bti:
                    s.type_info = bti
                if s.expr is not None:
                    s.expr.accept(_TypePropagator(bti, self._si))
            elif isinstance(s, ExpressionIR):
                s.accept(self)
        if n.final_expr is not None:
            n.final_expr.accept(self)

    def visit_einstein(self, n: EinsteinIR) -> None:
        self._stamp(n)
        for c in n.clauses or []:
            if not isinstance(c, EinsteinClauseIR):
                continue
            for idx in c.indices or []:
                if isinstance(idx, ExpressionIR):
                    idx.accept(self)
            if c.value is not None:
                c.value.accept(self)
            wc = c.where_clause
            if wc is not None:
                for ct in wc.constraints or []:
                    if isinstance(ct, ExpressionIR):
                        ct.accept(self)
            for _k, v in (c.variable_ranges or {}).items():
                if isinstance(v, RangeIR):
                    v.start.accept(self)
                    v.end.accept(self)
                elif isinstance(v, ExpressionIR):
                    v.accept(self)

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> None:
        self._stamp(n)
        n.body.accept(self)
        for _k, v in (n.loop_var_ranges or {}).items():
            if isinstance(v, RangeIR):
                v.start.accept(self)
                v.end.accept(self)
            elif isinstance(v, ExpressionIR):
                v.accept(self)
        wc = n.where_clause
        if wc is not None:
            for ct in wc.constraints or []:
                if isinstance(ct, ExpressionIR):
                    ct.accept(self)

    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> None:
        self._stamp(n)
        if n.primal_body is not None:
            n.primal_body.accept(self)
        if n.diff_body is not None:
            n.diff_body.accept(self)
        for _k, v in (n.loop_var_ranges or {}).items():
            if isinstance(v, RangeIR):
                v.start.accept(self)
                v.end.accept(self)
            elif isinstance(v, ExpressionIR):
                v.accept(self)

    def visit_rectangular_access(self, n: RectangularAccessIR) -> None:
        self._stamp(n)
        n.array.accept(self)
        for i in n.indices or []:
            if isinstance(i, ExpressionIR):
                i.accept(self)

    def visit_if_expression(self, n: IfExpressionIR) -> None:
        self._stamp(n)
        n.condition.accept(self)
        n.then_expr.accept(self)
        if n.else_expr is not None:
            n.else_expr.accept(self)

    def visit_function_call(self, n: FunctionCallIR) -> None:
        self._stamp(n)
        for a in n.arguments or []:
            a.accept(self)

    def visit_cast_expression(self, n: CastExpressionIR) -> None:
        self._stamp(n)
        n.expr.accept(self)

    def visit_builtin_call(self, n: BuiltinCallIR) -> None:
        self._stamp(n)
        for a in n.args or []:
            a.accept(self)

    def visit_differential(self, n: DifferentialIR) -> None:
        self._stamp(n)

    def visit_jagged_access(self, n: JaggedAccessIR) -> None:
        self._stamp(n)

    def visit_lambda(self, n: LambdaIR) -> None:
        self._stamp(n)

    def visit_range(self, n: RangeIR) -> None:
        self._stamp(n)
        n.start.accept(self)
        n.end.accept(self)

    def visit_array_comprehension(self, n: ArrayComprehensionIR) -> None:
        self._stamp(n)

    def visit_array_literal(self, n: ArrayLiteralIR) -> None:
        self._stamp(n)

    def visit_tuple_expression(self, n: TupleExpressionIR) -> None:
        self._stamp(n)

    def visit_tuple_access(self, n: TupleAccessIR) -> None:
        self._stamp(n)

    def visit_interpolated_string(self, n: InterpolatedStringIR) -> None:
        self._stamp(n)

    def visit_member_access(self, n: MemberAccessIR) -> None:
        self._stamp(n)
        n.object.accept(self)

    def visit_try_expression(self, n: TryExpressionIR) -> None:
        self._stamp(n)

    def visit_match_expression(self, n: MatchExpressionIR) -> None:
        self._stamp(n)

    def visit_where_expression(self, n: WhereExpressionIR) -> None:
        self._stamp(n)

    def visit_pipeline_expression(self, n: PipelineExpressionIR) -> None:
        self._stamp(n)

    def visit_function_value(self, n: FunctionValueIR) -> None:
        self._stamp(n)

    def visit_index_var(self, n: IndexVarIR) -> None:
        self._stamp(n)

    def visit_index_rest(self, n: IndexRestIR) -> None:
        self._stamp(n)
        if getattr(n, "defid", None) is not None:
            self.defids.add(n.defid)

    def visit_einstein_clause(self, n: EinsteinClauseIR) -> None:
        if n.value is not None:
            n.value.accept(self)

    def visit_lowered_reduction(self, n: Any) -> None:
        self._stamp(n)

    def visit_lowered_select_at_argmax(self, n: Any) -> None:
        self._stamp(n)

    def visit_lowered_comprehension(self, n: Any) -> None:
        self._stamp(n)

    def visit_lowered_einstein_clause(self, n: Any) -> None:
        self._stamp(n)

    def visit_lowered_einstein(self, n: Any) -> None:
        self._stamp(n)

    def visit_lowered_recurrence(self, n: Any) -> None:
        self._stamp(n)


def _propagate_ti(expr: Any, ti: Any, si: Any) -> None:
    if expr is None:
        return
    if isinstance(expr, BindingIR):
        if expr.expr is not None:
            _propagate_ti(expr.expr, ti, si)
        return
    if isinstance(expr, ExpressionIR):
        expr.accept(_TypePropagator(ti, si))


_set_type_info = _propagate_ti

def _bindings_in(block: Any, program: Optional[ProgramIR] = None) -> List[BindingIR]:
    if block is program or isinstance(block, ProgramIR):
        return [b for b in (program.bindings or []) if isinstance(b, BindingIR)] if program else []
    if isinstance(block, BlockExpressionIR):
        return [s for s in (block.statements or []) if isinstance(s, BindingIR)]
    return []

_bindings_in_block = _bindings_in

# ═══════════════════════════════════════════════════════════════════════════
# Expansion  (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════

def _primal_to_diff_map(bindings: List) -> Dict[DefId, BindingIR]:
    out: Dict[DefId, BindingIR] = {}
    lst = list(bindings or [])
    for i in range(len(lst) - 1):
        a, b = lst[i], lst[i + 1]
        if not isinstance(a, BindingIR) or not isinstance(b, BindingIR): continue
        if a.defid is None: continue
        bn = b.name or ""; an2 = a.name or ""
        if _is_diff_name(bn) and (bn == DIFF_PREFIX + an2 or bn == USER_DIFF_PREFIX + an2):
            out[a.defid] = b
    return out

_primal_to_following_diff_binding_map = _primal_to_diff_map

def _trans_deps(
    expr: Optional[ExpressionIR],
    B: Dict[DefId, BindingIR],
    dep_cache: Optional[_DependencyQueryCache] = None,
) -> Set[DefId]:
    out: Set[DefId] = set()
    vis: Set[DefId] = set()
    pending: List[DefId] = []
    roots = dep_cache.collect_defids(expr) if dep_cache is not None else _collect_defids(expr)
    for d in roots:
        pending.append(d)
    while pending:
        did = pending.pop()
        if did in vis:
            continue
        vis.add(did)
        out.add(did)
        b = B.get(did)
        if b is None or b.expr is None or isinstance(b.expr, FunctionValueIR):
            continue
        deps = dep_cache.collect_defids(b.expr) if dep_cache is not None else _collect_defids(b.expr)
        for d2 in deps:
            if d2 not in vis:
                pending.append(d2)
    return out

_transitive_primal_dep_defids_from_expr = _trans_deps


class _ExpansionVisitor(_Rewriter):
    """Phase-3 expansion: ``DifferentialIR``, quotients, ``print(@y)``, scoped blocks."""

    def __init__(
        self,
        D: Dict[DefId, IdentifierIR],
        SB: Dict[DefId, Any],
        SE: Dict[DefId, ExpressionIR],
        loc: SourceLocation,
        R: Any = None,
        P: Optional[ProgramIR] = None,
        dependency_cache: Optional[_DependencyQueryCache] = None,
    ) -> None:
        super().__init__(loc)
        self._D = D
        self._SB = SB
        self._SE = SE
        self._R = R
        self._P = P
        self._dep_cache = (
            dependency_cache
            if dependency_cache is not None and dependency_cache.bindings is SB
            else _DependencyQueryCache(SB)
        )

    def visit_differential(self, n: DifferentialIR) -> ExpressionIR:
        op = n.operand
        if isinstance(op, IdentifierIR) and op.defid is not None:
            ref = self._D.get(op.defid)
            if ref is not None:
                return IdentifierIR(ref.name, n.location, ref.defid, type_info=_ti(n), shape_info=_si(n))
        return op.accept(DiffVisitor(self._D, self._loc, self._SB, self._R))

    def visit_binary_op(self, n: BinaryOpIR) -> ExpressionIR:
        if (n.operator == BinaryOp.DIV and isinstance(n.left, DifferentialIR)
                and isinstance(n.right, DifferentialIR)):
            ql = n.location or self._loc
            nop = n.left.operand
            dop = n.right.operand
            if isinstance(dop, IdentifierIR) and dop.defid is not None:
                dd = dop.defid
                ne = self._SE.get(nop.defid) if isinstance(nop, IdentifierIR) and nop.defid is not None else nop
                if ne is None:
                    raise ValueError("Autodiff: numerator has no defining expr")
                cdn = dd in self._dep_cache.collect_defids(ne)
                rch = False
                if isinstance(nop, IdentifierIR) and nop.defid is not None:
                    rch = _is_reachable_with_cache(nop.defid, dd, self._SB, self._dep_cache)
                    if not rch:
                        den_b = self._SB.get(dd)
                        t_root = _rectangular_read_root_defid(den_b.expr) if den_b is not None and den_b.expr is not None else None
                        if t_root is not None:
                            rch = _is_reachable_with_cache(nop.defid, t_root, self._SB, self._dep_cache)
                if not cdn and not rch:
                    return _z(ql)
                den_binding = self._SB.get(dd)
                den_is_tensor = den_binding is not None and _tensor_rank_from_binding(den_binding) > 0
                if isinstance(ne, FunctionCallIR) and den_is_tensor:
                    # Keep @num/@den as d_num / d_den for tensor-denominator call quotients.
                    # Runtime executes these with cotangent seeding and avoids exploding
                    # symbolic Jacobians in expanded call bodies.
                    nL = n.left.accept(self)
                    nR = n.right.accept(self)
                    out = BinaryOpIR(BinaryOp.DIV, nL, nR, ql, type_info=_ti(n), shape_info=_si(n))
                    ti = _ti(n)
                    si = _si(n)
                    if ti or si:
                        _propagate_ti(out, ti, si)
                    return out
                legacy_directional = den_is_tensor and (
                    _tensor_rank_from_expr(ne, self._SB) > 0
                    or (
                        isinstance(ne, ReductionExpressionIR)
                        and ne.operation == ReductionOp.SUM
                    )
                )
                jv = JacobianVisitor(
                    dd,
                    ql,
                    self._SB,
                    self._R,
                    legacy_directional=legacy_directional,
                    dependency_cache=self._dep_cache,
                )
                der = _simplify(ne.accept(jv), ql)
                den_ti = None
                if dd in self._SB:
                    den_b = self._SB.get(dd)
                    den_ti = _ti(den_b)
                    if den_ti is None and den_b is not None:
                        den_ti = _ti(getattr(den_b, "expr", None))
                ti = _ti(der)
                if ti is None:
                    ti = den_ti if den_ti is not None else _ti(n)
                if ti is not None:
                    if isinstance(der, ExpressionIR):
                        der.type_info = ti
                        if hasattr(der, "shape_info"):
                            der.shape_info = None
                    _propagate_ti(der, ti, None)
                return der
            dids = self._dep_cache.collect_defids(dop)
            if len(dids) != 1:
                raise ValueError("Autodiff: @num/@(expr) denominator depends on != 1 variable")
            wd = next(iter(dids))
            legacy_directional = (
                _tensor_rank_from_expr(nop, self._SB) > 0
                or (
                    isinstance(nop, ReductionExpressionIR)
                    and nop.operation == ReductionOp.SUM
                )
            )
            dn = nop.accept(
                JacobianVisitor(
                    wd,
                    self._loc,
                    self._SB,
                    self._R,
                    legacy_directional=legacy_directional,
                    dependency_cache=self._dep_cache,
                )
            )
            dd_ = dop.accept(
                JacobianVisitor(
                    wd,
                    self._loc,
                    self._SB,
                    self._R,
                    legacy_directional=legacy_directional,
                    dependency_cache=self._dep_cache,
                )
            )
            der = BinaryOpIR(BinaryOp.DIV, dn, dd_, self._loc)
            ti = _ti(n)
            if ti is not None:
                if isinstance(der, ExpressionIR):
                    der.type_info = ti
                    if hasattr(der, "shape_info"):
                        der.shape_info = None
                _propagate_ti(der, ti, None)
            return der
        nL = n.left.accept(self)
        nR = n.right.accept(self)
        return BinaryOpIR(n.operator, nL, nR, n.location, type_info=_ti(n), shape_info=_si(n))

    def visit_unary_op(self, n: UnaryOpIR) -> ExpressionIR:
        return UnaryOpIR(
            n.operator,
            n.operand.accept(self),
            n.location,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_einstein(self, n: EinsteinIR) -> ExpressionIR:
        nc: List[EinsteinClauseIR] = []
        for c in n.clauses or []:
            cv = c.value.accept(self) if c.value is not None else None
            if cv is not None:
                cv = _unwrap_trivial_einstein_rhs(cv)
            nc.append(
                EinsteinClauseIR(
                    indices=c.indices,
                    value=cv,
                    location=c.location,
                    where_clause=c.where_clause,
                    variable_ranges=dict(c.variable_ranges) if c.variable_ranges else {},
                )
            )
        return EinsteinIR(
            clauses=nc,
            shape=n.shape,
            element_type=n.element_type,
            location=n.location,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        # ``_ensure_block_d`` mutates ``block.statements`` in place. The same ``BlockExpressionIR``
        # can be shared between ``tcx.function_ir_map`` (pub fn body) and a top-level expr subtree;
        # mutating ``n`` would corrupt callee bodies (e.g. recurrence ``u`` inside euler_decay).
        blk = BlockExpressionIR(
            list(n.statements or []),
            n.location or self._loc,
            n.final_expr,
            type_info=_ti(n),
            shape_info=_si(n),
        )
        _ensure_block_d(blk, self._SB, self._SE, self._D, self._loc, self._R)
        nsb = dict(self._SB)
        nse = dict(self._SE)
        child_dep_cache = self._dep_cache.fork(nsb)
        ns: List[Any] = []
        for s in blk.statements or []:
            if isinstance(s, BindingIR):
                v = _ExpansionVisitor(
                    self._D,
                    nsb,
                    nse,
                    self._loc,
                    self._R,
                    self._P,
                    dependency_cache=child_dep_cache,
                )
                ex = s.expr.accept(v) if s.expr is not None else None
                nb = BindingIR(name=s.name, expr=ex, location=s.location, defid=s.defid, type_info=_ti(s))
                if nb.defid is not None:
                    nsb[nb.defid] = nb
                    nse[nb.defid] = ex
                    child_dep_cache = child_dep_cache.fork(nsb)
                ns.append(nb)
            elif isinstance(s, ExpressionIR):
                v = _ExpansionVisitor(
                    self._D,
                    nsb,
                    nse,
                    self._loc,
                    self._R,
                    self._P,
                    dependency_cache=child_dep_cache,
                )
                ns.append(s.accept(v))
            else:
                ns.append(s)
        vfin = _ExpansionVisitor(
            self._D,
            nsb,
            nse,
            self._loc,
            self._R,
            self._P,
            dependency_cache=child_dep_cache,
        )
        nf = blk.final_expr.accept(vfin) if blk.final_expr is not None else None
        return BlockExpressionIR(ns, blk.location, nf, type_info=_ti(n), shape_info=_si(n))

    def visit_function_call(self, n: FunctionCallIR) -> ExpressionIR:
        na = [a.accept(self) for a in (n.arguments or [])]
        return FunctionCallIR(
            callee_expr=n.callee_expr,
            location=n.location,
            arguments=na,
            module_path=getattr(n, "module_path", None),
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_builtin_call(self, n: BuiltinCallIR) -> ExpressionIR:
        args = n.args or []
        if n.builtin_name == "print" and len(args) == 1 and isinstance(args[0], DifferentialIR):
            op = args[0].operand
            if isinstance(op, IdentifierIR) and op.defid is not None:
                yd, yn = op.defid, op.name or ""
                ye = self._SE.get(yd)
                if ye is not None:
                    try:
                        fv = DiffVisitor(self._D, self._loc, self._SB, self._R, pretty=True)
                        dr = _simplify(ye.accept(fv), n.location or self._loc)
                        pre: List[str] = []
                        P = self._P
                        if P is not None:
                            dm = _primal_to_diff_map(P.bindings or [])
                            needed = _trans_deps(ye, self._SB, self._dep_cache)
                            porder: Dict[DefId, int] = {}
                            for idx, bb in enumerate(P.bindings or []):
                                if (isinstance(bb, BindingIR) and bb.defid is not None
                                        and not _is_diff_name(bb.name or "")):
                                    if bb.defid not in porder:
                                        porder[bb.defid] = idx
                            for did in sorted((d for d in needed if d != yd), key=lambda d: porder.get(d, 10**9)):
                                pb = self._SB.get(did)
                                if pb is None or pb.expr is None or isinstance(pb.expr, FunctionValueIR):
                                    continue
                                if not self._dep_cache.collect_defids(pb.expr):
                                    continue
                                db = dm.get(did)
                                if db is None or db.expr is None:
                                    continue
                                nm = (pb.name if pb and getattr(pb, "name", None) else None) or "?"
                                pb_idx = ""
                                if isinstance(pb.expr, EinsteinIR):
                                    pb_cc = pb.expr.clauses or []
                                    if pb_cc and pb_cc[0].indices:
                                        pb_idx = ", ".join(_idx_str(i) for i in pb_cc[0].indices)
                                pre_lhs = "@" + nm + ("[" + pb_idx + "]" if pb_idx else "")
                                se = _simplify(db.expr, n.location or self._loc)
                                pre.append("let " + pre_lhs + " = " + _str_ir(se) + ";")
                        rhs = _str_ir_print_differential_rhs(dr, n.location or self._loc)
                        lhs = "@" + (yn or "?")
                        if isinstance(ye, EinsteinIR) and ye.clauses and len(ye.clauses) == 1:
                            idx_s = ", ".join(_idx_str(i) for i in (ye.clauses[0].indices or []))
                            if idx_s:
                                lhs += "[" + idx_s + "]"
                        if pre:
                            msg = "\n".join(pre) + "\n" + _fmt_print_msg(lhs, rhs)
                        else:
                            msg = _fmt_print_msg(lhs, rhs)
                        return BuiltinCallIR(
                            "print",
                            [LiteralIR(msg, n.location, type_info=STR)],
                            n.location,
                            defid=getattr(n, "defid", None),
                            type_info=_ti(n),
                            shape_info=_si(n),
                        )
                    except (ValueError, KeyError):
                        pass
        na = [a.accept(self) for a in args]
        return BuiltinCallIR(
            n.builtin_name,
            na,
            n.location,
            defid=getattr(n, "defid", None),
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_rectangular_access(self, n: RectangularAccessIR) -> ExpressionIR:
        return n

    def visit_if_expression(self, n: IfExpressionIR) -> ExpressionIR:
        nc = n.condition.accept(self)
        nt = n.then_expr.accept(self)
        ne = n.else_expr.accept(self) if n.else_expr is not None else None
        return IfExpressionIR(
            nc,
            nt,
            n.location or self._loc,
            else_expr=ne,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_cast_expression(self, n: CastExpressionIR) -> ExpressionIR:
        return n

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> ExpressionIR:
        return n

    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> ExpressionIR:
        return n

    def visit_jagged_access(self, n: JaggedAccessIR) -> ExpressionIR:
        return n

    def visit_lambda(self, n: LambdaIR) -> ExpressionIR:
        return n

    def visit_range(self, n: RangeIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_array_comprehension(self, n: ArrayComprehensionIR) -> ExpressionIR:
        return n

    def visit_module(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_array_literal(self, n: ArrayLiteralIR) -> ExpressionIR:
        return n

    def visit_tuple_expression(self, n: TupleExpressionIR) -> ExpressionIR:
        return n

    def visit_tuple_access(self, n: TupleAccessIR) -> ExpressionIR:
        return n

    def visit_interpolated_string(self, n: InterpolatedStringIR) -> ExpressionIR:
        return n

    def visit_member_access(self, n: MemberAccessIR) -> ExpressionIR:
        return n

    def visit_try_expression(self, n: TryExpressionIR) -> ExpressionIR:
        return n

    def visit_match_expression(self, n: MatchExpressionIR) -> ExpressionIR:
        return n

    def visit_where_expression(self, n: WhereExpressionIR) -> ExpressionIR:
        return n

    def visit_pipeline_expression(self, n: PipelineExpressionIR) -> ExpressionIR:
        return n

    def visit_function_value(self, n: FunctionValueIR) -> ExpressionIR:
        return n

    def visit_einstein_clause(self, n: EinsteinClauseIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_binding(self, n: BindingIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_program(self, n: ProgramIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_lowered_reduction(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("ExpansionVisitor", n)

    def visit_lowered_select_at_argmax(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("ExpansionVisitor", n)

    def visit_lowered_comprehension(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("ExpansionVisitor", n)

    def visit_lowered_einstein_clause(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("ExpansionVisitor", n)

    def visit_lowered_einstein(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("ExpansionVisitor", n)

    def visit_lowered_recurrence(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("ExpansionVisitor", n)

    def visit_literal_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_identifier_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_wildcard_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_tuple_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_array_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_rest_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_guard_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_or_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_constructor_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_binding_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_range_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)


def _expand(
    expr: ExpressionIR,
    D: Dict[DefId, IdentifierIR],
    SB: Dict[DefId, Any],
    SE: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    R: Any = None,
    P: Optional[ProgramIR] = None,
) -> ExpressionIR:
    """Expand DifferentialIR → _@* ref;  @num/@den → derivative quotient."""
    return expr.accept(_ExpansionVisitor(D, SB, SE, loc, R, P))


_expand_derivative_in_expr = _expand


def _expand_program(program: ProgramIR, D: Dict[DefId, IdentifierIR], loc: SourceLocation,
                    R: Any = None, init_B: Optional[Dict[DefId, Any]] = None,
                    target_binding_defids: Optional[Set[DefId]] = None,
                    target_statement_ids: Optional[Set[int]] = None) -> None:
    sb: Dict[DefId, Any] = dict(init_B) if init_B else {}
    se: Dict[DefId, ExpressionIR] = {d: b.expr for d, b in sb.items() if getattr(b, "expr", None) is not None}
    expand_binding_defids = target_binding_defids
    expand_statement_ids = target_statement_ids
    for b in program.bindings or []:
        if not isinstance(b, BindingIR) or b.expr is None: continue
        if expand_binding_defids is None or b.defid in expand_binding_defids:
            b.expr = _expand(b.expr, D, sb, se, b.expr.location or loc, R, program)
        if b.defid is not None: sb[b.defid] = b; se[b.defid] = b.expr
    stmts = program.statements or []
    for i, s in enumerate(stmts):
        if (not isinstance(s, BindingIR) and isinstance(s, ExpressionIR)
                and (expand_statement_ids is None or id(s) in expand_statement_ids)):
            stmts[i] = _expand(s, D, sb, se, getattr(s, "location", None) or loc, R, program)

_expand_derivative_nodes_in_program = _expand_program


def _ensure_block_d(block: BlockExpressionIR, SB: Dict[DefId, Any], SE: Dict[DefId, ExpressionIR],
                    D: Dict[DefId, IdentifierIR], loc: SourceLocation, R: Any) -> None:
    tgts, qps = _collect_targets_expr(block)
    if not tgts and not qps: return
    bb = _bindings_in(block, None)
    if not bb: return
    bd = {b.defid for b in bb if b.defid is not None}
    td: Set[DefId] = set()
    for did, _ in tgts: td.add(did)
    for n, d_ in qps: td.add(n); td.add(d_)
    td &= bd
    if not td: return

    bbd: Dict[DefId, BindingIR] = dict(SB)
    for b in bb:
        if b.defid is not None: bbd[b.defid] = b
    dep_cache = _DependencyQueryCache(bbd)
    b2d: Dict[DefId, Set[DefId]] = {}
    for b in bb:
        if b.defid is not None and b.expr is not None:
            if is_function_binding(b):
                b2d[b.defid] = set()
            else:
                b2d[b.defid] = _autodiff_primal_data_defids(b.expr, bbd, dep_cache)

    reach: Set[DefId] = set(td); wk = list(reach)
    while wk:
        did = wk.pop()
        for dep in b2d.get(did) or []:
            if dep in bd and dep not in reach: reach.add(dep); wk.append(dep)

    fwd: List[BindingIR] = []; seen: Set[DefId] = set()
    def vis(did: DefId) -> None:
        if did in seen or did not in reach: return
        seen.add(did); b = bbd.get(did)
        if b is None: return
        for dep in b2d.get(b.defid) or []:
            if dep in bd: vis(dep)
        fwd.append(b)
    for did in td: vis(did)
    if R is None: return

    qd = {d_ for _, d_ in qps}; lvs = {did for did in reach if not (b2d.get(did) or set())}
    sv: Dict[DefId, int] = {}
    for b in fwd:
        if b.defid is None: continue
        if b.defid in qd: sv[b.defid] = 1
        elif b.defid in lvs and b.defid in td: sv[b.defid] = 1
        else: sv[b.defid] = 0

    dre: Dict[DefId, ExpressionIR] = {}
    for did, ref in D.items(): dre[did] = ref
    upq = len(qps) > 0
    d2b: Dict[DefId, BindingIR] = {}
    for b in fwd:
        if b.defid is None or b.defid in D: continue
        bl = b.location or _LOC0
        if b.defid in sv and sv[b.defid] == 1: drhs = _fl(1, bl)
        elif upq and b.defid in lvs: drhs = _z(bl)
        else:
            for dep in b2d.get(b.defid) or []:
                if dep not in dre: dre[dep] = _z(bl)
            drhs = _fwd_expr(b, dre, bbd, b2d, bl, R)
            drhs = _inline_drhs(drhs, bl)
        ti = _ti(b) or (_ti(b.expr) if b.expr else None)
        si = _si(b) or (_si(b.expr) if b.expr else None)
        dd = R.allocate_for_local(); dn = USER_DIFF_PREFIX + (b.name or "")
        dr = IdentifierIR(dn, bl, dd, type_info=ti, shape_info=si)
        D[b.defid] = dr; dre[b.defid] = dr
        _propagate_ti(drhs, ti, si)
        d2b[b.defid] = BindingIR(name=dn, expr=drhs, location=b.location, defid=dd, type_info=ti)

    ns: List = []
    for s in block.statements or []:
        ns.append(s)
        if isinstance(s, BindingIR) and s.defid is not None and s.defid in d2b: ns.append(d2b[s.defid])
    object.__setattr__(block, "statements", ns)

# ═══════════════════════════════════════════════════════════════════════════
# Forward diff for top-level bindings  (Phase 2 core)
# ═══════════════════════════════════════════════════════════════════════════

def _inline_drhs(rhs: ExpressionIR, loc: SourceLocation) -> ExpressionIR:
    if isinstance(rhs, BlockExpressionIR) and rhs.final_expr is not None:
        stmts = [s for s in (rhs.statements or []) if isinstance(s, BindingIR)]
        if len(stmts) == 1 and stmts[0].expr is not None:
            if isinstance(rhs.final_expr, IdentifierIR) and rhs.final_expr.defid == stmts[0].defid:
                return stmts[0].expr
    return rhs

_inline_derivative_rhs_block = _inline_drhs

def _fwd_einstein(expr: EinsteinIR, dre: Dict[DefId, ExpressionIR],
                  B: Dict[DefId, BindingIR], loc: SourceLocation, R: Any) -> ExpressionIR:
    vis = DiffVisitor(dre, loc, B, R)
    nc: List[EinsteinClauseIR] = []
    for c in expr.clauses or []:
        try: dv = _unwrap_trivial_einstein_rhs(_simplify(c.value.accept(vis), loc))
        except (ValueError, KeyError): continue
        if _is_zero(dv): continue
        nc.append(EinsteinClauseIR(indices=list(c.indices or []), value=dv, location=c.location,
                  where_clause=c.where_clause, variable_ranges=dict(c.variable_ranges or {})))
    if not nc: return _z(loc)
    return EinsteinIR(clauses=nc, shape=expr.shape, element_type=expr.element_type,
                      location=expr.location, type_info=_ti(expr), shape_info=_si(expr))

def _fwd_expr(b: BindingIR, dre: Dict[DefId, ExpressionIR], B: Dict[DefId, BindingIR],
              b2d: Dict[DefId, Set[DefId]], loc: SourceLocation, R: Any) -> ExpressionIR:
    expr = b.expr
    if expr is None: return _z(loc)
    if isinstance(expr, FunctionValueIR): return _z(loc)
    if isinstance(expr, EinsteinIR): return _fwd_einstein(expr, dre, B, loc, R)
    vis = DiffVisitor(dre, loc, B, R)
    return expr.accept(vis)

_forward_d_y_expr = _fwd_expr

# ═══════════════════════════════════════════════════════════════════════════
# Pass entry point
# ═══════════════════════════════════════════════════════════════════════════

class AutodiffPass(BasePass):
    """Expand @expr and @y/@x into plain IR via forward-mode autodiff."""
    requires = [TypeInferencePass, UnifiedShapeAnalysisPass]

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        try:
            return self._core(ir, tcx)
        finally:
            from ..ir.nodes import clear_autodiff_only_fields
            clear_autodiff_only_fields(ir)

    def _core(self, program: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        bindings = _bindings_in(program, program) or []
        if not bindings:
            tcx.set_analysis(AutodiffPass, {"diff_block": None, "differential_targets": set(), "differential_buffer_by_defid": {}})
            return program

        # 1. Collect targets
        diff_targets: List[Tuple[DefId, str]] = []
        q_pairs: List[Tuple[DefId, DefId]] = []
        target_binding_defids: Set[DefId] = set()
        target_statement_ids: Set[int] = set()
        for b in bindings:
            if b.expr is None:
                continue
            bt, bq = _collect_targets_expr(b.expr)
            if bt or bq:
                if b.defid is not None:
                    target_binding_defids.add(b.defid)
                diff_targets.extend(bt)
                q_pairs.extend(bq)
        for s in program.statements or []:
            if isinstance(s, BindingIR) or not isinstance(s, ExpressionIR):
                continue
            st, sq = _collect_targets_expr(s)
            if st or sq:
                target_statement_ids.add(id(s))
                diff_targets.extend(st)
                q_pairs.extend(sq)

        # 2. Binding map
        B: Dict[DefId, BindingIR] = {}
        for b in bindings:
            if b.defid is not None: B[b.defid] = b
        fim = getattr(tcx, "function_ir_map", None) or {}
        for did, fn in fim.items():
            if did is not None and did not in B and isinstance(fn, BindingIR) and is_function_binding(fn):
                B[did] = fn

        # 3. Dep graph (top-level bindings only)
        # Function definitions (expr is FunctionValueIR) must not contribute inner DefIds: walking the
        # body pulls in callee-local bindings (e.g. recurrence tensor u inside euler_decay). Those
        # DefIds are not keys in B, so they become false "leaves" in lvs, pollute reach, and can
        # break runtime (seeds / tangents keyed by inner DefIds) even for unrelated calls like
        # let u = euler_decay(...) before any @-quotient.  Call-site autodiff still sees the body
        # via DiffVisitor / JacobianVisitor on FunctionCallIR.
        dep_cache = _DependencyQueryCache(B)
        b2d: Dict[DefId, Set[DefId]] = {}

        def _binding_deps(did: DefId) -> Set[DefId]:
            cached = b2d.get(did)
            if cached is not None:
                return cached
            b = B.get(did)
            if b is None or is_function_binding(b):
                out: Set[DefId] = set()
            else:
                out = _autodiff_primal_data_defids(b.expr, B, dep_cache)
            b2d[did] = out
            return out

        # 4. Reachable set
        td: Set[DefId] = set()
        for did, _ in diff_targets: td.add(did)
        for n, d in q_pairs: td.add(n); td.add(d)
        top = {b.defid for b in bindings if b.defid is not None}
        td_top = td & top
        reach: Set[DefId] = set(); wk = list(td_top)
        while wk:
            did = wk.pop()
            if did in reach: continue
            reach.add(did)
            b = B.get(did)
            if b is None: continue
            for dep in _binding_deps(did):
                if dep not in reach: wk.append(dep)

        # 5. Topo sort
        fwd: List[BindingIR] = []; seen: Set[DefId] = set()
        def _vis(did: DefId) -> None:
            if did in seen or did not in reach: return
            seen.add(did); b = B.get(did)
            if b is None: return
            for dep in _binding_deps(did): _vis(dep)
            fwd.append(b)
        for did in td_top: _vis(did)

        R = getattr(tcx, "resolver", None)
        if R is None:
            tcx.set_analysis(AutodiffPass, {"diff_block": None, "differential_targets": set(diff_targets), "differential_buffer_by_defid": {}})
            return program

        gti = gsi = None
        for b in bindings:
            if b.expr is not None and not isinstance(b.expr, FunctionValueIR):
                gti = _ti(b) or _ti(b.expr)
                gsi = _si(b) or _si(b.expr)
                if gti is not None:
                    break

        # 6. Create _@* identifiers + seeds
        D: Dict[DefId, IdentifierIR] = {}
        d2b: Dict[DefId, BindingIR] = {}
        sv: Dict[DefId, int] = {}
        qd = {d for _, d in q_pairs}
        lvs = {did for did in reach if not (b2d.get(did) or set())}
        for b in fwd:
            if b.defid is None or b.defid not in reach: continue
            dn = USER_DIFF_PREFIX + (b.name or "")
            dd = R.allocate_for_local()
            ti0 = _ti(b) or (_ti(b.expr) if b.expr else None) or gti
            si0 = _si(b) or (_si(b.expr) if b.expr else None) or gsi
            D[b.defid] = IdentifierIR(
                dn, b.location or _LOC0, dd, type_info=ti0, shape_info=si0
            )
            if b.defid in qd: sv[b.defid] = 1
            elif b.defid in lvs and (b.defid in td or not q_pairs): sv[b.defid] = 1
            else: sv[b.defid] = 0

        # 7. Build _@* RHS
        dre: Dict[DefId, ExpressionIR] = {}
        for did, ref in D.items(): dre[did] = ref
        drhs_map: Dict[DefId, ExpressionIR] = {}
        upq = len(q_pairs) > 0
        for b in fwd:
            if b.defid is None or b.defid not in reach: continue
            bl = b.location or _LOC0
            if b.defid in sv and sv[b.defid] == 1: drhs_map[b.defid] = _fl(1, bl)
            elif upq and b.defid in lvs: drhs_map[b.defid] = _z(bl)
            else:
                for dep in b2d.get(b.defid) or []:
                    if dep not in dre: dre[dep] = _z(bl)
                rhs = _fwd_expr(b, dre, B, b2d, bl, R)
                drhs_map[b.defid] = _inline_drhs(rhs if rhs is not None else _z(bl), bl)

        # 8. Create bindings with type info
        for b in fwd:
            if b.defid is None or b.defid not in reach: continue
            rhs = drhs_map.get(b.defid) or _z(b.location or _LOC0)
            ti = _ti(b) or (_ti(b.expr) if b.expr else None) or gti
            si = _si(b) or (_si(b.expr) if b.expr else None) or gsi
            _propagate_ti(rhs, ti, si)
            ref = D[b.defid]
            d2b[b.defid] = BindingIR(name=ref.name, expr=rhs, location=b.location, defid=ref.defid, type_info=ti)

        # 9. Insert after primals
        nb: List[BindingIR] = []
        for b in bindings:
            nb.append(b)
            db = d2b.get(b.defid)
            if db is not None: nb.append(db)
        program.bindings = nb
        non_b = [s for s in (program.statements or []) if not isinstance(s, BindingIR)]
        program.statements = nb + non_b

        # 10. Expand
        _expand_program(
            program,
            D,
            _LOC0,
            R,
            init_B=B,
            target_binding_defids=target_binding_defids,
            target_statement_ids=target_statement_ids,
        )

        # 11. Analysis
        dbl = [d2b[b.defid] for b in fwd if b.defid in d2b]
        adm: Dict[DefId, DefId] = {p: d2b[p].defid for p in d2b}
        tcx.set_analysis(AutodiffPass, {
            "diff_block": dbl or None,
            "differential_targets": set(diff_targets),
            "differential_buffer_by_defid": {},
            "autodiff_differential_map": adm,
            "differential_leaves": lvs,
        })
        return program
