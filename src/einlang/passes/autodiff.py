"""Autodiff pass — expand ``@expr`` and ``@y/@x`` into plain IR.

Design doc: ``docs/autodiff_design.md``

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

def _ti(n: Any) -> Any:
    return getattr(n, "type_info", None)

def _si(n: Any) -> Any:
    return getattr(n, "shape_info", None)

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
            sv = _simplify(c.value, c.location or loc)
            nc.append(EinsteinClauseIR(indices=c.indices, value=sv, location=c.location,
                      where_clause=c.where_clause, variable_ranges=c.variable_ranges))
        return EinsteinIR(clauses=nc, shape=expr.shape, element_type=expr.element_type,
                          location=expr.location, type_info=_ti(expr), shape_info=_si(expr))

    if isinstance(expr, ReductionExpressionIR):
        sb = _simplify(expr.body, loc)
        return ReductionExpressionIR(expr.operation, expr.loop_vars, sb, expr.location,
               where_clause=expr.where_clause, loop_var_ranges=expr.loop_var_ranges,
               type_info=_ti(expr), shape_info=_si(expr))

    return expr

# ═══════════════════════════════════════════════════════════════════════════
# log() lookup for power chain rule
# ═══════════════════════════════════════════════════════════════════════════

def _log_call(arg: ExpressionIR, bindings: Dict[DefId, Any], resolver: Any, loc: SourceLocation) -> ExpressionIR:
    for did, b in bindings.items():
        if not isinstance(b, BindingIR) or not isinstance(b.expr, FunctionValueIR):
            continue
        nm = b.name or ""
        if nm == "log" or nm.endswith("::log"):
            callee = IdentifierIR(nm, loc, did, type_info=b.type_info)
            return FunctionCallIR(callee_expr=callee, location=loc, arguments=[arg], type_info=F32)
    return FunctionCallIR(callee_expr=IdentifierIR("log", loc, None), location=loc, arguments=[arg], type_info=F32)


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
            nc.append(
                EinsteinClauseIR(
                    indices=[i.accept(self) for i in (c.indices or [])],
                    value=c.value.accept(self) if c.value is not None else None,
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

    def visit_identifier(self, n: IdentifierIR) -> ExpressionIR:
        if n.defid is not None and n.defid in self._m:
            return self._m[n.defid]
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
            return self._dm[op.defid]
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


def _collect_defids(expr: Optional[ExpressionIR]) -> Set[DefId]:
    if expr is None: return set()
    c = _DefIdCollector(); expr.accept(c); return c.defids


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
                 primal_subst: Optional[Dict[DefId, ExpressionIR]] = None) -> None:
        self._wrt = wrt; self._loc = loc; self._B = bindings
        self._R = resolver; self._sp = stmt_partial; self._wt = wrt_tangent
        self._ps = primal_subst

    # -- atoms ---------------------------------------------------------------
    def visit_identifier(self, n: IdentifierIR) -> ExpressionIR:
        if n.defid == self._wrt:
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
                return b.expr.accept(self)
        return _z(self._loc)

    def visit_literal(self, n: LiteralIR) -> ExpressionIR: return _z(self._loc)
    def visit_array_literal(self, n: ArrayLiteralIR) -> ExpressionIR: return _z(self._loc)

    # -- binary / unary ------------------------------------------------------
    def visit_binary_op(self, n: BinaryOpIR) -> ExpressionIR:
        L = n.left; R = n.right; loc = n.location or self._loc
        dL = L.accept(self); dR = R.accept(self)
        op = n.operator
        if op == BinaryOp.ADD: return BinaryOpIR(BinaryOp.ADD, dL, dR, loc)
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
        raise ValueError(f"Autodiff: unsupported binary op: {op}")

    def visit_unary_op(self, n: UnaryOpIR) -> ExpressionIR:
        d = n.operand.accept(self)
        if n.operator == UnaryOp.NEG: return UnaryOpIR(UnaryOp.NEG, d, n.location or self._loc)
        if n.operator == UnaryOp.POS: return d
        raise ValueError(f"Autodiff: unsupported unary op: {n.operator}")

    # -- reductions ----------------------------------------------------------
    def visit_reduction_expression(self, n: ReductionExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc; d_body = n.body.accept(self); op = n.operation
        if op == ReductionOp.SUM:
            return ReductionExpressionIR(ReductionOp.SUM, n.loop_vars, d_body, loc,
                   where_clause=n.where_clause, loop_var_ranges=n.loop_var_ranges,
                   type_info=_ti(n), shape_info=_si(n))
        if op == ReductionOp.MAX:
            return SelectAtArgmaxIR(n.body, d_body, n.loop_vars, loop_var_ranges=n.loop_var_ranges,
                   location=loc, type_info=_ti(n), shape_info=_si(n))
        if op == ReductionOp.MIN:
            return SelectAtArgmaxIR(n.body, d_body, n.loop_vars, loop_var_ranges=n.loop_var_ranges,
                   location=loc, type_info=_ti(n), shape_info=_si(n), use_argmin=True)
        if op == ReductionOp.PROD:
            return BinaryOpIR(BinaryOp.MUL,
                              BinaryOpIR(BinaryOp.DIV, n, n.body, loc), d_body, loc)
        raise ValueError(f"Autodiff: unsupported reduction: {op}")

    # -- indexing / cast / control flow --------------------------------------
    def visit_rectangular_access(self, n: RectangularAccessIR) -> ExpressionIR:
        return RectangularAccessIR(n.array.accept(self), list(n.indices or []),
                                   n.location or self._loc, type_info=_ti(n), shape_info=_si(n))
    def visit_cast_expression(self, n: CastExpressionIR) -> ExpressionIR:
        return n.expr.accept(self)
    def visit_if_expression(self, n: IfExpressionIR) -> ExpressionIR:
        dt = n.then_expr.accept(self)
        de = n.else_expr.accept(self) if n.else_expr is not None else _z(self._loc)
        return IfExpressionIR(condition=n.condition, then_expr=dt, location=n.location or self._loc,
                              else_expr=de, type_info=_ti(n), shape_info=_si(n))
    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        if n.final_expr is None:
            raise ValueError("Autodiff: JacobianVisitor block has no final expression")
        return n.final_expr.accept(self)

    # -- function call -------------------------------------------------------
    def visit_function_call(self, n: FunctionCallIR) -> ExpressionIR:
        loc = n.location or self._loc; args = n.arguments or []
        cdid = n.function_defid
        if cdid is None or cdid not in self._B:
            raise ValueError("Autodiff: JacobianVisitor callee is unresolved or not in binding map")
        b = self._B[cdid]
        if not isinstance(b.expr, FunctionValueIR):
            raise ValueError("Autodiff: JacobianVisitor callee is not a function value")
        fv = b.expr; ps = fv.parameters or []; body = fv.body
        rm = {p.defid: args[j] for j, p in enumerate(ps) if p.defid is not None and j < len(args)}

        rule_body = getattr(fv, "custom_diff_body", None)
        if rule_body is not None and len(ps) == len(args):
            if len(ps) == 1 and ps[0].defid is not None:
                return _sub_callee(_sub_wd(rule_body, rm, {ps[0].defid: args[0].accept(self)}, loc), fv, rm, loc)
            terms: List[ExpressionIR] = []
            for i, p in enumerate(ps):
                if p.defid is None: continue
                ud = {ps[j].defid: (_fl(1, loc) if j == i else _z(loc)) for j in range(len(ps)) if ps[j].defid is not None}
                coef = _simplify(_sub_callee(_sub_wd(rule_body, rm, ud, loc), fv, rm, loc), loc)
                terms.append(BinaryOpIR(BinaryOp.MUL, coef, args[i].accept(self), loc))
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
            iv = JacobianVisitor(p.defid, loc, self._B, self._R)
            terms.append(BinaryOpIR(BinaryOp.MUL, _sub(body.accept(iv), rm, loc), args[i].accept(self), loc))
        return _sum_terms(terms, loc)

    # -- einstein (index expansion for jacobians) ----------------------------
    def visit_einstein(self, n: EinsteinIR) -> ExpressionIR:
        return _diff_einstein_wrt(n, self._wrt, self._loc, self._B, self._R, self._wt,
                                  sp=self._sp, ps=self._ps)
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
    def visit_tuple_expression(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_tuple_access(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_interpolated_string(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
    def visit_member_access(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("JacobianVisitor", n)
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
                       ps: Optional[Dict[DefId, ExpressionIR]] = None) -> ExpressionIR:
    dc: List[EinsteinClauseIR] = []
    for clause in expr.clauses or []:
        val = clause.value
        if not isinstance(val, ReductionExpressionIR):
            d_val = val.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps))
            dc.append(EinsteinClauseIR(indices=clause.indices, value=d_val, location=clause.location,
                      where_clause=clause.where_clause, variable_ranges=dict(clause.variable_ranges or {})))
            continue
        inner = val.body; factors = _flatten_product(inner) if inner else None
        if not factors:
            if val.operation == ReductionOp.SUM:
                d_inner = inner.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps))
                dc.append(EinsteinClauseIR(indices=list(clause.indices or []),
                    value=ReductionExpressionIR(ReductionOp.SUM, list(val.loop_vars or []), d_inner, loc,
                          where_clause=val.where_clause, loop_var_ranges=_merged_lr(val, clause),
                          type_info=_ti(val), shape_info=_si(val)),
                    location=clause.location, where_clause=clause.where_clause,
                    variable_ranges=dict(clause.variable_ranges or {})))
            else:
                d_val = val.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps))
                dc.append(EinsteinClauseIR(indices=clause.indices, value=d_val, location=clause.location,
                          where_clause=clause.where_clause, variable_ranges=dict(clause.variable_ranges or {})))
            continue
        wrt_pos = [i for i, (a, _) in enumerate(factors) if isinstance(a, IdentifierIR) and a.defid == wrt]
        if not wrt_pos:
            if val.operation == ReductionOp.SUM:
                d_inner = inner.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps))
                dc.append(EinsteinClauseIR(indices=list(clause.indices or []),
                    value=ReductionExpressionIR(ReductionOp.SUM, list(val.loop_vars or []), d_inner, loc,
                          where_clause=val.where_clause, loop_var_ranges=_merged_lr(val, clause),
                          type_info=_ti(val), shape_info=_si(val)),
                    location=clause.location, where_clause=clause.where_clause,
                    variable_ranges=dict(clause.variable_ranges or {})))
            continue
        if val.operation == ReductionOp.SUM and len(wrt_pos) == 1 and len(factors) == 1:
            d_inner = inner.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps))
            dc.append(EinsteinClauseIR(indices=list(clause.indices or []),
                value=ReductionExpressionIR(ReductionOp.SUM, list(val.loop_vars or []), d_inner, loc,
                      where_clause=val.where_clause, loop_var_ranges=_merged_lr(val, clause),
                      type_info=_ti(val), shape_info=_si(val)),
                location=clause.location, where_clause=clause.where_clause,
                variable_ranges=dict(clause.variable_ranges or {})))
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
            dc.append(EinsteinClauseIR(indices=dvars, value=sel, location=clause.location,
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
            d_val = val.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps))
            dc.append(EinsteinClauseIR(indices=clause.indices, value=d_val, location=clause.location,
                      where_clause=clause.where_clause, variable_ranges=dict(clause.variable_ranges or {})))
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
                    ref = IdentifierIR(nm, loc, a.defid) if isinstance(a, IdentifierIR) else a
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
        pd = R.allocate_for_local()
        pn = s.name or "_p"
        pr = IdentifierIR(pn, s.location or loc, pd, type_info=_ti(s))
        primal_map[s.defid] = pr
        primal_stmts.append(
            BindingIR(
                name=pn,
                expr=subst_expr,
                location=s.location or loc,
                defid=pd,
                type_info=_ti(s),
            )
        )
    return primal_stmts, primal_map


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
    sp: Dict[DefId, ExpressionIR] = {}
    # Quotient Jacobians use wrt = a call-site DefId (denominator of @y/@x); parameters are
    # keys of ``param_subst``.  Forward-mode uses wrt = callee parameter DefId ∈ param_subst.
    # primal_subst must only apply to the quotient case (see tests using std::ml).
    ps_map = primal_map if wrt not in param_subst else None
    vis = JacobianVisitor(
        wrt, loc, B, R, stmt_partial=sp, wrt_tangent=wt, primal_subst=ps_map
    )
    diff_stmts: List[BindingIR] = []
    for s in block.statements or []:
        if not isinstance(s, BindingIR) or s.defid is None or s.expr is None:
            continue
        pv = _simplify(_sub(s.expr.accept(vis), primal_map, loc), loc)
        if _is_zero(pv):
            sp[s.defid] = _z(loc)
        else:
            dd = R.allocate_for_local()
            dn = DIFF_PREFIX + (s.name or "")
            dr = IdentifierIR(dn, s.location or loc, dd, type_info=_ti(s))
            sp[s.defid] = dr
            diff_stmts.append(
                BindingIR(
                    name=dn,
                    expr=pv,
                    location=s.location or loc,
                    defid=dd,
                    type_info=_ti(s),
                )
            )
    fp = _simplify(_sub(block.final_expr.accept(vis), primal_map, loc), loc)
    return diff_stmts, fp


def _callee_args_all_identifier_expressions(
    rm: Dict[DefId, ExpressionIR], ps: List[Any]
) -> bool:
    """True if every parameter is bound to a bare ``IdentifierIR`` (typical call sites)."""
    for p in ps:
        if p.defid is None:
            return False
        arg = rm.get(p.defid)
        if not isinstance(arg, IdentifierIR) or arg.defid is None:
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

    Requires every argument in ``rm`` to be a bare identifier so tangents attach to call-site
    ``DefId``s. Non-identifier arguments fall back to per-parameter ``JacobianVisitor`` sweeps.
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
        if not isinstance(arg, IdentifierIR) or arg.defid is None:
            raise ValueError("Autodiff: combined callee forward requires identifier arguments")
        dre[arg.defid] = da

    diff_stmts: List[BindingIR] = []
    bl = block.location or loc
    for s in block.statements or []:
        if not isinstance(s, BindingIR) or s.defid is None or s.expr is None:
            continue
        e_sub = _sub(s.expr, primal_map, loc)
        pv = _simplify(e_sub.accept(DiffVisitor(dre, loc, B, R)), loc)
        prim = primal_map.get(s.defid)
        if not isinstance(prim, IdentifierIR) or prim.defid is None:
            raise ValueError("Autodiff: primal_map missing identifier for callee local")
        pd = prim.defid
        if _is_zero(pv):
            dre[pd] = _z(bl)
            continue
        dd = R.allocate_for_local()
        dn = DIFF_PREFIX + (s.name or "")
        dr = IdentifierIR(dn, s.location or bl, dd, type_info=_ti(s))
        dre[pd] = dr
        diff_stmts.append(
            BindingIR(
                name=dn,
                expr=pv,
                location=s.location or bl,
                defid=dd,
                type_info=_ti(s),
            )
        )
    fe_sub = _sub(block.final_expr, primal_map, loc)
    fp = _simplify(fe_sub.accept(DiffVisitor(dre, loc, B, R)), loc)
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

def _sub_callee(expr: ExpressionIR, fv: FunctionValueIR,
                rm: Dict[DefId, ExpressionIR], loc: SourceLocation) -> ExpressionIR:
    cm: Dict[DefId, ExpressionIR] = dict(rm)
    body = fv.body
    if isinstance(body, BlockExpressionIR):
        for s in body.statements or []:
            if isinstance(s, BindingIR) and s.defid is not None and s.expr is not None:
                cm[s.defid] = _sub(s.expr, cm, loc)
    return _sub(expr, cm, loc)

def _sum_terms(terms: List[ExpressionIR], loc: SourceLocation) -> ExpressionIR:
    if not terms: return _z(loc)
    out = terms[0]
    for t in terms[1:]: out = BinaryOpIR(BinaryOp.ADD, out, t, loc)
    return out


def _callee_primal_subst_map(fv: FunctionValueIR, args: List[ExpressionIR]) -> Dict[DefId, ExpressionIR]:
    """Map each parameter defid to the corresponding call-site argument expression."""
    ps = fv.parameters or []
    return {p.defid: args[j] for j, p in enumerate(ps) if p.defid is not None and j < len(args)}


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
    ``DifferentialIR(IdentifierIR(param))``. When there is no custom rule and every argument
    is a bare identifier, one ``DiffVisitor`` sweep (``_diff_callee_block_combined_forward``)
    emits a single ``_@v`` per callee local. Otherwise per-parameter ``JacobianVisitor`` sweeps
    are summed (``_diff_callee_block_tangent`` in a loop).
    """
    ps = fv.parameters or []
    rule_body = getattr(fv, "custom_diff_body", None)
    if rule_body is not None and len(ps) == len(args):
        rm = _callee_primal_subst_map(fv, args)
        return _sub_callee(_sub_wd(rule_body, rm, tangent_by_param, loc), fv, rm, loc)

    body = fv.body
    if body is None:
        raise ValueError("Autodiff: user function has no body")
    if len(ps) != len(args):
        raise ValueError("Autodiff: arity mismatch")
    rm = _callee_primal_subst_map(fv, args)
    if isinstance(body, BlockExpressionIR) and R is not None:
        primal_stmts, primal_map = _callee_block_build_primal(body, rm, loc, R)
        if _callee_args_all_identifier_expressions(rm, ps):
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
        iv = JacobianVisitor(p.defid, loc, B, R, wrt_tangent=da)
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
                 resolver: Any = None, pretty: bool = False) -> None:
        self._d = d_map; self._loc = loc
        self._B: Dict[DefId, Any] = dict(bindings) if bindings else {}
        self._R = resolver; self._pretty = pretty

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
            return BinaryOpIR(BinaryOp.MUL,
                              BinaryOpIR(BinaryOp.DIV, n, n.body, loc), db, loc)
        raise ValueError(f"Autodiff: unsupported reduction: {op}")

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        if n.final_expr is None:
            raise ValueError("Autodiff: DiffVisitor block has no final expression")
        inl = _inline_block_lets(n)
        if inl is n:
            return n.final_expr.accept(self)
        return inl.accept(self)

    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> ExpressionIR: return n

    def visit_function_call(self, n: FunctionCallIR) -> ExpressionIR:
        loc = n.location or self._loc
        args = n.arguments or []
        cdid = n.function_defid
        if cdid is None or cdid not in self._B:
            raise ValueError("Autodiff: callee must be a resolved function")
        binding = self._B[cdid]
        if not isinstance(binding.expr, FunctionValueIR):
            raise ValueError("Autodiff: callee is not a function value")
        fv = binding.expr
        ps = fv.parameters or []
        tangent_by_param = {
            p.defid: args[i].accept(self) for i, p in enumerate(ps) if p.defid is not None
        }
        tang = _callee_forward_jvp(fv, args, tangent_by_param, loc, self._B, self._R)
        if self._pretty:
            return _wrap_tangent_binding(
                tang, cdid, fv, ps, list(args), n.callee_expr,
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
        return RectangularAccessIR(da, indices, loc, type_info=_ti(n), shape_info=_si(n))

    def visit_if_expression(self, n: IfExpressionIR) -> ExpressionIR:
        dt = n.then_expr.accept(self)
        de = n.else_expr.accept(self) if n.else_expr is not None else _z(self._loc)
        return IfExpressionIR(condition=n.condition, then_expr=dt, location=n.location or self._loc,
                              else_expr=de, type_info=_ti(n), shape_info=_si(n))

    def visit_cast_expression(self, n: CastExpressionIR) -> ExpressionIR:
        return n.expr.accept(self)

    def visit_einstein(self, n: EinsteinIR) -> ExpressionIR:
        nc: List[EinsteinClauseIR] = []
        for c in n.clauses or []:
            try: dv = _simplify(c.value.accept(self), c.location or self._loc)
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
    def visit_tuple_expression(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_tuple_access(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_interpolated_string(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
    def visit_member_access(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)
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

def _trans_deps(expr: Optional[ExpressionIR], B: Dict[DefId, BindingIR]) -> Set[DefId]:
    out: Set[DefId] = set()
    vis: Set[DefId] = set()
    pending: List[DefId] = []
    for d in _collect_defids(expr):
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
        for d2 in _collect_defids(b.expr):
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
    ) -> None:
        super().__init__(loc)
        self._D = D
        self._SB = SB
        self._SE = SE
        self._R = R
        self._P = P

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
                cdn = dd in _collect_defids(ne)
                rch = (isinstance(nop, IdentifierIR) and nop.defid is not None
                       and _is_reachable(nop.defid, dd, self._SB))
                if not cdn and not rch:
                    return _z(ql)
                jv = JacobianVisitor(dd, ql, self._SB, self._R)
                der = _simplify(ne.accept(jv), ql)
                ti = _ti(n)
                si = _si(n)
                if ti or si:
                    _propagate_ti(der, ti, si)
                return der
            dids = _collect_defids(dop)
            if len(dids) != 1:
                raise ValueError("Autodiff: @num/@(expr) denominator depends on != 1 variable")
            wd = next(iter(dids))
            dn = nop.accept(JacobianVisitor(wd, self._loc, self._SB, self._R))
            dd_ = dop.accept(JacobianVisitor(wd, self._loc, self._SB, self._R))
            der = BinaryOpIR(BinaryOp.DIV, dn, dd_, self._loc)
            ti = _ti(n)
            si = _si(n)
            if ti or si:
                _propagate_ti(der, ti, si)
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
        ns: List[Any] = []
        for s in blk.statements or []:
            if isinstance(s, BindingIR):
                v = _ExpansionVisitor(self._D, nsb, nse, self._loc, self._R, self._P)
                ex = s.expr.accept(v) if s.expr is not None else None
                nb = BindingIR(name=s.name, expr=ex, location=s.location, defid=s.defid, type_info=_ti(s))
                if nb.defid is not None:
                    nsb[nb.defid] = nb
                    nse[nb.defid] = ex
                ns.append(nb)
            elif isinstance(s, ExpressionIR):
                v = _ExpansionVisitor(self._D, nsb, nse, self._loc, self._R, self._P)
                ns.append(s.accept(v))
            else:
                ns.append(s)
        vfin = _ExpansionVisitor(self._D, nsb, nse, self._loc, self._R, self._P)
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
                            needed = _trans_deps(ye, self._SB)
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
                                if not _collect_defids(pb.expr):
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
                        rhs = _str_ir(dr)
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
        return n

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
                    R: Any = None, init_B: Optional[Dict[DefId, Any]] = None) -> None:
    sb: Dict[DefId, Any] = dict(init_B) if init_B else {}
    se: Dict[DefId, ExpressionIR] = {d: b.expr for d, b in sb.items() if getattr(b, "expr", None) is not None}
    for b in program.bindings or []:
        if not isinstance(b, BindingIR) or b.expr is None: continue
        b.expr = _expand(b.expr, D, sb, se, b.expr.location or loc, R, program)
        if b.defid is not None: sb[b.defid] = b; se[b.defid] = b.expr
    stmts = program.statements or []
    for i, s in enumerate(stmts):
        if not isinstance(s, BindingIR) and isinstance(s, ExpressionIR):
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
    b2d: Dict[DefId, Set[DefId]] = {}
    for b in bb:
        if b.defid is not None and b.expr is not None:
            if is_function_binding(b):
                b2d[b.defid] = set()
            else:
                b2d[b.defid] = _collect_defids(b.expr)

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
        if upq and b.defid in lvs: drhs: ExpressionIR = _z(bl)
        elif b.defid in sv and sv[b.defid] == 1: drhs = _fl(1, bl)
        else:
            for dep in b2d.get(b.defid) or []:
                if dep not in dre: dre[dep] = _z(bl)
            drhs = _fwd_expr(b, dre, bbd, b2d, bl, R)
            drhs = _inline_drhs(drhs, bl)
        dd = R.allocate_for_local(); dn = USER_DIFF_PREFIX + (b.name or "")
        dr = IdentifierIR(dn, bl, dd); D[b.defid] = dr; dre[b.defid] = dr
        ti = _ti(b) or (_ti(b.expr) if b.expr else None)
        si = _si(b) or (_si(b.expr) if b.expr else None)
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
                  B: Dict[DefId, BindingIR], loc: SourceLocation) -> ExpressionIR:
    vis = DiffVisitor(dre, loc, B)
    nc: List[EinsteinClauseIR] = []
    for c in expr.clauses or []:
        try: dv = _simplify(c.value.accept(vis), loc)
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
    if isinstance(expr, EinsteinIR): return _fwd_einstein(expr, dre, B, loc)
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
        diff_targets, q_pairs = _collect_targets(program)

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
        b2d: Dict[DefId, Set[DefId]] = {}
        for b in bindings:
            if b.defid is not None:
                if is_function_binding(b):
                    b2d[b.defid] = set()
                else:
                    b2d[b.defid] = _collect_defids(b.expr)

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
            for dep in b2d.get(b.defid) or []:
                if dep not in reach: wk.append(dep)

        # 5. Topo sort
        fwd: List[BindingIR] = []; seen: Set[DefId] = set()
        def _vis(did: DefId) -> None:
            if did in seen or did not in reach: return
            seen.add(did); b = B.get(did)
            if b is None: return
            for dep in b2d.get(b.defid) or []: _vis(dep)
            fwd.append(b)
        for did in td_top: _vis(did)

        R = getattr(tcx, "resolver", None)
        if R is None:
            tcx.set_analysis(AutodiffPass, {"diff_block": None, "differential_targets": set(diff_targets), "differential_buffer_by_defid": {}})
            return program

        # 6. Create _@* identifiers + seeds
        D: Dict[DefId, IdentifierIR] = {}
        d2b: Dict[DefId, BindingIR] = {}
        sv: Dict[DefId, int] = {}
        qd = {d for _, d in q_pairs}
        lvs = {did for did in reach if not (b2d.get(did) or set())}
        for b in fwd:
            if b.defid is None or b.defid not in reach: continue
            dn = USER_DIFF_PREFIX + (b.name or ""); dd = R.allocate_for_local()
            D[b.defid] = IdentifierIR(dn, b.location or _LOC0, dd)
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
        gti = gsi = None
        for b in bindings:
            if b.expr is not None and not isinstance(b.expr, FunctionValueIR):
                gti = _ti(b) or _ti(b.expr); gsi = _si(b) or _si(b.expr)
                if gti is not None: break
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
        _expand_program(program, D, _LOC0, R, init_B=B)

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
