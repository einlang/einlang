from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union, cast

from ...ir.nodes import (
    ArrayComprehensionIR,
    ArrayLiteralIR,
    BindingIR,
    BinaryOpIR,
    BlockExpressionIR,
    BuiltinCallIR,
    CastExpressionIR,
    DifferentialIR,
    EinsteinClauseIR,
    EinsteinIR,
    ExpressionIR,
    FunctionCallIR,
    FunctionValueIR,
    IRVisitor,
    IdentifierIR,
    IfExpressionIR,
    IndexRestIR,
    IndexVarIR,
    InterpolatedStringIR,
    JaggedAccessIR,
    LambdaIR,
    LiteralIR,
    MatchExpressionIR,
    MemberAccessIR,
    PipelineExpressionIR,
    ProgramIR,
    RangeIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    SelectAtArgmaxIR,
    TryExpressionIR,
    TupleAccessIR,
    TupleExpressionIR,
    UnaryOpIR,
    WhereClauseIR,
    WhereExpressionIR,
)
from ...shared.defid import DefId
from ...shared.source_location import SourceLocation
from ...shared.types import (
    BOOL,
    F32,
    UNKNOWN,
    BinaryOp,
    PrimitiveType,
    ReductionOp,
    UnaryOp,
)

DIFF_PREFIX = "_@"
USER_DIFF_PREFIX = "@"
_LOC0 = SourceLocation("", 0, 0)
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


@lru_cache(maxsize=None)
def _slot_names(cls: type) -> Tuple[str, ...]:
    out: List[str] = []
    for c in cls.__mro__:
        slots = getattr(c, "__slots__", ())
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


def _clone_ir_value(value: Any, memo: Dict[int, Any]) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    oid = id(value)
    cached = memo.get(oid)
    if cached is not None:
        return cached
    if isinstance(value, tuple):
        cloned = tuple(_clone_ir_value(v, memo) for v in value)
        memo[oid] = cloned
        return cloned
    if isinstance(value, list):
        cloned = [_clone_ir_value(v, memo) for v in value]
        memo[oid] = cloned
        return cloned
    if isinstance(value, dict):
        cloned = {
            _clone_ir_value(k, memo): _clone_ir_value(v, memo)
            for k, v in value.items()
        }
        memo[oid] = cloned
        return cloned
    if isinstance(value, IRVisitor):
        return value
    if hasattr(value, "__class__") and any(hasattr(c, "__slots__") for c in value.__class__.__mro__):
        cloned = value.__class__.__new__(value.__class__)
        memo[oid] = cloned
        for slot in _slot_names(value.__class__):
            current = getattr(value, slot, None)
            if slot in _SHARED_CLONE_SLOTS:
                setattr(cloned, slot, current)
            else:
                setattr(cloned, slot, _clone_ir_value(current, memo))
        return cloned
    return value


def _clone_ir_expr(expr: ExpressionIR) -> ExpressionIR:
    return cast(ExpressionIR, _clone_ir_value(expr, {}))


def _is_diff_name(name: str) -> bool:
    return name.startswith(DIFF_PREFIX) or name.startswith(USER_DIFF_PREFIX)


def _fl(v: float, loc: SourceLocation) -> LiteralIR:
    return LiteralIR(float(v), loc, type_info=F32)


def _z(loc: SourceLocation) -> LiteralIR:
    return _fl(0, loc)


def _unsupported_autodiff_ir(visitor: str, node: object) -> NoReturn:
    raise ValueError(
        f"Autodiff: {visitor} does not support IR node {type(node).__name__} "
        f"(do not substitute literal zero; add a visitor or lowering rule)"
    )


def _reject_lowered_ir(visitor: str, node: object) -> NoReturn:
    raise ValueError(
        f"Autodiff: {visitor} does not support {type(node).__name__}. "
        f"Run AutodiffPass before EinsteinLoweringPass; differentiate EinsteinIR only, not lowered-* IR."
    )


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
    return isinstance(target_type, PrimitiveType) and target_type.name in {
        "i8",
        "i32",
        "i64",
        "bool",
        "str",
        "range",
        "unit",
    }


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
        return (
            getattr(a, "function_defid", None) == getattr(b, "function_defid", None)
            and len(aa) == len(ba)
            and all(_seq(x, y) for x, y in zip(aa, ba))
        )
    if isinstance(a, BuiltinCallIR):
        aa, ba = a.args or [], b.args or []
        return (
            a.builtin_name == b.builtin_name
            and len(aa) == len(ba)
            and all(_seq(x, y) for x, y in zip(aa, ba))
        )
    if isinstance(a, CastExpressionIR):
        return getattr(a, "target_type", None) == getattr(b, "target_type", None) and _seq(a.expr, b.expr)
    return False


def _rc_index_lists_equivalent(
    outer: List[ExpressionIR], inner: List[ExpressionIR]
) -> bool:
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


_ir_structurally_equal = _seq


def _simplify(expr: ExpressionIR, loc: SourceLocation) -> ExpressionIR:
    if isinstance(expr, BinaryOpIR):
        L = _simplify(expr.left, loc)
        R = _simplify(expr.right, loc)
        op = expr.operator
        ti = _ti(expr)
        si = _si(expr)
        lz, rz = _is_zero(L), _is_zero(R)
        l1 = isinstance(L, LiteralIR) and L.value == 1
        r1 = isinstance(R, LiteralIR) and R.value == 1
        ll, rl = isinstance(L, LiteralIR), isinstance(R, LiteralIR)

        if op == BinaryOp.ADD:
            if lz:
                return R
            if rz:
                return L
            if isinstance(L, ReductionExpressionIR) and L.operation == ReductionOp.SUM and _is_zero(L.body):
                return R
            if isinstance(R, ReductionExpressionIR) and R.operation == ReductionOp.SUM and _is_zero(R.body):
                return L
            if ll and rl:
                return _fl(L.value + R.value, loc)
            if _seq(L, R):
                return BinaryOpIR(BinaryOp.MUL, _fl(2, loc), L, loc, type_info=ti, shape_info=si)
        elif op == BinaryOp.SUB:
            if rz:
                return L
            if ll and rl:
                return _fl(L.value - R.value, loc)
        elif op == BinaryOp.MUL:
            if lz or rz:
                return _z(loc)
            if l1:
                return R
            if r1:
                return L
            if ll and rl:
                return _fl(L.value * R.value, loc)
        elif op == BinaryOp.DIV:
            if lz:
                return _z(loc)
            if (
                isinstance(R, BinaryOpIR)
                and R.operator == BinaryOp.POW
                and isinstance(R.right, LiteralIR)
                and R.right.value == 2
            ):
                base = R.left
                if isinstance(L, BinaryOpIR) and L.operator == BinaryOp.MUL:
                    if _seq(L.left, base):
                        return BinaryOpIR(BinaryOp.DIV, L.right, base, loc, type_info=ti, shape_info=si)
                    if _seq(L.right, base):
                        return BinaryOpIR(BinaryOp.DIV, L.left, base, loc, type_info=ti, shape_info=si)
        elif op == BinaryOp.POW:
            if r1:
                return L

        return BinaryOpIR(op, L, R, loc, type_info=ti, shape_info=si)

    if isinstance(expr, UnaryOpIR):
        inner = _simplify(expr.operand, loc)
        if expr.operator == UnaryOp.NEG and isinstance(inner, UnaryOpIR) and inner.operator == UnaryOp.NEG:
            return inner.operand
        return UnaryOpIR(expr.operator, inner, expr.location or loc, type_info=_ti(expr), shape_info=_si(expr))

    if isinstance(expr, EinsteinIR):
        nc = []
        for c in expr.clauses or []:
            sv = _unwrap_trivial_einstein_rhs(_simplify(c.value, c.location or loc))
            nc.append(
                EinsteinClauseIR(
                    indices=c.indices,
                    value=sv,
                    location=c.location,
                    where_clause=c.where_clause,
                    variable_ranges=c.variable_ranges,
                )
            )
        return EinsteinIR(
            clauses=nc,
            shape=expr.shape,
            element_type=expr.element_type,
            location=expr.location,
            type_info=_ti(expr),
            shape_info=_si(expr),
        )

    if isinstance(expr, ReductionExpressionIR):
        sb = _simplify(expr.body, loc)
        return ReductionExpressionIR(
            expr.operation,
            expr.loop_vars,
            sb,
            expr.location,
            where_clause=expr.where_clause,
            loop_var_ranges=expr.loop_var_ranges,
            type_info=_ti(expr),
            shape_info=_si(expr),
        )

    if isinstance(expr, IfExpressionIR):
        nc = _simplify(expr.condition, loc)
        nt = _simplify(expr.then_expr, loc)
        ne = _simplify(expr.else_expr, loc) if expr.else_expr is not None else None
        return IfExpressionIR(nc, nt, expr.location or loc, else_expr=ne, type_info=_ti(expr), shape_info=_si(expr))

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
        return BlockExpressionIR(nss, expr.location or loc, nf, type_info=_ti(expr), shape_info=_si(expr))

    if isinstance(expr, CastExpressionIR):
        ie = _simplify(expr.expr, loc)
        return CastExpressionIR(ie, expr.target_type, expr.location or loc, type_info=_ti(expr), shape_info=_si(expr))

    if isinstance(expr, RectangularAccessIR):
        aa = _simplify(expr.array, loc)
        if _is_zero(aa):
            return _z(loc)
        nidx = [_simplify(i, loc) for i in (expr.indices or [])]
        return RectangularAccessIR(aa, nidx, expr.location or loc, type_info=_ti(expr), shape_info=_si(expr))

    return expr


def _unwrap_trivial_einstein_rhs(expr: ExpressionIR) -> ExpressionIR:
    while isinstance(expr, BlockExpressionIR) and expr.final_expr is not None:
        bindings = [
            s for s in (expr.statements or [])
            if isinstance(s, BindingIR) and s.defid is not None and s.expr is not None
        ]
        if bindings:
            break
        expr = expr.final_expr
    return expr


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
            return FunctionCallIR(callee_expr=callee, location=loc, arguments=[arg], type_info=out_ti, shape_info=out_si)
    return FunctionCallIR(
        callee_expr=IdentifierIR("log", loc, None, type_info=UNKNOWN),
        location=loc,
        arguments=[arg],
        type_info=out_ti,
        shape_info=out_si,
    )


def _pow_chain(
    node: BinaryOpIR,
    dL: ExpressionIR,
    dR: ExpressionIR,
    bindings: Dict[DefId, Any],
    resolver: Any,
    loc: SourceLocation,
) -> ExpressionIR:
    a, b = node.left, node.right
    ti = _ti(node) or F32
    si = _si(node)
    if isinstance(b, LiteralIR):
        n = b.value
        return BinaryOpIR(
            BinaryOp.MUL,
            BinaryOpIR(
                BinaryOp.MUL,
                _fl(n, loc),
                BinaryOpIR(BinaryOp.POW, a, _fl(n - 1, loc), loc, type_info=ti, shape_info=si),
                loc,
                type_info=ti,
                shape_info=si,
            ),
            dL,
            loc,
            type_info=ti,
            shape_info=si,
        )
    if isinstance(a, LiteralIR):
        c = a.value
        return BinaryOpIR(
            BinaryOp.MUL,
            BinaryOpIR(
                BinaryOp.MUL,
                node,
                _log_call(_fl(c, loc), bindings, resolver, loc),
                loc,
                type_info=ti,
                shape_info=si,
            ),
            dR,
            loc,
            type_info=ti,
            shape_info=si,
        )
    log_a = _log_call(a, bindings, resolver, loc)
    t1 = BinaryOpIR(
        BinaryOp.MUL,
        BinaryOpIR(BinaryOp.DIV, b, a, loc, type_info=ti, shape_info=si),
        dL,
        loc,
        type_info=ti,
        shape_info=si,
    )
    t2 = BinaryOpIR(BinaryOp.MUL, log_a, dR, loc, type_info=ti, shape_info=si)
    return BinaryOpIR(
        BinaryOp.MUL,
        node,
        BinaryOpIR(BinaryOp.ADD, t1, t2, loc, type_info=ti, shape_info=si),
        loc,
        type_info=ti,
        shape_info=si,
    )


class _Rewriter(IRVisitor[ExpressionIR]):
    def __init__(self, loc: SourceLocation) -> None:
        self._loc = loc

    @staticmethod
    def _same_items(old_items: Optional[List[Any]], new_items: List[Any]) -> bool:
        seq = list(old_items or [])
        return len(seq) == len(new_items) and all(a is b for a, b in zip(seq, new_items))

    @staticmethod
    def _same_mapping(old_map: Optional[Dict[Any, Any]], new_map: Optional[Dict[Any, Any]]) -> bool:
        if old_map is new_map:
            return True
        if not old_map and not new_map:
            return True
        if old_map is None or new_map is None or len(old_map) != len(new_map):
            return False
        return all(k in old_map and old_map[k] is v for k, v in new_map.items())

    def _rw_wc(self, wc: Any) -> Any:
        if wc is None:
            return None
        cc = getattr(wc, "constraints", None)
        if not cc:
            return wc
        nc = [c.accept(self) for c in cc]
        if self._same_items(cc, nc):
            return wc
        return WhereClauseIR(constraints=nc, location=self._loc)

    def _rw_lr(self, lr: Optional[Dict]) -> Optional[Dict]:
        if not lr:
            return lr
        out: Dict = {}
        changed = False
        for k, v in lr.items():
            if isinstance(v, RangeIR):
                ns = v.start.accept(self)
                ne = v.end.accept(self)
                if ns is v.start and ne is v.end:
                    out[k] = v
                else:
                    changed = True
                    out[k] = RangeIR(start=ns, end=ne, location=v.location or self._loc, type_info=v.type_info)
            elif isinstance(v, ExpressionIR):
                nv = v.accept(self)
                if nv is not v:
                    changed = True
                out[k] = nv
            else:
                out[k] = v
        return out if changed else lr

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
        nl = n.left.accept(self)
        nr = n.right.accept(self)
        if nl is n.left and nr is n.right:
            return n
        return BinaryOpIR(n.operator, nl, nr, loc, type_info=_ti(n), shape_info=_si(n))

    def visit_unary_op(self, n: UnaryOpIR) -> ExpressionIR:
        no = n.operand.accept(self)
        if no is n.operand:
            return n
        return UnaryOpIR(n.operator, no, n.location or self._loc, type_info=_ti(n), shape_info=_si(n))

    def visit_function_call(self, n: FunctionCallIR) -> ExpressionIR:
        na = [a.accept(self) for a in (n.arguments or [])]
        if self._same_items(n.arguments, na):
            return n
        return FunctionCallIR(
            callee_expr=n.callee_expr,
            location=n.location or self._loc,
            arguments=na,
            module_path=getattr(n, "module_path", None),
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_rectangular_access(self, n: RectangularAccessIR) -> ExpressionIR:
        na = n.array.accept(self)
        ni = [i.accept(self) for i in (n.indices or [])]
        if na is n.array and self._same_items(n.indices, ni):
            return n
        return RectangularAccessIR(na, ni, n.location or self._loc, type_info=_ti(n), shape_info=_si(n))

    def visit_jagged_access(self, n: JaggedAccessIR) -> ExpressionIR:
        return n

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        ns: List[Any] = []
        changed = False
        for s in n.statements or []:
            if isinstance(s, BindingIR) and s.expr is not None:
                nexpr = s.expr.accept(self)
                if nexpr is s.expr:
                    ns.append(s)
                    continue
                changed = True
                ns.append(BindingIR(name=s.name, expr=nexpr, location=s.location, defid=s.defid, type_info=_ti(s)))
            else:
                ns.append(s)
        nf = n.final_expr.accept(self) if n.final_expr is not None else None
        if nf is not n.final_expr:
            changed = True
        if not changed:
            return n
        return BlockExpressionIR(ns, loc, nf, type_info=_ti(n), shape_info=_si(n))

    def visit_if_expression(self, n: IfExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        nc = n.condition.accept(self)
        nt = n.then_expr.accept(self)
        ne = n.else_expr.accept(self) if n.else_expr is not None else None
        if nc is n.condition and nt is n.then_expr and ne is n.else_expr:
            return n
        return IfExpressionIR(condition=nc, then_expr=nt, location=loc, else_expr=ne, type_info=_ti(n), shape_info=_si(n))

    def visit_lambda(self, n: LambdaIR) -> ExpressionIR:
        return n

    def visit_differential(self, n: DifferentialIR) -> ExpressionIR:
        no = n.operand.accept(self)
        if no is n.operand:
            return n
        return DifferentialIR(no, n.location or self._loc, type_info=_ti(n), shape_info=_si(n))

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
        ne = n.expr.accept(self)
        if ne is n.expr:
            return n
        return CastExpressionIR(ne, n.target_type, n.location or self._loc, type_info=_ti(n), shape_info=_si(n))

    def visit_member_access(self, n: MemberAccessIR) -> ExpressionIR:
        no = n.object.accept(self)
        if no is n.object:
            return n
        return MemberAccessIR(no, n.member, n.location or self._loc, type_info=_ti(n), shape_info=_si(n))

    def visit_try_expression(self, n: TryExpressionIR) -> ExpressionIR:
        return n

    def visit_match_expression(self, n: MatchExpressionIR) -> ExpressionIR:
        return n

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> ExpressionIR:
        nb = n.body.accept(self)
        nwc = self._rw_wc(n.where_clause)
        nlr = self._rw_lr(n.loop_var_ranges)
        if nb is n.body and nwc is n.where_clause and nlr is n.loop_var_ranges:
            return n
        return ReductionExpressionIR(
            n.operation,
            n.loop_vars,
            nb,
            n.location or self._loc,
            where_clause=nwc,
            loop_var_ranges=nlr,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_builtin_call(self, n: BuiltinCallIR) -> ExpressionIR:
        na = [a.accept(self) for a in (n.args or [])]
        if self._same_items(n.args, na):
            return n
        return BuiltinCallIR(n.builtin_name, na, n.location or self._loc, defid=getattr(n, "defid", None), type_info=_ti(n), shape_info=_si(n))

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
        changed = False
        for c in n.clauses or []:
            cv = c.value.accept(self) if c.value is not None else None
            if cv is not None:
                cv = _unwrap_trivial_einstein_rhs(cv)
            ni = [i.accept(self) for i in (c.indices or [])]
            nwc = self._rw_wc(c.where_clause)
            nvr = self._rw_lr(c.variable_ranges)
            if cv is c.value and self._same_items(c.indices, ni) and nwc is c.where_clause and nvr is c.variable_ranges:
                nc.append(c)
            else:
                changed = True
                nc.append(
                    EinsteinClauseIR(
                        indices=ni,
                        value=cv,
                        location=c.location,
                        where_clause=nwc,
                        variable_ranges=dict(nvr or {}),
                    )
                )
        new_shape = tuple(s.accept(self) for s in n.shape) if n.shape else n.shape
        if n.shape:
            changed = changed or any(ns is not os for os, ns in zip(n.shape, new_shape))
        if not changed:
            return n
        return EinsteinIR(clauses=nc, shape=new_shape, element_type=n.element_type, location=n.location, type_info=_ti(n), shape_info=_si(n))

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
        nlr = self._rw_lr(n.loop_var_ranges)
        if pb is n.primal_body and db is n.diff_body and nlr is n.loop_var_ranges:
            return n
        return SelectAtArgmaxIR(
            pb,
            db,
            n.loop_vars,
            loop_var_ranges=nlr,
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
        if isinstance(expr, LiteralIR):
            return LiteralIR(expr.value, expr.location or self._loc, type_info=_ti(expr), shape_info=_si(expr))
        if isinstance(expr, IdentifierIR):
            return IdentifierIR(expr.name, expr.location or self._loc, expr.defid, type_info=_ti(expr), shape_info=_si(expr))
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
            return IndexRestIR(expr.name, expr.location or self._loc, expr.defid, type_info=_ti(expr), shape_info=_si(expr))
        return _clone_ir_expr(expr)

    def visit_identifier(self, n: IdentifierIR) -> ExpressionIR:
        if n.defid is not None and n.defid in self._m:
            return self._clone_subst_expr(self._m[n.defid])
        return n

    def visit_index_var(self, n: IndexVarIR) -> ExpressionIR:
        if n.defid is not None and n.defid in self._m:
            return self._clone_subst_expr(self._m[n.defid])
        return n


class _SubstDiffsVisitor(_SubstVisitor):
    def __init__(self, pm: Dict[DefId, ExpressionIR], dm: Dict[DefId, ExpressionIR], loc: SourceLocation) -> None:
        super().__init__(pm, loc)
        self._dm = dm

    def visit_differential(self, n: DifferentialIR) -> ExpressionIR:
        op = n.operand
        if isinstance(op, IdentifierIR) and op.defid is not None and op.defid in self._dm:
            return self._clone_subst_expr(self._dm[op.defid])
        return DifferentialIR(op.accept(self), n.location or self._loc, type_info=_ti(n), shape_info=_si(n))


def _sub(expr: ExpressionIR, m: Dict[DefId, ExpressionIR], loc: SourceLocation) -> ExpressionIR:
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
    if expr is None:
        return expr  # type: ignore[return-value]
    return expr.accept(_SubstDiffsVisitor(pm, dm, loc))


_substitute_with_diffs = _sub_wd


def _fresh_prod_interior_index_names(loop_names: Tuple[str, ...]) -> List[str]:
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
            nw = IndexVarIR(interior_name, vl, defid=new_did, range_ir=v.range_ir, type_info=_ti(v), shape_info=_si(v))
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
    if resolver is None:
        raise ValueError("Autodiff: prod pullback requires resolver for fresh DefIds")
    num = _prod_reduction_clone_interior_indices(n, loc, resolver)
    inner = BinaryOpIR(BinaryOp.MUL, BinaryOpIR(BinaryOp.DIV, num, n.body, loc), d_body, loc)
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
