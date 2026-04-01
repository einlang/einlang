from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._core import (
    DIFF_PREFIX,
    _AD_ZERO_TANGENT_BUILTINS,
    _cast_target_has_zero_tangent,
    _fl,
    _is_diff_name,
    _is_zero,
    _pow_chain,
    _rc_index_lists_equivalent,
    _reject_lowered_ir,
    _si,
    _simplify,
    _sub,
    _ti,
    _unsupported_autodiff_ir,
    _unwrap_trivial_einstein_rhs,
    _z,
)
from ._graph import _function_call_ir_label
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
    WhereExpressionIR,
)
from ...shared.defid import DefId
from ...shared.source_location import SourceLocation
from ...shared.types import F32, BinaryOp, ReductionOp, UnaryOp


def _inline_block_lets(block: BlockExpressionIR) -> ExpressionIR:
    if block.final_expr is None:
        return block
    stmts = [s for s in (block.statements or []) if isinstance(s, BindingIR)]
    if not stmts:
        return block.final_expr
    if len(stmts) == 1 and isinstance(block.final_expr, IdentifierIR):
        if block.final_expr.defid == stmts[0].defid and stmts[0].expr is not None:
            return stmts[0].expr
    return block


def _lift_block_binop(op: BinaryOp, L: ExpressionIR, R: ExpressionIR, loc: SourceLocation) -> ExpressionIR:
    sl: List = []
    sr: List = []
    l, r = L, R
    if isinstance(L, BlockExpressionIR) and L.final_expr is not None:
        sl = list(L.statements or [])
        l = L.final_expr
    if isinstance(R, BlockExpressionIR) and R.final_expr is not None:
        sr = list(R.statements or [])
        r = R.final_expr
    ti = _ti(L) or _ti(R)
    si = _si(L) or _si(R)
    if not sl and not sr:
        return BinaryOpIR(op, L, R, loc, type_info=ti or F32, shape_info=si)
    return BlockExpressionIR(sl + sr, loc, BinaryOpIR(op, l, r, loc, type_info=ti, shape_info=si), type_info=ti, shape_info=si)


class DiffVisitor(IRVisitor[ExpressionIR]):
    def __init__(
        self,
        d_map: Dict[DefId, ExpressionIR],
        loc: SourceLocation,
        bindings: Optional[Dict[DefId, Any]] = None,
        resolver: Any = None,
        pretty: bool = False,
        keep_primal_lets: bool = False,
    ) -> None:
        self._d = d_map
        self._loc = loc
        self._B: Dict[DefId, Any] = dict(bindings) if bindings else {}
        self._R = resolver
        self._pretty = pretty
        self._keep_primal_lets = keep_primal_lets

    def visit_identifier(self, n: IdentifierIR) -> ExpressionIR:
        if n.defid is not None:
            ref = self._d.get(n.defid)
            if ref is not None:
                if isinstance(ref, IdentifierIR):
                    return IdentifierIR(ref.name, n.location or self._loc, ref.defid, type_info=_ti(ref), shape_info=_si(ref))
                return ref
        raise ValueError("Autodiff: identifier not in differential map")

    def visit_literal(self, n: LiteralIR) -> ExpressionIR:
        return _z(self._loc)

    def visit_binary_op(self, n: BinaryOpIR) -> ExpressionIR:
        L = n.left
        R = n.right
        loc = n.location or self._loc
        dL = L.accept(self)
        dR = R.accept(self)
        op = n.operator
        ti = _ti(n) or F32
        si = _si(n)
        if op == BinaryOp.ADD:
            return _lift_block_binop(BinaryOp.ADD, dL, dR, loc) if self._pretty else BinaryOpIR(BinaryOp.ADD, dL, dR, loc, type_info=ti, shape_info=si)
        if op == BinaryOp.SUB:
            return _lift_block_binop(BinaryOp.SUB, dL, dR, loc) if self._pretty else BinaryOpIR(BinaryOp.SUB, dL, dR, loc, type_info=ti, shape_info=si)
        if op == BinaryOp.MUL:
            return BinaryOpIR(BinaryOp.ADD, BinaryOpIR(BinaryOp.MUL, L, dR, loc, type_info=ti, shape_info=si), BinaryOpIR(BinaryOp.MUL, R, dL, loc, type_info=ti, shape_info=si), loc, type_info=ti, shape_info=si)
        if op == BinaryOp.DIV:
            num = BinaryOpIR(BinaryOp.SUB, BinaryOpIR(BinaryOp.MUL, R, dL, loc, type_info=ti, shape_info=si), BinaryOpIR(BinaryOp.MUL, L, dR, loc, type_info=ti, shape_info=si), loc, type_info=ti, shape_info=si)
            den = BinaryOpIR(BinaryOp.POW, R, _fl(2, loc), loc, type_info=ti, shape_info=si)
            return BinaryOpIR(BinaryOp.DIV, num, den, loc, type_info=ti, shape_info=si)
        if op == BinaryOp.POW:
            return _pow_chain(n, dL, dR, self._B, self._R, loc)
        if op == BinaryOp.MOD:
            return dL
        if op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.LE, BinaryOp.GT, BinaryOp.GE, BinaryOp.AND, BinaryOp.OR):
            return _z(loc)
        raise ValueError(f"Autodiff: unsupported binary op: {op}")

    def visit_unary_op(self, n: UnaryOpIR) -> ExpressionIR:
        d = n.operand.accept(self)
        if n.operator == UnaryOp.NEG:
            return UnaryOpIR(UnaryOp.NEG, d, n.location or self._loc, type_info=_ti(n) or F32, shape_info=_si(n))
        if n.operator == UnaryOp.POS:
            return d
        raise ValueError(f"Autodiff: unsupported unary op: {n.operator}")

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        db = n.body.accept(self)
        op = n.operation
        if op == ReductionOp.SUM:
            return ReductionExpressionIR(ReductionOp.SUM, n.loop_vars, db, loc, where_clause=n.where_clause, loop_var_ranges=n.loop_var_ranges, type_info=_ti(n), shape_info=_si(n))
        if op == ReductionOp.MAX:
            return SelectAtArgmaxIR(n.body, db, n.loop_vars, loop_var_ranges=n.loop_var_ranges, location=loc, type_info=_ti(n), shape_info=_si(n))
        if op == ReductionOp.MIN:
            return SelectAtArgmaxIR(n.body, db, n.loop_vars, loop_var_ranges=n.loop_var_ranges, location=loc, type_info=_ti(n), shape_info=_si(n), use_argmin=True)
        if op == ReductionOp.PROD:
            from ._core import _prod_pullback_via_sum
            return _prod_pullback_via_sum(n, db, loc, self._R)
        raise ValueError(f"Autodiff: unsupported reduction: {op}")

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        if n.final_expr is None:
            raise ValueError("Autodiff: DiffVisitor block has no final expression")
        loc = n.location or self._loc
        stmts = [s for s in (n.statements or []) if isinstance(s, BindingIR) and s.defid is not None and s.expr is not None]
        if not stmts:
            return BlockExpressionIR([], loc, _simplify(n.final_expr.accept(self), loc), type_info=_ti(n), shape_info=_si(n))
        if len(stmts) == 1 and isinstance(n.final_expr, IdentifierIR) and n.final_expr.defid == stmts[0].defid:
            return stmts[0].expr.accept(self)
        d_ext: Dict[DefId, ExpressionIR] = dict(self._d)
        child = DiffVisitor(d_ext, loc, self._B, self._R, self._pretty, self._keep_primal_lets)
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
                dr = IdentifierIR(dn, s.location or loc, dd, type_info=_ti(s), shape_info=_si(s))
                d_ext[s.defid] = dr
                out_stmts.append(BindingIR(name=dn, expr=pv, location=s.location or loc, defid=dd, type_info=_ti(s)))
            else:
                d_ext[s.defid] = pv
        fp = _simplify(n.final_expr.accept(child), loc)
        if not out_stmts:
            return fp
        return BlockExpressionIR(out_stmts, loc, fp, type_info=_ti(n), shape_info=_si(n))

    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> ExpressionIR:
        return n

    def visit_function_call(self, n: FunctionCallIR) -> ExpressionIR:
        from ._callee import (
            _callee_forward_jvp,
            _pretty_use_callee_tangent_block_direct,
        )
        loc = n.location or self._loc
        args = n.arguments or []
        cdid = n.function_defid
        lab = _function_call_ir_label(n)
        if cdid is None or cdid not in self._B:
            detail = "missing function_defid" if cdid is None else "function_defid not in autodiff binding map"
            raise ValueError(f"Autodiff: cannot differentiate unresolved function call {lab!r} ({detail})")
        binding = self._B[cdid]
        if not isinstance(binding.expr, FunctionValueIR):
            raise ValueError(f"Autodiff: call {lab!r} resolves to non-function binding {binding.name or '?'}; expected a function value")
        fv = binding.expr
        ps = fv.parameters or []
        tangent_by_param = {p.defid: args[i].accept(self) for i, p in enumerate(ps) if p.defid is not None}
        tang = _callee_forward_jvp(fv, args, tangent_by_param, loc, self._B, self._R)
        if self._pretty:
            peeled = _peel_inlineable_tangent_blocks(tang)
            if _pretty_callee_tangent_inlineable(peeled):
                return peeled
            if _pretty_use_callee_tangent_block_direct(fv, list(args), peeled):
                return peeled
            return _wrap_tangent_binding(peeled, cdid, fv, ps, list(args), n.callee_expr, self._B, self._R, loc, _ti(n), _si(n))
        return tang

    def visit_rectangular_access(self, n: RectangularAccessIR) -> ExpressionIR:
        from ._expand import _propagate_ti
        loc = n.location or self._loc
        da = n.array.accept(self)
        if _is_zero(da):
            return _z(loc)
        indices = list(n.indices or [])
        if isinstance(da, EinsteinIR) and da.clauses and len(da.clauses) == 1:
            c = da.clauses[0]
            ci = c.indices or []
            if len(ci) == len(indices):
                rm: Dict[DefId, ExpressionIR] = {}
                for j, cidx in enumerate(ci):
                    if isinstance(cidx, (IndexVarIR, IdentifierIR)) and cidx.defid is not None and j < len(indices):
                        rm[cidx.defid] = indices[j]
                inl = _sub(c.value, rm, loc)
                ti = _ti(n)
                si = _si(n)
                if ti is not None or si is not None:
                    _propagate_ti(inl, ti, si)
                return inl
        if isinstance(da, RectangularAccessIR):
            dai = list(da.indices or [])
            if _rc_index_lists_equivalent(indices, dai):
                return da
        return RectangularAccessIR(da, indices, loc, type_info=_ti(n), shape_info=_si(n))

    def visit_if_expression(self, n: IfExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        dt = n.then_expr.accept(self)
        if isinstance(n.then_expr, BlockExpressionIR) and not isinstance(dt, BlockExpressionIR):
            dt = BlockExpressionIR([], n.then_expr.location or loc, dt, type_info=_ti(n.then_expr), shape_info=_si(n.then_expr))
        de = n.else_expr.accept(self) if n.else_expr is not None else _z(self._loc)
        if isinstance(n.else_expr, BlockExpressionIR) and not isinstance(de, BlockExpressionIR):
            de = BlockExpressionIR([], n.else_expr.location or loc, de, type_info=_ti(n.else_expr), shape_info=_si(n.else_expr))
        return IfExpressionIR(condition=n.condition, then_expr=dt, location=loc, else_expr=de, type_info=_ti(n), shape_info=_si(n))

    def visit_cast_expression(self, n: CastExpressionIR) -> ExpressionIR:
        if _cast_target_has_zero_tangent(n.target_type):
            return _z(self._loc)
        return n.expr.accept(self)

    def visit_einstein(self, n: EinsteinIR) -> ExpressionIR:
        nc: List[EinsteinClauseIR] = []
        for c in n.clauses or []:
            try:
                dv = _unwrap_trivial_einstein_rhs(_simplify(c.value.accept(self), c.location or self._loc))
            except (ValueError, KeyError):
                continue
            if _is_zero(dv):
                continue
            nc.append(EinsteinClauseIR(indices=list(c.indices or []), value=dv, location=c.location, where_clause=c.where_clause, variable_ranges=dict(c.variable_ranges or {})))
        if not nc:
            return _z(self._loc)
        return EinsteinIR(clauses=nc, shape=n.shape, element_type=n.element_type, location=n.location, type_info=_ti(n), shape_info=_si(n))

    def visit_lowered_einstein(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("DiffVisitor", n)

    def visit_differential(self, n: DifferentialIR) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)

    def visit_jagged_access(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)

    def visit_lambda(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)

    def visit_range(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)

    def visit_array_comprehension(self, n: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("DiffVisitor", n)

    def visit_array_literal(self, n: ArrayLiteralIR) -> ExpressionIR:
        return _z(self._loc)

    def visit_tuple_expression(self, n: TupleExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        return TupleExpressionIR([elem.accept(self) for elem in (n.elements or [])], loc, type_info=_ti(n), shape_info=_si(n))

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
    cur: ExpressionIR = expr
    for _ in range(64):
        if not isinstance(cur, BlockExpressionIR):
            return cur
        nxt = _inline_block_lets(cur)
        if nxt is cur:
            return cur
        cur = nxt
    return cur


class _PrettyCalleeTangentInlineableVisitor(IRVisitor[bool]):
    def _all(self, *nodes: Optional[ExpressionIR]) -> bool:
        for node in nodes:
            if node is not None and not node.accept(self):
                return False
        return True

    def _all_iter(self, nodes) -> bool:
        for node in nodes or []:
            if node is not None and not node.accept(self):
                return False
        return True

    def visit_block_expression(self, n: BlockExpressionIR) -> bool:
        if any(isinstance(stmt, BindingIR) for stmt in (n.statements or [])):
            return False
        if n.final_expr is None:
            return False
        return n.final_expr.accept(self)

    def visit_literal(self, n: LiteralIR) -> bool:
        return True

    def visit_identifier(self, n: IdentifierIR) -> bool:
        return True

    def visit_index_var(self, n: Any) -> bool:
        return True

    def visit_index_rest(self, n: Any) -> bool:
        return True

    def visit_unary_op(self, n: UnaryOpIR) -> bool:
        return self._all(n.operand)

    def visit_binary_op(self, n: BinaryOpIR) -> bool:
        return self._all(n.left, n.right)

    def visit_cast_expression(self, n: CastExpressionIR) -> bool:
        return self._all(n.expr)

    def visit_function_call(self, n: FunctionCallIR) -> bool:
        return self._all(n.callee_expr) and self._all_iter(n.arguments)

    def visit_builtin_call(self, n: BuiltinCallIR) -> bool:
        return self._all_iter(n.args)

    def visit_rectangular_access(self, n: RectangularAccessIR) -> bool:
        return self._all(n.array) and self._all_iter(n.indices)

    def visit_if_expression(self, n: IfExpressionIR) -> bool:
        return False

    def visit_differential(self, n: DifferentialIR) -> bool:
        return False

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> bool:
        return False

    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> bool:
        return False

    def visit_einstein(self, n: EinsteinIR) -> bool:
        return False

    def visit_lowered_einstein(self, n: Any) -> bool:
        return False

    def visit_jagged_access(self, n: Any) -> bool:
        return False

    def visit_lambda(self, n: Any) -> bool:
        return False

    def visit_range(self, n: Any) -> bool:
        return False

    def visit_array_comprehension(self, n: Any) -> bool:
        return False

    def visit_array_literal(self, n: ArrayLiteralIR) -> bool:
        return False

    def visit_tuple_expression(self, n: TupleExpressionIR) -> bool:
        return False

    def visit_tuple_access(self, n: TupleAccessIR) -> bool:
        return False

    def visit_interpolated_string(self, n: Any) -> bool:
        return False

    def visit_member_access(self, n: MemberAccessIR) -> bool:
        return False

    def visit_try_expression(self, n: Any) -> bool:
        return False

    def visit_match_expression(self, n: Any) -> bool:
        return False

    def visit_where_expression(self, n: Any) -> bool:
        return False

    def visit_pipeline_expression(self, n: Any) -> bool:
        return False

    def visit_module(self, n: Any) -> bool:
        return False

    def visit_program(self, n: Any) -> bool:
        return False

    def visit_binding(self, n: BindingIR) -> bool:
        return False

    def visit_literal_pattern(self, n: Any) -> bool:
        return False

    def visit_identifier_pattern(self, n: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, n: Any) -> bool:
        return False

    def visit_tuple_pattern(self, n: Any) -> bool:
        return False

    def visit_array_pattern(self, n: Any) -> bool:
        return False

    def visit_rest_pattern(self, n: Any) -> bool:
        return False

    def visit_guard_pattern(self, n: Any) -> bool:
        return False


def _pretty_callee_tangent_inlineable(e: ExpressionIR) -> bool:
    return e.accept(_PrettyCalleeTangentInlineableVisitor())


def _wrap_tangent_binding(
    tang: ExpressionIR,
    cdid: DefId,
    fv: FunctionValueIR,
    ps: List,
    args: List[ExpressionIR],
    callee_expr: ExpressionIR,
    B: Dict,
    R: Any,
    loc: SourceLocation,
    ti: Any,
    si: Any,
) -> ExpressionIR:
    cn = getattr(callee_expr, "name", None) or "f"
    dd = R.allocate_for_local()
    if len(args) == 1 and isinstance(args[0], IdentifierIR):
        an = args[0].name or ""
        dn = DIFF_PREFIX + cn + an if len(cn) == 1 else DIFF_PREFIX + cn + "_" + an
    elif len(args) > 1:
        dn = DIFF_PREFIX + cn + "_call"
    else:
        dn = DIFF_PREFIX + cn
    db = BindingIR(name=dn, expr=tang, location=loc, defid=dd, type_info=ti)
    dr = IdentifierIR(dn, loc, dd, type_info=ti, shape_info=si)
    return BlockExpressionIR([db], loc, dr, type_info=ti, shape_info=si)
