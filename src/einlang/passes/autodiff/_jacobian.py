from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ._core import (
    _AD_ZERO_TANGENT_BUILTINS,
    _cast_target_has_zero_tangent,
    _fl,
    _is_diff_name,
    _is_sum_reduction_of_zero,
    _is_zero,
    _pow_chain,
    _rc_index_lists_equivalent,
    _reject_lowered_ir,
    _si,
    _simplify,
    _sub,
    _sub_wd,
    _ti,
    _unsupported_autodiff_ir,
    _z,
)
from ._expr import _eval_const_expr
from ._graph import (
    _DependencyQueryCache,
    _binding_is_rect_slice_of_tensor,
    _einstein_clause_values_rect_read_tensor,
    _function_call_ir_label,
    _jacobian_rhs_depends_on_wrt,
    _rectangular_read_root_defid,
)
from ._tensor import (
    _alloc_wrt_gradient_axes,
    _append_cotangent_axes_to_clause_indices,
    _ensure_cotangent_axes_on_clause_indices,
    _flatten_rect_access,
    _is_bare_wrt_tensor_deriv,
    _jacobian_rect_read_wrt_slice_binding,
    _jacobian_tensor_id_wrt_slice_binding,
    _kronecker_delta_indices,
    _merge_primal_clause_with_cotangent_einstein,
    _reject_bare_wrt_tensor_jacobian,
    _tensor_rank_from_binding,
    _tensor_rank_from_expr,
)
from ...ir.nodes import (
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
    RangeIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    SelectAtArgmaxIR,
    TryExpressionIR,
    TupleAccessIR,
    TupleExpressionIR,
    UnaryOpIR,
    WhereClauseIR,
)
from ...shared.defid import DefId
from ...shared.source_location import SourceLocation
from ...shared.types import BOOL, F32, UNKNOWN, BinaryOp, PrimitiveType, ReductionOp, UnaryOp


class JacobianVisitor(IRVisitor[ExpressionIR]):
    def __init__(
        self,
        wrt: DefId,
        loc: SourceLocation,
        bindings: Dict[DefId, BindingIR],
        resolver: Any,
        stmt_partial: Optional[Dict[DefId, ExpressionIR]] = None,
        wrt_tangent: Optional[ExpressionIR] = None,
        primal_subst: Optional[Dict[DefId, ExpressionIR]] = None,
        shared_cotangent_axes: Optional[List[IndexVarIR]] = None,
        full_cotangent_axes: Optional[List[IndexVarIR]] = None,
        legacy_directional: bool = False,
        dependency_cache: Optional[_DependencyQueryCache] = None,
    ) -> None:
        self._wrt = wrt
        self._loc = loc
        self._B = bindings
        self._R = resolver
        self._sp = stmt_partial
        self._wt = wrt_tangent
        self._ps = primal_subst
        self._legacy_directional = legacy_directional
        self._dep_cache = (
            dependency_cache
            if dependency_cache is not None and dependency_cache.bindings is bindings
            else _DependencyQueryCache(bindings)
        )
        self._wrt_axes: Optional[List[IndexVarIR]] = None
        self._full_wrt_axes: Optional[List[IndexVarIR]] = None
        wb = bindings.get(wrt)
        if wb is not None and _tensor_rank_from_binding(wb) > 0:
            self._wrt_axes = (
                list(shared_cotangent_axes)
                if shared_cotangent_axes is not None
                else _alloc_wrt_gradient_axes(wb, wrt, resolver, loc)
            )
            self._full_wrt_axes = (
                list(full_cotangent_axes)
                if full_cotangent_axes is not None
                else list(self._wrt_axes)
            )

    def _wrap_if_cotangent_indices(self, inner: ExpressionIR, n: ReductionExpressionIR) -> ExpressionIR:
        ax = self._wrt_axes
        if not ax:
            return inner
        loc = n.location or self._loc
        axis_ids = {iv.defid for iv in ax if getattr(iv, "defid", None) is not None}
        from ._expr import _expr_uses_index_defids
        if not axis_ids or not _expr_uses_index_defids(inner, axis_ids):
            return inner
        mlr: Dict[DefId, RangeIR] = dict(n.loop_var_ranges or {})
        for iv in ax:
            did = getattr(iv, "defid", None)
            if did is not None and iv.range_ir is not None:
                mlr[did] = iv.range_ir
        return EinsteinIR(
            clauses=[EinsteinClauseIR(indices=list(ax), value=inner, location=loc, variable_ranges=mlr)],
            shape=None,
            element_type=None,
            location=loc,
            type_info=_ti(n),
            shape_info=None,
        )

    def visit_identifier(self, n: IdentifierIR) -> ExpressionIR:
        if n.defid == self._wrt:
            if self._wrt_axes:
                wb = self._B.get(self._wrt)
                return IdentifierIR(
                    n.name or "?",
                    n.location or self._loc,
                    n.defid,
                    type_info=_ti(n) or (_ti(wb) if wb is not None else None) or UNKNOWN,
                    shape_info=_si(n) or (_si(wb) if wb is not None else None),
                )
            return self._wt if self._wt is not None else _fl(1, self._loc)
        if n.defid is not None and self._sp is not None:
            pre = self._sp.get(n.defid)
            if pre is not None:
                return pre
        if n.defid is not None and self._ps is not None:
            sub = self._ps.get(n.defid)
            if sub is not None:
                return sub.accept(self)
        if n.defid is not None:
            b = self._B.get(n.defid)
            if b is not None and b.expr is not None:
                if _is_diff_name(b.name or ""):
                    return n
                w_b = self._B.get(self._wrt)
                slice_root = _rectangular_read_root_defid(w_b.expr) if w_b is not None and w_b.expr is not None else None
                tensor_alias = slice_root is not None and slice_root == n.defid
                if not tensor_alias and not _jacobian_rhs_depends_on_wrt(b.expr, self._wrt, self._B, self._dep_cache):
                    return _z(self._loc)
                if _binding_is_rect_slice_of_tensor(self._wrt, n.defid, self._B):
                    return IdentifierIR(n.name or "?", n.location or self._loc, n.defid, type_info=_ti(n), shape_info=_si(n))
                if (
                    isinstance(b.expr, EinsteinIR)
                    and n.defid != self._wrt
                    and self._wrt_axes is None
                    and _einstein_clause_values_rect_read_tensor(b.expr, n.defid)
                ):
                    return _z(self._loc)
                return b.expr.accept(self)
        return _z(self._loc)

    def visit_literal(self, n: LiteralIR) -> ExpressionIR:
        return _z(self._loc)

    def visit_array_literal(self, n: ArrayLiteralIR) -> ExpressionIR:
        return _z(self._loc)

    def visit_binary_op(self, n: BinaryOpIR) -> ExpressionIR:
        L = n.left
        R = n.right
        loc = n.location or self._loc
        dL = L.accept(self)
        dR = R.accept(self)
        _reject_bare_wrt_tensor_jacobian(dL, self._wrt, self._wrt_axes, "binary left")
        _reject_bare_wrt_tensor_jacobian(dR, self._wrt, self._wrt_axes, "binary right")
        op = n.operator
        ti = _ti(n) or F32
        si = _si(n)
        if op == BinaryOp.ADD:
            if _is_zero(dR) or _is_sum_reduction_of_zero(dR):
                return dL
            if _is_zero(dL) or _is_sum_reduction_of_zero(dL):
                return dR
            return BinaryOpIR(BinaryOp.ADD, dL, dR, loc, type_info=ti, shape_info=si)
        if op == BinaryOp.SUB:
            return BinaryOpIR(BinaryOp.SUB, dL, dR, loc, type_info=ti, shape_info=si)
        if op == BinaryOp.MUL:
            return BinaryOpIR(
                BinaryOp.ADD,
                BinaryOpIR(BinaryOp.MUL, L, dR, loc, type_info=ti, shape_info=si),
                BinaryOpIR(BinaryOp.MUL, R, dL, loc, type_info=ti, shape_info=si),
                loc,
                type_info=ti,
                shape_info=si,
            )
        if op == BinaryOp.DIV:
            num = BinaryOpIR(
                BinaryOp.SUB,
                BinaryOpIR(BinaryOp.MUL, R, dL, loc, type_info=ti, shape_info=si),
                BinaryOpIR(BinaryOp.MUL, L, dR, loc, type_info=ti, shape_info=si),
                loc,
                type_info=ti,
                shape_info=si,
            )
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
        _reject_bare_wrt_tensor_jacobian(d, self._wrt, self._wrt_axes, "unary")
        if n.operator == UnaryOp.NEG:
            return UnaryOpIR(UnaryOp.NEG, d, n.location or self._loc, type_info=_ti(n) or F32, shape_info=_si(n))
        if n.operator == UnaryOp.POS:
            return d
        raise ValueError(f"Autodiff: unsupported unary op: {n.operator}")

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        d_body = n.body.accept(self)
        op = n.operation
        if op == ReductionOp.SUM:
            inner = ReductionExpressionIR(ReductionOp.SUM, n.loop_vars, d_body, loc, where_clause=n.where_clause, loop_var_ranges=n.loop_var_ranges, type_info=_ti(n), shape_info=_si(n))
            return self._wrap_if_cotangent_indices(inner, n)
        if op == ReductionOp.MAX:
            inner = SelectAtArgmaxIR(n.body, d_body, n.loop_vars, loop_var_ranges=n.loop_var_ranges, location=loc, type_info=_ti(n), shape_info=_si(n))
            return self._wrap_if_cotangent_indices(inner, n)
        if op == ReductionOp.MIN:
            inner = SelectAtArgmaxIR(n.body, d_body, n.loop_vars, loop_var_ranges=n.loop_var_ranges, location=loc, type_info=_ti(n), shape_info=_si(n), use_argmin=True)
            return self._wrap_if_cotangent_indices(inner, n)
        if op == ReductionOp.PROD:
            from ._core import _prod_pullback_via_sum
            inner = _prod_pullback_via_sum(n, d_body, loc, self._R)
            return self._wrap_if_cotangent_indices(inner, n)
        raise ValueError(f"Autodiff: unsupported reduction: {op}")

    def visit_rectangular_access(self, n: RectangularAccessIR) -> ExpressionIR:
        loc = n.location or self._loc
        root_expr, full_indices = _flatten_rect_access(n)
        if isinstance(root_expr, IdentifierIR) and root_expr.defid is not None and self._wrt_axes is None:
            full_sl = _jacobian_rect_read_wrt_slice_binding(full_indices, self._wrt, loc, self._B, array_root=root_expr.defid)
            if full_sl is not None:
                return full_sl
        indices = list(n.indices or [])
        da = n.array.accept(self)
        if _is_zero(da):
            return _z(loc)
        if isinstance(da, IdentifierIR) and da.defid is not None and self._wrt_axes is None:
            sl = _jacobian_rect_read_wrt_slice_binding(indices, self._wrt, loc, self._B, array_root=da.defid)
            if sl is not None:
                return sl
        if isinstance(da, IdentifierIR) and da.defid == self._wrt and self._wrt_axes:
            if len(indices) != len(self._wrt_axes):
                return _z(loc)
            return _kronecker_delta_indices(indices, self._wrt_axes, loc)
        if isinstance(da, RectangularAccessIR):
            dai = list(da.indices or [])
            if _rc_index_lists_equivalent(indices, dai):
                return da
        if isinstance(da, EinsteinIR):
            used_axis_ids: Set[DefId] = set()
            full_axes = self._full_wrt_axes or self._wrt_axes or []
            current_axes = self._wrt_axes or full_axes
            axis_ids = {iv.defid for iv in full_axes if getattr(iv, "defid", None) is not None}
            if axis_ids:
                for c in da.clauses or []:
                    for ix in c.indices or []:
                        did = getattr(ix, "defid", None)
                        if did is not None and did in axis_ids:
                            used_axis_ids.add(did)
            if used_axis_ids:
                have = {getattr(ix, "defid", None) for ix in indices if getattr(ix, "defid", None) is not None}
                ext = list(indices)
                for p, full_iv in enumerate(full_axes):
                    full_did = getattr(full_iv, "defid", None)
                    if full_did is None or full_did not in used_axis_ids:
                        continue
                    current_iv = current_axes[p] if p < len(current_axes) else full_iv
                    current_did = getattr(current_iv, "defid", None)
                    if current_did is None:
                        continue
                    if current_did != full_did:
                        ext.append(current_iv)
                        continue
                    if current_did not in have and current_did in used_axis_ids:
                        ext.append(current_iv)
                        have.add(current_did)
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
        return IfExpressionIR(condition=n.condition, then_expr=dt, location=n.location or self._loc, else_expr=de, type_info=_ti(n), shape_info=_si(n))

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        if n.final_expr is None:
            raise ValueError("Autodiff: JacobianVisitor block has no final expression")
        loc = n.location or self._loc
        stmts = [s for s in (n.statements or []) if isinstance(s, BindingIR) and s.defid is not None and s.expr is not None]
        if not stmts:
            fp = _simplify(n.final_expr.accept(self), loc)
            return BlockExpressionIR([], loc, fp, type_info=_ti(n), shape_info=_si(n))
        if len(stmts) == 1 and isinstance(n.final_expr, IdentifierIR) and n.final_expr.defid == stmts[0].defid:
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
            full_cotangent_axes=self._full_wrt_axes,
            legacy_directional=self._legacy_directional,
            dependency_cache=self._dep_cache,
        )
        for s in stmts:
            pv = _simplify(s.expr.accept(child), loc)
            sp_ext[s.defid] = _z(loc) if _is_zero(pv) else pv
        der = n.final_expr.accept(child)
        return BlockExpressionIR(list(stmts), loc, der, type_info=_ti(n), shape_info=_si(n))

    def visit_function_call(self, n: FunctionCallIR) -> ExpressionIR:
        from ._callee import (
            _callee_arg_with_binding_metadata,
            _callee_forward_jvp,
            _diff_callee_block,
            _sub_callee,
            _sum_terms,
        )
        loc = n.location or self._loc
        args = n.arguments or []
        cdid = n.function_defid
        lab = _function_call_ir_label(n)
        if cdid is None or cdid not in self._B:
            detail = "missing function_defid" if cdid is None else "function_defid not in binding map"
            raise ValueError(f"Autodiff: JacobianVisitor cannot differentiate unresolved call {lab!r} ({detail})")
        b = self._B[cdid]
        if not isinstance(b.expr, FunctionValueIR):
            raise ValueError(f"Autodiff: JacobianVisitor call {lab!r} resolves to non-function binding {b.name or '?'}")
        fv = b.expr
        ps = fv.parameters or []
        body = fv.body
        rm = {p.defid: _callee_arg_with_binding_metadata(args[j], self._B) for j, p in enumerate(ps) if p.defid is not None and j < len(args)}
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
                    return _callee_forward_jvp(fv, args, tangent_by_param, loc, self._B, self._R)
                except Exception:
                    pass
        rule_body = getattr(fv, "custom_diff_body", None)
        if rule_body is not None and len(ps) == len(args):
            if len(ps) == 1 and ps[0].defid is not None:
                d_arg = self._wt if self._wt is not None else args[0].accept(self)
                _reject_bare_wrt_tensor_jacobian(d_arg, self._wrt, self._wrt_axes, "custom_diff d_arg")
                return _sub_callee(_sub_wd(rule_body, rm, {ps[0].defid: d_arg}, loc), fv, rm, loc, fold_body_bindings=False)
            terms: List[ExpressionIR] = []
            for i, p in enumerate(ps):
                if p.defid is None:
                    continue
                ud = {ps[j].defid: (_fl(1, loc) if j == i else _z(loc)) for j in range(len(ps)) if ps[j].defid is not None}
                coef = _simplify(_sub_callee(_sub_wd(rule_body, rm, ud, loc), fv, rm, loc, fold_body_bindings=False), loc)
                av = args[i].accept(self)
                _reject_bare_wrt_tensor_jacobian(av, self._wrt, self._wrt_axes, "custom_diff chain")
                terms.append(BinaryOpIR(BinaryOp.MUL, coef, av, loc, type_info=_ti(coef) or _ti(av) or _ti(n) or F32, shape_info=_si(coef) or _si(av) or _si(n)))
            return _sum_terms(terms, loc)
        if body is None:
            raise ValueError("Autodiff: JacobianVisitor cannot differentiate function with no body")
        if isinstance(body, BlockExpressionIR) and self._R is not None:
            return _sub_callee(_diff_callee_block(body, self._wrt, loc, self._B, self._R, rm, self._wt), fv, rm, loc)
        terms = []
        for i, p in enumerate(ps):
            if p.defid is None or i >= len(args):
                continue
            iv = JacobianVisitor(p.defid, loc, self._B, self._R, dependency_cache=self._dep_cache)
            av = args[i].accept(self)
            if isinstance(args[i], IdentifierIR) and isinstance(av, IdentifierIR) and args[i].defid is not None and _binding_is_rect_slice_of_tensor(self._wrt, args[i].defid, self._B):
                one_hot = _jacobian_tensor_id_wrt_slice_binding(
                    IdentifierIR(args[i].name or "?", args[i].location or loc, args[i].defid, type_info=_ti(args[i]), shape_info=_si(args[i])),
                    self._wrt,
                    loc,
                    self._B,
                    self._R,
                )
                if one_hot is not None:
                    av = one_hot
            _reject_bare_wrt_tensor_jacobian(av, self._wrt, self._wrt_axes, "call arg partial")
            coef = _sub(body.accept(iv), rm, loc)
            terms.append(BinaryOpIR(BinaryOp.MUL, coef, av, loc, type_info=_ti(coef) or _ti(av) or _ti(n) or F32, shape_info=_si(coef) or _si(av) or _si(n)))
        return _sum_terms(terms, loc)

    def visit_einstein(self, n: EinsteinIR) -> ExpressionIR:
        return _diff_einstein_wrt(
            n,
            self._wrt,
            self._loc,
            self._B,
            self._R,
            self._wt,
            sp=self._sp,
            ps=self._ps,
            cotangent_axes=self._full_wrt_axes or self._wrt_axes,
            legacy_directional=self._legacy_directional,
        )

    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> ExpressionIR:
        return n

    def visit_lowered_einstein(self, n: Any) -> ExpressionIR:
        _reject_lowered_ir("JacobianVisitor", n)

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
        return TupleExpressionIR([elem.accept(self) for elem in (n.elements or [])], loc, type_info=_ti(n), shape_info=_si(n))

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


def _flatten_product(expr: ExpressionIR) -> Optional[List[Tuple[ExpressionIR, List]]]:
    if isinstance(expr, RectangularAccessIR):
        arr = expr.array
        if isinstance(arr, IdentifierIR) and arr.defid is not None:
            return [(arr, list(expr.indices or []))]
        return None
    if isinstance(expr, BinaryOpIR) and expr.operator == BinaryOp.MUL:
        l = _flatten_product(expr.left)
        r = _flatten_product(expr.right)
        if l is not None and r is not None:
            return l + r
    return None


def _is_direct_einstein_tensor_jacobian(
    expr: Optional[ExpressionIR],
    wrt: DefId,
    bindings: Dict[DefId, BindingIR],
    dep_cache: Optional[_DependencyQueryCache] = None,
) -> bool:
    if not isinstance(expr, EinsteinIR):
        return False
    cache = dep_cache if dep_cache is not None else _DependencyQueryCache(bindings)
    saw_wrt = False
    for clause in expr.clauses or []:
        val = clause.value
        if not isinstance(val, ReductionExpressionIR):
            return False
        factors = _flatten_product(val.body)
        if not factors:
            return False
        clause_has_wrt = False
        for factor, _ in factors:
            if not isinstance(factor, IdentifierIR) or factor.defid is None:
                return False
            fd = factor.defid
            if fd == wrt:
                clause_has_wrt = True
                saw_wrt = True
                continue
            b = bindings.get(fd)
            if b is not None and b.expr is not None and _jacobian_rhs_depends_on_wrt(b.expr, wrt, bindings, cache):
                return False
        if not clause_has_wrt:
            return False
    return saw_wrt


def _build_deriv_idx(
    clause_indices: List,
    wrt_indices: List,
    wrt_id: IdentifierIR,
    resolver: Any,
    loc: SourceLocation,
    allow_reuse: bool,
    shared_axes: Optional[List[IndexVarIR]] = None,
) -> Tuple[List, Set[DefId], Dict]:
    ci_by_did: Dict[DefId, Any] = {}
    for c in clause_indices:
        did = getattr(c, "defid", None)
        if did is not None:
            ci_by_did[did] = c
    dvars: List = []
    new_dids: Set[DefId] = set()
    new_vr: Dict = {}
    for p in range(len(wrt_indices)):
        ip = wrt_indices[p]
        did_p = getattr(ip, "defid", None)
        if allow_reuse and did_p is not None and did_p in ci_by_did:
            dvars.append(ci_by_did[did_p])
        elif allow_reuse and shared_axes is not None and p < len(shared_axes) and isinstance(shared_axes[p], IndexVarIR):
            iv = shared_axes[p]
            dvars.append(iv)
            if iv.defid is not None:
                new_dids.add(iv.defid)
            if iv.defid is not None and iv.range_ir is not None:
                new_vr[iv.defid] = iv.range_ir
        else:
            nd = resolver.allocate_for_local()
            sh = MemberAccessIR(object=wrt_id, member="shape", location=loc, type_info=UNKNOWN)
            dim = LiteralIR(p, loc, type_info=PrimitiveType("i32"))
            sd = RectangularAccessIR(array=sh, indices=[dim], location=loc, type_info=UNKNOWN)
            rng = RangeIR(start=LiteralIR(0, loc, type_info=PrimitiveType("i32")), end=sd, location=loc, type_info=UNKNOWN)
            iv = IndexVarIR("_ad_%d" % p, loc, nd, range_ir=rng)
            dvars.append(iv)
            new_dids.add(nd)
            if iv.defid is not None and iv.range_ir is not None:
                new_vr[iv.defid] = iv.range_ir
    return dvars, new_dids, new_vr


def _merged_lr(val: ReductionExpressionIR, clause: EinsteinClauseIR) -> Dict:
    out: Dict = dict(val.loop_var_ranges or {})
    vr = clause.variable_ranges or {}
    for lv in val.loop_vars or []:
        did = getattr(lv, "defid", None)
        if did is not None and did not in out and did in vr:
            out[did] = vr[did]
    return out


def _diff_einstein_wrt(
    expr: EinsteinIR,
    wrt: DefId,
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    R: Any,
    wt: Optional[ExpressionIR] = None,
    sp: Optional[Dict[DefId, ExpressionIR]] = None,
    ps: Optional[Dict[DefId, ExpressionIR]] = None,
    cotangent_axes: Optional[List[IndexVarIR]] = None,
    legacy_directional: bool = False,
) -> ExpressionIR:
    dep_cache = _DependencyQueryCache(B)
    dc: List[EinsteinClauseIR] = []
    shared_axes: Optional[List[IndexVarIR]] = [] if legacy_directional else (list(cotangent_axes) if cotangent_axes is not None else [])
    expr_rank = _tensor_rank_from_expr(expr, B)
    align_shared_output_axes = bool(shared_axes) and expr_rank > 0 and expr_rank < len(shared_axes)
    for clause in expr.clauses or []:
        clause_shared_axes: Optional[List[IndexVarIR]] = shared_axes
        if align_shared_output_axes and shared_axes:
            mixed_axes = list(shared_axes)
            for p, ix in enumerate(clause.indices or []):
                if p >= len(mixed_axes):
                    break
                if isinstance(ix, IndexVarIR):
                    mixed_axes[p] = ix
            clause_shared_axes = mixed_axes
        val = clause.value
        if not isinstance(val, ReductionExpressionIR):
            jv = JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps, shared_cotangent_axes=clause_shared_axes, full_cotangent_axes=shared_axes, dependency_cache=dep_cache)
            d_val = _simplify(val.accept(jv), loc)
            mi, merged_v, nvr = _merge_primal_clause_with_cotangent_einstein(clause, d_val)
            mi, nvr = _ensure_cotangent_axes_on_clause_indices(mi, nvr, merged_v, clause_shared_axes)
            mi, nvr = _append_cotangent_axes_to_clause_indices(mi, nvr, clause_shared_axes)
            dc.append(EinsteinClauseIR(indices=mi, value=merged_v, location=clause.location, where_clause=clause.where_clause, variable_ranges=nvr))
            continue
        inner = val.body
        factors = _flatten_product(inner) if inner else None
        if not factors:
            if val.operation == ReductionOp.SUM:
                d_inner = inner.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps, shared_cotangent_axes=clause_shared_axes, full_cotangent_axes=shared_axes, dependency_cache=dep_cache))
                sum_val = ReductionExpressionIR(ReductionOp.SUM, list(val.loop_vars or []), d_inner, loc, where_clause=val.where_clause, loop_var_ranges=_merged_lr(val, clause), type_info=_ti(val), shape_info=_si(val))
                mi = list(clause.indices or [])
                nvr = dict(clause.variable_ranges or {})
                mi, nvr = _ensure_cotangent_axes_on_clause_indices(mi, nvr, sum_val, clause_shared_axes)
                mi, nvr = _append_cotangent_axes_to_clause_indices(mi, nvr, clause_shared_axes)
                dc.append(EinsteinClauseIR(indices=mi, value=sum_val, location=clause.location, where_clause=clause.where_clause, variable_ranges=nvr))
            else:
                d_val = val.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps, shared_cotangent_axes=clause_shared_axes, full_cotangent_axes=shared_axes, dependency_cache=dep_cache))
                mi = list(clause.indices or [])
                nvr = dict(clause.variable_ranges or {})
                mi, nvr = _ensure_cotangent_axes_on_clause_indices(mi, nvr, d_val, clause_shared_axes)
                mi, nvr = _append_cotangent_axes_to_clause_indices(mi, nvr, clause_shared_axes)
                dc.append(EinsteinClauseIR(indices=mi, value=d_val, location=clause.location, where_clause=clause.where_clause, variable_ranges=nvr))
            continue
        wrt_pos: List[int] = []
        for i, (a, _) in enumerate(factors):
            if not isinstance(a, IdentifierIR) or a.defid is None:
                continue
            if a.defid == wrt:
                wrt_pos.append(i)
                continue
            if ps is not None:
                sub = ps.get(a.defid)
                if isinstance(sub, IdentifierIR) and sub.defid == wrt:
                    wrt_pos.append(i)
        if not wrt_pos:
            if val.operation == ReductionOp.SUM:
                d_inner = inner.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps, shared_cotangent_axes=clause_shared_axes, full_cotangent_axes=shared_axes, dependency_cache=dep_cache))
                sum_val = ReductionExpressionIR(ReductionOp.SUM, list(val.loop_vars or []), d_inner, loc, where_clause=val.where_clause, loop_var_ranges=_merged_lr(val, clause), type_info=_ti(val), shape_info=_si(val))
                mi = list(clause.indices or [])
                nvr = dict(clause.variable_ranges or {})
                mi, nvr = _ensure_cotangent_axes_on_clause_indices(mi, nvr, sum_val, clause_shared_axes)
                mi, nvr = _append_cotangent_axes_to_clause_indices(mi, nvr, clause_shared_axes)
                dc.append(EinsteinClauseIR(indices=mi, value=sum_val, location=clause.location, where_clause=clause.where_clause, variable_ranges=nvr))
            continue
        first_wi = factors[wrt_pos[0]][1]
        wb = B.get(wrt)
        wid = IdentifierIR((wb.name if wb else "") or "?", loc, wrt, type_info=UNKNOWN)
        ci = list(clause.indices or [])
        allow_reuse = len(ci) < len(first_wi)
        dvars, new_dids, new_vr = _build_deriv_idx(ci, first_wi, wid, R, loc, allow_reuse, shared_axes=clause_shared_axes)
        lvs = list(val.loop_vars or [])
        mlr = _merged_lr(val, clause)
        if val.operation in (ReductionOp.MAX, ReductionOp.MIN):
            db: ExpressionIR = _fl(1, loc)
            for p in range(len(first_wi)):
                if getattr(dvars[p], "defid", None) in new_dids:
                    eq = BinaryOpIR(BinaryOp.EQ, first_wi[p], dvars[p], loc, type_info=BOOL)
                    db = IfExpressionIR(eq, db, loc, else_expr=_z(loc), type_info=_ti(val) or F32)
            sel = SelectAtArgmaxIR(val.body, db, lvs, loop_var_ranges=mlr, location=loc, type_info=_ti(val), shape_info=_si(val), use_argmin=(val.operation == ReductionOp.MIN))
            nvr = dict(clause.variable_ranges or {})
            nvr.update(new_vr)
            out_val: ExpressionIR = sel
            if not allow_reuse:
                sum_loops: List[IndexVarIR] = [ix for ix in ci if isinstance(ix, IndexVarIR)]
                if sum_loops:
                    sum_ranges: Dict[DefId, RangeIR] = {}
                    cvr = dict(clause.variable_ranges or {})
                    for lv in sum_loops:
                        did = getattr(lv, "defid", None)
                        if did is not None and did in cvr:
                            sum_ranges[did] = cvr[did]
                    out_val = ReductionExpressionIR(ReductionOp.SUM, sum_loops, sel, loc, loop_var_ranges=sum_ranges if sum_ranges else None, type_info=_ti(val), shape_info=None)
            dc.append(EinsteinClauseIR(indices=dvars, value=out_val, location=clause.location, where_clause=clause.where_clause, variable_ranges=nvr))
            continue
        if val.operation == ReductionOp.PROD and allow_reuse:
            exc = [BinaryOpIR(BinaryOp.NE, first_wi[p], dvars[p], loc, type_info=BOOL) for p in range(len(first_wi)) if getattr(dvars[p], "defid", None) in new_dids]
            ow = getattr(val, "where_clause", None)
            oc = list(getattr(ow, "constraints", None) or []) if ow else []
            pw = WhereClauseIR(constraints=oc + exc, location=loc) if (oc or exc) else None
            pr = ReductionExpressionIR(ReductionOp.PROD, lvs, val.body, loc, where_clause=pw, loop_var_ranges=mlr, type_info=_ti(val), shape_info=_si(val))
            nvr = dict(clause.variable_ranges or {})
            nvr.update(new_vr)
            dc.append(EinsteinClauseIR(indices=dvars, value=pr, location=clause.location, where_clause=clause.where_clause, variable_ranges=nvr))
            continue
        if val.operation != ReductionOp.SUM:
            d_val = val.accept(JacobianVisitor(wrt, loc, B, R, wrt_tangent=wt, stmt_partial=sp, primal_subst=ps, shared_cotangent_axes=clause_shared_axes, full_cotangent_axes=shared_axes, dependency_cache=dep_cache))
            mi = list(clause.indices or [])
            nvr = dict(clause.variable_ranges or {})
            mi, nvr = _ensure_cotangent_axes_on_clause_indices(mi, nvr, d_val, clause_shared_axes)
            mi, nvr = _append_cotangent_axes_to_clause_indices(mi, nvr, clause_shared_axes)
            dc.append(EinsteinClauseIR(indices=mi, value=d_val, location=clause.location, where_clause=clause.where_clause, variable_ranges=nvr))
            continue
        orig_rc: List[ExpressionIR] = list(getattr(val.where_clause, "constraints", None) or []) if val.where_clause else []
        rterms: List[ExpressionIR] = []
        for pos in wrt_pos:
            _, wi = factors[pos]
            others = [factors[i] for i in range(len(factors)) if i != pos]
            deltas = [BinaryOpIR(BinaryOp.EQ, wi[p], dvars[p], loc, type_info=BOOL) for p in range(len(wi)) if getattr(dvars[p], "defid", None) in new_dids]
            wh = WhereClauseIR(constraints=orig_rc + deltas, location=loc) if (orig_rc or deltas) else None
            if not others:
                body: ExpressionIR = _fl(1, loc)
            else:
                def _mkref(a: ExpressionIR, idxs: List) -> ExpressionIR:
                    bf = B.get(a.defid) if isinstance(a, IdentifierIR) else None
                    nm = a.name or (bf.name if bf else "") or ""
                    ref = IdentifierIR(nm, loc, a.defid, type_info=_ti(a), shape_info=_si(a)) if isinstance(a, IdentifierIR) else a
                    return RectangularAccessIR(ref, list(idxs), loc)
                body = _mkref(*others[0])
                for ae, il in others[1:]:
                    body = BinaryOpIR(BinaryOp.MUL, body, _mkref(ae, il), loc, type_info=_ti(val) or F32, shape_info=_si(val))
            rterms.append(ReductionExpressionIR(val.operation, lvs, body, loc, where_clause=wh, loop_var_ranges=mlr, type_info=_ti(val), shape_info=_si(val)))
        cv: ExpressionIR = rterms[0]
        for r in rterms[1:]:
            cv = BinaryOpIR(BinaryOp.ADD, cv, r, loc, type_info=_ti(val) or F32, shape_info=_si(val))
        ni = dvars if allow_reuse else (ci + dvars)
        nvr = dict(clause.variable_ranges or {})
        nvr.update(new_vr)
        dc.append(EinsteinClauseIR(indices=ni, value=cv, location=clause.location, where_clause=clause.where_clause, variable_ranges=nvr))
    if not dc:
        return _z(loc)
    dc = [
        EinsteinClauseIR(indices=c.indices, value=_simplify(c.value, loc), location=c.location, where_clause=c.where_clause, variable_ranges=dict(c.variable_ranges or {}))
        for c in dc
    ]
    return EinsteinIR(clauses=dc, shape=None, element_type=expr.element_type, location=expr.location, type_info=expr.type_info, shape_info=None)
