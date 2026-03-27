from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, cast

from ._core import (
    USER_DIFF_PREFIX,
    _LOC0,
    _Rewriter,
    _fl,
    _is_diff_name,
    _is_zero,
    _reject_lowered_ir,
    _si,
    _simplify,
    _ti,
    _unwrap_trivial_einstein_rhs,
    _z,
)
from ._forward import DiffVisitor
from ._graph import (
    _DefIdCollector,
    _DependencyQueryCache,
    _autodiff_primal_data_defids,
    _collect_defids,
    _collect_targets_expr,
    _is_reachable_with_cache,
    _rectangular_read_root_defid,
)
from ._jacobian import JacobianVisitor, _is_direct_einstein_tensor_jacobian
from ._print import _fmt_print_msg, _idx_str, _str_ir, _str_ir_print_differential_rhs
from ._tensor import _tensor_rank_from_binding, _tensor_rank_from_expr
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
    is_function_binding,
)
from ...shared.defid import DefId
from ...shared.source_location import SourceLocation
from ...shared.types import BinaryOp, F32, I32, STR, RectangularType


class _TypePropagator(_DefIdCollector):
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
            for _, v in (c.variable_ranges or {}).items():
                if isinstance(v, RangeIR):
                    v.start.accept(self)
                    v.end.accept(self)
                elif isinstance(v, ExpressionIR):
                    v.accept(self)

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> None:
        self._stamp(n)
        n.body.accept(self)
        for _, v in (n.loop_var_ranges or {}).items():
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
        for _, v in (n.loop_var_ranges or {}).items():
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

    def visit_function_call(self, n: Any) -> None:
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


def _bindings_in(block: Any, program: Optional[ProgramIR] = None) -> List[BindingIR]:
    if block is program or isinstance(block, ProgramIR):
        return [b for b in (program.bindings or []) if isinstance(b, BindingIR)] if program else []
    if isinstance(block, BlockExpressionIR):
        return [s for s in (block.statements or []) if isinstance(s, BindingIR)]
    return []


def _primal_to_diff_map(bindings: List) -> Dict[DefId, BindingIR]:
    out: Dict[DefId, BindingIR] = {}
    lst = list(bindings or [])
    for i in range(len(lst) - 1):
        a, b = lst[i], lst[i + 1]
        if not isinstance(a, BindingIR) or not isinstance(b, BindingIR):
            continue
        if a.defid is None:
            continue
        bn = b.name or ""
        an2 = a.name or ""
        if _is_diff_name(bn) and (bn == "_@" + an2 or bn == "@" + an2):
            out[a.defid] = b
    return out


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


class _ExpansionVisitor(_Rewriter):
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
        if n.operator == BinaryOp.DIV and isinstance(n.left, DifferentialIR) and isinstance(n.right, DifferentialIR):
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
                num_rank = _tensor_rank_from_expr(ne, self._SB)
                den_rank = _tensor_rank_from_binding(den_binding)
                if isinstance(ne, FunctionCallIR) and den_is_tensor and num_rank > 0 and num_rank >= den_rank:
                    nL = n.left.accept(self)
                    nR = n.right.accept(self)
                    out = BinaryOpIR(BinaryOp.DIV, nL, nR, ql, type_info=_ti(n), shape_info=_si(n))
                    ti = _ti(n)
                    si = _si(n)
                    if ti or si:
                        _propagate_ti(out, ti, si)
                    return out
                prefer_full_jacobian = den_is_tensor and num_rank > 0 and num_rank >= den_rank and _is_direct_einstein_tensor_jacobian(ne, dd, self._SB, self._dep_cache)
                legacy_directional = den_is_tensor and num_rank > 0 and not prefer_full_jacobian
                jv = JacobianVisitor(dd, ql, self._SB, self._R, legacy_directional=legacy_directional, dependency_cache=self._dep_cache)
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
                    if isinstance(der, BlockExpressionIR):
                        for stmt in der.statements or []:
                            if not isinstance(stmt, BindingIR) or stmt.expr is None:
                                continue
                            local_ti = stmt.type_info or _ti(stmt.expr)
                            local_si = _si(stmt) or _si(stmt.expr)
                            if local_ti is not None:
                                _propagate_ti(stmt.expr, local_ti, local_si)
                        if der.final_expr is not None:
                            _propagate_ti(der.final_expr, ti, None)
                    else:
                        _propagate_ti(der, ti, None)
                return der
            dids = self._dep_cache.collect_defids(dop)
            if len(dids) != 1:
                raise ValueError("Autodiff: @num/@(expr) denominator depends on != 1 variable")
            wd = next(iter(dids))
            den_b = self._SB.get(wd)
            num_rank = _tensor_rank_from_expr(nop, self._SB)
            prefer_full_jacobian = den_b is not None and _tensor_rank_from_binding(den_b) > 0 and num_rank > 0 and num_rank >= _tensor_rank_from_binding(den_b) and _is_direct_einstein_tensor_jacobian(nop, wd, self._SB, self._dep_cache)
            legacy_directional = den_b is not None and _tensor_rank_from_binding(den_b) > 0 and num_rank > 0 and not prefer_full_jacobian
            dn = nop.accept(JacobianVisitor(wd, self._loc, self._SB, self._R, legacy_directional=legacy_directional, dependency_cache=self._dep_cache))
            dd_ = dop.accept(JacobianVisitor(wd, self._loc, self._SB, self._R, legacy_directional=legacy_directional, dependency_cache=self._dep_cache))
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
        return UnaryOpIR(n.operator, n.operand.accept(self), n.location, type_info=_ti(n), shape_info=_si(n))

    def visit_einstein(self, n: EinsteinIR) -> ExpressionIR:
        nc: List[EinsteinClauseIR] = []
        for c in n.clauses or []:
            cv = c.value.accept(self) if c.value is not None else None
            if cv is not None:
                cv = _unwrap_trivial_einstein_rhs(cv)
            nc.append(EinsteinClauseIR(indices=c.indices, value=cv, location=c.location, where_clause=c.where_clause, variable_ranges=dict(c.variable_ranges) if c.variable_ranges else {}))
        return EinsteinIR(clauses=nc, shape=n.shape, element_type=n.element_type, location=n.location, type_info=_ti(n), shape_info=_si(n))

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        blk = BlockExpressionIR(list(n.statements or []), n.location or self._loc, n.final_expr, type_info=_ti(n), shape_info=_si(n))
        _ensure_block_d(blk, self._SB, self._SE, self._D, self._loc, self._R)
        nsb = dict(self._SB)
        nse = dict(self._SE)
        child_dep_cache = self._dep_cache.fork(nsb)
        ns: List[Any] = []
        for s in blk.statements or []:
            if isinstance(s, BindingIR):
                v = _ExpansionVisitor(self._D, nsb, nse, self._loc, self._R, self._P, dependency_cache=child_dep_cache)
                ex = s.expr.accept(v) if s.expr is not None else None
                nb = BindingIR(name=s.name, expr=ex, location=s.location, defid=s.defid, type_info=_ti(s))
                if nb.defid is not None:
                    nsb[nb.defid] = nb
                    nse[nb.defid] = ex
                    child_dep_cache = child_dep_cache.fork(nsb)
                ns.append(nb)
            elif isinstance(s, ExpressionIR):
                v = _ExpansionVisitor(self._D, nsb, nse, self._loc, self._R, self._P, dependency_cache=child_dep_cache)
                ns.append(s.accept(v))
            else:
                ns.append(s)
        vfin = _ExpansionVisitor(self._D, nsb, nse, self._loc, self._R, self._P, dependency_cache=child_dep_cache)
        nf = blk.final_expr.accept(vfin) if blk.final_expr is not None else None
        return BlockExpressionIR(ns, blk.location, nf, type_info=_ti(n), shape_info=_si(n))

    def visit_function_call(self, n: Any) -> ExpressionIR:
        na = [a.accept(self) for a in (n.arguments or [])]
        return type(n)(callee_expr=n.callee_expr, location=n.location, arguments=na, module_path=getattr(n, "module_path", None), type_info=_ti(n), shape_info=_si(n))

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
                            diff_binding_defids: Set[DefId] = {
                                b.defid for b in dm.values()
                                if isinstance(b, BindingIR) and b.defid is not None
                            }
                            needed = _trans_deps(ye, self._SB, self._dep_cache)
                            porder: Dict[DefId, int] = {}
                            for idx, bb in enumerate(P.bindings or []):
                                if isinstance(bb, BindingIR) and bb.defid is not None and not _is_diff_name(bb.name or ""):
                                    if bb.defid not in porder:
                                        porder[bb.defid] = idx
                            for did in sorted((d for d in needed if d != yd), key=lambda d: porder.get(d, 10 ** 9)):
                                pb = self._SB.get(did)
                                if pb is None or pb.expr is None or isinstance(pb.expr, FunctionValueIR):
                                    continue
                                db = dm.get(did)
                                if db is None or db.expr is None:
                                    continue
                                se = _simplify(db.expr, n.location or self._loc)
                                se_deps = self._dep_cache.collect_defids(se)
                                if not (se_deps & diff_binding_defids):
                                    continue
                                nm = (pb.name if pb and getattr(pb, "name", None) else None) or "?"
                                pb_idx = ""
                                if isinstance(pb.expr, EinsteinIR):
                                    pb_cc = pb.expr.clauses or []
                                    if pb_cc and pb_cc[0].indices:
                                        pb_idx = ", ".join(_idx_str(i) for i in pb_cc[0].indices)
                                pre_lhs = "@" + nm + ("[" + pb_idx + "]" if pb_idx else "")
                                pre.append("let " + pre_lhs + " = " + _str_ir(se) + ";")
                        rhs = _str_ir_print_differential_rhs(dr, n.location or self._loc)
                        lhs = "@" + (yn or "?")
                        if isinstance(ye, EinsteinIR) and ye.clauses and len(ye.clauses) == 1:
                            idx_s = ", ".join(_idx_str(i) for i in (ye.clauses[0].indices or []))
                            if idx_s:
                                lhs += "[" + idx_s + "]"
                        msg = "\n".join(pre) + "\n" + _fmt_print_msg(lhs, rhs) if pre else _fmt_print_msg(lhs, rhs)
                        return BuiltinCallIR("print", [LiteralIR(msg, n.location, type_info=STR)], n.location, defid=getattr(n, "defid", None), type_info=_ti(n), shape_info=_si(n))
                    except (ValueError, KeyError):
                        pass
        na = [a.accept(self) for a in args]
        return BuiltinCallIR(n.builtin_name, na, n.location, defid=getattr(n, "defid", None), type_info=_ti(n), shape_info=_si(n))

    def visit_rectangular_access(self, n: RectangularAccessIR) -> ExpressionIR:
        return n

    def visit_if_expression(self, n: IfExpressionIR) -> ExpressionIR:
        nc = n.condition.accept(self)
        nt = n.then_expr.accept(self)
        ne = n.else_expr.accept(self) if n.else_expr is not None else None
        return IfExpressionIR(nc, nt, n.location or self._loc, else_expr=ne, type_info=_ti(n), shape_info=_si(n))

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


def _expand(
    expr: ExpressionIR,
    D: Dict[DefId, IdentifierIR],
    SB: Dict[DefId, Any],
    SE: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    R: Any = None,
    P: Optional[ProgramIR] = None,
) -> ExpressionIR:
    return expr.accept(_ExpansionVisitor(D, SB, SE, loc, R, P))


def _expand_program(
    program: ProgramIR,
    D: Dict[DefId, IdentifierIR],
    loc: SourceLocation,
    R: Any = None,
    init_B: Optional[Dict[DefId, Any]] = None,
    target_binding_defids: Optional[Set[DefId]] = None,
    target_statement_ids: Optional[Set[int]] = None,
) -> None:
    sb: Dict[DefId, Any] = dict(init_B) if init_B else {}
    se: Dict[DefId, ExpressionIR] = {d: b.expr for d, b in sb.items() if getattr(b, "expr", None) is not None}
    for b in program.bindings or []:
        if not isinstance(b, BindingIR) or b.expr is None:
            continue
        if target_binding_defids is None or b.defid in target_binding_defids:
            b.expr = _expand(b.expr, D, sb, se, b.expr.location or loc, R, program)
        if b.defid is not None:
            sb[b.defid] = b
            se[b.defid] = b.expr
    stmts = program.statements or []
    for i, s in enumerate(stmts):
        if not isinstance(s, BindingIR) and isinstance(s, ExpressionIR) and (target_statement_ids is None or id(s) in target_statement_ids):
            stmts[i] = _expand(s, D, sb, se, getattr(s, "location", None) or loc, R, program)


def _ensure_block_d(
    block: BlockExpressionIR,
    SB: Dict[DefId, Any],
    SE: Dict[DefId, ExpressionIR],
    D: Dict[DefId, IdentifierIR],
    loc: SourceLocation,
    R: Any,
) -> None:
    tgts, qps = _collect_targets_expr(block)
    if not tgts and not qps:
        return
    bb = _bindings_in(block, None)
    if not bb:
        return
    bd = {b.defid for b in bb if b.defid is not None}
    td: Set[DefId] = set()
    for did, _ in tgts:
        td.add(did)
    for n, d_ in qps:
        td.add(n)
        td.add(d_)
    td &= bd
    if not td:
        return
    bbd: Dict[DefId, BindingIR] = dict(SB)
    for b in bb:
        if b.defid is not None:
            bbd[b.defid] = b
    dep_cache = _DependencyQueryCache(bbd)
    b2d: Dict[DefId, Set[DefId]] = {}
    for b in bb:
        if b.defid is not None and b.expr is not None:
            b2d[b.defid] = set() if is_function_binding(b) else _autodiff_primal_data_defids(b.expr, bbd, dep_cache)
    reach: Set[DefId] = set(td)
    wk = list(reach)
    while wk:
        did = wk.pop()
        for dep in b2d.get(did) or []:
            if dep in bd and dep not in reach:
                reach.add(dep)
                wk.append(dep)
    fwd: List[BindingIR] = []
    seen: Set[DefId] = set()

    def vis(did: DefId) -> None:
        if did in seen or did not in reach:
            return
        seen.add(did)
        b = bbd.get(did)
        if b is None:
            return
        for dep in b2d.get(b.defid) or []:
            if dep in bd:
                vis(dep)
        fwd.append(b)

    for did in td:
        vis(did)
    if R is None:
        return
    qd = {d_ for _, d_ in qps}
    lvs = {did for did in reach if not (b2d.get(did) or set())}
    sv: Dict[DefId, int] = {}
    for b in fwd:
        if b.defid is None:
            continue
        if b.defid in qd:
            sv[b.defid] = 1
        elif b.defid in lvs and b.defid in td:
            sv[b.defid] = 1
        else:
            sv[b.defid] = 0
    dre: Dict[DefId, ExpressionIR] = {did: ref for did, ref in D.items()}
    upq = len(qps) > 0
    d2b: Dict[DefId, BindingIR] = {}
    for b in fwd:
        if b.defid is None or b.defid in D:
            continue
        bl = b.location or _LOC0
        if b.defid in sv and sv[b.defid] == 1:
            drhs = _fl(1, bl)
        elif upq and b.defid in lvs:
            drhs = _z(bl)
        else:
            for dep in b2d.get(b.defid) or []:
                if dep not in dre:
                    dre[dep] = _z(bl)
            drhs = _fwd_expr(b, dre, bbd, b2d, bl, R)
            drhs = _inline_drhs(drhs, bl)
        ti = _ti(b) or (_ti(b.expr) if b.expr else None)
        si = _si(b) or (_si(b.expr) if b.expr else None)
        dd = R.allocate_for_local()
        dn = USER_DIFF_PREFIX + (b.name or "")
        dr = IdentifierIR(dn, bl, dd, type_info=ti, shape_info=si)
        D[b.defid] = dr
        dre[b.defid] = dr
        _propagate_ti(drhs, ti, si)
        d2b[b.defid] = BindingIR(name=dn, expr=drhs, location=b.location, defid=dd, type_info=ti)
    ns: List = []
    for s in block.statements or []:
        ns.append(s)
        if isinstance(s, BindingIR) and s.defid is not None and s.defid in d2b:
            ns.append(d2b[s.defid])
    object.__setattr__(block, "statements", ns)


def _inline_drhs(rhs: ExpressionIR, loc: SourceLocation) -> ExpressionIR:
    if isinstance(rhs, BlockExpressionIR) and rhs.final_expr is not None:
        stmts = [s for s in (rhs.statements or []) if isinstance(s, BindingIR)]
        if len(stmts) == 1 and stmts[0].expr is not None and isinstance(rhs.final_expr, IdentifierIR) and rhs.final_expr.defid == stmts[0].defid:
            return stmts[0].expr
    return rhs


def _fwd_einstein(
    expr: EinsteinIR,
    dre: Dict[DefId, ExpressionIR],
    B: Dict[DefId, BindingIR],
    loc: SourceLocation,
    R: Any,
) -> ExpressionIR:
    vis = DiffVisitor(dre, loc, B, R)
    nc: List[EinsteinClauseIR] = []
    for c in expr.clauses or []:
        try:
            dv = _unwrap_trivial_einstein_rhs(_simplify(c.value.accept(vis), loc))
        except (ValueError, KeyError):
            continue
        if _is_zero(dv):
            continue
        nc.append(EinsteinClauseIR(indices=list(c.indices or []), value=dv, location=c.location, where_clause=c.where_clause, variable_ranges=dict(c.variable_ranges or {})))
    if not nc:
        return _z(loc)
    return EinsteinIR(clauses=nc, shape=expr.shape, element_type=expr.element_type, location=expr.location, type_info=_ti(expr), shape_info=_si(expr))


def _fwd_expr(
    b: BindingIR,
    dre: Dict[DefId, ExpressionIR],
    B: Dict[DefId, BindingIR],
    b2d: Dict[DefId, Set[DefId]],
    loc: SourceLocation,
    R: Any,
) -> ExpressionIR:
    expr = b.expr
    if expr is None or isinstance(expr, FunctionValueIR):
        return _z(loc)
    if isinstance(expr, EinsteinIR):
        return _fwd_einstein(expr, dre, B, loc, R)
    return expr.accept(DiffVisitor(dre, loc, B, R))
