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
    _sub,
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
from ._jacobian import JacobianVisitor
from ._print import _fmt_print_msg, _idx_str, _str_ir, _str_ir_print_differential_rhs
from ._pullback import _shape_dim_expr, build_default_seed, build_seeded_pullback
from ._tensor import _flatten_rect_access, _tensor_rank_from_binding, _tensor_rank_from_expr
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
from ...shared.debug_trace import emit_debug_log
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
        node_ti = getattr(n, "type_info", None)
        node_is_rect = isinstance(node_ti, RectangularType) or isinstance(self._ti, RectangularType)
        if hasattr(n, "shape_info") and n.shape_info is None and self._si is not None and node_is_rect:
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
                    bsi = self._si if isinstance(bti, RectangularType) else None
                    s.expr.accept(_TypePropagator(bti, bsi))
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


def _set_result_metadata(expr: Any, ti: Any, si: Any) -> None:
    if expr is None:
        return
    if ti is not None and hasattr(expr, "type_info"):
        expr.type_info = ti
    if hasattr(expr, "shape_info"):
        expr.shape_info = si if isinstance(ti, RectangularType) else None


def _bindings_in(block: Any, program: Optional[ProgramIR] = None) -> List[BindingIR]:
    if block is program or isinstance(block, ProgramIR):
        return [b for b in (program.bindings or []) if isinstance(b, BindingIR)] if program else []
    if isinstance(block, BlockExpressionIR):
        return [s for s in (block.statements or []) if isinstance(s, BindingIR)]
    return []


def _rect_access_result_meta(expr: RectangularAccessIR, bindings: Optional[Dict[DefId, BindingIR]] = None) -> tuple:
    binding_map = bindings or {}
    base_rank = _tensor_rank_from_expr(expr.array, binding_map)
    if base_rank <= 0:
        return _ti(expr), _si(expr)
    result_rank = max(0, base_rank - len(expr.indices or []))
    if result_rank <= 0:
        arr_ti = _ti(expr.array)
        if isinstance(arr_ti, RectangularType):
            return getattr(arr_ti, "element_type", None) or _ti(expr), None
        expr_ti = _ti(expr)
        if isinstance(expr_ti, RectangularType):
            return getattr(expr_ti, "element_type", None) or expr_ti, None
        return expr_ti, None
    si = _si(expr)
    if isinstance(si, tuple) and len(si) >= result_rank:
        si = tuple(si[:result_rank])
    return _ti(expr), si


def _binding_shape_info(binding: Optional[BindingIR], bindings: Optional[Dict[DefId, BindingIR]] = None) -> Any:
    if binding is None:
        return None
    expr = getattr(binding, "expr", None)
    if isinstance(expr, RectangularAccessIR):
        _, si = _rect_access_result_meta(expr, bindings)
        if si is not None:
            return si
    return _si(binding) or _si(getattr(binding, "expr", None))


def _binding_type_info(binding: Optional[BindingIR], bindings: Optional[Dict[DefId, BindingIR]] = None) -> Any:
    if binding is None:
        return None
    expr = getattr(binding, "expr", None)
    if isinstance(expr, RectangularAccessIR):
        ti, _ = _rect_access_result_meta(expr, bindings)
        if ti is not None:
            return ti
    return _ti(binding) or _ti(getattr(binding, "expr", None))


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


def _unwrap_block_final_expr(expr: ExpressionIR) -> ExpressionIR:
    cur = expr
    while isinstance(cur, BlockExpressionIR) and cur.final_expr is not None:
        cur = cur.final_expr
    return cur


def _resolve_tensor_alias_projection(
    did: DefId,
    bindings: Dict[DefId, BindingIR],
) -> Optional[tuple]:
    seen: Set[DefId] = set()
    indices: List[ExpressionIR] = []
    cur_did: Optional[DefId] = did
    while cur_did is not None and cur_did not in seen:
        seen.add(cur_did)
        binding = bindings.get(cur_did)
        if binding is None or binding.expr is None:
            return None
        expr = binding.expr
        while isinstance(expr, CastExpressionIR):
            expr = expr.expr
        if isinstance(expr, RectangularAccessIR):
            root_expr, full_indices = _flatten_rect_access(expr)
            indices = list(full_indices) + indices
            if not isinstance(root_expr, IdentifierIR) or root_expr.defid is None:
                return None
            cur_did = root_expr.defid
            continue
        if isinstance(expr, IdentifierIR) and expr.defid is not None:
            cur_did = expr.defid
            continue
        if indices:
            return cur_did, indices, binding
        return None
    return None


def _project_tensor_expr_at_indices(
    expr: ExpressionIR,
    indices: List[ExpressionIR],
    loc: SourceLocation,
) -> ExpressionIR:
    expr = _unwrap_block_final_expr(expr)
    if isinstance(expr, EinsteinIR) and expr.clauses and len(expr.clauses) == 1:
        clause = expr.clauses[0]
        clause_indices = list(clause.indices or [])
        if len(clause_indices) == len(indices):
            sub_map: Dict[DefId, ExpressionIR] = {}
            for src_idx, target_idx in zip(clause_indices, indices):
                did = getattr(src_idx, "defid", None)
                if did is not None:
                    sub_map[did] = target_idx
            if clause.value is not None:
                return _simplify(_sub(clause.value, sub_map, loc), loc)
    if isinstance(expr, BlockExpressionIR) and expr.final_expr is not None:
        projected = _project_tensor_expr_at_indices(expr.final_expr, indices, loc)
        if not expr.statements:
            return projected
        return BlockExpressionIR(list(expr.statements or []), expr.location or loc, projected, type_info=_ti(projected), shape_info=_si(projected))
    if isinstance(expr, BinaryOpIR):
        return _simplify(
            BinaryOpIR(
                expr.operator,
                _project_tensor_expr_at_indices(expr.left, indices, loc),
                _project_tensor_expr_at_indices(expr.right, indices, loc),
                expr.location or loc,
                type_info=_ti(expr),
                shape_info=None,
            ),
            loc,
        )
    if isinstance(expr, UnaryOpIR):
        return UnaryOpIR(
            expr.operator,
            _project_tensor_expr_at_indices(expr.operand, indices, loc),
            expr.location or loc,
            type_info=_ti(expr),
            shape_info=None,
        )
    if isinstance(expr, CastExpressionIR):
        return CastExpressionIR(
            _project_tensor_expr_at_indices(expr.expr, indices, loc),
            expr.target_type,
            expr.location or loc,
            type_info=_ti(expr),
            shape_info=None,
        )
    if isinstance(expr, IfExpressionIR):
        return IfExpressionIR(
            expr.condition,
            _project_tensor_expr_at_indices(expr.then_expr, indices, loc),
            expr.location or loc,
            else_expr=_project_tensor_expr_at_indices(expr.else_expr, indices, loc) if expr.else_expr is not None else None,
            type_info=_ti(expr),
            shape_info=None,
        )
    if isinstance(expr, ReductionExpressionIR):
        return ReductionExpressionIR(
            expr.operation,
            list(expr.loop_vars or []),
            _project_tensor_expr_at_indices(expr.body, indices, loc),
            expr.location or loc,
            where_clause=expr.where_clause,
            loop_var_ranges=expr.loop_var_ranges,
            type_info=_ti(expr),
            shape_info=None,
        )
    return RectangularAccessIR(expr, list(indices), loc, type_info=_ti(expr), shape_info=None)


def _collapse_tensor_pullback_to_rank(
    expr: ExpressionIR,
    target_rank: int,
    bindings: Dict[DefId, BindingIR],
    loc: SourceLocation,
) -> ExpressionIR:
    expr_rank = _tensor_rank_from_expr(expr, bindings)
    if isinstance(expr, BlockExpressionIR) and expr.final_expr is not None:
        final_expr = _collapse_tensor_pullback_to_rank(expr.final_expr, target_rank, bindings, loc)
        return BlockExpressionIR(list(expr.statements or []), expr.location or loc, final_expr, type_info=_ti(expr), shape_info=_si(final_expr))
    if isinstance(expr, BinaryOpIR):
        return _simplify(
            BinaryOpIR(
                expr.operator,
                _collapse_tensor_pullback_to_rank(expr.left, target_rank, bindings, loc),
                _collapse_tensor_pullback_to_rank(expr.right, target_rank, bindings, loc),
                expr.location or loc,
                type_info=_ti(expr),
                shape_info=None,
            ),
            loc,
        )
    if isinstance(expr, UnaryOpIR):
        return UnaryOpIR(
            expr.operator,
            _collapse_tensor_pullback_to_rank(expr.operand, target_rank, bindings, loc),
            expr.location or loc,
            type_info=_ti(expr),
            shape_info=None,
        )
    if isinstance(expr, CastExpressionIR):
        return CastExpressionIR(
            _collapse_tensor_pullback_to_rank(expr.expr, target_rank, bindings, loc),
            expr.target_type,
            expr.location or loc,
            type_info=_ti(expr),
            shape_info=None,
        )
    if isinstance(expr, IfExpressionIR):
        return IfExpressionIR(
            expr.condition,
            _collapse_tensor_pullback_to_rank(expr.then_expr, target_rank, bindings, loc),
            expr.location or loc,
            else_expr=_collapse_tensor_pullback_to_rank(expr.else_expr, target_rank, bindings, loc) if expr.else_expr is not None else None,
            type_info=_ti(expr),
            shape_info=None,
        )
    expr = _unwrap_block_final_expr(expr)
    if isinstance(expr, EinsteinIR) and expr.clauses and len(expr.clauses) == 1:
        clause = expr.clauses[0]
        indices = list(clause.indices or [])
        expr_rank = max(expr_rank, len(indices))
        if expr_rank <= target_rank:
            return expr
        extra = expr_rank - target_rank
        if extra > 0 and extra <= len(indices) and clause.value is not None:
            red_indices = [idx for idx in indices[:extra] if isinstance(idx, (IndexVarIR, IdentifierIR))]
            kept_indices = indices[extra:]
            body = clause.value
            if red_indices:
                red_ranges = {
                    did: rng
                    for did, rng in (clause.variable_ranges or {}).items()
                    if did in {idx.defid for idx in red_indices if getattr(idx, "defid", None) is not None}
                }
                body = ReductionExpressionIR(
                    ReductionOp.SUM,
                    red_indices,
                    body,
                    loc,
                    loop_var_ranges=red_ranges,
                    type_info=_ti(body),
                    shape_info=None,
                )
            kept_ranges = {
                did: rng
                for did, rng in (clause.variable_ranges or {}).items()
                if did in {idx.defid for idx in kept_indices if getattr(idx, "defid", None) is not None}
            }
            return EinsteinIR(
                clauses=[EinsteinClauseIR(indices=kept_indices, value=body, location=clause.location, where_clause=clause.where_clause, variable_ranges=kept_ranges)],
                shape=None,
                element_type=expr.element_type,
                location=expr.location or loc,
                type_info=_ti(expr),
                shape_info=None,
            )
        if clause.value is not None and len(indices) > target_rank:
            body = clause.value
            kept_indices = list(indices)
            kept_ranges = dict(clause.variable_ranges or {})
            while len(kept_indices) > target_rank:
                drop_pos = None
                for pos in range(len(kept_indices) - 1, -1, -1):
                    idx = kept_indices[pos]
                    rng = kept_ranges.get(getattr(idx, "defid", None))
                    end = getattr(rng, "end", None)
                    if isinstance(end, LiteralIR):
                        try:
                            if int(end.value) == 1:
                                drop_pos = pos
                                break
                        except (TypeError, ValueError):
                            pass
                if drop_pos is None:
                    break
                idx = kept_indices.pop(drop_pos)
                did = getattr(idx, "defid", None)
                if did is not None:
                    kept_ranges.pop(did, None)
                    body = _sub(body, {did: LiteralIR(0, loc, type_info=I32)}, loc)
            if len(kept_indices) == target_rank:
                return EinsteinIR(
                    clauses=[EinsteinClauseIR(indices=kept_indices, value=body, location=clause.location, where_clause=clause.where_clause, variable_ranges=kept_ranges)],
                    shape=None,
                    element_type=expr.element_type,
                    location=expr.location or loc,
                    type_info=_ti(expr),
                    shape_info=None,
                )
    if expr_rank <= target_rank:
        return expr
    return expr


def _alias_indices_are_static_scalar(
    indices: List[ExpressionIR],
    bindings: Dict[DefId, BindingIR],
) -> bool:
    for idx in indices:
        if isinstance(idx, LiteralIR):
            continue
        did = getattr(idx, "defid", None)
        if did is None:
            return False
        binding = bindings.get(did)
        if binding is None or binding.expr is None:
            return False
        if _tensor_rank_from_expr(binding.expr, bindings) > 0:
            return False
    return True


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

    def _seed_binding_for_numerator(self, expr: Optional[ExpressionIR]) -> Optional[BindingIR]:
        if not isinstance(expr, IdentifierIR) or expr.defid is None:
            return None
        return self._SB.get(expr.defid)

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
                num_rank = _tensor_rank_from_expr(ne, self._SB)
                den_rank = _tensor_rank_from_binding(den_binding, self._SB)
                num_alias_projection = (
                    _resolve_tensor_alias_projection(nop.defid, self._SB)
                    if isinstance(nop, IdentifierIR) and nop.defid is not None
                    else None
                )
                den_alias_projection = _resolve_tensor_alias_projection(dd, self._SB)
                emit_debug_log(
                    "autodiff.quotient",
                    "_expand.py:visit_binary_op",
                    "expand_identifier_denominator_quotient",
                    {
                        "numerator_name": getattr(nop, "name", None),
                        "denominator_name": getattr(dop, "name", None),
                        "num_rank": num_rank,
                        "den_rank": den_rank,
                        "cdn": cdn,
                        "reachable": rch,
                        "num_alias_projection": num_alias_projection is not None,
                        "den_alias_projection": den_alias_projection is not None,
                    },
                )
                if num_alias_projection is not None and den_alias_projection is not None and num_rank > 0:
                    num_root, num_alias_indices, _ = num_alias_projection
                    den_root, den_alias_indices, _ = den_alias_projection
                    num_root_binding = self._SB.get(num_root)
                    if num_root_binding is not None and num_root_binding.expr is not None:
                        der_root = build_seeded_pullback(
                            num_root_binding.expr,
                            build_default_seed(
                                num_root_binding.expr,
                                self._SB,
                                self._R,
                                ql,
                                numerator_binding=num_root_binding,
                            ),
                            den_root,
                            self._SB,
                            self._R,
                            ql,
                            dependency_cache=self._dep_cache,
                        )
                        projected = RectangularAccessIR(
                            der_root,
                            list(num_alias_indices) + list(den_alias_indices),
                            ql,
                            type_info=_ti(n),
                            shape_info=_si(n),
                        )
                        if _ti(n) is not None:
                            _set_result_metadata(projected, _ti(n), _si(n))
                            _propagate_ti(projected, _ti(n), None)
                        return projected
                if num_rank == 0 and den_alias_projection is not None:
                    den_root, den_alias_indices, _ = den_alias_projection
                    try:
                        der_alias = _simplify(ne.accept(JacobianVisitor(dd, ql, self._SB, self._R, dependency_cache=self._dep_cache)), ql)
                    except Exception:
                        der_alias = None
                    if der_alias is not None and _tensor_rank_from_expr(der_alias, self._SB) <= 0:
                        der = der_alias
                        ti = _ti(n) or _binding_type_info(den_binding, self._SB)
                        if ti is not None:
                            _set_result_metadata(der, ti, _si(n))
                            _propagate_ti(der, ti, None)
                        return der
                    num_b = self._seed_binding_for_numerator(nop)
                    seed = build_default_seed(ne, self._SB, self._R, ql, numerator_binding=num_b)
                    der_root = build_seeded_pullback(
                        ne,
                        seed,
                        den_root,
                        self._SB,
                        self._R,
                        ql,
                        dependency_cache=self._dep_cache,
                    )
                    der_root_rank = _tensor_rank_from_expr(der_root, self._SB)
                    if der_root_rank <= 0 or not den_alias_indices:
                        der = der_root
                    elif _alias_indices_are_static_scalar(list(den_alias_indices), self._SB):
                        tmp_did = self._R.allocate_for_local() if self._R is not None else None
                        tmp_ref = IdentifierIR(
                            "__alias_root",
                            ql,
                            tmp_did,
                            type_info=_ti(der_root),
                            shape_info=_si(der_root),
                        )
                        tmp_binding = BindingIR(
                            name="__alias_root",
                            expr=der_root,
                            location=ql,
                            defid=tmp_did,
                            type_info=_ti(der_root),
                        )
                        projected = RectangularAccessIR(
                            tmp_ref,
                            list(den_alias_indices),
                            ql,
                            type_info=_binding_type_info(den_binding, self._SB) or _ti(n),
                            shape_info=_binding_shape_info(den_binding, self._SB) or _si(n),
                        )
                        der = BlockExpressionIR([tmp_binding], ql, projected, type_info=_ti(projected), shape_info=_si(projected))
                    else:
                        der = _project_tensor_expr_at_indices(der_root, list(den_alias_indices), ql)
                    der = _simplify(der, ql)
                    ti = _ti(n) or _binding_type_info(den_binding, self._SB)
                    if ti is not None:
                        _set_result_metadata(der, ti, _si(n))
                        _propagate_ti(der, ti, None)
                    return der
                if den_rank > 0:
                    alias_projection = den_alias_projection
                    result_binding = den_binding
                    num_b = self._seed_binding_for_numerator(nop)
                    seed = build_default_seed(ne, self._SB, self._R, ql, numerator_binding=num_b)
                    use_alias_projection = (
                        alias_projection is not None
                        and den_binding is not None
                    )
                    if use_alias_projection:
                        den_root, alias_indices, _ = alias_projection
                        der_root = build_seeded_pullback(
                            ne,
                            seed,
                            den_root,
                            self._SB,
                            self._R,
                            ql,
                            dependency_cache=self._dep_cache,
                        )
                        der = RectangularAccessIR(
                            der_root,
                            alias_indices,
                            ql,
                            type_info=_binding_type_info(den_binding, self._SB),
                            shape_info=_binding_shape_info(den_binding, self._SB),
                        )
                    else:
                        der = build_seeded_pullback(
                            ne,
                            seed,
                            dd,
                            self._SB,
                            self._R,
                            ql,
                            dependency_cache=self._dep_cache,
                        )
                    der = _collapse_tensor_pullback_to_rank(der, den_rank, self._SB, ql)
                    ti = _binding_type_info(result_binding, self._SB) or _ti(n)
                    si = _binding_shape_info(result_binding, self._SB)
                    _set_result_metadata(der, ti, si)
                    if ti is not None:
                        _propagate_ti(der, ti, None)
                    return der
                jv = JacobianVisitor(dd, ql, self._SB, self._R, dependency_cache=self._dep_cache)
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
                    _set_result_metadata(der, ti, _si(n))
                    _propagate_ti(der, ti, None)
                return der
            dids = self._dep_cache.collect_defids(dop)
            if len(dids) != 1:
                raise ValueError("Autodiff: @num/@(expr) denominator depends on != 1 variable")
            wd = next(iter(dids))
            den_b = self._SB.get(wd)
            ne = self._SE.get(nop.defid) if isinstance(nop, IdentifierIR) and nop.defid is not None else nop
            if ne is None:
                raise ValueError("Autodiff: numerator has no defining expr")
            cdn = wd in self._dep_cache.collect_defids(ne)
            rch = False
            if isinstance(nop, IdentifierIR) and nop.defid is not None:
                rch = _is_reachable_with_cache(nop.defid, wd, self._SB, self._dep_cache)
                if not rch and den_b is not None and den_b.expr is not None:
                    t_root = _rectangular_read_root_defid(den_b.expr)
                    if t_root is not None:
                        rch = _is_reachable_with_cache(nop.defid, t_root, self._SB, self._dep_cache)
            if not cdn and not rch:
                return _z(ql)
            num_b = self._seed_binding_for_numerator(nop)
            seed = build_default_seed(ne, self._SB, self._R, ql, numerator_binding=num_b)
            den_alias_projection = _resolve_tensor_alias_projection(wd, self._SB)
            use_alias_projection = den_alias_projection is not None and den_b is not None
            emit_debug_log(
                "autodiff.quotient",
                "_expand.py:visit_binary_op",
                "expand_expression_denominator_quotient",
                {
                    "numerator_name": getattr(nop, "name", None),
                    "denominator_defid": str(wd),
                    "reachable": rch,
                    "cdn": cdn,
                    "use_alias_projection": use_alias_projection,
                    "has_den_binding": den_b is not None,
                },
            )
            if use_alias_projection:
                den_root, alias_indices, _ = den_alias_projection
                der_root = build_seeded_pullback(
                    ne,
                    seed,
                    den_root,
                    self._SB,
                    self._R,
                    ql,
                    dependency_cache=self._dep_cache,
                )
                if _tensor_rank_from_expr(der_root, self._SB) <= 0 or not alias_indices:
                    der = der_root
                else:
                    der = RectangularAccessIR(
                        der_root,
                        alias_indices,
                        ql,
                        type_info=_binding_type_info(den_b, self._SB),
                        shape_info=_binding_shape_info(den_b, self._SB),
                    )
            else:
                der = build_seeded_pullback(
                    ne,
                    seed,
                    wd,
                    self._SB,
                    self._R,
                    ql,
                    dependency_cache=self._dep_cache,
                )
            den_rank = _tensor_rank_from_binding(den_b, self._SB)
            der = _collapse_tensor_pullback_to_rank(der, den_rank, self._SB, ql)
            ti = _binding_type_info(den_b, self._SB) or _ti(n)
            si = _binding_shape_info(den_b, self._SB) if den_b is not None else _si(n)
            if ti is not None:
                _set_result_metadata(der, ti, si)
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


class _ShapeAccessFolder(_Rewriter):
    def __init__(self, bindings: Dict[DefId, Any], loc: SourceLocation) -> None:
        super().__init__(loc)
        self._B = bindings

    def visit_identifier(self, n: IdentifierIR) -> ExpressionIR:
        return n

    def visit_literal(self, n: LiteralIR) -> ExpressionIR:
        return n

    def visit_index_var(self, n: IndexVarIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_index_rest(self, n: IndexRestIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_member_access(self, n: MemberAccessIR) -> ExpressionIR:
        obj = n.object.accept(self) if isinstance(n.object, ExpressionIR) else n.object
        return MemberAccessIR(obj, n.member, n.location or self._loc, type_info=_ti(n), shape_info=_si(n))

    def visit_rectangular_access(self, n: RectangularAccessIR) -> ExpressionIR:
        arr = n.array.accept(self) if isinstance(n.array, ExpressionIR) else n.array
        idxs = [idx.accept(self) if isinstance(idx, ExpressionIR) else idx for idx in (n.indices or [])]
        if (
            isinstance(arr, MemberAccessIR)
            and arr.member == "shape"
            and len(idxs) == 1
            and isinstance(idxs[0], LiteralIR)
            and isinstance(arr.object, ExpressionIR)
        ):
            try:
                dim_index = int(idxs[0].value)
                return _shape_dim_expr(arr.object, dim_index, n.location or self._loc, self._B)
            except (TypeError, ValueError):
                pass
        return RectangularAccessIR(arr, idxs, n.location or self._loc, type_info=_ti(n), shape_info=_si(n))

    def visit_binary_op(self, n: BinaryOpIR) -> ExpressionIR:
        return BinaryOpIR(
            n.operator,
            n.left.accept(self),
            n.right.accept(self),
            n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_unary_op(self, n: UnaryOpIR) -> ExpressionIR:
        return UnaryOpIR(n.operator, n.operand.accept(self), n.location or self._loc, type_info=_ti(n), shape_info=_si(n))

    def visit_cast_expression(self, n: CastExpressionIR) -> ExpressionIR:
        return CastExpressionIR(n.expr.accept(self), n.target_type, n.location or self._loc, type_info=_ti(n), shape_info=_si(n))

    def visit_if_expression(self, n: IfExpressionIR) -> ExpressionIR:
        return IfExpressionIR(
            n.condition.accept(self),
            n.then_expr.accept(self),
            n.location or self._loc,
            else_expr=n.else_expr.accept(self) if n.else_expr is not None else None,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_function_call(self, n: FunctionCallIR) -> ExpressionIR:
        return FunctionCallIR(
            callee_expr=n.callee_expr,
            location=n.location or self._loc,
            arguments=[a.accept(self) for a in (n.arguments or [])],
            module_path=n.module_path,
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

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        local_bindings = dict(self._B)
        stmts: List[Any] = []
        for stmt in n.statements or []:
            if isinstance(stmt, BindingIR):
                expr = stmt.expr.accept(_ShapeAccessFolder(local_bindings, stmt.location or self._loc)) if stmt.expr is not None else None
                nb = BindingIR(stmt.name, expr, type_info=_ti(stmt), location=stmt.location, defid=stmt.defid)
                stmts.append(nb)
                if nb.defid is not None:
                    local_bindings[nb.defid] = nb
            elif isinstance(stmt, ExpressionIR):
                stmts.append(stmt.accept(_ShapeAccessFolder(local_bindings, getattr(stmt, "location", None) or self._loc)))
            else:
                stmts.append(stmt)
        final_expr = n.final_expr.accept(_ShapeAccessFolder(local_bindings, n.location or self._loc)) if n.final_expr is not None else None
        return BlockExpressionIR(stmts, n.location or self._loc, final_expr, type_info=_ti(n), shape_info=_si(n))

    def visit_einstein(self, n: EinsteinIR) -> ExpressionIR:
        clauses: List[EinsteinClauseIR] = []
        for c in n.clauses or []:
            vr = {}
            for did, rng in (c.variable_ranges or {}).items():
                if isinstance(rng, RangeIR):
                    vr[did] = RangeIR(rng.start.accept(self), rng.end.accept(self), rng.location or self._loc, type_info=getattr(rng, "type_info", None))
                else:
                    vr[did] = rng
            clauses.append(
                EinsteinClauseIR(
                    indices=list(c.indices or []),
                    value=c.value.accept(self) if c.value is not None else None,
                    location=c.location or self._loc,
                    where_clause=c.where_clause,
                    variable_ranges=vr,
                )
            )
        return EinsteinIR(clauses=clauses, shape=n.shape, element_type=n.element_type, location=n.location or self._loc, type_info=_ti(n), shape_info=_si(n))

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> ExpressionIR:
        vr = {}
        for did, rng in (n.loop_var_ranges or {}).items():
            if isinstance(rng, RangeIR):
                vr[did] = RangeIR(rng.start.accept(self), rng.end.accept(self), rng.location or self._loc, type_info=getattr(rng, "type_info", None))
            else:
                vr[did] = rng
        return ReductionExpressionIR(
            n.operation,
            list(n.loop_vars or []),
            n.body.accept(self),
            n.location or self._loc,
            where_clause=n.where_clause,
            loop_var_ranges=vr,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> ExpressionIR:
        vr = {}
        for did, rng in (n.loop_var_ranges or {}).items():
            if isinstance(rng, RangeIR):
                vr[did] = RangeIR(rng.start.accept(self), rng.end.accept(self), rng.location or self._loc, type_info=getattr(rng, "type_info", None))
            else:
                vr[did] = rng
        return SelectAtArgmaxIR(
            n.primal_body.accept(self) if n.primal_body is not None else None,
            n.diff_body.accept(self) if n.diff_body is not None else None,
            list(n.loop_vars or []),
            loop_var_ranges=vr,
            location=n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
            use_argmin=getattr(n, "use_argmin", False),
        )

    def visit_differential(self, n: DifferentialIR) -> ExpressionIR:
        return n

    def visit_array_comprehension(self, n: ArrayComprehensionIR) -> ExpressionIR:
        return n

    def visit_array_literal(self, n: ArrayLiteralIR) -> ExpressionIR:
        return n

    def visit_tuple_expression(self, n: TupleExpressionIR) -> ExpressionIR:
        return n

    def visit_tuple_access(self, n: TupleAccessIR) -> ExpressionIR:
        return n

    def visit_interpolated_string(self, n: InterpolatedStringIR) -> ExpressionIR:
        return n

    def visit_try_expression(self, n: TryExpressionIR) -> ExpressionIR:
        return n

    def visit_match_expression(self, n: MatchExpressionIR) -> ExpressionIR:
        return n

    def visit_where_expression(self, n: WhereExpressionIR) -> ExpressionIR:
        return n

    def visit_pipeline_expression(self, n: PipelineExpressionIR) -> ExpressionIR:
        return n

    def visit_lambda(self, n: LambdaIR) -> ExpressionIR:
        return n

    def visit_function_value(self, n: FunctionValueIR) -> ExpressionIR:
        return n

    def visit_range(self, n: RangeIR) -> ExpressionIR:
        return cast(ExpressionIR, n)


def _fold_shape_accesses_program(
    program: ProgramIR,
    loc: SourceLocation,
    init_B: Optional[Dict[DefId, Any]] = None,
) -> None:
    sb: Dict[DefId, Any] = dict(init_B) if init_B else {}
    for b in program.bindings or []:
        if not isinstance(b, BindingIR) or b.expr is None:
            continue
        b.expr = b.expr.accept(_ShapeAccessFolder(sb, b.expr.location or loc))
        if b.defid is not None:
            sb[b.defid] = b


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
