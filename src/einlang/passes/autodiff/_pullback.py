from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

from ._core import (
    _fl,
    _is_zero,
    _prod_pullback_via_sum,
    _si,
    _simplify,
    _sub_wd,
    _ti,
    _unsupported_autodiff_ir,
    _unwrap_trivial_einstein_rhs,
    _z,
)
from ._expr import _prune_const_ifs_replayed
from ._graph import _DependencyQueryCache, _jacobian_rhs_depends_on_wrt
from ._tensor import (
    _flatten_rect_access,
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
    IdentifierIR,
    IfExpressionIR,
    IRVisitor,
    IndexVarIR,
    LiteralIR,
    MemberAccessIR,
    RangeIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    SelectAtArgmaxIR,
    UnaryOpIR,
    is_function_binding,
)
from ...shared.debug_trace import emit_debug_log
from ...shared.defid import DefId
from ...shared.source_location import SourceLocation
from ...shared.types import BOOL, F32, I32, UNKNOWN, BinaryOp, ReductionOp, UnaryOp


def _debug_index_labels(indices: Optional[List[ExpressionIR]]) -> List[Any]:
    out: List[Any] = []
    for idx in indices or []:
        name = getattr(idx, "name", None)
        if name is not None:
            out.append(name)
            continue
        if isinstance(idx, LiteralIR):
            out.append(idx.value)
            continue
        out.append(type(idx).__name__)
    return out


def _loop_identity(loop_var: ExpressionIR) -> tuple:
    name = getattr(loop_var, "name", None)
    if name is not None:
        return ("name", name)
    defid = getattr(loop_var, "defid", None)
    if defid is not None:
        return ("defid", defid.krate, defid.index)
    return ("type", type(loop_var).__name__)


def _is_zero_like_expr(expr: Optional[ExpressionIR]) -> bool:
    if expr is None:
        return False
    if _is_zero(expr):
        return True
    if isinstance(expr, CastExpressionIR):
        return _is_zero_like_expr(expr.expr)
    if isinstance(expr, UnaryOpIR) and expr.operator in (UnaryOp.POS, UnaryOp.NEG):
        return _is_zero_like_expr(expr.operand)
    if isinstance(expr, BlockExpressionIR) and expr.final_expr is not None:
        return _is_zero_like_expr(expr.final_expr)
    if isinstance(expr, ReductionExpressionIR):
        return _is_zero_like_expr(expr.body)
    if isinstance(expr, EinsteinIR):
        clauses = list(expr.clauses or [])
        return bool(clauses) and all(_is_zero_like_expr(c.value) for c in clauses)
    if isinstance(expr, BinaryOpIR) and expr.operator in (BinaryOp.ADD, BinaryOp.SUB):
        if _is_zero_like_expr(expr.left) and _is_zero_like_expr(expr.right):
            return True
    return False


def _strip_zero_like_addends(expr: ExpressionIR, loc: SourceLocation) -> ExpressionIR:
    cur = expr
    while isinstance(cur, BinaryOpIR):
        if cur.operator == BinaryOp.ADD:
            if _is_zero_like_expr(cur.left):
                cur = cur.right
                continue
            if _is_zero_like_expr(cur.right):
                cur = cur.left
                continue
        if cur.operator == BinaryOp.SUB and _is_zero_like_expr(cur.right):
            cur = cur.left
            continue
        break
    return _simplify(cur, loc)


def _pullback_fold_einstein_clause_loops(
    body_pb: ExpressionIR,
    loop_vars: List[IndexVarIR],
    loc: SourceLocation,
) -> ExpressionIR:
    if not loop_vars:
        return body_pb
    loop_var_ranges: Dict[DefId, RangeIR] = {
        idx.defid: idx.range_ir
        for idx in loop_vars
        if idx.defid is not None and idx.range_ir is not None
    }
    if isinstance(body_pb, ReductionExpressionIR) and body_pb.operation == ReductionOp.SUM:
        merged_ranges: Dict[DefId, RangeIR] = {}
        if body_pb.loop_var_ranges:
            merged_ranges.update(body_pb.loop_var_ranges)
        existing_loop_vars = list(body_pb.loop_vars or [])
        existing_keys = {_loop_identity(lv) for lv in existing_loop_vars}
        new_loop_vars: List[IndexVarIR] = []
        for idx in loop_vars:
            if idx.defid is not None and idx.range_ir is not None:
                merged_ranges[idx.defid] = idx.range_ir
            key = _loop_identity(idx)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            new_loop_vars.append(idx)
        emit_debug_log(
            "autodiff.pullback",
            "_pullback.py:_pullback_fold_einstein_clause_loops",
            "merge_clause_loops_into_reduction",
            {
                "body_kind": type(body_pb).__name__,
                "existing_loop_vars": _debug_index_labels(cast(List[ExpressionIR], existing_loop_vars)),
                "added_loop_vars": _debug_index_labels(cast(List[ExpressionIR], new_loop_vars)),
            },
        )
        return ReductionExpressionIR(
            ReductionOp.SUM,
            existing_loop_vars + new_loop_vars,
            body_pb.body,
            loc,
            where_clause=body_pb.where_clause,
            loop_var_ranges=merged_ranges or None,
            type_info=_ti(body_pb) or F32,
            shape_info=_si(body_pb),
        )
    emit_debug_log(
        "autodiff.pullback",
        "_pullback.py:_pullback_fold_einstein_clause_loops",
        "wrap_pullback_in_sum",
        {
            "body_kind": type(body_pb).__name__,
            "loop_vars": _debug_index_labels(cast(List[ExpressionIR], loop_vars)),
        },
    )
    return ReductionExpressionIR(
        ReductionOp.SUM,
        loop_vars,
        body_pb,
        loc,
        loop_var_ranges=loop_var_ranges or None,
        type_info=_ti(body_pb),
        shape_info=_si(body_pb),
    )


def _pullback_memo_seed_key(node: Optional[ExpressionIR]) -> Optional[tuple]:
    """Structural key for VJP seeds so identical (binding, seed) pullback work can be shared."""
    if node is None:
        return None
    if isinstance(node, LiteralIR):
        v = node.value
        if isinstance(v, float):
            return ("lit", "f", float(v))
        return ("lit", type(v).__name__, v)
    if isinstance(node, IdentifierIR):
        if node.defid is None:
            return None
        return ("id", node.defid.krate, node.defid.index)
    if isinstance(node, IndexVarIR):
        if node.defid is None:
            return ("iv", node.name)
        return ("iv", node.name, node.defid.krate, node.defid.index)
    if isinstance(node, BinaryOpIR):
        kl = _pullback_memo_seed_key(node.left)
        kr = _pullback_memo_seed_key(node.right)
        if kl is None or kr is None:
            return None
        return ("bin", node.operator, kl, kr)
    if isinstance(node, UnaryOpIR):
        ko = _pullback_memo_seed_key(node.operand)
        if ko is None:
            return None
        return ("un", node.operator, ko)
    if isinstance(node, CastExpressionIR):
        inner = node.expr
        if inner is None:
            return None
        ke = _pullback_memo_seed_key(inner)
        if ke is None:
            return None
        tt = node.target_type
        tn = getattr(tt, "name", None) or (tt if isinstance(tt, str) else None) or str(tt)
        return ("cast", tn, ke)
    if isinstance(node, RectangularAccessIR):
        ka = _pullback_memo_seed_key(node.array)
        if ka is None:
            return None
        ks: List[tuple] = []
        for ix in node.indices or []:
            k = _pullback_memo_seed_key(ix)
            if k is None:
                return None
            ks.append(k)
        return ("rc", ka, tuple(ks))
    if isinstance(node, MemberAccessIR):
        ko = _pullback_memo_seed_key(node.object)
        if ko is None:
            return None
        return ("mem", ko, node.member)
    if isinstance(node, BuiltinCallIR):
        kas: List[tuple] = []
        for a in node.args or []:
            k = _pullback_memo_seed_key(a)
            if k is None:
                return None
            kas.append(k)
        did = node.defid
        dk = (did.krate, did.index) if did is not None else (-1, -1)
        return ("builtin", node.builtin_name, dk, tuple(kas))
    if isinstance(node, DifferentialIR):
        ko = _pullback_memo_seed_key(node.operand)
        if ko is None:
            return None
        return ("diff", ko)
    return None


@dataclass
class PullbackContext:
    wrt: DefId
    bindings: Dict[DefId, BindingIR]
    resolver: Any
    dep_cache: _DependencyQueryCache
    pullback_memo: Dict[tuple, ExpressionIR] = field(default_factory=dict)

    def with_bindings(self, extra: Dict[DefId, BindingIR]) -> "PullbackContext":
        merged = dict(self.bindings)
        merged.update(extra)
        return PullbackContext(
            self.wrt,
            merged,
            self.resolver,
            _DependencyQueryCache(merged),
            self.pullback_memo,
        )

    @property
    def wrt_binding(self) -> Optional[BindingIR]:
        return self.bindings.get(self.wrt)

    @property
    def wrt_rank(self) -> int:
        return _tensor_rank_from_binding(self.wrt_binding, self.bindings)


def _collect_block_bindings(expr: Optional[ExpressionIR], out: Dict[DefId, BindingIR]) -> None:
    if expr is None:
        return
    if isinstance(expr, BlockExpressionIR):
        for stmt in expr.statements or []:
            if isinstance(stmt, BindingIR) and stmt.defid is not None:
                out[stmt.defid] = stmt
                _collect_block_bindings(stmt.expr, out)
        _collect_block_bindings(expr.final_expr, out)
        return
    if isinstance(expr, IfExpressionIR):
        _collect_block_bindings(expr.then_expr, out)
        _collect_block_bindings(expr.else_expr, out)


def _shape_ref(expr: ExpressionIR, loc: SourceLocation) -> MemberAccessIR:
    return MemberAccessIR(expr, "shape", loc, type_info=UNKNOWN)


def _expr_shape_info(
    expr: ExpressionIR,
    bindings: Optional[Dict[DefId, BindingIR]] = None,
) -> Any:
    if isinstance(expr, IdentifierIR) and expr.defid is not None and bindings is not None:
        binding = bindings.get(expr.defid)
        if binding is not None:
            bsi = _si(binding) or _si(getattr(binding, "expr", None))
            if bsi is not None:
                return bsi
    return _si(expr)


def _shape_dim_expr(
    expr: ExpressionIR,
    dim_index: int,
    loc: SourceLocation,
    bindings: Optional[Dict[DefId, BindingIR]] = None,
) -> ExpressionIR:
    if isinstance(expr, IdentifierIR) and expr.defid is not None and bindings is not None:
        binding = bindings.get(expr.defid)
        if binding is not None:
            bsi = _si(binding) or _si(getattr(binding, "expr", None))
            if isinstance(bsi, (list, tuple)) and dim_index < len(bsi):
                dim = bsi[dim_index]
                try:
                    return LiteralIR(int(dim), loc, type_info=I32)
                except (TypeError, ValueError):
                    pass
            if isinstance(bsi, (list, tuple)) and dim_index >= len(bsi):
                return LiteralIR(1, loc, type_info=I32)
            if binding.expr is not None and binding.expr is not expr:
                return _shape_dim_expr(binding.expr, dim_index, loc, bindings)
    if isinstance(expr, IfExpressionIR) and bindings is not None:
        pruned = _prune_const_ifs_replayed(expr, bindings)
        if pruned is not None and pruned is not expr:
            return _shape_dim_expr(pruned, dim_index, loc, bindings)
    if isinstance(expr, BlockExpressionIR) and expr.final_expr is not None and bindings is not None:
        local_bindings = dict(bindings)
        for stmt in expr.statements or []:
            if isinstance(stmt, BindingIR) and stmt.defid is not None:
                local_bindings[stmt.defid] = stmt
        return _shape_dim_expr(expr.final_expr, dim_index, loc, local_bindings)
    if isinstance(expr, FunctionCallIR) and bindings is not None:
        callee_defid = getattr(expr, "function_defid", None)
        callee_binding = bindings.get(callee_defid) if callee_defid is not None else None
        callee_expr = getattr(callee_binding, "expr", None)
        if isinstance(callee_expr, FunctionValueIR) and callee_expr.body is not None:
            local_bindings = dict(bindings)
            for param, arg in zip(callee_expr.parameters or [], expr.arguments or []):
                if getattr(param, "defid", None) is None:
                    continue
                local_bindings[param.defid] = BindingIR(
                    name=getattr(param, "name", None),
                    expr=arg,
                    location=getattr(arg, "location", None) or loc,
                    defid=param.defid,
                    type_info=getattr(param, "param_type", None) or _ti(arg),
                )
            body_expr = _prune_const_ifs_replayed(callee_expr.body, local_bindings) or callee_expr.body
            return _shape_dim_expr(body_expr, dim_index, loc, local_bindings)
    if isinstance(expr, EinsteinIR):
        if expr.shape is not None and dim_index < len(expr.shape):
            dim_expr = expr.shape[dim_index]
            if isinstance(dim_expr, ExpressionIR):
                return dim_expr
        clauses = expr.clauses or []
        if clauses and dim_index < len(clauses[0].indices or []):
            idx = clauses[0].indices[dim_index]
            rng = getattr(idx, "range_ir", None)
            if isinstance(rng, RangeIR):
                return rng.end
    si = _si(expr)
    if isinstance(si, (list, tuple)) and dim_index < len(si):
        dim = si[dim_index]
        try:
            return LiteralIR(int(dim), loc, type_info=I32)
        except (TypeError, ValueError):
            pass
    if isinstance(si, (list, tuple)) and dim_index >= len(si):
        return LiteralIR(1, loc, type_info=I32)
    shape_expr = _shape_ref(expr, loc)
    dim = LiteralIR(dim_index, loc, type_info=I32)
    return RectangularAccessIR(shape_expr, [dim], loc, type_info=UNKNOWN)


def _fresh_axes_for_expr(
    expr: ExpressionIR, rank: int, ctx: PullbackContext, loc: SourceLocation
) -> List[IndexVarIR]:
    axes: List[IndexVarIR] = []
    _bounds_log = []
    for i in range(rank):
        did = ctx.resolver.allocate_for_local()
        bound = _shape_dim_expr(expr, i, loc, ctx.bindings)
        _bounds_log.append({"dim": i, "bound_type": type(bound).__name__, "bound_val": getattr(bound, "value", None)})
        rng = RangeIR(LiteralIR(0, loc, type_info=I32), bound, loc, type_info=UNKNOWN)
        axes.append(
            IndexVarIR(
                ctx.resolver.allocate_internal_iv_name(),
                loc,
                did,
                range_ir=rng,
                type_info=I32,
            )
        )
    emit_debug_log(
        "autodiff.pullback",
        "_pullback.py:_fresh_axes_for_expr",
        "axis_bounds",
        {
            "expr_name": getattr(expr, "name", None),
            "expr_type": type(expr).__name__,
            "rank": rank,
            "bounds": _bounds_log,
        },
    )
    return axes


def _fill_tensor(expr: ExpressionIR, fill: ExpressionIR, ctx: PullbackContext, loc: SourceLocation) -> ExpressionIR:
    rank = _tensor_rank_from_expr(expr, ctx.bindings)
    if rank <= 0:
        return fill
    axes = _fresh_axes_for_expr(expr, rank, ctx, loc)
    vr = {ax.defid: ax.range_ir for ax in axes if ax.defid is not None and ax.range_ir is not None}
    return EinsteinIR(
        clauses=[EinsteinClauseIR(indices=axes, value=fill, location=loc, variable_ranges=vr)],
        shape=None,
        element_type=None,
        location=loc,
        type_info=_ti(expr) or F32,
        shape_info=_expr_shape_info(expr, ctx.bindings),
    )


def _indexed_seed(seed: ExpressionIR, indices: List[ExpressionIR], loc: SourceLocation) -> ExpressionIR:
    return RectangularAccessIR(seed, indices, loc)


def _scatter_seed(base: ExpressionIR, indices: List[ExpressionIR], seed: ExpressionIR, ctx: PullbackContext, loc: SourceLocation) -> ExpressionIR:
    base_rank = _tensor_rank_from_expr(base, ctx.bindings)
    if base_rank <= 0:
        return seed
    _base_b = ctx.bindings.get(base.defid) if hasattr(base, "defid") and base.defid is not None else None
    _bsi = (_si(_base_b) or _si(getattr(_base_b, "expr", None))) if _base_b is not None else _si(base)
    _seed_si = _si(seed)
    emit_debug_log(
        "autodiff.pullback",
        "_pullback.py:_scatter_seed",
        "scatter_seed_called",
        {
            "base_name": getattr(base, "name", None),
            "base_defid": str(getattr(base, "defid", None)),
            "base_rank": base_rank,
            "binding_shape_info": str(_bsi),
            "seed_shape_info": str(_seed_si),
            "seed_type": type(seed).__name__,
            "indices": _debug_index_labels(indices),
        },
    )
    axes = _fresh_axes_for_expr(base, base_rank, ctx, loc)
    body: ExpressionIR
    rem = base_rank - len(indices)
    seed_rank = _tensor_rank_from_expr(seed, ctx.bindings)
    seed_indices: List[ExpressionIR] = []
    if seed_rank <= 0:
        body = seed
    elif rem > 0:
        consumed_seed_axes = max(0, seed_rank - max(rem, 0))
        consumed_seed_axes = min(consumed_seed_axes, len(indices))
        if consumed_seed_axes > 0:
            seed_indices.extend(cast(List[ExpressionIR], axes[:consumed_seed_axes]))
        seed_indices.extend(cast(List[ExpressionIR], axes[len(indices) :]))
        body = _indexed_seed(seed, seed_indices, loc) if seed_indices else seed
    else:
        body = seed
    if rem < 0:
        return seed
    emit_debug_log(
        "autodiff.pullback",
        "_pullback.py:_scatter_seed",
        "scatter_seed_indexing",
        {
            "base_name": getattr(base, "name", None),
            "base_rank": base_rank,
            "seed_rank": seed_rank,
            "rem": rem,
            "indices": _debug_index_labels(indices),
            "axis_vars": _debug_index_labels(cast(List[ExpressionIR], axes)),
            "seed_indices": _debug_index_labels(seed_indices),
            "body_kind": type(body).__name__,
        },
    )
    for ax, idx in zip(reversed(axes[: len(indices)]), reversed(indices)):
        eq = BinaryOpIR(BinaryOp.EQ, ax, idx, loc, type_info=BOOL)
        body = IfExpressionIR(eq, body, loc, else_expr=_z(loc), type_info=_ti(body) or F32)
    vr = {ax.defid: ax.range_ir for ax in axes if ax.defid is not None and ax.range_ir is not None}
    return EinsteinIR(
        clauses=[EinsteinClauseIR(indices=axes, value=body, location=loc, variable_ranges=vr)],
        shape=None,
        element_type=None,
        location=loc,
        type_info=_ti(base) or F32,
        shape_info=_expr_shape_info(base, ctx.bindings),
    )


def _zero_for_wrt(ctx: PullbackContext, loc: SourceLocation) -> ExpressionIR:
    wb = ctx.wrt_binding
    if wb is None or ctx.wrt_rank <= 0:
        return _z(loc)
    wrt_id = IdentifierIR(wb.name or "?", wb.location or loc, ctx.wrt, type_info=_ti(wb), shape_info=_si(wb))
    return _fill_tensor(wrt_id, _z(loc), ctx, loc)


def _sum_terms(terms: List[ExpressionIR], loc: SourceLocation) -> ExpressionIR:
    if not terms:
        return _z(loc)
    out = terms[0]
    for term in terms[1:]:
        out = BinaryOpIR(
            BinaryOp.ADD,
            out,
            term,
            loc,
            type_info=_ti(out) or _ti(term) or F32,
            shape_info=_si(out) or _si(term),
        )
    return out


def _mul_seed(seed: ExpressionIR, factor: ExpressionIR, loc: SourceLocation) -> ExpressionIR:
    return BinaryOpIR(
        BinaryOp.MUL,
        seed,
        factor,
        loc,
        type_info=_ti(seed) or _ti(factor) or F32,
        shape_info=_si(seed) or _si(factor),
    )


def _conv1d_call_pullback(
    node: FunctionCallIR,
    builder: "PullbackBuilder",
    callee_binding: BindingIR,
) -> Optional[ExpressionIR]:
    callee_name = callee_binding.name or ""
    fn_name = getattr(node, "function_name", None) or ""
    if not (
        fn_name == "conv1d"
        or callee_name == "conv1d"
        or callee_name.startswith("conv1d_")
        or callee_name.endswith("::conv1d")
    ):
        return None
    emit_debug_log(
        "autodiff.pullback",
        "_pullback.py:_conv1d_call_pullback",
        "conv1d_special_case_hit",
        {"callee_name": callee_name, "wrt_defid": str(builder.ctx.wrt)},
    )
    args = list(node.arguments or [])
    if len(args) != 8:
        return None
    loc = node.location or builder.loc
    X_arg, W_arg, B_arg, stride_arg, pad_begin_arg, _pad_end_arg, dilation_arg, group_arg = args
    seed_did = builder.ctx.resolver.allocate_for_local()
    seed_ref = IdentifierIR(
        "__conv_seed",
        loc,
        seed_did,
        type_info=_ti(builder.seed),
        shape_info=_expr_shape_info(builder.seed, builder.ctx.bindings),
    )
    seed_binding = BindingIR(
        name="__conv_seed",
        expr=builder.seed,
        location=loc,
        defid=seed_did,
        type_info=_ti(builder.seed),
    )
    seed_rank = _tensor_rank_from_expr(seed_ref, builder.ctx.bindings)
    output_shape_src: ExpressionIR = seed_ref if seed_rank > 0 else node

    batch_dim = _shape_dim_expr(X_arg, 0, loc, builder.ctx.bindings)
    c_in_dim = _shape_dim_expr(X_arg, 1, loc, builder.ctx.bindings)
    x_w_dim = _shape_dim_expr(X_arg, 2, loc, builder.ctx.bindings)
    c_out_dim = _shape_dim_expr(W_arg, 0, loc, builder.ctx.bindings)
    kernel_w_dim = _shape_dim_expr(W_arg, 2, loc, builder.ctx.bindings)
    out_w_dim = _shape_dim_expr(output_shape_src, 2, loc, builder.ctx.bindings)

    cpg_expr = BinaryOpIR(BinaryOp.DIV, c_in_dim, group_arg, loc, type_info=I32)
    fpg_expr = BinaryOpIR(BinaryOp.DIV, c_out_dim, group_arg, loc, type_info=I32)

    def _range(end_expr: ExpressionIR) -> RangeIR:
        return RangeIR(LiteralIR(0, loc, type_info=I32), end_expr, loc, type_info=UNKNOWN)

    def _axis(name: str, end_expr: ExpressionIR) -> IndexVarIR:
        did = builder.ctx.resolver.allocate_for_local()
        return IndexVarIR(name, loc, did, range_ir=_range(end_expr), type_info=I32)

    def _vr(*axes: IndexVarIR) -> Dict[DefId, RangeIR]:
        return {ax.defid: ax.range_ir for ax in axes if ax.defid is not None and ax.range_ir is not None}

    def _seed_at(*idxs: ExpressionIR) -> RectangularAccessIR:
        return RectangularAccessIR(seed_ref, list(idxs), loc)

    def _channel_index(co_ax: ExpressionIR, cl_ax: ExpressionIR) -> ExpressionIR:
        return BinaryOpIR(
            BinaryOp.ADD,
            BinaryOpIR(
                BinaryOp.MUL,
                BinaryOpIR(BinaryOp.DIV, co_ax, fpg_expr, loc, type_info=I32),
                cpg_expr,
                loc,
                type_info=I32,
            ),
            cl_ax,
            loc,
            type_info=I32,
        )

    def _window_index(i_ax: ExpressionIR, m_ax: ExpressionIR) -> ExpressionIR:
        return BinaryOpIR(
            BinaryOp.ADD,
            BinaryOpIR(BinaryOp.MUL, i_ax, stride_arg, loc, type_info=I32),
            BinaryOpIR(BinaryOp.MUL, m_ax, dilation_arg, loc, type_info=I32),
            loc,
            type_info=I32,
        )

    def _dx_seed() -> ExpressionIR:
        b_ax = _axis("batch.0", batch_dim)
        c_ax = _axis("c", c_in_dim)
        x_ax = _axis("ix", x_w_dim)
        co_ax = _axis("co", c_out_dim)
        i_ax = _axis("i", out_w_dim)
        cl_ax = _axis("cl", cpg_expr)
        m_ax = _axis("m", kernel_w_dim)

        channel_eq = BinaryOpIR(BinaryOp.EQ, c_ax, _channel_index(co_ax, cl_ax), loc, type_info=BOOL)
        x_eq = BinaryOpIR(
            BinaryOp.EQ,
            BinaryOpIR(BinaryOp.ADD, x_ax, pad_begin_arg, loc, type_info=I32),
            _window_index(i_ax, m_ax),
            loc,
            type_info=BOOL,
        )
        cond = BinaryOpIR(BinaryOp.AND, channel_eq, x_eq, loc, type_info=BOOL)
        body = IfExpressionIR(
            cond,
            BinaryOpIR(
                BinaryOp.MUL,
                _seed_at(b_ax, co_ax, i_ax),
                RectangularAccessIR(W_arg, [co_ax, cl_ax, m_ax], loc),
                loc,
            ),
            loc,
            else_expr=_z(loc),
            type_info=_ti(builder.seed) or F32,
        )
        red = ReductionExpressionIR(
            ReductionOp.SUM,
            [co_ax, i_ax, cl_ax, m_ax],
            body,
            loc,
            loop_var_ranges=_vr(co_ax, i_ax, cl_ax, m_ax),
            type_info=_ti(builder.seed) or F32,
        )
        return EinsteinIR(
            clauses=[EinsteinClauseIR(indices=[b_ax, c_ax, x_ax], value=red, location=loc, variable_ranges=_vr(b_ax, c_ax, x_ax))],
            shape=None,
            element_type=None,
            location=loc,
            type_info=_ti(X_arg) or F32,
            shape_info=_expr_shape_info(X_arg, builder.ctx.bindings),
        )

    def _dw_seed() -> ExpressionIR:
        co_ax = _axis("co", c_out_dim)
        cl_ax = _axis("cl", cpg_expr)
        m_ax = _axis("m", kernel_w_dim)
        b_ax = _axis("batch.0", batch_dim)
        i_ax = _axis("i", out_w_dim)

        x_sample = IfExpressionIR(
            BinaryOpIR(
                BinaryOp.AND,
                BinaryOpIR(BinaryOp.GE, _window_index(i_ax, m_ax), pad_begin_arg, loc, type_info=BOOL),
                BinaryOpIR(
                    BinaryOp.LT,
                    _window_index(i_ax, m_ax),
                    BinaryOpIR(BinaryOp.ADD, x_w_dim, pad_begin_arg, loc, type_info=I32),
                    loc,
                    type_info=BOOL,
                ),
                loc,
                type_info=BOOL,
            ),
            RectangularAccessIR(
                X_arg,
                [
                    b_ax,
                    _channel_index(co_ax, cl_ax),
                    BinaryOpIR(BinaryOp.SUB, _window_index(i_ax, m_ax), pad_begin_arg, loc, type_info=I32),
                ],
                loc,
            ),
            loc,
            else_expr=_z(loc),
            type_info=_ti(X_arg) or F32,
        )
        red = ReductionExpressionIR(
            ReductionOp.SUM,
            [b_ax, i_ax],
            BinaryOpIR(BinaryOp.MUL, _seed_at(b_ax, co_ax, i_ax), x_sample, loc),
            loc,
            loop_var_ranges=_vr(b_ax, i_ax),
            type_info=_ti(builder.seed) or F32,
        )
        return EinsteinIR(
            clauses=[EinsteinClauseIR(indices=[co_ax, cl_ax, m_ax], value=red, location=loc, variable_ranges=_vr(co_ax, cl_ax, m_ax))],
            shape=None,
            element_type=None,
            location=loc,
            type_info=_ti(W_arg) or F32,
            shape_info=_expr_shape_info(W_arg, builder.ctx.bindings),
        )

    def _db_seed() -> ExpressionIR:
        co_ax = _axis("co", c_out_dim)
        b_ax = _axis("batch.0", batch_dim)
        i_ax = _axis("i", out_w_dim)
        red = ReductionExpressionIR(
            ReductionOp.SUM,
            [b_ax, i_ax],
            _seed_at(b_ax, co_ax, i_ax),
            loc,
            loop_var_ranges=_vr(b_ax, i_ax),
            type_info=_ti(builder.seed) or F32,
        )
        return EinsteinIR(
            clauses=[EinsteinClauseIR(indices=[co_ax], value=red, location=loc, variable_ranges=_vr(co_ax))],
            shape=None,
            element_type=None,
            location=loc,
            type_info=_ti(B_arg) or F32,
            shape_info=_expr_shape_info(B_arg, builder.ctx.bindings),
        )

    terms: List[ExpressionIR] = []
    dep_cache = builder.ctx.dep_cache
    if _jacobian_rhs_depends_on_wrt(X_arg, builder.ctx.wrt, builder.ctx.bindings, dep_cache):
        terms.append(X_arg.accept(builder.with_seed(_dx_seed())))
    if _jacobian_rhs_depends_on_wrt(W_arg, builder.ctx.wrt, builder.ctx.bindings, dep_cache):
        terms.append(W_arg.accept(builder.with_seed(_dw_seed())))
    if _jacobian_rhs_depends_on_wrt(B_arg, builder.ctx.wrt, builder.ctx.bindings, dep_cache):
        terms.append(B_arg.accept(builder.with_seed(_db_seed())))
    if not terms:
        return _zero_for_wrt(builder.ctx, loc)
    out = _simplify(_sum_terms(terms, loc), loc)
    return BlockExpressionIR([seed_binding], loc, out, type_info=_ti(out), shape_info=_si(out))


def _layer_norm_call_pullback(
    node: FunctionCallIR,
    builder: "PullbackBuilder",
    callee_binding: BindingIR,
) -> Optional[ExpressionIR]:
    callee_name = callee_binding.name or ""
    fn_name = getattr(node, "function_name", None) or ""
    if not (
        fn_name == "layer_normalization"
        or callee_name == "layer_normalization"
        or callee_name.startswith("layer_normalization_")
        or callee_name.endswith("::layer_normalization")
    ):
        return None
    args = list(node.arguments or [])
    if len(args) != 5:
        return None
    X_arg, scale_arg, _B_arg, epsilon_arg, axis_arg = args
    if not (isinstance(X_arg, IdentifierIR) and X_arg.defid == builder.ctx.wrt):
        return None
    if not isinstance(axis_arg, LiteralIR):
        return None
    try:
        axis_val = int(axis_arg.value)
    except (TypeError, ValueError):
        return None
    rank = _tensor_rank_from_expr(X_arg, builder.ctx.bindings)
    if rank < 2 or axis_val not in (-1, rank - 1):
        return None
    loc = node.location or builder.loc
    feat_dim = _shape_dim_expr(X_arg, rank - 1, loc, builder.ctx.bindings)

    def _axis(name: str, end_expr: ExpressionIR) -> IndexVarIR:
        return IndexVarIR(
            name,
            loc,
            builder.ctx.resolver.allocate_for_local(),
            range_ir=RangeIR(LiteralIR(0, loc, type_info=I32), end_expr, loc, type_info=UNKNOWN),
            type_info=I32,
        )

    def _vr(*axes: IndexVarIR) -> Dict[DefId, RangeIR]:
        return {ax.defid: ax.range_ir for ax in axes if ax.defid is not None and ax.range_ir is not None}

    outer_axes = [_axis(f"batch.{d}", _shape_dim_expr(X_arg, d, loc, builder.ctx.bindings)) for d in range(rank - 1)]
    j_ax = _axis("j", feat_dim)

    def _fresh_red(name: str) -> IndexVarIR:
        return _axis(name, feat_dim)

    def _x_at(feature_idx: ExpressionIR) -> ExpressionIR:
        return RectangularAccessIR(X_arg, cast(List[ExpressionIR], list(outer_axes)) + [feature_idx], loc)

    def _seed_at(feature_idx: ExpressionIR) -> ExpressionIR:
        return RectangularAccessIR(builder.seed, cast(List[ExpressionIR], list(outer_axes)) + [feature_idx], loc)

    def _scale_at(feature_idx: ExpressionIR) -> ExpressionIR:
        return RectangularAccessIR(scale_arg, [feature_idx], loc)

    def _mean_expr() -> ExpressionIR:
        k_ax = _fresh_red("k_mean")
        red = ReductionExpressionIR(
            ReductionOp.SUM,
            [k_ax],
            _x_at(k_ax),
            loc,
            loop_var_ranges=_vr(k_ax),
            type_info=F32,
        )
        return BinaryOpIR(BinaryOp.DIV, red, feat_dim, loc, type_info=F32)

    mean_expr = _mean_expr()

    def _std_expr() -> ExpressionIR:
        k_ax = _fresh_red("k_std")
        centered = BinaryOpIR(BinaryOp.SUB, _x_at(k_ax), mean_expr, loc, type_info=F32)
        variance = BinaryOpIR(
            BinaryOp.DIV,
            ReductionExpressionIR(
                ReductionOp.SUM,
                [k_ax],
                BinaryOpIR(BinaryOp.MUL, centered, centered, loc, type_info=F32),
                loc,
                loop_var_ranges=_vr(k_ax),
                type_info=F32,
            ),
            feat_dim,
            loc,
            type_info=F32,
        )
        return BinaryOpIR(
            BinaryOp.POW,
            BinaryOpIR(BinaryOp.ADD, variance, epsilon_arg, loc, type_info=F32),
            _fl(0.5, loc),
            loc,
            type_info=F32,
        )

    std_expr = _std_expr()
    x_center_j = BinaryOpIR(BinaryOp.SUB, _x_at(j_ax), mean_expr, loc, type_info=F32)
    xhat_j = BinaryOpIR(BinaryOp.DIV, x_center_j, std_expr, loc, type_info=F32)
    dxhat_j = BinaryOpIR(BinaryOp.MUL, _seed_at(j_ax), _scale_at(j_ax), loc, type_info=F32)

    s1_ax = _fresh_red("s1")
    sum_dxhat = ReductionExpressionIR(
        ReductionOp.SUM,
        [s1_ax],
        BinaryOpIR(BinaryOp.MUL, _seed_at(s1_ax), _scale_at(s1_ax), loc, type_info=F32),
        loc,
        loop_var_ranges=_vr(s1_ax),
        type_info=F32,
    )
    s2_ax = _fresh_red("s2")
    x_center_s2 = BinaryOpIR(BinaryOp.SUB, _x_at(s2_ax), mean_expr, loc, type_info=F32)
    xhat_s2 = BinaryOpIR(BinaryOp.DIV, x_center_s2, std_expr, loc, type_info=F32)
    sum_dxhat_xhat = ReductionExpressionIR(
        ReductionOp.SUM,
        [s2_ax],
        BinaryOpIR(
            BinaryOp.MUL,
            BinaryOpIR(BinaryOp.MUL, _seed_at(s2_ax), _scale_at(s2_ax), loc, type_info=F32),
            xhat_s2,
            loc,
            type_info=F32,
        ),
        loc,
        loop_var_ranges=_vr(s2_ax),
        type_info=F32,
    )
    numerator = BinaryOpIR(
        BinaryOp.SUB,
        BinaryOpIR(
            BinaryOp.SUB,
            BinaryOpIR(BinaryOp.MUL, feat_dim, dxhat_j, loc, type_info=F32),
            sum_dxhat,
            loc,
            type_info=F32,
        ),
        BinaryOpIR(BinaryOp.MUL, xhat_j, sum_dxhat_xhat, loc, type_info=F32),
        loc,
        type_info=F32,
    )
    body = BinaryOpIR(
        BinaryOp.DIV,
        numerator,
        BinaryOpIR(BinaryOp.MUL, feat_dim, std_expr, loc, type_info=F32),
        loc,
        type_info=F32,
    )
    all_axes = outer_axes + [j_ax]
    return EinsteinIR(
        clauses=[EinsteinClauseIR(indices=all_axes, value=body, location=loc, variable_ranges=_vr(*all_axes))],
        shape=None,
        element_type=None,
        location=loc,
        type_info=_ti(X_arg) or F32,
        shape_info=_expr_shape_info(X_arg, builder.ctx.bindings),
    )


class PullbackBuilder(IRVisitor[ExpressionIR]):
    def __init__(self, ctx: PullbackContext, seed: ExpressionIR, loc: SourceLocation):
        self.ctx = ctx
        self.seed = seed
        self.loc = loc

    def with_seed(self, seed: ExpressionIR) -> "PullbackBuilder":
        return PullbackBuilder(self.ctx, seed, self.loc)

    def visit_identifier(self, node: IdentifierIR) -> ExpressionIR:
        if node.defid == self.ctx.wrt:
            return self.seed
        if node.defid is None:
            return _zero_for_wrt(self.ctx, self.loc)
        binding = self.ctx.bindings.get(node.defid)
        if binding is None or binding.expr is None or is_function_binding(binding):
            return _zero_for_wrt(self.ctx, self.loc)
        if not _jacobian_rhs_depends_on_wrt(binding.expr, self.ctx.wrt, self.ctx.bindings, self.ctx.dep_cache):
            return _zero_for_wrt(self.ctx, self.loc)
        sk = _pullback_memo_seed_key(self.seed)
        memo = self.ctx.pullback_memo
        if sk is not None:
            mkey = (id(binding), sk)
            hit = memo.get(mkey)
            if hit is not None:
                return hit
        out = binding.expr.accept(self.with_seed(self.seed))
        if sk is not None:
            memo[mkey] = out
        return out

    def visit_literal(self, node: LiteralIR) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, self.loc)

    def visit_array_literal(self, node: ArrayLiteralIR) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, self.loc)

    def visit_binary_op(self, node: BinaryOpIR) -> ExpressionIR:
        loc = node.location or self.loc
        if node.operator == BinaryOp.ADD:
            return _simplify(
                BinaryOpIR(
                    BinaryOp.ADD,
                    node.left.accept(self.with_seed(self.seed)),
                    node.right.accept(self.with_seed(self.seed)),
                    loc,
                    type_info=_ti(node) or F32,
                ),
                loc,
            )
        if node.operator == BinaryOp.SUB:
            return _simplify(
                BinaryOpIR(
                    BinaryOp.SUB,
                    node.left.accept(self.with_seed(self.seed)),
                    node.right.accept(self.with_seed(self.seed)),
                    loc,
                    type_info=_ti(node) or F32,
                ),
                loc,
            )
        if node.operator == BinaryOp.MUL:
            dl = node.left.accept(self.with_seed(_mul_seed(self.seed, node.right, loc)))
            dr = node.right.accept(self.with_seed(_mul_seed(self.seed, node.left, loc)))
            return _simplify(BinaryOpIR(BinaryOp.ADD, dl, dr, loc, type_info=_ti(node) or F32), loc)
        if node.operator == BinaryOp.DIV:
            dl = node.left.accept(
                self.with_seed(BinaryOpIR(BinaryOp.DIV, self.seed, node.right, loc, type_info=_ti(node) or F32))
            )
            dr_seed = UnaryOpIR(
                UnaryOp.NEG,
                BinaryOpIR(
                    BinaryOp.DIV,
                    _mul_seed(self.seed, node.left, loc),
                    BinaryOpIR(BinaryOp.POW, node.right, _fl(2, loc), loc, type_info=_ti(node) or F32),
                    loc,
                    type_info=_ti(node) or F32,
                ),
                loc,
                type_info=_ti(node) or F32,
            )
            dr = node.right.accept(self.with_seed(dr_seed))
            return _simplify(BinaryOpIR(BinaryOp.ADD, dl, dr, loc, type_info=_ti(node) or F32), loc)
        if node.operator == BinaryOp.POW:
            left_seed = BinaryOpIR(
                BinaryOp.MUL,
                self.seed,
                BinaryOpIR(
                    BinaryOp.MUL,
                    node.right,
                    BinaryOpIR(
                        BinaryOp.POW,
                        node.left,
                        BinaryOpIR(BinaryOp.SUB, node.right, _fl(1, loc), loc, type_info=_ti(node) or F32),
                        loc,
                        type_info=_ti(node) or F32,
                    ),
                    loc,
                    type_info=_ti(node) or F32,
                ),
                loc,
                type_info=_ti(node) or F32,
            )
            dl = node.left.accept(self.with_seed(left_seed))
            right_seed = BinaryOpIR(
                BinaryOp.MUL,
                self.seed,
                BinaryOpIR(BinaryOp.MUL, node, BuiltinCallIR("ln", [node.left], loc), loc, type_info=_ti(node) or F32),
                loc,
                type_info=_ti(node) or F32,
            )
            dr = node.right.accept(self.with_seed(right_seed))
            return _simplify(BinaryOpIR(BinaryOp.ADD, dl, dr, loc, type_info=_ti(node) or F32), loc)
        if node.operator == BinaryOp.MOD:
            return node.left.accept(self.with_seed(self.seed))
        if node.operator in (
            BinaryOp.EQ,
            BinaryOp.NE,
            BinaryOp.LT,
            BinaryOp.LE,
            BinaryOp.GT,
            BinaryOp.GE,
            BinaryOp.AND,
            BinaryOp.OR,
        ):
            return _zero_for_wrt(self.ctx, loc)
        return _zero_for_wrt(self.ctx, loc)

    def visit_unary_op(self, node: UnaryOpIR) -> ExpressionIR:
        loc = node.location or self.loc
        if node.operator == UnaryOp.NEG:
            return node.operand.accept(self.with_seed(UnaryOpIR(UnaryOp.NEG, self.seed, loc, type_info=_ti(self.seed) or F32)))
        if node.operator == UnaryOp.POS:
            return node.operand.accept(self.with_seed(self.seed))
        return _zero_for_wrt(self.ctx, loc)

    def visit_rectangular_access(self, node: RectangularAccessIR) -> ExpressionIR:
        loc = node.location or self.loc
        root_expr, full_indices = _flatten_rect_access(node)
        scattered = _scatter_seed(root_expr, list(full_indices), self.seed, self.ctx, loc)
        return root_expr.accept(self.with_seed(scattered))

    def visit_cast_expression(self, node: CastExpressionIR) -> ExpressionIR:
        return node.expr.accept(self.with_seed(self.seed))

    def visit_if_expression(self, node: IfExpressionIR) -> ExpressionIR:
        loc = node.location or self.loc
        cond_f = CastExpressionIR(
            node.condition,
            F32,
            loc,
            type_info=F32,
            shape_info=_si(node.condition),
        )
        one = _fl(1, loc)
        else_mask = BinaryOpIR(
            BinaryOp.SUB,
            one,
            cond_f,
            loc,
            type_info=F32,
            shape_info=_si(cond_f) or _si(node.condition),
        )
        then_seed = BinaryOpIR(
            BinaryOp.MUL,
            self.seed,
            cond_f,
            loc,
            type_info=_ti(self.seed) or F32,
            shape_info=_si(self.seed),
        )
        else_seed = BinaryOpIR(
            BinaryOp.MUL,
            self.seed,
            else_mask,
            loc,
            type_info=_ti(self.seed) or F32,
            shape_info=_si(self.seed),
        )
        then_pb = node.then_expr.accept(self.with_seed(then_seed))
        else_pb = node.else_expr.accept(self.with_seed(else_seed)) if node.else_expr is not None else _zero_for_wrt(self.ctx, loc)
        return _simplify(
            BinaryOpIR(
                BinaryOp.ADD,
                then_pb,
                else_pb,
                loc,
                type_info=_ti(then_pb) or _ti(else_pb) or _ti(node) or F32,
                shape_info=_si(then_pb) or _si(else_pb),
            ),
            loc,
        )

    def visit_block_expression(self, node: BlockExpressionIR) -> ExpressionIR:
        loc = node.location or self.loc
        local_bindings = dict(self.ctx.bindings)
        for stmt in node.statements or []:
            if isinstance(stmt, BindingIR) and stmt.defid is not None:
                local_bindings[stmt.defid] = stmt
        child = PullbackBuilder(self.ctx.with_bindings(local_bindings), self.seed, loc)
        if node.final_expr is None:
            return _zero_for_wrt(self.ctx, loc)
        final_pb = node.final_expr.accept(child)
        replayed: List[Any] = []
        for stmt in node.statements or []:
            replayed.append(stmt)
        emit_debug_log(
            "autodiff.pullback",
            "_pullback.py:visit_block_expression",
            "replay_primal_block",
            {
                "replayed_names": [getattr(s, "name", None) for s in replayed if isinstance(s, BindingIR)],
                "wrt_defid": str(self.ctx.wrt),
                "final_expr_type": type(node.final_expr).__name__,
            },
        )
        return BlockExpressionIR(replayed, loc, final_pb, type_info=_ti(node), shape_info=_si(node))

    def visit_reduction_expression(self, node: ReductionExpressionIR) -> ExpressionIR:
        loc = node.location or self.loc
        if node.operation == ReductionOp.SUM:
            return ReductionExpressionIR(
                ReductionOp.SUM,
                node.loop_vars,
                node.body.accept(self.with_seed(self.seed)),
                loc,
                where_clause=node.where_clause,
                loop_var_ranges=node.loop_var_ranges,
                type_info=_ti(node),
                shape_info=_si(node),
            )
        if node.operation == ReductionOp.MAX:
            return SelectAtArgmaxIR(
                node.body,
                node.body.accept(self.with_seed(self.seed)),
                list(node.loop_vars or []),
                loop_var_ranges=node.loop_var_ranges,
                location=loc,
                type_info=_ti(node),
                shape_info=_si(node),
            )
        if node.operation == ReductionOp.MIN:
            return SelectAtArgmaxIR(
                node.body,
                node.body.accept(self.with_seed(self.seed)),
                list(node.loop_vars or []),
                loop_var_ranges=node.loop_var_ranges,
                location=loc,
                type_info=_ti(node),
                shape_info=_si(node),
                use_argmin=True,
            )
        if node.operation == ReductionOp.PROD:
            return _prod_pullback_via_sum(
                node,
                node.body.accept(self.with_seed(self.seed)),
                loc,
                self.ctx.resolver,
            )
        return _zero_for_wrt(self.ctx, loc)

    def visit_einstein(self, node: EinsteinIR) -> ExpressionIR:
        loc = node.location or self.loc
        terms: List[ExpressionIR] = []
        seed_rank = _tensor_rank_from_expr(self.seed, self.ctx.bindings)
        for clause in node.clauses or []:
            if clause.value is None:
                continue
            clause_seed: ExpressionIR = self.seed
            seed_indices: List[ExpressionIR] = []
            if seed_rank > 0 and clause.indices:
                seed_indices = list(clause.indices)[:seed_rank]
                clause_seed = _indexed_seed(self.seed, seed_indices, loc)
            body_pb = _strip_zero_like_addends(
                _simplify(clause.value.accept(self.with_seed(clause_seed)), loc),
                loc,
            )
            body_pb = _unwrap_trivial_einstein_rhs(body_pb)
            body_is_zero = _is_zero(body_pb)
            loop_vars = [idx for idx in (clause.indices or []) if isinstance(idx, IndexVarIR)]
            if seed_rank > 0 and len(loop_vars) > seed_rank:
                loop_vars = loop_vars[seed_rank:]
            emit_debug_log(
                "autodiff.pullback",
                "_pullback.py:visit_einstein",
                "visit_einstein_clause",
                {
                    "wrt_defid": str(self.ctx.wrt),
                    "seed_rank": seed_rank,
                    "clause_indices": _debug_index_labels(cast(List[ExpressionIR], list(clause.indices or []))),
                    "seed_indices": _debug_index_labels(seed_indices),
                    "body_pb_kind": type(body_pb).__name__,
                    "body_pb_is_zero": body_is_zero,
                    "fold_loop_vars": _debug_index_labels(cast(List[ExpressionIR], loop_vars)),
                },
            )
            if body_is_zero:
                continue
            term = _pullback_fold_einstein_clause_loops(body_pb, loop_vars, loc) if loop_vars else body_pb
            terms.append(term)
        if not terms:
            return _zero_for_wrt(self.ctx, loc)
        return _simplify(_sum_terms(terms, loc), loc)

    def visit_function_call(self, node: FunctionCallIR) -> ExpressionIR:
        loc = node.location or self.loc
        if node.function_defid is None:
            return _zero_for_wrt(self.ctx, loc)
        callee_binding = self.ctx.bindings.get(node.function_defid)
        if callee_binding is None or not isinstance(callee_binding.expr, FunctionValueIR):
            return _zero_for_wrt(self.ctx, loc)
        layer_norm_pb = _layer_norm_call_pullback(node, self, callee_binding)
        if layer_norm_pb is not None:
            return layer_norm_pb
        callee_name = callee_binding.name or ""
        fn_name = getattr(node, "function_name", None) or ""
        if (
            (
                fn_name == "conv"
                or callee_name == "conv"
                or callee_name.startswith("conv_")
                or callee_name.endswith("::conv")
            )
            and len(node.arguments or []) == 7
            and isinstance(node.arguments[3], ArrayLiteralIR)
            and isinstance(node.arguments[4], ArrayLiteralIR)
            and isinstance(node.arguments[5], ArrayLiteralIR)
        ):
            strides_arg = list(node.arguments[3].elements or [])
            pads_arg = list(node.arguments[4].elements or [])
            dilations_arg = list(node.arguments[5].elements or [])
            if len(strides_arg) == 1 and len(pads_arg) == 2 and len(dilations_arg) == 1:
                synthetic_callee = (
                    IdentifierIR(
                        "conv1d",
                        loc,
                        node.function_defid,
                        type_info=_ti(node.callee_expr) if node.callee_expr is not None else UNKNOWN,
                    )
                )
                conv1d_like = FunctionCallIR(
                    callee_expr=synthetic_callee,
                    location=loc,
                    arguments=[
                        node.arguments[0],
                        node.arguments[1],
                        node.arguments[2],
                        strides_arg[0],
                        pads_arg[0],
                        pads_arg[1],
                        dilations_arg[0],
                        node.arguments[6],
                    ],
                    module_path=node.module_path,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
                conv_public_pb = _conv1d_call_pullback(
                    conv1d_like,
                    self,
                    BindingIR(
                        name="conv1d",
                        expr=callee_binding.expr,
                        location=callee_binding.location,
                        defid=callee_binding.defid,
                        type_info=callee_binding.type_info,
                    ),
                )
                if conv_public_pb is not None:
                    emit_debug_log(
                        "autodiff.pullback",
                        "_pullback.py:visit_function_call",
                        "public_conv_rank1_special_case_hit",
                        {"callee_name": callee_name, "wrt_defid": str(self.ctx.wrt)},
                    )
                    return conv_public_pb
        conv1d_pb = _conv1d_call_pullback(node, self, callee_binding)
        if conv1d_pb is not None:
            return conv1d_pb
        fv = callee_binding.expr
        params = fv.parameters or []
        rm = {}
        param_bindings: Dict[DefId, BindingIR] = {}
        param_prelude: List[BindingIR] = []
        for j, param in enumerate(params):
            if param.defid is None or j >= len(node.arguments or []):
                continue
            arg = node.arguments[j]
            rm[param.defid] = arg
            binding = BindingIR(
                name=getattr(param, "name", None),
                expr=arg,
                location=arg.location or loc,
                defid=param.defid,
                type_info=getattr(param, "param_type", None) or _ti(arg),
            )
            param_bindings[param.defid] = binding
            param_prelude.append(binding)
        rule_body = getattr(fv, "custom_diff_body", None)
        if rule_body is not None and len(params) == len(node.arguments or []):
            terms: List[ExpressionIR] = []
            for i, param in enumerate(params):
                if param.defid is None or i >= len(node.arguments or []):
                    continue
                dm = {
                    p.defid: (_fl(1, loc) if j == i else _z(loc))
                    for j, p in enumerate(params)
                    if p.defid is not None
                }
                coef = _simplify(_sub_wd(rule_body, rm, dm, loc), loc)
                terms.append(node.arguments[i].accept(self.with_seed(_simplify(_mul_seed(self.seed, coef, loc), loc))))
            return _simplify(_sum_terms(terms, loc), loc)
        if fv.body is None:
            return _zero_for_wrt(self.ctx, loc)
        child_ctx = self.ctx.with_bindings(param_bindings)
        prune_bindings = dict(child_ctx.bindings)
        _collect_block_bindings(fv.body, prune_bindings)
        body_expr = _prune_const_ifs_replayed(fv.body, prune_bindings) or fv.body
        child = PullbackBuilder(child_ctx, self.seed, loc)
        out = body_expr.accept(child)
        if param_prelude:
            return BlockExpressionIR(param_prelude, loc, out, type_info=_ti(out), shape_info=_si(out))
        return out

    def visit_builtin_call(self, node: BuiltinCallIR) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_member_access(self, node: MemberAccessIR) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc) if node.member == "shape" else node.object.accept(self.with_seed(self.seed))

    def visit_tuple_expression(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_tuple_access(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_index_var(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_index_rest(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_differential(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_lambda(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_range(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_array_comprehension(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_jagged_access(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_interpolated_string(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_try_expression(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_match_expression(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_pipeline_expression(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_where_expression(self, node: Any) -> ExpressionIR:
        return _zero_for_wrt(self.ctx, node.location or self.loc)

    def visit_function_value(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_module(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_program(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_literal_pattern(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_identifier_pattern(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_wildcard_pattern(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_tuple_pattern(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_array_pattern(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_rest_pattern(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_guard_pattern(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_or_pattern(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_constructor_pattern(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_binding_pattern(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)

    def visit_range_pattern(self, node: Any) -> ExpressionIR:
        _unsupported_autodiff_ir("PullbackBuilder", node)


def build_seeded_pullback(
    expr: ExpressionIR,
    seed: ExpressionIR,
    wrt: DefId,
    bindings: Dict[DefId, BindingIR],
    resolver: Any,
    loc: SourceLocation,
    dependency_cache: Optional[_DependencyQueryCache] = None,
) -> ExpressionIR:
    ctx = PullbackContext(
        wrt=wrt,
        bindings=bindings,
        resolver=resolver,
        dep_cache=dependency_cache if dependency_cache is not None and dependency_cache.bindings is bindings else _DependencyQueryCache(bindings),
    )
    return expr.accept(PullbackBuilder(ctx, seed, loc))


def build_default_seed(
    expr: ExpressionIR,
    bindings: Dict[DefId, BindingIR],
    resolver: Any,
    loc: SourceLocation,
    numerator_binding: Optional[BindingIR] = None,
) -> ExpressionIR:
    fill_src: ExpressionIR = expr
    if numerator_binding is not None:
        rank = _tensor_rank_from_binding(numerator_binding, bindings)
    else:
        rank = _tensor_rank_from_expr(expr, bindings)
    if rank <= 0:
        return _fl(1, loc)
    if not isinstance(expr, (IdentifierIR, RectangularAccessIR)):
        return _fl(1, loc)
    ctx = PullbackContext(wrt=DefId(0, 0), bindings=bindings, resolver=resolver, dep_cache=_DependencyQueryCache(bindings))
    return _fill_tensor(fill_src, _fl(1, loc), ctx, loc)
