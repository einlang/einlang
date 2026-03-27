from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ._core import _fl, _is_zero, _si, _ti, _z
from ._expr import _expr_uses_index_defids
from ._graph import _rectangular_read_root_defid
from ...ir.nodes import (
    ArrayLiteralIR,
    BindingIR,
    BinaryOpIR,
    CastExpressionIR,
    EinsteinClauseIR,
    EinsteinIR,
    ExpressionIR,
    IdentifierIR,
    IfExpressionIR,
    IndexVarIR,
    LiteralIR,
    MemberAccessIR,
    RangeIR,
    RectangularAccessIR,
)
from ...shared.defid import DefId
from ...shared.source_location import SourceLocation
from ...shared.types import BOOL, F32, UNKNOWN, BinaryOp, PrimitiveType, RectangularType


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
    wid = IdentifierIR(wb.name or "_wrt", wb.location or loc, wrt, type_info=_ti(wb) or UNKNOWN, shape_info=_si(wb))
    rank = _tensor_rank_from_binding(wb)
    if rank <= 0:
        return []
    out: List[IndexVarIR] = []
    for p in range(rank):
        nd = resolver.allocate_for_local()
        sh = MemberAccessIR(object=wid, member="shape", location=loc, type_info=UNKNOWN)
        dim = LiteralIR(p, loc, type_info=PrimitiveType("i32"))
        sd = RectangularAccessIR(array=sh, indices=[dim], location=loc, type_info=UNKNOWN)
        rng = RangeIR(start=LiteralIR(0, loc, type_info=PrimitiveType("i32")), end=sd, location=loc, type_info=UNKNOWN)
        out.append(IndexVarIR("_jcot_%d" % p, loc, nd, range_ir=rng))
    return out


def _jacobian_rect_read_wrt_slice_binding(
    read_indices: List[ExpressionIR],
    wrt: DefId,
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    array_root: Optional[DefId] = None,
) -> Optional[ExpressionIR]:
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
    shape_ref = MemberAccessIR(object=tensor_id, member="shape", location=loc, type_info=UNKNOWN)
    for p in range(len(slice_idxs)):
        did = resolver.allocate_for_local()
        dim = LiteralIR(p, loc, type_info=PrimitiveType("i32"))
        bound = RectangularAccessIR(array=shape_ref, indices=[dim], location=loc, type_info=UNKNOWN)
        rng = RangeIR(start=LiteralIR(0, loc, type_info=PrimitiveType("i32")), end=bound, location=loc, type_info=UNKNOWN)
        iv = IndexVarIR("_jslice_%d" % p, loc, did, range_ir=rng)
        dyn_idxs.append(iv)
        var_ranges[did] = rng
    body: ExpressionIR = _fl(1, loc)
    for d, s in zip(dyn_idxs, slice_idxs):
        eq = BinaryOpIR(BinaryOp.EQ, d, s, loc, type_info=BOOL)
        body = IfExpressionIR(eq, body, loc, else_expr=_z(loc), type_info=F32)
    return EinsteinIR(
        clauses=[EinsteinClauseIR(indices=dyn_idxs, value=body, location=loc, variable_ranges=var_ranges)],
        shape=None,
        element_type=None,
        location=loc,
        type_info=_ti(tensor_id),
        shape_info=_si(tensor_id),
    )


def _flatten_rect_access(expr: RectangularAccessIR) -> Tuple[ExpressionIR, List[ExpressionIR]]:
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
    inner = _kronecker_delta_indices(accessed[1:], wrt_axes[1:], loc) if accessed[1:] else _fl(1, loc)
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
