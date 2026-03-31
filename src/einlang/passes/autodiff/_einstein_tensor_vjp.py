"""Generalized Einstein cotangent construction for autodiff (multilinear / reduction VJP in IR).

This is the Einlang counterpart of the NumPy ``tensor_vjp`` in ``examples/julia_style_conv_vjp_numpy.py``:
for a clause whose reduction body is a product of rectangular reads, derivatives w.r.t. one tensor
factor are built by summing over the same loops with all *other* factors as frozen coefficients and
optional Kronecker-style ``where`` constraints — no op-specific conv/pool/matmul logic.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ._core import _fl, _is_zero, _si, _ti, _z
from ...ir.nodes import (
    BindingIR,
    BinaryOpIR,
    EinsteinClauseIR,
    ExpressionIR,
    IdentifierIR,
    IfExpressionIR,
    IndexVarIR,
    LiteralIR,
    MemberAccessIR,
    RangeIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    SelectAtArgmaxIR,
    WhereClauseIR,
)
from ...shared.defid import DefId
from ...shared.source_location import SourceLocation
from ...shared.types import BOOL, F32, UNKNOWN, BinaryOp, PrimitiveType, ReductionOp


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


def _if_product_with_zero_else(
    inner: Optional[ExpressionIR],
) -> Optional[Tuple[ExpressionIR, ExpressionIR]]:
    """If inner is ``if cond { product-of-reads } else { 0 }``, return ``(cond, then_expr)``."""
    if inner is None or not isinstance(inner, IfExpressionIR):
        return None
    el = inner.else_expr
    if el is None or not _is_zero(el):
        return None
    then_e = inner.then_expr
    if then_e is None or _flatten_product(then_e) is None:
        return None
    return (inner.condition, then_e)


def _wrt_positions_in_factors(
    factors: List[Tuple[ExpressionIR, List]],
    wrt: DefId,
    ps: Optional[Dict[DefId, ExpressionIR]],
) -> List[int]:
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
    return wrt_pos


def _build_deriv_idx(
    clause_indices: List,
    wrt_indices: List,
    wrt_id: IdentifierIR,
    resolver: object,
    loc: SourceLocation,
    allow_reuse: bool,
    shared_axes: Optional[List[IndexVarIR]] = None,
) -> Tuple[List, Set[DefId], Dict]:
    ci_by_did: Dict[DefId, object] = {}
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
            iv = IndexVarIR(resolver.allocate_internal_iv_name(), loc, nd, range_ir=rng)
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


def _mk_tensor_ref(a: ExpressionIR, idxs: List, loc: SourceLocation, B: Dict[DefId, BindingIR]) -> ExpressionIR:
    bf = B.get(a.defid) if isinstance(a, IdentifierIR) else None
    nm = a.name or (bf.name if bf else "") or ""
    ref = IdentifierIR(nm, loc, a.defid, type_info=_ti(a), shape_info=_si(a)) if isinstance(a, IdentifierIR) else a
    return RectangularAccessIR(ref, list(idxs), loc)


def einstein_multilinear_sum_vjp_clause(
    clause: EinsteinClauseIR,
    val: ReductionExpressionIR,
    factors: List[Tuple[ExpressionIR, List]],
    wrt_pos: List[int],
    wrt: DefId,
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    R: object,
    clause_shared_axes: Optional[List[IndexVarIR]],
    mask_cond: Optional[ExpressionIR] = None,
) -> EinsteinClauseIR:
    first_wi = factors[wrt_pos[0]][1]
    wb = B.get(wrt)
    wid = IdentifierIR((wb.name if wb else "") or "?", loc, wrt, type_info=UNKNOWN)
    ci = list(clause.indices or [])
    allow_reuse = len(ci) < len(first_wi)
    dvars, new_dids, new_vr = _build_deriv_idx(ci, first_wi, wid, R, loc, allow_reuse, shared_axes=clause_shared_axes)
    lvs = list(val.loop_vars or [])
    mlr = _merged_lr(val, clause)
    orig_rc: List[ExpressionIR] = list(getattr(val.where_clause, "constraints", None) or []) if val.where_clause else []
    rterms: List[ExpressionIR] = []
    for pos in wrt_pos:
        _, wi = factors[pos]
        others = [factors[i] for i in range(len(factors)) if i != pos]
        deltas = [
            BinaryOpIR(BinaryOp.EQ, wi[p], dvars[p], loc, type_info=BOOL)
            for p in range(len(wi))
            if getattr(dvars[p], "defid", None) in new_dids
        ]
        wh = WhereClauseIR(constraints=orig_rc + deltas, location=loc) if (orig_rc or deltas) else None
        if not others:
            body: ExpressionIR = _fl(1, loc)
        else:
            body = _mk_tensor_ref(others[0][0], others[0][1], loc, B)
            for ae, il in others[1:]:
                body = BinaryOpIR(BinaryOp.MUL, body, _mk_tensor_ref(ae, il, loc, B), loc, type_info=_ti(val) or F32, shape_info=_si(val))
        if mask_cond is not None:
            body = IfExpressionIR(
                mask_cond,
                body,
                loc,
                else_expr=_fl(0.0, loc),
                type_info=_ti(val) or F32,
                shape_info=_si(val),
            )
        rterms.append(
            ReductionExpressionIR(val.operation, lvs, body, loc, where_clause=wh, loop_var_ranges=mlr, type_info=_ti(val), shape_info=_si(val))
        )
    cv: ExpressionIR = rterms[0]
    for r in rterms[1:]:
        cv = BinaryOpIR(BinaryOp.ADD, cv, r, loc, type_info=_ti(val) or F32, shape_info=_si(val))
    ni = dvars if allow_reuse else (ci + dvars)
    nvr = dict(clause.variable_ranges or {})
    nvr.update(new_vr)
    return EinsteinClauseIR(indices=ni, value=cv, location=clause.location, where_clause=clause.where_clause, variable_ranges=nvr)


def einstein_maxmin_vjp_clause(
    clause: EinsteinClauseIR,
    val: ReductionExpressionIR,
    factors: List[Tuple[ExpressionIR, List]],
    wrt_pos: List[int],
    wrt: DefId,
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    R: object,
    clause_shared_axes: Optional[List[IndexVarIR]],
    use_argmin: bool,
) -> EinsteinClauseIR:
    first_wi = factors[wrt_pos[0]][1]
    wb = B.get(wrt)
    wid = IdentifierIR((wb.name if wb else "") or "?", loc, wrt, type_info=UNKNOWN)
    ci = list(clause.indices or [])
    allow_reuse = len(ci) < len(first_wi)
    dvars, new_dids, new_vr = _build_deriv_idx(ci, first_wi, wid, R, loc, allow_reuse, shared_axes=clause_shared_axes)
    lvs = list(val.loop_vars or [])
    mlr = _merged_lr(val, clause)
    db: ExpressionIR = _fl(1, loc)
    for p in range(len(first_wi)):
        if getattr(dvars[p], "defid", None) in new_dids:
            eq = BinaryOpIR(BinaryOp.EQ, first_wi[p], dvars[p], loc, type_info=BOOL)
            db = IfExpressionIR(eq, db, loc, else_expr=_z(loc), type_info=_ti(val) or F32)
    sel = SelectAtArgmaxIR(val.body, db, lvs, loop_var_ranges=mlr, location=loc, type_info=_ti(val), shape_info=_si(val), use_argmin=use_argmin)
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
            out_val = ReductionExpressionIR(
                ReductionOp.SUM,
                sum_loops,
                sel,
                loc,
                loop_var_ranges=sum_ranges if sum_ranges else None,
                type_info=_ti(val),
                shape_info=None,
            )
    return EinsteinClauseIR(indices=dvars, value=out_val, location=clause.location, where_clause=clause.where_clause, variable_ranges=nvr)


def einstein_prod_reuse_vjp_clause(
    clause: EinsteinClauseIR,
    val: ReductionExpressionIR,
    factors: List[Tuple[ExpressionIR, List]],
    wrt_pos: List[int],
    wrt: DefId,
    loc: SourceLocation,
    R: object,
    clause_shared_axes: Optional[List[IndexVarIR]],
) -> EinsteinClauseIR:
    first_wi = factors[wrt_pos[0]][1]
    wb = B.get(wrt)
    wid = IdentifierIR((wb.name if wb else "") or "?", loc, wrt, type_info=UNKNOWN)
    ci = list(clause.indices or [])
    allow_reuse = len(ci) < len(first_wi)
    dvars, new_dids, new_vr = _build_deriv_idx(ci, first_wi, wid, R, loc, allow_reuse, shared_axes=clause_shared_axes)
    lvs = list(val.loop_vars or [])
    mlr = _merged_lr(val, clause)
    exc = [
        BinaryOpIR(BinaryOp.NE, first_wi[p], dvars[p], loc, type_info=BOOL)
        for p in range(len(first_wi))
        if getattr(dvars[p], "defid", None) in new_dids
    ]
    ow = getattr(val, "where_clause", None)
    oc = list(getattr(ow, "constraints", None) or []) if ow else []
    pw = WhereClauseIR(constraints=oc + exc, location=loc) if (oc or exc) else None
    pr = ReductionExpressionIR(ReductionOp.PROD, lvs, val.body, loc, where_clause=pw, loop_var_ranges=mlr, type_info=_ti(val), shape_info=_si(val))
    nvr = dict(clause.variable_ranges or {})
    nvr.update(new_vr)
    return EinsteinClauseIR(indices=dvars, value=pr, location=clause.location, where_clause=clause.where_clause, variable_ranges=nvr)


def einstein_structured_reduction_vjp(
    clause: EinsteinClauseIR,
    val: ReductionExpressionIR,
    wrt: DefId,
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    R: object,
    ps: Optional[Dict[DefId, ExpressionIR]],
    clause_shared_axes: Optional[List[IndexVarIR]],
) -> Optional[EinsteinClauseIR]:
    inner = val.body
    mask_cond: Optional[ExpressionIR] = None
    product_inner = inner
    peeled = _if_product_with_zero_else(inner)
    if peeled is not None:
        mask_cond, product_inner = peeled
    factors = _flatten_product(product_inner) if product_inner else None
    if not factors:
        return None
    wrt_pos = _wrt_positions_in_factors(factors, wrt, ps)
    if not wrt_pos:
        return None
    op = val.operation
    first_wi = factors[wrt_pos[0]][1]
    ci = list(clause.indices or [])
    allow_reuse = len(ci) < len(first_wi)
    if op in (ReductionOp.MAX, ReductionOp.MIN):
        return einstein_maxmin_vjp_clause(
            clause, val, factors, wrt_pos, wrt, loc, B, R, clause_shared_axes, use_argmin=(op == ReductionOp.MIN)
        )
    if op == ReductionOp.PROD and allow_reuse:
        return einstein_prod_reuse_vjp_clause(clause, val, factors, wrt_pos, wrt, loc, R, clause_shared_axes)
    if op != ReductionOp.SUM:
        return None
    return einstein_multilinear_sum_vjp_clause(
        clause, val, factors, wrt_pos, wrt, loc, B, R, clause_shared_axes, mask_cond=mask_cond
    )
