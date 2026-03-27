from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from ._core import DIFF_PREFIX, _fl, _is_zero, _si, _simplify, _sub, _sub_wd, _ti, _z
from ._expr import _expr_uses_index_defids, _prune_const_ifs_replayed
from ._graph import _DependencyQueryCache
from ...ir.nodes import (
    ArrayLiteralIR,
    BindingIR,
    BinaryOpIR,
    BlockExpressionIR,
    CastExpressionIR,
    EinsteinIR,
    ExpressionIR,
    FunctionValueIR,
    IdentifierIR,
    IfExpressionIR,
    LiteralIR,
    RectangularAccessIR,
    UnaryOpIR,
)
from ...shared.defid import DefId
from ...shared.source_location import SourceLocation
from ...shared.types import F32, BinaryOp


def _callee_block_build_primal(
    block: BlockExpressionIR,
    pm: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    R: Any,
) -> Tuple[List[BindingIR], Dict[DefId, ExpressionIR]]:
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
        primal_stmts.append(BindingIR(name=pn, expr=subst_expr, location=s.location or loc, defid=pd, type_info=bti))
    return primal_stmts, primal_map


def _callee_replay_block(
    block: BlockExpressionIR,
    pm: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    R: Any,
) -> BlockExpressionIR:
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
        stmts_out.append(BindingIR(name=pn, expr=sub_e, location=s.location or bl, defid=pd, type_info=bti))
    fe_out = _callee_replay_expression(block.final_expr, pm, loc, R) if block.final_expr is not None else None
    return BlockExpressionIR(stmts_out, bl, fe_out, type_info=_ti(block), shape_info=_si(block))


def _callee_replay_expression(
    expr: ExpressionIR,
    pm: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    R: Any,
) -> ExpressionIR:
    if isinstance(expr, BlockExpressionIR):
        return _callee_replay_block(expr, pm, loc, R)
    if isinstance(expr, IfExpressionIR):
        el = expr.location or loc
        cond_s = _sub(expr.condition, pm, loc)
        pm_then = dict(pm)
        then_r = _callee_replay_expression(expr.then_expr, pm_then, loc, R)
        else_r = _callee_replay_expression(expr.else_expr, dict(pm), loc, R) if expr.else_expr is not None else None
        return IfExpressionIR(cond_s, then_r, el, else_expr=else_r, type_info=_ti(expr), shape_info=_si(expr))
    return _sub(expr, pm, loc)


def _stmt_partial_for_replayed_callee_final(
    sp: Dict[DefId, ExpressionIR],
    primal_map: Dict[DefId, ExpressionIR],
    block: BlockExpressionIR,
) -> Dict[DefId, ExpressionIR]:
    out: Dict[DefId, ExpressionIR] = dict(sp)
    for s in block.statements or []:
        if not isinstance(s, BindingIR) or s.defid is None or s.defid not in sp:
            continue
        prim = primal_map.get(s.defid)
        if isinstance(prim, IdentifierIR) and prim.defid is not None:
            out[prim.defid] = sp[s.defid]
    return out


def _tensor_root_defid_for_jacobian_fe(expr: Optional[ExpressionIR]) -> Optional[DefId]:
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
    from ._jacobian import JacobianVisitor
    if block.final_expr is None:
        raise ValueError("Autodiff: callee block has no final expression")
    dep_cache = _DependencyQueryCache(B)
    sp: Dict[DefId, ExpressionIR] = {}
    ps_map = primal_map if wrt not in param_subst else None
    vis = JacobianVisitor(wrt, loc, B, R, stmt_partial=sp, wrt_tangent=wt, primal_subst=ps_map, dependency_cache=dep_cache)
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
            axis_ids = {iv.defid for iv in (vis._wrt_axes or []) if getattr(iv, "defid", None) is not None}
            if axis_ids and _expr_uses_index_defids(pv, axis_ids):
                sp[s.defid] = pv
            else:
                sp[s.defid] = dr
            diff_stmts.append(BindingIR(name=dn, expr=pv, location=s.location or loc, defid=dd, type_info=bti))
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
        shared_cotangent_axes=vis._wrt_axes,
        full_cotangent_axes=vis._full_wrt_axes,
        dependency_cache=dep_cache,
    )
    fp = _simplify(_sub(fe_rep.accept(vis_fe), pm_fp, loc), loc)
    return diff_stmts, fp


def _callee_args_support_combined_forward(rm: Dict[DefId, ExpressionIR], ps: List[Any]) -> bool:
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
    primal_stmts: List[BindingIR],
    primal_map: Dict[DefId, ExpressionIR],
    rm: Dict[DefId, ExpressionIR],
    tangent_by_param: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    R: Any,
    ps: List[Any],
) -> Tuple[List[BindingIR], ExpressionIR]:
    from ._forward import DiffVisitor
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
        if isinstance(arg, (LiteralIR, ArrayLiteralIR)):
            continue
        if not isinstance(arg, IdentifierIR) or arg.defid is None:
            raise ValueError("Autodiff: combined callee forward requires identifier or literal arguments")
        dre[arg.defid] = da
    diff_stmts: List[BindingIR] = []
    bl = block.location or loc
    replayed_by_defid: Dict[DefId, BindingIR] = {}
    for s, replayed_stmt in zip((stmt for stmt in (block.statements or []) if isinstance(stmt, BindingIR) and stmt.defid is not None and stmt.expr is not None), primal_stmts):
        replayed_by_defid[s.defid] = replayed_stmt
    for s in block.statements or []:
        if not isinstance(s, BindingIR) or s.defid is None or s.expr is None:
            continue
        replayed = replayed_by_defid.get(s.defid)
        if replayed is None or replayed.expr is None:
            raise ValueError("Autodiff: missing replayed primal binding for callee local")
        pv = _simplify(replayed.expr.accept(DiffVisitor(dre, loc, B, R)), loc)
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
        diff_stmts.append(BindingIR(name=dn, expr=pv, location=s.location or bl, defid=dd, type_info=bti))
    pm_fe = dict(primal_map)
    fe_rep = _callee_replay_expression(block.final_expr, pm_fe, bl, R)
    replayed_local_bindings = {stmt.defid: stmt for stmt in primal_stmts if stmt.defid is not None}
    fe_rep = _prune_const_ifs_replayed(fe_rep, replayed_local_bindings)
    dre_final: Dict[DefId, ExpressionIR] = dict(dre)
    for p in ps:
        if p.defid is None:
            continue
        arg = rm[p.defid]
        if not isinstance(arg, IdentifierIR) or arg.defid is None or p.defid == arg.defid:
            continue
        da0 = tangent_by_param.get(p.defid)
        if da0 is None:
            continue
        dre_final[p.defid] = dre_final.get(arg.defid, da0)
    fp = _simplify(fe_rep.accept(DiffVisitor(dre_final, loc, B, R, pretty=False, keep_primal_lets=True)), loc)
    return diff_stmts, fp


def _diff_callee_block(
    block: BlockExpressionIR,
    wrt: DefId,
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    R: Any,
    pm: Dict[DefId, ExpressionIR],
    wt: Optional[ExpressionIR] = None,
) -> ExpressionIR:
    if block.final_expr is None:
        raise ValueError("Autodiff: callee block has no final expression")
    primal_stmts, primal_map = _callee_block_build_primal(block, pm, loc, R)
    diff_stmts, fp = _diff_callee_block_tangent(block, wrt, loc, B, R, primal_map, pm, wt)
    out_stmts = primal_stmts + diff_stmts
    if out_stmts:
        return BlockExpressionIR(out_stmts, block.location or loc, fp, type_info=_ti(block), shape_info=_si(block))
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
    if not terms:
        return _z(loc)
    out = terms[0]
    for t in terms[1:]:
        out = BinaryOpIR(BinaryOp.ADD, out, t, loc, type_info=_ti(out) or _ti(t) or F32, shape_info=_si(out) or _si(t))
    return out


def _callee_arg_with_binding_metadata(arg: ExpressionIR, bindings: Dict[DefId, BindingIR]) -> ExpressionIR:
    if isinstance(arg, LiteralIR):
        cloned = LiteralIR(arg.value, arg.location, type_info=_ti(arg), shape_info=_si(arg))
    elif isinstance(arg, IdentifierIR):
        cloned = IdentifierIR(arg.name, arg.location, arg.defid, type_info=_ti(arg), shape_info=_si(arg))
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
    ps = fv.parameters or []
    return {p.defid: _callee_arg_with_binding_metadata(args[j], bindings) for j, p in enumerate(ps) if p.defid is not None and j < len(args)}


def _callee_primal_let_binding_count(fv: FunctionValueIR) -> int:
    body = fv.body
    if not isinstance(body, BlockExpressionIR):
        return 0
    return sum(1 for s in (body.statements or []) if isinstance(s, BindingIR))


def _pretty_use_callee_tangent_block_direct(fv: FunctionValueIR, args: List[ExpressionIR], peeled: ExpressionIR) -> bool:
    if getattr(fv, "custom_diff_body", None) is not None:
        return False
    if len(args) != 1 or not isinstance(args[0], IdentifierIR):
        return False
    if not isinstance(peeled, BlockExpressionIR):
        return False
    if not any(isinstance(s, BindingIR) for s in (peeled.statements or [])):
        return False
    return _callee_primal_let_binding_count(fv) >= 2


def _flatten_add_terms(terms: List[ExpressionIR], loc: SourceLocation) -> ExpressionIR:
    if not terms:
        return _z(loc)
    if len(terms) == 1:
        return terms[0]
    ms: List[BindingIR] = []
    fs: List[ExpressionIR] = []
    for t in terms:
        if isinstance(t, BlockExpressionIR) and t.final_expr is not None:
            for s in t.statements or []:
                if isinstance(s, BindingIR):
                    ms.append(s)
            fs.append(t.final_expr)
        else:
            fs.append(t)
    acc = fs[0]
    for f in fs[1:]:
        acc = BinaryOpIR(BinaryOp.ADD, acc, f, loc, type_info=_ti(acc) or _ti(f) or F32, shape_info=_si(acc) or _si(f))
    return BlockExpressionIR(ms, loc, acc, type_info=_ti(acc), shape_info=_si(acc)) if ms else acc


def _callee_forward_jvp(
    fv: FunctionValueIR,
    args: List[ExpressionIR],
    tangent_by_param: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    B: Dict[DefId, BindingIR],
    R: Any,
) -> ExpressionIR:
    from ._jacobian import JacobianVisitor
    ps = fv.parameters or []
    rule_body = getattr(fv, "custom_diff_body", None)
    if rule_body is not None and len(ps) == len(args):
        rm = _callee_primal_subst_map(fv, args, B)
        return _sub_callee(_sub_wd(rule_body, rm, tangent_by_param, loc), fv, rm, loc, fold_body_bindings=False)
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
            all_diff, acc = _diff_callee_block_combined_forward(body, primal_stmts, primal_map, rm, tangent_by_param, loc, B, R, ps)
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
                acc = BinaryOpIR(BinaryOp.ADD, acc, fe, loc, type_info=_ti(acc) or _ti(fe) or F32, shape_info=_si(acc) or _si(fe))
        out: ExpressionIR = BlockExpressionIR(primal_stmts + all_diff, body.location or loc, acc, type_info=_ti(body), shape_info=_si(body)) if (primal_stmts or all_diff) else acc
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
    return _sub_callee(_flatten_add_terms(terms, loc), fv, rm, loc)
