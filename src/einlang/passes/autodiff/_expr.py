from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Union

from ._graph import _collect_defids
from ...ir.nodes import (
    BinaryOpIR,
    BindingIR,
    BlockExpressionIR,
    BuiltinCallIR,
    CastExpressionIR,
    EinsteinIR,
    ExpressionIR,
    FunctionCallIR,
    IdentifierIR,
    IfExpressionIR,
    IndexVarIR,
    LiteralIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    SelectAtArgmaxIR,
    UnaryOpIR,
)
from ...shared.defid import DefId
from ...shared.types import BinaryOp


def _expr_uses_index_defids(expr: Optional[ExpressionIR], defids: Set[DefId]) -> bool:
    if expr is None or not defids:
        return False
    if isinstance(expr, IndexVarIR):
        return expr.defid is not None and expr.defid in defids
    if isinstance(expr, (LiteralIR, IdentifierIR)):
        return False
    if isinstance(expr, UnaryOpIR):
        return _expr_uses_index_defids(expr.operand, defids)
    if isinstance(expr, BinaryOpIR):
        return _expr_uses_index_defids(expr.left, defids) or _expr_uses_index_defids(expr.right, defids)
    if isinstance(expr, IfExpressionIR):
        if _expr_uses_index_defids(expr.condition, defids):
            return True
        if _expr_uses_index_defids(expr.then_expr, defids):
            return True
        return _expr_uses_index_defids(expr.else_expr, defids)
    if isinstance(expr, RectangularAccessIR):
        if _expr_uses_index_defids(expr.array, defids):
            return True
        for ix in expr.indices or []:
            if _expr_uses_index_defids(ix, defids):
                return True
        return False
    if isinstance(expr, ReductionExpressionIR):
        ow = getattr(expr, "where_clause", None)
        if ow is not None:
            for c in getattr(ow, "constraints", None) or []:
                if _expr_uses_index_defids(c, defids):
                    return True
        return _expr_uses_index_defids(expr.body, defids)
    if isinstance(expr, CastExpressionIR):
        return _expr_uses_index_defids(expr.expr, defids)
    if isinstance(expr, EinsteinIR):
        for c in expr.clauses or []:
            for ix in c.indices or []:
                did = getattr(ix, "defid", None)
                if did is not None and did in defids:
                    return True
            cw = getattr(c, "where_clause", None)
            if cw is not None:
                for cn in getattr(cw, "constraints", None) or []:
                    if _expr_uses_index_defids(cn, defids):
                        return True
            if _expr_uses_index_defids(c.value, defids):
                return True
        return False
    if isinstance(expr, SelectAtArgmaxIR):
        return _expr_uses_index_defids(expr.primal_body, defids) or _expr_uses_index_defids(
            expr.diff_body, defids
        )
    if isinstance(expr, FunctionCallIR):
        for a in expr.arguments or []:
            if _expr_uses_index_defids(a, defids):
                return True
        return False
    if isinstance(expr, BlockExpressionIR):
        if expr.final_expr is not None and _expr_uses_index_defids(expr.final_expr, defids):
            return True
        for s in expr.statements or []:
            if isinstance(s, BindingIR) and s.expr is not None and _expr_uses_index_defids(s.expr, defids):
                return True
        return False
    return False


def _eval_const_expr(
    expr: Optional[ExpressionIR],
    bindings: Dict[DefId, BindingIR],
    max_depth: int = 32,
    _vis: Optional[Set[DefId]] = None,
    subst: Optional[Dict[DefId, ExpressionIR]] = None,
) -> Optional[Union[int, float, bool]]:
    if expr is None or max_depth <= 0:
        return None
    if isinstance(expr, LiteralIR):
        return expr.value
    if isinstance(expr, CastExpressionIR):
        return _eval_const_expr(expr.expr, bindings, max_depth - 1, _vis, subst)
    if isinstance(expr, IdentifierIR) and expr.defid is not None:
        vis = _vis if _vis is not None else set()
        if expr.defid in vis:
            return None
        vis.add(expr.defid)
        if subst is not None:
            sub = subst.get(expr.defid)
            if sub is not None:
                sv = _eval_const_expr(sub, bindings, max_depth - 1, vis, subst)
                if sv is not None:
                    return sv
        b = bindings.get(expr.defid)
        if b is None or b.expr is None:
            return None
        return _eval_const_expr(b.expr, bindings, max_depth - 1, vis, subst)
    if isinstance(expr, BuiltinCallIR):
        if expr.builtin_name == "len" and len(expr.args or []) == 1:
            cur: Optional[ExpressionIR] = expr.args[0]
            hops = 0
            while isinstance(cur, IdentifierIR) and cur.defid is not None and hops < 16:
                if subst is not None:
                    sub = subst.get(cur.defid)
                    if sub is not None:
                        cur_try = sub
                        if hasattr(cur_try, "elements"):
                            cur = cur_try
                            break
                        cur = cur_try
                        hops += 1
                        if not isinstance(cur, IdentifierIR):
                            break
                b = bindings.get(cur.defid)
                if b is None or b.expr is None:
                    break
                cur = b.expr
                hops += 1
            if hasattr(cur, "elements"):
                return len(cur.elements or [])
        return None
    if isinstance(expr, UnaryOpIR):
        v = _eval_const_expr(expr.operand, bindings, max_depth - 1, _vis, subst)
        if v is None:
            return None
        if expr.operator.name == "NEG":
            return -float(v)
        if expr.operator.name == "POS":
            return +float(v)
        return None
    if isinstance(expr, BinaryOpIR):
        lv = _eval_const_expr(expr.left, bindings, max_depth - 1, _vis, subst)
        rv = _eval_const_expr(expr.right, bindings, max_depth - 1, _vis, subst)
        if lv is None or rv is None:
            return None
        op = expr.operator
        if op == BinaryOp.ADD:
            return float(lv) + float(rv)
        if op == BinaryOp.SUB:
            return float(lv) - float(rv)
        if op == BinaryOp.MUL:
            return float(lv) * float(rv)
        if op == BinaryOp.DIV:
            if float(rv) == 0.0:
                return None
            return float(lv) / float(rv)
        if op == BinaryOp.MOD:
            if float(rv) == 0.0:
                return None
            return int(float(lv)) % int(float(rv))
        if op == BinaryOp.POW:
            return float(lv) ** float(rv)
        if op == BinaryOp.EQ:
            return lv == rv
        if op == BinaryOp.NE:
            return lv != rv
        if op == BinaryOp.LT:
            return float(lv) < float(rv)
        if op == BinaryOp.LE:
            return float(lv) <= float(rv)
        if op == BinaryOp.GT:
            return float(lv) > float(rv)
        if op == BinaryOp.GE:
            return float(lv) >= float(rv)
        if op == BinaryOp.AND:
            return bool(lv) and bool(rv)
        if op == BinaryOp.OR:
            return bool(lv) or bool(rv)
        return None
    return None


def _collapse_empty_block_wrappers(expr: Optional[ExpressionIR]) -> Optional[ExpressionIR]:
    cur = expr
    while isinstance(cur, BlockExpressionIR) and not (cur.statements or []) and cur.final_expr is not None:
        cur = cur.final_expr
    return cur


def _collapse_nested_empty_blocks(expr: Optional[ExpressionIR]) -> Optional[ExpressionIR]:
    cur = expr
    while isinstance(cur, BlockExpressionIR) and not (cur.statements or []) and isinstance(cur.final_expr, BlockExpressionIR):
        cur = cur.final_expr
    return cur


def _prune_const_ifs_replayed(
    expr: Optional[ExpressionIR],
    bindings: Dict[DefId, BindingIR],
) -> Optional[ExpressionIR]:
    if expr is None:
        return None
    if isinstance(expr, IfExpressionIR):
        cond = _prune_const_ifs_replayed(expr.condition, bindings)
        then_expr = _prune_const_ifs_replayed(expr.then_expr, bindings)
        else_expr = _prune_const_ifs_replayed(expr.else_expr, bindings) if expr.else_expr is not None else None
        cond_value = _eval_const_expr(cond, bindings)
        if isinstance(cond_value, (bool, int)):
            return _collapse_nested_empty_blocks(then_expr if bool(cond_value) else else_expr)
        if cond is expr.condition and then_expr is expr.then_expr and else_expr is expr.else_expr:
            return expr
        return IfExpressionIR(
            cond if cond is not None else expr.condition,
            then_expr if then_expr is not None else expr.then_expr,
            expr.location,
            else_expr=else_expr,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )
    if isinstance(expr, BlockExpressionIR):
        changed = False
        ns: List[Any] = []
        for s in expr.statements or []:
            if isinstance(s, BindingIR) and s.expr is not None:
                ne = _prune_const_ifs_replayed(s.expr, bindings)
                if ne is s.expr:
                    ns.append(s)
                else:
                    changed = True
                    ns.append(
                        BindingIR(
                            name=s.name,
                            expr=ne,
                            location=s.location,
                            defid=s.defid,
                            type_info=s.type_info,
                        )
                    )
            else:
                ns.append(s)
        nf = _prune_const_ifs_replayed(expr.final_expr, bindings) if expr.final_expr is not None else None
        if nf is not expr.final_expr:
            changed = True
        if not changed:
            return expr
        return BlockExpressionIR(
            ns,
            expr.location,
            nf,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )
    return expr


__all__ = [
    "_collapse_empty_block_wrappers",
    "_collapse_nested_empty_blocks",
    "_eval_const_expr",
    "_expr_uses_index_defids",
    "_prune_const_ifs_replayed",
]
