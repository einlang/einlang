from __future__ import annotations

from typing import Any, List, Optional, Set, cast

from ._core import _Rewriter
from ._expr import _collapse_empty_block_wrappers
from ._graph import _DependencyQueryCache, _collect_defids
from ...ir.nodes import (
    BindingIR,
    BlockExpressionIR,
    ExpressionIR,
    IdentifierIR,
    IfExpressionIR,
    IndexRestIR,
    IndexVarIR,
    SourceLocation,
)


def _is_diff_name(name: str) -> bool:
    return name.startswith("_@") or name.startswith("@")


def _idx_str(idx: Any) -> str:
    if isinstance(idx, IndexVarIR):
        return idx.name or "?"
    if isinstance(idx, IndexRestIR):
        return f"..{idx.name}" if idx.name else ".."
    if isinstance(idx, IdentifierIR):
        return idx.name or "?"
    return "?"


def _str_ir(expr: ExpressionIR) -> str:
    cleaned = expr.accept(_CleanPrintBlocksRewriter(getattr(expr, "location", None) or SourceLocation("", 0, 0)))
    return str(_collapse_empty_block_wrappers(cleaned) or cleaned)


def _live_primal_defids_for_print_display(
    stmts: List[Any],
    final_expr: Optional[ExpressionIR],
    dep_cache: Optional[_DependencyQueryCache] = None,
) -> Set[Any]:
    def _deps(expr: Optional[ExpressionIR]) -> Set[Any]:
        if dep_cache is not None:
            return set(dep_cache.collect_defids(expr))
        return _collect_defids(expr)

    need: Set[Any] = set()
    if final_expr is not None:
        need |= _deps(final_expr)
    for s in stmts:
        if isinstance(s, BindingIR) and s.expr is not None and _is_diff_name(s.name or ""):
            need |= _deps(s.expr)
    primal_by_did = {
        s.defid: s
        for s in stmts
        if isinstance(s, BindingIR) and s.defid is not None and not _is_diff_name(s.name or "")
    }
    while True:
        prev = len(need)
        for pd in list(need):
            pb = primal_by_did.get(pd)
            if pb is not None and pb.expr is not None:
                need |= _deps(pb.expr)
        if len(need) == prev:
            break
    return set(primal_by_did.keys()) & need


class _TrimDeadPrimalPrintRewriter(_Rewriter):
    def __init__(self, loc: SourceLocation, dep_cache: Optional[_DependencyQueryCache] = None) -> None:
        super().__init__(loc)
        self._dep_cache = dep_cache

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        if n.final_expr is None:
            return n
        ns: List[Any] = []
        for s in n.statements or []:
            if isinstance(s, BindingIR) and s.expr is not None:
                ns.append(
                    BindingIR(
                        name=s.name,
                        expr=s.expr.accept(self),
                        location=s.location,
                        defid=s.defid,
                        type_info=getattr(s, "type_info", None),
                    )
                )
            elif isinstance(s, ExpressionIR):
                ns.append(s.accept(self))
            else:
                ns.append(s)
        nf = n.final_expr.accept(self)
        live = _live_primal_defids_for_print_display(ns, nf, self._dep_cache)
        out = []
        for s in ns:
            if isinstance(s, BindingIR):
                nm = s.name or ""
                if _is_diff_name(nm) or s.defid is None or s.defid in live:
                    out.append(s)
            else:
                out.append(s)
        return BlockExpressionIR(
            out,
            loc,
            nf,
            type_info=getattr(n, "type_info", None),
            shape_info=getattr(n, "shape_info", None),
        )


class _CleanPrintBlocksRewriter(_Rewriter):
    def __init__(self, loc: SourceLocation, preserve_outer_block: bool = False) -> None:
        super().__init__(loc)
        self._preserve_outer_block = preserve_outer_block

    def _child(self, preserve_outer_block: bool = False) -> "_CleanPrintBlocksRewriter":
        return _CleanPrintBlocksRewriter(self._loc, preserve_outer_block=preserve_outer_block)

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        rewritten = cast(BlockExpressionIR, super().visit_block_expression(n))
        collapsed_final = (
            _collapse_empty_block_wrappers(rewritten.final_expr)
            if rewritten.final_expr is not None
            else None
        )
        if collapsed_final is not rewritten.final_expr:
            rewritten = BlockExpressionIR(
                list(rewritten.statements or []),
                rewritten.location or self._loc,
                collapsed_final,
                type_info=getattr(rewritten, "type_info", None),
                shape_info=getattr(rewritten, "shape_info", None),
            )
        if self._preserve_outer_block:
            return rewritten
        return _collapse_empty_block_wrappers(rewritten) or rewritten

    def visit_if_expression(self, n: IfExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        nc = n.condition.accept(self._child())
        nt = n.then_expr.accept(self._child(preserve_outer_block=isinstance(n.then_expr, BlockExpressionIR)))
        ne = (
            n.else_expr.accept(self._child(preserve_outer_block=isinstance(n.else_expr, BlockExpressionIR)))
            if n.else_expr is not None
            else None
        )
        if nc is n.condition and nt is n.then_expr and ne is n.else_expr:
            return n
        return IfExpressionIR(
            condition=nc,
            then_expr=nt,
            location=loc,
            else_expr=ne,
            type_info=getattr(n, "type_info", None),
            shape_info=getattr(n, "shape_info", None),
        )


def _str_ir_print_differential_rhs(
    expr: ExpressionIR,
    loc: SourceLocation,
    dep_cache: Optional[_DependencyQueryCache] = None,
) -> str:
    trimmed = expr.accept(_TrimDeadPrimalPrintRewriter(loc, dep_cache))
    cleaned = trimmed.accept(_CleanPrintBlocksRewriter(loc))
    return str(_collapse_empty_block_wrappers(cleaned) or cleaned)


def _fmt_print_msg(lhs: str, rhs: str) -> str:
    return "let " + lhs + " = " + rhs.rstrip("\n") + ";"


__all__ = [
    "_CleanPrintBlocksRewriter",
    "_TrimDeadPrimalPrintRewriter",
    "_fmt_print_msg",
    "_idx_str",
    "_live_primal_defids_for_print_display",
    "_str_ir",
    "_str_ir_print_differential_rhs",
]
