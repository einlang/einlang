"""
Tree Shaking Pass — removes unreachable functions from ProgramIR.

Walks the IR starting from entry points (top-level statements) and
transitively collects all reachable function DefIds via FunctionCallIR
and identifier (function) references.  Functions not in the reachable set are
pruned from ProgramIR.functions.

This runs as the LAST pass, after monomorphization and lowering, so
generic template functions (never called directly) are naturally pruned.
"""

import logging
from typing import Set

from ..ir.nodes import (
    ExpressionIR, ProgramIR, FunctionDefIR, BindingIR,
    FunctionCallIR, BuiltinCallIR,
    IdentifierIR, LiteralIR, BinaryOpIR, UnaryOpIR,
    RectangularAccessIR, JaggedAccessIR, MemberAccessIR, TupleAccessIR,
    ArrayLiteralIR, ArrayComprehensionIR,
    BlockExpressionIR, IfExpressionIR, MatchExpressionIR,
    LambdaIR, ArrowExpressionIR, PipelineExpressionIR,
    CastExpressionIR, InterpolatedStringIR, TupleExpressionIR,
    ReductionExpressionIR, WhereExpressionIR,
    EinsteinDeclarationIR, VariableDeclarationIR,
    RangeIR, TryExpressionIR, ConstantDefIR,
    is_function_binding,
)
from ..shared.defid import DefId

logger = logging.getLogger(__name__)


def _collect_defid_refs(node, refs: Set[DefId]) -> None:
    """Recursively collect all function DefIds referenced from an IR node."""
    if node is None:
        return

    if isinstance(node, FunctionCallIR):
        for arg in node.arguments:
            _collect_defid_refs(arg, refs)
        _collect_defid_refs(getattr(node, 'callee_expr', None), refs)
        return

    if isinstance(node, IdentifierIR) and getattr(node, 'defid', None) is not None:
        refs.add(node.defid)
        return

    if isinstance(node, BuiltinCallIR):
        for arg in (getattr(node, 'args', None) or getattr(node, 'arguments', None) or []):
            _collect_defid_refs(arg, refs)
        return

    if isinstance(node, BlockExpressionIR):
        for stmt in (node.statements or []):
            _collect_defid_refs(stmt, refs)
        _collect_defid_refs(node.final_expr, refs)
        return

    if isinstance(node, VariableDeclarationIR):
        _collect_defid_refs(getattr(node, 'value', None), refs)
        return

    if isinstance(node, IfExpressionIR):
        _collect_defid_refs(node.condition, refs)
        _collect_defid_refs(node.then_expr, refs)
        _collect_defid_refs(node.else_expr, refs)
        return

    if isinstance(node, BinaryOpIR):
        _collect_defid_refs(node.left, refs)
        _collect_defid_refs(node.right, refs)
        return

    if isinstance(node, UnaryOpIR):
        _collect_defid_refs(node.operand, refs)
        return

    if isinstance(node, RectangularAccessIR):
        _collect_defid_refs(node.array, refs)
        for idx in (node.indices or []):
            _collect_defid_refs(idx, refs)
        return

    if isinstance(node, JaggedAccessIR):
        _collect_defid_refs(getattr(node, 'array', None), refs)
        _collect_defid_refs(getattr(node, 'index', None), refs)
        return

    if isinstance(node, MemberAccessIR):
        _collect_defid_refs(node.object, refs)
        return

    if isinstance(node, TupleAccessIR):
        _collect_defid_refs(getattr(node, 'tuple_expr', None), refs)
        return

    if isinstance(node, CastExpressionIR):
        _collect_defid_refs(node.expr, refs)
        return

    if isinstance(node, ArrayLiteralIR):
        for elem in (node.elements or []):
            _collect_defid_refs(elem, refs)
        return

    if isinstance(node, ArrayComprehensionIR):
        _collect_defid_refs(getattr(node, 'body', None), refs)
        for r in getattr(node, 'ranges', None) or []:
            _collect_defid_refs(r, refs)
        for v in getattr(node, 'loop_vars', None) or []:
            _collect_defid_refs(v, refs)
        for c in getattr(node, 'constraints', None) or []:
            _collect_defid_refs(c, refs)
        return

    if isinstance(node, TupleExpressionIR):
        for elem in (node.elements or []):
            _collect_defid_refs(elem, refs)
        return

    if isinstance(node, ReductionExpressionIR):
        _collect_defid_refs(node.body, refs)
        return

    if isinstance(node, WhereExpressionIR):
        _collect_defid_refs(node.expr, refs)
        _collect_defid_refs(getattr(node, 'condition', None), refs)
        return

    if isinstance(node, EinsteinDeclarationIR):
        for clause in (getattr(node, 'clauses', None) or []):
            _collect_defid_refs(getattr(clause, 'value', None), refs)
        return

    if isinstance(node, LambdaIR):
        _collect_defid_refs(node.body, refs)
        return

    if isinstance(node, ArrowExpressionIR):
        _collect_defid_refs(getattr(node, 'body', None), refs)
        return

    if isinstance(node, PipelineExpressionIR):
        _collect_defid_refs(getattr(node, 'left', None), refs)
        _collect_defid_refs(getattr(node, 'right', None), refs)
        return

    if isinstance(node, InterpolatedStringIR):
        for part in (node.parts or []):
            _collect_defid_refs(part, refs)
        return

    if isinstance(node, MatchExpressionIR):
        _collect_defid_refs(getattr(node, 'scrutinee', None), refs)
        for arm in (getattr(node, 'arms', None) or []):
            _collect_defid_refs(getattr(arm, 'body', None), refs)
            _collect_defid_refs(getattr(arm, 'guard', None), refs)
        return

    if isinstance(node, TryExpressionIR):
        _collect_defid_refs(getattr(node, 'body', None), refs)
        _collect_defid_refs(getattr(node, 'handler', None), refs)
        return

    if isinstance(node, RangeIR):
        _collect_defid_refs(node.start, refs)
        _collect_defid_refs(node.end, refs)
        return

    # LoweredEinsteinIR / LoweredEinsteinClauseIR
    tn = type(node).__name__
    if tn == 'LoweredEinsteinIR':
        for item in (getattr(node, 'items', None) or []):
            _collect_defid_refs(item, refs)
        for s in (getattr(node, 'shape', None) or []):
            _collect_defid_refs(s, refs)
        return

    if tn == 'LoweredEinsteinClauseIR':
        _collect_defid_refs(getattr(node, 'body', None), refs)
        for loop in (getattr(node, 'loops', None) or []):
            _collect_defid_refs(getattr(loop, 'iterable', None), refs)
        for b in (getattr(node, 'bindings', None) or []):
            _collect_defid_refs(getattr(b, 'value', None) or getattr(b, 'expr', None), refs)
        for g in (getattr(node, 'guards', None) or []):
            _collect_defid_refs(getattr(g, 'condition', None), refs)
        return

    if tn == 'LoweredReductionIR':
        _collect_defid_refs(getattr(node, 'body', None), refs)
        for loop in (getattr(node, 'loops', None) or []):
            _collect_defid_refs(getattr(loop, 'iterable', None), refs)
        for b in (getattr(node, 'bindings', None) or []):
            _collect_defid_refs(getattr(b, 'value', None) or getattr(b, 'expr', None), refs)
        for g in (getattr(node, 'guards', None) or []):
            _collect_defid_refs(getattr(g, 'condition', None), refs)
        return

    if tn == 'LoweredComprehensionIR':
        _collect_defid_refs(getattr(node, 'body', None), refs)
        _collect_defid_refs(getattr(node, 'iterable', None), refs)
        return


def tree_shake(ir: ProgramIR) -> ProgramIR:
    """Remove unreachable functions from the program IR.

    1. Seed the reachable set from top-level statements.
    2. Transitively follow FunctionCallIR / IdentifierIR (callee) edges.
    3. Filter ProgramIR.functions to the reachable set.
    """
    func_by_defid = {}
    for func in ir.functions:
        did = getattr(func, 'defid', None)
        if did is not None:
            func_by_defid[did] = func

    reachable: Set[DefId] = set()

    # Seed: top-level statements
    for stmt in (ir.statements or []):
        _collect_defid_refs(stmt, reachable)

    # Transitive closure
    worklist = list(reachable & set(func_by_defid.keys()))
    visited: Set[DefId] = set()
    while worklist:
        did = worklist.pop()
        if did in visited:
            continue
        visited.add(did)
        func = func_by_defid.get(did)
        if func is None:
            continue
        new_refs: Set[DefId] = set()
        _collect_defid_refs(func.body, new_refs)
        for ref in new_refs:
            reachable.add(ref)
            if ref not in visited and ref in func_by_defid:
                worklist.append(ref)

    before = len(ir.functions)
    kept_defids = {f.defid for f in ir.functions
                   if getattr(f, 'defid', None) is not None and f.defid in reachable}
    kept = [f for f in ir.functions
            if getattr(f, 'defid', None) is not None and f.defid in kept_defids]
    after = len(kept)

    if before != after:
        logger.debug(f"[TreeShaking] {before} → {after} functions ({before - after} pruned)")

    new_statements = [s for s in (ir.statements or [])
                      if not is_function_binding(s) or (getattr(s, 'defid', None) in kept_defids)]
    object.__setattr__(ir, 'statements', new_statements)
    object.__setattr__(ir, 'bindings', [s for s in new_statements if isinstance(s, BindingIR)])
    return ir
