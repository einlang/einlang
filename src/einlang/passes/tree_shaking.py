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
from typing import Any, Set

from ..ir.nodes import (
    ProgramIR, BindingIR,
    FunctionCallIR, BuiltinCallIR,
    IdentifierIR, BinaryOpIR, UnaryOpIR,
    RectangularAccessIR, JaggedAccessIR, MemberAccessIR, TupleAccessIR,
    ArrayLiteralIR, ArrayComprehensionIR,
    BlockExpressionIR, IfExpressionIR, MatchExpressionIR,
    LambdaIR, PipelineExpressionIR,
    CastExpressionIR, InterpolatedStringIR, TupleExpressionIR,
    ReductionExpressionIR, WhereExpressionIR,
    RangeIR, TryExpressionIR,
    IRVisitor,
    is_function_binding, is_einstein_binding,
)
from ..shared.defid import DefId

logger = logging.getLogger(__name__)


def _visit_opt(visitor: "DefidRefsCollector", node: Any) -> None:
    if node is not None and hasattr(node, "accept"):
        node.accept(visitor)


def _visit_many(visitor: "DefidRefsCollector", nodes: Any) -> None:
    if not nodes:
        return
    for n in nodes:
        if n is not None and hasattr(n, "accept"):
            n.accept(visitor)


class DefidRefsCollector(IRVisitor[None]):
    """Recursively collect all function DefIds referenced from an IR node into _refs."""

    def __init__(self, refs: Set[DefId]) -> None:
        self._refs = refs

    def visit_function_call(self, node: FunctionCallIR) -> None:
        _visit_many(self, node.arguments)
        _visit_opt(self, getattr(node, "callee_expr", None))

    def visit_identifier(self, node: IdentifierIR) -> None:
        did = getattr(node, "defid", None)
        if did is not None:
            self._refs.add(did)

    def visit_builtin_call(self, node: BuiltinCallIR) -> None:
        args = getattr(node, "args", None) or getattr(node, "arguments", None) or []
        _visit_many(self, args)

    def visit_block_expression(self, node: BlockExpressionIR) -> None:
        _visit_many(self, node.statements)
        _visit_opt(self, node.final_expr)

    def visit_binding(self, node: BindingIR) -> None:
        _visit_opt(self, getattr(node, "value", None) or getattr(node, "expr", None))
        if is_einstein_binding(node):
            expr = getattr(node, "expr", None)
            clauses = getattr(expr, "clauses", None) or []
            for clause in clauses:
                _visit_opt(self, getattr(clause, "value", None))

    def visit_if_expression(self, node: IfExpressionIR) -> None:
        node.condition.accept(self)
        node.then_expr.accept(self)
        _visit_opt(self, node.else_expr)

    def visit_binary_op(self, node: BinaryOpIR) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_unary_op(self, node: UnaryOpIR) -> None:
        node.operand.accept(self)

    def visit_rectangular_access(self, node: RectangularAccessIR) -> None:
        node.array.accept(self)
        _visit_many(self, node.indices)

    def visit_jagged_access(self, node: JaggedAccessIR) -> None:
        base = getattr(node, "base", None) or getattr(node, "array", None)
        _visit_opt(self, base)
        chain = getattr(node, "index_chain", None)
        if chain is not None:
            _visit_many(self, chain)
        else:
            _visit_opt(self, getattr(node, "index", None))

    def visit_member_access(self, node: MemberAccessIR) -> None:
        node.object.accept(self)

    def visit_tuple_access(self, node: TupleAccessIR) -> None:
        _visit_opt(self, getattr(node, "tuple_expr", None))

    def visit_cast_expression(self, node: CastExpressionIR) -> None:
        node.expr.accept(self)

    def visit_array_literal(self, node: ArrayLiteralIR) -> None:
        _visit_many(self, node.elements)

    def visit_array_comprehension(self, node: ArrayComprehensionIR) -> None:
        _visit_opt(self, getattr(node, "body", None))
        _visit_many(self, getattr(node, "ranges", None))
        _visit_many(self, getattr(node, "loop_vars", None))
        _visit_many(self, getattr(node, "constraints", None))

    def visit_tuple_expression(self, node: TupleExpressionIR) -> None:
        _visit_many(self, node.elements)

    def visit_reduction_expression(self, node: Any) -> None:
        node.body.accept(self)

    def visit_where_expression(self, node: WhereExpressionIR) -> None:
        node.expr.accept(self)
        _visit_many(self, getattr(node, "constraints", None))

    def visit_lambda(self, node: LambdaIR) -> None:
        node.body.accept(self)

    def visit_pipeline_expression(self, node: PipelineExpressionIR) -> None:
        _visit_opt(self, getattr(node, "left", None))
        _visit_opt(self, getattr(node, "right", None))

    def visit_interpolated_string(self, node: InterpolatedStringIR) -> None:
        _visit_many(self, node.parts)

    def visit_match_expression(self, node: MatchExpressionIR) -> None:
        node.scrutinee.accept(self)
        for arm in getattr(node, "arms", None) or []:
            _visit_opt(self, getattr(arm, "body", None))
            _visit_opt(self, getattr(arm, "guard", None))

    def visit_try_expression(self, node: TryExpressionIR) -> None:
        _visit_opt(self, getattr(node, "operand", None))

    def visit_range(self, node: RangeIR) -> None:
        node.start.accept(self)
        node.end.accept(self)

    def visit_lowered_einstein(self, node: Any) -> None:
        _visit_many(self, getattr(node, "items", None))
        _visit_many(self, getattr(node, "shape", None))

    def visit_lowered_einstein_clause(self, node: Any) -> None:
        _visit_opt(self, getattr(node, "body", None))
        for loop in getattr(node, "loops", None) or []:
            _visit_opt(self, getattr(loop, "iterable", None))
        for b in getattr(node, "bindings", None) or []:
            _visit_opt(self, getattr(b, "value", None) or getattr(b, "expr", None))
        for g in getattr(node, "guards", None) or []:
            _visit_opt(self, getattr(g, "condition", None))

    def visit_lowered_reduction(self, node: Any) -> None:
        _visit_opt(self, getattr(node, "body", None))
        for loop in getattr(node, "loops", None) or []:
            _visit_opt(self, getattr(loop, "iterable", None))
        for b in getattr(node, "bindings", None) or []:
            _visit_opt(self, getattr(b, "value", None) or getattr(b, "expr", None))
        for g in getattr(node, "guards", None) or []:
            _visit_opt(self, getattr(g, "condition", None))

    def visit_lowered_comprehension(self, node: Any) -> None:
        _visit_opt(self, getattr(node, "body", None))
        _visit_opt(self, getattr(node, "iterable", None))

    # No-op / no DefIds for the rest
    def visit_literal(self, node: Any) -> None:
        pass

    def visit_index_var(self, node: Any) -> None:
        _visit_opt(self, getattr(node, "range_ir", None))

    def visit_index_rest(self, node: Any) -> None:
        pass

    def visit_module(self, node: Any) -> None:
        pass

    def visit_program(self, node: ProgramIR) -> None:
        _visit_many(self, node.statements)

    def visit_literal_pattern(self, node: Any) -> None:
        pass

    def visit_identifier_pattern(self, node: Any) -> None:
        pass

    def visit_wildcard_pattern(self, node: Any) -> None:
        pass

    def visit_tuple_pattern(self, node: Any) -> None:
        _visit_many(self, getattr(node, "elements", None))

    def visit_array_pattern(self, node: Any) -> None:
        _visit_many(self, getattr(node, "elements", None))

    def visit_rest_pattern(self, node: Any) -> None:
        pass

    def visit_guard_pattern(self, node: Any) -> None:
        inner = getattr(node, "inner_pattern", None)
        guard_expr = getattr(node, "guard_expr", None)
        _visit_opt(self, inner)
        _visit_opt(self, guard_expr)

    def visit_or_pattern(self, node: Any) -> None:
        _visit_many(self, getattr(node, "alternatives", None))

    def visit_constructor_pattern(self, node: Any) -> None:
        _visit_many(self, getattr(node, "patterns", None))

    def visit_binding_pattern(self, node: Any) -> None:
        _visit_opt(self, getattr(node, "inner_pattern", None))

    def visit_range_pattern(self, node: Any) -> None:
        pass

    def visit_function_value(self, node: Any) -> None:
        _visit_opt(self, getattr(node, "body", None))

    def visit_einstein(self, node: Any) -> None:
        pass

    def visit_einstein_clause(self, node: Any) -> None:
        pass

    def visit_lowered_recurrence(self, node: Any) -> None:
        _visit_opt(self, getattr(node, "initial", None))
        rec_loop = getattr(node, "recurrence_loop", None)
        _visit_opt(self, getattr(rec_loop, "iterable", None) if rec_loop else None)
        _visit_opt(self, getattr(node, "body", None))


def _collect_defid_refs(node: Any, refs: Set[DefId]) -> None:
    """Recursively collect all function DefIds referenced from an IR node."""
    if node is None:
        return
    if hasattr(node, "accept"):
        node.accept(DefidRefsCollector(refs))


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
