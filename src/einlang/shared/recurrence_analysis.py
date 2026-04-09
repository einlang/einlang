"""Shared recurrence-analysis helpers for compiler passes and runtimes.

These helpers operate only on generic IR structure and are intentionally kept
out of `passes/` and `backends/` so either side can depend on them.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..ir.nodes import (
    BinaryOpIR,
    IRVisitor,
    LiteralIR,
    RangeIR,
)
from ..shared.types import BinaryOp


class _BodyReferencesDefidVisitor(IRVisitor[bool]):
    """True iff the tree contains an IdentifierIR or IndexVarIR for target defid."""

    def __init__(self, target_defid: Any) -> None:
        self._target = target_defid

    def references(self, expr: Any) -> bool:
        if expr is None or self._target is None:
            return False
        return expr.accept(self)

    def _any(self, *nodes: Any) -> bool:
        for node in nodes:
            if node is not None and node.accept(self):
                return True
        return False

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_identifier(self, node: Any) -> bool:
        return node.defid == self._target

    def visit_index_var(self, node: Any) -> bool:
        return node.defid == self._target

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_binary_op(self, node: Any) -> bool:
        return self._any(node.left, node.right)

    def visit_function_call(self, node: Any) -> bool:
        return self._any(node.callee_expr, *(node.arguments or []))

    def visit_rectangular_access(self, node: Any) -> bool:
        if self._any(node.array):
            return True
        return self._any(*(node.indices or []))

    def visit_jagged_access(self, node: Any) -> bool:
        return self._any(node.base, *((node.index_chain or [])))

    def visit_block_expression(self, node: Any) -> bool:
        for stmt in (node.statements or []):
            if self._any(stmt):
                return True
        return self._any(node.final_expr)

    def visit_if_expression(self, node: Any) -> bool:
        return self._any(node.condition, node.then_expr, node.else_expr)

    def visit_lambda(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_unary_op(self, node: Any) -> bool:
        return self._any(node.operand)

    def visit_range(self, node: Any) -> bool:
        return self._any(node.start, node.end)

    def visit_array_comprehension(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return self._any(*(node.elements or []))

    def visit_tuple_expression(self, node: Any) -> bool:
        return self._any(*(node.elements or []))

    def visit_tuple_access(self, node: Any) -> bool:
        return self._any(node.tuple_expr)

    def visit_interpolated_string(self, node: Any) -> bool:
        return self._any(*(node.parts or []))

    def visit_cast_expression(self, node: Any) -> bool:
        return self._any(node.expr)

    def visit_member_access(self, node: Any) -> bool:
        return self._any(node.object)

    def visit_try_expression(self, node: Any) -> bool:
        return self._any(node.operand)

    def visit_match_expression(self, node: Any) -> bool:
        if self._any(node.scrutinee):
            return True
        for arm in (node.arms or []):
            if self._any(arm.body):
                return True
        return False

    def visit_reduction_expression(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_lowered_reduction(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_where_expression(self, node: Any) -> bool:
        return self._any(node.expr, *(node.constraints or []))

    def visit_pipeline_expression(self, node: Any) -> bool:
        return self._any(node.left, node.right)

    def visit_builtin_call(self, node: Any) -> bool:
        return self._any(*(node.args or []))

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return self._any(*(node.patterns or []))

    def visit_array_pattern(self, node: Any) -> bool:
        return self._any(*(node.patterns or []))

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return self._any(node.inner_pattern, node.guard_expr)

    def visit_binding(self, node: Any) -> bool:
        return self._any(node.expr)

    def visit_program(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_lowered_einstein(self, node: Any) -> bool:
        return self._any(*(node.items or []))

    def visit_lowered_recurrence(self, node: Any) -> bool:
        return self._any(node.initial, node.body, node.recurrence_loop)

    def visit_or_pattern(self, node: Any) -> bool:
        return self._any(*(node.alternatives or []))

    def visit_constructor_pattern(self, node: Any) -> bool:
        return self._any(*(node.patterns or []))

    def visit_binding_pattern(self, node: Any) -> bool:
        return self._any(node.inner_pattern)

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_function_value(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_einstein(self, node: Any) -> bool:
        return self._any(*(node.clauses or []))

    def visit_einstein_clause(self, node: Any) -> bool:
        return self._any(node.value, node.where_clause)


class _IndexExprIsLoopVarVisitor(IRVisitor[bool]):
    """True if expr is exactly the loop variable."""

    def __init__(self, loop_defid: Any) -> None:
        self._loop_defid = loop_defid

    def visit_identifier(self, node: Any) -> bool:
        return node.defid == self._loop_defid

    def visit_index_var(self, node: Any) -> bool:
        return node.defid == self._loop_defid

    def visit_binary_op(self, node: Any) -> bool:
        return False

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_rectangular_access(self, node: Any) -> bool:
        return False

    def visit_function_call(self, node: Any) -> bool:
        return False

    def visit_unary_op(self, node: Any) -> bool:
        return False

    def visit_block_expression(self, node: Any) -> bool:
        return False

    def visit_if_expression(self, node: Any) -> bool:
        return False

    def visit_lambda(self, node: Any) -> bool:
        return False

    def visit_range(self, node: Any) -> bool:
        return False

    def visit_array_comprehension(self, node: Any) -> bool:
        return False

    def visit_jagged_access(self, node: Any) -> bool:
        return False

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return False

    def visit_tuple_expression(self, node: Any) -> bool:
        return False

    def visit_tuple_access(self, node: Any) -> bool:
        return False

    def visit_interpolated_string(self, node: Any) -> bool:
        return False

    def visit_cast_expression(self, node: Any) -> bool:
        return False

    def visit_member_access(self, node: Any) -> bool:
        return False

    def visit_try_expression(self, node: Any) -> bool:
        return False

    def visit_match_expression(self, node: Any) -> bool:
        return False

    def visit_reduction_expression(self, node: Any) -> bool:
        return False

    def visit_where_expression(self, node: Any) -> bool:
        return False

    def visit_pipeline_expression(self, node: Any) -> bool:
        return False

    def visit_builtin_call(self, node: Any) -> bool:
        return False

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return False

    def visit_array_pattern(self, node: Any) -> bool:
        return False

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return False

    def visit_binding(self, node: Any) -> bool:
        return False

    def visit_program(self, node: Any) -> bool:
        return False

    def visit_lowered_reduction(self, node: Any) -> bool:
        return False

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein(self, node: Any) -> bool:
        return False

    def visit_lowered_recurrence(self, node: Any) -> bool:
        return False

    def visit_or_pattern(self, node: Any) -> bool:
        return False

    def visit_constructor_pattern(self, node: Any) -> bool:
        return False

    def visit_binding_pattern(self, node: Any) -> bool:
        return False

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_function_value(self, node: Any) -> bool:
        return False

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False


def _index_expr_is_loop_var(expr: Any, loop_defid: Any) -> bool:
    if expr is None or loop_defid is None:
        return False
    return expr.accept(_IndexExprIsLoopVarVisitor(loop_defid))


class _IndexExprIsLoopVarOrOffsetVisitor(IRVisitor[bool]):
    """True if expr is loop_var or (loop_var +/- const)."""

    def __init__(self, loop_defid: Any) -> None:
        self._loop_defid = loop_defid

    def visit_identifier(self, node: Any) -> bool:
        return node.defid == self._loop_defid

    def visit_index_var(self, node: Any) -> bool:
        return node.defid == self._loop_defid

    def visit_binary_op(self, node: Any) -> bool:
        if node.operator not in (BinaryOp.ADD, BinaryOp.SUB):
            return False
        if not _index_expr_is_loop_var(node.left, self._loop_defid):
            return False
        if isinstance(node.right, LiteralIR):
            try:
                int(node.right.value)
                return True
            except (TypeError, ValueError):
                return False
        return False

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_rectangular_access(self, node: Any) -> bool:
        return False

    def visit_function_call(self, node: Any) -> bool:
        return False

    def visit_unary_op(self, node: Any) -> bool:
        return False

    def visit_block_expression(self, node: Any) -> bool:
        return False

    def visit_if_expression(self, node: Any) -> bool:
        return False

    def visit_lambda(self, node: Any) -> bool:
        return False

    def visit_range(self, node: Any) -> bool:
        return False

    def visit_array_comprehension(self, node: Any) -> bool:
        return False

    def visit_jagged_access(self, node: Any) -> bool:
        return False

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return False

    def visit_tuple_expression(self, node: Any) -> bool:
        return False

    def visit_tuple_access(self, node: Any) -> bool:
        return False

    def visit_interpolated_string(self, node: Any) -> bool:
        return False

    def visit_cast_expression(self, node: Any) -> bool:
        return False

    def visit_member_access(self, node: Any) -> bool:
        return False

    def visit_try_expression(self, node: Any) -> bool:
        return False

    def visit_match_expression(self, node: Any) -> bool:
        return False

    def visit_reduction_expression(self, node: Any) -> bool:
        return False

    def visit_where_expression(self, node: Any) -> bool:
        return False

    def visit_pipeline_expression(self, node: Any) -> bool:
        return False

    def visit_builtin_call(self, node: Any) -> bool:
        return False

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return False

    def visit_array_pattern(self, node: Any) -> bool:
        return False

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return False

    def visit_binding(self, node: Any) -> bool:
        return False

    def visit_program(self, node: Any) -> bool:
        return False

    def visit_lowered_reduction(self, node: Any) -> bool:
        return False

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein(self, node: Any) -> bool:
        return False

    def visit_lowered_recurrence(self, node: Any) -> bool:
        return False

    def visit_or_pattern(self, node: Any) -> bool:
        return False

    def visit_constructor_pattern(self, node: Any) -> bool:
        return False

    def visit_binding_pattern(self, node: Any) -> bool:
        return False

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_function_value(self, node: Any) -> bool:
        return False

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False


def _index_expr_is_loop_var_or_offset(expr: Any, loop_defid: Any) -> bool:
    if expr is None or loop_defid is None:
        return False
    return expr.accept(_IndexExprIsLoopVarOrOffsetVisitor(loop_defid))


class _ExprIsLoopVarOrMinusOneVisitor(IRVisitor[bool]):
    """True if expr is loop_var or (loop_var - 1)."""

    def __init__(self, loop_defid: Any) -> None:
        self._loop_defid = loop_defid

    def visit_identifier(self, node: Any) -> bool:
        return node.defid == self._loop_defid

    def visit_index_var(self, node: Any) -> bool:
        return node.defid == self._loop_defid

    def visit_binary_op(self, node: Any) -> bool:
        if node.operator != BinaryOp.SUB:
            return False
        if not _index_expr_is_loop_var(node.left, self._loop_defid):
            return False
        if isinstance(node.right, LiteralIR):
            try:
                return int(node.right.value) == 1
            except (TypeError, ValueError):
                return False
        return False

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_rectangular_access(self, node: Any) -> bool:
        return False

    def visit_function_call(self, node: Any) -> bool:
        return False

    def visit_unary_op(self, node: Any) -> bool:
        return False

    def visit_block_expression(self, node: Any) -> bool:
        return False

    def visit_if_expression(self, node: Any) -> bool:
        return False

    def visit_lambda(self, node: Any) -> bool:
        return False

    def visit_range(self, node: Any) -> bool:
        return False

    def visit_array_comprehension(self, node: Any) -> bool:
        return False

    def visit_jagged_access(self, node: Any) -> bool:
        return False

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return False

    def visit_tuple_expression(self, node: Any) -> bool:
        return False

    def visit_tuple_access(self, node: Any) -> bool:
        return False

    def visit_interpolated_string(self, node: Any) -> bool:
        return False

    def visit_cast_expression(self, node: Any) -> bool:
        return False

    def visit_member_access(self, node: Any) -> bool:
        return False

    def visit_try_expression(self, node: Any) -> bool:
        return False

    def visit_match_expression(self, node: Any) -> bool:
        return False

    def visit_reduction_expression(self, node: Any) -> bool:
        return False

    def visit_where_expression(self, node: Any) -> bool:
        return False

    def visit_pipeline_expression(self, node: Any) -> bool:
        return False

    def visit_builtin_call(self, node: Any) -> bool:
        return False

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return False

    def visit_array_pattern(self, node: Any) -> bool:
        return False

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return False

    def visit_binding(self, node: Any) -> bool:
        return False

    def visit_program(self, node: Any) -> bool:
        return False

    def visit_lowered_reduction(self, node: Any) -> bool:
        return False

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein(self, node: Any) -> bool:
        return False

    def visit_lowered_recurrence(self, node: Any) -> bool:
        return False

    def visit_or_pattern(self, node: Any) -> bool:
        return False

    def visit_constructor_pattern(self, node: Any) -> bool:
        return False

    def visit_binding_pattern(self, node: Any) -> bool:
        return False

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_function_value(self, node: Any) -> bool:
        return False

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False


def _expr_is_loop_var_or_minus_one(expr: Any, loop_defid: Any) -> bool:
    if expr is None or loop_defid is None:
        return False
    return expr.accept(_ExprIsLoopVarOrMinusOneVisitor(loop_defid))


class _LHSReadIndexListsCollector(IRVisitor[List[List[Any]]]):
    """Collect index lists from accesses whose array references target_defid."""

    def __init__(self, target_defid: Any) -> None:
        self._target_defid = target_defid

    def _empty(self) -> List[List[Any]]:
        return []

    def visit_rectangular_access(self, node: Any) -> List[List[Any]]:
        out: List[List[Any]] = []
        if _BodyReferencesDefidVisitor(self._target_defid).references(node.array):
            if node.indices:
                out.append(list(node.indices))
        out.extend(node.array.accept(self))
        return out

    def visit_binary_op(self, node: Any) -> List[List[Any]]:
        out = node.left.accept(self)
        out.extend(node.right.accept(self))
        return out

    def visit_function_call(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for arg in node.arguments or []:
            out.extend(arg.accept(self))
        out.extend(node.callee_expr.accept(self))
        return out

    def visit_block_expression(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for stmt in node.statements or []:
            out.extend(stmt.accept(self))
        if node.final_expr is not None:
            out.extend(node.final_expr.accept(self))
        return out

    def visit_binding(self, node: Any) -> List[List[Any]]:
        return node.expr.accept(self)

    def visit_lambda(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_if_expression(self, node: Any) -> List[List[Any]]:
        out = node.condition.accept(self)
        out.extend(node.then_expr.accept(self))
        if node.else_expr is not None:
            out.extend(node.else_expr.accept(self))
        return out

    def visit_literal(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_identifier(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_index_var(self, node: Any) -> List[List[Any]]:
        return node.range_ir.accept(self) if node.range_ir is not None else self._empty()

    def visit_index_rest(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_unary_op(self, node: Any) -> List[List[Any]]:
        return node.operand.accept(self)

    def visit_range(self, node: Any) -> List[List[Any]]:
        out = node.start.accept(self)
        out.extend(node.end.accept(self))
        return out

    def visit_array_comprehension(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_jagged_access(self, node: Any) -> List[List[Any]]:
        out = node.base.accept(self)
        for idx in node.index_chain or []:
            out.extend(idx.accept(self))
        return out

    def visit_module(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_array_literal(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for element in node.elements or []:
            out.extend(element.accept(self))
        return out

    def visit_tuple_expression(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for element in node.elements or []:
            out.extend(element.accept(self))
        return out

    def visit_tuple_access(self, node: Any) -> List[List[Any]]:
        return node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for part in node.parts or []:
            if hasattr(part, "accept"):
                out.extend(part.accept(self))
        return out

    def visit_cast_expression(self, node: Any) -> List[List[Any]]:
        return node.expr.accept(self)

    def visit_member_access(self, node: Any) -> List[List[Any]]:
        return node.object.accept(self)

    def visit_try_expression(self, node: Any) -> List[List[Any]]:
        return node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> List[List[Any]]:
        out = node.scrutinee.accept(self)
        for arm in node.arms or []:
            out.extend(arm.body.accept(self))
        return out

    def visit_reduction_expression(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_where_expression(self, node: Any) -> List[List[Any]]:
        out = node.expr.accept(self)
        for constraint in node.constraints or []:
            out.extend(constraint.accept(self))
        return out

    def visit_pipeline_expression(self, node: Any) -> List[List[Any]]:
        out = node.left.accept(self)
        out.extend(node.right.accept(self))
        return out

    def visit_builtin_call(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for arg in node.args or []:
            out.extend(arg.accept(self))
        return out

    def visit_literal_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_identifier_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_wildcard_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_tuple_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_array_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_rest_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_guard_pattern(self, node: Any) -> List[List[Any]]:
        out = node.inner_pattern.accept(self)
        out.extend(node.guard_expr.accept(self))
        return out

    def visit_program(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for stmt in node.statements or []:
            out.extend(stmt.accept(self))
        return out

    def visit_lowered_reduction(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_lowered_comprehension(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_lowered_einstein_clause(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_lowered_einstein(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for item in node.items or []:
            out.extend(item.accept(self))
        return out

    def visit_lowered_recurrence(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        if node.initial is not None:
            out.extend(node.initial.accept(self))
        if node.recurrence_loop is not None and node.recurrence_loop.iterable is not None:
            out.extend(node.recurrence_loop.iterable.accept(self))
        if node.body is not None:
            out.extend(node.body.accept(self))
        return out

    def visit_or_pattern(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for alt in node.alternatives or []:
            out.extend(alt.accept(self))
        return out

    def visit_constructor_pattern(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for pattern in node.patterns or []:
            out.extend(pattern.accept(self))
        return out

    def visit_binding_pattern(self, node: Any) -> List[List[Any]]:
        return node.inner_pattern.accept(self)

    def visit_range_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_function_value(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self) if node.body is not None else self._empty()

    def visit_einstein(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_einstein_clause(self, node: Any) -> List[List[Any]]:
        return self._empty()


def _collect_lhs_read_index_lists(body: Any, target_defid: Any) -> List[List[Any]]:
    if body is None:
        return []
    return body.accept(_LHSReadIndexListsCollector(target_defid))


def _recurrence_dims(
    lowered: Any,
    variable_defid: Any,
    clause_indices: Optional[List[Any]] = None,
) -> List[int]:
    loops = lowered.loops or []
    if not loops or variable_defid is None:
        return []
    loop_defids = [loop.variable.defid for loop in loops]
    read_index_lists = _collect_lhs_read_index_lists(lowered.body, variable_defid)
    if not read_index_lists:
        return []
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops) if clause_indices else None
    recurrence: List[int] = []
    for k in range(len(loops)):
        out_dim = loop_dims[k] if loop_dims is not None and k < len(loop_dims) else k
        for idx_list in read_index_lists:
            if out_dim >= len(idx_list):
                continue
            if not _index_expr_is_loop_var(idx_list[out_dim], loop_defids[k]):
                recurrence.append(k)
                break
    return recurrence


def _index_expr_is_strictly_backward(expr: Any, loop_defid: Any) -> bool:
    if expr is None or loop_defid is None:
        return False
    if not isinstance(expr, BinaryOpIR) or expr.operator != BinaryOp.SUB:
        return False
    if not _index_expr_is_loop_var(expr.left, loop_defid):
        return False
    if isinstance(expr.right, LiteralIR):
        try:
            return int(expr.right.value) > 0
        except (TypeError, ValueError):
            return False
    return False


def _recurrence_dims_for_hybrid(
    lowered: Any,
    variable_defid: Any,
    clause_indices: Optional[List[Any]] = None,
) -> List[int]:
    loops = lowered.loops or []
    if not loops or variable_defid is None:
        return []
    loop_defids = [loop.variable.defid for loop in loops]
    read_index_lists = _collect_lhs_read_index_lists(lowered.body, variable_defid)
    if not read_index_lists:
        return []
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops) if clause_indices else None
    result: List[int] = []
    for k in range(len(loops)):
        out_dim = loop_dims[k] if loop_dims is not None and k < len(loop_dims) else k
        if all(
            out_dim < len(idx_list)
            and _index_expr_is_strictly_backward(idx_list[out_dim], loop_defids[k])
            for idx_list in read_index_lists
        ):
            result.append(k)
    return result


def _recurrence_dims_for_hybrid_or_full(
    lowered: Any,
    variable_defid: Any,
    clause_indices: Optional[List[Any]] = None,
) -> List[int]:
    loops = lowered.loops or []
    if not loops or variable_defid is None:
        return []
    recurrence_for_hybrid = _recurrence_dims_for_hybrid(lowered, variable_defid, clause_indices)
    if not recurrence_for_hybrid:
        return []
    loop_defids = [loop.variable.defid for loop in loops]
    read_index_lists = _collect_lhs_read_index_lists(lowered.body, variable_defid)
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops) if clause_indices else None
    for k in range(len(loops)):
        if k in recurrence_for_hybrid:
            continue
        out_dim = loop_dims[k] if loop_dims is not None and k < len(loop_dims) else k
        for idx_list in read_index_lists:
            if out_dim >= len(idx_list):
                continue
            if not _index_expr_is_loop_var_or_offset(idx_list[out_dim], loop_defids[k]):
                return _recurrence_dims(lowered, variable_defid, clause_indices)
    return recurrence_for_hybrid


def _reduction_var_bounded_by_loop_var(
    read_index_expr: Any,
    loop_defid: Any,
    reduction_ranges: Any,
) -> bool:
    if reduction_ranges is None or not read_index_expr or loop_defid is None:
        return False
    read_defid = getattr(read_index_expr, "defid", None)
    if read_defid is None:
        return False
    loop_struct = reduction_ranges.get(read_defid) if isinstance(reduction_ranges, dict) else None
    if loop_struct is None:
        return False
    iterable = loop_struct.iterable
    if not isinstance(iterable, RangeIR):
        return False
    return _expr_is_loop_var_or_minus_one(iterable.end, loop_defid)


def _loop_dims_from_clause_indices(clause_indices: List[Any], loops: List[Any]) -> Optional[List[int]]:
    if not clause_indices or not loops:
        return None
    out: List[int] = []
    loop_pos = 0
    for pos, idx in enumerate(clause_indices):
        if isinstance(idx, LiteralIR):
            continue
        if loop_pos >= len(loops):
            return None
        out.append(pos)
        loop_pos += 1
    return out if loop_pos == len(loops) else None


def _literal_numeric_value(expr: Any) -> Optional[float]:
    if not isinstance(expr, LiteralIR):
        return None
    try:
        return float(expr.value)
    except (TypeError, ValueError):
        return None


def _index_expr_offset(expr: Any, loop_defid: Any) -> Optional[int]:
    if _index_expr_is_loop_var(expr, loop_defid):
        return 0
    if not isinstance(expr, BinaryOpIR):
        return None
    if expr.operator == BinaryOp.ADD:
        if _index_expr_is_loop_var(expr.left, loop_defid):
            val = _literal_numeric_value(expr.right)
            return int(val) if val is not None and float(val).is_integer() else None
        if _index_expr_is_loop_var(expr.right, loop_defid):
            val = _literal_numeric_value(expr.left)
            return int(val) if val is not None and float(val).is_integer() else None
        return None
    if expr.operator == BinaryOp.SUB and _index_expr_is_loop_var(expr.left, loop_defid):
        val = _literal_numeric_value(expr.right)
        return -int(val) if val is not None and float(val).is_integer() else None
    return None


def _backward_offset_from_index_expr(expr: Any, loop_defid: Any) -> Optional[int]:
    offset = _index_expr_offset(expr, loop_defid)
    if offset is None or offset >= 0:
        return None
    return -offset

