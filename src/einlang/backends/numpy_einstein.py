"""NumPy backend Einstein execution: variable decl, lowered einstein/clause; env only."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ir.nodes import (
    LiteralIR, RangeIR, LoweredEinsteinIR, LoweredEinsteinClauseIR, LoweredRecurrenceIR,
    LoweredReductionIR, ReductionExpressionIR, BinaryOpIR, UnaryOpIR, RectangularAccessIR, IndexVarIR,
    FunctionCallIR, IdentifierIR, IfExpressionIR, BlockExpressionIR,
    is_function_binding, is_einstein_binding,
    IRVisitor, BindingIR,
)
from ..shared.defid import DefId
from ..shared.types import BinaryOp
from ..utils.config import DEFAULT_EINSTEIN_LOOP_MAX
from .numpy_helpers import _reject_non_lowered


class _BodyReferencesDefidVisitor(IRVisitor[bool]):
    """Visitor that returns True iff the tree contains an IdentifierIR or IndexVarIR with defid == target_defid."""

    def __init__(self, target_defid: Any) -> None:
        self._target = target_defid

    def references(self, expr: Any) -> bool:
        """True if expr (IR with accept) contains any node with defid == self._target."""
        if expr is None or self._target is None:
            return False
        accept = getattr(expr, "accept", None)
        if accept is None:
            return False
        return accept(self)

    def _any(self, *nodes: Any) -> bool:
        for n in nodes:
            if n is not None and getattr(n, "accept", None) is not None:
                if n.accept(self):
                    return True
        return False

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_identifier(self, node: Any) -> bool:
        return getattr(node, "defid", None) == self._target

    def visit_index_var(self, node: Any) -> bool:
        return getattr(node, "defid", None) == self._target

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_binary_op(self, node: Any) -> bool:
        return self._any(getattr(node, "left", None), getattr(node, "right", None))

    def visit_function_call(self, node: Any) -> bool:
        args = getattr(node, "arguments", []) or []
        return self._any(getattr(node, "callee_expr", None), *args)

    def visit_rectangular_access(self, node: Any) -> bool:
        if self._any(getattr(node, "array", None)):
            return True
        for idx in getattr(node, "indices", []) or []:
            if self._any(idx):
                return True
        return False

    def visit_jagged_access(self, node: Any) -> bool:
        return self._any(getattr(node, "array", None), *((getattr(node, "indices", None) or [])))

    def visit_block_expression(self, node: Any) -> bool:
        for stmt in getattr(node, "statements", []) or []:
            if self._any(stmt):
                return True
        return self._any(getattr(node, "final_expr", None))

    def visit_if_expression(self, node: Any) -> bool:
        return self._any(
            getattr(node, "condition", None),
            getattr(node, "then_expr", None),
            getattr(node, "else_expr", None),
        )

    def visit_lambda(self, node: Any) -> bool:
        return self._any(getattr(node, "body", None))

    def visit_unary_op(self, node: Any) -> bool:
        return self._any(getattr(node, "operand", None))

    def visit_range(self, node: Any) -> bool:
        return self._any(getattr(node, "start", None), getattr(node, "end", None))

    def visit_array_comprehension(self, node: Any) -> bool:
        return self._any(getattr(node, "body", None))

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return self._any(*(getattr(node, "elements", []) or []))

    def visit_tuple_expression(self, node: Any) -> bool:
        return self._any(*(getattr(node, "elements", []) or []))

    def visit_tuple_access(self, node: Any) -> bool:
        return self._any(getattr(node, "tuple_expr", None))

    def visit_interpolated_string(self, node: Any) -> bool:
        return self._any(*(getattr(node, "parts", []) or []))

    def visit_cast_expression(self, node: Any) -> bool:
        return self._any(getattr(node, "expr", None))

    def visit_member_access(self, node: Any) -> bool:
        return self._any(getattr(node, "object_expr", None))

    def visit_try_expression(self, node: Any) -> bool:
        return self._any(
            getattr(node, "try_expr", None),
            getattr(node, "else_expr", None),
        )

    def visit_match_expression(self, node: Any) -> bool:
        if self._any(getattr(node, "scrutinee", None)):
            return True
        for arm in getattr(node, "arms", []) or []:
            if self._any(getattr(arm, "body", None)):
                return True
        return False

    def visit_reduction_expression(self, node: Any) -> bool:
        return self._any(getattr(node, "body", None))

    def visit_lowered_reduction(self, node: Any) -> bool:
        return self._any(getattr(node, "body", None))

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return self._any(getattr(node, "body", None))

    def visit_where_expression(self, node: Any) -> bool:
        return self._any(getattr(node, "operand", None), getattr(node, "condition", None))

    def visit_pipeline_expression(self, node: Any) -> bool:
        return self._any(*(getattr(node, "stages", []) or []))

    def visit_builtin_call(self, node: Any) -> bool:
        return self._any(getattr(node, "callee_expr", None), *(getattr(node, "arguments", []) or []))

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return self._any(*(getattr(node, "elements", []) or []))

    def visit_array_pattern(self, node: Any) -> bool:
        return self._any(*(getattr(node, "elements", []) or []))

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return self._any(getattr(node, "inner_pattern", None), getattr(node, "condition", None))

    def visit_or_pattern(self, node: Any) -> bool:
        for alt in getattr(node, "alternatives", []) or []:
            if self._any(alt):
                return True
        return False

    def visit_constructor_pattern(self, node: Any) -> bool:
        return self._any(*(getattr(node, "patterns", []) or []))

    def visit_binding_pattern(self, node: Any) -> bool:
        return self._any(getattr(node, "inner_pattern", None))

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_function_value(self, node: Any) -> bool:
        return self._any(getattr(node, "body", None))

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False

    def visit_binding(self, node: Any) -> bool:
        """Recurse into binding RHS (e.g. let z_cell = ... * state[t-1,...] + ...)."""
        expr = getattr(node, "expr", None)
        return self._any(expr) if expr is not None else False

    def visit_program(self, node: Any) -> bool:
        return self._any(*(getattr(node, "statements", []) or []))


class _ReductionDimsCounter(IRVisitor[int]):
    """Visitor that returns the max number of reduction dimensions in any LoweredReductionIR in the tree (0 if none)."""

    def visit_lowered_reduction(self, node: Any) -> int:
        loops = node.loops or []
        body_count = node.body.accept(self)
        return max(len(loops), body_count)

    def visit_binary_op(self, node: Any) -> int:
        return max(node.left.accept(self), node.right.accept(self))

    def visit_rectangular_access(self, node: Any) -> int:
        n = node.array.accept(self)
        for idx in node.indices:
            n = max(n, idx.accept(self))
        return n

    def visit_function_call(self, node: Any) -> int:
        n = node.callee_expr.accept(self)
        for a in node.arguments:
            n = max(n, a.accept(self))
        return n

    def visit_unary_op(self, node: Any) -> int:
        return node.operand.accept(self)

    def visit_if_expression(self, node: Any) -> int:
        c = node.condition.accept(self)
        t = node.then_expr.accept(self)
        e = node.else_expr.accept(self) if node.else_expr is not None else 0
        return max(c, t, e)

    def visit_block_expression(self, node: Any) -> int:
        n = 0
        for stmt in node.statements:
            n = max(n, stmt.accept(self))
        final = node.final_expr.accept(self) if node.final_expr is not None else 0
        return max(n, final)

    def visit_lambda(self, node: Any) -> int:
        return node.body.accept(self)

    def visit_range(self, node: Any) -> int:
        return max(node.start.accept(self), node.end.accept(self))

    def visit_array_comprehension(self, node: Any) -> int:
        return node.body.accept(self)

    def visit_where_expression(self, node: Any) -> int:
        n = node.expr.accept(self)
        for c in node.constraints:
            n = max(n, c.accept(self))
        return n

    def visit_pipeline_expression(self, node: Any) -> int:
        return max(node.left.accept(self), node.right.accept(self))

    def visit_builtin_call(self, node: Any) -> int:
        n = 0
        for a in node.args:
            n = max(n, a.accept(self))
        return n

    def visit_literal(self, node: Any) -> int:
        return 0

    def visit_identifier(self, node: Any) -> int:
        return 0

    def visit_index_var(self, node: Any) -> int:
        return node.range_ir.accept(self) if node.range_ir is not None else 0

    def visit_index_rest(self, node: Any) -> int:
        return 0

    def visit_jagged_access(self, node: Any) -> int:
        n = node.base.accept(self)
        for idx in node.index_chain:
            n = max(n, idx.accept(self))
        return n

    def visit_module(self, node: Any) -> int:
        return 0

    def visit_array_literal(self, node: Any) -> int:
        n = 0
        for e in node.elements:
            n = max(n, e.accept(self))
        return n

    def visit_tuple_expression(self, node: Any) -> int:
        n = 0
        for e in node.elements:
            n = max(n, e.accept(self))
        return n

    def visit_tuple_access(self, node: Any) -> int:
        return node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> int:
        n = 0
        for p in node.parts:
            n = max(n, p.accept(self))
        return n

    def visit_cast_expression(self, node: Any) -> int:
        return node.expr.accept(self)

    def visit_member_access(self, node: Any) -> int:
        return node.object.accept(self)

    def visit_try_expression(self, node: Any) -> int:
        return node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> int:
        n = node.scrutinee.accept(self)
        for arm in node.arms:
            n = max(n, arm.body.accept(self))
        return n

    def visit_reduction_expression(self, node: Any) -> int:
        return node.body.accept(self)

    def visit_literal_pattern(self, node: Any) -> int:
        return 0

    def visit_identifier_pattern(self, node: Any) -> int:
        return 0

    def visit_wildcard_pattern(self, node: Any) -> int:
        return 0

    def visit_tuple_pattern(self, node: Any) -> int:
        n = 0
        for e in node.elements:
            n = max(n, e.accept(self))
        return n

    def visit_array_pattern(self, node: Any) -> int:
        n = 0
        for e in node.elements:
            n = max(n, e.accept(self))
        return n

    def visit_rest_pattern(self, node: Any) -> int:
        return 0

    def visit_guard_pattern(self, node: Any) -> int:
        return max(node.inner_pattern.accept(self), node.guard_expr.accept(self))

    def visit_binding(self, node: Any) -> int:
        return node.expr.accept(self)

    def visit_program(self, node: Any) -> int:
        n = 0
        for stmt in node.statements:
            n = max(n, stmt.accept(self))
        return n


def _count_reduction_dims_in_expr(expr: Any) -> int:
    """Return max number of reduction dimensions in any LoweredReductionIR in expr (0 if none)."""
    if expr is None:
        return 0
    return expr.accept(_ReductionDimsCounter())


def _clause_body_summary_default(node: Any, max_len: int) -> str:
    kind = type(node).__name__.replace("IR", "").lower()
    return kind[:max_len]


class _ClauseBodySummaryVisitor(IRVisitor[str]):
    """Short string for RHS of a clause (for profile output)."""

    def __init__(self, max_len: int = 60) -> None:
        self._max_len = max_len

    def visit_lowered_reduction(self, node: Any) -> str:
        op = getattr(node, "operation", "?")
        inner = node.body.accept(_ClauseBodySummaryVisitor(20)) if node.body is not None else "?"
        s = f"{op}({inner})" if inner != "?" else op
        return s[:self._max_len]

    def visit_binary_op(self, node: Any) -> str:
        op = getattr(node, "operator", None)
        op_str = str(getattr(op, "name", op)) if op is not None else "?"
        left = node.left.accept(_ClauseBodySummaryVisitor(15)) if node.left is not None else "?"
        right = node.right.accept(_ClauseBodySummaryVisitor(15)) if node.right is not None else "?"
        return f"{left} {op_str} {right}"[:self._max_len]

    def visit_function_call(self, node: Any) -> str:
        name = getattr(node, "function_name", None) or getattr(node, "callee_expr", None)
        if name is not None and not isinstance(name, str):
            name = getattr(name, "name", str(name))
        fn = name if isinstance(name, str) else "call"
        return f"{fn}(...)"[:self._max_len]

    def visit_rectangular_access(self, node: Any) -> str:
        arr = getattr(node, "array", None)
        arr_name = getattr(arr, "name", None) if arr is not None and hasattr(arr, "name") else None
        return (arr_name or "[]")[:self._max_len]

    def visit_identifier(self, node: Any) -> str:
        return (getattr(node, "name", None) or "?")[:self._max_len]

    def visit_unary_op(self, node: Any) -> str:
        op = getattr(node, "operator", None)
        op_str = str(getattr(op, "name", op)) if op is not None else "?"
        inner = node.operand.accept(_ClauseBodySummaryVisitor(20)) if node.operand is not None else "?"
        return f"{op_str}({inner})"[:self._max_len]

    def _default(self, node: Any) -> str:
        return _clause_body_summary_default(node, self._max_len)

    def visit_literal(self, node: Any) -> str:
        return self._default(node)

    def visit_index_var(self, node: Any) -> str:
        return self._default(node)

    def visit_index_rest(self, node: Any) -> str:
        return self._default(node)

    def visit_jagged_access(self, node: Any) -> str:
        return self._default(node)

    def visit_block_expression(self, node: Any) -> str:
        return self._default(node)

    def visit_if_expression(self, node: Any) -> str:
        return self._default(node)

    def visit_lambda(self, node: Any) -> str:
        return self._default(node)

    def visit_range(self, node: Any) -> str:
        return self._default(node)

    def visit_array_comprehension(self, node: Any) -> str:
        return self._default(node)

    def visit_module(self, node: Any) -> str:
        return self._default(node)

    def visit_array_literal(self, node: Any) -> str:
        return self._default(node)

    def visit_tuple_expression(self, node: Any) -> str:
        return self._default(node)

    def visit_tuple_access(self, node: Any) -> str:
        return self._default(node)

    def visit_interpolated_string(self, node: Any) -> str:
        return self._default(node)

    def visit_cast_expression(self, node: Any) -> str:
        return self._default(node)

    def visit_member_access(self, node: Any) -> str:
        return self._default(node)

    def visit_try_expression(self, node: Any) -> str:
        return self._default(node)

    def visit_match_expression(self, node: Any) -> str:
        return self._default(node)

    def visit_reduction_expression(self, node: Any) -> str:
        return self._default(node)

    def visit_where_expression(self, node: Any) -> str:
        return self._default(node)

    def visit_pipeline_expression(self, node: Any) -> str:
        return self._default(node)

    def visit_builtin_call(self, node: Any) -> str:
        return self._default(node)

    def visit_literal_pattern(self, node: Any) -> str:
        return self._default(node)

    def visit_identifier_pattern(self, node: Any) -> str:
        return self._default(node)

    def visit_wildcard_pattern(self, node: Any) -> str:
        return self._default(node)

    def visit_tuple_pattern(self, node: Any) -> str:
        return self._default(node)

    def visit_array_pattern(self, node: Any) -> str:
        return self._default(node)

    def visit_rest_pattern(self, node: Any) -> str:
        return self._default(node)

    def visit_guard_pattern(self, node: Any) -> str:
        return self._default(node)

    def visit_binding(self, node: Any) -> str:
        return self._default(node)

    def visit_program(self, node: Any) -> str:
        return self._default(node)

    def visit_lowered_comprehension(self, node: Any) -> str:
        return self._default(node)

    def visit_lowered_einstein_clause(self, node: Any) -> str:
        return self._default(node)

    def visit_lowered_einstein(self, node: Any) -> str:
        return self._default(node)

    def visit_lowered_recurrence(self, node: Any) -> str:
        return self._default(node)

    def visit_or_pattern(self, node: Any) -> str:
        return self._default(node)

    def visit_constructor_pattern(self, node: Any) -> str:
        return self._default(node)

    def visit_binding_pattern(self, node: Any) -> str:
        return self._default(node)

    def visit_range_pattern(self, node: Any) -> str:
        return self._default(node)

    def visit_function_value(self, node: Any) -> str:
        return self._default(node)

    def visit_einstein(self, node: Any) -> str:
        return self._default(node)

    def visit_einstein_clause(self, node: Any) -> str:
        return self._default(node)


def _clause_body_summary(body: Any, max_len: int = 60) -> str:
    """Short string for RHS of a clause (for profile output)."""
    if body is None:
        return "?"
    return body.accept(_ClauseBodySummaryVisitor(max_len))


class _ReductionUsesClauseVarVisitor(IRVisitor[bool]):
    """True if any LoweredReductionIR has a loop whose iterable references a clause loop var (dynamic bounds)."""

    def __init__(self, clause_loop_defids: List[Any]) -> None:
        self._clause_loop_defids = clause_loop_defids

    def _any(self, *nodes: Any) -> bool:
        for n in nodes:
            if n is not None and getattr(n, "accept", None) is not None and n.accept(self):
                return True
        return False

    def visit_lowered_reduction(self, node: Any) -> bool:
        for loop in node.loops:
            it = getattr(loop, "iterable", None)
            if it is not None and any(
                _BodyReferencesDefidVisitor(d).references(it) for d in self._clause_loop_defids
            ):
                return True
        return node.body.accept(self)

    def visit_binary_op(self, node: Any) -> bool:
        return node.left.accept(self) or node.right.accept(self)

    def visit_rectangular_access(self, node: Any) -> bool:
        if node.array.accept(self):
            return True
        for idx in node.indices:
            if idx.accept(self):
                return True
        return False

    def visit_function_call(self, node: Any) -> bool:
        if node.callee_expr.accept(self):
            return True
        for a in node.arguments:
            if a.accept(self):
                return True
        return False

    def visit_unary_op(self, node: Any) -> bool:
        return node.operand.accept(self)

    def visit_if_expression(self, node: Any) -> bool:
        return (
            node.condition.accept(self)
            or node.then_expr.accept(self)
            or (node.else_expr.accept(self) if node.else_expr is not None else False)
        )

    def visit_block_expression(self, node: Any) -> bool:
        for stmt in node.statements:
            if stmt.accept(self):
                return True
        return (
            node.final_expr.accept(self)
            if node.final_expr is not None
            else False
        )

    def visit_lambda(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_range(self, node: Any) -> bool:
        return node.start.accept(self) or node.end.accept(self)

    def visit_array_comprehension(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_where_expression(self, node: Any) -> bool:
        if node.expr.accept(self):
            return True
        for c in node.constraints:
            if c.accept(self):
                return True
        return False

    def visit_pipeline_expression(self, node: Any) -> bool:
        return node.left.accept(self) or node.right.accept(self)

    def visit_builtin_call(self, node: Any) -> bool:
        return any(a.accept(self) for a in node.args)

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_identifier(self, node: Any) -> bool:
        return False

    def visit_index_var(self, node: Any) -> bool:
        return (
            node.range_ir.accept(self)
            if node.range_ir is not None
            else False
        )

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_jagged_access(self, node: Any) -> bool:
        if node.base.accept(self):
            return True
        return any(idx.accept(self) for idx in node.index_chain)

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_tuple_expression(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_tuple_access(self, node: Any) -> bool:
        return node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> bool:
        return any(p.accept(self) for p in node.parts)

    def visit_cast_expression(self, node: Any) -> bool:
        return node.expr.accept(self)

    def visit_member_access(self, node: Any) -> bool:
        return node.object.accept(self)

    def visit_try_expression(self, node: Any) -> bool:
        return node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> bool:
        if node.scrutinee.accept(self):
            return True
        return any(arm.body.accept(self) for arm in node.arms)

    def visit_reduction_expression(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_array_pattern(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return node.inner_pattern.accept(self) or node.guard_expr.accept(self)

    def visit_binding(self, node: Any) -> bool:
        return node.expr.accept(self)

    def visit_program(self, node: Any) -> bool:
        return any(stmt.accept(self) for stmt in node.statements)

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_einstein(self, node: Any) -> bool:
        return any(item.accept(self) for item in node.items)

    def visit_lowered_recurrence(self, node: Any) -> bool:
        if getattr(node, "initial", None) is not None and node.initial.accept(self):
            return True
        rloop = getattr(node, "recurrence_loop", None)
        if rloop is not None and getattr(rloop, "iterable", None) is not None:
            if rloop.iterable.accept(self):
                return True
        return (
            getattr(node, "body", None) is not None
            and node.body.accept(self)
        )

    def visit_or_pattern(self, node: Any) -> bool:
        return any(alt.accept(self) for alt in (getattr(node, "alternatives", []) or []))

    def visit_constructor_pattern(self, node: Any) -> bool:
        return any(p.accept(self) for p in node.patterns)

    def visit_binding_pattern(self, node: Any) -> bool:
        return node.inner_pattern.accept(self)

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_function_value(self, node: Any) -> bool:
        return node.body.accept(self) if node.body is not None else False

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False


def _reduction_uses_clause_var_in_bounds(expr: Any, clause_loop_defids: List[Any]) -> bool:
    """True if any LoweredReductionIR in expr has a loop whose iterable references a clause loop var (dynamic bounds)."""
    if expr is None or not clause_loop_defids:
        return False
    return expr.accept(_ReductionUsesClauseVarVisitor(clause_loop_defids))


def _merge_defids_by_name(a: Dict[str, List[Any]], b: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Merge two name->defids dicts (a and b) into one. Modifies and returns a."""
    for k, v in b.items():
        a.setdefault(k, []).extend(v)
    return a


class _DefidsByNameCollector(IRVisitor[Dict[str, List[Any]]]):
    """Collect all (name -> list of defids) from IdentifierIR/IndexVarIR in the tree."""

    def _empty(self) -> Dict[str, List[Any]]:
        return {}

    def _one(self, name: str, defid: Any) -> Dict[str, List[Any]]:
        if name is not None and defid is not None:
            return {name: [defid]}
        return self._empty()

    def visit_literal(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_identifier(self, node: Any) -> Dict[str, List[Any]]:
        return self._one(node.name, node.defid)

    def visit_index_var(self, node: Any) -> Dict[str, List[Any]]:
        out = self._one(node.name, node.defid)
        if node.range_ir is not None:
            _merge_defids_by_name(out, node.range_ir.accept(self))
        return out

    def visit_index_rest(self, node: Any) -> Dict[str, List[Any]]:
        return self._one(node.name, node.defid)

    def visit_binary_op(self, node: Any) -> Dict[str, List[Any]]:
        out = node.left.accept(self)
        _merge_defids_by_name(out, node.right.accept(self))
        return out

    def visit_unary_op(self, node: Any) -> Dict[str, List[Any]]:
        return node.operand.accept(self)

    def visit_rectangular_access(self, node: Any) -> Dict[str, List[Any]]:
        out = node.array.accept(self)
        for idx in node.indices:
            _merge_defids_by_name(out, idx.accept(self))
        return out

    def visit_function_call(self, node: Any) -> Dict[str, List[Any]]:
        out = node.callee_expr.accept(self)
        for a in node.arguments:
            _merge_defids_by_name(out, a.accept(self))
        return out

    def visit_if_expression(self, node: Any) -> Dict[str, List[Any]]:
        out = node.condition.accept(self)
        _merge_defids_by_name(out, node.then_expr.accept(self))
        if node.else_expr is not None:
            _merge_defids_by_name(out, node.else_expr.accept(self))
        return out

    def visit_block_expression(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for stmt in node.statements:
            _merge_defids_by_name(out, stmt.accept(self))
        if node.final_expr is not None:
            _merge_defids_by_name(out, node.final_expr.accept(self))
        return out

    def visit_lambda(self, node: Any) -> Dict[str, List[Any]]:
        return node.body.accept(self)

    def visit_range(self, node: Any) -> Dict[str, List[Any]]:
        out = node.start.accept(self)
        _merge_defids_by_name(out, node.end.accept(self))
        return out

    def visit_lowered_reduction(self, node: Any) -> Dict[str, List[Any]]:
        out = node.body.accept(self)
        for lp in node.loops:
            if lp.variable is not None:
                _merge_defids_by_name(out, lp.variable.accept(self))
            if lp.iterable is not None:
                _merge_defids_by_name(out, lp.iterable.accept(self))
        return out

    def visit_jagged_access(self, node: Any) -> Dict[str, List[Any]]:
        out = node.base.accept(self)
        for idx in node.index_chain:
            _merge_defids_by_name(out, idx.accept(self))
        return out

    def visit_module(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_array_literal(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for e in node.elements:
            _merge_defids_by_name(out, e.accept(self))
        return out

    def visit_tuple_expression(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for e in node.elements:
            _merge_defids_by_name(out, e.accept(self))
        return out

    def visit_tuple_access(self, node: Any) -> Dict[str, List[Any]]:
        return node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for p in node.parts:
            _merge_defids_by_name(out, p.accept(self))
        return out

    def visit_cast_expression(self, node: Any) -> Dict[str, List[Any]]:
        return node.expr.accept(self)

    def visit_member_access(self, node: Any) -> Dict[str, List[Any]]:
        return node.object.accept(self)

    def visit_try_expression(self, node: Any) -> Dict[str, List[Any]]:
        return node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> Dict[str, List[Any]]:
        out = node.scrutinee.accept(self)
        for arm in node.arms:
            _merge_defids_by_name(out, arm.body.accept(self))
        return out

    def visit_reduction_expression(self, node: Any) -> Dict[str, List[Any]]:
        return node.body.accept(self)

    def visit_where_expression(self, node: Any) -> Dict[str, List[Any]]:
        out = node.expr.accept(self)
        for c in node.constraints:
            _merge_defids_by_name(out, c.accept(self))
        return out

    def visit_pipeline_expression(self, node: Any) -> Dict[str, List[Any]]:
        out = node.left.accept(self)
        _merge_defids_by_name(out, node.right.accept(self))
        return out

    def visit_builtin_call(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for a in node.args:
            _merge_defids_by_name(out, a.accept(self))
        return out

    def visit_array_comprehension(self, node: Any) -> Dict[str, List[Any]]:
        return node.body.accept(self)

    def visit_literal_pattern(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_identifier_pattern(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_wildcard_pattern(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_tuple_pattern(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for e in node.elements:
            _merge_defids_by_name(out, e.accept(self))
        return out

    def visit_array_pattern(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for e in node.elements:
            _merge_defids_by_name(out, e.accept(self))
        return out

    def visit_rest_pattern(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_guard_pattern(self, node: Any) -> Dict[str, List[Any]]:
        out = node.inner_pattern.accept(self)
        _merge_defids_by_name(out, node.guard_expr.accept(self))
        return out

    def visit_binding(self, node: Any) -> Dict[str, List[Any]]:
        return node.expr.accept(self)

    def visit_program(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for stmt in node.statements:
            _merge_defids_by_name(out, stmt.accept(self))
        return out

    def visit_lowered_comprehension(self, node: Any) -> Dict[str, List[Any]]:
        return node.body.accept(self)

    def visit_lowered_einstein_clause(self, node: Any) -> Dict[str, List[Any]]:
        return node.body.accept(self)

    def visit_lowered_einstein(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for item in node.items:
            _merge_defids_by_name(out, item.accept(self))
        return out

    def visit_lowered_recurrence(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        if getattr(node, "initial", None) is not None:
            _merge_defids_by_name(out, node.initial.accept(self))
        rloop = getattr(node, "recurrence_loop", None)
        if rloop is not None and getattr(rloop, "iterable", None) is not None:
            _merge_defids_by_name(out, rloop.iterable.accept(self))
        if getattr(node, "body", None) is not None:
            _merge_defids_by_name(out, node.body.accept(self))
        return out

    def visit_or_pattern(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for alt in getattr(node, "alternatives", []) or []:
            _merge_defids_by_name(out, alt.accept(self))
        return out

    def visit_constructor_pattern(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for p in node.patterns:
            _merge_defids_by_name(out, p.accept(self))
        return out

    def visit_binding_pattern(self, node: Any) -> Dict[str, List[Any]]:
        return node.inner_pattern.accept(self)

    def visit_range_pattern(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_function_value(self, node: Any) -> Dict[str, List[Any]]:
        return node.body.accept(self) if node.body is not None else self._empty()

    def visit_einstein(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_einstein_clause(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()


def _collect_defids_by_name(expr: Any) -> Dict[str, List[Any]]:
    """Collect all (name -> list of defids) from IdentifierIR/IndexVarIR in expr (e.g. for binding aliases)."""
    if expr is None:
        return {}
    return expr.accept(_DefidsByNameCollector())


class _BodyContainsCallUsingLoopVarVisitor(IRVisitor[bool]):
    """True if the tree contains a FunctionCallIR whose callee or any argument references a loop var."""

    def __init__(self, loop_defids: List[Any]) -> None:
        self._loop_defids = loop_defids

    def visit_function_call(self, node: Any) -> bool:
        for defid in self._loop_defids:
            if _BodyReferencesDefidVisitor(defid).references(node.callee_expr):
                return True
            for a in node.arguments:
                if _BodyReferencesDefidVisitor(defid).references(a):
                    return True
        return False

    def visit_rectangular_access(self, node: Any) -> bool:
        if node.array.accept(self):
            return True
        return any(idx.accept(self) for idx in node.indices)

    def visit_binary_op(self, node: Any) -> bool:
        return node.left.accept(self) or node.right.accept(self)

    def visit_block_expression(self, node: Any) -> bool:
        for stmt in node.statements:
            if stmt.accept(self):
                return True
        return (
            node.final_expr.accept(self)
            if node.final_expr is not None
            else False
        )

    def visit_binding(self, node: Any) -> bool:
        return node.expr.accept(self)

    def visit_guard_pattern(self, node: Any) -> bool:
        return node.inner_pattern.accept(self) or node.guard_expr.accept(self)

    def visit_or_pattern(self, node: Any) -> bool:
        return any(alt.accept(self) for alt in (getattr(node, "alternatives", []) or []))

    def visit_constructor_pattern(self, node: Any) -> bool:
        return any(p.accept(self) for p in node.patterns)

    def _false(self) -> bool:
        return False

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_identifier(self, node: Any) -> bool:
        return False

    def visit_index_var(self, node: Any) -> bool:
        return (
            node.range_ir.accept(self)
            if node.range_ir is not None
            else False
        )

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_unary_op(self, node: Any) -> bool:
        return node.operand.accept(self)

    def visit_if_expression(self, node: Any) -> bool:
        return (
            node.condition.accept(self)
            or node.then_expr.accept(self)
            or (node.else_expr.accept(self) if node.else_expr is not None else False)
        )

    def visit_lambda(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_range(self, node: Any) -> bool:
        return node.start.accept(self) or node.end.accept(self)

    def visit_array_comprehension(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_jagged_access(self, node: Any) -> bool:
        if node.base.accept(self):
            return True
        return any(idx.accept(self) for idx in node.index_chain)

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_tuple_expression(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_tuple_access(self, node: Any) -> bool:
        return node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> bool:
        return any(p.accept(self) for p in node.parts)

    def visit_cast_expression(self, node: Any) -> bool:
        return node.expr.accept(self)

    def visit_member_access(self, node: Any) -> bool:
        return node.object.accept(self)

    def visit_try_expression(self, node: Any) -> bool:
        return node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> bool:
        if node.scrutinee.accept(self):
            return True
        return any(arm.body.accept(self) for arm in node.arms)

    def visit_reduction_expression(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_where_expression(self, node: Any) -> bool:
        if node.expr.accept(self):
            return True
        return any(c.accept(self) for c in node.constraints)

    def visit_pipeline_expression(self, node: Any) -> bool:
        return node.left.accept(self) or node.right.accept(self)

    def visit_builtin_call(self, node: Any) -> bool:
        return any(a.accept(self) for a in node.args)

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_array_pattern(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_binding_pattern(self, node: Any) -> bool:
        return node.inner_pattern.accept(self)

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_program(self, node: Any) -> bool:
        return any(stmt.accept(self) for stmt in node.statements)

    def visit_lowered_reduction(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_einstein(self, node: Any) -> bool:
        return any(item.accept(self) for item in node.items)

    def visit_lowered_recurrence(self, node: Any) -> bool:
        if getattr(node, "initial", None) is not None and node.initial.accept(self):
            return True
        rloop = getattr(node, "recurrence_loop", None)
        if rloop is not None and getattr(rloop, "iterable", None) is not None:
            if rloop.iterable.accept(self):
                return True
        return (
            getattr(node, "body", None) is not None
            and node.body.accept(self)
        )

    def visit_function_value(self, node: Any) -> bool:
        return node.body.accept(self) if node.body is not None else False

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False


def _body_contains_call_using_loop_var(expr: Any, loop_defids: List[Any]) -> bool:
    """True if body contains a FunctionCallIR whose arguments (or callee) reference a loop var."""
    if not expr or not loop_defids:
        return False
    return expr.accept(_BodyContainsCallUsingLoopVarVisitor(loop_defids))


class _DefidsInCallArgsCollector(IRVisitor[None]):
    """Collect into _out any loop defid referenced inside a call's arguments or callee. Tracks _inside_call for RectangularAccessIR indices."""

    def __init__(self, loop_defids: List[Any], out: set, inside_call: bool = False) -> None:
        self._loop_set = {d for d in loop_defids if d is not None}
        self._out = out
        self._inside_call = inside_call

    def visit_identifier(self, node: Any) -> None:
        if self._inside_call and node.defid is not None and node.defid in self._loop_set:
            self._out.add(node.defid)

    def visit_index_var(self, node: Any) -> None:
        if self._inside_call and node.defid is not None and node.defid in self._loop_set:
            self._out.add(node.defid)
        if node.range_ir is not None:
            node.range_ir.accept(self)

    def visit_index_rest(self, node: Any) -> None:
        if self._inside_call and node.defid is not None and node.defid in self._loop_set:
            self._out.add(node.defid)

    def visit_rectangular_access(self, node: Any) -> None:
        node.array.accept(self)
        if self._inside_call:
            for idx in node.indices:
                idx.accept(self)

    def visit_binary_op(self, node: Any) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_function_call(self, node: Any) -> None:
        prev = self._inside_call
        self._inside_call = True
        try:
            for a in node.arguments:
                a.accept(self)
            node.callee_expr.accept(self)
        finally:
            self._inside_call = prev

    def visit_block_expression(self, node: Any) -> None:
        for stmt in node.statements:
            stmt.accept(self)
        if node.final_expr is not None:
            node.final_expr.accept(self)

    def visit_binding(self, node: Any) -> None:
        node.expr.accept(self)

    def visit_guard_pattern(self, node: Any) -> None:
        node.inner_pattern.accept(self)
        node.guard_expr.accept(self)

    def visit_or_pattern(self, node: Any) -> None:
        for alt in getattr(node, "alternatives", []) or []:
            alt.accept(self)

    def visit_constructor_pattern(self, node: Any) -> None:
        for p in node.patterns:
            p.accept(self)

    def _recurse_none(self) -> None:
        return None

    def visit_literal(self, node: Any) -> None:
        return None

    def visit_unary_op(self, node: Any) -> None:
        node.operand.accept(self)

    def visit_if_expression(self, node: Any) -> None:
        node.condition.accept(self)
        node.then_expr.accept(self)
        if node.else_expr is not None:
            node.else_expr.accept(self)

    def visit_lambda(self, node: Any) -> None:
        node.body.accept(self)

    def visit_range(self, node: Any) -> None:
        node.start.accept(self)
        node.end.accept(self)

    def visit_array_comprehension(self, node: Any) -> None:
        node.body.accept(self)

    def visit_jagged_access(self, node: Any) -> None:
        node.base.accept(self)
        for idx in node.index_chain:
            idx.accept(self)

    def visit_module(self, node: Any) -> None:
        return None

    def visit_array_literal(self, node: Any) -> None:
        for e in node.elements:
            e.accept(self)

    def visit_tuple_expression(self, node: Any) -> None:
        for e in node.elements:
            e.accept(self)

    def visit_tuple_access(self, node: Any) -> None:
        node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> None:
        for p in node.parts:
            p.accept(self)

    def visit_cast_expression(self, node: Any) -> None:
        node.expr.accept(self)

    def visit_member_access(self, node: Any) -> None:
        node.object.accept(self)

    def visit_try_expression(self, node: Any) -> None:
        node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> None:
        node.scrutinee.accept(self)
        for arm in node.arms:
            arm.body.accept(self)

    def visit_reduction_expression(self, node: Any) -> None:
        node.body.accept(self)

    def visit_where_expression(self, node: Any) -> None:
        node.expr.accept(self)
        for c in node.constraints:
            c.accept(self)

    def visit_pipeline_expression(self, node: Any) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_builtin_call(self, node: Any) -> None:
        for a in node.args:
            a.accept(self)

    def visit_literal_pattern(self, node: Any) -> None:
        return None

    def visit_identifier_pattern(self, node: Any) -> None:
        return None

    def visit_wildcard_pattern(self, node: Any) -> None:
        return None

    def visit_tuple_pattern(self, node: Any) -> None:
        for e in node.elements:
            e.accept(self)

    def visit_array_pattern(self, node: Any) -> None:
        for e in node.elements:
            e.accept(self)

    def visit_rest_pattern(self, node: Any) -> None:
        return None

    def visit_binding_pattern(self, node: Any) -> None:
        node.inner_pattern.accept(self)

    def visit_range_pattern(self, node: Any) -> None:
        return None

    def visit_program(self, node: Any) -> None:
        for stmt in node.statements:
            stmt.accept(self)

    def visit_lowered_reduction(self, node: Any) -> None:
        node.body.accept(self)

    def visit_lowered_comprehension(self, node: Any) -> None:
        node.body.accept(self)

    def visit_lowered_einstein_clause(self, node: Any) -> None:
        node.body.accept(self)

    def visit_lowered_einstein(self, node: Any) -> None:
        for item in node.items:
            item.accept(self)

    def visit_lowered_recurrence(self, node: Any) -> None:
        if getattr(node, "initial", None) is not None:
            node.initial.accept(self)
        rloop = getattr(node, "recurrence_loop", None)
        if rloop is not None and getattr(rloop, "iterable", None) is not None:
            rloop.iterable.accept(self)
        if getattr(node, "body", None) is not None:
            node.body.accept(self)

    def visit_function_value(self, node: Any) -> None:
        if node.body is not None:
            node.body.accept(self)

    def visit_einstein(self, node: Any) -> None:
        return None

    def visit_einstein_clause(self, node: Any) -> None:
        return None


def _collect_defids_in_call_args(expr: Any, loop_defids: List[Any], out: set, inside_call: bool = False) -> None:
    """Add to out any loop defid referenced inside a call's arguments or callee (not in outer indexing like result[j])."""
    if expr is None or not loop_defids:
        return
    expr.accept(_DefidsInCallArgsCollector(loop_defids, out, inside_call))


def _loop_defids_in_call_args(body: Any, loop_defids: List[Any]) -> set:
    """Loop defids that appear in any call's arguments (or callee). Those dims must be scalar for call-scalar hybrid."""
    loop_set = {d for d in loop_defids if d is not None}
    out: set = set()
    _collect_defids_in_call_args(body, list(loop_set), out)
    return out


def _body_is_elementwise_call(body: Any, loop_defids: List[Any]) -> bool:
    """True if the body is (or ends in) a single function call and every loop var appears in that call's arguments.
    Such clauses are element-wise and must use the vectorized path (one call with full array), not the scalar loop."""
    if body is None or not loop_defids:
        return False
    loop_set = {d for d in loop_defids if d is not None}
    if not loop_set:
        return False
    in_call = _loop_defids_in_call_args(body, loop_defids)
    if in_call != loop_set:
        return False
    if isinstance(body, FunctionCallIR):
        return True
    if isinstance(body, BlockExpressionIR):
        return _body_is_elementwise_call(getattr(body, "final_expr", None), loop_defids)
    return False


class _IndexExprIsLoopVarVisitor(IRVisitor[bool]):
    """True iff expr is IdentifierIR or IndexVarIR with defid == loop_defid."""

    def __init__(self, loop_defid: Any) -> None:
        self._loop_defid = loop_defid

    def visit_identifier(self, node: Any) -> bool:
        return getattr(node, "defid", None) == self._loop_defid

    def visit_index_var(self, node: Any) -> bool:
        return getattr(node, "defid", None) == self._loop_defid

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_binary_op(self, node: Any) -> bool:
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


class _IndexExprIsBackwardVisitor(IRVisitor[bool]):
    """True if expr is loop_var or (loop_var - positive_const)."""

    def __init__(self, loop_defid: Any) -> None:
        self._loop_defid = loop_defid

    def visit_identifier(self, node: Any) -> bool:
        return getattr(node, "defid", None) == self._loop_defid

    def visit_index_var(self, node: Any) -> bool:
        return getattr(node, "defid", None) == self._loop_defid

    def visit_binary_op(self, node: Any) -> bool:
        if node.operator != BinaryOp.SUB and getattr(node.operator, "value", None) != "-":
            return False
        if not _index_expr_is_loop_var(node.left, self._loop_defid):
            return False
        if isinstance(node.right, LiteralIR):
            try:
                return int(getattr(node.right, "value", 0)) > 0
            except (TypeError, ValueError):
                pass
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


def _index_expr_is_backward(expr: Any, loop_defid: Any) -> bool:
    """True if expr is loop_var or (loop_var - positive_const). Recurrence dim: no loop var on RHS, only t-1 style."""
    if expr is None or loop_defid is None:
        return False
    return expr.accept(_IndexExprIsBackwardVisitor(loop_defid))


class _IndexExprIsStrictlyBackwardVisitor(IRVisitor[bool]):
    """True only for (loop_var - positive_const). False for loop_var (same index)."""

    def __init__(self, loop_defid: Any) -> None:
        self._loop_defid = loop_defid

    def visit_identifier(self, node: Any) -> bool:
        return False

    def visit_index_var(self, node: Any) -> bool:
        return False

    def visit_binary_op(self, node: Any) -> bool:
        if node.operator != BinaryOp.SUB and getattr(node.operator, "value", None) != "-":
            return False
        if not _index_expr_is_loop_var(node.left, self._loop_defid):
            return False
        if isinstance(node.right, LiteralIR):
            try:
                return int(getattr(node.right, "value", 0)) > 0
            except (TypeError, ValueError):
                pass
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


def _index_expr_is_strictly_backward(expr: Any, loop_defid: Any) -> bool:
    """True only for (loop_var - positive_const). False for loop_var (same index). Used so hybrid recurrence is only t-1 style."""
    if expr is None or loop_defid is None:
        return False
    return expr.accept(_IndexExprIsStrictlyBackwardVisitor(loop_defid))


class _IndexExprIsLoopVarOrOffsetVisitor(IRVisitor[bool]):
    """True if expr is loop_var or (loop_var ± const)."""

    def __init__(self, loop_defid: Any) -> None:
        self._loop_defid = loop_defid

    def visit_identifier(self, node: Any) -> bool:
        return getattr(node, "defid", None) == self._loop_defid

    def visit_index_var(self, node: Any) -> bool:
        return getattr(node, "defid", None) == self._loop_defid

    def visit_binary_op(self, node: Any) -> bool:
        op = getattr(node, "operator", None)
        op_val = getattr(op, "value", None) if op is not None else None
        if op not in (BinaryOp.ADD, BinaryOp.SUB) and op_val not in ("+", "-"):
            return False
        if not _index_expr_is_loop_var(node.left, self._loop_defid):
            return False
        if isinstance(node.right, LiteralIR):
            try:
                int(getattr(node.right, "value", 0))
                return True
            except (TypeError, ValueError):
                pass
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
    """True if expr is loop_var or (loop_var ± const). Such dims are not recurrence; we can vectorize (e.g. i, i-1, i+1)."""
    if expr is None or loop_defid is None:
        return False
    return expr.accept(_IndexExprIsLoopVarOrOffsetVisitor(loop_defid))


class _ExprIsLoopVarOrMinusOneVisitor(IRVisitor[bool]):
    """True if expr is loop_var or (loop_var - 1)."""

    def __init__(self, loop_defid: Any) -> None:
        self._loop_defid = loop_defid

    def visit_identifier(self, node: Any) -> bool:
        return getattr(node, "defid", None) == self._loop_defid

    def visit_index_var(self, node: Any) -> bool:
        return getattr(node, "defid", None) == self._loop_defid

    def visit_binary_op(self, node: Any) -> bool:
        if node.operator != BinaryOp.SUB and getattr(node.operator, "value", None) != "-":
            return False
        if not _index_expr_is_loop_var(node.left, self._loop_defid):
            return False
        if isinstance(node.right, LiteralIR):
            try:
                return int(getattr(node.right, "value", 0)) == 1
            except (TypeError, ValueError):
                pass
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
    """True if expr is loop_var or (loop_var - 1). Used to detect reduction upper bound like 0..(j-1)."""
    if expr is None or loop_defid is None:
        return False
    return expr.accept(_ExprIsLoopVarOrMinusOneVisitor(loop_defid))


def _recurrence_dims_for_hybrid(
    lowered: Any, variable_defid: Any, clause_indices: Optional[List[Any]] = None
) -> List[int]:
    """Loop indices k where every LHS read is strictly backward (e.g. t-1). Same timestep (e.g. t) is not recurrence.
    Not used for partition/step: we need to accept both backward-in-time and same-timestep; use _recurrence_dims."""
    loops = getattr(lowered, "loops", None) or []
    if not loops or variable_defid is None:
        return []
    loop_defids = [getattr(lp.variable, "defid", None) for lp in loops]
    read_index_lists = _collect_lhs_read_index_lists(lowered.body, variable_defid)
    if not read_index_lists:
        return []
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops) if clause_indices else None
    result: List[int] = []
    for k in range(len(loops)):
        out_d = loop_dims[k] if loop_dims is not None and k < len(loop_dims) else k
        if all(
            out_d < len(idx_list) and _index_expr_is_strictly_backward(idx_list[out_d], loop_defids[k])
            for idx_list in read_index_lists
        ):
            result.append(k)
    return result


def _recurrence_dims_for_hybrid_or_full(
    lowered: Any, variable_defid: Any, clause_indices: Optional[List[Any]] = None
) -> List[int]:
    """Strict backward only (every read t-1). Returns [] if any read is same timestep (t).
    For partition/step we accept backward-in-time and same-timestep; use _recurrence_dims. This is for hybrid vectorized path only."""
    loops = getattr(lowered, "loops", None) or []
    if not loops or variable_defid is None:
        return []
    recurrence_for_hybrid = _recurrence_dims_for_hybrid(lowered, variable_defid, clause_indices)
    if not recurrence_for_hybrid:
        return []
    loop_defids = [getattr(lp.variable, "defid", None) for lp in loops]
    read_index_lists = _collect_lhs_read_index_lists(lowered.body, variable_defid)
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops) if clause_indices else None
    for k in range(len(loops)):
        if k in recurrence_for_hybrid:
            continue
        out_d = loop_dims[k] if loop_dims is not None and k < len(loop_dims) else k
        for idx_list in read_index_lists:
            if out_d >= len(idx_list):
                continue
            if not _index_expr_is_loop_var_or_offset(idx_list[out_d], loop_defids[k]):
                return _recurrence_dims(lowered, variable_defid, clause_indices)
    return recurrence_for_hybrid


class _LHSReadIndexListsCollector(IRVisitor[List[List[Any]]]):
    """Collect index lists from RectangularAccessIR nodes whose array references target_defid (LHS read positions)."""

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
        for a in node.arguments:
            out.extend(a.accept(self))
        out.extend(node.callee_expr.accept(self))
        return out

    def visit_block_expression(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for stmt in node.statements:
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
        return (
            node.range_ir.accept(self)
            if node.range_ir is not None
            else self._empty()
        )

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
        for idx in node.index_chain:
            out.extend(idx.accept(self))
        return out

    def visit_module(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_array_literal(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for e in node.elements:
            out.extend(e.accept(self))
        return out

    def visit_tuple_expression(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for e in node.elements:
            out.extend(e.accept(self))
        return out

    def visit_tuple_access(self, node: Any) -> List[List[Any]]:
        return node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for p in node.parts:
            out.extend(p.accept(self))
        return out

    def visit_cast_expression(self, node: Any) -> List[List[Any]]:
        return node.expr.accept(self)

    def visit_member_access(self, node: Any) -> List[List[Any]]:
        return node.object.accept(self)

    def visit_try_expression(self, node: Any) -> List[List[Any]]:
        return node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> List[List[Any]]:
        out = node.scrutinee.accept(self)
        for arm in node.arms:
            out.extend(arm.body.accept(self))
        return out

    def visit_reduction_expression(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_where_expression(self, node: Any) -> List[List[Any]]:
        out = node.expr.accept(self)
        for c in node.constraints:
            out.extend(c.accept(self))
        return out

    def visit_pipeline_expression(self, node: Any) -> List[List[Any]]:
        out = node.left.accept(self)
        out.extend(node.right.accept(self))
        return out

    def visit_builtin_call(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for a in node.args:
            out.extend(a.accept(self))
        return out

    def visit_literal_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_identifier_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_wildcard_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_tuple_pattern(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for e in node.elements:
            out.extend(e.accept(self))
        return out

    def visit_array_pattern(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for e in node.elements:
            out.extend(e.accept(self))
        return out

    def visit_rest_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_guard_pattern(self, node: Any) -> List[List[Any]]:
        out = node.inner_pattern.accept(self)
        out.extend(node.guard_expr.accept(self))
        return out

    def visit_program(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for stmt in node.statements:
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
        for item in node.items:
            out.extend(item.accept(self))
        return out

    def visit_lowered_recurrence(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        if getattr(node, "initial", None) is not None:
            out.extend(node.initial.accept(self))
        rloop = getattr(node, "recurrence_loop", None)
        if rloop is not None and getattr(rloop, "iterable", None) is not None:
            out.extend(rloop.iterable.accept(self))
        if getattr(node, "body", None) is not None:
            out.extend(node.body.accept(self))
        return out

    def visit_or_pattern(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for alt in getattr(node, "alternatives", []) or []:
            out.extend(alt.accept(self))
        return out

    def visit_constructor_pattern(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for p in node.patterns:
            out.extend(p.accept(self))
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
    """Collect index lists from array accesses whose array references target_defid (LHS read positions)."""
    if body is None:
        return []
    return body.accept(_LHSReadIndexListsCollector(target_defid))


def _recurrence_dims(lowered: Any, variable_defid: Any, clause_indices: Optional[List[Any]] = None) -> List[int]:
    """Return loop indices k where any LHS read differs from the write (recurrence).
    Accepts both backward-in-time (e.g. t-1) and same-timestep (e.g. state[t,0] when writing state[t,1]);
    we add k if any read is not the loop var, so clauses with mixed t-1 and t reads still get k and run in timestep-major.
    When clause_indices is set, index read lists by output dim loop_dims[k], not k."""
    loops = getattr(lowered, "loops", None) or []
    if not loops or variable_defid is None:
        return []
    loop_defids = [getattr(lp.variable, "defid", None) for lp in loops]
    read_index_lists = _collect_lhs_read_index_lists(lowered.body, variable_defid)
    if not read_index_lists:
        return []
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops) if clause_indices else None
    recurrence: List[int] = []
    for k in range(len(loops)):
        out_d = loop_dims[k] if loop_dims is not None and k < len(loop_dims) else k
        for idx_list in read_index_lists:
            if out_d >= len(idx_list):
                continue
            if not _index_expr_is_loop_var(idx_list[out_d], loop_defids[k]):
                recurrence.append(k)
                break
    return recurrence


def _reduction_var_bounded_by_loop_var(
    read_index_expr: Any,
    loop_defid: Any,
    reduction_ranges: Any,
) -> bool:
    """True if read_index_expr is a reduction variable whose range end is loop_var or loop_var-1."""
    if reduction_ranges is None or not read_index_expr or loop_defid is None:
        return False
    read_defid = getattr(read_index_expr, "defid", None)
    if read_defid is None:
        return False
    loop_struct = reduction_ranges.get(read_defid) if isinstance(reduction_ranges, dict) else None
    if loop_struct is None:
        return False
    iterable = getattr(loop_struct, "iterable", None)
    if not isinstance(iterable, RangeIR):
        return False
    end = getattr(iterable, "end", None)
    return _expr_is_loop_var_or_minus_one(end, loop_defid)


def _loop_dims_from_clause_indices(clause_indices: List[Any], loops: List[Any]) -> Optional[List[int]]:
    """For each loop index k, return the output-dimension index (position in clause_indices). Literals have no loop."""
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


def _slice_list_from_clause_indices(
    clause_indices: List[Any],
    lowered: Any,
    expr_evaluator: Any,
) -> Optional[List[Any]]:
    """Build full-dimension slice list: literal idx -> scalar (int), other indices -> slice from loop range.
    Rule: literal idx / constant value -> scalar, other indices -> vectorize (slice)."""
    if not clause_indices:
        return None
    out: List[Any] = []
    loop_pos = 0
    loops = getattr(lowered, "loops", None) or []
    for idx in clause_indices:
        literal_val = None
        if isinstance(idx, LiteralIR):
            try:
                literal_val = int(idx.value)
            except (TypeError, ValueError):
                pass
        elif getattr(idx, "value", None) is not None:
            try:
                literal_val = int(idx.value)
            except (TypeError, ValueError):
                pass
        if literal_val is not None:
            out.append(literal_val)
            continue
        if loop_pos < len(loops):
            try:
                start, end = _extract_loop_range(loops[loop_pos], expr_evaluator)
                out.append(slice(int(start), int(end)))
            except (RuntimeError, TypeError, ValueError):
                return None
            loop_pos += 1
        else:
            return None
    return out if loop_pos == len(loops) else None


def _extract_loop_range(loop, evaluator) -> Tuple[int, int]:
    """Return (start, end) for the loop range; both must be concrete int. Raises if missing or dependent."""
    it = getattr(loop, "iterable", None)
    if it is None:
        raise RuntimeError("loop has no iterable; cannot extract range")
    if isinstance(it, LiteralIR) and isinstance(getattr(it, "value", None), range):
        r = it.value
        start = getattr(r, "start", 0)
        stop = getattr(r, "stop", r.start)
        try:
            return (int(start), int(stop))
        except (TypeError, ValueError) as e:
            raise RuntimeError("loop range start/stop must be int; got dependent or non-int") from e
    if isinstance(it, RangeIR):
        end_ev = evaluator(it.end)
        if not isinstance(end_ev, (int, np.integer)):
            raise RuntimeError("loop range end must be int; got dependent or non-int")
        start_node = getattr(it, "start", None)
        if start_node is not None:
            start_ev = evaluator(start_node)
            if not isinstance(start_ev, (int, np.integer)):
                raise RuntimeError("loop range start must be int; got dependent or non-int")
            return (int(start_ev), int(end_ev))
        return (0, int(end_ev))
    raise RuntimeError("loop iterable is not a range or literal range; cannot extract (start, end)")


def _eval_clause_body_with_broadcast_loops(
    clause: Any,
    output_shape: List[int],
    evaluator: Any,
    backend: Any,
    loop_ranges_override: Optional[List[Tuple[int, int]]] = None,
) -> Optional[Any]:
    """Evaluate clause body once with loop vars set to broadcast arrays. Returns result or None on failure.
    If loop_ranges_override is set, use (start,end) per dimension instead of extracting from clause (for chunked execution)."""
    loops = getattr(clause, "loops", None) or []
    if not loops or getattr(clause, "guards", None) or getattr(clause, "bindings", None):
        return None
    clause_ndim = len(loops)
    n_red = _count_reduction_dims_in_expr(clause.body)
    ndim = clause_ndim + n_red
    loop_info: List[Tuple[Any, Tuple[int, int], str]] = []
    for dim, lp in enumerate(loops):
        defid = getattr(lp.variable, "defid", None)
        if defid is None:
            return None
        if loop_ranges_override is not None and dim < len(loop_ranges_override):
            r = loop_ranges_override[dim]
        else:
            try:
                r = _extract_loop_range(lp, evaluator)
            except RuntimeError:
                return None
        name = getattr(lp.variable, "name", None)
        loop_info.append((defid, r, name))
    clause_loop_defids = [defid for (defid, _, _) in loop_info]
    if _reduction_uses_clause_var_in_bounds(clause.body, clause_loop_defids):
        return None
    try:
        with backend.env.scope():
            for dim, (defid, rng, name) in enumerate(loop_info):
                start, end = rng
                sz = end - start
                shape = [1] * ndim
                shape[dim] = sz
                arr = np.arange(start, end, dtype=np.intp).reshape(shape)
                backend.env.set_value(defid, arr, name=name)
            parallel_shape_tuple = tuple(output_shape)
            try:
                setattr(backend, "_vectorize_parallel_shape", parallel_shape_tuple)
                return clause.body.accept(backend)
            finally:
                setattr(backend, "_vectorize_parallel_shape", None)
    except Exception:
        return None


def _try_slice_vectorize_if_clause(
    lowered: Any,
    output: np.ndarray,
    expr_evaluator: Any,
    backend: Any,
) -> Optional[np.ndarray]:
    """
    When body is (if loop_var < bound then then_expr else else_expr), vectorize over
    [0..bound) and fill the rest with else_expr. Speeds up patterns like
    emb[p,d] = if p < t then dec_tok_emb[tokens[p], d] + ... else 0.
    """
    body = getattr(lowered, "body", None)
    if not isinstance(body, IfExpressionIR):
        return None
    cond = getattr(body, "condition", None)
    if not isinstance(cond, BinaryOpIR):
        return None
    op = getattr(cond, "operator", None)
    lt_ops = (BinaryOp.LT, "<")
    if op not in lt_ops:
        return None
    left = getattr(cond, "left", None)
    right = getattr(cond, "right", None)
    if left is None or right is None:
        return None
    loops = getattr(lowered, "loops", None) or []
    if not loops:
        return None
    loop_defids = [getattr(lp.variable, "defid", None) for lp in loops]
    left_defid = getattr(left, "defid", None) if isinstance(left, (IdentifierIR, IndexVarIR)) else None
    right_defid = getattr(right, "defid", None) if isinstance(right, (IdentifierIR, IndexVarIR)) else None
    bound_side = None
    dim = -1
    if left_defid is not None and left_defid in loop_defids:
        bound_side = right
        dim = loop_defids.index(left_defid)
    elif right_defid is not None and right_defid in loop_defids:
        bound_side = left
        dim = loop_defids.index(right_defid)
    if bound_side is None or dim < 0:
        return None
    try:
        if isinstance(bound_side, LiteralIR):
            bound = int(getattr(bound_side, "value", 0))
        elif getattr(bound_side, "defid", None) is not None:
            bound = backend.env.get_value(bound_side.defid)
            bound = int(bound) if bound is not None else None
        else:
            bound = bound_side.accept(backend)
            bound = int(bound) if bound is not None else None
        if bound is None or bound < 0:
            return None
    except (TypeError, ValueError, AttributeError):
        return None
    full_shape = list(output.shape)
    if dim >= len(full_shape) or bound > full_shape[dim]:
        return None
    override: List[Tuple[int, int]] = []
    for i, lp in enumerate(loops):
        if i == dim:
            override.append((0, bound))
        else:
            try:
                r = _extract_loop_range(lp, expr_evaluator)
                override.append(r)
            except RuntimeError:
                return None
    slice_shape = [override[i][1] - override[i][0] for i in range(len(override))]
    result = _try_vectorize_clause(
        lowered, slice_shape, output.dtype, expr_evaluator, backend,
        loop_ranges_override=override,
    )
    if result is None:
        return None
    result = np.asarray(result)
    if result.shape != tuple(slice_shape):
        return None
    else_expr = getattr(body, "else_expr", None)
    else_val = 0
    if isinstance(else_expr, LiteralIR):
        v = getattr(else_expr, "value", 0)
        try:
            else_val = float(v) if isinstance(v, (int, float)) else 0
        except (TypeError, ValueError):
            else_val = 0
    elif else_expr is not None:
        try:
            else_val = else_expr.accept(backend)
            if isinstance(else_val, np.ndarray) and else_val.ndim == 0:
                else_val = float(else_val)
            elif not isinstance(else_val, (int, float)):
                else_val = 0
        except Exception:
            else_val = 0
    output.fill(else_val)
    sl: List[Any] = [slice(None)] * output.ndim
    sl[dim] = slice(0, bound)
    output[tuple(sl)] = result.astype(output.dtype, copy=False)
    return output


def _try_vectorize_clause(
    clause,
    output_shape,
    dtype,
    evaluator,
    backend=None,
    loop_ranges_override: Optional[List[Tuple[int, int]]] = None,
):
    """
    General vectorization: set loop variables to broadcast numpy arrays,
    evaluate the body once, and let numpy handle everything.
    Falls back to None if any operation doesn't support array-valued indices.
    If loop_ranges_override is set, use those (start,end) per dimension for chunked execution.
    """
    if backend is None:
        return None
    loops = clause.loops or []
    if not loops:
        return None
    if clause.guards:
        return None
    if clause.bindings:
        return None

    clause_ndim = len(loops)
    n_red = _count_reduction_dims_in_expr(clause.body)
    ndim = clause_ndim + n_red

    loop_info: List[Tuple[Any, Tuple[int, int], str]] = []
    for dim, lp in enumerate(loops):
        defid = getattr(lp.variable, "defid", None)
        if defid is None:
            return None
        if loop_ranges_override is not None and dim < len(loop_ranges_override):
            r = loop_ranges_override[dim]
        else:
            r = _extract_loop_range(lp, evaluator)
        name = getattr(lp.variable, "name", None)
        loop_info.append((defid, r, name))

    clause_loop_defids = [defid for (defid, _, _) in loop_info]
    if _reduction_uses_clause_var_in_bounds(clause.body, clause_loop_defids):
        return None

    try:
        result = _eval_clause_body_with_broadcast_loops(
            clause, output_shape, evaluator, backend, loop_ranges_override=loop_ranges_override
        )
        if result is None:
            return None
        if isinstance(result, np.ndarray):
            expected = tuple(output_shape)
            ranges = [(start, end) for (_, (start, end), _) in loop_info]
            range_is_full = len(ranges) == len(expected) and all(
                start == 0 and end == expected[dim] for dim, (start, end) in enumerate(ranges)
            )
            if result.shape == expected:
                return result.astype(dtype, copy=False)
            if not range_is_full:
                full = np.zeros(expected, dtype=dtype)
                slices = tuple(slice(int(start), int(end)) for (start, end) in ranges)
                full[slices] = result.astype(dtype, copy=False)
                return full
            if result.size == np.prod(expected):
                return result.reshape(expected).astype(dtype, copy=False)
            try:
                return np.broadcast_to(result, expected).copy().astype(dtype, copy=False)
            except ValueError:
                return None
        if isinstance(result, (int, float, np.integer, np.floating)):
            return np.full(output_shape, result, dtype=dtype)
    except Exception:
        return None
    return None


def _try_hybrid_vectorize_clause(
    clause: Any,
    output_shape: List[int],
    output: np.ndarray,
    variable_defid: Any,
    expr_evaluator: Any,
    backend: Any,
    clause_indices: Optional[List[Any]] = None,
) -> Optional[np.ndarray]:
    """
    When the body has recurrence on a subset of dimensions: iterate over those
    (scalar), vectorize over the rest. Writes into output and returns it, or None on failure.
    When clause_indices is set (and has literals), build slice in output space and use
    recurrence_dims from _recurrence_dims_for_hybrid_or_full(..., clause_indices).
    """
    from ..runtime.compute.lowered_execution import execute_lowered_loops
    loops = getattr(clause, "loops", None) or []
    if not loops or getattr(clause, "guards", None):
        return None
    if getattr(clause, "bindings", None):
        return None
    ndim = len(loops)
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops) if clause_indices else None
    recurrence_dims = (
        _recurrence_dims_for_hybrid_or_full(clause, variable_defid, clause_indices)
        if variable_defid else []
    )
    if not (0 < len(recurrence_dims) < ndim):
        return None
    loop_info: List[Tuple[Any, Optional[Tuple[int, int]], str]] = []
    for dim, lp in enumerate(loops):
        defid = getattr(lp.variable, "defid", None)
        if defid is None:
            return None
        try:
            r = _extract_loop_range(lp, expr_evaluator)
        except RuntimeError:
            r = None  # range depends on another loop var (e.g. j in k..n); extract inside loop
        name = getattr(lp.variable, "name", None)
        loop_info.append((defid, r, name))
    recurrence_loops = [loops[d] for d in recurrence_dims]
    _MAX = int(DEFAULT_EINSTEIN_LOOP_MAX)
    n_iter = [0]
    output_ndim = output.ndim
    has_literal = bool(clause_indices and any(isinstance(idx, LiteralIR) for idx in clause_indices))
    body_defids_by_name = _collect_defids_by_name(clause.body)

    try:
        for rec_context in execute_lowered_loops(recurrence_loops, {}, expr_evaluator):
            n_iter[0] += 1
            if n_iter[0] > _MAX:
                raise RuntimeError(
                    f"Einstein clause loop iterations exceeded limit ({_MAX}). "
                    "Reduce clause range or increase config.DEFAULT_EINSTEIN_LOOP_MAX."
                )
            with backend.env.scope():
                for dim in range(ndim):
                    defid, range_val, name = loop_info[dim]
                    if dim in recurrence_dims:
                        val = rec_context[defid]
                        backend.env.set_value(defid, val, name=name)
                        for other_defid in body_defids_by_name.get(name, []):
                            if other_defid != defid:
                                backend.env.set_value(other_defid, val, name=name)
                    else:
                        start, end = range_val if range_val is not None else (None, None)
                        if range_val is None:
                            try:
                                start, end = _extract_loop_range(
                                    loops[dim], lambda e: e.accept(backend)
                                )
                            except RuntimeError:
                                return None
                        sz = end - start
                        shape = [1] * ndim
                        shape[dim] = sz
                        arr = np.arange(start, end, dtype=np.intp).reshape(shape)
                        backend.env.set_value(defid, arr, name=name)
                result = clause.body.accept(backend)
            if not isinstance(result, np.ndarray):
                return None
            if has_literal and clause_indices is not None and loop_dims is not None and len(clause_indices) == output_ndim:
                slice_list_out: List[Any] = []
                loop_pos = 0
                for out_d, idx in enumerate(clause_indices):
                    if isinstance(idx, LiteralIR):
                        try:
                            slice_list_out.append(int(idx.value))
                        except (TypeError, ValueError):
                            return None
                    elif loop_pos < len(loops):
                        k = loop_pos
                        loop_pos += 1
                        if k in recurrence_dims:
                            v = rec_context[loop_info[k][0]]
                            slice_list_out.append(int(v) if hasattr(v, "__int__") else v)
                        else:
                            r = loop_info[k][1]
                            if r is None:
                                try:
                                    r = _extract_loop_range(
                                        loops[k], lambda e: e.accept(backend)
                                    )
                                except RuntimeError:
                                    return None
                            start, end = r
                            slice_list_out.append(slice(int(start), int(end)))
                    else:
                        return None
                if len(slice_list_out) != output_ndim:
                    return None
                to_write = result
                if to_write.ndim == ndim:
                    to_write = np.squeeze(to_write, axis=tuple(recurrence_dims))
                elif to_write.ndim != ndim - len(recurrence_dims):
                    axes = [d for d in recurrence_dims if d < to_write.ndim]
                    to_write = np.squeeze(to_write, axis=tuple(axes)) if axes else to_write
                output[tuple(slice_list_out)] = to_write.astype(output.dtype)
            else:
                slice_list: List[Any] = []
                for dim in range(ndim):
                    if dim in recurrence_dims:
                        slice_list.append(rec_context[loop_info[dim][0]])
                    else:
                        r = loop_info[dim][1]
                        if r is None:
                            try:
                                r = _extract_loop_range(
                                    loops[dim], lambda e: e.accept(backend)
                                )
                            except RuntimeError:
                                return None
                            start, end = r
                            slice_list.append(slice(int(start), int(end)))
                        else:
                            start, end = r
                            slice_list.append(slice(int(start), int(end)))
                try:
                    n_rec = len(recurrence_dims)
                    if result.ndim == ndim:
                        squeezed = np.squeeze(result, axis=tuple(recurrence_dims))
                    elif result.ndim == ndim - n_rec:
                        squeezed = result
                    else:
                        axes = [d for d in recurrence_dims if d < result.ndim]
                        squeezed = np.squeeze(result, axis=tuple(axes)) if axes else result
                except ValueError:
                    return None
                output[tuple(slice_list)] = squeezed.astype(output.dtype)
        return output
    except Exception:
        return None


def _try_call_scalar_vectorize_clause(
    clause: Any,
    output_shape: List[int],
    output: np.ndarray,
    scalar_loop_indices: List[int],
    expr_evaluator: Any,
    backend: Any,
) -> Optional[np.ndarray]:
    """
    When body has a non-element-wise call using some loop vars: iterate over those (scalar),
    vectorize over the rest. E.g. topk_2d_row_values(X, i, ...)[j]: scalar over i, vector over j.
    """
    from ..runtime.compute.lowered_execution import execute_lowered_loops
    loops = getattr(clause, "loops", None) or []
    if not loops or clause.guards or clause.bindings:
        return None
    ndim = len(loops)
    scalar_set = set(scalar_loop_indices)
    if not (0 < len(scalar_set) < ndim):
        return None
    vector_dims = [d for d in range(ndim) if d not in scalar_set]
    loop_info: List[Tuple[Any, Tuple[int, int], str]] = []
    for dim, lp in enumerate(loops):
        defid = getattr(lp.variable, "defid", None)
        if defid is None:
            return None
        try:
            r = _extract_loop_range(lp, expr_evaluator)
        except RuntimeError:
            return None
        name = getattr(lp.variable, "name", None)
        loop_info.append((defid, r, name))
    scalar_loops = [loops[d] for d in scalar_loop_indices]
    _MAX = int(DEFAULT_EINSTEIN_LOOP_MAX)
    n_iter = [0]
    try:
        for scalar_context in execute_lowered_loops(scalar_loops, {}, expr_evaluator):
            n_iter[0] += 1
            if n_iter[0] > _MAX:
                return None
            with backend.env.scope():
                for dim in range(ndim):
                    defid, (start, end), name = loop_info[dim]
                    if dim in scalar_set:
                        backend.env.set_value(defid, scalar_context[defid], name=name)
                    else:
                        sz = end - start
                        shape = [1] * ndim
                        shape[dim] = sz
                        arr = np.arange(start, end, dtype=np.intp).reshape(shape)
                        backend.env.set_value(defid, arr, name=name)
                result = clause.body.accept(backend)
            if not isinstance(result, np.ndarray):
                return None
            slice_list: List[Any] = []
            for dim in range(ndim):
                if dim in scalar_set:
                    slice_list.append(scalar_context[loop_info[dim][0]])
                else:
                    start, end = loop_info[dim][1]
                    slice_list.append(slice(int(start), int(end)))
            try:
                squeezed = np.squeeze(result, axis=tuple(scalar_loop_indices))
            except ValueError:
                return None
            output[tuple(slice_list)] = squeezed.astype(output.dtype)
        return output
    except Exception:
        return None


class EinsteinExecutionMixin:
    def visit_lowered_einstein_clause(self, node: LoweredEinsteinClauseIR) -> Any:
        stack = getattr(self, "_variable_decl_stack", None)
        variable_decl = stack[-1] if stack else None
        return self._execute_lowered_einstein_clause(node, variable_decl)

    def visit_lowered_einstein(self, node: LoweredEinsteinIR) -> Any:
        stack = getattr(self, "_variable_decl_stack", None)
        variable_decl = stack[-1] if stack else None
        return self._execute_lowered_einstein(node, variable_decl)

    def visit_lowered_recurrence(self, node: LoweredRecurrenceIR) -> Any:
        """Execute recurrence isolated out of Einstein: run initial once, then for each recurrence_loop value run body clauses."""
        stack = getattr(self, "_variable_decl_stack", None)
        variable_decl = stack[-1] if stack else None
        if variable_decl is None:
            raise RuntimeError("LoweredRecurrenceIR executed without variable_decl on stack")
        from ..runtime.compute.lowered_execution import execute_lowered_loops
        output = self._execute_lowered_einstein(node.initial, variable_decl)
        binding = getattr(variable_decl, "_binding", None) or variable_decl
        variable_key = getattr(binding, "defid", None) or getattr(variable_decl, "defid", None)
        variable_defid = variable_key
        tensor_shape = list(output.shape) if output is not None else None
        tensor_element_type = getattr(node.initial, "element_type", None)

        def expr_eval(e: Any) -> Any:
            return e.accept(self)

        recurrence_loops_for_outer = [node.recurrence_loop]
        _MAX = int(DEFAULT_EINSTEIN_LOOP_MAX)
        n_iter = [0]
        first_ctx_iter = execute_lowered_loops(recurrence_loops_for_outer, {}, expr_eval)
        first_ctx = next(first_ctx_iter, None)
        step_items: List[Any] = []
        if first_ctx is not None:
            for item in node.body.items:
                step_ok = self._execute_lowered_einstein_clause_one_recurrence_step(
                    item, variable_decl, output, variable_key, variable_defid,
                    first_ctx, recurrence_loops_for_outer, expr_eval, tensor_shape, tensor_element_type,
                )
                if step_ok is not None and variable_key is not None:
                    self.env.set_value(variable_key, output)
                    step_items.append(item)
                else:
                    self.env.set_value(variable_key, output)
                    full_result = self._execute_lowered_einstein_clause(
                        item, variable_decl,
                        shape=tensor_shape, element_type=tensor_element_type,
                        pre_allocated_output=None,
                    )
                    if full_result is not None and variable_key is not None:
                        clause_indices = getattr(item, "indices", None) or []
                        if clause_indices and len(clause_indices) == output.ndim:
                            slices_list = _slice_list_from_clause_indices(clause_indices, item, expr_eval) or []
                            if len(slices_list) == output.ndim:
                                if full_result.shape == output.shape:
                                    output[tuple(slices_list)] = full_result[tuple(slices_list)].astype(output.dtype)
                                else:
                                    output[tuple(slices_list)] = full_result.astype(output.dtype)
                                self.env.set_value(variable_key, output)
        else:
            step_items = list(node.body.items)
        for rec_context in first_ctx_iter:
            n_iter[0] += 1
            if n_iter[0] > _MAX:
                raise RuntimeError(
                    f"Einstein recurrence loop iterations exceeded limit ({_MAX}). "
                    "Reduce clause range or increase config.DEFAULT_EINSTEIN_LOOP_MAX."
                )
            for item in step_items:
                result = self._execute_lowered_einstein_clause_one_recurrence_step(
                    item, variable_decl, output, variable_key, variable_defid,
                    rec_context, recurrence_loops_for_outer, expr_eval, tensor_shape, tensor_element_type,
                )
                if result is not None and variable_key is not None:
                    self.env.set_value(variable_key, output)
        return output

    def _primitive_type_to_numpy_dtype(self, type_obj: Any) -> Optional[Any]:
        from ..shared.types import PrimitiveType
        if not isinstance(type_obj, PrimitiveType):
            return None
        type_name = type_obj.name.lower()
        dtype_map = {
            "i8": np.int8, "i32": np.int32, "i64": np.int64,
            "f16": np.float16, "f32": np.float32, "f64": np.float64,
            "bool": np.bool_, "int": np.int32, "float": np.float32,
        }
        try:
            import ml_dtypes
            dtype_map["bf16"] = ml_dtypes.bfloat16
            dtype_map["f8e4m3"] = ml_dtypes.float8_e4m3fn
        except ImportError:
            pass
        return dtype_map.get(type_name)

    def _type_info_to_numpy_dtype(self, type_info: Any) -> Optional[Any]:
        from ..shared.types import PrimitiveType, RectangularType, Type, TypeKind
        if type_info is None:
            return None
        if isinstance(type_info, Type) and hasattr(type_info, "kind") and type_info.kind == TypeKind.UNKNOWN:
            return None
        if isinstance(type_info, PrimitiveType):
            return self._primitive_type_to_numpy_dtype(type_info)
        if isinstance(type_info, RectangularType):
            return self._type_info_to_numpy_dtype(type_info.element_type)
        return None

    def _dtype_for_clause_result(self, clause_body: Any, tensor_element_type: Any) -> Any:
        """Dtype from type pass only: tensor_element_type, then clause body type_info."""
        dtype = self._type_info_to_numpy_dtype(tensor_element_type)
        if dtype is not None:
            return dtype
        if clause_body is None:
            return np.int32
        type_info = getattr(clause_body, "type_info", None)
        if type_info is not None:
            dtype = self._type_info_to_numpy_dtype(type_info)
            if dtype is not None:
                return dtype
        if isinstance(clause_body, (LoweredReductionIR, ReductionExpressionIR)):
            body_expr = getattr(clause_body, "body", None)
            if body_expr is not None:
                ti = getattr(body_expr, "type_info", None)
                if ti is not None:
                    dtype = self._type_info_to_numpy_dtype(ti)
                    if dtype is not None:
                        return dtype
        return np.int32

    def _get_defid_for_pattern_var(self, var_name: str, pattern: Any) -> Optional[DefId]:
        if hasattr(pattern, "name") and pattern.name == var_name:
            return pattern.defid
        if hasattr(pattern, "inner_pattern"):
            return self._get_defid_for_pattern_var(var_name, pattern.inner_pattern)
        if hasattr(pattern, "patterns"):
            for nested in pattern.patterns:
                defid = self._get_defid_for_pattern_var(var_name, nested)
                if defid is not None:
                    return defid
        return None

    def _execute_lowered_einstein(self, lowered_einstein: LoweredEinsteinIR, variable_decl: Any) -> Any:
        from ..runtime.compute.lowered_execution import execute_lowered_loops, execute_lowered_bindings, check_lowered_guards
        binding = getattr(variable_decl, "_binding", None)
        variable_key = (getattr(binding, "defid", None) if binding else None) or getattr(variable_decl, "defid", None)
        tensor_shape = lowered_einstein.shape
        tensor_element_type = lowered_einstein.element_type
        output_shape = None
        if tensor_shape:
            output_shape = []
            for shape_dim in tensor_shape:
                dim_value = shape_dim.accept(self)
                if isinstance(dim_value, (int, np.integer)):
                    output_shape.append(int(dim_value))
                elif isinstance(dim_value, np.ndarray) and dim_value.ndim == 0:
                    try:
                        output_shape.append(int(dim_value))
                    except (TypeError, ValueError):
                        output_shape = None
                        break
                else:
                    output_shape = None
                    break
        if not output_shape and lowered_einstein.items:
            output_shape = self._shape_from_all_items(lowered_einstein.items)
        elif output_shape and lowered_einstein.items:
            # Compiler shape may underestimate for multi-segment declarations
            # (e.g. _compute_shape_union picks a symbolic expr that evaluates to
            # the first clause's end, not the union).  Widen to cover all items.
            items_shape = self._shape_from_all_items(lowered_einstein.items)
            if items_shape and len(items_shape) == len(output_shape):
                output_shape = [max(a, b) for a, b in zip(output_shape, items_shape)]
        if not output_shape and lowered_einstein.items:
            raise RuntimeError(
                "Einstein declaration has no shape from compiler. "
                "Compiler must set shape (union of clause ranges) on LoweredEinsteinIR."
            )
        if not output_shape:
            output_shape = [1]
        dtype = self._type_info_to_numpy_dtype(tensor_element_type)
        if dtype is None:
            dtype = np.int32

        # Multi-segment: reuse existing array if this variable was already
        # declared (e.g. pad's `let result[i in 0..p] = ...; let result[i in p..n] = ...;`)
        existing = self.env.get_value(variable_key) if variable_key is not None else None
        if existing is not None and isinstance(existing, np.ndarray):
            needed = tuple(output_shape)
            current = existing.shape
            if len(needed) == len(current) and any(n > c for n, c in zip(needed, current)):
                new_shape = tuple(max(n, c) for n, c in zip(needed, current))
                output = np.zeros(new_shape, dtype=existing.dtype)
                slices = tuple(slice(0, s) for s in current)
                output[slices] = existing
                self.env.set_value(variable_key, output)
            else:
                output = existing
        else:
            output = np.zeros(output_shape, dtype=dtype)
            if variable_key is not None:
                self.env.set_value(variable_key, output)

        def expr_eval(e: Any) -> Any:
            return e.accept(self)

        items = lowered_einstein.items
        binding = getattr(variable_decl, "_binding", None)
        variable_defid = (getattr(binding, "defid", None) if binding else None) or getattr(variable_decl, "defid", None)

        # Execution order (mental model): (1) run all non-recurrence clauses in source order; (2) for each value of
        # the recurrence dimension (outermost), run all recurrence clauses in source order. So: recurrence dim
        # outermost, all clauses that write at that index run inside, preserving .ein clause order.
        # Partition: recurrence = any backward-in-time (t-1) or same-timestep (state[t,0] when writing state[t,1]).
        # Use _recurrence_dims so we accept both; clauses like LSTM hidden (reads t-1 and t) stay in recurrence_items and run after cell.
        recurrence_items: List[Any] = []
        non_recurrence_items: List[Any] = []
        recurrence_loops_for_outer: Optional[List[Any]] = None
        if len(items) > 1 and variable_defid is not None:
            for it in items:
                clause_indices = getattr(it, "indices", None) or []
                loops_it = getattr(it, "loops", None) or []
                # RecurrenceOrderPass may set recurrence_dims_override for same-timestep dependent clauses.
                rec_dims = getattr(it, "recurrence_dims_override", None)
                if rec_dims is None:
                    rec_dims = _recurrence_dims_for_hybrid(it, variable_defid, clause_indices)
                if not rec_dims:
                    rec_dims = _recurrence_dims(it, variable_defid, clause_indices)
                body_refs = _BodyReferencesDefidVisitor(variable_defid).references(getattr(it, "body", None))
                # Recurrence = has recurrence dim(s). Allow pure recurrence (only t) so t is extracted as outer loop.
                # When len(rec_dims) < len(loops_it) we vectorize over the rest; when equal we run one scalar/point per t.
                has_rec = bool(
                    rec_dims
                    and body_refs
                    and 0 < len(rec_dims) <= len(loops_it)
                )
                if has_rec:
                    recurrence_items.append(it)
                    if recurrence_loops_for_outer is None:
                        recurrence_loops_for_outer = [it.loops[d] for d in rec_dims]
                else:
                    non_recurrence_items.append(it)
        use_timestep_major = bool(recurrence_items and recurrence_loops_for_outer)

        # Run non-recurrence items first (e.g. state[0,...] initial conditions).
        for item in non_recurrence_items:
            result = self._execute_lowered_einstein_clause(
                item, variable_decl,
                shape=tensor_shape, element_type=tensor_element_type,
                pre_allocated_output=output,
            )
            if result is not None and variable_key is not None:
                if result.shape == output.shape:
                    if result is not output:
                        output[:] = result.astype(output.dtype)
                elif result.shape != output.shape:
                    slices_list_nr: List[Any] = []
                    clause_indices = getattr(item, "indices", None) or []
                    if clause_indices and len(clause_indices) == output.ndim:
                        slices_list_nr = _slice_list_from_clause_indices(clause_indices, item, expr_eval) or []
                    if len(slices_list_nr) != output.ndim and getattr(item, "loops", None):
                        try:
                            for lp in item.loops:
                                start, end = _extract_loop_range(lp, expr_eval)
                                slices_list_nr.append(slice(int(start), int(end)))
                        except RuntimeError:
                            slices_list_nr = []
                    if len(slices_list_nr) == output.ndim:
                        output[tuple(slices_list_nr)] = result.astype(output.dtype)
                    elif result.size == 1 and getattr(item, "indices", None) and all(
                        isinstance(idx, LiteralIR) for idx in item.indices
                    ):
                        idx_tuple = tuple(int(idx.value) for idx in item.indices)
                        output[idx_tuple] = result.flat[0] if result.size == 1 else result
                self.env.set_value(variable_key, output)

        # When we have recurrence items: run recurrence dim outermost, all recurrence clauses inside (timestep-major).
        if use_timestep_major and recurrence_loops_for_outer:
            from ..runtime.compute.lowered_execution import execute_lowered_loops
            _MAX = int(DEFAULT_EINSTEIN_LOOP_MAX)
            n_iter = [0]
            try:
                # Items whose step returns None: run full clause once and write slice (e.g. clauses with literal indices).
                step_items: List[Any] = []
                first_ctx_iter = execute_lowered_loops(recurrence_loops_for_outer, {}, expr_eval)
                first_ctx = next(first_ctx_iter, None)
                if first_ctx is not None:
                    for idx, item in enumerate(recurrence_items):
                        step_ok = self._execute_lowered_einstein_clause_one_recurrence_step(
                            item, variable_decl, output, variable_key, variable_defid,
                            first_ctx, recurrence_loops_for_outer, expr_eval, tensor_shape, tensor_element_type,
                        )
                        if step_ok is not None and variable_key is not None:
                            self.env.set_value(variable_key, output)
                            step_items.append(item)
                        else:
                            # Run full clause using existing output so body reads current state (e.g. state[t,0] for hidden).
                            self.env.set_value(variable_key, output)
                            full_result = self._execute_lowered_einstein_clause(
                                item, variable_decl,
                                shape=tensor_shape, element_type=tensor_element_type,
                                pre_allocated_output=output,
                            )
                            if full_result is not None and variable_key is not None:
                                clause_indices = getattr(item, "indices", None) or []
                                if clause_indices and len(clause_indices) == output.ndim:
                                    slices_list = _slice_list_from_clause_indices(clause_indices, item, expr_eval) or []
                                    if len(slices_list) == output.ndim:
                                        if full_result.shape == output.shape:
                                            output[tuple(slices_list)] = full_result[tuple(slices_list)].astype(output.dtype)
                                        else:
                                            output[tuple(slices_list)] = full_result.astype(output.dtype)
                                        self.env.set_value(variable_key, output)
                else:
                    step_items = list(recurrence_items)
                rec_loop = first_ctx_iter if first_ctx is not None else execute_lowered_loops(recurrence_loops_for_outer, {}, expr_eval)
                for rec_context in rec_loop:
                    n_iter[0] += 1
                    if n_iter[0] > _MAX:
                        raise RuntimeError(
                            f"Einstein recurrence loop iterations exceeded limit ({_MAX}). "
                            "Reduce clause range or increase config.DEFAULT_EINSTEIN_LOOP_MAX."
                        )
                    for item in step_items:
                        result = self._execute_lowered_einstein_clause_one_recurrence_step(
                            item, variable_decl, output, variable_key, variable_defid,
                            rec_context, recurrence_loops_for_outer, expr_eval, tensor_shape, tensor_element_type,
                        )
                        if result is not None and variable_key is not None:
                            self.env.set_value(variable_key, output)
                return output
            except Exception:
                use_timestep_major = False
        # Recurrence items not run in timestep-major (fallback or single-clause): run them in clause order.
        run_in_clause_order = recurrence_items if recurrence_items else items
        for clause_idx, item in enumerate(run_in_clause_order):
            result = self._execute_lowered_einstein_clause(
                item, variable_decl,
                shape=tensor_shape, element_type=tensor_element_type,
                pre_allocated_output=output,
            )
            if result is not None and variable_key is not None:
                if result.shape == output.shape:
                    if result is not output:
                        output[:] = result.astype(output.dtype)
                elif result.shape != output.shape:
                    slices_list: List[Any] = []
                    clause_indices = getattr(item, "indices", None) or []
                    if clause_indices and len(clause_indices) == output.ndim:
                        slices_list = _slice_list_from_clause_indices(clause_indices, item, expr_eval) or []
                    if len(slices_list) != output.ndim and getattr(item, "loops", None):
                        slices_list = []
                        try:
                            for lp in item.loops:
                                start, end = _extract_loop_range(lp, expr_eval)
                                slices_list.append(slice(int(start), int(end)))
                        except RuntimeError:
                            slices_list = []
                    if len(slices_list) == output.ndim:
                        output[tuple(slices_list)] = result.astype(output.dtype)
                    elif result.size == 1 and getattr(item, "indices", None) and all(
                        isinstance(idx, LiteralIR) for idx in item.indices
                    ):
                        idx_tuple = tuple(int(idx.value) for idx in item.indices)
                        output[idx_tuple] = result.flat[0] if result.size == 1 else result
                self.env.set_value(variable_key, output)
        return output

    def _execute_lowered_einstein_clause_one_recurrence_step(
        self,
        item: Any,
        variable_decl: Any,
        output: np.ndarray,
        variable_key: Any,
        variable_defid: Any,
        rec_context: Dict[Any, Any],
        recurrence_loops_outer: List[Any],
        expr_eval: Any,
        tensor_shape: Optional[List] = None,
        tensor_element_type: Optional[Any] = None,
    ) -> Optional[Any]:
        """Run one clause for one recurrence step (rec_context); vectorize over other dims. Used by timestep-major.
        recurrence_loops_outer: loops we iterate over (same order as rec_context keys); use their variable defids
        so every clause gets the current timestep even if its own loop var has a different defid."""
        from ..runtime.compute.lowered_execution import execute_lowered_bindings
        loops = getattr(item, "loops", None) or []
        if not loops:
            return None
        guards = getattr(item, "guards", None) or []
        clause_indices = getattr(item, "indices", None) or []
        recurrence_dims = getattr(item, "recurrence_dims_override", None)
        if not recurrence_dims and variable_defid:
            recurrence_dims = _recurrence_dims(item, variable_defid, clause_indices)
        recurrence_dims = recurrence_dims or []
        ndim = len(loops)
        # Allow clause to run when we have at least as many loops as outer (so we can bind current step).
        if not recurrence_dims or len(loops) < len(recurrence_loops_outer):
            return None
        loop_info: List[Tuple[Any, Tuple[int, int], str]] = []
        for dim, lp in enumerate(loops):
            defid = getattr(lp.variable, "defid", None)
            if defid is None:
                return None
            try:
                r = _extract_loop_range(lp, expr_eval)
            except RuntimeError:
                return None
            name = getattr(lp.variable, "name", None)
            loop_info.append((defid, r, name))
        # Map clause dim k to k-th outer loop's defid (so all outer vars bound from rec_context for current step).
        outer_rec_defids = []
        for k in range(min(len(recurrence_loops_outer), ndim)):
            outer_defid = getattr(recurrence_loops_outer[k].variable, "defid", None)
            if outer_defid is None:
                return None
            outer_rec_defids.append((k, outer_defid))
        outer_dims_set = {d for d, _ in outer_rec_defids}
        inner_recurrence_dims = [d for d in recurrence_dims if d not in outer_dims_set]
        # All clause dims bound from current step (e.g. diagonal clause with i==j in row-major): run body once, write one cell.
        if len(outer_dims_set) == ndim and not inner_recurrence_dims:
            with self.env.scope():
                self.env.set_value(variable_defid, output)
                for _d, odef in outer_rec_defids:
                    if odef in rec_context:
                        self.env.set_value(odef, rec_context[odef])
                for dim in range(ndim):
                    defid, (_start, _end), name = loop_info[dim]
                    od = next((o for d, o in outer_rec_defids if d == dim), None)
                    if od is not None and od in rec_context:
                        self.env.set_value(defid, rec_context[od], name=name)
                    else:
                        self.env.set_value(defid, np.arange(_start, _end, dtype=np.intp), name=name)
                bindings = getattr(item, "bindings", None) or []
                if bindings:
                    from ..runtime.compute.lowered_execution import execute_lowered_bindings
                    loop_context = {getattr(lp.variable, "defid", None): self.env.get_value(getattr(lp.variable, "defid", None)) for lp in loops if getattr(lp.variable, "defid", None) is not None}
                    full_context = execute_lowered_bindings(bindings, loop_context, expr_eval)
                    for d, val in full_context.items():
                        if d is not None:
                            self.env.set_value(d, val)
                if guards:
                    from ..runtime.compute.lowered_execution import check_lowered_guards
                    step_ctx = {loop_info[dim][0]: rec_context.get(odef) for dim, (_, odef) in enumerate(outer_rec_defids) if odef in rec_context}
                    if not check_lowered_guards(guards, step_ctx, lambda e: self._to_bool(e.accept(self))):
                        if variable_key is not None:
                            self.env.set_value(variable_key, output)
                        return output
                _saved_rec = getattr(self, "_einstein_recurrence_clause", False)
                _saved_vec = getattr(self, "_vectorize_parallel_shape", None)
                try:
                    self._einstein_recurrence_clause = True  # so reduction in body uses env (current L), not fast path
                    self._vectorize_parallel_shape = None    # force scalar reduction path to use env
                    res = item.body.accept(self)
                finally:
                    self._einstein_recurrence_clause = _saved_rec
                    self._vectorize_parallel_shape = _saved_vec
            if res is not None:
                res = np.asarray(res, dtype=output.dtype)
                slice_list: List[Any] = []
                for dim in range(ndim):
                    od = next((o for d, o in outer_rec_defids if d == dim), None)
                    if od is not None and od in rec_context:
                        v = rec_context[od]
                        slice_list.append(int(v) if isinstance(v, (np.integer, np.floating, int, float)) else v)
                    else:
                        slice_list.append(slice(loop_info[dim][1][0], loop_info[dim][1][1]))
                if len(slice_list) == output.ndim and res.size == 1:
                    val = res.flat[0]
                    output[tuple(slice_list)] = val
                    if variable_key is not None:
                        self.env.set_value(variable_key, output)
                    return output
        # Inner recurrence dims: isolate as outer loop (iterate in order). In the body they are literals; vectorize other dims.
        if inner_recurrence_dims:
            from ..runtime.compute.lowered_execution import execute_lowered_loops
            inner_loops = [loops[d] for d in inner_recurrence_dims]
            _saved = getattr(self, "_einstein_recurrence_clause", False)
            try:
                self._einstein_recurrence_clause = True
                for inner_ctx in execute_lowered_loops(inner_loops, {}, expr_eval):
                    with self.env.scope():
                        self.env.set_value(variable_defid, output)
                        for _d, odef in outer_rec_defids:
                            if odef in rec_context:
                                self.env.set_value(odef, rec_context[odef])
                        for dim in range(ndim):
                            defid, (start, end), name = loop_info[dim]
                            if dim in outer_dims_set:
                                od = next((o for d, o in outer_rec_defids if d == dim), None)
                                if od is not None and od in rec_context:
                                    self.env.set_value(defid, rec_context[od], name=name)
                                else:
                                    self.env.set_value(defid, np.arange(start, end, dtype=np.intp), name=name)
                            elif dim in inner_recurrence_dims:
                                if defid in inner_ctx:
                                    self.env.set_value(defid, inner_ctx[defid], name=name)
                                else:
                                    self.env.set_value(defid, np.arange(start, end, dtype=np.intp), name=name)
                            else:
                                sz = end - start
                                shape = [1] * ndim
                                shape[dim] = sz
                                arr = np.arange(start, end, dtype=np.intp).reshape(shape)
                                self.env.set_value(defid, arr, name=name)
                        bindings = getattr(item, "bindings", None) or []
                        if bindings:
                            loop_context = {}
                            for lp in loops:
                                d = getattr(lp.variable, "defid", None)
                                if d is not None:
                                    loop_context[d] = self.env.get_value(d)
                            full_context = execute_lowered_bindings(bindings, loop_context, expr_eval)
                            for d, val in full_context.items():
                                if d is not None:
                                    self.env.set_value(d, val)
                        res = item.body.accept(self)
                    if res is None:
                        continue
                        res = np.asarray(res, dtype=output.dtype)
                    # Build slice: literals (outer + inner recurrence) as scalar indices; other dims as slice() for vectorized write.
                    # Use loop_pos to index loop_info (only advance when we consume a loop), so literal indices don't skip a dimension.
                    slice_list: List[Any] = []
                    loop_pos = 0
                    for pos in range(min(len(clause_indices), output.ndim)):
                        idx = clause_indices[pos] if pos < len(clause_indices) else None
                        if isinstance(idx, LiteralIR):
                            try:
                                slice_list.append(int(idx.value))
                            except (TypeError, ValueError):
                                break
                            continue
                        if loop_pos >= len(loop_info):
                            break
                        if loop_pos in outer_dims_set:
                            od = next((o for d, o in outer_rec_defids if d == loop_pos), None)
                            v = rec_context[od] if od is not None and od in rec_context else inner_ctx.get(loop_info[loop_pos][0], 0)
                            slice_list.append(int(v) if isinstance(v, (np.integer, np.floating)) else v)
                        elif loop_pos in inner_recurrence_dims:
                            v = inner_ctx.get(loop_info[loop_pos][0], 0)
                            slice_list.append(int(v) if isinstance(v, (np.integer, np.floating)) else v)
                        else:
                            _, (start, end), _ = loop_info[loop_pos]
                            slice_list.append(slice(int(start), int(end)))
                        loop_pos += 1
                    if len(slice_list) == output.ndim:
                        if res.size == 1:
                            output[tuple(slice_list)] = res.flat[0]
                        else:
                            output[tuple(slice_list)] = res
                self.env.set_value(variable_key, output)
                return output
            finally:
                self._einstein_recurrence_clause = _saved
        # When body is a block (e.g. RNN recurrence with let + if), use scalar iteration over non-recurrence dims
        # so that reductions in the body see the same env as the clause's loop vars (avoids vectorization bugs).
        # Only when clause indices match loop count (no literal indices like LSTM state[t, slot, b, h]) so slice building is correct.
        non_rec_loops = [loops[i] for i in range(ndim) if i not in recurrence_dims]
        if (
            isinstance(getattr(item, "body", None), BlockExpressionIR)
            and non_rec_loops
            and clause_indices
            and len(clause_indices) == len(loops)
        ):
            from ..runtime.compute.lowered_execution import execute_lowered_loops
            _saved_rec = getattr(self, "_einstein_recurrence_clause", False)
            try:
                self._einstein_recurrence_clause = True
                for inner_ctx in execute_lowered_loops(non_rec_loops, {}, expr_eval):
                    with self.env.scope():
                        self.env.set_value(variable_defid, output)
                        for _d, odef in outer_rec_defids:
                            if odef in rec_context:
                                self.env.set_value(odef, rec_context[odef])
                        for dim in range(ndim):
                            defid, (start, end), name = loop_info[dim]
                            if dim in recurrence_dims:
                                od = next((o for d, o in outer_rec_defids if d == dim), None)
                                if od is not None and od in rec_context:
                                    self.env.set_value(defid, rec_context[od], name=name)
                                else:
                                    self.env.set_value(defid, np.arange(start, end, dtype=np.intp), name=name)
                            else:
                                if defid in inner_ctx:
                                    self.env.set_value(defid, inner_ctx[defid], name=name)
                                else:
                                    self.env.set_value(defid, np.arange(start, end, dtype=np.intp), name=name)
                        bindings = getattr(item, "bindings", None) or []
                        if bindings:
                            loop_context = {}
                            for lp in loops:
                                d = getattr(lp.variable, "defid", None)
                                if d is not None:
                                    loop_context[d] = self.env.get_value(d)
                            full_context = execute_lowered_bindings(bindings, loop_context, expr_eval)
                            for d, val in full_context.items():
                                if d is not None:
                                    self.env.set_value(d, val)
                        res = item.body.accept(self)
                    if not isinstance(res, np.ndarray):
                        continue
                    scalar = res.flat[0] if res.size == 1 else res
                    slice_list_scalar: List[Any] = []
                    for pos, idx in enumerate(clause_indices):
                        if pos >= len(loops):
                            break
                        if isinstance(idx, LiteralIR):
                            try:
                                slice_list_scalar.append(int(idx.value))
                            except (TypeError, ValueError):
                                break
                        elif pos in recurrence_dims:
                            od = next((o for d, o in outer_rec_defids if d == pos), None)
                            if od is not None and od in rec_context:
                                slice_list_scalar.append(rec_context[od])
                            else:
                                slice_list_scalar.append(inner_ctx.get(loop_info[pos][0], 0))
                        else:
                            slice_list_scalar.append(inner_ctx.get(loop_info[pos][0], 0))
                    if len(slice_list_scalar) == output.ndim:
                        output[tuple(slice_list_scalar)] = np.asarray(scalar, dtype=output.dtype)
                self.env.set_value(variable_key, output)
                return output
            finally:
                self._einstein_recurrence_clause = _saved_rec
        _saved_recurrence_flag = getattr(self, "_einstein_recurrence_clause", False)
        _saved_vectorize_shape = getattr(self, "_vectorize_parallel_shape", None)
        try:
            self._einstein_recurrence_clause = True  # so reduction in body uses env (correct t), not fast path
            self._vectorize_parallel_shape = None  # force scalar reduction path to use env's t
            with self.env.scope():
                self.env.set_value(variable_defid, output)
                # Bind outer recurrence loop vars so body sees current timestep (body may reference outer defid).
                for _d, outer_defid in outer_rec_defids:
                    if outer_defid in rec_context:
                        self.env.set_value(outer_defid, rec_context[outer_defid])
                for dim in range(ndim):
                    defid, (start, end), name = loop_info[dim]
                    if dim in recurrence_dims:
                        outer_defid = next((od for d, od in outer_rec_defids if d == dim), None)
                        if outer_defid is not None and outer_defid in rec_context:
                            self.env.set_value(defid, rec_context[outer_defid], name=name)
                        else:
                            sz = end - start
                            shape = [1] * ndim
                            shape[dim] = sz
                            arr = np.arange(start, end, dtype=np.intp).reshape(shape)
                            self.env.set_value(defid, arr, name=name)
                    else:
                        sz = end - start
                        shape = [1] * ndim
                        shape[dim] = sz
                        arr = np.arange(start, end, dtype=np.intp).reshape(shape)
                        self.env.set_value(defid, arr, name=name)
                bindings = getattr(item, "bindings", None) or []
                if bindings:
                    loop_context = {}
                    for lp in loops:
                        d = getattr(lp.variable, "defid", None)
                        if d is not None:
                            loop_context[d] = self.env.get_value(d)
                    full_context = execute_lowered_bindings(bindings, loop_context, expr_eval)
                    for d, val in full_context.items():
                        if d is not None:
                            self.env.set_value(d, val)
                result = item.body.accept(self)
            result = np.asarray(result) if result is not None else None
            if result is not None and not isinstance(result, np.ndarray):
                result = np.asarray(result)
            if result is None or not isinstance(result, np.ndarray):
                return None
            # Build output slice from clause indices (same length as output.ndim; literals and recurrence bound).
            if not clause_indices or len(clause_indices) != output.ndim:
                return None
            slice_list: List[Any] = []
            loop_pos = 0
            for idx in clause_indices:
                # Literal or constant (e.g. 10, 0 in u[t, i, 10, 0]) so we don't consume a loop.
                literal_val = None
                if isinstance(idx, LiteralIR):
                    try:
                        literal_val = int(idx.value)
                    except (TypeError, ValueError):
                        pass
                elif getattr(idx, "value", None) is not None:
                    try:
                        literal_val = int(idx.value)
                    except (TypeError, ValueError):
                        pass
                if literal_val is not None:
                    slice_list.append(literal_val)
                    continue
                if loop_pos < len(loops):
                    if loop_pos in recurrence_dims:
                        outer_defid = next((od for d, od in outer_rec_defids if d == loop_pos), None)
                        if outer_defid is not None and outer_defid in rec_context:
                            slice_list.append(rec_context[outer_defid])
                        else:
                            start, end = loop_info[loop_pos][1]
                            slice_list.append(slice(int(start), int(end)))
                    else:
                        start, end = loop_info[loop_pos][1]
                        slice_list.append(slice(int(start), int(end)))
                    loop_pos += 1
                else:
                    return None
            if loop_pos != len(loops):
                return None
            try:
                n_outer = len(recurrence_loops_outer)
                if result.ndim == ndim:
                    squeezed = np.squeeze(result, axis=tuple(range(n_outer)))
                elif result.ndim == ndim - n_outer:
                    squeezed = result
                else:
                    axes = [d for d in range(n_outer) if d < result.ndim]
                    squeezed = np.squeeze(result, axis=tuple(axes)) if axes else result
            except ValueError:
                return None
            output[tuple(slice_list)] = squeezed.astype(output.dtype)
            return output
        finally:
            self._einstein_recurrence_clause = _saved_recurrence_flag
            self._vectorize_parallel_shape = _saved_vectorize_shape

    def _shape_from_all_items(self, items: List) -> Optional[List[int]]:
        """Compute output shape from the max absolute end across ALL items (not just the first)."""
        if not items:
            return None
        rank = None
        max_ends: Optional[List[int]] = None
        for item in items:
            loops = getattr(item, "loops", None) or []
            if not loops:
                continue
            if rank is None:
                rank = len(loops)
                max_ends = [0] * rank
            if len(loops) != rank:
                return None
            for d, loop in enumerate(loops):
                it = getattr(loop, "iterable", None)
                if it is None:
                    return None
                if isinstance(it, LiteralIR) and isinstance(getattr(it, "value", None), range):
                    end = int(it.value.stop)
                elif isinstance(it, RangeIR):
                    try:
                        end = int(it.end.accept(self))
                    except (TypeError, ValueError):
                        return None
                else:
                    try:
                        r = it.accept(self)
                        end = len(r) if hasattr(r, "__len__") else None
                    except Exception:
                        return None
                    if end is None:
                        return None
                if end > max_ends[d]:
                    max_ends[d] = end
        return max_ends

    def _clause_set_output(self, variable_defid: Any, output: Any) -> None:
        """Set clause result in env."""
        if variable_defid is not None:
            self.env.set_value(variable_defid, output)

    def _execute_lowered_einstein_clause(
        self,
        lowered: LoweredEinsteinClauseIR,
        variable_decl: Any,
        shape: Optional[List] = None,
        element_type: Optional[Any] = None,
        pre_allocated_output: Optional[Any] = None,
    ) -> Any:
        from ..runtime.compute.lowered_execution import execute_lowered_loops, execute_lowered_bindings, check_lowered_guards
        loc = getattr(lowered, "location", None) or getattr(variable_decl, "location", None)
        line = int(getattr(loc, "line", 0) or 0)
        _clause_name = (
            getattr(variable_decl, "name", None)
            or getattr(getattr(variable_decl, "_binding", None), "name", None)
            or ""
        )
        _clause_rhs = _clause_body_summary(getattr(lowered, "body", None))
        bucket_size = getattr(self, "_profile_bucket_size", 0)
        _profile_clauses = getattr(self, "_profile_functions", False) or getattr(self, "_profile_statements", False)
        t0 = time.perf_counter() if (bucket_size > 0 or _profile_clauses) else 0
        def _record_profile(shape: Optional[tuple] = None, path: Optional[str] = None) -> None:
            if bucket_size > 0 and getattr(self, "_profile_buckets", None) is not None:
                key = (line // bucket_size) * bucket_size
                self._profile_buckets[key] = self._profile_buckets.get(key, 0) + (time.perf_counter() - t0)
            if _profile_clauses and line and t0:
                elapsed = time.perf_counter() - t0
                parts = [f"[profile] clause L{line}"]
                if _clause_name or _clause_rhs:
                    lhs = _clause_name or "?"
                    parts.append(f" {lhs} = {_clause_rhs}")
                if shape is not None:
                    parts.append(f" {shape}")
                parts.append(f": \033[32m{elapsed:.3f}s\033[0m")
                if path:
                    parts.append(f" [{path}]")
                print("".join(parts), flush=True)

        clause_indices = getattr(lowered, "indices", None) or []
        binding = getattr(variable_decl, "_binding", None)
        variable_defid = (getattr(binding, "defid", None) if binding else None) or getattr(variable_decl, "defid", None)
        body_node = getattr(lowered, "body", None)
        has_recurrence = bool(
            variable_defid is not None
            and body_node is not None
            and _BodyReferencesDefidVisitor(variable_defid).references(body_node)
            and len(_recurrence_dims(lowered, variable_defid, clause_indices)) > 0
        )
        self._einstein_recurrence_clause = has_recurrence

        def cell_index(full_context: dict) -> Optional[tuple]:
            out = []
            loop_pos = 0
            for idx in clause_indices:
                if isinstance(idx, LiteralIR):
                    try:
                        out.append(int(idx.value))
                    except (TypeError, ValueError):
                        break
                elif loop_pos < len(lowered.loops):
                    defid = lowered.loops[loop_pos].variable.defid
                    v = full_context.get(defid)
                    if v is None and defid is not None:
                        v = self.env.get_value(defid)
                    if v is None:
                        break
                    out.append(v)
                    loop_pos += 1
                else:
                    break
            return tuple(out) if len(out) == len(clause_indices) else None

        if pre_allocated_output is not None:
            output = pre_allocated_output
            if variable_defid is not None:
                self.env.set_value(variable_defid, output)
        else:
            output_shape = None
            if shape:
                output_shape = []
                for shape_dim in shape:
                    dim_value = shape_dim.accept(self)
                    if isinstance(dim_value, (int, np.integer)):
                        output_shape.append(int(dim_value))
                    elif isinstance(dim_value, np.ndarray) and dim_value.ndim == 0:
                        try:
                            output_shape.append(int(dim_value))
                        except (TypeError, ValueError):
                            output_shape = None
                            break
                    else:
                        output_shape = None
                        break
            if output_shape is None and lowered.loops:
                output_shape = []
                for loop in lowered.loops:
                    it = getattr(loop, "iterable", None)
                    if it is None:
                        output_shape = None
                        break
                    if isinstance(it, LiteralIR) and isinstance(getattr(it, "value", None), range):
                        output_shape.append(int(it.value.stop))
                    elif isinstance(it, RangeIR):
                        try:
                            end_val = int(it.end.accept(self))
                        except (TypeError, ValueError):
                            output_shape = None
                            break
                        output_shape.append(end_val)
                    else:
                        try:
                            r = it.accept(self)
                            output_shape.append(len(r) if hasattr(r, "__len__") else None)
                        except Exception:
                            output_shape = None
                        if output_shape and output_shape[-1] is None:
                            output_shape = None
                            break
            if output_shape is None:
                output_shape = [int(getattr(idx, "value", 0)) + 1 if isinstance(idx, LiteralIR) else 1 for idx in clause_indices] if clause_indices else [1]
            dtype = self._dtype_for_clause_result(lowered.body, element_type)
            output = np.zeros(output_shape, dtype=dtype)

        def expr_evaluator(expr: Any) -> Any:
            return expr.accept(self)

        has_literal_idx = any(isinstance(idx, LiteralIR) for idx in clause_indices)
        body_node = lowered.body
        loop_defids = [lp.variable.defid for lp in (lowered.loops or [])]
        has_call_using_loop = _body_contains_call_using_loop_var(body_node, [d for d in loop_defids if d is not None])
        # When body has a call that uses loop vars in its args (e.g. topk_2d_row_values(X, i, ...)), those vars must be scalar.
        # Try call-scalar first so we don't use wrong full-vectorize result (array-valued row index).
        if (
            lowered.loops
            and not has_literal_idx
            and has_call_using_loop
        ):
            scalar_defids = _loop_defids_in_call_args(body_node, loop_defids)
            scalar_loop_indices_call = [
                dim
                for dim, lp in enumerate(lowered.loops)
                if lp.variable.defid in scalar_defids
            ]
            if 0 < len(scalar_loop_indices_call) < len(lowered.loops):
                call_hybrid_out = _try_call_scalar_vectorize_clause(
                    lowered,
                    list(output.shape),
                    output,
                    scalar_loop_indices_call,
                    expr_evaluator,
                    backend=self,
                )
                if call_hybrid_out is not None:
                    if variable_defid:
                        self._clause_set_output(variable_defid, output)
                    self._einstein_call_scalar = getattr(self, "_einstein_call_scalar", 0) + 1
                    _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path="call-scalar")
                    return output
        # Literal idx / self-ref (recurrence) -> scalar; other indices -> vectorize.
        # When body has recurrence (reads LHS at different index), try hybrid first so we read prior timestep correctly.
        recurrence_needs_scalar = False
        if (
            lowered.loops
            and variable_defid is not None
            and _BodyReferencesDefidVisitor(variable_defid).references(body_node)
        ):
            recurrence_dims = _recurrence_dims(lowered, variable_defid, clause_indices)
            if 0 < len(recurrence_dims) < len(lowered.loops):
                hybrid_out = _try_hybrid_vectorize_clause(
                    lowered, list(output.shape), output, variable_defid, expr_evaluator, backend=self,
                    clause_indices=clause_indices,
                )
                if hybrid_out is not None:
                    if variable_defid:
                        self._clause_set_output(variable_defid, output)
                    self._einstein_hybrid = getattr(self, "_einstein_hybrid", 0) + 1
                    _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path="hybrid")
                    return output
                recurrence_needs_scalar = True  # hybrid failed; use scalar path so we read LHS[t-1] correctly
        # Try full vectorize over loop dims (literal idx -> fixed slice; other dims -> vectorize).
        if lowered.loops:
            # Slice-vectorize: body "if p < t then ... else 0" -> vectorize over [0..t), fill rest (e.g. emb in decode).
            if not recurrence_needs_scalar and not getattr(lowered, "guards", None) and not getattr(lowered, "bindings", None):
                slice_vec = _try_slice_vectorize_if_clause(lowered, output, expr_evaluator, backend=self)
                if slice_vec is not None:
                    if variable_defid:
                        self._clause_set_output(variable_defid, output)
                    self._einstein_vectorized = getattr(self, "_einstein_vectorized", 0) + 1
                    _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path="vectorized")
                    return output
            # Optional chunked execution to reduce peak memory (env EINLANG_CHUNK_ELEMENTS > 0).
            chunk_threshold = int(os.environ.get("EINLANG_CHUNK_ELEMENTS", "0") or "0")
            if (
                chunk_threshold > 0
                and output.size > chunk_threshold
                and not recurrence_needs_scalar
                and not has_literal_idx
                and len(lowered.loops) == output.ndim
            ):
                try:
                    full_ranges = [_extract_loop_range(lp, expr_evaluator) for lp in lowered.loops]
                    if len(full_ranges) == output.ndim and output.shape[0] > 1:
                        rest_size = max(1, output.size // output.shape[0])
                        chunk_rows = max(1, min(output.shape[0], chunk_threshold // rest_size))
                        if chunk_rows < output.shape[0]:
                            all_ok = True
                            for start in range(0, output.shape[0], chunk_rows):
                                end = min(start + chunk_rows, output.shape[0])
                                override = [(start, end)] + list(full_ranges[1:])
                                chunk_shape = [end - start] + list(output.shape[1:])
                                chunk_result = _try_vectorize_clause(
                                    lowered, chunk_shape, output.dtype, expr_evaluator, backend=self,
                                    loop_ranges_override=override,
                                )
                                if chunk_result is None:
                                    all_ok = False
                                    break
                                output[start:end, ...] = chunk_result.astype(output.dtype, copy=False)
                            if all_ok:
                                if variable_defid:
                                    self._clause_set_output(variable_defid, output)
                                self._einstein_vectorized = getattr(self, "_einstein_vectorized", 0) + 1
                                _path = getattr(self, "_last_reduction_fast_path", None) or "vectorized"
                                if hasattr(self, "_last_reduction_fast_path"):
                                    delattr(self, "_last_reduction_fast_path")
                                _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path=_path)
                                return output
                except (RuntimeError, TypeError, ValueError):
                    pass
            # When clause has literal indices, vectorize only over loop dims so result shape matches loop dims.
            vec_shape = list(output.shape)
            if has_literal_idx and len(clause_indices) == output.ndim:
                try:
                    vec_shape = [int(_extract_loop_range(lp, expr_evaluator)[1]) - int(_extract_loop_range(lp, expr_evaluator)[0]) for lp in lowered.loops]
                except (RuntimeError, TypeError, ValueError):
                    vec_shape = list(output.shape)
            vec_result = _try_vectorize_clause(
                lowered, vec_shape, output.dtype, expr_evaluator, backend=self,
            )
            if recurrence_needs_scalar and vec_result is not None:
                vec_result = None  # force scalar path so recurrence reads prior timestep correctly
            if vec_result is not None:
                vec_result = np.asarray(vec_result)
                slice_list_from_indices = (
                    _slice_list_from_clause_indices(clause_indices, lowered, expr_evaluator)
                    if has_literal_idx and len(clause_indices) == output.ndim
                    else None
                )
                if vec_result.shape == output.shape:
                    if pre_allocated_output is not None and lowered.loops:
                        slices_list_partial: List[Any] = []
                        try:
                            if slice_list_from_indices is not None:
                                slices_list_partial = slice_list_from_indices
                            else:
                                for lp in lowered.loops:
                                    start, end = _extract_loop_range(lp, expr_evaluator)
                                    slices_list_partial.append(slice(int(start), int(end)))
                        except RuntimeError:
                            slices_list_partial = []
                        if slice_list_from_indices is not None and len(slices_list_partial) == output.ndim:
                            output[tuple(slices_list_partial)] = vec_result.astype(output.dtype, copy=False)
                        else:
                            range_is_full_partial = (
                                len(slices_list_partial) == len(lowered.loops)
                                and all(s.start == 0 and s.stop == output.shape[i] for i, s in enumerate(slices_list_partial) if isinstance(s, slice))
                            )
                            if len(slices_list_partial) == len(lowered.loops) and not range_is_full_partial:
                                recurrence_dims = _recurrence_dims(lowered, variable_defid, clause_indices) if _BodyReferencesDefidVisitor(variable_defid).references(lowered.body) else []
                                if recurrence_dims:
                                    hybrid_out = _try_hybrid_vectorize_clause(
                                        lowered, list(output.shape), output, variable_defid, expr_evaluator, backend=self,
                                        clause_indices=clause_indices,
                                    )
                                    if hybrid_out is not None:
                                        if variable_defid:
                                            self._clause_set_output(variable_defid, output)
                                        self._einstein_hybrid = getattr(self, "_einstein_hybrid", 0) + 1
                                        _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path="hybrid")
                                        return output
                                    vec_result = None
                                else:
                                    output[tuple(slices_list_partial)] = vec_result[tuple(slices_list_partial)].astype(output.dtype, copy=False)
                            else:
                                output[:] = vec_result
                    else:
                        output[:] = vec_result
                if vec_result is not None:
                    if slice_list_from_indices is not None and len(slice_list_from_indices) == output.ndim:
                        output[tuple(slice_list_from_indices)] = vec_result.astype(output.dtype, copy=False)
                    elif vec_result.shape != output.shape and vec_result.size == output.size:
                        output[:] = vec_result.reshape(output.shape)
                    elif pre_allocated_output is not None and vec_result.ndim == output.ndim:
                        slices_list: List[Any] = []
                        try:
                            if slice_list_from_indices is not None and len(slice_list_from_indices) == output.ndim:
                                slices_list = slice_list_from_indices
                            else:
                                for lp in lowered.loops:
                                    start, end = _extract_loop_range(lp, expr_evaluator)
                                    slices_list.append(slice(int(start), int(end)))
                        except RuntimeError:
                            slices_list = []
                        if slice_list_from_indices is not None and len(slices_list) == output.ndim:
                            output[tuple(slices_list)] = vec_result.astype(output.dtype, copy=False)
                        else:
                            range_is_full = (
                                len(slices_list) == len(lowered.loops)
                                and all(s.start == 0 and s.stop == output.shape[i] for i, s in enumerate(slices_list))
                            )
                            if len(slices_list) == len(lowered.loops):
                                if range_is_full:
                                    np.copyto(output, np.broadcast_to(vec_result.astype(output.dtype, copy=False), output.shape))
                                else:
                                    output[tuple(slices_list)] = vec_result[tuple(slices_list)].astype(output.dtype, copy=False)
                            else:
                                output[:] = np.broadcast_to(vec_result, output.shape)
                    else:
                        output[:] = np.broadcast_to(vec_result, output.shape)
                    if variable_defid:
                        self._clause_set_output(variable_defid, output)
                    self._einstein_vectorized = getattr(self, "_einstein_vectorized", 0) + 1
                    _path = getattr(self, "_last_reduction_fast_path", None) or "vectorized"
                    if hasattr(self, "_last_reduction_fast_path"):
                        delattr(self, "_last_reduction_fast_path")
                    _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path=_path)
                    return output

        # Fallback: call-scalar hybrid when only some loop vars in call args (e.g. topk) and full vectorize failed.
        if (
            lowered.loops
            and not has_literal_idx
            and has_call_using_loop
        ):
            scalar_defids = _loop_defids_in_call_args(body_node, loop_defids)
            scalar_loop_indices = [
                dim
                for dim, lp in enumerate(lowered.loops)
                if getattr(lp.variable, "defid", None) in scalar_defids
            ]
            if 0 < len(scalar_loop_indices) < len(lowered.loops):
                call_hybrid_out = _try_call_scalar_vectorize_clause(
                    lowered,
                    list(output.shape),
                    output,
                    scalar_loop_indices,
                    expr_evaluator,
                    backend=self,
                )
                if call_hybrid_out is not None:
                    if variable_defid:
                        self._clause_set_output(variable_defid, output)
                    self._einstein_call_scalar = getattr(self, "_einstein_call_scalar", 0) + 1
                    _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path="call-scalar")
                    return output

        # Element-wise call (e.g. gelu(fc1[s,k])): must run once with full array, not scalar loop.
        if (
            lowered.loops
            and _body_is_elementwise_call(body_node, loop_defids)
        ):
            elem_result = _eval_clause_body_with_broadcast_loops(
                lowered, list(output.shape), expr_evaluator, self
            )
            if elem_result is not None and isinstance(elem_result, np.ndarray):
                assigned = False
                if elem_result.shape == output.shape:
                    output[:] = elem_result.astype(output.dtype, copy=False)
                    assigned = True
                elif elem_result.size == output.size:
                    output.reshape(-1)[:] = elem_result.reshape(-1).astype(output.dtype, copy=False)
                    assigned = True
                else:
                    try:
                        np.copyto(output, np.broadcast_to(elem_result, output.shape))
                        assigned = True
                    except (ValueError, TypeError):
                        pass
                if assigned:
                    if variable_defid:
                        self._clause_set_output(variable_defid, output)
                    self._einstein_vectorized = getattr(self, "_einstein_vectorized", 0) + 1
                    _path = getattr(self, "_last_reduction_fast_path", None) or "vectorized"
                    if hasattr(self, "_last_reduction_fast_path"):
                        delattr(self, "_last_reduction_fast_path")
                    _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path=_path)
                    return output

        self._einstein_scalar = getattr(self, "_einstein_scalar", 0) + 1
        _loop_defid_to_name = {}
        _loops = lowered.loops
        for lp in _loops:
            v = getattr(lp, "variable", None)
            if v and getattr(v, "defid", None):
                _loop_defid_to_name[v.defid] = getattr(v, "name", None)
        _body = lowered.body
        _bindings = getattr(lowered, "bindings", None) or []
        _guards = getattr(lowered, "guards", None) or []
        # Precompute cell_index spec: list of (is_literal, value_or_defid) so we avoid getattr per iteration.
        _cell_index_spec: List[Any] = []
        _loop_pos = 0
        for idx in clause_indices:
            if isinstance(idx, LiteralIR):
                try:
                    _cell_index_spec.append((True, int(idx.value)))
                except (TypeError, ValueError):
                    _cell_index_spec = None
                    break
            elif _loop_pos < len(_loops):
                _defid = getattr(_loops[_loop_pos].variable, "defid", None)
                _cell_index_spec.append((False, _defid))
                _loop_pos += 1
            else:
                _cell_index_spec = None
                break
        if _cell_index_spec is not None and _loop_pos != len(_loops):
            _cell_index_spec = None
        _loop_defids_tuple = tuple(getattr(lp.variable, "defid", None) for lp in _loops)

        with self.env.scope():
            if not _loops:
                if all(isinstance(idx, LiteralIR) for idx in clause_indices):
                    idx_tuple = tuple(int(idx.value) for idx in clause_indices)
                else:
                    idx_tuple = None
                if idx_tuple is not None:
                    value = _body.accept(self)
                    if value is not None:
                        if isinstance(value, np.ndarray):
                            if value.ndim == 0:
                                value = value.item()
                            elif value.size == 1:
                                value = value.flatten()[0].item()
                        elif isinstance(value, np.generic):
                            value = value.item()
                        output[idx_tuple] = value
            else:
                _MAX = int(DEFAULT_EINSTEIN_LOOP_MAX)
                _n = [0]
                _set_value = self.env.set_value
                _to_bool = self._to_bool
                for loop_context in execute_lowered_loops(_loops, {}, expr_evaluator):
                    _n[0] += 1
                    if _n[0] > _MAX:
                        raise RuntimeError("Einstein clause loop iterations exceeded limit.")
                    if _bindings:
                        full_context = execute_lowered_bindings(_bindings, loop_context, expr_evaluator)
                    else:
                        full_context = loop_context
                    for defid, val in full_context.items():
                        if defid is not None:
                            _set_value(defid, val, name=_loop_defid_to_name.get(defid))
                    if _guards and not check_lowered_guards(_guards, full_context, lambda e: _to_bool(e.accept(self))):
                        continue
                    try:
                        value = _body.accept(self)
                    except IndexError:
                        continue
                    if _cell_index_spec is not None:
                        _parts = []
                        for _is_lit, _v in _cell_index_spec:
                            if _is_lit:
                                _parts.append(_v)
                            else:
                                _p = full_context.get(_v)
                                if _p is None and _v is not None:
                                    _p = self.env.get_value(_v)
                                if _p is None:
                                    _parts = None
                                    break
                                _parts.append(_p)
                        idx_tuple = tuple(_parts) if _parts is not None and len(_parts) == len(clause_indices) else None
                    else:
                        idx_tuple = cell_index(full_context)
                    if idx_tuple is None:
                        idx_tuple = tuple(full_context.get(d) for d in _loop_defids_tuple)
                    if idx_tuple is None or len(idx_tuple) != output.ndim:
                        continue
                    if isinstance(value, np.ndarray):
                        if value.ndim == 0:
                            value = value.item()
                        elif value.size == 1:
                            value = value.flatten()[0].item()
                    elif isinstance(value, np.generic):
                        value = value.item()
                    if len(idx_tuple) == 1:
                        output[idx_tuple[0]] = value
                    else:
                        output[idx_tuple] = value

        if variable_defid:
            self._clause_set_output(variable_defid, output)
        _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path="scalar")
        return output
