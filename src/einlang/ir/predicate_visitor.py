from __future__ import annotations

from typing import Any, Iterable

from .nodes import IRVisitor


class RecursivePredicateVisitor(IRVisitor[bool]):
    """Reusable recursive bool visitor for IR tree predicates.

    Subclasses override :meth:`matches` for the node kinds they care about.
    Default traversal walks the IR tree structurally and returns ``True`` as
    soon as any matching descendant is found.
    """

    def matches(self, node: Any) -> bool:
        return False

    def visit(self, node: Any) -> bool:
        if node is None:
            return False
        accept = getattr(node, "accept", None)
        if callable(accept):
            return bool(accept(self))
        return False

    def visit_any(self, *nodes: Any) -> bool:
        for node in nodes:
            if self.visit(node):
                return True
        return False

    def visit_iter(self, nodes: Iterable[Any]) -> bool:
        for node in nodes:
            if self.visit(node):
                return True
        return False

    def visit_program(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.modules or []) or self.visit_iter(node.statements or [])

    def visit_module(self, node: Any) -> bool:
        return (
            self.matches(node)
            or self.visit_iter(node.constants or [])
            or self.visit_iter(node.functions or [])
            or self.visit_iter(node.submodules or [])
        )

    def visit_diff_rule(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.body)

    def visit_literal(self, node: Any) -> bool:
        return self.matches(node)

    def visit_identifier(self, node: Any) -> bool:
        return self.matches(node)

    def visit_index_var(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.range_ir)

    def visit_index_rest(self, node: Any) -> bool:
        return self.matches(node)

    def visit_binary_op(self, node: Any) -> bool:
        return self.matches(node) or self.visit_any(node.left, node.right)

    def visit_unary_op(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.operand)

    def visit_differential(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.operand)

    def visit_function_call(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.callee_expr) or self.visit_iter(node.arguments or [])

    def visit_builtin_call(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.args or [])

    def visit_parameter(self, node: Any) -> bool:
        return self.matches(node)

    def visit_rectangular_access(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.array) or self.visit_iter(node.indices or [])

    def visit_jagged_access(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.base) or self.visit_iter(node.index_chain or [])

    def visit_block_expression(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.statements or []) or self.visit(node.final_expr)

    def visit_if_expression(self, node: Any) -> bool:
        return self.matches(node) or self.visit_any(node.condition, node.then_expr, node.else_expr)

    def visit_lambda(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.parameters or []) or self.visit(node.body)

    def visit_function_value(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.parameters or []) or self.visit(node.body) or self.visit(node.custom_diff_body)

    def visit_range(self, node: Any) -> bool:
        return self.matches(node) or self.visit_any(node.start, node.end)

    def visit_array_comprehension(self, node: Any) -> bool:
        return (
            self.matches(node)
            or self.visit(node.body)
            or self.visit_iter(node.loop_vars or [])
            or self.visit_iter(node.ranges or [])
        )

    def visit_array_literal(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.elements or [])

    def visit_tuple_expression(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.elements or [])

    def visit_tuple_access(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.tuple_expr)

    def visit_interpolated_string(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(part for part in (node.parts or []) if hasattr(part, "accept"))

    def visit_cast_expression(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.expr)

    def visit_member_access(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.object)

    def visit_try_expression(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.operand)

    def visit_match_expression(self, node: Any) -> bool:
        if self.matches(node) or self.visit(node.scrutinee):
            return True
        for arm in node.arms or []:
            if self.visit(arm.pattern) or self.visit(arm.body):
                return True
        return False

    def visit_reduction_expression(self, node: Any) -> bool:
        return (
            self.matches(node)
            or self.visit_iter(node.loop_vars or [])
            or self.visit(node.body)
            or self.visit(node.where_clause)
        )

    def visit_select_at_argmax(self, node: Any) -> bool:
        return (
            self.matches(node)
            or self.visit(node.primal_body)
            or self.visit(node.diff_body)
            or self.visit_iter(node.loop_vars or [])
        )

    def visit_where_expression(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.expr) or self.visit_iter(node.constraints or [])

    def visit_pipeline_expression(self, node: Any) -> bool:
        return self.matches(node) or self.visit_any(node.left, node.right)

    def visit_literal_pattern(self, node: Any) -> bool:
        return self.matches(node)

    def visit_identifier_pattern(self, node: Any) -> bool:
        return self.matches(node)

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return self.matches(node)

    def visit_tuple_pattern(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.patterns or [])

    def visit_array_pattern(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.patterns or [])

    def visit_rest_pattern(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.pattern)

    def visit_guard_pattern(self, node: Any) -> bool:
        return self.matches(node) or self.visit_any(node.inner_pattern, node.guard_expr)

    def visit_or_pattern(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.alternatives or [])

    def visit_constructor_pattern(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.patterns or [])

    def visit_binding_pattern(self, node: Any) -> bool:
        return self.matches(node) or self.visit_any(node.identifier_pattern, node.inner_pattern)

    def visit_range_pattern(self, node: Any) -> bool:
        return self.matches(node)

    def visit_binding(self, node: Any) -> bool:
        return self.matches(node) or self.visit(node.expr)

    def visit_einstein_clause(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.indices or []) or self.visit(node.value) or self.visit(node.where_clause)

    def visit_einstein(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.clauses or []) or self.visit_iter(node.shape or [])

    def visit_lowered_reduction(self, node: Any) -> bool:
        return (
            self.matches(node)
            or self.visit(node.body)
            or self.visit_iter(node.bindings or [])
            or self.visit_iter(getattr(guard, "condition", None) for guard in (node.guards or []))
            or self.visit_iter(loop.variable for loop in (node.loops or []))
            or self.visit_iter(loop.iterable for loop in (node.loops or []))
        )

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return (
            self.matches(node)
            or self.visit(node.body)
            or self.visit_iter(node.bindings or [])
            or self.visit_iter(getattr(guard, "condition", None) for guard in (node.guards or []))
            or self.visit_iter(loop.variable for loop in (node.loops or []))
            or self.visit_iter(loop.iterable for loop in (node.loops or []))
        )

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return (
            self.matches(node)
            or self.visit(node.body)
            or self.visit_iter(node.bindings or [])
            or self.visit_iter(getattr(guard, "condition", None) for guard in (node.guards or []))
            or self.visit_iter(node.indices or [])
            or self.visit_iter(loop.variable for loop in (node.loops or []))
            or self.visit_iter(loop.iterable for loop in (node.loops or []))
        )

    def visit_lowered_einstein(self, node: Any) -> bool:
        return self.matches(node) or self.visit_iter(node.items or []) or self.visit_iter(node.shape or [])

    def visit_lowered_recurrence(self, node: Any) -> bool:
        return (
            self.matches(node)
            or self.visit(node.initial)
            or self.visit(node.body)
            or self.visit(getattr(node.recurrence_loop, "variable", None))
            or self.visit(getattr(node.recurrence_loop, "iterable", None))
        )

    def visit_lowered_select_at_argmax(self, node: Any) -> bool:
        return (
            self.matches(node)
            or self.visit(node.primal_body)
            or self.visit(node.diff_body)
            or self.visit_iter(node.bindings or [])
            or self.visit_iter(getattr(guard, "condition", None) for guard in (node.guards or []))
            or self.visit_iter(loop.variable for loop in (node.loops or []))
            or self.visit_iter(loop.iterable for loop in (node.loops or []))
        )
