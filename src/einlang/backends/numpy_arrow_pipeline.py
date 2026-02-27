"""Arrow and pipeline application: apply component/right to left value via env-backed backend."""

from typing import Any

from ..ir.nodes import IRVisitor, FunctionCallIR, LiteralIR


def apply_arrow_component(comp: Any, result: Any, location: Any, backend: Any) -> Any:
    applier = ArrowComponentApplier(result, location, backend)
    return comp.accept(applier)


def apply_pipeline_right(right_expr: Any, left_value: Any, location: Any, backend: Any) -> Any:
    applier = PipelineApplier(left_value, location, backend)
    return right_expr.accept(applier)


def _delegate(node: Any, backend: Any) -> Any:
    return node.accept(backend)


class ArrowComponentApplier(IRVisitor[Any]):
    def __init__(self, result: Any, location: Any, backend: Any):
        self.result = result
        self.location = location
        self.backend = backend

    def visit_function_call(self, expr: FunctionCallIR) -> Any:
        new_call = FunctionCallIR(
            callee_expr=expr.callee_expr,
            location=self.location,
            arguments=[LiteralIR(value=self.result, location=self.location)] + expr.arguments,
        )
        return new_call.accept(self.backend)

    def visit_literal(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_identifier(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_index_var(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_binary_op(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_unary_op(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_rectangular_access(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_jagged_access(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_block_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_if_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_lambda(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_range(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_array_comprehension(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_lowered_comprehension(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_array_literal(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_tuple_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_tuple_access(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_interpolated_string(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_cast_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_member_access(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_try_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_match_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_reduction_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_lowered_reduction(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_where_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_arrow_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_pipeline_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_builtin_call(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_einstein_declaration(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_variable_declaration(self, node: Any) -> Any:
        if hasattr(node, "value") and node.value:
            return node.value.accept(self)
        return None
    def visit_literal_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_identifier_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_wildcard_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_tuple_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_array_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_rest_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_guard_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_function_def(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_constant_def(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_module(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_program(self, node: Any) -> Any:
        return _delegate(node, self.backend)


class PipelineApplier(IRVisitor[Any]):
    def __init__(self, left_value: Any, location: Any, backend: Any):
        self.left_value = left_value
        self.location = location
        self.backend = backend

    def visit_program(self, node: Any) -> Any:
        return None

    def visit_function_call(self, expr: FunctionCallIR) -> Any:
        new_call = FunctionCallIR(
            callee_expr=expr.callee_expr,
            location=self.location,
            arguments=[LiteralIR(value=self.left_value, location=self.location)] + expr.arguments,
        )
        return new_call.accept(self.backend)

    def visit_literal(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_identifier(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_index_var(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_binary_op(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_unary_op(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_rectangular_access(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_jagged_access(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_block_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_if_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_lambda(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_range(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_array_comprehension(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_lowered_comprehension(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_array_literal(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_tuple_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_tuple_access(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_interpolated_string(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_cast_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_member_access(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_try_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_match_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_reduction_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_lowered_reduction(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_where_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_arrow_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_pipeline_expression(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_builtin_call(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_einstein_declaration(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_variable_declaration(self, node: Any) -> Any:
        if hasattr(node, "value") and node.value:
            return node.value.accept(self)
        return None
    def visit_literal_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_identifier_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_wildcard_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_tuple_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_array_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_rest_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_guard_pattern(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_function_def(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_constant_def(self, node: Any) -> Any:
        return _delegate(node, self.backend)
    def visit_module(self, node: Any) -> Any:
        return _delegate(node, self.backend)
