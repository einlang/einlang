"""
Visitor Helper Utilities

Common visitor patterns for expression analysis without isinstance/hasattr.
"""

from typing import Set, Optional, List, Any
from ..ir.nodes import (
    ExpressionIR, IdentifierIR, BinaryOpIR, LiteralIR,
    RectangularAccessIR, FunctionCallIR, IRVisitor,
    is_function_binding, is_einstein_binding
)


class VariableExtractor(IRVisitor[Set[str]]):
    """Extract variable names from expression - visitor pattern"""
    
    def visit_program(self, node) -> Set[str]:
        """Visit program - not used for variable extraction"""
        return set()
    
    def visit_identifier(self, expr: IdentifierIR) -> Set[str]:
        return {expr.name}
    
    def visit_binary_op(self, expr: BinaryOpIR) -> Set[str]:
        vars: Set[str] = set()
        vars.update(expr.left.accept(self))
        vars.update(expr.right.accept(self))
        return vars
    
    # Default: empty set
    def visit_literal(self, node) -> Set[str]:
        return set()
    
    def visit_function_call(self, node) -> Set[str]:
        return set()
    
    def visit_unary_op(self, node) -> Set[str]:
        return set()
    
    def visit_rectangular_access(self, node) -> Set[str]:
        return set()
    
    def visit_jagged_access(self, node) -> Set[str]:
        return set()
    
    def visit_block_expression(self, node) -> Set[str]:
        return set()
    
    def visit_if_expression(self, node) -> Set[str]:
        return set()
    
    def visit_lambda(self, node) -> Set[str]:
        return set()
    
    def visit_range(self, node) -> Set[str]:
        return set()
    
    def visit_array_comprehension(self, node) -> Set[str]:
        return set()
    
    def visit_array_literal(self, node) -> Set[str]:
        return set()
    
    def visit_tuple_expression(self, node) -> Set[str]:
        return set()
    
    def visit_tuple_access(self, node) -> Set[str]:
        return set()
    
    def visit_interpolated_string(self, node) -> Set[str]:
        return set()
    
    def visit_cast_expression(self, node) -> Set[str]:
        return set()
    
    def visit_member_access(self, node) -> Set[str]:
        return set()
    
    def visit_try_expression(self, node) -> Set[str]:
        return set()
    
    def visit_match_expression(self, node) -> Set[str]:
        return set()
    
    def visit_reduction_expression(self, node) -> Set[str]:
        return set()
    
    def visit_where_expression(self, node) -> Set[str]:
        return set()
    
    def visit_pipeline_expression(self, node) -> Set[str]:
        return set()
    
    def visit_builtin_call(self, node) -> Set[str]:
        return set()
    
    def visit_binding(self, node) -> Set[str]:
        return set()
    
    def visit_literal_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_identifier_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_wildcard_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_tuple_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_array_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_rest_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_guard_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_module(self, node) -> Set[str]:
        return set()


class ConstantEvaluator(IRVisitor[Optional[int]]):
    """Evaluate constant expression to integer - visitor pattern"""
    
    def visit_program(self, node) -> Optional[int]:
        """Visit program - not used for constant evaluation"""
        return None
    
    def visit_literal(self, expr: LiteralIR) -> Optional[int]:
        if isinstance(expr.value, int):
            return expr.value
        return None
    
    # Default: None
    def visit_identifier(self, node) -> Optional[int]:
        return None
    
    def visit_binary_op(self, node) -> Optional[int]:
        return None
    
    def visit_function_call(self, node) -> Optional[int]:
        return None
    
    def visit_unary_op(self, node) -> Optional[int]:
        return None
    
    def visit_rectangular_access(self, node) -> Optional[int]:
        return None
    
    def visit_jagged_access(self, node) -> Optional[int]:
        return None
    
    def visit_block_expression(self, node) -> Optional[int]:
        return None
    
    def visit_if_expression(self, node) -> Optional[int]:
        return None
    
    def visit_lambda(self, node) -> Optional[int]:
        return None
    
    def visit_range(self, node) -> Optional[int]:
        return None
    
    def visit_array_comprehension(self, node) -> Optional[int]:
        return None
    
    def visit_array_literal(self, node) -> Optional[int]:
        return None
    
    def visit_tuple_expression(self, node) -> Optional[int]:
        return None
    
    def visit_tuple_access(self, node) -> Optional[int]:
        return None
    
    def visit_interpolated_string(self, node) -> Optional[int]:
        return None
    
    def visit_cast_expression(self, node) -> Optional[int]:
        return None
    
    def visit_member_access(self, node) -> Optional[int]:
        return None
    
    def visit_try_expression(self, node) -> Optional[int]:
        return None
    
    def visit_match_expression(self, node) -> Optional[int]:
        return None
    
    def visit_reduction_expression(self, node) -> Optional[int]:
        return None
    
    def visit_where_expression(self, node) -> Optional[int]:
        return None
    
    def visit_pipeline_expression(self, node) -> Optional[int]:
        return None
    
    def visit_builtin_call(self, node) -> Optional[int]:
        return None
    
    def visit_binding(self, node) -> Optional[int]:
        return None
    
    def visit_literal_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_identifier_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_wildcard_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_tuple_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_array_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_rest_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_guard_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_module(self, node) -> Optional[int]:
        return None


class ArrayAccessCollector(IRVisitor[List[RectangularAccessIR]]):
    """Collect all array accesses from expression - visitor pattern"""
    
    def visit_program(self, node) -> List[RectangularAccessIR]:
        """Visit program - not used for array access collection"""
        return []
    
    def visit_rectangular_access(self, expr: RectangularAccessIR) -> List[RectangularAccessIR]:
        accesses = [expr]
        accesses.extend(expr.array.accept(self))
        for idx in (getattr(expr, 'indices', None) or []):
            if idx is not None and hasattr(idx, 'accept'):
                accesses.extend(idx.accept(self))
        return accesses
    
    def visit_binary_op(self, expr: BinaryOpIR) -> List[RectangularAccessIR]:
        accesses = []
        accesses.extend(expr.left.accept(self))
        accesses.extend(expr.right.accept(self))
        return accesses
    
    def visit_function_call(self, expr: FunctionCallIR) -> List[RectangularAccessIR]:
        accesses = []
        for arg in expr.arguments:
            accesses.extend(arg.accept(self))
        return accesses
    
    # Default: empty list
    def visit_literal(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_identifier(self, node) -> List[RectangularAccessIR]:
        return []

    def visit_index_var(self, node) -> List[RectangularAccessIR]:
        return []

    def visit_index_rest(self, node) -> List[RectangularAccessIR]:
        return []

    def visit_unary_op(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_jagged_access(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_block_expression(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_if_expression(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_lambda(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_range(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_array_comprehension(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_array_literal(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_tuple_expression(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_tuple_access(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_interpolated_string(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_cast_expression(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_member_access(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_try_expression(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_match_expression(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_reduction_expression(self, node) -> List[RectangularAccessIR]:
        # CRITICAL FIX: Recurse into reduction body to find array accesses
        # This is needed for windowed operations like max[di in 0..2](image[i+di])
        if hasattr(node, 'body') and node.body:
            return node.body.accept(self)
        return []
    
    def visit_where_expression(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_pipeline_expression(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_builtin_call(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_binding(self, node) -> List[RectangularAccessIR]:
        if not (is_function_binding(node) or is_einstein_binding(node)):
            if hasattr(node, 'value') and node.value:
                return node.value.accept(self)
        return []
    
    def visit_literal_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_identifier_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_wildcard_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_tuple_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_array_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_rest_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_guard_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_module(self, node) -> List[RectangularAccessIR]:
        return []

