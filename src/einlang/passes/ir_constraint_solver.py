"""
IR Constraint Solver for Einstein Notation Range Inference 

Pure IR-based constraint solving for deserialization IR nodes.

Core Constraint: index_expr < dimension_bound
Solve for: target_var range

Given:
  - Constraint: An index expression must be < dimension_bound
  - Example: i * 2 < 4 (accessing image[i*2+di] where dim size is 4)

Compute:
  - Range for target_var using inverse reasoning
  - Example: i * 2 < 4 → i < 4 / 2 → i in [0, 2)


to IR representation (works on IR nodes).
"""

from __future__ import annotations
from typing import Optional
from contextlib import contextmanager
import logging

from ..ir.nodes import ExpressionIR, IdentifierIR, BinaryOpIR, LiteralIR, IRVisitor, is_function_binding, is_einstein_binding
from ..shared.types import BinaryOp, PrimitiveType
from .range_info import RangeInfo, DynamicRange

logger = logging.getLogger(__name__)


def make_literal(value: int, location=None) -> LiteralIR:
    """Create a typed literal IR node"""
    from ..shared.source_location import SourceLocation
    if location is None:
        location = SourceLocation('<generated>', 0, 0, 0, 0)
    return LiteralIR(value=value, location=location, type_info=PrimitiveType(name='i32'))


class IRConstraintSolver(IRVisitor):
    """
    Solves range constraints on IR nodes using inverse reasoning.
    
    Constraint form: index_expr < dimension_bound
    Solve for: target_var range
    
    Works directly with IR nodes via pattern matching:
    - IdentifierIR(target_var) < bound → target_var in [0, bound)
    - (target_var / k) < bound → target_var in [0, bound * k)
    - (target_var * k) < bound → target_var in [0, ceil(bound / k))
    - (target_var + k) < bound → target_var in [0, bound - k)
    - (target_var - k) < bound → target_var in [0, bound + k)
    
    Example:
        Constraint: i * 2 < 4
        Solve for: i
        Result: i in [0, 2)
    """
    
    def __init__(self, target_var: str, dimension_bound: ExpressionIR):
        """
        Args:
            target_var: Variable to solve for
            dimension_bound: Upper bound constraint (dimension size)
        """
        self.target_var = target_var
        self.dimension_bound = self._ensure_typed(dimension_bound)
        self.result = None
    
    def _is_double_variable_addition(self, node: BinaryOpIR) -> bool:
        """Check if expression is 'var + var' with same variable."""
        return (node.operator == BinaryOp.ADD and
                isinstance(node.left, IdentifierIR) and 
                isinstance(node.right, IdentifierIR) and
                node.left.name == self.target_var and
                node.right.name == self.target_var)
    
    @contextmanager
    def scoped_bound(self, new_bound: ExpressionIR):
        """Context manager for temporarily adjusting dimension bound"""
        saved_bound = self.dimension_bound
        self.dimension_bound = new_bound
        try:
            yield
        finally:
            self.dimension_bound = saved_bound
    
    def visit_identifier(self, node: IdentifierIR) -> Optional[RangeInfo]:
        """Direct indexing: i → [0, bound)"""
        if node.name == self.target_var:
            logger.debug(f"[IndexRange] {self.target_var} in [0, {self.dimension_bound})")
            return DynamicRange(start=make_literal(0, node.location), end=self.dimension_bound)
        return None
    
    def visit_binary_op(self, node: BinaryOpIR) -> Optional[RangeInfo]:
        """Handle arithmetic operations and nested expressions"""
        
        left_has_target = self._contains_target(node.left)
        right_has_target = self._contains_target(node.right)
        
        # Addition: adjust bound based on constant terms
        if node.operator == BinaryOp.ADD:
            # (i + k) < bound or (k + i) < bound → i < bound - k
            # Caller (implicit range detector) passes raw shape; then to_output_range
            # (maximize_over_reduction_vars) substitutes k with max(k) for intersection.
            if left_has_target and not right_has_target:
                # i + k < bound → i < bound - k
                adjusted_bound = self._build_sub(self.dimension_bound, node.right)
                logger.debug(f"[IndexRange] Adjusting for addition: bound - {node.right}")
                with self.scoped_bound(adjusted_bound):
                    return node.left.accept(self)
            elif right_has_target and not left_has_target:
                # k + i < bound → i < bound - k
                adjusted_bound = self._build_sub(self.dimension_bound, node.left)
                logger.debug(f"[IndexRange] Adjusting for addition: bound - {node.left}")
                with self.scoped_bound(adjusted_bound):
                    return node.right.accept(self)
            elif left_has_target and right_has_target:
                # Special case: i + i → 2*i, so i + i < bound → i < bound/2
                if self._is_double_variable_addition(node):
                    # i + i < bound → 2*i < bound → i < bound/2
                    end = self._build_ceil_div(self.dimension_bound, make_literal(2, node.location))
                    logger.debug(f"[IndexRange] {self.target_var} + {self.target_var} → [0, ceil({self.dimension_bound} / 2))")
                    return DynamicRange(start=make_literal(0, node.location), end=end)
                # Can't isolate in general case
                return None
            else:
                return None
        
        # Subtraction
        elif node.operator == BinaryOp.SUB:
            # (i - k) < bound → i < bound + k
            # (k - i) < bound → i > k - bound (not handled - complex)
            if left_has_target and not right_has_target:
                # i - k < bound → i < bound + k
                adjusted_bound = self._build_add(self.dimension_bound, node.right)
                logger.debug(f"[IndexRange] Adjusting for subtraction: bound + {node.right}")
                with self.scoped_bound(adjusted_bound):
                    return node.left.accept(self)
            elif right_has_target and not left_has_target:
                # k - i < bound → -i < bound - k → i > k - bound
                # This requires a lower bound, not upper bound - not supported
                logger.debug("[IndexRange] Cannot solve (k - i) < bound for i (requires lower bound)")
                return None
            else:
                return None
        
        # Division: i / k → [0, bound * k)
        elif node.operator == BinaryOp.DIV:
            if isinstance(node.left, IdentifierIR) and node.left.name == self.target_var:
                if not right_has_target:  # k must be constant wrt target
                    k = self._ensure_typed(node.right)
                    end = self._build_mul(self.dimension_bound, k)
                    logger.debug(f"[IndexRange] {self.target_var} / {k} → [0, {self.dimension_bound} * {k})")
                    return DynamicRange(start=make_literal(0, node.location), end=end)
        
        # Multiplication: i * k → [0, ceil(bound / k))
        elif node.operator == BinaryOp.MUL:
            target_node = None
            constant_node = None
            
            if isinstance(node.left, IdentifierIR) and node.left.name == self.target_var:
                target_node, constant_node = node.left, node.right
            elif isinstance(node.right, IdentifierIR) and node.right.name == self.target_var:
                target_node, constant_node = node.right, node.left
            
            if target_node and not self._contains_target(constant_node):
                k = self._ensure_typed(constant_node)
                end = self._build_ceil_div(self.dimension_bound, k)
                logger.debug(f"[IndexRange] {self.target_var} * {k} → [0, ceil({self.dimension_bound} / {k}))")
                return DynamicRange(start=make_literal(0, node.location), end=end)
            
            # Nested case: (i % r) * coeff
            if left_has_target and not right_has_target:
                return node.left.accept(self)
            elif right_has_target and not left_has_target:
                return node.right.accept(self)
        
        # Modulo: i % k → no range constraint
        elif node.operator == BinaryOp.MOD:
            if isinstance(node.left, IdentifierIR) and node.left.name == self.target_var:
                logger.debug(f"[IndexRange] {self.target_var} % k → modulo doesn't constrain range")
                return None
        
        return None
    
    def visit_literal(self, node: LiteralIR) -> Optional[RangeInfo]:
        return None
    
    # Stub implementations for other abstract visitor methods (not used in constraint solving)
    def visit_array_comprehension(self, node) -> Optional[RangeInfo]:
        return None
    def visit_array_literal(self, node) -> Optional[RangeInfo]:
        return None
    def visit_array_pattern(self, node) -> Optional[RangeInfo]:
        return None
    def visit_arrow_expression(self, node) -> Optional[RangeInfo]:
        return None
    def visit_block_expression(self, node) -> Optional[RangeInfo]:
        return None
    def visit_builtin_call(self, node) -> Optional[RangeInfo]:
        return None
    def visit_cast_expression(self, node) -> Optional[RangeInfo]:
        # Unwrap cast
        if hasattr(node, 'expr'):
            return node.expr.accept(self)
        return None
    def visit_binding(self, node) -> Optional[RangeInfo]:
        return None
    def visit_function_call(self, node) -> Optional[RangeInfo]:
        return None
    def visit_guard_pattern(self, node) -> Optional[RangeInfo]:
        return None
    def visit_identifier_pattern(self, node) -> Optional[RangeInfo]:
        return None
    def visit_if_expression(self, node) -> Optional[RangeInfo]:
        return None
    def visit_interpolated_string(self, node) -> Optional[RangeInfo]:
        return None
    def visit_jagged_access(self, node) -> Optional[RangeInfo]:
        return None
    def visit_lambda(self, node) -> Optional[RangeInfo]:
        return None
    def visit_literal_pattern(self, node) -> Optional[RangeInfo]:
        return None
    def visit_match_expression(self, node) -> Optional[RangeInfo]:
        return None
    def visit_member_access(self, node) -> Optional[RangeInfo]:
        return None
    def visit_module(self, node) -> Optional[RangeInfo]:
        return None
    def visit_pipeline_expression(self, node) -> Optional[RangeInfo]:
        return None
    def visit_program(self, node) -> Optional[RangeInfo]:
        return None
    def visit_range(self, node) -> Optional[RangeInfo]:
        return None
    def visit_rectangular_access(self, node) -> Optional[RangeInfo]:
        return None
    def visit_reduction_expression(self, node) -> Optional[RangeInfo]:
        return None
    def visit_rest_pattern(self, node) -> Optional[RangeInfo]:
        return None
    def visit_try_expression(self, node) -> Optional[RangeInfo]:
        return None
    def visit_tuple_access(self, node) -> Optional[RangeInfo]:
        return None
    def visit_tuple_expression(self, node) -> Optional[RangeInfo]:
        return None
    def visit_tuple_pattern(self, node) -> Optional[RangeInfo]:
        return None
    def visit_unary_op(self, node) -> Optional[RangeInfo]:
        # Could handle unary minus: -i < bound → i > -bound
        # But not needed for current use case
        return None
    def visit_where_expression(self, node) -> Optional[RangeInfo]:
        return None
    def visit_wildcard_pattern(self, node) -> Optional[RangeInfo]:
        return None
    
    # Helper methods for IR manipulation
    
    def _contains_target(self, expr: ExpressionIR) -> bool:
        """Check if expression contains target variable"""
        if isinstance(expr, IdentifierIR):
            return expr.name == self.target_var
        elif isinstance(expr, BinaryOpIR):
            return self._contains_target(expr.left) or self._contains_target(expr.right)
        return False
    
    def _ensure_typed(self, expr: ExpressionIR) -> ExpressionIR:
        """Ensure IR node has type annotation"""
        if not hasattr(expr, 'type_info') or expr.type_info is None:
            expr.type_info = PrimitiveType(name='i32')
        return expr
    
    def _build_mul(self, a: ExpressionIR, b: ExpressionIR) -> BinaryOpIR:
        """Build IR for: a * b"""
        result = BinaryOpIR(operator=BinaryOp.MUL, left=a, right=b, location=a.location)
        result.type_info = PrimitiveType(name='i32')
        return result
    
    def _build_add(self, a: ExpressionIR, b: ExpressionIR) -> BinaryOpIR:
        """Build IR for: a + b"""
        result = BinaryOpIR(operator=BinaryOp.ADD, left=a, right=b, location=a.location)
        result.type_info = PrimitiveType(name='i32')
        return result
    
    def _build_sub(self, a: ExpressionIR, b: ExpressionIR) -> BinaryOpIR:
        """Build IR for: a - b"""
        result = BinaryOpIR(operator=BinaryOp.SUB, left=a, right=b, location=a.location)
        result.type_info = PrimitiveType(name='i32')
        return result
    
    def _build_div(self, a: ExpressionIR, b: ExpressionIR) -> BinaryOpIR:
        """Build IR for: a / b"""
        result = BinaryOpIR(operator=BinaryOp.DIV, left=a, right=b, location=a.location)
        result.type_info = PrimitiveType(name='i32')
        return result
    
    def _build_ceil_div(self, numerator: ExpressionIR, divisor: ExpressionIR) -> BinaryOpIR:
        """
        Build IR for ceiling division: ceil(numerator / divisor)
        
        Implementation: (numerator + divisor - 1) / divisor
        """
        # divisor - 1
        divisor_minus_one = self._build_sub(divisor, make_literal(1, numerator.location))
        
        # numerator + (divisor - 1)
        adjusted_numerator = self._build_add(numerator, divisor_minus_one)
        
        # (numerator + divisor - 1) / divisor
        return self._build_div(adjusted_numerator, divisor)


def _collect_defids(expr) -> set:
    """Collect all DefIds referenced by IdentifierIR/IndexVarIR nodes in an expression."""
    from ..ir.nodes import IndexVarIR, RectangularAccessIR, MemberAccessIR
    out = set()
    if expr is None:
        return out
    if isinstance(expr, (IdentifierIR, IndexVarIR)):
        d = getattr(expr, 'defid', None)
        if d is not None:
            out.add(d)
        return out
    if isinstance(expr, BinaryOpIR):
        out |= _collect_defids(expr.left)
        out |= _collect_defids(expr.right)
        return out
    if hasattr(expr, 'operand'):
        out |= _collect_defids(getattr(expr, 'operand'))
    if isinstance(expr, RectangularAccessIR):
        out |= _collect_defids(expr.array)
        for idx in (expr.indices or []):
            out |= _collect_defids(idx)
    if isinstance(expr, MemberAccessIR):
        out |= _collect_defids(getattr(expr, 'object', None))
    return out


def solve_index_constraint(
    index_expr: ExpressionIR,
    target_var: str,
    dimension_bound: ExpressionIR,
) -> Optional[RangeInfo]:
    """
    Solve constraint: index_expr < dimension_bound for target_var
    
    Args:
        index_expr: Expression containing target_var (e.g., i*2+di)
        target_var: Variable to solve for (e.g., 'i')
        dimension_bound: Upper bound (e.g., image.shape[0])
        
    Returns:
        RangeInfo for target_var, or None if cannot solve
    """
    solver = IRConstraintSolver(target_var, dimension_bound)
    return index_expr.accept(solver)
