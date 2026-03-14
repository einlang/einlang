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

from ..ir.nodes import (
    ExpressionIR, IdentifierIR, IndexVarIR, BinaryOpIR, LiteralIR, UnaryOpIR,
    RectangularAccessIR, MemberAccessIR, IRVisitor,
    is_function_binding, is_einstein_binding,
)
from ..shared.defid import DefId
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
    - (target_var - k) < bound → target_var in [k, bound + k)
    
    Example:
        Constraint: i * 2 < 4
        Solve for: i
        Result: i in [0, 2)
    """
    
    def __init__(self, target_var: Optional[str], dimension_bound: ExpressionIR,
                 target_defid: Optional[DefId] = None):
        """
        Args:
            target_var: Variable name to solve for (used as fallback if target_defid is None)
            dimension_bound: Upper bound constraint (dimension size)
            target_defid: DefId of target variable (preferred over name matching)
        """
        self.target_var = target_var
        self.target_defid = target_defid
        self.dimension_bound = self._ensure_typed(dimension_bound)
        self.result = None

    def _is_target(self, node: ExpressionIR) -> bool:
        """Check if a leaf node IS the target variable (by DefId first, name fallback)."""
        if not isinstance(node, (IdentifierIR, IndexVarIR)):
            return False
        if self.target_defid is not None:
            return node.defid == self.target_defid
        return self.target_var is not None and node.name == self.target_var
    
    def _is_double_variable_addition(self, node: BinaryOpIR) -> bool:
        """Check if expression is 'var + var' with same variable."""
        return (node.operator == BinaryOp.ADD and
                self._is_target(node.left) and
                self._is_target(node.right))
    
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
        if self._is_target(node):
            logger.debug(f"[IndexRange] {node.name} in [0, {self.dimension_bound})")
            return DynamicRange(start=make_literal(0, node.location), end=self.dimension_bound)
        return None

    def visit_index_var(self, node: IndexVarIR) -> Optional[RangeInfo]:
        """Einstein index variable: same as identifier for range solving purposes."""
        if self._is_target(node):
            logger.debug(f"[IndexRange] {node.name} in [0, {self.dimension_bound})")
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
            if left_has_target and not right_has_target:
                # (expr - k) in [0, bound): expr in [k, bound + k)
                # Upper bound: expr < bound + k
                k = self._ensure_typed(node.right)
                adjusted_bound = self._build_add(self.dimension_bound, k)
                logger.debug(f"[IndexRange] Adjusting for subtraction: bound + {k}, lower = f({k})")
                with self.scoped_bound(adjusted_bound):
                    upper_result = node.left.accept(self)
                if upper_result is None:
                    return None
                # Lower bound: expr >= k  →  solve "expr < k" and take .end as start
                with self.scoped_bound(k):
                    lower_result = node.left.accept(self)
                if isinstance(upper_result, DynamicRange):
                    start = (lower_result.end
                             if lower_result is not None and isinstance(lower_result, DynamicRange)
                             else k)
                    return DynamicRange(start=start, end=upper_result.end)
                return upper_result
            elif right_has_target and not left_has_target:
                # k - i < bound → -i < bound - k → i > k - bound
                # This requires a lower bound, not upper bound - not supported
                logger.debug("[IndexRange] Cannot solve (k - i) < bound for i (requires lower bound)")
                return None
            else:
                return None
        
        # Division: i / k → [0, bound * k)
        elif node.operator == BinaryOp.DIV:
            if self._is_target(node.left):
                if not right_has_target:  # k must be constant wrt target
                    k = self._ensure_typed(node.right)
                    end = self._build_mul(self.dimension_bound, k)
                    logger.debug(f"[IndexRange] {node.left.name} / {k} → [0, {self.dimension_bound} * {k})")
                    return DynamicRange(start=make_literal(0, node.location), end=end)
        
        # Multiplication: i * k → [0, ceil(bound / k))
        elif node.operator == BinaryOp.MUL:
            target_node = None
            constant_node = None
            
            if self._is_target(node.left):
                target_node, constant_node = node.left, node.right
            elif self._is_target(node.right):
                target_node, constant_node = node.right, node.left
            
            if target_node and not self._contains_target(constant_node):
                k = self._ensure_typed(constant_node)
                end = self._build_ceil_div(self.dimension_bound, k)
                logger.debug(f"[IndexRange] {target_node.name} * {k} → [0, ceil({self.dimension_bound} / {k}))")
                return DynamicRange(start=make_literal(0, node.location), end=end)
            
            # Nested case: (i % r) * coeff
            if left_has_target and not right_has_target:
                return node.left.accept(self)
            elif right_has_target and not left_has_target:
                return node.right.accept(self)
        
        # Modulo: i % k → no range constraint
        elif node.operator == BinaryOp.MOD:
            if self._is_target(node.left):
                logger.debug(f"[IndexRange] {node.left.name} % k → modulo doesn't constrain range")
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
        if isinstance(expr, (IdentifierIR, IndexVarIR)):
            return self._is_target(expr)
        elif isinstance(expr, BinaryOpIR):
            return self._contains_target(expr.left) or self._contains_target(expr.right)
        return False
    
    def _ensure_typed(self, expr: ExpressionIR) -> ExpressionIR:
        """Ensure IR node has type annotation"""
        if getattr(expr, 'type_info', None) is None:
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


class _DefidsInExprCollector(IRVisitor[None]):
    """Collect DefIds from IdentifierIR/IndexVarIR in an expression tree into _out."""

    def __init__(self, out: set) -> None:
        self._out = out

    def visit_identifier(self, node: IdentifierIR) -> None:
        if node.defid is not None:
            self._out.add(node.defid)

    def visit_index_var(self, node: IndexVarIR) -> None:
        if node.defid is not None:
            self._out.add(node.defid)

    def visit_binary_op(self, node: BinaryOpIR) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_unary_op(self, node: UnaryOpIR) -> None:
        node.operand.accept(self)

    def visit_rectangular_access(self, node: RectangularAccessIR) -> None:
        node.array.accept(self)
        for idx in node.indices or []:
            idx.accept(self)

    def visit_member_access(self, node: MemberAccessIR) -> None:
        node.object.accept(self)

    def visit_try_expression(self, node) -> None:
        if node.operand is not None:
            node.operand.accept(self)

    def visit_literal(self, node) -> None:
        pass

    def visit_index_rest(self, node) -> None:
        pass

    def visit_function_call(self, node) -> None:
        pass

    def visit_jagged_access(self, node) -> None:
        pass

    def visit_block_expression(self, node) -> None:
        pass

    def visit_if_expression(self, node) -> None:
        pass

    def visit_lambda(self, node) -> None:
        pass

    def visit_range(self, node) -> None:
        pass

    def visit_array_comprehension(self, node) -> None:
        pass

    def visit_module(self, node) -> None:
        pass

    def visit_array_literal(self, node) -> None:
        pass

    def visit_tuple_expression(self, node) -> None:
        pass

    def visit_tuple_access(self, node) -> None:
        pass

    def visit_interpolated_string(self, node) -> None:
        pass

    def visit_cast_expression(self, node) -> None:
        pass

    def visit_match_expression(self, node) -> None:
        pass

    def visit_reduction_expression(self, node) -> None:
        pass

    def visit_where_expression(self, node) -> None:
        pass

    def visit_pipeline_expression(self, node) -> None:
        pass

    def visit_builtin_call(self, node) -> None:
        pass

    def visit_literal_pattern(self, node) -> None:
        pass

    def visit_identifier_pattern(self, node) -> None:
        pass

    def visit_wildcard_pattern(self, node) -> None:
        pass

    def visit_tuple_pattern(self, node) -> None:
        pass

    def visit_array_pattern(self, node) -> None:
        pass

    def visit_rest_pattern(self, node) -> None:
        pass

    def visit_guard_pattern(self, node) -> None:
        pass

    def visit_or_pattern(self, node) -> None:
        pass

    def visit_constructor_pattern(self, node) -> None:
        pass

    def visit_binding_pattern(self, node) -> None:
        pass

    def visit_range_pattern(self, node) -> None:
        pass

    def visit_function_value(self, node) -> None:
        pass

    def visit_einstein(self, node) -> None:
        pass

    def visit_einstein_clause(self, node) -> None:
        pass

    def visit_binding(self, node) -> None:
        pass

    def visit_program(self, node) -> None:
        pass

    def visit_lowered_reduction(self, node) -> None:
        pass

    def visit_lowered_comprehension(self, node) -> None:
        pass

    def visit_lowered_einstein_clause(self, node) -> None:
        pass

    def visit_lowered_einstein(self, node) -> None:
        pass

    def visit_lowered_recurrence(self, node) -> None:
        pass


def _collect_defids(expr) -> set:
    """Collect all DefIds referenced by IdentifierIR/IndexVarIR nodes in an expression."""
    out: set = set()
    if expr is None:
        return out
    if expr is not None:
        expr.accept(_DefidsInExprCollector(out))
    return out


def solve_index_constraint(
    index_expr: ExpressionIR,
    target_var: Optional[str],
    dimension_bound: ExpressionIR,
    target_defid: Optional[DefId] = None,
) -> Optional[RangeInfo]:
    """
    Solve constraint: index_expr < dimension_bound for target_var
    
    Args:
        index_expr: Expression containing target_var (e.g., i*2+di)
        target_var: Variable name to solve for (fallback if target_defid is None)
        dimension_bound: Upper bound (e.g., image.shape[0])
        target_defid: DefId of target variable (preferred over name matching)
        
    Returns:
        RangeInfo for target_var, or None if cannot solve
    """
    solver = IRConstraintSolver(target_var, dimension_bound, target_defid=target_defid)
    return index_expr.accept(solver)
