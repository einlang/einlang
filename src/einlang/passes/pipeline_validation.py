"""
Pipeline Type Validation Pass

Rust Pattern: N/A (Einlang-specific feature)
Reference: PASS_SYSTEM_DESIGN.md

Design Pattern: Visitor pattern for IR traversal (no isinstance/hasattr)

Validates type compatibility in pipeline expressions at compile-time.
"""

import logging
from typing import Optional

from ..passes.base import BasePass, TyCtxt
from ..passes.type_inference import TypeInferencePass
from ..ir.nodes import (
    ProgramIR, PipelineExpressionIR, IRVisitor, IRNode,
    BindingIR, BlockExpressionIR, IfExpressionIR,
    is_function_binding, is_einstein_binding, is_constant_binding
)
from ..shared.types import Type, FunctionType, UNKNOWN
from ..shared.source_location import SourceLocation

logger = logging.getLogger("einlang.passes.pipeline_validation")

class PipelineTypeValidationPass(BasePass):
    """
    Validates type compatibility in pipeline expressions at compile-time.
    
    Rust Pattern: N/A (Einlang-specific feature)
    
    This pass checks:
    - Standard pipeline (|>): Output type of step N matches input type of step N+1
    - Option pipeline (?>): Function accepts Option type, returns Option type
    - Result pipeline (!>): Function accepts Result type, returns Result type
    """
    requires = [TypeInferencePass]  # Depends on type inference (needs type information)
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """
        Validate pipeline expressions.
        
        Rust Pattern: Analysis pass stores results in TyCtxt
        """
        logger.debug("Starting pipeline type validation")
        
        
        visitor = PipelineTypeValidator(tcx)
        
        # Visit all functions
        for func in ir.functions:
            func.body.accept(visitor)
        
        # Visit all constants
        for const in ir.constants:
            const.value.accept(visitor)
        
        # Visit top-level statements
        for stmt in ir.statements:
            if hasattr(stmt, 'accept'):
                stmt.accept(visitor)
        
        logger.debug("Pipeline type validation complete")
        return ir

class PipelineTypeValidator(IRVisitor[None]):
    """
    Visitor to validate pipeline expressions.
    
    Rust Pattern: rustc_hir::intravisit::Visitor
    """
    
    def __init__(self, tcx: TyCtxt):
        self.tcx = tcx
    
    def visit_program(self, node: ProgramIR) -> None:
        """Visit program and validate all pipeline expressions"""
        # Visit all statements
        for stmt in node.statements:
            stmt.accept(self)
        # Visit all functions
        for func in node.functions:
            func.accept(self)
        # Visit all constants
        for const in node.constants:
            const.accept(self)
    
    def visit_pipeline_expression(self, node: PipelineExpressionIR) -> None:
        """Validate pipeline expression"""
        # Get left side type
        left_type = node.left.type_info if hasattr(node.left, 'type_info') else None
        
        # Get right side type (should be a function)
        right_type = node.right.type_info if hasattr(node.right, 'type_info') else None
        
        logger.debug(f"Pipeline validation: left_type={left_type}, right_type={right_type}, operator={node.operator}")
        
        # Skip validation if types are unknown (inference may not have run yet)
        if left_type is None or right_type is None or left_type == UNKNOWN or right_type == UNKNOWN:
            logger.debug(f"Pipeline validation skipped: types unknown for {node.location}")
            # Still visit nested expressions
            if node.left:
                node.left.accept(self)
            if node.right:
                node.right.accept(self)
            return
        
        # Validate based on operator
        if node.operator == "|>":
            # Standard pipeline: x |> f
            # f must accept x's type, and return type becomes new left type
            self._validate_standard_pipeline(node, left_type, right_type)
        elif node.operator == "?>":
            # Option pipeline: x ?> f
            # x must be Option<T>, f must accept T and return Option<U>
            self._validate_option_pipeline(node, left_type, right_type)
        elif node.operator == "!>":
            # Result pipeline: x !> f
            # x must be Result<T, E>, f must accept T and return Result<U, E>
            self._validate_result_pipeline(node, left_type, right_type)
        
        # Visit nested expressions
        if node.left:
            node.left.accept(self)
        if node.right:
            node.right.accept(self)
    
    def _validate_standard_pipeline(
        self, 
        node: PipelineExpressionIR, 
        left_type: Type, 
        right_type: Type
    ) -> None:
        """Validate standard pipeline (|>)"""
        # Right side should be a function
        if not isinstance(right_type, FunctionType):
            # Right side is not a function type - this is handled by other passes
            return
        
        # Get function input and output types
        func_input_type = right_type.param_types[0] if right_type.param_types else None
        func_output_type = right_type.return_type
        
        if func_input_type is None or func_input_type == UNKNOWN:
            logger.debug(f"Standard pipeline validation skipped: function input type is UNKNOWN")
            return
        
        # Check if left type is compatible with function input type
        if not self._types_compatible(left_type, func_input_type):
            self.tcx.reporter.report_error(
                f"type mismatch in pipeline: cannot pass `{left_type}` to function expecting `{func_input_type}`",
                node.location
            )
    
    def _validate_option_pipeline(
        self, 
        node: PipelineExpressionIR, 
        left_type: Type, 
        right_type: Type
    ) -> None:
        """Validate option pipeline (?>)"""
        # Left side must be Option<T>
        # TODO: Check if left_type is OptionType (when OptionType is implemented)
        # For now, basic validation
        if not isinstance(right_type, FunctionType):
            return
        
        func_input_type = right_type.param_types[0] if right_type.param_types else None
        func_output_type = right_type.return_type
        
        if func_input_type is None or func_input_type == UNKNOWN:
            return
        
        # TODO: Validate Option type compatibility when OptionType is implemented
        logger.debug(f"Option pipeline validation: left_type={left_type}, func_input={func_input_type}, func_output={func_output_type}")
    
    def _validate_result_pipeline(
        self, 
        node: PipelineExpressionIR, 
        left_type: Type, 
        right_type: Type
    ) -> None:
        """Validate result pipeline (!>)"""
        # Note: ResultType not yet in type system, so we do basic validation
        if not isinstance(right_type, FunctionType):
            return
        
        func_input_type = right_type.param_types[0] if right_type.param_types else None
        func_output_type = right_type.return_type
        
        # Basic validation: function should accept some type
        logger.debug(f"Result pipeline validation: left_type={left_type}, func_input={func_input_type}, func_output={func_output_type}")
    
    def _types_compatible(self, source_type: Type, target_type: Type) -> bool:
        """
        Check if source type is compatible with target type.
        
        Rules:
        - Same types are compatible
        - Unknown types are compatible (runtime check needed)
        - Widening is allowed (i32 -> i64, f32 -> f64)
        """
        from ..shared.types import PrimitiveType
        
        # Handle UNKNOWN - allow (runtime check needed)
        if source_type == UNKNOWN or target_type == UNKNOWN:
            return True
        
        # Same types are compatible
        if source_type == target_type:
            return True
        
        # Widening rules (i32 -> i64, f32 -> f64)
        if isinstance(source_type, PrimitiveType) and isinstance(target_type, PrimitiveType):
            if source_type.name == "i32" and target_type.name == "i64":
                return True
            if source_type.name == "f32" and target_type.name == "f64":
                return True
        
        # Other combinations not compatible
        return False
    
    # Visitor methods for traversing IR
    def visit_binding(self, node: BindingIR) -> None:
        """Visit bindings"""
        if is_function_binding(node):
            if node.body:
                node.body.accept(self)
        elif is_einstein_binding(node):
            for clause in getattr(node, 'clauses', []) or []:
                clause.accept(self)
        else:
            if hasattr(node, 'value') and node.value:
                node.value.accept(self)
    
    def visit_block_expression(self, node: BlockExpressionIR) -> None:
        """Visit block expressions"""
        if hasattr(node, 'statements'):
            for stmt in node.statements:
                if hasattr(stmt, 'accept'):
                    stmt.accept(self)
        if hasattr(node, 'final_expr') and node.final_expr:
            node.final_expr.accept(self)
    
    def visit_if_expression(self, node: IfExpressionIR) -> None:
        """Visit if expressions"""
        if node.condition:
            node.condition.accept(self)
        if node.then_expr:
            node.then_expr.accept(self)
        if node.else_expr:
            node.else_expr.accept(self)
    
    # Default implementations for other nodes
    def visit_literal(self, node) -> None:
        pass
    
    def visit_identifier(self, node) -> None:
        pass
    
    def visit_binary_op(self, node) -> None:
        if node.left:
            node.left.accept(self)
        if node.right:
            node.right.accept(self)
    
    def visit_unary_op(self, node) -> None:
        if node.operand:
            node.operand.accept(self)
    
    def visit_function_call(self, node) -> None:
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_rectangular_access(self, node) -> None:
        if node.array:
            node.array.accept(self)
        for idx in (node.indices or []):
            if idx is not None and hasattr(idx, 'accept'):
                idx.accept(self)
    
    def visit_jagged_access(self, node) -> None:
        if node.base:
            node.base.accept(self)
        for idx in (node.index_chain or []):
            if idx is not None and hasattr(idx, 'accept'):
                idx.accept(self)
    
    def visit_array_literal(self, node) -> None:
        for elem in node.elements:
            elem.accept(self)
    
    def visit_tuple_expression(self, node) -> None:
        for elem in node.elements:
            elem.accept(self)
    
    def visit_lambda(self, node) -> None:
        if node.body:
            node.body.accept(self)
    
    def visit_match_expression(self, node) -> None:
        if node.scrutinee:
            node.scrutinee.accept(self)
        for arm in node.arms:
            if arm.pattern:
                arm.pattern.accept(self)
            if arm.body:
                arm.body.accept(self)
    
    def visit_where_expression(self, node) -> None:
        if node.expr:
            node.expr.accept(self)
        for constraint in node.constraints:
            constraint.accept(self)
    
    # Add stub implementations for all other IR node types
    def visit_range(self, node) -> None:
        pass
    
    def visit_array_comprehension(self, node) -> None:
        pass
    
    def visit_tuple_access(self, node) -> None:
        pass
    
    def visit_interpolated_string(self, node) -> None:
        pass
    
    def visit_cast_expression(self, node) -> None:
        if node.expr:
            node.expr.accept(self)
    
    def visit_member_access(self, node) -> None:
        if node.object:
            node.object.accept(self)
    
    def visit_try_expression(self, node) -> None:
        if node.expr:
            node.expr.accept(self)
    
    def visit_reduction_expression(self, node) -> None:
        if node.body:
            node.body.accept(self)
        if node.where_clause:
            for constraint in node.where_clause.constraints:
                constraint.accept(self)
    
    def visit_builtin_call(self, node) -> None:
        for arg in node.args:
            arg.accept(self)
    
    def visit_module(self, node) -> None:
        pass
    
    # Pattern visitors (no-op)
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


