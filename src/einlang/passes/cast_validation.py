"""
Cast Validation Pass

Rust Pattern: Type Cast Validation
Reference: TYPE_ANALYSIS_DESIGN.md
"""

from typing import Optional, Any
from ..passes.base import BasePass, TyCtxt
from ..passes.type_inference import TypeInferencePass
from ..ir.nodes import (
    ProgramIR, ExpressionIR, CastExpressionIR, IRVisitor
)
from ..shared.defid import DefId
from ..shared.types import (
    Type,
    TypeKind,
    I32,
    I64,
    F32,
    F64,
    BOOL,
    STR,
    UNKNOWN,
)

# Structured sets of primitive types for cast rules (object identity / equality)
_NUMERIC_PRIMITIVES = (I32, I64, F32, F64)


def _is_numeric_primitive(ty: Any) -> bool:
    return ty in _NUMERIC_PRIMITIVES


def _type_display(ty: Any) -> str:
    """Display type for error messages; prefer .name for primitives, else str(ty)."""
    if ty is None:
        return "?"
    if getattr(ty, "name", None) is not None:
        return ty.name
    return str(ty)


class CastValidationPass(BasePass):
    """
    Cast validation pass.
    
    Validates type casts to ensure they are safe and valid.
    """
    requires = [TypeInferencePass]  # Depends on type inference (needs type information)
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """Validate casts in IR"""
        validator = CastValidator(tcx)
        
        # Validate casts in all expressions
        visitor = CastValidationVisitor(validator)
        
        # Process all functions
        for func in ir.functions:
            func.body.accept(visitor)
        
        # Process all statements
        for stmt in ir.statements:
            stmt.accept(visitor)
        
        return ir


class CastValidator:
    """Cast validator - validates type casts"""
    
    def __init__(self, tcx: TyCtxt):
        self.tcx = tcx
    
    def _get_expression_type(self, expr: ExpressionIR) -> Optional[Any]:
        """Get type of expression (type object, e.g. PrimitiveType)."""
        if not hasattr(expr, 'type_info') or not expr.type_info:
            return None
        return expr.type_info

    def _get_target_type(self, cast_expr: CastExpressionIR) -> Optional[Any]:
        """Get target type as type object (never None for valid IR)."""
        return getattr(cast_expr, 'target_type', None)

    def validate_cast(self, cast_expr: CastExpressionIR) -> bool:
        """Validate type cast. Source and target are type objects."""
        source_type = self._get_expression_type(cast_expr.expr)
        target_type = self._get_target_type(cast_expr)

        if source_type is None or target_type is None:
            return False

        return self._is_valid_cast(source_type, target_type)

    def _is_valid_cast(self, source_type: Any, target_type: Any) -> bool:
        """Check if cast from source_type to target_type is valid (both type objects)."""
        if source_type == target_type:
            return True

        # Unknown allows anything
        if source_type == UNKNOWN or getattr(source_type, "kind", None) == TypeKind.UNKNOWN:
            return True
        if target_type == UNKNOWN or getattr(target_type, "kind", None) == TypeKind.UNKNOWN:
            return True

        # Array/tensor types: valid if element types are valid cast
        src_elem = getattr(source_type, "element_type", None)
        tgt_elem = getattr(target_type, "element_type", None)
        if src_elem is not None and tgt_elem is not None:
            return self._is_valid_cast(src_elem, tgt_elem)

        # Primitive rules using type objects (no string comparison)
        if getattr(source_type, "kind", None) != TypeKind.PRIMITIVE:
            return False
        if getattr(target_type, "kind", None) != TypeKind.PRIMITIVE:
            return False

        # Numeric <-> numeric
        if _is_numeric_primitive(source_type) and _is_numeric_primitive(target_type):
            return True
        # Bool <-> numeric
        if source_type == BOOL and _is_numeric_primitive(target_type):
            return True
        if _is_numeric_primitive(source_type) and target_type == BOOL:
            return True
        # Same non-numeric primitive (str/str, bool/bool)
        if source_type == STR and target_type == STR:
            return True
        if source_type == BOOL and target_type == BOOL:
            return True
        return False


class CastValidationVisitor(IRVisitor[None]):
    """Visitor to validate casts in IR"""
    
    def __init__(self, validator: CastValidator):
        self.validator = validator
    
    def visit_program(self, node: ProgramIR) -> None:
        """Visit program and validate all casts"""
        # Visit all statements
        for stmt in node.statements:
            stmt.accept(self)
        # Visit all functions
        for func in node.functions:
            func.accept(self)
        # Visit all constants
        for const in node.constants:
            const.accept(self)
    
    def visit_cast_expression(self, expr: CastExpressionIR) -> None:
        """Validate cast expression"""
        is_valid = self.validator.validate_cast(expr)
        
        if not is_valid:
            source_type = self.validator._get_expression_type(expr.expr)
            target_type = self.validator._get_target_type(expr)
            self.validator.tcx.reporter.report_error(
                message=f"Cannot cast {_type_display(source_type)} to {_type_display(target_type)}. Invalid type cast.",
                location=expr.location,
                code="E1003",
                note="Valid casts: numeric types (i32, i64, f32, f64), bool to numeric, numeric to bool, same type.",
            )
        
        # Process operand
        expr.expr.accept(self)
    
    # Required visitor methods (no-op for other nodes)
    def visit_literal(self, node) -> None:
        pass
    
    def visit_identifier(self, node) -> None:
        pass
    
    def visit_binary_op(self, node) -> None:
        pass
    
    def visit_function_call(self, node) -> None:
        pass
    
    def visit_unary_op(self, node) -> None:
        pass
    
    def visit_rectangular_access(self, node) -> None:
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
    
    def visit_array_literal(self, node) -> None:
        pass
    
    def visit_tuple_expression(self, node) -> None:
        pass
    
    def visit_tuple_access(self, node) -> None:
        pass
    
    def visit_interpolated_string(self, node) -> None:
        pass
    
    def visit_member_access(self, node) -> None:
        pass
    
    def visit_try_expression(self, node) -> None:
        pass
    
    def visit_match_expression(self, node) -> None:
        pass
    
    def visit_reduction_expression(self, node) -> None:
        """Visit reduction expression - traverse body and where clause"""
        node.body.accept(self)
        if node.where_clause:
            for constraint in node.where_clause.constraints:
                constraint.accept(self)
    
    def visit_where_expression(self, node) -> None:
        """Visit where expression - traverse expr and constraints"""
        node.expr.accept(self)
        for constraint in node.constraints:
            constraint.accept(self)
    
    def visit_arrow_expression(self, node) -> None:
        pass
    
    def visit_pipeline_expression(self, node) -> None:
        pass
    
    def visit_builtin_call(self, node) -> None:
        pass
    
    def visit_einstein_declaration(self, node) -> None:
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
    
    def visit_function_def(self, node) -> None:
        pass
    
    def visit_constant_def(self, node) -> None:
        pass
    
    def visit_module(self, node) -> None:
        pass

    def visit_variable_declaration(self, node) -> Any:
        """Visit variable declaration - recurse into value"""
        if hasattr(node, 'value') and node.value:
            return node.value.accept(self)
        return None

