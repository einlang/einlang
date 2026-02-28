"""
Exhaustiveness Checking Pass

Rust Pattern: Rust Exhaustiveness Checking
Reference: PATTERN_MATCHING_DESIGN.md
"""

from typing import List, Set, Optional, Any
from ..passes.base import BasePass, TyCtxt
from ..passes.type_inference import TypeInferencePass
from ..ir.nodes import (
    ProgramIR, ExpressionIR, MatchExpressionIR, MatchArmIR,
    PatternIR, LiteralPatternIR, WildcardPatternIR,
    IdentifierPatternIR, TuplePatternIR, ArrayPatternIR,
    OrPatternIR, ConstructorPatternIR, BindingPatternIR, RangePatternIR,
    IRVisitor,
    is_function_binding, is_einstein_binding, is_constant_binding
)
from ..shared.defid import DefId


class ExhaustivenessPass(BasePass):
    """
    Exhaustiveness checking pass.
    
    Checks if match expressions are exhaustive (cover all possible values).
    """
    requires = [TypeInferencePass]  # Depends on type inference (needs type information)
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """Check exhaustiveness in IR"""
        checker = ExhaustivenessChecker(tcx)
        
        # Check exhaustiveness in all expressions
        visitor = ExhaustivenessVisitor(checker)
        
        # Process all functions
        for func in ir.functions:
            func.body.accept(visitor)
        
        # Process all statements
        for stmt in ir.statements:
            stmt.accept(visitor)
        
        return ir


class ExhaustivenessChecker:
    """Exhaustiveness checker - checks if match expressions are exhaustive"""
    
    def __init__(self, tcx: TyCtxt):
        self.tcx = tcx
    
    def check_exhaustiveness(self, match_expr: MatchExpressionIR) -> bool:
        """Check if match expression is exhaustive"""
        # Get scrutinee type
        scrutinee_type = self._get_scrutinee_type(match_expr.scrutinee)
        
        if scrutinee_type is None:
            # Cannot check exhaustiveness without type
            return True  # Assume exhaustive
        
        # Collect patterns
        patterns = [arm.pattern for arm in match_expr.arms]
        
        # Check if patterns cover all cases
        return self._patterns_cover_type(patterns, scrutinee_type)
    
    def _get_scrutinee_type(self, scrutinee: ExpressionIR) -> Optional[str]:
        """Get type of scrutinee expression"""
        # Direct attribute access - trust IR structure
        if hasattr(scrutinee, 'type_info') and scrutinee.type_info:
            type_info = scrutinee.type_info
            
            # Handle PrimitiveType objects (most common case)
            if hasattr(type_info, 'name'):
                return type_info.name
            elif hasattr(type_info, '__name__'):
                return type_info.__name__
            elif isinstance(type_info, str):
                return type_info
        
        return None
    
    def _flatten_patterns(self, patterns: List[PatternIR]) -> List[PatternIR]:
        """Flatten or-patterns into individual alternatives for coverage analysis."""
        flat: List[PatternIR] = []
        for p in patterns:
            if isinstance(p, OrPatternIR):
                flat.extend(self._flatten_patterns(p.alternatives))
            else:
                flat.append(p)
        return flat
    
    def _patterns_cover_type(self, patterns: List[PatternIR], type_name: str) -> bool:
        """Check if patterns cover all cases for type"""
        flat = self._flatten_patterns(patterns)
        for pattern in flat:
            if isinstance(pattern, BindingPatternIR):
                if self._is_wildcard(pattern) or self._is_wildcard(pattern.inner_pattern):
                    return True
            elif hasattr(pattern, 'inner_pattern'):  # GuardPatternIR
                if self._is_wildcard(pattern.inner_pattern):
                    return True
            elif self._is_wildcard(pattern):
                return True
        
        if type_name in ('int', 'i32', 'i64'):
            return self._check_integer_exhaustiveness(flat)
        
        if type_name in ('bool', 'boolean'):
            return self._check_boolean_exhaustiveness(flat)
        
        return False
    
    def _is_wildcard(self, pattern: PatternIR) -> bool:
        """Check if pattern is a catch-all (wildcard, identifier, or binding-of-wildcard)."""
        if isinstance(pattern, WildcardPatternIR):
            return True
        if isinstance(pattern, IdentifierPatternIR):
            return True
        if isinstance(pattern, BindingPatternIR):
            return self._is_wildcard(pattern.inner_pattern)
        return False
    
    def _check_integer_exhaustiveness(self, patterns: List[PatternIR]) -> bool:
        """Check if integer patterns are exhaustive"""
        for pattern in patterns:
            if self._is_wildcard(pattern):
                return True
        return False
    
    def _check_boolean_exhaustiveness(self, patterns: List[PatternIR]) -> bool:
        """Check if boolean patterns are exhaustive"""
        has_true = False
        has_false = False
        has_wildcard = False
        
        for pattern in patterns:
            if self._is_wildcard(pattern):
                has_wildcard = True
            elif hasattr(pattern, 'value'):  # LiteralPatternIR
                if pattern.value is True:
                    has_true = True
                elif pattern.value is False:
                    has_false = True
        
        return has_wildcard or (has_true and has_false)
    
    def find_uncovered_cases(self, match_expr: MatchExpressionIR) -> List[str]:
        """Find uncovered cases in match expression"""
        scrutinee_type = self._get_scrutinee_type(match_expr.scrutinee)
        
        if scrutinee_type is None:
            return []
        
        patterns = [arm.pattern for arm in match_expr.arms]
        flat = self._flatten_patterns(patterns)
        
        uncovered = []
        
        if not any(self._is_wildcard(p) for p in flat):
            if scrutinee_type in ('bool', 'boolean'):
                has_true = False
                has_false = False
                for p in flat:
                    if hasattr(p, 'value'):
                        if p.value is True:
                            has_true = True
                        elif p.value is False:
                            has_false = True
                
                if not has_true:
                    uncovered.append("true")
                if not has_false:
                    uncovered.append("false")
            elif scrutinee_type in ('int', 'i32', 'i64', 'f32', 'f64'):
                uncovered.append("other values")
        
        return uncovered


class ExhaustivenessVisitor(IRVisitor[None]):
    """Visitor to check exhaustiveness in IR"""
    
    def __init__(self, checker: ExhaustivenessChecker):
        self.checker = checker
    
    def visit_program(self, node: ProgramIR) -> None:
        """Visit program and check exhaustiveness of all match expressions"""
        # Visit all statements
        for stmt in node.statements:
            stmt.accept(self)
        # Visit all functions
        for func in node.functions:
            func.accept(self)
        # Visit all constants
        for const in node.constants:
            const.accept(self)
    
    def visit_match_expression(self, expr: MatchExpressionIR) -> None:
        """Check exhaustiveness of match expression"""
        is_exhaustive = self.checker.check_exhaustiveness(expr)
        
        if not is_exhaustive:
            uncovered = self.checker.find_uncovered_cases(expr)
            if uncovered:
                missing = ", ".join(f"`{u}`" for u in uncovered)
                single = len(uncovered) == 1
                label = f"pattern {missing} not covered" if single else f"patterns {missing} not covered"
                self.checker.tcx.reporter.report_error(
                    f"non-exhaustive patterns: {missing} not covered",
                    location=expr.location,
                    code="E0004",
                    label=label,
                    help="ensure that all possible cases are being handled by adding a match arm with a wildcard pattern `_`",
                )
        
        # Process scrutinee and arms
        expr.scrutinee.accept(self)
        for arm in expr.arms:
            arm.pattern.accept(self)
            arm.body.accept(self)
    
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
    
    def visit_cast_expression(self, node) -> None:
        pass
    
    def visit_member_access(self, node) -> None:
        pass
    
    def visit_try_expression(self, node) -> None:
        pass
    
    def visit_reduction_expression(self, node) -> None:
        pass
    
    def visit_where_expression(self, node) -> None:
        pass
    
    def visit_pipeline_expression(self, node) -> None:
        pass
    
    def visit_builtin_call(self, node) -> None:
        pass
    
    def visit_binding(self, node) -> None:
        if is_function_binding(node) or is_einstein_binding(node):
            return None
        if hasattr(node, 'value') and node.value:
            return node.value.accept(self)
        return None
    
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
        if hasattr(node, 'alternatives'):
            for alt in node.alternatives:
                alt.accept(self)
    
    def visit_constructor_pattern(self, node) -> None:
        if hasattr(node, 'patterns'):
            for p in node.patterns:
                p.accept(self)
    
    def visit_binding_pattern(self, node) -> None:
        if hasattr(node, 'inner_pattern'):
            node.inner_pattern.accept(self)
    
    def visit_range_pattern(self, node) -> None:
        pass
    
    def visit_module(self, node) -> None:
        pass

