"""
AST Visitor Pattern and Scope Management

This module provides:
1. AST validation errors (for fail-fast error reporting)
2. ScopedASTAnalyzer (scope management without visitor pattern)
3. ASTVisitor (abstract visitor - for future when AST nodes get accept() methods)
4. ScopedASTVisitor (visitor + scope management - for future migration)

Design:
- Abstract base class with visit_* methods for each AST node type
- Type-safe (mypy can check)
- Extensible (add new visitors without changing nodes)
- Standard compiler pattern (LLVM, Rust MIR, Swift SIL)

Future Work:
- Add accept() methods to AST nodes in shared/nodes.py
- Migrate analyzers from ScopedASTAnalyzer to ScopedASTVisitor
"""

from typing import TypeVar, Generic, Dict, List, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
from contextlib import contextmanager

if TYPE_CHECKING:
    from .nodes import Literal

T = TypeVar('T')


# ============================================
# AST VALIDATION ERRORS (Industry Pattern)
# ============================================

class ASTValidationError(Exception):
    """
    Base class for AST validation errors.
    
    Industry Standard: AST passes validate preconditions and invariants (fail-fast).
    - Rust: HIR validation checks well-formedness
    - Einlang: AST passes validate semantic rules
    """

class TypeValidationError(ASTValidationError):
    """Type validation error - raised when type rules violated."""

class ShapeValidationError(ASTValidationError):
    """Shape validation error - raised when shape rules violated."""

class PrecisionValidationError(ASTValidationError):
    """Precision validation error - raised when precision rules violated."""


# ============================================
# SCOPED AST ANALYZER (Current Implementation)
# ============================================

class ScopedASTAnalyzer(Generic[T]):
    """
    Base class for AST analyzers that need scope management.
    
    NOTE: This is NOT a visitor pattern implementation!
    - AST nodes currently don't have accept() methods
    - Analyzers do manual isinstance() checks
    - For the proper visitor pattern version, see ScopedASTVisitor below
    
    Industry Best Practice: Extract common scope logic to avoid duplication.
    Follows RAII pattern with context managers for automatic scope cleanup.
    
    Design Philosophy (from LLVM/Rust/Swift):
    - Scope is TEMPORARY (only during analysis pass)
    - Results are PERMANENT (attached to AST nodes)
    - Lexical scoping with shadowing support
    - Clean scope entry/exit via context managers
    
    Subclasses:
    - TypeInferenceEngine: T = TypeInfo (tracks numeric precision)
    - TypeChecker: T = TypeInfo (tracks variable types)
    - ShapeAnalyzer: T = ShapeInfo (tracks tensor shapes)
    
    Usage:
        class MyAnalyzer(ScopedASTAnalyzer[MyType]):
            def analyze_function(self, node):
                with self._scope():
                    self._set_var("x", my_value)
                    # Analyze function body
                # Scope automatically exits
    """
    
    def __init__(self):
        # Scope stack: List[Dict[var_name, value]]
        # Index 0 is global scope, higher indices are nested scopes
        self._scope_stack: List[Dict[str, T]] = [{}]
    
    # =========================================================================
    # Scope Management (Protected API for Subclasses)
    # =========================================================================
    
    @contextmanager
    def _scope(self):
        """
        Context manager for entering/exiting a scope (RAII pattern).
        
        Industry Best Practice (C++ RAII, Python with-statement):
        - Automatic resource management
        - Exception-safe (scope exits even on error)
        - Clear scope boundaries in code
        
        Usage:
            with self._scope():
                # New scope active here
                self._set_var("x", value)
                # Do work...
            # Scope automatically exits
        """
        self._push_scope()
        try:
            yield
        finally:
            self._pop_scope()
    
    def _push_scope(self) -> None:
        """Enter a new scope (prefer using _scope() context manager)"""
        self._scope_stack.append({})
    
    def _pop_scope(self) -> None:
        """Exit current scope (prefer using _scope() context manager)"""
        if len(self._scope_stack) > 1:
            self._scope_stack.pop()
    
    def _set_var(self, var_name: str, value: T) -> None:
        """Set variable data in current scope (with shadowing)"""
        self._scope_stack[-1][var_name] = value
    
    def _get_var(self, var_name: str) -> Optional[T]:
        """Get variable data, checking scopes from innermost to outermost"""
        for scope in reversed(self._scope_stack):
            if var_name in scope:
                return scope[var_name]
        return None
    
    def _current_scope(self) -> Dict[str, T]:
        """Get current scope dictionary"""
        return self._scope_stack[-1]
    
    def _global_scope(self) -> Dict[str, T]:
        """Get global scope dictionary"""
        return self._scope_stack[0]
    
    def _scope_depth(self) -> int:
        """Get current scope depth"""
        return len(self._scope_stack)


# ============================================
# AST VISITOR PATTERN (Active Implementation)
# ============================================

class ASTVisitor(ABC, Generic[T]):
    """
    Base AST visitor with default traversal for all nodes.
    
    Provides:
    - Default traversal for non-leaf nodes (visits children automatically)
    - Enforced implementation for leaf nodes (Literal, Identifier, etc.)
    - Type-safe polymorphic dispatch
    
    Leaf nodes that MUST be implemented:
    - visit_literal, visit_identifier, visit_module_access
    - visit_use_statement, visit_module_declaration
    
    All other nodes have default traversal. Override to add custom behavior.
    
    Usage:
        class MyAnalyzer(ASTVisitor[Result]):
            # Custom logic for specific nodes
            def visit_binary_expression(self, node) -> Result:
                left = node.left.accept(self)
                right = node.right.accept(self)
                return combine(left, right)
        
            # Implement required leaf nodes
            def visit_literal(self, node) -> Result:
                return Result(node.value)
            
            def visit_identifier(self, node) -> Result:
                return self.lookup(node.name)
    """
    
    # Leaf nodes - NO DEFAULT IMPLEMENTATION
    # Subclasses must explicitly handle these
    @abstractmethod
    def visit_literal(self, node: 'Literal') -> T:
        raise NotImplementedError(f"{self.__class__.__name__} must implement visit_literal()")
    
    @abstractmethod
    def visit_identifier(self, node) -> T:
        raise NotImplementedError(f"{self.__class__.__name__} must implement visit_identifier()")
    
    def visit_index_var(self, node) -> T:
        """Visit variable index slot (name + optional range). Delegate to range_expr if present."""
        if getattr(node, "range_expr", None) is not None:
            node.range_expr.accept(self)

    def visit_index_rest(self, node) -> T:
        """Visit rest index slot (..name). Leaf node."""
        pass

    # Expressions with children
    def visit_binary_expression(self, node) -> T:
        node.left.accept(self)
        node.right.accept(self)
    
    def visit_unary_expression(self, node) -> T:
        node.operand.accept(self)
    
    def visit_function_call(self, node) -> T:
        node.function_expr.accept(self)
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_array_literal(self, node) -> T:
        for elem in node.elements:
            elem.accept(self)
    
    def visit_rectangular_access(self, node) -> T:
        node.base_expr.accept(self)
        for idx in node.indices:
            idx.accept(self)
    
    def visit_jagged_access(self, node) -> T:
        node.base_expr.accept(self)
        for idx in node.index_chain:
            idx.accept(self)
    
    def visit_array_comprehension(self, node) -> T:
        node.expr.accept(self)
        if node.constraints:
            for constraint in node.constraints:
                constraint.accept(self)
    
    def visit_reduction_expression(self, node) -> T:
        node.body.accept(self)
        if node.where_clause:
            for constraint in node.where_clause.constraints:
                constraint.accept(self)
    
    def visit_where_expression(self, node) -> T:
        node.expr.accept(self)
        for constraint in node.where_clause.constraints:
            constraint.accept(self)
    
    def visit_if_expression(self, node) -> T:
        node.condition.accept(self)
        node.then_block.accept(self)
        if node.else_block:
            node.else_block.accept(self)
    
    def visit_lambda_expression(self, node) -> T:
        node.body.accept(self)
    
    def visit_block_expression(self, node) -> T:
        for stmt in node.statements:
            stmt.accept(self)
        if node.final_expr:
            node.final_expr.accept(self)
    
    def visit_cast_expression(self, node) -> T:
        node.expr.accept(self)
    
    def visit_member_access(self, node) -> T:
        node.object.accept(self)
    
    @abstractmethod
    def visit_module_access(self, node) -> T:
        raise NotImplementedError(f"{self.__class__.__name__} must implement visit_module_access()")
    
    def visit_method_call(self, node) -> T:
        node.object.accept(self)
        node.method_expr.accept(self)
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_tuple_expression(self, node) -> T:
        for elem in node.elements:
            elem.accept(self)
    
    def visit_pipeline_expression(self, node) -> T:
        node.left.accept(self)
        node.right.accept(self)
    
    def visit_try_expression(self, node) -> T:
        node.operand.accept(self)
    
    def visit_interpolated_string(self, node) -> T:
        # Trust: all parts are Expression nodes (Literal or InterpolationPart) with accept()
        for part in node.parts:
            part.accept(self)
    
    def visit_range(self, node) -> T:
        if node.start:
            node.start.accept(self)
        if node.end:
            node.end.accept(self)
    
    # Pattern matching
    def visit_match_expression(self, node) -> T:
        """Visit match expression - default implementation visits scrutinee and patterns only"""
        scrutinee_result = node.scrutinee.accept(self)
        arm_results = [arm.pattern.accept(self) for arm in node.arms]
        # NOTE: Arm bodies are NOT visited by default - subclasses should override
        # to visit bodies with proper scope setup
        return scrutinee_result, arm_results
    
    def visit_literal_pattern(self, node) -> T:
        """Visit literal pattern"""
        return node.value.accept(self)
    
    def visit_identifier_pattern(self, node) -> T:
        """Visit identifier pattern"""
        # Leaf node - no children
        return node.name
    
    def visit_wildcard_pattern(self, node) -> T:
        """Visit wildcard pattern"""
        # Leaf node - no children
        return None
    
    def visit_tuple_pattern(self, node) -> T:
        """Visit tuple pattern"""
        return [p.accept(self) for p in node.patterns]
    
    def visit_array_pattern(self, node) -> T:
        """Visit array pattern"""
        for p in node.patterns:
            p.accept(self)
    
    def visit_rest_pattern(self, node) -> T:
        """Visit rest pattern: ..pattern"""
        node.pattern.accept(self)
    
    def visit_guard_pattern(self, node) -> T:
        """Visit guard pattern"""
        pattern_result = node.pattern.accept(self)
        guard_result = node.guard.accept(self)
        return pattern_result, guard_result
    
    # Statements
    def visit_program(self, node) -> T:
        for stmt in node.statements:
            stmt.accept(self)
    
    def visit_function_definition(self, node) -> T:
        if node.body:
            node.body.accept(self)
    
    def visit_variable_declaration(self, node) -> T:
        if node.value:
            node.value.accept(self)
    
    def visit_expression_statement(self, node) -> T:
        node.expr.accept(self)
    
    def visit_einstein_declaration(self, node) -> T:
        for clause in node.clauses:
            clause.accept(self)

    def visit_enum_definition(self, node) -> T:
        """Visit enum definition - no children to visit"""
        # Enum definitions are type-level, no expression children
        pass
    
    def visit_struct_definition(self, node) -> T:
        """Visit struct definition - no children to visit"""
        # Struct definitions are type-level, no expression children
        pass
    
    def visit_constructor_pattern(self, node) -> T:
        """Visit constructor pattern: Circle(r), Point { x, y }"""
        return [p.accept(self) for p in node.patterns]
    
    def visit_or_pattern(self, node) -> T:
        """Visit or pattern: pat1 | pat2 | ..."""
        return [alt.accept(self) for alt in node.alternatives]
    
    def visit_binding_pattern(self, node) -> T:
        """Visit binding pattern: name @ pattern"""
        return node.pattern.accept(self)
    
    def visit_range_pattern(self, node) -> T:
        """Visit range pattern: start..end or start..=end"""
        pass
    
    @abstractmethod
    def visit_use_statement(self, node) -> T:
        raise NotImplementedError(f"{self.__class__.__name__} must implement visit_use_statement()")
    
    @abstractmethod
    def visit_module_declaration(self, node) -> T:
        raise NotImplementedError(f"{self.__class__.__name__} must implement visit_module_declaration()")
    
    def visit_inline_module(self, node) -> T:
        for stmt in node.body:
            stmt.accept(self)


class ScopedASTVisitor(ASTVisitor[T], Generic[T]):
    """
    AST visitor with scope management (like ScopedIRVisitor).
    
    Combines:
    - Visitor pattern (polymorphic dispatch via visit_* methods)
    - Scope management (lexical scoping with context managers)
    
    Industry Pattern: Like LLVM analysis passes, Rust HIR checker.
    
    ✅ AST nodes now have accept() methods - this class is ready to use!
    Use this instead of ScopedASTAnalyzer for all new analysis code.
    
    Usage:
        class MyAnalyzer(ScopedASTVisitor[MyResult]):
            def visit_function_definition(self, node: FunctionDefinition) -> MyResult:
                with self._scope():
                    # Bind parameters
                    for param in node.parameters:
                        self._set_var(param.name, param_info)
                    
                    # Visit body - visitor pattern handles dispatch!
                    # Trust: node.body is BlockExpression with accept() method
                    node.body.accept(self)
                
                return result
    """
    
    def __init__(self):
        # Scope stack: List[Dict[var_name, value]]
        self._scope_stack: List[Dict[str, T]] = [{}]
    
    # =========================================================================
    # Scope Management API (Public - for use by subclasses and helpers)
    # =========================================================================
    
    @contextmanager
    def scope(self):
        """
        Context manager for entering/exiting a scope (RAII pattern).
        
        Usage:
            with self.scope():
                self.set_var("x", value)
                # Do work in new scope...
            # Scope automatically exits
        """
        self._scope_stack.append({})
        try:
            yield
        finally:
            if len(self._scope_stack) > 1:
                self._scope_stack.pop()
    
    def set_var(self, var_name: str, value: T) -> None:
        """Set variable in current scope (shadows outer scopes)."""
        self._scope_stack[-1][var_name] = value
    
    def get_var(self, var_name: str) -> Optional[T]:
        """Get variable from scope chain (inner → outer, natural shadowing)."""
        for scope in reversed(self._scope_stack):
            if var_name in scope:
                return scope[var_name]
        return None

