"""
Constraint Classifier Pass

Rust Pattern: Constraint Classification, Dependency Resolution
Reference: CONSTRAINT_ANALYSIS_DESIGN.md
"""

from __future__ import annotations
from enum import Enum
from typing import List, Dict, Set, Optional, Any
from ..passes.base import BasePass, TyCtxt
from ..ir.nodes import (
    ProgramIR, ExpressionIR, BinaryOpIR, WhereExpressionIR,
    ReductionExpressionIR, WhereClauseIR, IRVisitor, IdentifierIR
)
from ..shared.defid import DefId
from ..shared.types import BinaryOp


class ConstraintType(Enum):
    """Constraint type classification"""
    INDEX_RANGE = "INDEX_RANGE"  # i in 0..N
    INDEX_RELATIONAL = "INDEX_RELATIONAL"  # i < j, i >= 0
    BINDING = "BINDING"  # j = i + 1
    VALUE_RELATIONAL = "VALUE_RELATIONAL"  # A[i] > 0
    UNKNOWN = "UNKNOWN"


class ConstraintClassifierPass(BasePass):
    """
    Constraint classification pass.
    
    Classifies constraints in where clauses and comprehensions into types
    for proper handling.
    """
    requires = []  # No dependencies
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """Classify constraints in IR"""
        classifier = ConstraintClassifier(tcx)
        
        # Classify constraints in all expressions
        visitor = ConstraintClassificationVisitor(classifier)
        
        # Process all functions
        for func in ir.functions:
            func.body.accept(visitor)
        
        # Process all statements
        for stmt in ir.statements:
            stmt.accept(visitor)
        
        # Store classification results in TyCtxt
        tcx.set_analysis(ConstraintClassifierPass, classifier.classifications)
        
        return ir


class ConstraintClassifier:
    """Constraint classifier - classifies constraints by type"""
    
    def __init__(self, tcx: TyCtxt):
        self.tcx = tcx
        self.classifications: Dict[ExpressionIR, ConstraintType] = {}
        self.index_vars: Set[str] = set()  # Set of index variable names
    
    def classify_constraint(self, constraint: ExpressionIR) -> ConstraintType:
        """Classify constraint type - visitor pattern"""
        return constraint.accept(ConstraintTypeClassifier(self))
    
    def _involves_index_vars(self, expr: ExpressionIR) -> bool:
        """Check if expression involves index variables - visitor pattern"""
        return expr.accept(IndexVarChecker(self.index_vars))
    
    def resolve_binding_order(self, constraints: List[ExpressionIR]) -> List[ExpressionIR]:
        """Resolve binding order based on dependencies"""
        # Build dependency graph
        graph = self._build_dependency_graph(constraints)
        
        # Topological sort
        ordered = self._topological_sort(graph, constraints)
        
        return ordered
    
    def _build_dependency_graph(self, constraints: List[ExpressionIR]) -> Dict[ExpressionIR, Set[ExpressionIR]]:
        """Build dependency graph for bindings"""
        graph: Dict[ExpressionIR, Set[ExpressionIR]] = {c: set() for c in constraints}
        
        # Find bindings (j = i + 1)
        bindings = [c for c in constraints if self.classify_constraint(c) == ConstraintType.BINDING]
        
        for binding in bindings:
            # Extract bound variable and dependencies using visitor pattern
            binding_info = binding.accept(BindingExtractor())
            if binding_info:
                bound_var, deps = binding_info
                
                # Add edges from dependencies to this binding
                for dep_var in deps:
                    for other_binding in bindings:
                        other_info = other_binding.accept(BindingExtractor())
                        if other_info and other_info[0] == dep_var:
                            graph[binding].add(other_binding)
        
        return graph
    
    def _extract_variables(self, expr: ExpressionIR) -> Set[str]:
        """Extract variable names from expression - visitor pattern"""
        return expr.accept(VariableExtractor())
    
    def _topological_sort(self, graph: Dict[ExpressionIR, Set[ExpressionIR]], 
                         constraints: List[ExpressionIR]) -> List[ExpressionIR]:
        """Topological sort of constraints"""
        in_degree = {c: len(graph[c]) for c in constraints}
        queue = [c for c, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            constraint = queue.pop(0)
            result.append(constraint)
            
            for other_constraint in constraints:
                if constraint in graph[other_constraint]:
                    in_degree[other_constraint] -= 1
                    if in_degree[other_constraint] == 0:
                        queue.append(other_constraint)
        
        # Add remaining constraints (non-bindings)
        for constraint in constraints:
            if constraint not in result:
                result.append(constraint)
        
        return result


class ConstraintClassificationVisitor(IRVisitor[None]):
    """Visitor to classify constraints in IR"""
    
    def __init__(self, classifier: ConstraintClassifier):
        self.classifier = classifier
    
    def visit_program(self, node: ProgramIR) -> None:
        """Visit program and classify all constraints"""
        # Visit all statements
        for stmt in node.statements:
            stmt.accept(self)
        # Visit all functions
        for func in node.functions:
            func.accept(self)
        # Visit all constants
        for const in node.constants:
            const.accept(self)
    
    def visit_where_expression(self, expr: WhereExpressionIR) -> None:
        """Classify constraints in where expression"""
        for constraint in expr.constraints:
            constraint_type = self.classifier.classify_constraint(constraint)
            self.classifier.classifications[constraint] = constraint_type
            constraint.accept(self)
    
    def visit_reduction_expression(self, expr: ReductionExpressionIR) -> None:
        """Classify constraints in reduction expression"""
        if expr.where_clause:
            for constraint in expr.where_clause.constraints:
                constraint_type = self.classifier.classify_constraint(constraint)
                self.classifier.classifications[constraint] = constraint_type
                constraint.accept(self)
    
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
    
    def visit_match_expression(self, node) -> None:
        pass
    
    def visit_arrow_expression(self, node) -> None:
        pass
    
    def visit_pipeline_expression(self, node) -> None:
        pass
    
    def visit_builtin_call(self, node) -> None:
        pass
    
    def visit_function_ref(self, node) -> None:
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


    def visit_variable_declaration(self, node) -> None:
        """Visit variable declaration - recurse into value"""
        if hasattr(node, 'value') and node.value:
            node.value.accept(self)

class ConstraintTypeClassifier(IRVisitor[ConstraintType]):
    """Visitor to classify constraint type"""
    
    def __init__(self, classifier: ConstraintClassifier):
        self.classifier = classifier
    
    def visit_binary_op(self, expr: BinaryOpIR) -> ConstraintType:
        """Classify binary operation constraint using visitor pattern (dictionary dispatch)"""
        def _classify_relational() -> ConstraintType:
            """Classify relational operator constraint"""
            if self.classifier._involves_index_vars(expr):
                return ConstraintType.INDEX_RELATIONAL
            else:
                return ConstraintType.VALUE_RELATIONAL
        
        op = getattr(expr, "operator", None)
        if op == BinaryOp.IN:
            return ConstraintType.INDEX_RANGE
        if op == BinaryOp.ASSIGN:
            return ConstraintType.BINDING
        if op in (BinaryOp.LT, BinaryOp.GT, BinaryOp.LE, BinaryOp.GE, BinaryOp.EQ, BinaryOp.NE):
            return _classify_relational()
        
        return ConstraintType.UNKNOWN
    
    # Default: UNKNOWN for other expression types
    def visit_literal(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_identifier(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_function_call(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_unary_op(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_rectangular_access(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_jagged_access(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_block_expression(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_if_expression(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_lambda(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_range(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_array_comprehension(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_array_literal(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_tuple_expression(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_tuple_access(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_interpolated_string(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_cast_expression(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_member_access(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_try_expression(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_match_expression(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_reduction_expression(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_where_expression(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_arrow_expression(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_pipeline_expression(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_builtin_call(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_function_ref(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_einstein_declaration(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_literal_pattern(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_identifier_pattern(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_wildcard_pattern(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_tuple_pattern(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_array_pattern(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_rest_pattern(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_guard_pattern(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_function_def(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_constant_def(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN
    
    def visit_module(self, node) -> ConstraintType:
        return ConstraintType.UNKNOWN


    def visit_variable_declaration(self, node) -> ConstraintType:
        """Visit variable declaration - recurse into value"""
        if hasattr(node, 'value') and node.value:
            return node.value.accept(self)
        return None

class IndexVarChecker(IRVisitor[bool]):
    """Visitor to check if expression involves index variables"""
    
    def __init__(self, index_vars: Set[str]):
        self.index_vars = index_vars
    
    def visit_identifier(self, expr: IdentifierIR) -> bool:
        """Check if identifier is an index variable"""
        return expr.name in self.index_vars
    
    def visit_binary_op(self, expr: BinaryOpIR) -> bool:
        """Check if binary operation involves index variables"""
        return (expr.left.accept(self) or expr.right.accept(self))
    
    # Default: False for other expression types
    def visit_literal(self, node) -> bool:
        return False
    
    def visit_function_call(self, node) -> bool:
        return False
    
    def visit_unary_op(self, node) -> bool:
        return False
    
    def visit_rectangular_access(self, node) -> bool:
        return False
    
    def visit_jagged_access(self, node) -> bool:
        return False
    
    def visit_block_expression(self, node) -> bool:
        return False
    
    def visit_if_expression(self, node) -> bool:
        return False
    
    def visit_lambda(self, node) -> bool:
        return False
    
    def visit_range(self, node) -> bool:
        return False
    
    def visit_array_comprehension(self, node) -> bool:
        return False
    
    def visit_array_literal(self, node) -> bool:
        return False
    
    def visit_tuple_expression(self, node) -> bool:
        return False
    
    def visit_tuple_access(self, node) -> bool:
        return False
    
    def visit_interpolated_string(self, node) -> bool:
        return False
    
    def visit_cast_expression(self, node) -> bool:
        return False
    
    def visit_member_access(self, node) -> bool:
        return False
    
    def visit_try_expression(self, node) -> bool:
        return False
    
    def visit_match_expression(self, node) -> bool:
        return False
    
    def visit_reduction_expression(self, node) -> bool:
        return False
    
    def visit_where_expression(self, node) -> bool:
        return False
    
    def visit_arrow_expression(self, node) -> bool:
        return False
    
    def visit_pipeline_expression(self, node) -> bool:
        return False
    
    def visit_builtin_call(self, node) -> bool:
        return False
    
    def visit_function_ref(self, node) -> bool:
        return False
    
    def visit_einstein_declaration(self, node) -> bool:
        return False
    
    def visit_literal_pattern(self, node) -> bool:
        return False
    
    def visit_identifier_pattern(self, node) -> bool:
        return False
    
    def visit_wildcard_pattern(self, node) -> bool:
        return False
    
    def visit_tuple_pattern(self, node) -> bool:
        return False
    
    def visit_array_pattern(self, node) -> bool:
        return False
    
    def visit_rest_pattern(self, node) -> bool:
        return False
    
    def visit_guard_pattern(self, node) -> bool:
        return False
    
    def visit_function_def(self, node) -> bool:
        return False
    
    def visit_constant_def(self, node) -> bool:
        return False
    
    def visit_module(self, node) -> bool:
        return False


    def visit_variable_declaration(self, node) -> bool:
        """Visit variable declaration - recurse into value"""
        if hasattr(node, 'value') and node.value:
            return node.value.accept(self)
        return None

class BindingExtractor(IRVisitor):
    """Visitor to extract binding information (bound_var, dependencies)"""
    
    def visit_binary_op(self, expr: BinaryOpIR) -> Optional[tuple[str, Set[str]]]:
        """Extract binding from binary operation"""
        if getattr(expr, "operator", None) == BinaryOp.ASSIGN:
            # Left side is the bound variable - direct attribute access
            bound_var = expr.left.name if hasattr(expr.left, 'name') else None
            if bound_var:
                # Right side contains dependencies
                deps = expr.right.accept(VariableExtractor())
                return (bound_var, deps)
        
        return None
    
    # Default: None for other expression types
    def visit_literal(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_identifier(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_function_call(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_unary_op(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_rectangular_access(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_jagged_access(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_block_expression(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_if_expression(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_lambda(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_range(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_array_comprehension(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_array_literal(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_tuple_expression(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_tuple_access(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_interpolated_string(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_cast_expression(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_member_access(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_try_expression(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_match_expression(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_reduction_expression(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_where_expression(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_arrow_expression(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_pipeline_expression(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_builtin_call(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_function_ref(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_einstein_declaration(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_literal_pattern(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_identifier_pattern(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_wildcard_pattern(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_tuple_pattern(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_array_pattern(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_rest_pattern(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_guard_pattern(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_function_def(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_constant_def(self, node) -> Optional[tuple[str, Set[str]]]:
        return None
    
    def visit_module(self, node) -> Optional[tuple[str, Set[str]]]:
        return None


    def visit_variable_declaration(self, node):
        """Visit variable declaration - recurse into value"""
        if hasattr(node, 'value') and node.value:
            return node.value.accept(self)
        return None

class VariableExtractor(IRVisitor):
    """Visitor to extract variable names from expression"""
    
    def visit_program(self, node: ProgramIR) -> Set[str]:
        """Visit program - not used for variable extraction"""
        return set()
    
    def visit_identifier(self, expr: IdentifierIR) -> Set[str]:
        """Extract identifier name"""
        return {expr.name}
    
    def visit_binary_op(self, expr: BinaryOpIR) -> Set[str]:
        """Extract variables from binary operation"""
        vars: Set[str] = set()
        vars.update(expr.left.accept(self))
        vars.update(expr.right.accept(self))
        return vars
    
    # Default: empty set for other expression types
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
    
    def visit_arrow_expression(self, node) -> Set[str]:
        return set()
    
    def visit_pipeline_expression(self, node) -> Set[str]:
        return set()
    
    def visit_builtin_call(self, node) -> Set[str]:
        return set()
    
    def visit_function_ref(self, node) -> Set[str]:
        return set()
    
    def visit_einstein_declaration(self, node) -> Set[str]:
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
    
    def visit_function_def(self, node) -> Set[str]:
        return set()
    
    def visit_constant_def(self, node) -> Set[str]:
        return set()
    
    def visit_module(self, node) -> Set[str]:
        return set()

    def visit_variable_declaration(self, node) -> Any:
        """Visit variable declaration - recurse into value"""
        if hasattr(node, 'value') and node.value:
            return node.value.accept(self)
        return None

