"""
Einstein Declaration Grouping Pass

Rust Pattern: Analysis pass (no direct Rust equivalent - Einlang-specific)
Reference: PASS_SYSTEM_DESIGN.md

Design Pattern: Visitor pattern for IR traversal (no isinstance/hasattr)

Groups Einstein declarations by array name on IR. Used by RestPattern and ShapeAnalysis
(via tcx.get_analysis). Mono service runs this when analyzing specialized function bodies.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..passes.base import BasePass, TyCtxt
from ..passes.ast_to_ir import ASTToIRLoweringPass
from ..ir.nodes import (
    ProgramIR, EinsteinDeclarationIR, IRVisitor, IRNode,
    FunctionDefIR, BlockExpressionIR, IfExpressionIR
)
from ..shared.source_location import SourceLocation
from ..shared.defid import DefId

logger = logging.getLogger("einlang.passes.einstein_grouping")


@dataclass
class DeclarationGroup:
    """
    Groups related Einstein tensor declarations that operate on the same array/tensor.
    
    Rust Pattern: N/A (Einlang-specific)
    """
    array_name: str
    declarations: List[EinsteinDeclarationIR]
    max_dimensions: List[int]
    has_simple_assignments: bool = False
    has_einstein_assignments: bool = False
    is_complete: bool = False  # Set by coverage analysis when group has complete coverage


class EinsteinDeclarationGroupingPass(BasePass):
    """
    Einstein declaration grouping pass.
    
    Rust Pattern: N/A (Einlang-specific feature)
    
    Responsibility: ANALYSIS ONLY (no transformation)
    - Groups: Einstein declarations by array name
    - Tracks: Declaration sequences and dependencies
    - Enables: Recurrence relation analysis (Fibonacci, etc.)
    - Stores: Group metadata in TyCtxt
    
    Industry Pattern (LLVM/Rust/Swift):
    - Foundational pass (runs first, no dependencies)
    - Simple grouping algorithm
    - Results used by downstream passes
    """
    requires = [ASTToIRLoweringPass]  # Depends on AST→IR lowering
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """
        Group Einstein declarations by array name.
        
        Rust Pattern: Analysis pass stores results in TyCtxt
        """
        logger.debug("Starting Einstein declaration grouping analysis")
        
        # Collect Einstein declarations using visitor pattern
        visitor = EinsteinDeclarationCollector()
        
        # Visit function bodies
        for func in ir.functions:
            func.body.accept(visitor)
        
        # Visit constant values
        for const in ir.constants:
            const.value.accept(visitor)
        
        # Visit top-level statements (may include Einstein declarations)
        for stmt in ir.statements:
            # Statements can be ExpressionIR or EinsteinDeclarationIR
            if isinstance(stmt, EinsteinDeclarationIR):
                visitor.declarations.append(stmt)
            elif hasattr(stmt, 'accept'):
                stmt.accept(visitor)
        
        einstein_declarations = visitor.declarations
        
        if not einstein_declarations:
            logger.debug("No Einstein declarations found")
            # Store empty result in TyCtxt
            tcx.set_analysis(EinsteinDeclarationGroupingPass, {})
            return ir
        
        # Group declarations by array name
        groups = self._group_declarations_by_array(einstein_declarations, tcx)
        
        # Store grouping information in TyCtxt (Rust pattern: analysis results in TyCtxt)
        tcx.set_analysis(EinsteinDeclarationGroupingPass, groups)
        
        # Calculate statistics
        total_declarations = len(einstein_declarations)
        grouped_declarations = sum(len(group.declarations) for group in groups.values())
        
        logger.debug(f"Einstein declaration grouping complete: "
                    f"{len(groups)} groups, {total_declarations} total declarations, "
                    f"{grouped_declarations} grouped")
        
        return ir
    
    def _group_declarations_by_array(
        self,
        declarations: List[EinsteinDeclarationIR],
        tcx: TyCtxt,
    ) -> Dict[str, DeclarationGroup]:
        """Group declarations by array name (same name in same scope = same variable)."""
        groups: Dict[str, List[EinsteinDeclarationIR]] = {}

        for decl in declarations:
            key = decl.name
            if key not in groups:
                groups[key] = []
            groups[key].append(decl)
        
        # Create DeclarationGroup objects
        declaration_groups = {}
        for _key, decl_list in groups.items():
            # All declarations in group should have same name (same variable)
            array_name = decl_list[0].name
            
            # Sort declarations by their position in the program
            decl_list.sort(key=lambda d: d.location.line if d.location else 0)
            
            # Fail on rank mismatch: all clauses in each decl must have the same rank
            for decl in decl_list:
                clauses = decl.clauses or []
                if len(clauses) < 2:
                    continue
                ranks = [len(c.indices) for c in clauses]
                if len(set(ranks)) > 1:
                    tcx.reporter.report_error(
                        f"Einstein declaration '{decl.name}' has clauses with different ranks: {ranks}. All clauses must have the same rank.",
                        location=decl.location,
                    )
            
            # All clauses have the same rank; use first clause to get rank
            rank = 0
            for decl in decl_list:
                if decl.clauses:
                    rank = len(decl.clauses[0].indices)
                    break
            if rank == 0:
                rank = 1

            # Actual array size from literal indices in every clause (same rank)
            actual_dimensions = [0] * rank
            from ..passes.visitor_helpers import ConstantEvaluator
            evaluator = ConstantEvaluator()
            for decl in decl_list:
                for clause in (decl.clauses or []):
                    for i, idx_expr in enumerate(clause.indices or []):
                        if i < rank:
                            idx_value = idx_expr.accept(evaluator)
                            if idx_value is not None and isinstance(idx_value, int):
                                actual_dimensions[i] = max(actual_dimensions[i], idx_value)
            
            # Convert max index to array size: size = max_index + 1
            # Ensure we have at least 1 element in each dimension
            actual_dimensions = [max(1, dim + 1) for dim in actual_dimensions]
            
            group = DeclarationGroup(
                array_name=array_name,
                declarations=decl_list,
                max_dimensions=actual_dimensions,
                has_simple_assignments=True,  # Assume simple assignments
                has_einstein_assignments=False  # Assume no Einstein assignments for now
            )
            # Use array name as key (for backward compatibility with consumers)
            declaration_groups[array_name] = group
        
        return declaration_groups


class EinsteinDeclarationCollector(IRVisitor[None]):
    """
    Visitor to collect Einstein declarations from IR.
    Grouping is done by DefId (same DefId = same variable = same group).
    
    Rust Pattern: rustc_hir::intravisit::Visitor
    """
    
    def __init__(self):
        self.declarations: List[EinsteinDeclarationIR] = []
    
    def visit_program(self, node: ProgramIR) -> None:
        """Visit program and collect from all statements"""
        # Visit all statements (visitor pattern handles dispatch)
        for stmt in node.statements:
            stmt.accept(self)
        # Visit all functions
        for func in node.functions:
            func.accept(self)
    
    def visit_einstein_declaration(self, node: EinsteinDeclarationIR) -> None:
        """Collect Einstein declaration"""
        # DefId identifies the variable - declarations with same DefId are grouped together
        self.declarations.append(node)
        # Einstein declarations use 'array_name', not 'name'
        node_name = getattr(node, 'array_name', None) or getattr(node, 'name', None) or '<unknown>'
        clause_counts = [len(c.indices) for c in (node.clauses or [])]
        logger.debug(f"  - {node_name} with {len(node.clauses or [])} clause(s), indices per clause: {clause_counts}")
    
    def visit_variable_declaration(self, node) -> None:
        """Visit variable declaration - recurse into value"""
        if node.value:
            node.value.accept(self)
    
    def visit_function_def(self, node: FunctionDefIR) -> None:
        """Collect from function bodies"""
        if node.body:
            node.body.accept(self)
    
    def visit_block_expression(self, node: BlockExpressionIR) -> None:
        """Visit block expressions"""
        # Visit all statements in block (visitor pattern handles dispatch)
        for stmt in node.statements:
            stmt.accept(self)
        
        # Visit final expression if present
        if node.final_expr:
            node.final_expr.accept(self)
    
    def visit_if_expression(self, node: IfExpressionIR) -> None:
        """Visit if expressions"""
        if node.condition:
            node.condition.accept(self)
        if node.then_expr:
            node.then_expr.accept(self)
        if node.else_expr:
            node.else_expr.accept(self)
    
    # Default implementations for other nodes (no-op for collection)
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
        if getattr(node, 'base', None):
            node.base.accept(self)
        for idx in (getattr(node, 'index_chain', None) or []):
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

    def visit_lowered_comprehension(self, node) -> None:
        """Recurse into lowered comprehension body (no Einstein decl to collect)."""
        if node.body:
            node.body.accept(self)

    def visit_lowered_reduction(self, node) -> None:
        """Recurse into lowered reduction body and guards (no Einstein decl to collect)."""
        if node.body:
            node.body.accept(self)
        for guard in getattr(node, 'guards', None) or []:
            if getattr(guard, 'condition', None):
                guard.condition.accept(self)
    
    def visit_arrow_expression(self, node) -> None:
        for comp in node.components:
            comp.accept(self)
    
    def visit_pipeline_expression(self, node) -> None:
        if node.left:
            node.left.accept(self)
        if node.right:
            node.right.accept(self)
    
    def visit_builtin_call(self, node) -> None:
        for arg in node.args:
            arg.accept(self)
    
    def visit_constant_def(self, node) -> None:
        if node.value:
            node.value.accept(self)
    
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

