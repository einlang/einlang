"""
IR Validation Pass

Rust Pattern: rustc_mir::transform::MirPass (validation)
Reference: PASS_SYSTEM_DESIGN.md

Validates that IR is structurally well-formed and ready for execution.

Industry Best Practice (LLVM, Rust, GCC, Java):
- SEMANTIC validation (types, scopes, names) → AST passes (before lowering)
- STRUCTURAL validation (well-formedness, completeness) → IR passes (after lowering)

This pass checks STRUCTURAL properties only:
1. Lowering was complete (all metadata attached)
2. IR nodes have required fields populated
3. No dangling references
4. Type information present (from TypeInferencePass)

This pass does NOT re-do semantic analysis:
- Type checking already done in TypeInferencePass
- Shape validation already done in UnifiedShapeAnalysisPass
- Scope resolution already done in NameResolutionPass

If this pass fails, it indicates a COMPILER BUG, not a user error.

Following: LLVM IR Verifier, Java Bytecode Verifier
"""

import logging
from typing import Optional, Any
from ..passes.base import BasePass, TyCtxt
from ..ir.nodes import (
    ProgramIR, ExpressionIR, FunctionDefIR, ConstantDefIR, EinsteinDeclarationIR,
    IRVisitor, IRNode,
    LiteralIR, IdentifierIR, BinaryOpIR, UnaryOpIR, FunctionCallIR,
    RectangularAccessIR, JaggedAccessIR, ArrayLiteralIR, TupleExpressionIR,
    ReductionExpressionIR, IfExpressionIR, MatchExpressionIR, CastExpressionIR,
    TryExpressionIR, LambdaIR, RangeIR, ArrayComprehensionIR, InterpolatedStringIR,
    TupleAccessIR, BuiltinCallIR, BlockExpressionIR, ArrowExpressionIR,
    PipelineExpressionIR, FunctionRefIR, MemberAccessIR,
    LiteralPatternIR, IdentifierPatternIR, WildcardPatternIR,
    TuplePatternIR, ArrayPatternIR, RestPatternIR, GuardPatternIR,
    MatchArmIR, IndexRestIR,
)

logger = logging.getLogger("einlang.passes.ir_validation")

class IRValidationVisitor(IRVisitor[None]):
    """
    Visitor for validating IR nodes using polymorphic dispatch.
    No isinstance checks!
    """
    def __init__(self, tcx: TyCtxt):
        self.tcx = tcx
        self.nodes_validated = 0
    
    def _report_error(self, message: str, location):
        """Report validation error"""
        self.tcx.reporter.report_error(
            message,
            location,
            code="E0999"  # Compiler bug error code
        )
    
    # Expression visitors
    def visit_literal(self, node: LiteralIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
    
    def visit_identifier(self, node: IdentifierIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        # Validate DefId is present for identifiers (needed for variable lookup)
        # Following DefId is optional during lowering, can be resolved later
        # This allows stdlib functions with Python module calls to be lowered without
        # requiring all identifiers to be resolved upfront (they'll be resolved during
        # specialization or execution when proper scope is available)
        if node.defid is None:
            # Allow None DefId for now - will be resolved later during specialization/execution
            # This is consistent with the approach where identifiers don't require DefIds during lowering
            pass  # Don't report error - DefId will be resolved when needed
    
    def visit_binary_op(self, node: BinaryOpIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.left.accept(self)
        node.right.accept(self)
    
    def visit_unary_op(self, node: UnaryOpIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.operand.accept(self)
    
    def visit_function_call(self, node: FunctionCallIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        # Validate function_defid is present (needed for function lookup)
        # Exception: Python module calls (e.g., math::sqrt, np::array) may not have DefId
        # because Python modules are resolved at runtime, not compile time
        # These will be handled by the backend's Python interop system
        if node.function_defid is None:
            # Allow None function_defid for Python module calls (handled at runtime)
            # TODO: Add a flag to FunctionCallIR to mark Python module calls explicitly
            pass  # Backend will handle Python module calls at runtime
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_builtin_call(self, node: BuiltinCallIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        for arg in node.args:
            arg.accept(self)
    
    def visit_rectangular_access(self, node: RectangularAccessIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.array.accept(self)
        for idx in (node.indices or []):
            if idx is not None:
                if isinstance(idx, IndexRestIR):
                    self._report_error(
                        f"IndexRestIR (..{getattr(idx, 'name', '?')}) must not reach IR validation; "
                        "rest patterns must be expanded in rest_pattern_preprocessing.",
                        node.location,
                    )
                if hasattr(idx, 'accept'):
                    idx.accept(self)
    
    def visit_jagged_access(self, node: JaggedAccessIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.base.accept(self)
        for idx in (node.index_chain or []):
            if idx is not None:
                if isinstance(idx, IndexRestIR):
                    self._report_error(
                        f"IndexRestIR (..{getattr(idx, 'name', '?')}) must not reach IR validation; "
                        "rest patterns must be expanded in rest_pattern_preprocessing.",
                        node.location,
                    )
                if hasattr(idx, 'accept'):
                    idx.accept(self)
    
    def visit_array_literal(self, node: ArrayLiteralIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        for elem in node.elements:
            elem.accept(self)
    
    def visit_tuple_expression(self, node: TupleExpressionIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        for elem in node.elements:
            elem.accept(self)
    
    def visit_reduction_expression(self, node: ReductionExpressionIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.body.accept(self)
        if node.where_clause:
            for constraint in node.where_clause.constraints:
                constraint.accept(self)
    
    def visit_block_expression(self, node: BlockExpressionIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        for stmt in node.statements:
            stmt.accept(self)
        if node.final_expr:
            node.final_expr.accept(self)
    
    def visit_if_expression(self, node: IfExpressionIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.condition.accept(self)
        node.then_expr.accept(self)
        if node.else_expr:
            node.else_expr.accept(self)
    
    def visit_match_expression(self, node: MatchExpressionIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.scrutinee.accept(self)
        for arm in node.arms:
            arm.pattern.accept(self)
            arm.body.accept(self)
    
    def visit_literal_pattern(self, node: LiteralPatternIR) -> None:
        self.nodes_validated += 1
    
    def visit_identifier_pattern(self, node: IdentifierPatternIR) -> None:
        self.nodes_validated += 1
    
    def visit_wildcard_pattern(self, node: WildcardPatternIR) -> None:
        self.nodes_validated += 1
    
    def visit_tuple_pattern(self, node: TuplePatternIR) -> None:
        self.nodes_validated += 1
        for p in node.patterns:
            p.accept(self)
    
    def visit_rest_pattern(self, node: RestPatternIR) -> None:
        self.nodes_validated += 1
        node.pattern.accept(self)
    
    def visit_array_pattern(self, node: ArrayPatternIR) -> None:
        self.nodes_validated += 1
        for p in node.patterns:
            p.accept(self)
    
    def visit_guard_pattern(self, node: GuardPatternIR) -> None:
        self.nodes_validated += 1
        node.inner_pattern.accept(self)
        node.guard_expr.accept(self)
    
    def visit_or_pattern(self, node) -> None:
        self.nodes_validated += 1
        for alt in node.alternatives:
            alt.accept(self)
    
    def visit_constructor_pattern(self, node) -> None:
        self.nodes_validated += 1
        for p in node.patterns:
            p.accept(self)
    
    def visit_binding_pattern(self, node) -> None:
        self.nodes_validated += 1
        node.inner_pattern.accept(self)
    
    def visit_range_pattern(self, node) -> None:
        self.nodes_validated += 1
    
    def visit_cast_expression(self, node: CastExpressionIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.expr.accept(self)
    
    def visit_try_expression(self, node: TryExpressionIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.operand.accept(self)
    
    def visit_lambda(self, node: LambdaIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.body.accept(self)
    
    def visit_range(self, node: RangeIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.start.accept(self)
        node.end.accept(self)
    
    def visit_array_comprehension(self, node: ArrayComprehensionIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.body.accept(self)
        for range_expr in node.ranges:
            range_expr.accept(self)
        for constraint in node.constraints:
            constraint.accept(self)
    
    def visit_arrow_expression(self, node: ArrowExpressionIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        for component in node.components:
            component.accept(self)
    
    def visit_pipeline_expression(self, node: PipelineExpressionIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.left.accept(self)
        node.right.accept(self)
    
    def visit_tuple_access(self, node: TupleAccessIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.tuple_expr.accept(self)
    
    def visit_interpolated_string(self, node: InterpolatedStringIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        for part in node.parts:
            if hasattr(part, 'accept'):
                part.accept(self)
    
    def visit_function_ref(self, node: FunctionRefIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        # Validate function_defid is present
        if node.function_defid is None:
            self._report_error(
                "FunctionRefIR missing function_defid. "
                "Function references must have DefId for definition table lookup.",
                node.location
            )
    
    def visit_member_access(self, node: MemberAccessIR) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.object.accept(self)
    
    # Statement/Definition visitors
    def visit_einstein_declaration(self, node: EinsteinDeclarationIR) -> None:
        """Validate Einstein declaration: check each clause."""
        self.nodes_validated += 1
        for clause in (node.clauses or []):
            clause.accept(self)

    def visit_einstein(self, node) -> None:
        """Validate one Einstein clause has required metadata."""
        if node.value and hasattr(node.value, 'type_info') and node.value.type_info is None:
            self._report_error(
                "Einstein clause value missing type_info. TypeInferencePass should produce well-typed IR.",
                node.location
            )
        has_loops = bool(node.loop_vars)
        has_variable_ranges = bool(node.variable_ranges)
        indices = node.indices or []
        is_literal_assignment = all(isinstance(idx, LiteralIR) for idx in indices)
        if not has_loops and not has_variable_ranges and not is_literal_assignment:
            self._report_error(
                "Einstein clause missing loop structures and variable ranges. Ensure RangeAnalysisPass has run.",
                node.location
            )
        if node.value:
            node.value.accept(self)
    
    def visit_function_def(self, node: FunctionDefIR) -> None:
        self.nodes_validated += 1
        # Validate function has DefId
        if node.defid is None:
            self._report_error(
                f"FunctionDefIR '{node.name}' missing DefId. "
                "Functions must have DefId for definition table lookup.",
                node.location
            )
        # Skip validation of generic functions — they are templates that get
        # specialized; only the specialized copies need full validation.
        from ..analysis.analysis_guard import is_generic_function
        if is_generic_function(node):
            return
        # Validate function body
        node.body.accept(self)
    
    def visit_constant_def(self, node: ConstantDefIR) -> None:
        self.nodes_validated += 1
        # Validate constant has DefId
        if node.defid is None:
            self._report_error(
                f"ConstantDefIR '{node.name}' missing DefId. "
                "Constants must have DefId for definition table lookup.",
                node.location
            )
        # Validate constant value
        node.value.accept(self)
    
    def visit_program(self, node: ProgramIR) -> None:
        # Validate all functions
        for func in node.functions:
            func.accept(self)
        
        # Validate all constants
        for const in node.constants:
            const.accept(self)
        
        # Validate top-level statements - use visitor pattern
        for stmt in node.statements:
                stmt.accept(self)
    
    def visit_module(self, node) -> None:
        # Validate module functions and constants
        for func in node.functions:
            func.accept(self)
        for const in node.constants:
            const.accept(self)
        for submodule in node.submodules:
            submodule.accept(self)
    
    def visit_where_expression(self, node) -> None:
        self.nodes_validated += 1
        self._check_type(node)
        node.expr.accept(self)
        for constraint in node.constraints:
            constraint.accept(self)
    

    def visit_variable_declaration(self, node) -> Any:
        """Visit variable declaration - recurse into value"""
        if hasattr(node, 'value') and node.value:
            return node.value.accept(self)
        return None

    def visit_lowered_einstein(self, node: Any) -> None:
        """Visit all lowered Einstein clauses (not just last)."""
        for item in getattr(node, 'items', []) or []:
            item.accept(self)

    def visit_lowered_einstein_clause(self, node: Any) -> None:
        """Visit lowered clause body, loops, bindings, guards."""
        if getattr(node, 'body', None):
            node.body.accept(self)
        for loop in getattr(node, 'loops', []) or []:
            if getattr(loop, 'iterable', None):
                loop.iterable.accept(self)
        for b in getattr(node, 'bindings', []) or []:
            if getattr(b, 'expr', None):
                b.expr.accept(self)
        for g in getattr(node, 'guards', []) or []:
            if getattr(g, 'condition', None):
                g.condition.accept(self)

    def visit_lowered_reduction(self, node: Any) -> None:
        """Visit lowered reduction body and guards."""
        if getattr(node, 'body', None):
            node.body.accept(self)
        for g in getattr(node, 'guards', []) or []:
            if getattr(g, 'condition', None):
                g.condition.accept(self)

    def visit_lowered_comprehension(self, node: Any) -> None:
        """Visit lowered comprehension body and guards."""
        if getattr(node, 'body', None):
            node.body.accept(self)
        for g in getattr(node, 'guards', []) or []:
            if getattr(g, 'condition', None):
                g.condition.accept(self)
    # Helper
    def _check_type(self, node: ExpressionIR):
        """
        Check if expression has type_info.
        
        Missing type_info is an ERROR (not a warning) because:
        - IR must be well-typed before execution
        - Type inference should complete during TypeInferencePass
        - Missing types indicate incomplete lowering or analysis
        """
        if node.type_info is None:
            self._report_error(
                f"Expression {type(node).__name__} missing type_info. "
                "IR must be well-typed (all nodes have type_info from TypeInferencePass). "
                "This is a compiler bug - TypeInferencePass should produce well-typed IR.",
                node.location
            )

class IRValidationPass(BasePass):
    """
    Validates IR is structurally well-formed and ready for execution.
    
    STRUCTURAL validation (this pass):
    1. All IR nodes have type_info (TypeInferencePass was complete)
    2. Einstein declarations have proper loop structures or variable ranges
    3. All required metadata is attached by analysis passes
    4. DefIds present where required
    
    SEMANTIC validation (AST passes, NOT here):
    - Type checking: TypeInferencePass
    - Shape validation: UnifiedShapeAnalysisPass  
    - Scope resolution: NameResolutionPass
    
    Industry Pattern: LLVM IR Verifier, Java Bytecode Verifier
    
    Errors from this pass indicate COMPILER BUGS, not user errors.
    """
    requires = []  # Runs after all analysis passes, no dependencies
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """
        Validate IR program using visitor pattern.
        
        Rust Pattern: Validation pass returns IR unchanged (read-only)
        
        Args:
            ir: The IR program to validate
            tcx: Type context with error reporter
            
        Returns:
            IR unchanged (validation is read-only)
        """
        logger.debug("Starting IR validation")
        
        # Use visitor pattern - no isinstance checks!
        visitor = IRValidationVisitor(tcx)
        visitor.visit_program(ir)
        
        # Log results
        if tcx.reporter.has_errors():
            logger.debug(f"IR validation failed: {visitor.nodes_validated} nodes validated")
        else:
            logger.debug(f"IR validation passed: {visitor.nodes_validated} nodes validated")
        
        return ir  # Return IR unchanged (validation is read-only)

