"""
Range Analysis Pass

Rust Pattern: Range Inference, SCEV-like Analysis
Reference: RANGE_ANALYSIS_DESIGN.md
"""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple, Any
from ..passes.base import BasePass, TyCtxt
from ..passes.shape_analysis import UnifiedShapeAnalysisPass
from ..ir.nodes import (
    ProgramIR, ExpressionIR, BinaryOpIR, IdentifierIR, IndexVarIR, IndexRestIR,
    ReductionExpressionIR, WhereClauseIR, RangeIR, IRVisitor, LiteralIR,
    EinsteinDeclarationIR, FunctionDefIR, ParameterIR, IfExpressionIR,
    RectangularAccessIR, MemberAccessIR, TupleAccessIR, BuiltinCallIR,
    VariableDeclarationIR,
)
from ..ir.scoped_visitor import ScopedIRVisitor
from ..shared.defid import DefId, fixed_builtin_defid
from ..shared.source_location import SourceLocation
from ..shared.types import BinaryOp, infer_literal_type, UNKNOWN
from .visitor_helpers import ConstantEvaluator, VariableExtractor
from .implicit_range_detector import ImplicitRangeDetector

logger = logging.getLogger(__name__)


class Range:
    """Range representation"""
    __slots__ = ('start', 'end')
    
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
    
    @property
    def size(self) -> int:
        """Get range size"""
        return max(0, self.end - self.start)
    
    def __repr__(self) -> str:
        return f"Range({self.start}, {self.end})"

class RangeAnalysisPass(BasePass):
    """
    Range analysis pass.
    
    Infers ranges for index variables in Einstein notation from multiple
    constraint sources.
    
    Runs BEFORE shape analysis (shape analysis needs ranges for offset calculation)
    """
    requires = []  # No dependency on shape analysis (shape depends on range, not the other way)
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """Analyze ranges in IR"""
        analyzer = RangeAnalyzer(tcx)
        analyzer.program_ir = ir  # Store program IR in analyzer for visitor access
        setattr(tcx, 'program_ir', ir)
        
        # Analyze ranges in all expressions
        visitor = RangeAnalysisVisitor(analyzer)
        
        # Process all functions (visit FunctionDefIR, not just body)
        # This ensures visit_function_def is called and scopes are set up
        for func in ir.functions:
            func.accept(visitor)  # aligned: visit function node itself
        
        # CRITICAL: Also process stdlib module functions (they're in ir.modules, not ir.functions)
        for module in ir.modules:
            for func in module.functions:
                func.accept(visitor)
        
        # Process all statements (including Einstein declarations)
        for stmt in ir.statements:
            stmt.accept(visitor)
        
        tcx.set_analysis(RangeAnalysisPass, analyzer)
        return ir

    def process_specialized_functions(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """
        Run range analysis on specialized function bodies (incremental specialization).
        Type pass calls this after adding specialized functions so range handles its part.
        """
        specialized_funcs = getattr(tcx, 'specialized_functions', [])
        if not specialized_funcs:
            return ir
        try:
            analyzer = tcx.get_analysis(RangeAnalysisPass)
        except RuntimeError:
            return ir
        analyzer.program_ir = ir
        visitor = RangeAnalysisVisitor(analyzer)
        for func in specialized_funcs:
            try:
                logger.debug(f"[RangeAnalysis] Processing specialized function {func.name}")
                func.accept(visitor)
            except Exception as e:
                logger.warning(f"Range analysis failed for specialized {func.name}: {e}")
        return ir

def _range_ir_from_literal_bounds(start: int, end: int, location: Any) -> RangeIR:
    loc = location if hasattr(location, "file") else SourceLocation("", 0, 0)
    return RangeIR(
        start=LiteralIR(value=start, location=loc, type_info=infer_literal_type(start)),
        end=LiteralIR(value=end, location=loc, type_info=infer_literal_type(end)),
        location=loc,
        type_info=UNKNOWN,
    )


def _to_range_ir(rng: Any, loc: Any) -> Optional[RangeIR]:
    if rng is None:
        return None
    if isinstance(rng, RangeIR):
        return rng
    if isinstance(rng, range):
        return _range_ir_from_literal_bounds(rng.start, rng.stop, loc)
    return None


class RangeAnalyzer:
    """Range analyzer - infers ranges for index variables. Keys are DefId only. Stores RangeIR only."""

    def __init__(self, tcx: TyCtxt):
        self.tcx = tcx
        self.ranges: Dict[DefId, List[RangeIR]] = {}

    def add_range(self, defid: DefId, range_ir: RangeIR) -> None:
        if defid is None:
            raise ValueError("add_range: defid must not be None.")
        if defid not in self.ranges:
            self.ranges[defid] = []
        self.ranges[defid].append(range_ir)

    def set_range(self, defid: DefId, range_ir: RangeIR) -> None:
        if defid is None:
            raise ValueError("set_range: defid must not be None.")
        self.ranges[defid] = [range_ir]

    def get_range(self, defid: Optional[DefId]) -> Optional[RangeIR]:
        if defid is None:
            return None
        if defid not in self.ranges or not self.ranges[defid]:
            raise ValueError(
                f"DefId {defid.krate}:{defid.index} not found in range analyzer. "
                "Ensure RangeAnalysisPass has run and variable_ranges or set_range registered this defid."
            )
        return self._intersect_ranges_ir(self.ranges[defid])

    def _intersect_ranges_ir(self, range_irs: List[RangeIR]) -> Optional[RangeIR]:
        """Intersection of ranges: [max(starts), min(ends)] via TSC max/min. Not union."""
        if not range_irs:
            return None
        if len(range_irs) == 1:
            return range_irs[0]
        loc = getattr(range_irs[0], "location", None) or SourceLocation("", 0, 0)
        start_exprs = [r.start for r in range_irs]
        end_exprs = [r.end for r in range_irs]
        start_ir = BuiltinCallIR("max", start_exprs, loc, type_info=UNKNOWN, defid=fixed_builtin_defid("max")) if len(start_exprs) > 1 else start_exprs[0]
        end_ir = BuiltinCallIR("min", end_exprs, loc, type_info=UNKNOWN, defid=fixed_builtin_defid("min")) if len(end_exprs) > 1 else end_exprs[0]
        return RangeIR(start=start_ir, end=end_ir, location=loc, type_info=UNKNOWN)
    
    def infer_range_from_constraints(self, target_defid: DefId, constraints: List[ExpressionIR]) -> Optional[Range]:
        """Infer range from relational constraints. Uses DefId only."""
        from ..passes.constraint_classifier import ConstraintClassifierPass, ConstraintType

        try:
            classifications = self.tcx.get_analysis(ConstraintClassifierPass)
        except RuntimeError:
            classifications = {}

        lower_bounds = []
        upper_bounds = []

        for constraint in constraints:
            constraint_type = classifications.get(constraint, None)

            if constraint_type == ConstraintType.INDEX_RELATIONAL:
                if isinstance(constraint, BinaryOpIR):
                    if self._involves_variable(constraint, target_defid):
                        bound = self._extract_bound(constraint, target_defid)
                        if bound is not None:
                            is_lower, value = bound
                            if is_lower:
                                lower_bounds.append(value)
                            else:
                                upper_bounds.append(value)
        
        # Merge bounds
        start = max(lower_bounds) if lower_bounds else None
        end = min(upper_bounds) if upper_bounds else None
        
        if start is not None and end is not None and start < end:
            return Range(start, end)
        
        return None
    
    def _involves_variable(self, expr: ExpressionIR, target_defid: DefId) -> bool:
        """Check if expression involves variable by DefId."""
        return expr.accept(VariableInvolvementChecker(target_defid))

    def _extract_bound(self, constraint: BinaryOpIR, target_defid: DefId) -> Optional[Tuple[bool, int]]:
        """Extract bound from constraint (is_lower, value). Uses DefId only."""
        evaluator = ConstantEvaluator()
        left_defid = getattr(constraint.left, 'defid', None)
        right_defid = getattr(constraint.right, 'defid', None)

        if left_defid == target_defid:
            if constraint.operator == BinaryOp.LT:
                value = constraint.right.accept(evaluator)
                if value is not None:
                    return (False, value)
            elif constraint.operator == BinaryOp.LE:
                value = constraint.right.accept(evaluator)
                if value is not None:
                    return (False, value + 1)

        if right_defid == target_defid:
            if constraint.operator == BinaryOp.GT:
                value = constraint.left.accept(evaluator)
                if value is not None:
                    return (True, value + 1)
            elif constraint.operator == BinaryOp.GE:
                value = constraint.left.accept(evaluator)
                if value is not None:
                    return (True, value)

        return None
    
    def _evaluate_constant(self, expr: ExpressionIR) -> Optional[int]:
        """Evaluate constant expression to integer - visitor pattern"""
        return expr.accept(ConstantEvaluator())

class RangeAnalysisVisitor(ScopedIRVisitor[ParameterIR]):
    """
    Visitor to analyze ranges in IR (aligned with scoped visitor pattern).
    
    RangeAnalysisEngine extends ScopedASTVisitor, automatically
    tracks function parameters in scope.
    
    Alignment: Extends ScopedIRVisitor to get automatic parameter tracking.
    This eliminates manual var_definitions dict building.
    """
    
    def __init__(self, analyzer: RangeAnalyzer):
        super().__init__()
        self.analyzer = analyzer
        self._current_einstein_stack = []
        self._current_function_einstein_decls = []

    @contextmanager
    def _einstein_scope(self, node: EinsteinDeclarationIR):
        """Scope for processing an Einstein declaration; pops stack on exit."""
        self._current_einstein_stack.append(node)
        try:
            yield
        finally:
            if self._current_einstein_stack and self._current_einstein_stack[-1] == node:
                self._current_einstein_stack.pop()
    
    def visit_program(self, node: ProgramIR) -> None:
        """Visit program and analyze ranges in all statements and functions"""
        # Visit all statements
        for stmt in node.statements:
            stmt.accept(self)
        # Visit all functions
        for func in node.functions:
            func.accept(self)
        # Visit all constants
        for const in node.constants:
            const.accept(self)
    
    def visit_range(self, expr: RangeIR) -> None:
        """Extract range from RangeIR"""
        # Evaluate start and end
        start = self.analyzer._evaluate_constant(expr.start)
        end = self.analyzer._evaluate_constant(expr.end)
        
        if start is not None and end is not None:
            # Range is used in comprehensions, not directly stored
            pass
    
    def visit_reduction_expression(self, expr: ReductionExpressionIR) -> None:
        from ..ir.nodes import RangeIR, LiteralIR
        import logging
        logger = logging.getLogger(__name__)
        if expr.body:
            expr.body.accept(self)
        if expr.where_clause:
            for constraint in expr.where_clause.constraints:
                constraint.accept(self)
        program = getattr(self.analyzer, 'program_ir', None)

        scope_stack: List[Dict[DefId, Any]] = []
        if program:
            program_scope: Dict[DefId, Any] = {}
            for stmt in program.statements:
                if isinstance(stmt, VariableDeclarationIR) and getattr(stmt, 'defid', None) is not None:
                    program_scope[stmt.defid] = stmt.value
                    val_did = getattr(stmt.value, 'defid', None)
                    if val_did is not None and val_did != stmt.defid:
                        program_scope[val_did] = stmt.value
            scope_stack.append(program_scope)
        scope_stack.extend(self._scope_stack)
        detector = ImplicitRangeDetector(scope_stack, self.analyzer.tcx)
        detector._current_clause = None
        detector.infer_reduction_ranges_from_where(expr)
        from ..ir.nodes import EinsteinDeclarationIR, EinsteinIR, IdentifierIR, IndexVarIR
        for loop_var_ident in (expr.loop_vars or []):
            if not isinstance(loop_var_ident, (IdentifierIR, IndexVarIR)):
                continue
            loop_var_defid = getattr(loop_var_ident, 'defid', None)
            if loop_var_defid is None or loop_var_defid in expr.loop_var_ranges:
                continue
            implicit_range = detector.infer_implicit_range(expr.body, loop_var_defid)
            if implicit_range:
                if hasattr(implicit_range, 'to_range_ir') and callable(getattr(implicit_range, 'to_range_ir')):
                    range_ir = implicit_range.to_range_ir(expr.location)
                else:
                    from ..ir.nodes import LiteralIR
                    start_ir = LiteralIR(value=implicit_range.start, location=expr.location, type_info=infer_literal_type(implicit_range.start))
                    end_ir = LiteralIR(value=implicit_range.stop, location=expr.location, type_info=infer_literal_type(implicit_range.stop))
                    range_ir = RangeIR(start=start_ir, end=end_ir, location=expr.location, type_info=UNKNOWN)
                expr.loop_var_ranges[loop_var_defid] = range_ir
                continue
            range_ir = detector.infer_reduction_var_range_from_body(
                expr.body, loop_var_defid, expr.location
            )
            if range_ir:
                expr.loop_var_ranges[loop_var_defid] = range_ir

        for loop_var_ident in (expr.loop_vars or []):
            if not isinstance(loop_var_ident, (IdentifierIR, IndexVarIR)):
                continue
            loop_var_defid = getattr(loop_var_ident, 'defid', None)
            if loop_var_defid is None:
                continue
            if loop_var_defid in expr.loop_var_ranges:
                continue
            loc = getattr(expr, 'location', None) or getattr(loop_var_ident, 'location', None) or SourceLocation('', 0, 0)
            cause = detector.diagnose_reduction_range_failure(expr.body, loop_var_defid)
            display_name = getattr(loop_var_ident, 'name', None) or '?'
            tcx = getattr(self.analyzer, 'tcx', None)
            if tcx and getattr(tcx, 'reporter', None):
                tcx.reporter.report_error(
                    f"Range for reduction loop variable '{display_name}' (defid={loop_var_defid.krate}:{loop_var_defid.index}) could not be inferred. {cause}",
                    loc,
                    help="Ensure the reduction body uses the variable in an array index (e.g. A[i]) or add an explicit range (e.g. sum[i in 0..N](body)).",
                )
            else:
                raise ValueError(
                    f"Range for reduction loop variable '{display_name}' (defid={loop_var_defid.krate}:{loop_var_defid.index}) could not be inferred. {cause} "
                    "Ensure RangeAnalysisPass runs before EinsteinLoweringPass and that implicit_range_detector can infer the range."
                )

    # Required visitor methods (no-op for other nodes)
    def visit_literal(self, node) -> None:
        pass
    
    def visit_identifier(self, node) -> None:
        pass
    
    def visit_binary_op(self, node) -> None:
        from ..ir.nodes import ReductionExpressionIR as RedIR
        left = getattr(node, 'left', None)
        right = getattr(node, 'right', None)
        if left and hasattr(left, 'accept'):
            node.left.accept(self)
        if right and hasattr(right, 'accept'):
            node.right.accept(self)

    def visit_function_call(self, node) -> None:
        if hasattr(node, 'arguments') and node.arguments:
            for arg in node.arguments:
                if hasattr(arg, 'accept'):
                    arg.accept(self)

    def visit_unary_op(self, node) -> None:
        pass
    
    def visit_rectangular_access(self, node) -> None:
        pass
    
    def visit_jagged_access(self, node) -> None:
        pass
    
    def visit_block_expression(self, node) -> None:
        """Visit block expression - register all bindings in block then traverse (each stmt exposes get_defid_binding)."""
        stmts = getattr(node, 'statements', None) or []
        final = getattr(node, 'final_expr', None)
        for stmt in stmts:
            get_binding = getattr(stmt, 'get_defid_binding', None)
            if get_binding is not None:
                binding = get_binding()
                if binding is not None:
                    self.set_var(binding[0], binding[1])
        for stmt in stmts:
            if hasattr(stmt, 'accept'):
                stmt.accept(self)
        
        # Visit final expression
        if hasattr(node, 'final_expr') and node.final_expr and hasattr(node.final_expr, 'accept'):
            node.final_expr.accept(self)
    
    def visit_if_expression(self, node) -> None:
        if hasattr(node, 'condition') and node.condition and hasattr(node.condition, 'accept'):
            node.condition.accept(self)
        if hasattr(node, 'then_expr') and node.then_expr and hasattr(node.then_expr, 'accept'):
            node.then_expr.accept(self)
        if hasattr(node, 'else_expr') and node.else_expr and hasattr(node.else_expr, 'accept'):
            node.else_expr.accept(self)

    def visit_lambda(self, node) -> None:
        pass
    
    def visit_array_comprehension(self, node) -> None:
        if hasattr(node, 'body') and node.body and hasattr(node.body, 'accept'):
            node.body.accept(self)
        for r in getattr(node, 'ranges', []) or []:
            if hasattr(r, 'accept'):
                r.accept(self)
        for c in getattr(node, 'constraints', []) or []:
            if hasattr(c, 'accept'):
                c.accept(self)

    def visit_array_literal(self, node) -> None:
        pass
    
    def visit_tuple_expression(self, node) -> None:
        pass
    
    def visit_tuple_access(self, node) -> None:
        pass
    
    def visit_interpolated_string(self, node) -> None:
        pass
    
    def visit_cast_expression(self, node) -> None:
        """Visit cast expression - traverse into inner expression"""
        if hasattr(node, 'expr') and node.expr:
            node.expr.accept(self)
    
    def visit_member_access(self, node) -> None:
        pass
    
    def visit_try_expression(self, node) -> None:
        pass
    
    def visit_match_expression(self, node) -> None:
        pass
    
    def visit_where_expression(self, node) -> None:
        pass
    
    def visit_arrow_expression(self, node) -> None:
        pass
    
    def visit_pipeline_expression(self, node) -> None:
        pass
    
    def visit_builtin_call(self, node) -> None:
        if hasattr(node, 'arguments') and node.arguments:
            for arg in node.arguments:
                if hasattr(arg, 'accept'):
                    arg.accept(self)

    def visit_function_ref(self, node) -> None:
        pass
    
    def visit_einstein_declaration(self, node) -> None:
        """Analyze ranges for Einstein declaration variables. Process every clause."""
        clauses = getattr(node, 'clauses', None) or []
        if not clauses:
            return
        with self._einstein_scope(node):
            for clause in clauses:
                self._process_einstein_clause(node, clause)

    def _process_einstein_clause(self, declaration, clause) -> None:
        """Process one Einstein clause: set variable_ranges on the clause and register in analyzer."""
        from ..ir.nodes import EinsteinDeclarationIR, IdentifierIR, IndexVarIR, IndexRestIR, RectangularAccessIR, ArrayLiteralIR, LiteralIR
        import numpy as np
        node = declaration
        # variable_ranges is keyed by DefId (index variable identity), not by name
        existing_ranges = getattr(clause, 'variable_ranges', {}) or {}
        variable_ranges = {}  # DefId -> range or RangeIR

        if clause.value:
            clause.value.accept(self)

        program = getattr(self.analyzer, 'program_ir', None)
        def _defid_key(d):
            if d is None:
                return None
            if isinstance(d, DefId):
                return d
            if hasattr(d, 'krate') and hasattr(d, 'index'):
                return DefId(d.krate, d.index)
            if hasattr(d, '__getitem__') and len(d) >= 2:
                return DefId(d[0], d[1])
            return d
        clause_scope_stack: List[Dict[DefId, Any]] = []
        if program:
            from ..ir.nodes import VariableDeclarationIR
            program_scope: Dict[DefId, Any] = {}
            for stmt in program.statements:
                if isinstance(stmt, VariableDeclarationIR) and getattr(stmt, 'defid', None) is not None:
                    val = stmt.value
                    key = _defid_key(stmt.defid)
                    if key is not None:
                        program_scope[key] = val
                    if getattr(val, 'defid', None) is not None and val.defid != stmt.defid:
                        vkey = _defid_key(val.defid)
                        if vkey is not None:
                            program_scope[vkey] = val
            clause_scope_stack.append(program_scope)
        clause_scope_stack.extend(self._scope_stack)
        detector = ImplicitRangeDetector(clause_scope_stack, self.analyzer.tcx)
        prior_decls = []
        if hasattr(self, '_current_function_einstein_decls') and self._current_function_einstein_decls:
            try:
                current_index = self._current_function_einstein_decls.index(node)
                prior_decls = self._current_function_einstein_decls[:current_index]
            except ValueError:
                pass
        elif program:
            for stmt in program.statements:
                decl = stmt.value if isinstance(stmt, VariableDeclarationIR) and getattr(stmt, 'value', None) else stmt
                if decl is node:
                    break
                if isinstance(decl, EinsteinDeclarationIR):
                    prior_decls.append(decl)
        detector.set_prior_declarations(prior_decls)
        detector._current_clause = clause
        detector.set_current_declaration(node)

        from ..ir.nodes import ReductionExpressionIR as RedIR
        loop_var_ranges: Dict[DefId, RangeIR] = getattr(clause.value, 'loop_var_ranges', {}) if isinstance(clause.value, RedIR) else {}
        if not loop_var_ranges and clause.value:
            red = detector.find_reduction_in_value(clause.value)
            if red is not None:
                loop_var_ranges = getattr(red, 'loop_var_ranges', {}) or {}

        for idx_expr in (clause.indices or []):
                if not isinstance(idx_expr, (IdentifierIR, IndexVarIR, IndexRestIR)) or not getattr(idx_expr, "name", None):
                    continue
                defid = getattr(idx_expr, "defid", None)
                var_name = idx_expr.name
                if defid is None:
                    raise ValueError(
                        f"Index variable '{var_name}' must have defid in clause indices. "
                        "Ensure NameResolutionPass runs on AST before ASTToIRLoweringPass."
                    )
                # Prefer this clause's LHS range (real range for this clause) over existing_ranges.
                if isinstance(idx_expr, IndexVarIR) and getattr(idx_expr, "range_ir", None) is not None:
                    variable_ranges[defid] = idx_expr.range_ir
                    continue
                if defid in existing_ranges:
                    variable_ranges[defid] = existing_ranges[defid]
                    continue
                # No LHS range: fall back to where clause if it has explicit "var in start..end" (not relational guards).
                range_obj = detector.get_range_from_where_clause(clause.where_clause, defid=defid)
                if range_obj is not None:
                    rir = _to_range_ir(range_obj, clause.location)
                    if rir is not None:
                        variable_ranges[defid] = rir
                    else:
                        variable_ranges[defid] = range_obj
                    continue
                
                # Use implicit range detector 
                range_obj = None
                if range_obj is None:
                    implicit_range = detector.infer_implicit_range(clause.value, defid)
                    if implicit_range:
                        # Convert RangeInfo to RangeIR using built-in method
                        from ..passes.range_info import StaticRange, DynamicRange
                        range_obj = implicit_range.to_range_ir(clause.location)
                
                if range_obj is None:
                    range_ir_det = detector.infer_clause_index_range_from_reduction_body(clause, defid, getattr(node, "location", clause.location))
                    if range_ir_det is not None:
                        range_obj = range_ir_det
                if range_obj:
                    clause_ranges = getattr(clause, 'variable_ranges', None) or {}
                    if defid in clause_ranges and isinstance(clause_ranges[defid], RangeIR):
                        variable_ranges[defid] = clause_ranges[defid]
                    elif defid in existing_ranges and isinstance(existing_ranges[defid], RangeIR):
                        variable_ranges[defid] = existing_ranges[defid]
                    else:
                        rir = _to_range_ir(range_obj, getattr(node, "location", clause.location))
                        variable_ranges[defid] = rir if rir is not None else range_obj
                else:
                    # Don't report error yet - try to infer from program statements
                    range_obj = detector.infer_range_from_program_statements(self.analyzer.tcx, defid, node, clause.indices)
                    if range_obj:
                        rir = _to_range_ir(range_obj, getattr(node, "location", clause.location))
                        variable_ranges[defid] = rir if rir is not None else range_obj
                    else:
                        # Check if this variable is part of a reduction expression
                        from ..ir.nodes import ReductionExpressionIR
                        if isinstance(clause.value, ReductionExpressionIR):
                            red_expr = clause.value
                            if defid in loop_var_ranges:
                                variable_ranges[defid] = loop_var_ranges[defid]
                                continue
                        
                        # Check if the variable appears in array accesses in the value expression
                        # If so, its range can be inferred at runtime from the array shape
                        # Don't report a compile-time error - allow runtime inference
                        from ..ir.nodes import RectangularAccessIR, IdentifierIR, IndexVarIR
                        def has_array_access_with_defid(expr, target_defid: DefId):
                            """Check if expression contains array access using this variable (by DefId)."""
                            if isinstance(expr, RectangularAccessIR):
                                for idx in expr.indices:
                                    if isinstance(idx, (IdentifierIR, IndexVarIR)) and getattr(idx, 'defid', None) == target_defid:
                                        return True
                                    elif hasattr(idx, 'left') or hasattr(idx, 'right'):
                                        if hasattr(idx, 'left') and has_array_access_with_defid(idx.left, target_defid):
                                            return True
                                        if hasattr(idx, 'right') and has_array_access_with_defid(idx.right, target_defid):
                                            return True
                            elif hasattr(expr, 'left') and hasattr(expr, 'right'):
                                return (has_array_access_with_defid(expr.left, target_defid) or
                                       has_array_access_with_defid(expr.right, target_defid))
                            elif hasattr(expr, 'then_expr') and hasattr(expr, 'else_expr'):
                                result = (has_array_access_with_defid(expr.then_expr, target_defid) or
                                         has_array_access_with_defid(expr.else_expr, target_defid))
                                if hasattr(expr, 'condition'):
                                    result = result or has_array_access_with_defid(expr.condition, target_defid)
                                return result
                            elif hasattr(expr, 'arguments'):
                                from ..ir.nodes import FunctionCallIR
                                if isinstance(expr, FunctionCallIR):
                                    for arg in expr.arguments:
                                        if has_array_access_with_defid(arg, target_defid):
                                            return True
                            elif hasattr(expr, 'body'):
                                return has_array_access_with_defid(expr.body, target_defid)
                            return False
                        
                        if has_array_access_with_defid(clause.value, defid):
                            clause_ranges = getattr(clause, 'variable_ranges', None) or {}
                            if defid in clause_ranges and isinstance(clause_ranges[defid], RangeIR):
                                variable_ranges[defid] = clause_ranges[defid]
                            elif defid in existing_ranges and isinstance(existing_ranges[defid], RangeIR):
                                variable_ranges[defid] = existing_ranges[defid]
                            elif defid not in variable_ranges:
                                range_ir_red = detector.infer_clause_index_range_from_reduction_body(clause, defid, getattr(node, "location", clause.location))
                                if range_ir_red is not None:
                                    variable_ranges[defid] = range_ir_red
                            if defid not in variable_ranges:
                                implicit_range = detector.infer_implicit_range(clause.value, defid)
                                if implicit_range:
                                    if hasattr(implicit_range, 'to_range_ir') and callable(implicit_range.to_range_ir):
                                        range_ir = implicit_range.to_range_ir(getattr(node, 'location', clause.location))
                                        if range_ir is not None:
                                            variable_ranges[defid] = range_ir
                                    else:
                                        from ..ir.nodes import LiteralIR
                                        start_ir = LiteralIR(value=implicit_range.start, location=node.location, type_info=infer_literal_type(implicit_range.start))
                                        end_ir = LiteralIR(value=implicit_range.stop, location=node.location, type_info=infer_literal_type(implicit_range.stop))
                                        range_ir = RangeIR(start=start_ir, end=end_ir, location=node.location, type_info=UNKNOWN)
                                        variable_ranges[defid] = range_ir
                            if defid not in variable_ranges:
                                inferred_range = detector.infer_range_from_array_access(
                                    defid, clause.value, node,
                                    loop_var_ranges=loop_var_ranges,
                                    clause_indices=clause.indices,
                                    einstein_clause=clause,
                                    location=getattr(node, "location", clause.location),
                                )
                                if inferred_range:
                                    rir = _to_range_ir(inferred_range, getattr(node, "location", clause.location))
                                    if rir is not None:
                                        variable_ranges[defid] = rir
                                elif getattr(clause, 'variable_ranges', None) and defid in clause.variable_ranges:
                                    variable_ranges[defid] = clause.variable_ranges[defid]
                            continue
                        
                        # STDLIB: Don't report error for stdlib functions without type annotations
                        # Runtime range inference will handle these cases. Still register defid so get_range() does not raise.
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.debug(f"Could not determine compile-time range for defid {defid} in {node.name}, will use runtime inference")
                        inferred_range = detector.infer_range_from_array_access(
                            defid, clause.value, node,
                            loop_var_ranges=loop_var_ranges,
                            clause_indices=clause.indices,
                            einstein_clause=clause,
                            location=getattr(node, "location", clause.location),
                        )
                        if inferred_range:
                            rir = _to_range_ir(inferred_range, getattr(node, "location", clause.location))
                            if rir is not None:
                                variable_ranges[defid] = rir
                        if defid not in variable_ranges:
                            loc = getattr(node, "location", clause.location) or SourceLocation("", 0, 0)
                            tcx = getattr(self.analyzer, 'tcx', None)
                            if tcx and getattr(tcx, 'reporter', None):
                                tcx.reporter.report_error(
                                    f"Range for index variable '{getattr(idx_expr, 'name', None) or '?'}' (defid={defid.krate}:{defid.index}) could not be inferred.",
                                    loc,
                                    help="Ensure the RHS uses the variable in an array index or add an explicit range in the where clause (e.g. var in 0..N).",
                                )
                            else:
                                raise ValueError(
                                    f"Range for index variable '{getattr(idx_expr, 'name', None) or '?'}' (defid={defid.krate}:{defid.index}) could not be inferred. "
                                    "Ensure RangeAnalysisPass runs and implicit_range_detector can infer the range."
                                )

        existing_on_clause = getattr(clause, 'variable_ranges', None) or {}
        for defid_existing, rng in existing_on_clause.items():
            if defid_existing not in variable_ranges:
                variable_ranges[defid_existing] = rng
            elif isinstance(rng, RangeIR):
                variable_ranges[defid_existing] = rng
        for red_defid, red_rng in loop_var_ranges.items():
            if red_defid is None:
                raise ValueError(
                    "loop_var_ranges must not have None defid; reduction loop var must have defid from name resolution."
                )
            if red_defid not in variable_ranges:
                variable_ranges[red_defid] = red_rng
        object.__setattr__(clause, 'variable_ranges', variable_ranges if variable_ranges else {})

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
    
    def visit_function_def(self, node: FunctionDefIR) -> None:
        """
        Visit function definition - aligned pattern.
        
        Pre-populate parameters in _var_definitions before analyzing body.
        Pattern: Use ScopedIRVisitor to automatically track parameters in scope.
        
        This ensures ImplicitRangeDetector can find function parameters when
        inferring ranges for Einstein declarations.
        
        ALIGNED SEQUENTIAL PROCESSING: Collect all Einstein declarations
        and analyze them in order, allowing later declarations to reference earlier ones.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Skip generic functions (no type annotations); callsite uses specialized.
        from ..analysis.analysis_guard import should_analyze_function
        tcx = getattr(self.analyzer, 'tcx', None)
        do_analyze = should_analyze_function(node, tcx=tcx)
        if not do_analyze:
            logger.debug(f"Skipping generic function {node.name} (will be monomorphized)")
            return

        logger.debug(f"[RangeAnalysis] Visiting function {node.name}, parameters: {[p.name for p in node.parameters]}")
        # ALIGNED: Collect all Einstein declarations in function body (in order)
        from ..ir.nodes import EinsteinDeclarationIR
        einstein_decls = []
        if hasattr(node, 'body') and node.body and hasattr(node.body, 'statements'):
            for stmt in node.body.statements:
                if isinstance(stmt, EinsteinDeclarationIR):
                    einstein_decls.append(stmt)
        
        self._current_function_einstein_decls = einstein_decls
        self._current_function = node

        with self.scope():
            for param in node.parameters:
                did = getattr(param, 'defid', None)
                if did is None:
                    raise RuntimeError(
                        f"Parameter '{getattr(param, 'name', None)}' has no defid in function '{getattr(node, 'name', None)}'. "
                        "Ensure name resolution runs before range analysis and assigns defids to parameters."
                    )
                self.set_var(did, param)
                logger.debug(f"[RangeAnalysis] Pre-registered parameter '{param.name}' in scope")
            if hasattr(node, 'body') and node.body:
                node.body.accept(self)
        
        self._current_function_einstein_decls = []
        self._current_function = None

    def visit_constant_def(self, node) -> None:
        pass
    
    def visit_module(self, node) -> None:
        pass

    def visit_variable_declaration(self, node) -> None:
        """Visit variable declaration - register in scope (by defid) then recurse into value"""
        did = getattr(node, 'defid', None)
        if did is None:
            raise RuntimeError(
                f"Variable declaration (pattern={getattr(node, 'pattern', None)}) has no defid. "
                "Ensure name resolution runs before range analysis and assigns defids to let-bindings."
            )
        self.set_var(did, getattr(node, 'value', node))
        if hasattr(node, 'value') and node.value:
            node.value.accept(self)

class VariableInvolvementChecker(IRVisitor[bool]):
    """Check if expression involves a variable by DefId."""

    def __init__(self, target_defid: DefId):
        self.target_defid = target_defid

    def visit_identifier(self, expr: IdentifierIR) -> bool:
        return getattr(expr, 'defid', None) == self.target_defid

    def visit_index_var(self, expr) -> bool:
        return getattr(expr, 'defid', None) == self.target_defid
    
    def visit_binary_op(self, expr: BinaryOpIR) -> bool:
        return (expr.left.accept(self) or expr.right.accept(self))
    
    # Default: False
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
    
    def visit_program(self, node) -> bool:
        return False
    
    # Additional stubs for abstract methods
    def visit_if_expression(self, node) -> bool:
        return False
    
    def visit_lambda(self, node) -> bool:
        return False
    
    def visit_range(self, node) -> bool:
        return False
    
    def visit_array_comprehension(self, node) -> bool:
        return False
    
    def visit_module(self, node) -> bool:
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
    
    def visit_einstein_declaration(self, node) -> bool:
        return False
    
    def visit_variable_declaration(self, node) -> bool:
        return False
    
    def visit_constant_def(self, node) -> bool:
        return False
    
    def visit_function_def(self, node) -> bool:
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

    def visit_variable_declaration(self, node) -> Any:
        """Visit variable declaration - recurse into value"""
        if hasattr(node, 'value') and node.value:
            return node.value.accept(self)
        return None

