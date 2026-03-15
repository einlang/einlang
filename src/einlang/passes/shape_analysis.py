"""
Shape Analysis Pass

Rust Pattern: Tensor Shape Inference, Coverage Analysis
Reference: SHAPE_ANALYSIS_DESIGN.md
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from ..passes.base import BasePass, TyCtxt
from ..passes.rest_pattern_preprocessing import RestPatternPreprocessingPass
from ..ir.nodes import (
    ProgramIR, ExpressionIR, ArrayLiteralIR, ArrayComprehensionIR,
    FunctionCallIR, BindingIR, is_einstein_binding, is_function_binding, RectangularAccessIR,
    IRVisitor, RangeIR, IdentifierIR, LiteralIR, MemberAccessIR, BinaryOpIR, DifferentialIR,
)
from ..shared.defid import DefId
from ..shared.source_location import SourceLocation
from ..shared.types import BinaryOp
from .visitor_helpers import ConstantEvaluator, ArrayAccessCollector, ExprInvolvesVarVisitor

logger = logging.getLogger(__name__)


def _none_stride_offset() -> Tuple[Optional[int], Optional[int]]:
    return (None, None)


class _StrideOffsetExtractor(IRVisitor[Tuple[Optional[int], Optional[int]]]):
    """Extract (stride, max_offset) from index expressions like i, i*2, i+1, i*2+di. Uses variable_ranges for offset vars."""

    def __init__(
        self,
        index_var: str,
        variable_ranges: Optional[Dict[DefId, Any]] = None,
    ) -> None:
        self._index_var = index_var
        self._variable_ranges = variable_ranges or {}

    def visit_identifier(self, node: IdentifierIR) -> Tuple[Optional[int], Optional[int]]:
        if node.name == self._index_var:
            return (None, 0)
        return _none_stride_offset()

    def visit_literal(self, node: LiteralIR) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_binary_op(self, node: BinaryOpIR) -> Tuple[Optional[int], Optional[int]]:
        from ..ir.nodes import LiteralIR as LiteralIRNode
        left, right = node.left, node.right
        if node.operator == BinaryOp.MUL:
            stride = None
            if isinstance(left, IdentifierIR) and left.name == self._index_var and isinstance(right, LiteralIRNode):
                stride = int(right.value)
            elif isinstance(right, IdentifierIR) and right.name == self._index_var and isinstance(left, LiteralIRNode):
                stride = int(left.value)
            if stride:
                logger.debug(f"[ShapeAnalysis] Found stride pattern: {self._index_var}*{stride}")
                return (stride, 0)
        if node.operator == BinaryOp.SUB:
            if isinstance(left, IdentifierIR) and left.name == self._index_var and isinstance(right, IdentifierIR):
                logger.debug(f"[ShapeAnalysis] Subtraction pattern: {self._index_var}-{right.name}, using offset=0")
                return (None, 0)
        if node.operator == BinaryOp.ADD:
            stride = None
            if isinstance(left, BinaryOpIR) and left.operator == BinaryOp.MUL:
                if isinstance(left.left, IdentifierIR) and left.left.name == self._index_var and isinstance(left.right, LiteralIRNode):
                    stride = int(left.right.value)
                elif isinstance(left.right, IdentifierIR) and left.right.name == self._index_var and isinstance(left.left, LiteralIRNode):
                    stride = int(left.left.value)
                if stride:
                    if isinstance(right, IdentifierIR):
                        max_off = _range_end_to_int(self._variable_ranges.get(right.defid) if right.defid else None)
                        if max_off is not None:
                            return (stride, max_off)
                    elif isinstance(right, LiteralIRNode):
                        return (stride, int(right.value))
            if isinstance(left, IdentifierIR) and left.name == self._index_var and isinstance(right, LiteralIRNode):
                return (None, int(right.value))
            if isinstance(right, IdentifierIR) and right.name == self._index_var and isinstance(left, LiteralIRNode):
                return (None, int(left.value))
            if isinstance(left, IdentifierIR) and left.name == self._index_var and isinstance(right, IdentifierIR):
                max_off = _range_end_to_int(self._variable_ranges.get(right.defid) if right.defid else None)
                if max_off is not None:
                    return (None, max_off)
            if isinstance(right, IdentifierIR) and right.name == self._index_var and isinstance(left, IdentifierIR):
                max_off = _range_end_to_int(self._variable_ranges.get(left.defid) if left.defid else None)
                if max_off is not None:
                    return (None, max_off)
        return _none_stride_offset()

    def visit_index_var(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_index_rest(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_rectangular_access(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_jagged_access(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_function_call(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_unary_op(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_block_expression(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_if_expression(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_lambda(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_range(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_array_comprehension(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_module(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_array_literal(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_tuple_expression(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_tuple_access(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_interpolated_string(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_cast_expression(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_member_access(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_try_expression(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_match_expression(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_reduction_expression(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_where_expression(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_pipeline_expression(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_builtin_call(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_literal_pattern(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_identifier_pattern(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_wildcard_pattern(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_tuple_pattern(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_array_pattern(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_rest_pattern(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_guard_pattern(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_binding(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()

    def visit_program(self, node: Any) -> Tuple[Optional[int], Optional[int]]:
        return _none_stride_offset()


def _range_end_to_int(range_obj: Any) -> Optional[int]:
    """Return (end - 1) as int when range has literal end (max offset for exclusive range)."""
    if range_obj is None:
        return None
    end = range_obj.end
    if end is None:
        if isinstance(range_obj, range):
            return range_obj.stop - 1
        return None
    if isinstance(end, LiteralIR) and isinstance(end.value, (int, float)):
        return int(end.value) - 1
    return None


def _range_bound_to_int(range_obj: Any, bound: str) -> Optional[int]:
    """Return start or end as int. bound is 'start' or 'end'. For range() use start/stop."""
    if range_obj is None:
        return None
    if isinstance(range_obj, range):
        return range_obj.start if bound == 'start' else range_obj.stop
    attr = range_obj.start if bound == 'start' else range_obj.end
    if attr is None:
        return None
    if isinstance(attr, LiteralIR) and isinstance(attr.value, (int, float)):
        return int(attr.value)
    return None


class UnifiedShapeAnalysisPass(BasePass):
    """
    Unified shape analysis pass.
    
    Combines shape resolution and coverage analysis for Einstein notation.
    
    Runs AFTER range analysis (needs ranges to compute output dims with offsets)
    """
    requires = [RestPatternPreprocessingPass]  # RestPattern → Range → Shape
    # Note: RangeAnalysisPass dependency is implicit via pass ordering (not declared to avoid import cycles)
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """Analyze shapes in IR"""
        try:
            analyzer = ShapeAnalyzer(tcx)
            
            # Analyze shapes in all expressions
            visitor = ShapeAnalysisVisitor(analyzer)
            
            # Process all functions
            for func in ir.functions:
                func.body.accept(visitor)
            
            # Process all statements
            for stmt_idx, stmt in enumerate(ir.statements):
                try:
                    stmt.accept(visitor)
                except Exception as e:
                    raise

            # : Propagate shape to literal-index Einstein decls from their group
            self._propagate_shape_to_literal_index_decls(ir, tcx, analyzer)

            # defid_to_shape: Variables/Einstein use defid for lookup
            # expr_shapes still captures shapes from analyzer traversal
            defid_to_shape = {}

            # Store both ExpressionIR -> shape and DefId -> shape mappings
            # CRITICAL: Convert all shapes in analyzer.shapes to tuples for hashability
            expr_shapes = {}
            for expr, shape in analyzer.shapes.items():
                if isinstance(shape, list):
                    expr_shapes[expr] = tuple(shape)
                else:
                    expr_shapes[expr] = shape
            
            tcx.set_analysis(UnifiedShapeAnalysisPass, {
                'expr_shapes': expr_shapes,  # ExpressionIR -> Tuple[int, ...]
                'defid_shapes': defid_to_shape   # DefId -> Tuple[int, ...] (for variables)
            })
            
            return ir
        except TypeError as e:
            if "unhashable type" in str(e):
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Unhashable type error in shape analysis: {e}\n{traceback.format_exc()}")
            raise

    def _propagate_shape_to_literal_index_decls(
        self, ir: ProgramIR, tcx: TyCtxt, analyzer: "ShapeAnalyzer"
    ) -> None:
        """Propagate shape from loop-bearing Einstein decls to literal-index decls in same group."""
        try:
            from .einstein_grouping import EinsteinDeclarationGroupingPass
            grouping = tcx.get_analysis(EinsteinDeclarationGroupingPass)
        except RuntimeError:
            return
        if not isinstance(grouping, dict):
            return
        for array_name, group in grouping.items():
            decls = group.declarations or []
            if not decls:
                continue
            # Collect shapes from decls that have them (loop-bearing)
            shapes_in_group = []
            decls_without_shape = []
            for decl in decls:
                shape = analyzer.get_shape(decl)
                if shape:
                    shapes_in_group.append(tuple(shape) if isinstance(shape, list) else shape)
                else:
                    decls_without_shape.append(decl)
            if not shapes_in_group or not decls_without_shape:
                continue
            # Use max shape (per dimension) so we cover all indices
            max_rank = max(len(s) for s in shapes_in_group)
            unified = []
            for i in range(max_rank):
                dims = [s[i] for s in shapes_in_group if i < len(s)]
                if dims:
                    unified.append(max(dims))
            if not unified:
                continue
            unified = tuple(unified)
            for decl in decls_without_shape:
                analyzer.set_shape(decl, unified)
                logger.debug(f"[ShapeAnalysis] Propagated shape {unified} to literal-index decl {decl.name or '?'}")

    def process_specialized_functions(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """
        Run shape analysis on specialized function bodies (incremental specialization).
        Type pass calls this after adding specialized functions so shape handles its part.
        """
        specialized_funcs = getattr(tcx, 'specialized_functions', [])
        if not specialized_funcs:
            return ir
        try:
            data = tcx.get_analysis(UnifiedShapeAnalysisPass)
            expr_shapes = dict(data.get('expr_shapes', {}))
            defid_to_shape = dict(data.get('defid_shapes', {}))
        except RuntimeError:
            return ir
        analyzer = ShapeAnalyzer(tcx)
        analyzer.shapes = expr_shapes
        analyzer.defid_to_shape = defid_to_shape
        visitor = ShapeAnalysisVisitor(analyzer)
        for func in specialized_funcs:
            try:
                logger.debug(f"[ShapeAnalysis] Processing specialized function {func.name}")
                if func.body:
                    func.body.accept(visitor)
            except Exception as e:
                logger.warning(f"Shape analysis failed for specialized {func.name}: {e}")
        # analyzer.shapes / defid_to_shape are the same dicts as expr_shapes / defid_to_shape (updated in place)
        tcx.set_analysis(UnifiedShapeAnalysisPass, {
            'expr_shapes': expr_shapes,
            'defid_shapes': defid_to_shape
        })
        return ir

class ShapeAnalyzer:
    """Shape analyzer - infers shapes for arrays and tensors"""
    
    def __init__(self, tcx: TyCtxt):
        self.tcx = tcx
        self.shapes: Dict[ExpressionIR, tuple] = {}  # Expression -> shape (tuple for hashability)
        self.defid_to_shape: Dict[DefId, tuple] = {}  # DefId -> shape (for variables)
    
    def set_shape(self, expr: ExpressionIR, shape) -> None:
        """Set shape for expression - shape should be tuple"""
        # Ensure shape is a tuple for hashability
        if isinstance(shape, list):
            shape = tuple(shape)
        elif shape is not None and not isinstance(shape, tuple):
            shape = tuple(shape)
        if shape is not None:
            self.shapes[expr] = shape
    
    def get_shape(self, expr: ExpressionIR) -> Optional[tuple]:
        """Get shape for expression - returns tuple"""
        # CRITICAL FIX: For IdentifierIR, look up shape by DefId
        # This is needed to find the shape of variables like "image"
        from ..ir.nodes import IdentifierIR
        if isinstance(expr, IdentifierIR) and expr.defid:
            return self.defid_to_shape.get(expr.defid, None)
        return self.shapes.get(expr, None)

    def _is_array_literal_element(self, elem: ExpressionIR) -> bool:
        """True if element is an array literal (nested array). ElementTypeChecker.visit_array_literal -> 'array'."""
        from ..ir.nodes import CastExpressionIR
        if isinstance(elem, ArrayLiteralIR):
            return True
        if isinstance(elem, CastExpressionIR):
            return isinstance(elem.expr, ArrayLiteralIR)
        return False

    def infer_array_literal_shape(self, expr: ArrayLiteralIR) -> Optional[tuple]:
        """Infer shape from array literal - returns tuple for hashability. Rejects mixed scalar/array ()."""
        if not expr.elements:
            return (0,)

        from ..ir.nodes import CastExpressionIR

        first_elem = expr.elements[0]
        first_is_array = self._is_array_literal_element(first_elem)
        loc = expr.location

        if first_is_array:
            # All elements must be arrays and have the same shape
            element_shape = self._infer_array_literal_shape_from_element(first_elem)
            for i, elem in enumerate(expr.elements[1:], start=1):
                if not self._is_array_literal_element(elem):
                    self.tcx.reporter.report_error(
                        "Array literal has inconsistent element types: "
                        "element 0 is array, but element {} is not an array".format(i),
                        location=(elem.location or loc) or loc,
                    )
                    return None
                elem_shape = self._infer_array_literal_shape_from_element(elem)
                if (
                    elem_shape is not None
                    and element_shape is not None
                    and elem_shape != element_shape
                ):
                    return None
            if element_shape is not None:
                return (len(expr.elements),) + (element_shape if isinstance(element_shape, tuple) else tuple(element_shape))
            return (len(expr.elements),)
        else:
            # All elements must be scalars (no nested array)
            for i, elem in enumerate(expr.elements[1:], start=1):
                if self._is_array_literal_element(elem):
                    self.tcx.reporter.report_error(
                        "Array literal has inconsistent element types: "
                        "element 0 is scalar, but element {} is array".format(i),
                        location=(elem.location or loc) or loc,
                    )
                    return None
            return (len(expr.elements),)

    def _infer_array_literal_shape_from_element(self, elem: ExpressionIR) -> Optional[tuple]:
        """Shape of a single element (for nested array: recursive shape; for scalar: None)."""
        if isinstance(elem, ArrayLiteralIR):
            return self.infer_array_literal_shape(elem)
        from ..ir.nodes import CastExpressionIR
        if isinstance(elem, CastExpressionIR):
            return self.get_shape(elem.expr) or (
                self.infer_array_literal_shape(elem.expr) if isinstance(elem.expr, ArrayLiteralIR) else None
            )
        return self.get_shape(elem)
    
    def infer_comprehension_shape(self, expr: ArrayComprehensionIR) -> Optional[tuple]:
        """Infer shape from array comprehension - returns tuple for hashability"""
        total_size = 1
        for range_expr in expr.ranges:
            range_size = self._get_range_size(range_expr)
            if range_size is None:
                return None
            total_size *= range_size
        return (total_size,) if total_size >= 1 else None
    
    def _get_range_size(self, range_expr: ExpressionIR) -> Optional[int]:
        """Get size of range expression. Only RangeIR has .start/.end; comprehension over collection (e.g. IdentifierIR) has no size here."""
        if not isinstance(range_expr, RangeIR):
            return None
        evaluator = ConstantEvaluator()
        start = range_expr.start.accept(evaluator)
        end = range_expr.end.accept(evaluator)
        if start is not None and end is not None:
            return max(0, end - start)
        return None
    
    def resolve_symbolic_shape(self, shape_expr: ExpressionIR) -> Optional[int]:
        """Resolve symbolic shape expression to concrete value"""
        evaluator = ConstantEvaluator()
        if isinstance(shape_expr, FunctionCallIR) and shape_expr.function_name == "shape" and len(shape_expr.arguments) >= 2:
            array_expr = shape_expr.arguments[0]
            dim_expr = shape_expr.arguments[1]
            
            array_shape = self.get_shape(array_expr)
            dim = dim_expr.accept(evaluator)
            
            if array_shape and dim is not None and 0 <= dim < len(array_shape):
                return array_shape[dim]
        return None

    def _evaluate_shape_dim_expr(self, expr: ExpressionIR) -> Optional[int]:
        """
        Evaluate a dependent-range end expression of the form array.shape[dim]
        (RectangularAccessIR over MemberAccessIR .shape with literal index).
        Returns the constant dimension size when shape is known, else None.
        """
        if not isinstance(expr, RectangularAccessIR):
            return None
        array_expr = expr.array
        indices = expr.indices or []
        if not isinstance(array_expr, MemberAccessIR) or array_expr.member != 'shape':
            return None
        arr = array_expr.object
        if not indices or not isinstance(indices[0], LiteralIR):
            return None
        dim_val = indices[0].value
        if not isinstance(dim_val, (int, float)):
            return None
        dim = int(dim_val)
        shape = self.get_shape(arr)
        if shape is None or not isinstance(shape, (list, tuple)) or dim < 0 or dim >= len(shape):
            return None
        size = shape[dim]
        if isinstance(size, (int, float)):
            return int(size)
        return None

    def _resolve_dependent_ranges_on_decl(self, decl: BindingIR) -> None:
        """
        Resolve dependent ranges on clauses when shape is known.
        Range pass leaves 0..array.shape[dim]; we replace end with LiteralIR(size) when we know the shape.
        Mutates clause.variable_ranges[defid].end in place.
        """
        from ..shared.types import infer_literal_type
        for clause in (decl.clauses or []):
            variable_ranges = clause.variable_ranges or {}
            for defid, rng in variable_ranges.items():
                if not isinstance(rng, RangeIR):
                    continue
                end_expr = rng.end
                if end_expr is None or isinstance(end_expr, LiteralIR):
                    continue
                resolved = self._evaluate_shape_dim_expr(end_expr)
                if resolved is not None:
                    loc = rng.location or decl.location
                    new_end = LiteralIR(
                        value=resolved,
                        location=loc or SourceLocation('', 0, 0),
                        type_info=infer_literal_type(resolved),
                    )
                    object.__setattr__(rng, 'end', new_end)
                    logger.debug("[ShapeAnalysis] Resolved dependent range end to %s", resolved)

    def infer_einstein_shape(self, decl: BindingIR) -> Optional[tuple]:
        """Infer shape as max of output indices of each clause (per dimension)."""
        from ..passes.range_info import StaticRange
        import logging
        logger = logging.getLogger(__name__)
        clauses = decl.clauses or []
        if not clauses:
            return None

        rank = 0
        for c in clauses:
            if c.indices:
                rank = len(c.indices)
                break
        if rank == 0:
            return None

        def shape_for_clause(clause) -> Tuple[Optional[int], ...]:
            value_expr = clause.value
            variable_ranges = clause.variable_ranges or {}
            shape: List[Optional[int]] = []
            for idx in (clause.indices or []):
                from ..ir.nodes import IndexVarIR
                if isinstance(idx, LiteralIR):
                    v = idx.value
                    try:
                        extent = int(v) + 1 if v is not None else 1
                    except (TypeError, ValueError):
                        shape.append(None)
                        continue
                    shape.append(max(1, extent))
                    continue
                index_var = idx.name
                defid = idx.defid
                if index_var is None:
                    continue
                range_obj = variable_ranges.get(defid) if defid else None
                has_explicit_range = range_obj is not None
                if has_explicit_range:
                    if isinstance(range_obj, StaticRange):
                        shape.append(range_obj.end)
                    elif isinstance(range_obj, range):
                        shape.append(range_obj.stop)
                    elif isinstance(range_obj, RangeIR) and range_obj.end is not None:
                        end_expr = range_obj.end
                        if isinstance(end_expr, LiteralIR) and isinstance(end_expr.value, (int, float)):
                            shape.append(int(end_expr.value))
                        else:
                            s = self._infer_shape_from_arrays(index_var, value_expr, variable_ranges)
                            shape.append(s)
                    else:
                        s = self._infer_shape_from_arrays(index_var, value_expr, variable_ranges)
                        shape.append(s)
                else:
                    s = self._infer_shape_from_arrays(index_var, value_expr, variable_ranges)
                    shape.append(s)
            while len(shape) < rank:
                shape.append(None)
            return tuple(shape[:rank])

        shapes: List[Tuple[Optional[int], ...]] = []
        for clause in clauses:
            s = shape_for_clause(clause)
            if len(s) != rank:
                logger.debug(f"[infer_einstein_shape] {decl.name} clause shape rank mismatch: {len(s)} vs {rank}")
                return None
            shapes.append(s)
            logger.debug(f"[infer_einstein_shape] {decl.name} clause shape: {s}")

        combined = []
        for d in range(rank):
            dim_vals = [s[d] for s in shapes if s[d] is not None]
            if not dim_vals:
                return None
            try:
                dim_max = max(dim_vals)
            except TypeError:
                dim_max = dim_vals[0]
            combined.append(dim_max)
        result = tuple(combined) if combined else None
        logger.debug(f"[infer_einstein_shape] {decl.name} final shape: {result}")
        return result
    
    def _infer_shape_from_arrays(
        self, index_var: str, value_expr: ExpressionIR,
        variable_ranges: Optional[Dict[DefId, Any]] = None,
    ) -> Optional[int]:
        """
        Infer shape dimension from array accesses in value expression.
        Uses variable_ranges (clause-scoped) for offset variable ranges.
        """
        from ..ir.nodes import BinaryOpIR, IdentifierIR
        import logging
        logger = logging.getLogger(__name__)
        variable_ranges = variable_ranges or {}
        
        # Find array accesses that use this index variable
        accesses = value_expr.accept(ArrayAccessCollector())
        logger.debug(f"[_infer_shape_from_arrays] {index_var}: found {len(accesses)} array accesses")
        
        for access in accesses:
            logger.debug(f"[_infer_shape_from_arrays] {index_var}: checking access with {len(access.indices)} indices")
            # Find position of index_var in indices
            for dim_idx, idx_expr in enumerate(access.indices):
                # Check if this index expression involves our index_var
                involves = self._expr_involves_var(idx_expr, index_var)
                logger.debug(f"[_infer_shape_from_arrays] {index_var}: dim {dim_idx} involves {index_var}? {involves}")
                if not involves:
                    continue
                
                # Get shape of array being accessed
                array_shape = self.get_shape(access.array)
                logger.debug(f"[_infer_shape_from_arrays] {index_var}: array shape: {array_shape}")
                if not array_shape or dim_idx >= len(array_shape):
                    logger.debug(f"[_infer_shape_from_arrays] {index_var}: array shape unavailable or dim out of range")
                    continue
                
                input_dim_size = array_shape[dim_idx]
                logger.debug(f"[_infer_shape_from_arrays] {index_var}: input_dim_size: {input_dim_size}")
                
                # Detect stride and offset patterns: i*stride+offset, i+offset, etc.
                stride, max_offset = self._extract_stride_and_offset(idx_expr, index_var, variable_ranges)
                logger.debug(f"[_infer_shape_from_arrays] {index_var}: stride={stride}, max_offset={max_offset}")
                
                if max_offset is not None:
                    if stride and stride > 1:
                        # Strided access: output_dim = floor((input_dim - max_offset - 1) / stride) + 1
                        import math
                        safe_dim_size = max(0, math.floor((input_dim_size - max_offset - 1) / stride) + 1)
                        logger.debug(f"[_infer_shape_from_arrays] {index_var}: strided access, returning safe_dim_size: {safe_dim_size}")
                        return safe_dim_size
                    elif max_offset > 0:
                        # Simple offset: output_dim = input_dim - max_offset
                        safe_dim_size = max(0, input_dim_size - max_offset)
                        logger.debug(f"[_infer_shape_from_arrays] {index_var}: simple offset, returning safe_dim_size: {safe_dim_size}")
                        return safe_dim_size
                    else:
                        # No offset
                        logger.debug(f"[_infer_shape_from_arrays] {index_var}: no offset, returning input_dim_size: {input_dim_size}")
                        return input_dim_size
        
        logger.debug(f"[_infer_shape_from_arrays] {index_var}: returning None (no valid access found)")
        return None
    
    def _expr_involves_var(self, expr: ExpressionIR, var_name: str) -> bool:
        """Check if expression involves a variable"""
        return expr.accept(ExprInvolvesVarVisitor(var_name))

    def _extract_stride_and_offset(
        self, idx_expr: ExpressionIR, index_var: str,
        variable_ranges: Optional[Dict[DefId, Any]] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Extract stride and maximum offset from index expression. Returns (stride, max_offset)."""
        return idx_expr.accept(_StrideOffsetExtractor(index_var, variable_ranges))

    def _extract_max_offset(
        self, idx_expr: ExpressionIR, index_var: str,
        variable_ranges: Optional[Dict[DefId, Any]] = None,
    ) -> Optional[int]:
        """Extract maximum offset from index expression. Uses variable_ranges (clause-scoped) when provided."""
        _stride, max_offset = idx_expr.accept(_StrideOffsetExtractor(index_var, variable_ranges))
        return max_offset

    def check_perfect_partition(self, declarations: List[BindingIR]) -> bool:
        """Check if declarations form perfect partition"""
        # Collect all index combinations
        all_combinations: Set[Tuple] = set()
        
        for decl in declarations:
            combinations = self._get_index_combinations(decl)
            all_combinations.update(combinations)
        
        # Check for overlaps
        total_size = sum(len(self._get_index_combinations(d)) for d in declarations)
        if len(all_combinations) != total_size:
            return False  # Overlaps found
        
        return True
    
    def _get_index_combinations(self, decl: BindingIR) -> Set[Tuple]:
        """Get all index combinations for declaration. Uses clause variable_ranges only (no global dict)."""
        combinations: Set[Tuple] = set()
        clauses = decl.clauses or []
        if not clauses or not clauses[0].indices:
            return combinations
        rank = len(clauses[0].indices)

        try:
            ranges_list = []
            for dim in range(rank):
                dim_start, dim_end = None, None
                for clause in clauses:
                    if not clause.indices or dim >= len(clause.indices):
                        continue
                    idx = clause.indices[dim]
                    defid = idx.defid
                    if not defid:
                        continue
                    variable_ranges = clause.variable_ranges or {}
                    range_obj = variable_ranges.get(defid)
                    if not range_obj:
                        continue
                    start = _range_bound_to_int(range_obj, 'start')
                    end = _range_bound_to_int(range_obj, 'end')
                    if start is None or end is None:
                        continue
                    if dim_start is None or start < dim_start:
                        dim_start = start
                    if dim_end is None or end > dim_end:
                        dim_end = end
                if dim_start is None or dim_end is None:
                    return set()
                ranges_list.append(range(dim_start, dim_end))

            from itertools import product
            for combo in product(*ranges_list):
                combinations.add(combo)
        except RuntimeError:
            pass

        return combinations

class ShapeAnalysisVisitor(IRVisitor[None]):
    """Visitor to analyze shapes in IR"""
    
    def __init__(self, analyzer: ShapeAnalyzer):
        self.analyzer = analyzer
    
    def visit_program(self, node: ProgramIR) -> None:
        """Visit program and analyze shapes in all statements and functions"""
        for stmt in node.statements:
            if stmt is not None:
                stmt.accept(self)
        # Visit all functions
        for func in node.functions:
            func.accept(self)
        # Visit all constants
        for const in node.constants:
            const.accept(self)
    
    def visit_array_literal(self, expr: ArrayLiteralIR) -> None:
        """Infer shape from array literal"""
        # CRITICAL: Process elements FIRST so their shapes are available when inferring this array's shape
        # This allows recursive shape inference for multi-dimensional arrays
        for elem in expr.elements:
            elem.accept(self)
        
        # Now infer shape (elements have been visited and their shapes are stored)
        shape = self.analyzer.infer_array_literal_shape(expr)
        
        if shape:
            # shape is already a tuple from infer_array_literal_shape
            self.analyzer.set_shape(expr, shape)
            
            # Attach shape_info to IR node (for range analysis to access)
            # Use tuple for shape_info (hashable, immutable)
            expr.shape_info = shape if isinstance(shape, tuple) else tuple(shape)
        else:
            pass
    def visit_array_comprehension(self, expr: ArrayComprehensionIR) -> None:
        """Infer shape from array comprehension"""
        shape = self.analyzer.infer_comprehension_shape(expr)
        if shape:
            # shape is already a tuple from infer_comprehension_shape
            self.analyzer.set_shape(expr, shape)
        
        # Process body and range
        expr.body.accept(self)
        for range_expr in expr.ranges:
            range_expr.accept(self)
    
    def visit_binding(self, node: BindingIR) -> None:
        if is_einstein_binding(node):
            from ..shared.types import infer_literal_type
            shape_tuple = self.analyzer.infer_einstein_shape(node)
            if shape_tuple:
                self.analyzer.set_shape(node, shape_tuple)
                # Store shape on the IR node so lowering/backend can read it without lookup
                shape_list = []
                loc = node.location
                for dim in shape_tuple:
                    if isinstance(dim, int):
                        shape_list.append(LiteralIR(
                            value=dim,
                            location=loc,
                            shape_info=None,
                            type_info=infer_literal_type(dim),
                        ))
                if shape_list:
                    node.expr.shape = shape_list
            # Resolve dependent ranges (0..array.shape[dim]) to literals when we know array shape
            self.analyzer._resolve_dependent_ranges_on_decl(node)
            # Process each clause's value
            for clause in (node.clauses or []):
                if clause.value:
                    clause.value.accept(self)
        elif is_function_binding(node):
            pass
        else:
            if node.value:
                node.value.accept(self)
                if node.defid:
                    shape = self.analyzer.get_shape(node.value)
                    if shape:
                        self.analyzer.defid_to_shape[node.defid] = shape
    
    # Required visitor methods (no-op for other nodes)
    def visit_literal(self, node) -> None:
        pass
    
    def visit_identifier(self, node) -> None:
        pass
    
    def visit_binary_op(self, node: BinaryOpIR) -> None:
        # Recurse so operand shapes are available.
        if node.left:
            node.left.accept(self)
        if node.right:
            node.right.accept(self)
        # Gradient quotient (df/dx): when both operands are gradient-typed, result shape is
        # elementwise (same as numerator). Design allows scalar or elementwise; we use elementwise.
        if node.operator == BinaryOp.DIV and isinstance(node.left, DifferentialIR) and isinstance(node.right, DifferentialIR):
            num_shape = self.analyzer.get_shape(node.left)
            if num_shape:
                shape = tuple(num_shape) if not isinstance(num_shape, tuple) else num_shape
                self.analyzer.set_shape(node, shape)
                node.shape_info = shape
    
    def visit_function_call(self, node) -> None:
        pass
    
    def visit_unary_op(self, node) -> None:
        pass
    
    def visit_differential(self, node: DifferentialIR) -> None:
        """DifferentialIR(operand) has the same shape as operand (AUTODIFF_IMPLEMENTATION.md §5)."""
        node.operand.accept(self)
        operand_shape = self.analyzer.get_shape(node.operand)
        if operand_shape:
            shape = tuple(operand_shape) if not isinstance(operand_shape, tuple) else operand_shape
            self.analyzer.set_shape(node, shape)
            node.shape_info = shape

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
    
    def visit_tuple_expression(self, node) -> None:
        pass
    
    def visit_tuple_access(self, node) -> None:
        pass
    
    def visit_interpolated_string(self, node) -> None:
        pass
    
    def visit_cast_expression(self, node) -> None:
        """Visit cast expression - cast doesn't change shape, so visit inner expression"""
        # Cast doesn't change shape, but we need to visit the inner expression
        # so its shape is available
        if node.expr:
            node.expr.accept(self)
            # Get the shape from the inner expression and propagate it to the cast
            inner_shape = self.analyzer.get_shape(node.expr)
            if inner_shape:
                # Ensure shape is a tuple (hashable)
                if isinstance(inner_shape, list):
                    inner_shape = tuple(inner_shape)
                self.analyzer.set_shape(node, inner_shape)
    
    def visit_member_access(self, node) -> None:
        pass
    
    def visit_try_expression(self, node) -> None:
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
    
    
    def visit_module(self, node) -> None:
        pass


