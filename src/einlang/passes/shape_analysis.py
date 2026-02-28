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
    FunctionCallIR, BindingIR, is_einstein_binding, RectangularAccessIR,
    IRVisitor, RangeIR, IdentifierIR, LiteralIR, MemberAccessIR,
)
from ..shared.defid import DefId
from ..shared.source_location import SourceLocation
from ..shared.types import BinaryOp
from .visitor_helpers import ConstantEvaluator, ArrayAccessCollector

logger = logging.getLogger(__name__)


def _range_end_to_int(range_obj: Any) -> Optional[int]:
    """Return (end - 1) as int when range has literal end (max offset for exclusive range)."""
    if range_obj is None:
        return None
    end = getattr(range_obj, 'end', None)
    if end is None:
        if isinstance(range_obj, range):
            return range_obj.stop - 1
        return None
    if isinstance(end, LiteralIR) and isinstance(getattr(end, 'value', None), (int, float)):
        return int(end.value) - 1
    return None


def _range_bound_to_int(range_obj: Any, bound: str) -> Optional[int]:
    """Return start or end as int. bound is 'start' or 'end'. For range() use start/stop."""
    if range_obj is None:
        return None
    if isinstance(range_obj, range):
        return range_obj.start if bound == 'start' else range_obj.stop
    attr = getattr(range_obj, bound, None)
    if attr is None:
        return None
    if isinstance(attr, LiteralIR) and isinstance(getattr(attr, 'value', None), (int, float)):
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
            decls = getattr(group, "declarations", None) or []
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
                logger.debug(f"[ShapeAnalysis] Propagated shape {unified} to literal-index decl {decl.name}")

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
        if isinstance(expr, IdentifierIR) and hasattr(expr, 'defid') and expr.defid:
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
        loc = getattr(expr, "location", None)

        if first_is_array:
            # All elements must be arrays and have the same shape
            element_shape = self._infer_array_literal_shape_from_element(first_elem)
            for i, elem in enumerate(expr.elements[1:], start=1):
                if not self._is_array_literal_element(elem):
                    self.tcx.reporter.report_error(
                        "Array literal has inconsistent element types: "
                        "element 0 is array, but element {} is not an array".format(i),
                        location=getattr(elem, "location", loc) or loc,
                    )
                    return None
                elem_shape = self._infer_array_literal_shape_from_element(elem)
                if (
                    elem_shape is not None
                    and element_shape is not None
                    and elem_shape != element_shape
                ):
                    self.tcx.reporter.report_error(
                        "Array literal has inconsistent shapes: "
                        "element 0 has shape {}, but element {} has shape {}".format(
                            element_shape, i, elem_shape
                        ),
                        location=getattr(elem, "location", loc) or loc,
                    )
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
                        location=getattr(elem, "location", loc) or loc,
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
        """Get size of range expression"""
        evaluator = ConstantEvaluator()
        # Direct attribute access - trust IR structure
        if hasattr(range_expr, 'start') and hasattr(range_expr, 'end'):
            start = range_expr.start.accept(evaluator)
            end = range_expr.end.accept(evaluator)
            if start is not None and end is not None:
                return max(0, end - start)
        return None
    
    def resolve_symbolic_shape(self, shape_expr: ExpressionIR) -> Optional[int]:
        """Resolve symbolic shape expression to concrete value"""
        evaluator = ConstantEvaluator()
        # Direct attribute access - trust IR structure
        if hasattr(shape_expr, 'function_name') and shape_expr.function_name == "shape":
            if hasattr(shape_expr, 'arguments') and len(shape_expr.arguments) >= 2:
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
        array_expr = getattr(expr, 'array', None)
        indices = getattr(expr, 'indices', None) or []
        if not isinstance(array_expr, MemberAccessIR) or getattr(array_expr, 'member', None) != 'shape':
            return None
        arr = getattr(array_expr, 'object', None)
        if not indices or not isinstance(indices[0], LiteralIR):
            return None
        dim_val = getattr(indices[0], 'value', None)
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
            variable_ranges = getattr(clause, 'variable_ranges', None) or {}
            for defid, rng in variable_ranges.items():
                if not isinstance(rng, RangeIR):
                    continue
                end_expr = getattr(rng, 'end', None)
                if end_expr is None or isinstance(end_expr, LiteralIR):
                    continue
                resolved = self._evaluate_shape_dim_expr(end_expr)
                if resolved is not None:
                    loc = getattr(rng, 'location', None) or getattr(decl, 'location', None)
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

        def shape_for_clause(clause) -> Optional[tuple]:
            value_expr = clause.value
            variable_ranges = getattr(clause, 'variable_ranges', None) or {}
            shape = []
            for idx in (clause.indices or []):
                from ..ir.nodes import IndexVarIR
                if isinstance(idx, LiteralIR):
                    v = getattr(idx, 'value', None)
                    try:
                        extent = int(v) + 1 if v is not None else 1
                    except (TypeError, ValueError):
                        return None
                    shape.append(max(1, extent))
                    continue
                index_var = getattr(idx, 'name', None)
                defid = getattr(idx, 'defid', None)
                if index_var is None:
                    continue
                range_obj = variable_ranges.get(defid) if defid else None
                has_explicit_range = range_obj is not None
                if has_explicit_range:
                    if isinstance(range_obj, StaticRange):
                        shape.append(range_obj.end)
                    elif isinstance(range_obj, range):
                        shape.append(range_obj.stop)
                    elif isinstance(range_obj, RangeIR) and getattr(range_obj, 'end', None) is not None:
                        end_expr = range_obj.end
                        if isinstance(end_expr, LiteralIR) and isinstance(getattr(end_expr, 'value', None), (int, float)):
                            shape.append(int(end_expr.value))
                        else:
                            s = self._infer_shape_from_arrays(index_var, value_expr, variable_ranges)
                            if s is not None:
                                shape.append(s)
                            else:
                                return None
                    else:
                        s = self._infer_shape_from_arrays(index_var, value_expr, variable_ranges)
                        if s is not None:
                            shape.append(s)
                        else:
                            return None
                else:
                    s = self._infer_shape_from_arrays(index_var, value_expr, variable_ranges)
                    if s is not None:
                        shape.append(s)
                    else:
                        return None
            return tuple(shape) if shape else None

        shapes = []
        for clause in clauses:
            s = shape_for_clause(clause)
            if s is None:
                logger.debug(f"[infer_einstein_shape] {decl.name} clause shape: None")
                return None
            shapes.append(s)
            logger.debug(f"[infer_einstein_shape] {decl.name} clause shape: {s}")

        # All clauses must have the same rank; fail on mismatch
        rank = len(shapes[0])
        if not all(len(s) == rank for s in shapes):
            ranks = [len(s) for s in shapes]
            self.tcx.reporter.report_error(
                f"Einstein declaration '{decl.name}' has clauses with different ranks: {ranks}. All clauses must have the same rank.",
                location=decl.location,
            )
            return None
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
        from ..ir.nodes import IdentifierIR, BinaryOpIR
        
        if isinstance(expr, IdentifierIR):
            return expr.name == var_name
        elif isinstance(expr, BinaryOpIR):
            return (self._expr_involves_var(expr.left, var_name) or 
                    self._expr_involves_var(expr.right, var_name))
        return False
    
    def _extract_stride_and_offset(
        self, idx_expr: ExpressionIR, index_var: str,
        variable_ranges: Optional[Dict[DefId, Any]] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract stride and maximum offset from index expression.
        Uses variable_ranges (clause-scoped) when provided; no global range dict.
        Returns: (stride, max_offset)
        """
        from ..ir.nodes import BinaryOpIR, IdentifierIR, LiteralIR
        import logging
        logger = logging.getLogger(__name__)
        variable_ranges = variable_ranges or {}
        
        # Simple case: just the variable itself (i)
        if isinstance(idx_expr, IdentifierIR) and idx_expr.name == index_var:
            return (None, 0)
        
        # Pattern: i*stride (BinaryOp with MUL)
        if isinstance(idx_expr, BinaryOpIR) and idx_expr.operator == BinaryOp.MUL:
            # Check if it's i*literal or literal*i
            stride = None
            if isinstance(idx_expr.left, IdentifierIR) and idx_expr.left.name == index_var and isinstance(idx_expr.right, LiteralIR):
                stride = int(idx_expr.right.value)
            elif isinstance(idx_expr.right, IdentifierIR) and idx_expr.right.name == index_var and isinstance(idx_expr.left, LiteralIR):
                stride = int(idx_expr.left.value)
            
            if stride:
                logger.debug(f"[ShapeAnalysis] Found stride pattern: {index_var}*{stride}")
                return (stride, 0)
        
        # Pattern: i - variable (BinaryOp with SUB)
        # Example: data[i-k] in cumulative operations
        if isinstance(idx_expr, BinaryOpIR) and idx_expr.operator == BinaryOp.SUB:
            left = idx_expr.left
            right = idx_expr.right
            
            # Pattern: i - variable_offset (e.g., i-k)
            if isinstance(left, IdentifierIR) and left.name == index_var and isinstance(right, IdentifierIR):
                # For cumulative operations like cumsum[i] = sum[k](data[i-k]),
                # k is typically a reduction variable with dependent range (e.g., 0..i+1)
                # We can't statically determine k's range, but we know that:
                # - The expression i-k with valid k values will stay in array bounds
                # - Therefore, i can range from 0 to len(array)-1
                # - So we return offset=0 to use the full array dimension
                logger.debug(f"[ShapeAnalysis] Subtraction pattern: {index_var}-{right.name}")
                logger.debug(f"[ShapeAnalysis] Assuming cumulative operation pattern, using offset=0")
                return (None, 0)
        
        # Pattern: i*stride + offset (BinaryOp with ADD, left is MUL)
        if isinstance(idx_expr, BinaryOpIR) and idx_expr.operator == BinaryOp.ADD:
            left = idx_expr.left
            right = idx_expr.right
            
            # Check if left is i*stride pattern
            stride = None
            if isinstance(left, BinaryOpIR) and left.operator == BinaryOp.MUL:
                # Check if it's i*literal or literal*i
                if isinstance(left.left, IdentifierIR) and left.left.name == index_var and isinstance(left.right, LiteralIR):
                    stride = int(left.right.value)
                elif isinstance(left.right, IdentifierIR) and left.right.name == index_var and isinstance(left.left, LiteralIR):
                    stride = int(left.left.value)
                
                if stride:
                    # Found i*stride pattern, now extract offset from right side
                    max_offset = None
                    if isinstance(right, IdentifierIR):
                        right_defid = getattr(right, 'defid', None)
                        offset_range = variable_ranges.get(right_defid) if right_defid else None
                        logger.debug(f"[ShapeAnalysis] Strided pattern: {index_var}*{stride}+{right.name}, range: {offset_range}")
                        if offset_range is not None:
                            max_offset = _range_end_to_int(offset_range)
                        if max_offset is not None:
                            logger.debug(f"[ShapeAnalysis] Max offset for {right.name}: {max_offset}")
                            return (stride, max_offset)
                    elif isinstance(right, LiteralIR):
                        # Pattern: i*stride + literal
                        max_offset = int(right.value)
                        logger.debug(f"[ShapeAnalysis] Strided pattern: {index_var}*{stride}+{max_offset}")
                        return (stride, max_offset)
            
            # Pattern: i + literal (e.g., i+1)
            if isinstance(left, IdentifierIR) and left.name == index_var and isinstance(right, LiteralIR):
                return (None, int(right.value))
            
            # Pattern: literal + i (e.g., 1+i)
            if isinstance(right, IdentifierIR) and right.name == index_var and isinstance(left, LiteralIR):
                return (None, int(left.value))
            
            # Pattern: i + variable_offset (e.g., i+di)
            if isinstance(left, IdentifierIR) and left.name == index_var and isinstance(right, IdentifierIR):
                right_defid = getattr(right, 'defid', None)
                offset_range = variable_ranges.get(right_defid) if right_defid else None
                logger.debug(f"[ShapeAnalysis] Looking up range for {right.name}: {offset_range}")
                if offset_range is not None:
                    max_offset = _range_end_to_int(offset_range)
                    if max_offset is not None:
                        logger.debug(f"[ShapeAnalysis] Max offset for {right.name}: {max_offset}")
                        return (None, max_offset)
            
            # Pattern: variable_offset + i (e.g., di+i)
            if isinstance(right, IdentifierIR) and right.name == index_var and isinstance(left, IdentifierIR):
                left_defid = getattr(left, 'defid', None)
                offset_range = variable_ranges.get(left_defid) if left_defid else None
                if offset_range is not None:
                    max_offset = _range_end_to_int(offset_range)
                    if max_offset is not None:
                        return (None, max_offset)
        
        # Unable to determine
        return (None, None)
    
    def _extract_max_offset(
        self, idx_expr: ExpressionIR, index_var: str,
        variable_ranges: Optional[Dict[DefId, Any]] = None,
    ) -> Optional[int]:
        """Extract maximum offset from index expression. Uses variable_ranges (clause-scoped) when provided."""
        from ..ir.nodes import BinaryOpIR, IdentifierIR, LiteralIR
        import logging
        logger = logging.getLogger(__name__)
        variable_ranges = variable_ranges or {}
        
        # Simple case: just the variable itself (i)
        if isinstance(idx_expr, IdentifierIR) and idx_expr.name == index_var:
            return 0
        
        # Pattern: i*stride + offset (BinaryOp with ADD, left is MUL)
        # Example: i*2+di
        if isinstance(idx_expr, BinaryOpIR) and idx_expr.operator == BinaryOp.ADD:
            left = idx_expr.left
            right = idx_expr.right
            
            # Check if left is i*stride pattern
            if isinstance(left, BinaryOpIR) and left.operator == BinaryOp.MUL:
                # Check if it's i*literal or literal*i
                if ((isinstance(left.left, IdentifierIR) and left.left.name == index_var and isinstance(left.right, LiteralIR)) or
                    (isinstance(left.right, IdentifierIR) and left.right.name == index_var and isinstance(left.left, LiteralIR))):
                    # Found i*stride pattern, now extract offset from right side
                    if isinstance(right, IdentifierIR):
                        right_defid = getattr(right, 'defid', None)
                        offset_range = variable_ranges.get(right_defid) if right_defid else None
                        logger.debug(f"[ShapeAnalysis] Strided pattern: {index_var}*stride+{right.name}, range: {offset_range}")
                        if offset_range is not None:
                            max_offset = _range_end_to_int(offset_range)
                            if max_offset is not None:
                                return max_offset
                    elif isinstance(right, LiteralIR):
                        return int(right.value)
            
            # Pattern: i + literal (e.g., i+1)
            if (isinstance(left, IdentifierIR) and left.name == index_var and
                isinstance(right, LiteralIR)):
                return int(right.value)
            
            # Pattern: literal + i (e.g., 1+i)
            if (isinstance(right, IdentifierIR) and right.name == index_var and
                isinstance(left, LiteralIR)):
                return int(left.value)
            
            # Pattern: i + variable_offset (e.g., i+di)
            if isinstance(left, IdentifierIR) and left.name == index_var and isinstance(right, IdentifierIR):
                right_defid = getattr(right, 'defid', None)
                offset_range = variable_ranges.get(right_defid) if right_defid else None
                logger.debug(f"[ShapeAnalysis] Looking up range for {right.name}: {offset_range}")
                if offset_range is not None:
                    max_offset = _range_end_to_int(offset_range)
                    if max_offset is not None:
                        return max_offset
            
            # Pattern: variable_offset + i (e.g., di+i)
            if isinstance(right, IdentifierIR) and right.name == index_var and isinstance(left, IdentifierIR):
                left_defid = getattr(left, 'defid', None)
                offset_range = variable_ranges.get(left_defid) if left_defid else None
                if offset_range is not None:
                    max_offset = _range_end_to_int(offset_range)
                    if max_offset is not None:
                        return max_offset
        
        # Unable to determine offset
        return None
    
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
                    defid = getattr(idx, 'defid', None)
                    if not defid:
                        continue
                    variable_ranges = getattr(clause, 'variable_ranges', None) or {}
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
        # Visit all statements
        for stmt in node.statements:
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
    
    def visit_einstein_declaration(self, expr: BindingIR) -> None:
        """Infer shape for Einstein declaration; store on IR and in analyzer; resolve dependent ranges when shape is known."""
        from ..shared.types import infer_literal_type
        shape_tuple = self.analyzer.infer_einstein_shape(expr)
        if shape_tuple:
            self.analyzer.set_shape(expr, shape_tuple)
            # Store shape on the IR node so lowering/backend can read it without lookup
            shape_list = []
            loc = getattr(expr, 'location', None)
            for dim in shape_tuple:
                if isinstance(dim, int):
                    shape_list.append(LiteralIR(
                        value=dim,
                        location=loc,
                        shape_info=None,
                        type_info=infer_literal_type(dim),
                    ))
            if shape_list:
                expr.expr.shape = shape_list
        # Resolve dependent ranges (0..array.shape[dim]) to literals when we know array shape
        self.analyzer._resolve_dependent_ranges_on_decl(expr)
        # Process each clause's value
        for clause in (expr.clauses or []):
            if clause.value:
                clause.value.accept(self)
    
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
        if hasattr(node, 'expr') and node.expr:
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
    
    def visit_arrow_expression(self, node) -> None:
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
    
    def visit_function_def(self, node) -> None:
        pass
    
    def visit_constant_def(self, node) -> None:
        pass
    
    def visit_module(self, node) -> None:
        pass

    def visit_variable_declaration(self, node) -> Any:
        """Visit variable declaration - recurse into value and store shape by DefId"""
        if hasattr(node, 'value') and node.value:
            node.value.accept(self)
            # CRITICAL FIX: Store shape by DefId for later lookup
            # This allows get_shape() to find shapes for IdentifierIR nodes
            if hasattr(node, 'defid') and node.defid:
                shape = self.analyzer.get_shape(node.value)
                if shape:
                    self.analyzer.defid_to_shape[node.defid] = shape
        return None

