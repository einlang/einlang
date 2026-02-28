"""
Implicit Range Detector for - 

Infers ranges from array usage patterns using IR visitor pattern.
ImplicitRangeDetector adapted for IR structure.

Returns RangeInfo (not Python range) to support dynamic bounds.
"""

from typing import List, Optional, Dict, Tuple, Any, Union
import logging

from ..ir.nodes import (
    ExpressionIR, IdentifierIR, IndexVarIR, RectangularAccessIR, ArrayLiteralIR,
    BinaryOpIR, UnaryOpIR, ReductionExpressionIR, WhereExpressionIR,
    IRVisitor, LiteralIR, MemberAccessIR, CastExpressionIR,
    BindingIR, FunctionCallIR, IfExpressionIR, RangeIR,
    TupleAccessIR, BuiltinCallIR,
    is_function_binding, is_einstein_binding,
)
from ..shared.defid import DefId
from ..shared.source_location import SourceLocation
from ..shared.types import BinaryOp, infer_literal_type, UNKNOWN
from .range_info import RangeInfo, StaticRange, DynamicRange
from .ir_constraint_solver import solve_index_constraint

logger = logging.getLogger(__name__)


class ImplicitRangeDetector(IRVisitor[None]):
    """
    Detects implicit ranges from array usage patterns using IR visitor pattern.
    
    Examples:
        let result[i] = data[i]      → infer: i in 0..len(data)
        let out[i,j] = mat[i,j]      → infer: i in 0..mat.shape[0], j in 0..mat.shape[1]
        let out[i] = data[i*2]       → infer: i in 0..(len(data)/2)
        let out[i] = data[i+1]       → infer: i in 0..(len(data)-1)
    """
    
    def __init__(self, var_definitions_scope_stack: List[Dict[DefId, Any]], tcx: Any = None):
        """
        Args:
            var_definitions_scope_stack: Scopes from outermost to innermost; lookup searches innermost first.
            tcx: Type context (for shape analysis results)
        """
        self._var_definitions_scope_stack = list(var_definitions_scope_stack) if var_definitions_scope_stack else []
        self._current_clause = None
        self._current_declaration = None
        self._tcx = tcx
        self.accesses: List[Tuple[Optional[DefId], int, ExpressionIR, ExpressionIR]] = []
        self._function_decls_by_defid: Dict[DefId, Any] = {}

    def _resolve_var_def(self, array_defid: DefId) -> Any:
        if array_defid is None:
            return None
        for scope in reversed(self._var_definitions_scope_stack):
            var_def = scope.get(array_defid)
            if var_def is not None:
                return var_def
        var_def = self._function_decls_by_defid.get(array_defid)
        if var_def is not None:
            return var_def
        if self._tcx and hasattr(self._tcx, 'program_ir'):
            program = getattr(self._tcx, 'program_ir', None)
            if program:
                for stmt in program.statements:
                    if isinstance(stmt, BindingIR):
                        if getattr(stmt, 'defid', None) == array_defid:
                            return stmt.value
        return None
    
    def set_prior_declarations(self, decls: List[Any]) -> None:
        """
        Register prior Einstein declarations in this function (by DefId only).
        ALIGNED: Enables range inference from prior declarations (e.g. shifted[..batch, j] then doubled[...] = shifted[...] * 2).
        """
        for decl in decls:
            if is_einstein_binding(decl):
                did = getattr(decl, 'defid', None)
                if did is not None:
                    self._function_decls_by_defid[did] = decl

    def set_current_declaration(self, declaration: Any) -> None:
        """Set the Einstein declaration being analyzed (for recurrence detection). array_name == node.array_name."""
        self._current_declaration = declaration


    def infer_implicit_range(self, expr: ExpressionIR, defid: DefId) -> Optional[RangeInfo]:
        """
        Infer implicit range for an index variable from how it's used.
        defid is required; variable identity is by DefId only.
        """
        if defid is None:
            raise ValueError("infer_implicit_range requires defid")
        self._target_defid = defid
        try:
            return self._infer_implicit_range_impl(expr)
        finally:
            self._target_defid = None

    def _name_from_defid(self) -> Optional[str]:
        """Resolve name from _current_clause by defid (for solver/logging only)."""
        ctx = self._current_clause
        target_defid = getattr(self, '_target_defid', None)
        if ctx is None or target_defid is None:
            return None
        if hasattr(ctx, 'indices') and ctx.indices:
            for idx in ctx.indices:
                if getattr(idx, 'defid', None) == target_defid:
                    return getattr(idx, 'name', None)
        if hasattr(ctx, 'loop_vars') and ctx.loop_vars:
            for i in ctx.loop_vars:
                if getattr(i, 'defid', None) == target_defid:
                    return getattr(i, 'name', None)
        return None

    def _is_direct_index_expr(self, index_expr: ExpressionIR) -> bool:
        """True if index is exactly the target variable (by defid only)."""
        target_defid = getattr(self, '_target_defid', None)
        if target_defid is None:
            return False
        return isinstance(index_expr, (IdentifierIR, IndexVarIR)) and getattr(index_expr, 'defid', None) == target_defid

    def _infer_implicit_range_impl(self, expr: ExpressionIR) -> Optional[RangeInfo]:
        array_accesses = self._find_array_accesses(expr)
        if not array_accesses:
            return None
        
        def expression_complexity(access_tuple):
            """Estimate complexity by counting IR nodes"""
            _, _, index_expr, _ = access_tuple
            if index_expr is None:
                return 0
            counter = _ComplexityCounter()
            index_expr.accept(counter)
            return counter.count

        sorted_accesses = sorted(array_accesses, key=expression_complexity)
        direct_matches = []
        indirect_matches = []

        for array_defid, index_position, index_expr, base_expr in sorted_accesses:
            logger.debug(f"[ImplicitRangeDetector] Trying access: defid {array_defid} pos {index_position}")
            try:
                decl = getattr(self, '_current_declaration', None)
                if array_defid is not None and decl is not None and array_defid == getattr(decl, 'defid', None):
                    logger.debug(f"[ImplicitRangeDetector] Recurrence: accessing LHS array at dim {index_position}")
                    shape = self._get_array_shape(array_defid, index_position)
                    if shape is None and hasattr(decl, 'name') and hasattr(decl, 'location'):
                        from ..shared.types import PrimitiveType
                        loc = getattr(decl, 'location', None) or SourceLocation('<unknown>', 0, 0, 0, 0)
                        array_id = IdentifierIR(name=decl.name, location=loc, defid=getattr(decl, 'defid', None))
                        shape_member = MemberAccessIR(object=array_id, member='shape', location=loc)
                        index_lit = LiteralIR(value=index_position, location=loc, type_info=PrimitiveType(name='i32'))
                        shape = RectangularAccessIR(array=shape_member, indices=[index_lit], location=loc)
                    if shape is not None:
                        if isinstance(shape, int):
                            if self._is_direct_index_expr(index_expr):
                                direct_matches.append(StaticRange(0, shape))
                            else:
                                indirect_matches.append(StaticRange(0, shape))
                        else:
                            start_lit = LiteralIR(value=0, location=shape.location, type_info=PrimitiveType(name='i32'))
                            r = DynamicRange(start=start_lit, end=shape)
                            if self._is_direct_index_expr(index_expr):
                                direct_matches.append(r)
                            else:
                                indirect_matches.append(r)
                    continue
                result = self._try_array_literal(base_expr, index_position, index_expr)
                if result:
                    if self._is_direct_index_expr(index_expr):
                        direct_matches.append(result)
                    else:
                        indirect_matches.append(result)
                    continue

                result = self._try_intermediate_access(index_expr)
                if result:
                    if self._is_direct_index_expr(index_expr):
                        direct_matches.append(result)
                    else:
                        indirect_matches.append(result)
                    continue

                if array_defid is None:
                    continue
                shape = self._get_array_shape(array_defid, index_position)
                if shape is not None:
                    constraint_result = self._extract_constraint_from_array_access_ir(
                        base_expr, index_position, index_expr, shape
                    )
                    
                    if constraint_result:
                        constraint_range = constraint_result
                        if constraint_range:
                            logger.debug(f"[ImplicitRangeDetector] Constraint solver inferred: {constraint_range}")
                            if self._is_direct_index_expr(index_expr):
                                direct_matches.append(constraint_range)
                            else:
                                indirect_matches.append(constraint_range)
                            continue
                
                from ..shared.types import PrimitiveType
                shape = self._get_array_shape(array_defid, index_position)
                if shape is not None:
                    if self._is_direct_index_expr(index_expr):
                        if isinstance(shape, int):
                            direct_matches.append(StaticRange(0, shape))
                        else:
                            start_lit = LiteralIR(value=0, location=shape.location, type_info=PrimitiveType(name='i32'))
                            direct_matches.append(DynamicRange(start=start_lit, end=shape))
                    elif isinstance(index_expr, BinaryOpIR) and index_expr.operator in (BinaryOp.ADD, BinaryOp.SUB):
                        target_defid = getattr(self, '_target_defid', None)
                        left_defid = getattr(index_expr.left, 'defid', None)
                        left_is_target = target_defid is not None and left_defid == target_defid
                        if left_is_target:
                            if isinstance(shape, int):
                                indirect_matches.append(StaticRange(0, shape))
                            else:
                                start_lit = LiteralIR(value=0, location=shape.location, type_info=PrimitiveType(name='i32'))
                                indirect_matches.append(DynamicRange(start=start_lit, end=shape))
                        else:
                            # Complex binary op, use conservative bound
                            logger.debug(f"[ImplicitRangeDetector] [Fallback] Complex binary op, using shape as bound")
                            if isinstance(shape, int):
                                indirect_matches.append(StaticRange(0, shape))
                            else:
                                start_lit = LiteralIR(value=0, location=shape.location, type_info=PrimitiveType(name='i32'))
                                indirect_matches.append(DynamicRange(start=start_lit, end=shape))
                    else:
                        # Complex expression, use shape as conservative bound
                        logger.debug(f"[ImplicitRangeDetector] [Fallback] Complex expression, using shape as bound")
                        if isinstance(shape, int):
                            indirect_matches.append(StaticRange(0, shape))
                        else:
                            start_lit = LiteralIR(value=0, location=shape.location, type_info=PrimitiveType(name='i32'))
                            indirect_matches.append(DynamicRange(start=start_lit, end=shape))
                        
            except Exception as e:
                import traceback
                logger.debug(f"[ImplicitRangeDetector] Range inference failed from defid {array_defid} pos {index_position}: {e}")
                logger.debug(f"[ImplicitRangeDetector] Traceback:\n{traceback.format_exc()}")
                continue  # Try next array access
        
        # Combine all matches
        # CRITICAL FIX : Distinguish between Einstein index variables and reduction variables
        # - For Einstein index variables (var_name in node's indices/loop_vars): take intersection of ALL matches
        #   This ensures bounds safety (e.g., arr[i] + arr[i+1] requires i in intersection of both constraints)
        # - For reduction variables (called from reduction context): prioritize direct accesses
        #   This handles cases like sum[k](signal[i+k] * kernel[k]) where kernel[k] is more reliable
        
        is_einstein_index = False
        ctx = self._current_clause
        target_defid = getattr(self, '_target_defid', None)
        if target_defid is not None and ctx is not None:
            if hasattr(ctx, 'indices') and ctx.indices:
                for idx in ctx.indices:
                    if getattr(idx, 'defid', None) == target_defid:
                        is_einstein_index = True
                        break
            if not is_einstein_index and hasattr(ctx, 'loop_vars') and ctx.loop_vars:
                for i in ctx.loop_vars:
                    if getattr(i, 'defid', None) == target_defid:
                        is_einstein_index = True
                        break
        
        if is_einstein_index:
            # Einstein index variable: take intersection of all matches (most restrictive)
            # This matches the constraint solver approach which ensures all accesses are in bounds
            all_matches = direct_matches + indirect_matches
            if all_matches:
                # For StaticRange, take min by 'end'; for DynamicRange, prefer first found
                # (more complex merging would require symbolic reasoning)
                static_matches = [m for m in all_matches if isinstance(m, StaticRange)]
                if static_matches:
                    result_range = min(static_matches, key=lambda r: r.end)
                    return result_range
                else:
                    # All dynamic ranges - return first (TODO: merge dynamic ranges symbolically)
                    return all_matches[0]
        else:
            # Reduction variable: prioritize direct accesses (more reliable)
            # This matches the approach for reduction variables
            if direct_matches:
                static_matches = [m for m in direct_matches if isinstance(m, StaticRange)]
                if static_matches:
                    result_range = min(static_matches, key=lambda r: r.end)
                    return result_range
                else:
                    return direct_matches[0]
            elif indirect_matches:
                static_matches = [m for m in indirect_matches if isinstance(m, StaticRange)]
                if static_matches:
                    result_range = min(static_matches, key=lambda r: r.end)
                    return result_range
                else:
                    return indirect_matches[0]
        
        return None
    
    def _find_array_accesses(self, expr: ExpressionIR) -> List[Tuple[Optional[DefId], int, ExpressionIR, ExpressionIR]]:
        """Find array accesses that use the target variable (by defid). Returns (array_defid, index_position, index_expr, base_expr)."""
        target_defid = getattr(self, '_target_defid', None)
        if target_defid is None:
            raise ValueError("_find_array_accesses requires _target_defid")
        self.accesses = []
        expr.accept(self)
        return self.accesses


    def _get_array_shape(self, array_defid: DefId, index_position: int) -> Union[int, ExpressionIR, None]:
        """
        Get array shape for a specific dimension. Lookup by DefId via scope stack.
        Returns: int, ExpressionIR, or None.
        """
        var_def = self._resolve_var_def(array_defid)
        if var_def is None:
            logger.debug(f"[_get_array_shape] No definition for defid {array_defid}")
            return None
        _name = getattr(var_def, 'name', None) or ''
        _defid_for_id = array_defid

        if self._tcx:
            from .shape_analysis import UnifiedShapeAnalysisPass
            try:
                shape_data = self._tcx.get_analysis(UnifiedShapeAnalysisPass)
                if shape_data and isinstance(shape_data, dict) and 'defid_shapes' in shape_data:
                    shape = shape_data['defid_shapes'].get(array_defid)
                    if shape and isinstance(shape, (list, tuple)) and index_position < len(shape):
                        size = shape[index_position]
                        if isinstance(size, int):
                            logger.debug(f"[_get_array_shape] Found shape from shape analysis: defid {array_defid}[{index_position}] = {size}")
                            return size
            except (RuntimeError, AttributeError):
                pass

        from ..shared.types import RectangularType, PrimitiveType
        from ..shared.source_location import SourceLocation

        type_info = getattr(var_def, 'param_type', None) or getattr(var_def, 'type_info', None)
        if isinstance(type_info, RectangularType) and type_info.shape and index_position < len(type_info.shape):
            dim = type_info.shape[index_position]
            if dim is None or dim == '?':
                if not _name:
                    return None
                loc = getattr(var_def, 'location', None) or SourceLocation('<unknown>', 0, 0, 0, 0)
                array_id = IdentifierIR(name=_name, location=loc, defid=_defid_for_id)
                shape_member = MemberAccessIR(object=array_id, member='shape', location=loc)
                index_lit = LiteralIR(value=index_position, location=loc, type_info=PrimitiveType(name='i32'))
                return RectangularAccessIR(array=shape_member, indices=[index_lit], location=loc)
            if isinstance(dim, int):
                return dim

        from ..ir.nodes import ParameterIR
        if isinstance(var_def, ParameterIR) and (type_info is None or not isinstance(type_info, RectangularType)):
            loc = getattr(var_def, 'location', None) or SourceLocation('<unknown>', 0, 0, 0, 0)
            array_id = IdentifierIR(name=_name, location=loc, defid=_defid_for_id)
            shape_member = MemberAccessIR(object=array_id, member='shape', location=loc)
            index_lit = LiteralIR(value=index_position, location=loc, type_info=PrimitiveType(name='i32'))
            return RectangularAccessIR(array=shape_member, indices=[index_lit], location=loc)

        if isinstance(var_def, BindingIR) and getattr(var_def, 'value', None) is not None:
            var_def = var_def.value
        if hasattr(var_def, 'shape_info') and var_def.shape_info:
            shape = var_def.shape_info
            if isinstance(shape, (list, tuple)) and index_position < len(shape):
                size = shape[index_position]
                if isinstance(size, int):
                    logger.debug(f"[_get_array_shape] Found shape from shape_info: defid {array_defid}[{index_position}] = {size}")
                    return size

        from ..ir.nodes import CastExpressionIR
        if isinstance(var_def, CastExpressionIR) and getattr(var_def, 'expr', None) is not None:
            var_def = var_def.expr

        if isinstance(var_def, ArrayLiteralIR):
            if index_position == 0:
                size = len(var_def.elements)
                logger.debug(f"[_get_array_shape] Found shape from ArrayLiteralIR (1D): defid {array_defid}[{index_position}] = {size}")
                return size
            current = var_def
            for dim in range(index_position):
                if isinstance(current, CastExpressionIR) and getattr(current, 'expr', None) is not None:
                    current = current.expr
                if isinstance(current, ArrayLiteralIR) and len(current.elements) > 0:
                    current = current.elements[0]
                else:
                    return None
            if isinstance(current, CastExpressionIR) and getattr(current, 'expr', None) is not None:
                current = current.expr
            if isinstance(current, ArrayLiteralIR):
                return len(current.elements)

        if is_einstein_binding(var_def):
            max_size = None
            for clause in (var_def.clauses or []):
                var_ranges = getattr(clause, 'variable_ranges', None) or {}
                if not (clause.indices or []) or index_position >= len(clause.indices):
                    continue
                idx_expr = clause.indices[index_position]
                did = getattr(idx_expr, 'defid', None)
                range_ir = var_ranges.get(did) if did else None
                if range_ir:
                    if isinstance(range_ir, range):
                        s = range_ir.stop
                        if isinstance(s, int) and (max_size is None or s > max_size):
                            max_size = s
                    elif hasattr(range_ir, 'end') and isinstance(range_ir.end, LiteralIR):
                        s = range_ir.end.value
                        if isinstance(s, int) and (max_size is None or s > max_size):
                            max_size = s
            if max_size is not None:
                logger.debug(f"[_get_array_shape] Found shape from variable_ranges: defid {array_defid}[{index_position}] = {max_size}")
                return max_size
            num_dims = len(var_def.shape) if getattr(var_def, 'shape', None) else 0
            if num_dims == 0 and var_def.clauses and var_def.clauses[0].indices:
                num_dims = len(var_def.clauses[0].indices)
            if index_position < num_dims:
                if not _name:
                    return None
                loc = getattr(var_def, 'location', None) or SourceLocation('<unknown>', 0, 0, 0, 0)
                array_id = IdentifierIR(name=_name, location=loc, defid=_defid_for_id)
                shape_member = MemberAccessIR(object=array_id, member='shape', location=loc)
                index_lit = LiteralIR(value=index_position, location=loc, type_info=PrimitiveType(name='i32'))
                return RectangularAccessIR(array=shape_member, indices=[index_lit], location=loc)

        if hasattr(var_def, 'shape') and var_def.shape and isinstance(var_def.shape, (list, tuple)) and index_position < len(var_def.shape):
            return var_def.shape[index_position]

        logger.debug(f"[_get_array_shape] Could not find shape for defid {array_defid}[{index_position}]")
        return None

    def _try_array_literal(self, base_expr: ExpressionIR, index_position: int, index_expr: ExpressionIR) -> Optional[RangeInfo]:
        """Try to infer range from array literal (no str key)."""
        if not isinstance(base_expr, ArrayLiteralIR):
            return None
        logger.debug(f"[ImplicitRangeDetector] Array literal detected, computing static length from base_expr")
        # Compute static length for this dimension
        current = base_expr
        for dim in range(index_position):
            if isinstance(current, ArrayLiteralIR) and len(current.elements) > 0:
                current = current.elements[0]
            else:
                logger.debug(f"[ImplicitRangeDetector] Cannot compute static length for array literal dimension {index_position}")
                return None
        
        if not isinstance(current, ArrayLiteralIR):
            logger.debug("[ImplicitRangeDetector] Cannot compute static length for array literal")
            return None
        
        array_len = len(current.elements)
        logger.debug(f"[ImplicitRangeDetector] Computed static length: {array_len}")
        
        if self._is_direct_index_expr(index_expr):
            return StaticRange(0, array_len)
        
        # Complex expression - will be handled by other methods
        logger.debug("[ImplicitRangeDetector] Array literal with complex index, will try other methods")
        return None
    
    def _try_intermediate_access(self, index_expr: ExpressionIR) -> Optional[RangeInfo]:
        """Try to infer range from intermediate array access like data[indices[i]]"""
        unwrapped_expr = index_expr
        if isinstance(index_expr, CastExpressionIR):
            unwrapped_expr = index_expr.expr
        if not isinstance(unwrapped_expr, RectangularAccessIR):
            return None
        array_defid = getattr(unwrapped_expr.array, 'defid', None) if isinstance(unwrapped_expr.array, IdentifierIR) else None
        if array_defid is None:
            return None
        target_defid = getattr(self, '_target_defid', None)
        if target_defid is None:
            return None
        for idx, intermediate_idx in enumerate(unwrapped_expr.indices or []):
            if not self._expr_uses_defid(intermediate_idx, target_defid):
                continue
            if self._is_direct_index_expr(intermediate_idx):
                shape = self._get_array_shape(array_defid, idx)
                if shape is not None:
                    if isinstance(shape, int):
                        return StaticRange(0, shape)
                    from ..shared.types import PrimitiveType
                    start_lit = LiteralIR(value=0, location=shape.location, type_info=PrimitiveType(name='i32'))
                    return DynamicRange(start=start_lit, end=shape)
        return None

    def _extract_constraint_from_array_access_ir(
        self,
        array_expr: ExpressionIR,
        index_position: int,
        index_expr: ExpressionIR,
        shape: Union[int, ExpressionIR]
    ) -> Optional[RangeInfo]:
        """Extract constraint and solve for target variable (by defid). Name resolved from clause for solver."""
        target_var = self._name_from_defid()
        if target_var is None:
            return None
        shape_desc = shape if isinstance(shape, int) else f"<dynamic>"
        logger.debug(f"[ImplicitRangeDetector] [IRConstraintSolver] Extracting constraint for defid: {index_expr} < {shape_desc}")
        if isinstance(index_expr, BinaryOpIR):
            if index_expr.operator == BinaryOp.SUB and isinstance(index_expr.right, (IdentifierIR, IndexVarIR)):
                logger.debug("[IRConstraintSolver] Recurrence relation detected")
                return None
        from ..ir.nodes import LiteralIR, MemberAccessIR, RectangularAccessIR
        from ..shared.source_location import SourceLocation
        from ..shared.types import PrimitiveType
        if self._is_direct_index_expr(index_expr):
            if isinstance(shape, int):
                return StaticRange(0, shape)
            start_lit = LiteralIR(value=0, location=shape.location, type_info=PrimitiveType(name='i32'))
            return DynamicRange(start=start_lit, end=shape)
        location = array_expr.location if hasattr(array_expr, 'location') else SourceLocation('<unknown>', 0, 0, 0, 0)
        if isinstance(shape, int):
            dimension_bound = LiteralIR(value=shape, location=location, type_info=PrimitiveType(name='i32'))
        else:
            dimension_bound = shape
        range_result = solve_index_constraint(index_expr, target_var, dimension_bound)
        
        if range_result:
            logger.debug(f"[IRConstraintSolver] Raw result: {range_result}")
            output_range = self._to_output_range_ir(range_result)
            if output_range is not None:
                logger.debug(f"[IRConstraintSolver] After to_output_range_ir: {output_range}")
                return output_range
            return range_result
        else:
            logger.debug(f"[IRConstraintSolver] Could not solve constraint")
            return None
    
    def _find_reduction_in_expr(self, expr: ExpressionIR) -> Optional[ReductionExpressionIR]:
        """Find ReductionExpressionIR inside expr (e.g. BinaryOp(reduction, literal) for sum[...](...)/4)."""
        if expr is None:
            return None
        if isinstance(expr, ReductionExpressionIR):
            return expr
        if isinstance(expr, BinaryOpIR):
            found = self._find_reduction_in_expr(expr.left)
            if found:
                return found
            return self._find_reduction_in_expr(expr.right)
        if isinstance(expr, UnaryOpIR):
            return self._find_reduction_in_expr(expr.operand)
        if isinstance(expr, FunctionCallIR):
            for arg in expr.arguments:
                found = self._find_reduction_in_expr(arg)
                if found:
                    return found
        if isinstance(expr, WhereExpressionIR):
            return self._find_reduction_in_expr(expr.expr)
        return None
    
    def _to_output_range_ir(self, range_result: RangeInfo) -> Optional[RangeInfo]:
        """
        to_output_range / maximize_over_loop_vars.

        After the constraint solver produces a raw DynamicRange, its end expression
        may reference other Einstein output indices or reduction variables.
        eliminates these by substituting each variable with its **max value**
        (end - 1 of its range).

        - Reduction variables: use their loop_var_ranges entry → end - 1.
        - Other output indices whose range is already resolved (in
          clause.variable_ranges): use that range's end - 1.
        - Other output indices whose range is NOT yet resolved: treat as
          StaticRange(0, 1) → end - 1 = 0   (later output indices
          in lexicographic order get minimized to maximize the current range).
        """
        from ..ir.nodes import RangeIR
        from ..shared.types import PrimitiveType
        from ..shared.source_location import SourceLocation

        if not isinstance(range_result, DynamicRange):
            return None

        from .ir_constraint_solver import _collect_defids
        end_defids = _collect_defids(range_result.end)
        if not end_defids:
            return range_result

        target_defid = getattr(self, '_target_defid', None)
        clause = self._current_clause
        loc = getattr(range_result.end, 'location', None) or SourceLocation('<generated>', 0, 0, 0, 0)
        i32 = PrimitiveType(name='i32')
        one = LiteralIR(value=1, location=loc, type_info=i32)

        substitute_map: dict = {}

        # --- 1. Reduction variables: substitute with max (end - 1) ---
        value_expr = clause.value if (clause and hasattr(clause, 'value')) else None
        if value_expr is not None:
            reduction_expr = (
                value_expr if isinstance(value_expr, ReductionExpressionIR)
                else self._find_reduction_in_expr(value_expr)
            )
            if isinstance(reduction_expr, ReductionExpressionIR) and getattr(reduction_expr, 'loop_var_ranges', None):
                for rv_defid, range_ir in reduction_expr.loop_var_ranges.items():
                    if rv_defid not in end_defids:
                        continue
                    if not isinstance(range_ir, RangeIR):
                        continue
                    end_expr = getattr(range_ir, 'end', None)
                    if end_expr is None:
                        continue
                    if isinstance(end_expr, LiteralIR) and isinstance(end_expr.value, (int, float)):
                        substitute_map[rv_defid] = LiteralIR(
                            value=max(0, int(end_expr.value) - 1), location=loc, type_info=i32,
                        )
                    else:
                        substitute_map[rv_defid] = BinaryOpIR(
                            operator=BinaryOp.SUB, left=end_expr, right=one,
                            location=loc, type_info=i32,
                        )

        # --- 2. Other output indices: substitute with max (end - 1) of their range ---
        if clause is not None and target_defid is not None:
            output_index_defids = set()
            for idx in (getattr(clause, 'indices', None) or []):
                did = getattr(idx, 'defid', None)
                if did is not None and did != target_defid:
                    output_index_defids.add(did)

            clause_ranges = getattr(clause, 'variable_ranges', {}) or {}
            for oi_defid in output_index_defids:
                if oi_defid not in end_defids or oi_defid in substitute_map:
                    continue
                resolved = clause_ranges.get(oi_defid)
                if isinstance(resolved, RangeIR) and getattr(resolved, 'end', None) is not None:
                    end_expr = resolved.end
                    if isinstance(end_expr, LiteralIR) and isinstance(end_expr.value, (int, float)):
                        substitute_map[oi_defid] = LiteralIR(
                            value=max(0, int(end_expr.value) - 1), location=loc, type_info=i32,
                        )
                    else:
                        substitute_map[oi_defid] = BinaryOpIR(
                            operator=BinaryOp.SUB, left=end_expr, right=one,
                            location=loc, type_info=i32,
                        )
                else:
                    # Range not yet resolved → StaticRange(0, 1) → end - 1 = 0
                    substitute_map[oi_defid] = LiteralIR(value=0, location=loc, type_info=i32)

        if not substitute_map:
            return range_result

        new_end = self._substitute_defids_in_expr(range_result.end, substitute_map)
        if new_end is range_result.end:
            return range_result

        return DynamicRange(start=range_result.start, end=new_end)
    
    def _substitute_defids_in_expr(self, expr: ExpressionIR, substitute_map: dict) -> ExpressionIR:
        """Replace IdentifierIR/IndexVarIR by DefId: when expr.defid is in substitute_map, return substitute_map[expr.defid]. Clone tree."""
        from ..shared.source_location import SourceLocation
        if expr is None:
            return expr
        if isinstance(expr, (IdentifierIR, IndexVarIR)):
            expr_defid = getattr(expr, 'defid', None)
            if expr_defid is not None and expr_defid in substitute_map:
                return substitute_map[expr_defid]
        if isinstance(expr, LiteralIR):
            return expr
        if isinstance(expr, BinaryOpIR):
            new_left = self._substitute_defids_in_expr(expr.left, substitute_map)
            new_right = self._substitute_defids_in_expr(expr.right, substitute_map)
            if new_left is expr.left and new_right is expr.right:
                return expr
            loc = getattr(expr, 'location', None) or SourceLocation('<generated>', 0, 0, 0, 0)
            return BinaryOpIR(
                operator=expr.operator, left=new_left, right=new_right,
                location=loc, type_info=getattr(expr, 'type_info', None)
            )
        if isinstance(expr, UnaryOpIR):
            new_operand = self._substitute_defids_in_expr(expr.operand, substitute_map)
            if new_operand is expr.operand:
                return expr
            loc = getattr(expr, 'location', None) or SourceLocation('<generated>', 0, 0, 0, 0)
            return UnaryOpIR(operator=expr.operator, operand=new_operand, location=loc, type_info=getattr(expr, 'type_info', None))
        return expr
    
    def _expr_uses_defid(self, expr: ExpressionIR, defid: Optional[DefId]) -> bool:
        """Check if an expression contains a node with the given defid (IdentifierIR/IndexVarIR)."""
        if expr is None or defid is None or not hasattr(expr, 'accept'):
            return False

        class DefIdChecker(IRVisitor[bool]):
            def __init__(self, target_defid: DefId):
                self.target_defid = target_defid

            def visit_identifier(self, node: IdentifierIR) -> bool:
                return getattr(node, 'defid', None) == self.target_defid

            def visit_index_var(self, node: IndexVarIR) -> bool:
                return getattr(node, 'defid', None) == self.target_defid

            def visit_literal(self, node: LiteralIR) -> bool:
                return False

            def visit_binary_op(self, node: BinaryOpIR) -> bool:
                return node.left.accept(self) or node.right.accept(self)

            def visit_unary_op(self, node: UnaryOpIR) -> bool:
                return node.operand.accept(self)

            def visit_rectangular_access(self, node: RectangularAccessIR) -> bool:
                if node.array and node.array.accept(self):
                    return True
                for idx in (getattr(node, 'indices', None) or []):
                    if idx is not None and hasattr(idx, 'accept') and idx.accept(self):
                        return True
                return False

            def visit_member_access(self, node: MemberAccessIR) -> bool:
                return node.object.accept(self) if hasattr(node, 'object') else False

            def visit_cast_expression(self, node: CastExpressionIR) -> bool:
                return node.expr.accept(self) if hasattr(node, 'expr') else False

            def visit_reduction_expression(self, node: ReductionExpressionIR) -> bool:
                return node.body.accept(self) if hasattr(node, 'body') else False

            def visit_where_expression(self, node: WhereExpressionIR) -> bool:
                return node.expr.accept(self) if hasattr(node, 'expr') else False

            def visit_function_call(self, node) -> bool:
                if hasattr(node, 'arguments'):
                    return any(arg.accept(self) for arg in node.arguments)
                return False

            def visit_if_expression(self, node) -> bool:
                r = False
                if hasattr(node, 'condition') and node.condition:
                    r = r or node.condition.accept(self)
                if hasattr(node, 'then_expr') and node.then_expr:
                    r = r or node.then_expr.accept(self)
                if hasattr(node, 'else_expr') and node.else_expr:
                    r = r or node.else_expr.accept(self)
                return r

            def visit_identifier_pattern(self, node) -> bool:
                return False

            def visit_literal_pattern(self, node) -> bool:
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

            def visit_binding(self, node) -> bool:
                if is_function_binding(node) or is_einstein_binding(node):
                    return False
                if hasattr(node, 'value') and node.value:
                    return node.value.accept(self)
                return False

            def visit_array_literal(self, node) -> bool:
                return False

            def visit_array_comprehension(self, node) -> bool:
                return False

            def visit_block_expression(self, node) -> bool:
                return False

            def visit_arrow_expression(self, node) -> bool:
                return False

            def visit_pipeline_expression(self, node) -> bool:
                return False

            def visit_builtin_call(self, node) -> bool:
                return False

            def visit_module(self, node) -> bool:
                return False

            def visit_program(self, node) -> bool:
                return False

            def visit_match_expression(self, node) -> bool:
                return False

            def visit_try_expression(self, node) -> bool:
                return False

            def visit_interpolated_string(self, node) -> bool:
                return False

            def visit_tuple_expression(self, node) -> bool:
                return False

            def visit_tuple_access(self, node) -> bool:
                return False

            def visit_jagged_access(self, node) -> bool:
                return False

            def visit_lambda(self, node) -> bool:
                return node.body.accept(self) if hasattr(node, 'body') and node.body else False

            def visit_range(self, node) -> bool:
                return False

        return expr.accept(DefIdChecker(defid))

    def _find_rectangular_accesses(self, expr: ExpressionIR) -> List[RectangularAccessIR]:
        """Collect all RectangularAccessIR nodes in expression (handles binary_op, reduction body, etc.)."""
        result: List[RectangularAccessIR] = []

        def collect(e: Any) -> None:
            if e is None:
                return
            if isinstance(e, RectangularAccessIR):
                result.append(e)
                return
            if isinstance(e, BinaryOpIR):
                collect(e.left)
                collect(e.right)
                return
            for attr in ('body', 'left', 'right', 'operand', 'condition', 'then_expr', 'else_expr', 'expr', 'object'):
                if hasattr(e, attr):
                    collect(getattr(e, attr))
            if hasattr(e, 'arguments'):
                for a in getattr(e, 'arguments', []):
                    collect(a)
            if hasattr(e, 'elements'):
                for x in getattr(e, 'elements', []) or []:
                    collect(x)

        collect(expr)
        return result

    def _index_uses_var(self, idx: ExpressionIR, target_defid: DefId, var_name: Optional[str]) -> bool:
        if self._expr_uses_defid(idx, target_defid):
            return True
        if var_name and isinstance(idx, (IdentifierIR, IndexVarIR)) and getattr(idx, 'name', None) == var_name:
            return True
        return False

    def infer_reduction_var_range_from_body(
        self, expr: ExpressionIR, target_defid: DefId, location: Any, var_name: Optional[str] = None
    ) -> Optional[RangeIR]:
        """
        Infer reduction variable range from array accesses in body. Match by DefId, or by name if var_name given.
        Returns RangeIR(0, array.shape[idx_pos]) when the variable is used at index position idx_pos.
        """
        from ..shared.types import infer_literal_type, UNKNOWN
        if expr is None or target_defid is None:
            return None
        accesses = self._find_rectangular_accesses(expr)
        for access in accesses:
            if not access.indices:
                continue
            indices_flat: List[ExpressionIR] = []
            for idx in access.indices:
                if isinstance(idx, list):
                    indices_flat.extend(idx)
                else:
                    indices_flat.append(idx)
            for idx_pos, idx in enumerate(indices_flat):
                if not self._index_uses_var(idx, target_defid, var_name):
                    continue
                array_expr = getattr(access, 'array', None)
                if array_expr is None:
                    continue
                loc = location or (getattr(access, 'location', None))
                if loc is None:
                    from ..shared.source_location import SourceLocation
                    loc = SourceLocation('', 0, 0)
                if idx_pos == 0:
                    from ..shared.defid import fixed_builtin_defid
                    end_expr = BuiltinCallIR(
                        builtin_name='len',
                        args=[array_expr],
                        location=loc,
                        defid=fixed_builtin_defid('len'),
                    )
                else:
                    shape_access = MemberAccessIR(
                        object=array_expr,
                        member='shape',
                        location=loc,
                    )
                    dim_lit = LiteralIR(
                        value=idx_pos,
                        location=loc,
                        type_info=infer_literal_type(idx_pos),
                    )
                    end_expr = RectangularAccessIR(
                        array=shape_access,
                        indices=[dim_lit],
                        location=loc,
                    )
                start_lit = LiteralIR(
                    value=0,
                    location=loc,
                    type_info=infer_literal_type(0),
                )
                return RangeIR(
                    start=start_lit,
                    end=end_expr,
                    location=loc,
                    type_info=UNKNOWN,
                )
        return None

    def diagnose_reduction_range_failure(
        self, expr: Optional[ExpressionIR], target_defid: DefId
    ) -> str:
        """Return a short cause string when reduction loop var range could not be inferred. Uses DefId only."""
        if expr is None:
            return "Reduction has no body."
        accesses = self._find_rectangular_accesses(expr)
        if not accesses:
            return "Reduction body has no array accesses (e.g. A[i]); add explicit range (e.g. sum[i in 0..N](body))."
        uses_var = False
        for access in accesses:
            if not access.indices:
                continue
            indices_flat: List[ExpressionIR] = []
            for idx in access.indices:
                if isinstance(idx, list):
                    indices_flat.extend(idx)
                else:
                    indices_flat.append(idx)
            for i in indices_flat:
                if self._expr_uses_defid(i, target_defid):
                    uses_var = True
                    break
            if uses_var:
                break
        if not uses_var:
            return "Reduction body has no array access that uses this variable."
        for access in accesses:
            if isinstance(access.array, IdentifierIR):
                array_defid = getattr(access.array, 'defid', None)
                if array_defid is not None and self._resolve_var_def(array_defid) is not None:
                    return "Array shape could not be determined for the accessed array."
                break
        return "Could not determine range from body."

    def infer_clause_index_range_from_reduction_body(
        self,
        clause: Any,
        target_defid: DefId,
        location: Any,
    ) -> Optional[RangeIR]:
        """
        Infer range for a clause index (e.g. j) from array accesses in reduction body.
        E.g. result[j] = sum[k](x[j-k]) -> j in 0..(x.shape[0] + k). Uses constraint solver;
        does not substitute reduction vars (end = shape + k). Match by DefId.
        """
        from ..shared.types import infer_literal_type, UNKNOWN
        from ..shared.source_location import SourceLocation
        if clause is None or not getattr(clause, 'value', None):
            return None
        red = clause.value
        if not isinstance(red, ReductionExpressionIR) or not getattr(red, 'body', None):
            return None
        body = red.body
        loc = location or getattr(clause, 'location', None) or getattr(red, 'location', None)
        if loc is None:
            loc = SourceLocation('', 0, 0)
        prev_clause = self._current_clause
        prev_defid = getattr(self, '_target_defid', None)
        self._current_clause = clause
        self._target_defid = target_defid
        try:
            target_var = self._name_from_defid()
            if not target_var:
                return None
            # Reduction loop variable DefIds — the solved range must NOT
            # reference any of these ("forbidden defids" pattern).
            reduction_defids = set()
            for lv in (getattr(red, 'loop_vars', None) or []):
                d = getattr(lv, 'defid', None)
                if d is not None:
                    reduction_defids.add(d)
            accesses = self._find_rectangular_accesses(body)
            for access in accesses:
                if not access.indices or not isinstance(access.array, IdentifierIR):
                    continue
                array_defid = getattr(access.array, 'defid', None)
                if array_defid is None:
                    continue
                indices_flat: List[ExpressionIR] = []
                for idx in access.indices:
                    if isinstance(idx, list):
                        indices_flat.extend(idx)
                    else:
                        indices_flat.append(idx)
                for idx_pos, idx in enumerate(indices_flat):
                    if not self._expr_uses_defid(idx, target_defid):
                        continue
                    shape = self._get_array_shape(array_defid, idx_pos)
                    if shape is None:
                        continue
                    if isinstance(shape, int):
                        start_lit = LiteralIR(value=0, location=loc, type_info=infer_literal_type(0))
                        end_lit = LiteralIR(value=shape, location=loc, type_info=infer_literal_type(shape))
                        return RangeIR(start=start_lit, end=end_lit, location=loc, type_info=UNKNOWN)
                    dimension_bound = shape
                    range_result = solve_index_constraint(idx, target_var, dimension_bound)
                    if range_result is None:
                        continue
                    # Substitute reduction vars with their max value (end-1)
                    # so the range only references outer-scope variables.
                    maximized = self._maximize_over_reduction_vars(range_result)
                    if maximized is not None:
                        range_result = maximized
                    # Verify no reduction variable DefIds remain.
                    if isinstance(range_result, DynamicRange) and reduction_defids:
                        from .ir_constraint_solver import _collect_defids
                        if _collect_defids(range_result.end) & reduction_defids:
                            continue
                    start_lit = LiteralIR(value=0, location=loc, type_info=infer_literal_type(0))
                    if isinstance(range_result, StaticRange):
                        end_lit = LiteralIR(value=range_result.end, location=loc, type_info=infer_literal_type(range_result.end))
                        return RangeIR(start=start_lit, end=end_lit, location=loc, type_info=UNKNOWN)
                    if isinstance(range_result, DynamicRange) and hasattr(range_result, 'end'):
                        return RangeIR(start=start_lit, end=range_result.end, location=loc, type_info=UNKNOWN)
            return None
        finally:
            self._current_clause = prev_clause
            self._target_defid = prev_defid

    def _expr_uses_var(self, expr: ExpressionIR, target_defid: DefId) -> bool:
        """Check if an expression uses a variable by DefId. Delegates to _expr_uses_defid."""
        return self._expr_uses_defid(expr, target_defid)

    def _expr_uses_var_legacy(self, expr: ExpressionIR, var_name: str) -> bool:
        if expr is None or not hasattr(expr, 'accept'):
            return False
        class VarChecker(IRVisitor[bool]):
            def __init__(self, target: str):
                self.target = target

            def visit_identifier(self, node: IdentifierIR) -> bool:
                return getattr(node, 'name', None) == self.target

            def visit_index_var(self, node: IndexVarIR) -> bool:
                return getattr(node, 'name', None) == self.target

            def visit_literal(self, node: LiteralIR) -> bool:
                return False

            def visit_binary_op(self, node: BinaryOpIR) -> bool:
                return node.left.accept(self) or node.right.accept(self)

            def visit_unary_op(self, node: UnaryOpIR) -> bool:
                return node.operand.accept(self)

            def visit_rectangular_access(self, node: RectangularAccessIR) -> bool:
                result = node.array.accept(self) if hasattr(node, 'array') else False
                for idx in (getattr(node, 'indices', None) or []):
                    if idx is not None and hasattr(idx, 'accept') and idx.accept(self):
                        return True
                return result
            
            def visit_member_access(self, node: MemberAccessIR) -> bool:
                return node.object.accept(self) if hasattr(node, 'object') else False
            
            def visit_cast_expression(self, node: CastExpressionIR) -> bool:
                return node.expr.accept(self) if hasattr(node, 'expr') else False
            
            def visit_reduction_expression(self, node: ReductionExpressionIR) -> bool:
                return node.body.accept(self) if hasattr(node, 'body') else False
            
            def visit_where_expression(self, node: WhereExpressionIR) -> bool:
                return node.expr.accept(self) if hasattr(node, 'expr') else False
            
            # Default visitor methods (no-op for other nodes)
            def visit_array_comprehension(self, node) -> bool:
                return False
            
            def visit_array_literal(self, node) -> bool:
                return False
            
            def visit_array_pattern(self, node) -> bool:
                return False
            
            def visit_arrow_expression(self, node) -> bool:
                return False
            
            def visit_block_expression(self, node) -> bool:
                return False
            
            def visit_builtin_call(self, node) -> bool:
                return False
            
            def visit_function_call(self, node) -> bool:
                # Check all arguments for variable usage
                if hasattr(node, 'arguments'):
                    return any(arg.accept(self) for arg in node.arguments)
                return False
            
            def visit_guard_pattern(self, node) -> bool:
                return False
            
            def visit_identifier_pattern(self, node) -> bool:
                return False
            
            def visit_if_expression(self, node) -> bool:
                # Check condition, then_expr, and else_expr
                result = False
                if hasattr(node, 'condition'):
                    result = result or node.condition.accept(self)
                if hasattr(node, 'then_expr'):
                    result = result or node.then_expr.accept(self)
                if hasattr(node, 'else_expr') and node.else_expr:
                    result = result or node.else_expr.accept(self)
                return result
            
            def visit_interpolated_string(self, node) -> bool:
                return False
            
            def visit_jagged_access(self, node) -> bool:
                return False
            
            def visit_lambda(self, node) -> bool:
                return False
            
            def visit_literal_pattern(self, node) -> bool:
                return False
            
            def visit_match_expression(self, node) -> bool:
                return False
            
            def visit_module(self, node) -> bool:
                return False
            
            def visit_pipeline_expression(self, node) -> bool:
                return False
            
            def visit_program(self, node) -> bool:
                return False
            
            def visit_range(self, node) -> bool:
                return False
            
            def visit_rest_pattern(self, node) -> bool:
                return False
            
            def visit_try_expression(self, node) -> bool:
                return False
            
            def visit_tuple_access(self, node) -> bool:
                return False
            
            def visit_tuple_expression(self, node) -> bool:
                return False
            
            def visit_tuple_pattern(self, node) -> bool:
                return False
            
            def visit_wildcard_pattern(self, node) -> bool:
                return False
            
            def visit_binding(self, node) -> bool:
                if is_function_binding(node) or is_einstein_binding(node):
                    return False
                if hasattr(node, 'value') and node.value:
                    return node.value.accept(self)
                return False
        
        if expr is None or not hasattr(expr, 'accept'):
            return False
        return expr.accept(VarChecker(var_name))
    
    # ============================================
    # IR VISITOR PATTERN METHODS
    # ============================================
    
    def visit_identifier(self, node: IdentifierIR) -> None:
        pass
    
    def visit_literal(self, node: LiteralIR) -> None:
        pass
    
    def visit_rectangular_access(self, node: RectangularAccessIR) -> None:
        """Collect array accesses using our variable. Store (array_defid, index_position, index_expr, base_expr)."""
        array_defid = getattr(node.array, 'defid', None) if isinstance(node.array, IdentifierIR) else None
        indices = getattr(node, 'indices', None) or []
        target_defid = getattr(self, '_target_defid', None)
        if target_defid is None:
            return
        for idx_pos, idx in enumerate(indices):
            if idx is None:
                continue
            uses = self._expr_uses_defid(idx, target_defid)
            if uses:
                logger.debug(f"[ImplicitRangeDetector] Found access: defid {array_defid} at dimension={idx_pos}")
                self.accesses.append((array_defid, idx_pos, idx, node.array))
                    
        
        # Continue traversal
        if hasattr(node, 'array'):
            node.array.accept(self)
        for idx in (getattr(node, 'indices', None) or []):
            if idx is not None and hasattr(idx, 'accept'):
                idx.accept(self)
    
    def visit_reduction_expression(self, node: ReductionExpressionIR) -> None:
        """Visit reduction expression - traverse into body to find array accesses"""
        if node.body:
            node.body.accept(self)
    
    def visit_where_expression(self, node: WhereExpressionIR) -> None:
        """Visit where expression - traverse into inner expression"""
        if node.expr:
            node.expr.accept(self)
    
    def visit_binary_op(self, node: BinaryOpIR) -> None:
        """Traverse both sides"""
        if node.left:
            node.left.accept(self)
        if node.right:
            node.right.accept(self)
    
    def visit_unary_op(self, node: UnaryOpIR) -> None:
        """Traverse operand"""
        if node.operand:
            node.operand.accept(self)
    
    def visit_member_access(self, node: MemberAccessIR) -> None:
        """Traverse object"""
        if hasattr(node, 'object'):
            node.object.accept(self)
    
    def visit_cast_expression(self, node: CastExpressionIR) -> None:
        """Traverse expression"""
        if hasattr(node, 'expr'):
            node.expr.accept(self)
    
    # Default visitor methods (no-op for other nodes)
    def visit_array_literal(self, node: ArrayLiteralIR) -> None:
        pass
    
    def visit_function_call(self, node) -> None:
        """Traverse into function call arguments to find array accesses"""
        if hasattr(node, 'arguments'):
            for arg in node.arguments:
                arg.accept(self)
    
    def visit_if_expression(self, node) -> None:
        """Traverse into condition, then_expr, and else_expr to find array accesses"""
        if hasattr(node, 'condition'):
            node.condition.accept(self)
        if hasattr(node, 'then_expr'):
            node.then_expr.accept(self)
        if hasattr(node, 'else_expr') and node.else_expr:
            node.else_expr.accept(self)
    
    def visit_match_expression(self, node) -> None:
        pass
    
    def visit_array_comprehension(self, node) -> None:
        pass
    
    def visit_array_pattern(self, node) -> None:
        pass
    
    def visit_arrow_expression(self, node) -> None:
        pass
    
    def visit_block_expression(self, node) -> None:
        for stmt in (getattr(node, 'statements', None) or []):
            if stmt is not None and hasattr(stmt, 'accept'):
                stmt.accept(self)
        final = getattr(node, 'final_expr', None)
        if final is not None and hasattr(final, 'accept'):
            final.accept(self)

    def visit_builtin_call(self, node) -> None:
        pass
    
    def visit_binding(self, node) -> None:
        if is_function_binding(node) or is_einstein_binding(node):
            return
        if hasattr(node, 'value') and node.value:
            node.value.accept(self)
    
    def visit_guard_pattern(self, node) -> None:
        pass
    
    def visit_identifier_pattern(self, node) -> None:
        pass
    
    def visit_interpolated_string(self, node) -> None:
        pass
    
    def visit_jagged_access(self, node) -> None:
        pass
    
    def visit_lambda(self, node) -> None:
        pass
    
    def visit_literal_pattern(self, node) -> None:
        pass
    
    def visit_module(self, node) -> None:
        pass
    
    def visit_pipeline_expression(self, node) -> None:
        pass
    
    def visit_program(self, node) -> None:
        pass
    
    def visit_range(self, node) -> None:
        pass
    
    def visit_rest_pattern(self, node) -> None:
        pass
    
    def visit_try_expression(self, node) -> None:
        pass
    
    def visit_tuple_access(self, node) -> None:
        pass
    
    def visit_tuple_expression(self, node) -> None:
        pass
    
    def visit_tuple_pattern(self, node) -> None:
        pass
    
    def visit_wildcard_pattern(self, node) -> None:
        pass
    
    def _get_max_offset_from_binary_op(self, binary_op: 'BinaryOpIR', expr: ExpressionIR) -> Optional[int]:
        """
        Extract the maximum offset value from a binary operation like i+k or i-k.
        Uses _target_defid to identify the target variable (do not rely on name).
        """
        from ..ir.nodes import IdentifierIR, LiteralIR, ReductionExpressionIR

        if binary_op.operator not in (BinaryOp.ADD, BinaryOp.SUB):
            return None

        target_defid = getattr(self, '_target_defid', None)
        offset_side = None
        if target_defid is not None:
            if getattr(binary_op.left, 'defid', None) == target_defid:
                offset_side = binary_op.right
            elif getattr(binary_op.right, 'defid', None) == target_defid:
                offset_side = binary_op.left
        if offset_side is None:
            return None

        if isinstance(offset_side, LiteralIR):
            try:
                offset_val = int(offset_side.value)
                if binary_op.operator == BinaryOp.SUB:
                    return -offset_val
                return offset_val
            except (ValueError, TypeError):
                return None
        elif isinstance(offset_side, (IdentifierIR, IndexVarIR)):
            offset_defid = getattr(offset_side, 'defid', None)
            max_val = self._find_reduction_var_max_by_defid(offset_defid, expr)
            array_access_max = self._infer_var_max_from_array_accesses_by_defid(offset_defid, expr)
            
            # Take the minimum (most restrictive) of explicit range and array access inference
            # CRITICAL FIX: If we found an explicit range of 0, it's likely wrong (e.g., from 0..1 range)
            # Prefer array access inference in this case, as it's more reliable
            if array_access_max is not None:
                if max_val is None:
                    max_val = array_access_max
                elif max_val == 0:
                    # Explicit range of 0 is suspicious - prefer array access inference
                    max_val = array_access_max
                else:
                    # Use the more restrictive (smaller) value
                    max_val = min(max_val, array_access_max)
                
            
            if max_val is not None:
                if binary_op.operator == BinaryOp.SUB:
                    return -max_val
                else:
                    return max_val

        return None
    
    def _extract_mult_and_offset_from_binary_op(self, binary_op: 'BinaryOpIR', expr: ExpressionIR) -> Tuple[Optional[int], Optional[int]]:
        """Extract (mult_factor, max_offset) from binary op; uses _target_defid to identify target variable."""
        from ..ir.nodes import IdentifierIR, LiteralIR, BinaryOpIR
        target_defid = getattr(self, '_target_defid', None)
        if target_defid is None:
            return (None, None)
        if binary_op.operator == BinaryOp.ADD:
            mult_factor = None
            offset_side = None
            if isinstance(binary_op.left, BinaryOpIR) and binary_op.left.operator == BinaryOp.MUL:
                mult_factor, _ = self._extract_mult_from_binary_op(binary_op.left)
                offset_side = binary_op.right
            elif isinstance(binary_op.right, BinaryOpIR) and binary_op.right.operator == BinaryOp.MUL:
                mult_factor, _ = self._extract_mult_from_binary_op(binary_op.right)
                offset_side = binary_op.left
            else:
                offset_side = binary_op.right if getattr(binary_op.left, 'defid', None) == target_defid else binary_op.left
            max_offset = None
            if offset_side is not None:
                if mult_factor is None:
                    max_offset = self._get_max_offset_from_binary_op(binary_op, expr)
                else:
                    max_offset = self._get_max_offset_value(offset_side, expr)
            return (mult_factor, max_offset)
        if binary_op.operator == BinaryOp.MUL:
            mult_factor, _ = self._extract_mult_from_binary_op(binary_op)
            return (mult_factor, None)
        max_offset = self._get_max_offset_from_binary_op(binary_op, expr)
        return (None, max_offset)

    def _extract_mult_from_binary_op(self, binary_op: 'BinaryOpIR') -> Tuple[Optional[int], Optional[ExpressionIR]]:
        """Extract multiplication factor from i*factor or factor*i; uses _target_defid."""
        from ..ir.nodes import IdentifierIR, LiteralIR
        target_defid = getattr(self, '_target_defid', None)
        if target_defid is None or binary_op.operator != BinaryOp.MUL:
            return (None, None)
        if getattr(binary_op.left, 'defid', None) == target_defid and isinstance(binary_op.right, LiteralIR) and isinstance(binary_op.right.value, (int, float)):
            return (int(binary_op.right.value), binary_op.right)
        if getattr(binary_op.right, 'defid', None) == target_defid and isinstance(binary_op.left, LiteralIR) and isinstance(binary_op.left.value, (int, float)):
            return (int(binary_op.left.value), binary_op.left)
        return (None, None)
    
    def _get_max_offset_value(self, expr: ExpressionIR, enclosing_expr: ExpressionIR) -> Optional[int]:
        """Get maximum value of an offset expression by defid (e.g. max di from reduction loop_var_ranges)."""
        from ..ir.nodes import IdentifierIR, ReductionExpressionIR, LiteralIR, RangeIR
        offset_defid = getattr(expr, 'defid', None)
        if offset_defid is None or not isinstance(expr, (IdentifierIR, IndexVarIR)):
            return None
        if enclosing_expr:
            max_val = self._find_reduction_var_max_by_defid(offset_defid, enclosing_expr)
            if max_val is not None:
                return max_val
            class ReductionFinder(IRVisitor[Optional[int]]):
                def __init__(self, target_defid: DefId):
                    self.target_defid = target_defid
                    self.max_val = None
                def visit_reduction_expression(self, node: ReductionExpressionIR) -> Optional[int]:
                    if hasattr(node, 'loop_var_ranges') and node.loop_var_ranges and self.target_defid in node.loop_var_ranges:
                        range_ir = node.loop_var_ranges[self.target_defid]
                        if isinstance(range_ir, RangeIR):
                            if hasattr(range_ir, 'end') and isinstance(range_ir.end, LiteralIR):
                                end_val = range_ir.end.value
                                if isinstance(end_val, (int, float)):
                                    self.max_val = int(end_val) - 1
                                    return self.max_val
                            elif hasattr(range_ir, 'end'):
                                from ..passes.const_folding import ConstantFolder
                                folder = ConstantFolder()
                                try:
                                    end_eval = range_ir.end.accept(folder)
                                    if isinstance(end_eval, LiteralIR) and isinstance(end_eval.value, (int, float)):
                                        self.max_val = int(end_eval.value) - 1
                                        return self.max_val
                                except Exception:
                                    pass
                    if hasattr(node, 'body'):
                        node.body.accept(self)
                    return self.max_val

                def visit_binary_op(self, node: BinaryOpIR) -> Optional[int]:
                    node.left.accept(self)
                    node.right.accept(self)
                    return self.max_val

                def visit_rectangular_access(self, node: RectangularAccessIR) -> Optional[int]:
                    for idx in (getattr(node, 'indices', None) or []):
                        if idx is not None and hasattr(idx, 'accept'):
                            idx.accept(self)
                    return self.max_val

                def visit_identifier(self, node: IdentifierIR) -> Optional[int]:
                    return self.max_val

                def visit_literal(self, node) -> Optional[int]:
                    return self.max_val

                def visit_unary_op(self, node) -> Optional[int]:
                    return node.operand.accept(self) if hasattr(node, 'operand') else self.max_val

                def visit_function_call(self, node) -> Optional[int]:
                    if hasattr(node, 'arguments'):
                        for arg in node.arguments:
                            arg.accept(self)
                    return self.max_val

                def visit_binding(self, node) -> Optional[int]:
                    if is_function_binding(node):
                        return self.max_val
                    if is_einstein_binding(node):
                        for clause in getattr(node, 'clauses', None) or []:
                            if hasattr(clause, 'value') and clause.value:
                                clause.value.accept(self)
                        return self.max_val
                    if hasattr(node, 'value'):
                        node.value.accept(self)
                    return self.max_val

                def visit_array_literal(self, node) -> Optional[int]:
                    return self.max_val

                def visit_array_comprehension(self, node) -> Optional[int]:
                    if hasattr(node, 'expr'):
                        node.expr.accept(self)
                    return self.max_val

                def visit_where_expression(self, node) -> Optional[int]:
                    if hasattr(node, 'expr'):
                        node.expr.accept(self)
                    return self.max_val

                def visit_member_access(self, node) -> Optional[int]:
                    if hasattr(node, 'object'):
                        node.object.accept(self)
                    return self.max_val

                def visit_cast_expression(self, node) -> Optional[int]:
                    if hasattr(node, 'expr'):
                        node.expr.accept(self)
                    return self.max_val

                def visit_if_expression(self, node) -> Optional[int]:
                    if hasattr(node, 'condition'):
                        node.condition.accept(self)
                    if hasattr(node, 'then_expr'):
                        node.then_expr.accept(self)
                    if hasattr(node, 'else_expr'):
                        node.else_expr.accept(self)
                    return self.max_val

                def visit_lambda(self, node) -> Optional[int]:
                    if hasattr(node, 'body'):
                        node.body.accept(self)
                    return self.max_val

                def visit_block_expression(self, node) -> Optional[int]:
                    if hasattr(node, 'statements'):
                        for stmt in node.statements:
                            stmt.accept(self)
                    if hasattr(node, 'final_expr'):
                        node.final_expr.accept(self)
                    return self.max_val

                def visit_program(self, node) -> Optional[int]:
                    return self.max_val

                def visit_range(self, node) -> Optional[int]:
                    return self.max_val

                def visit_jagged_access(self, node) -> Optional[int]:
                    return self.max_val

                def visit_tuple_expression(self, node) -> Optional[int]:
                    return self.max_val

                def visit_tuple_access(self, node) -> Optional[int]:
                    return self.max_val

                def visit_interpolated_string(self, node) -> Optional[int]:
                    return self.max_val

                def visit_try_expression(self, node) -> Optional[int]:
                    return self.max_val

                def visit_match_expression(self, node) -> Optional[int]:
                    return self.max_val

                def visit_arrow_expression(self, node) -> Optional[int]:
                    return self.max_val

                def visit_pipeline_expression(self, node) -> Optional[int]:
                    return self.max_val

                def visit_builtin_call(self, node) -> Optional[int]:
                    return self.max_val

                def visit_module(self, node) -> Optional[int]:
                    return self.max_val

                def visit_literal_pattern(self, node) -> Optional[int]:
                    return self.max_val

                def visit_identifier_pattern(self, node) -> Optional[int]:
                    return self.max_val

                def visit_wildcard_pattern(self, node) -> Optional[int]:
                    return self.max_val

                def visit_tuple_pattern(self, node) -> Optional[int]:
                    return self.max_val

                def visit_array_pattern(self, node) -> Optional[int]:
                    return self.max_val

                def visit_rest_pattern(self, node) -> Optional[int]:
                    return self.max_val

                def visit_guard_pattern(self, node) -> Optional[int]:
                    return self.max_val
                
            finder = ReductionFinder(offset_defid)
            max_val = enclosing_expr.accept(finder)
            if max_val is not None:
                return max_val
            if isinstance(enclosing_expr, ReductionExpressionIR) and hasattr(enclosing_expr, 'loop_var_ranges') and enclosing_expr.loop_var_ranges and offset_defid in enclosing_expr.loop_var_ranges:
                range_ir = enclosing_expr.loop_var_ranges[offset_defid]
                if isinstance(range_ir, RangeIR) and hasattr(range_ir, 'end') and isinstance(range_ir.end, LiteralIR):
                    end_val = range_ir.end.value
                    if isinstance(end_val, (int, float)):
                        return int(end_val) - 1
            return max_val
        return None
    
    def _find_reduction_var_max_by_defid(self, defid: Optional[DefId], expr: ExpressionIR) -> Optional[int]:
        """Find max value of a reduction variable by defid (from loop_var_ranges)."""
        if defid is None:
            return None
        from ..ir.nodes import ReductionExpressionIR, LiteralIR, RangeIR
        if isinstance(expr, ReductionExpressionIR):
            if hasattr(expr, 'loop_var_ranges') and expr.loop_var_ranges and defid in expr.loop_var_ranges:
                range_ir = expr.loop_var_ranges[defid]
                if isinstance(range_ir, RangeIR) and hasattr(range_ir, 'end') and isinstance(range_ir.end, LiteralIR):
                    end_val = range_ir.end.value
                    if isinstance(end_val, (int, float)):
                        return int(end_val) - 1
        return self._find_reduction_var_max_visitor(defid, expr)

    def _find_reduction_var_max_visitor(self, target_defid: DefId, expr: ExpressionIR) -> Optional[int]:
        """Visitor that finds reduction loop_var_ranges[target_defid] and returns max (end-1)."""
        from ..ir.nodes import ReductionExpressionIR, LiteralIR, RangeIR
        result = [None]

        class ReductionMaxFinder(IRVisitor[None]):
            def visit_reduction_expression(self, node: ReductionExpressionIR) -> None:
                if hasattr(node, 'loop_var_ranges') and node.loop_var_ranges and target_defid in node.loop_var_ranges:
                    range_ir = node.loop_var_ranges[target_defid]
                    if isinstance(range_ir, RangeIR) and hasattr(range_ir, 'end') and isinstance(range_ir.end, LiteralIR):
                        end_val = range_ir.end.value
                        if isinstance(end_val, (int, float)):
                            result[0] = int(end_val) - 1
                if hasattr(node, 'body') and node.body:
                    node.body.accept(self)

            def visit_binary_op(self, node: BinaryOpIR) -> None:
                if node.left:
                    node.left.accept(self)
                if node.right:
                    node.right.accept(self)

            def visit_rectangular_access(self, node: RectangularAccessIR) -> None:
                if node.array:
                    node.array.accept(self)
                for idx in (getattr(node, 'indices', None) or []):
                    if idx is not None and hasattr(idx, 'accept'):
                        idx.accept(self)

        for _ in (None,):
            pass
        missing_visitors = [
            'visit_identifier', 'visit_literal', 'visit_member_access', 'visit_cast_expression',
            'visit_where_expression', 'visit_function_call', 'visit_if_expression',
            'visit_array_literal', 'visit_array_comprehension', 'visit_block_expression',
            'visit_arrow_expression', 'visit_pipeline_expression', 'visit_builtin_call',
            'visit_function_def', 'visit_constant_def',
            'visit_einstein_declaration', 'visit_module', 'visit_program',
            'visit_match_expression', 'visit_try_expression', 'visit_interpolated_string',
            'visit_tuple_expression', 'visit_tuple_access', 'visit_jagged_access',
            'visit_unary_op', 'visit_range', 'visit_identifier_pattern', 'visit_literal_pattern',
            'visit_wildcard_pattern', 'visit_tuple_pattern', 'visit_array_pattern',
            'visit_rest_pattern', 'visit_guard_pattern', 'visit_variable_declaration',
        ]
        finder = ReductionMaxFinder()
        for m in missing_visitors:
            if not hasattr(finder, m):
                setattr(finder, m, lambda self, n: None)
        expr.accept(finder)
        return result[0]

    def _find_reduction_var_max(self, target_defid: Optional[DefId], expr: ExpressionIR) -> Optional[int]:
        """Find max from reduction body by DefId. Delegates to _find_reduction_var_max_by_defid."""
        return self._find_reduction_var_max_by_defid(target_defid, expr) if target_defid else None

    def _find_reduction_var_max_old(self, var_name: str, expr: ExpressionIR) -> Optional[int]:
        """Find the maximum value of a reduction variable by searching for reduction expressions"""
        from ..ir.nodes import ReductionExpressionIR, LiteralIR
        
        if isinstance(expr, ReductionExpressionIR):
            if hasattr(expr, 'loop_var_ranges') and expr.loop_var_ranges:
                var_defid = next((getattr(i, 'defid', None) for i in (expr.loop_vars or []) if getattr(i, 'name', None) == var_name), None)
                if var_defid is not None and var_defid in expr.loop_var_ranges:
                    range_ir = expr.loop_var_ranges[var_defid]
                    if hasattr(range_ir, 'end'):
                        if isinstance(range_ir.end, LiteralIR):
                            # Range is [start, end), so max value is end - 1
                            end_val = range_ir.end.value
                            if isinstance(end_val, (int, float)):
                                max_val = int(end_val) - 1
                                return max_val
        
        class ReductionFinder(IRVisitor[None]):
            def __init__(self, target_var: str):
                self.target_var = target_var
                self.max_val = None
            
            def visit_reduction_expression(self, node: ReductionExpressionIR) -> None:
                # Check if this reduction defines the target variable
                if hasattr(node, 'loop_var_ranges') and node.loop_var_ranges:
                    var_defid = next((getattr(i, 'defid', None) for i in (node.loop_vars or []) if getattr(i, 'name', None) == var_name), None)
                    if var_defid is not None and var_defid in node.loop_var_ranges:
                        range_ir = node.loop_var_ranges[var_defid]
                        if hasattr(range_ir, 'end') and isinstance(range_ir.end, LiteralIR):
                            end_val = range_ir.end.value
                            if isinstance(end_val, (int, float)):
                                self.max_val = int(end_val) - 1
                
                # Continue searching in the body
                if hasattr(node, 'body'):
                    node.body.accept(self)
            
            # Default visitor methods (no-op for other nodes)
            def visit_array_comprehension(self, node) -> None:
                pass
            def visit_array_literal(self, node) -> None:
                pass
            def visit_array_pattern(self, node) -> None:
                pass
            def visit_arrow_expression(self, node) -> None:
                pass
            def visit_binary_op(self, node) -> None:
                if hasattr(node, 'left'):
                    node.left.accept(self)
                if hasattr(node, 'right'):
                    node.right.accept(self)
            def visit_block_expression(self, node) -> None:
                pass
            def visit_builtin_call(self, node) -> None:
                pass
            def visit_cast_expression(self, node) -> None:
                if hasattr(node, 'expr'):
                    node.expr.accept(self)
            def visit_function_call(self, node) -> None:
                if hasattr(node, 'arguments'):
                    for arg in node.arguments:
                        arg.accept(self)
            def visit_guard_pattern(self, node) -> None:
                pass
            def visit_identifier(self, node) -> None:
                pass
            def visit_identifier_pattern(self, node) -> None:
                pass
            def visit_if_expression(self, node) -> None:
                if hasattr(node, 'condition'):
                    node.condition.accept(self)
                if hasattr(node, 'then_expr'):
                    node.then_expr.accept(self)
                if hasattr(node, 'else_expr') and node.else_expr:
                    node.else_expr.accept(self)
            def visit_interpolated_string(self, node) -> None:
                pass
            def visit_jagged_access(self, node) -> None:
                pass
            def visit_lambda(self, node) -> None:
                pass
            def visit_literal(self, node) -> None:
                pass
            def visit_literal_pattern(self, node) -> None:
                pass
            def visit_match_expression(self, node) -> None:
                pass
            def visit_member_access(self, node) -> None:
                if hasattr(node, 'object'):
                    node.object.accept(self)
            def visit_module(self, node) -> None:
                pass
            def visit_pipeline_expression(self, node) -> None:
                pass
            def visit_program(self, node) -> None:
                pass
            def visit_range(self, node) -> None:
                pass
            def visit_rectangular_access(self, node) -> None:
                if hasattr(node, 'array'):
                    node.array.accept(self)
                for idx in (getattr(node, 'indices', None) or []):
                    if idx is not None and hasattr(idx, 'accept'):
                        idx.accept(self)
            def visit_rest_pattern(self, node) -> None:
                pass
            def visit_try_expression(self, node) -> None:
                pass
            def visit_tuple_access(self, node) -> None:
                pass
            def visit_tuple_expression(self, node) -> None:
                pass
            def visit_tuple_pattern(self, node) -> None:
                pass
            def visit_unary_op(self, node) -> None:
                if hasattr(node, 'operand'):
                    node.operand.accept(self)
            def visit_binding(self, node) -> None:
                if is_function_binding(node) or is_einstein_binding(node):
                    return
                if hasattr(node, 'value'):
                    node.value.accept(self)
            def visit_where_expression(self, node) -> None:
                if hasattr(node, 'expr'):
                    node.expr.accept(self)
            def visit_wildcard_pattern(self, node) -> None:
                pass
        
        finder = ReductionFinder(var_name)
        expr.accept(finder)
        return finder.max_val

    def _infer_var_max_from_array_accesses_by_defid(self, target_defid: Optional[DefId], expr: ExpressionIR) -> Optional[int]:
        if target_defid is None:
            return None
        prev = getattr(self, '_target_defid', None)
        self._target_defid = target_defid
        try:
            array_accesses = self._find_array_accesses(expr)
        finally:
            self._target_defid = prev
        max_val = None
        for array_defid, index_position, index_expr, base_expr in array_accesses:
            if array_defid is None:
                continue
            shape = self._get_array_shape(array_defid, index_position)
            if shape is None or not isinstance(shape, int):
                continue
            from ..ir.nodes import BinaryOpIR, LiteralIR
            if isinstance(index_expr, (IdentifierIR, IndexVarIR)) and getattr(index_expr, 'defid', None) == target_defid:
                candidate_max = shape - 1
                if max_val is None or candidate_max < max_val:
                    max_val = candidate_max
            elif isinstance(index_expr, BinaryOpIR) and index_expr.operator in (BinaryOp.ADD, BinaryOp.SUB):
                other_defid = None
                if getattr(index_expr.left, 'defid', None) == target_defid and isinstance(index_expr.right, (IdentifierIR, IndexVarIR)):
                    other_defid = getattr(index_expr.right, 'defid', None)
                elif getattr(index_expr.right, 'defid', None) == target_defid and isinstance(index_expr.left, (IdentifierIR, IndexVarIR)):
                    other_defid = getattr(index_expr.left, 'defid', None)
                other_var_max = None
                if other_defid and self._current_clause:
                    var_ranges = getattr(self._current_clause, 'variable_ranges', None) or {}
                    range_ir = var_ranges.get(other_defid)
                    if range_ir and hasattr(range_ir, 'end') and isinstance(range_ir.end, LiteralIR):
                        end_val = range_ir.end.value
                        if isinstance(end_val, (int, float)):
                            other_var_max = int(end_val) - 1
                if index_expr.operator == BinaryOp.ADD:
                    candidate_max = max(0, shape - other_var_max - 1) if other_var_max is not None else shape - 1
                else:
                    candidate_max = shape - 1
                if max_val is None or candidate_max < max_val:
                    max_val = candidate_max
        return max_val

    def _infer_var_max_from_array_accesses(self, target_defid: Optional[DefId], expr: ExpressionIR) -> Optional[int]:
        return self._infer_var_max_from_array_accesses_by_defid(target_defid, expr) if target_defid else None

    def get_range_from_where_clause(self, where_clause: Any, defid: Optional[DefId] = None) -> Any:
        if not where_clause:
            return None
        if hasattr(where_clause, 'ranges') and where_clause.ranges and defid is not None:
            range_obj = where_clause.ranges.get(defid)
            if range_obj is not None:
                if isinstance(range_obj, RangeIR):
                    return range_obj
                if isinstance(range_obj, range):
                    return range_obj
        if not hasattr(where_clause, 'constraints') or not where_clause.constraints or defid is None:
            return None
        # Flatten AND chains so ``a && b`` becomes [a, b]
        flat_constraints: list = []
        def _flatten(c):
            if isinstance(c, BinaryOpIR) and getattr(c, 'operator', None) == BinaryOp.AND:
                _flatten(c.left)
                _flatten(c.right)
            else:
                flat_constraints.append(c)
        for c in where_clause.constraints:
            _flatten(c)
        for constraint in flat_constraints:
            if isinstance(constraint, BinaryOpIR) and getattr(constraint, "operator", None) == BinaryOp.IN:
                left = constraint.left
                if isinstance(left, (IdentifierIR, IndexVarIR)) and getattr(left, 'defid', None) == defid:
                    if isinstance(constraint.right, RangeIR):
                        r = constraint.right
                        start_expr, end_expr = r.start, r.end
                        start_val = None
                        if isinstance(start_expr, LiteralIR):
                            try:
                                start_val = int(start_expr.value)
                            except (ValueError, TypeError):
                                pass
                        end_val = None
                        if isinstance(end_expr, LiteralIR):
                            try:
                                end_val = int(end_expr.value)
                            except (ValueError, TypeError):
                                pass
                        if start_val is not None and end_val is not None:
                            from ..shared.types import PrimitiveType
                            loc = getattr(r, "location", None) or SourceLocation("", 0, 0)
                            return RangeIR(
                                start=LiteralIR(value=start_val, location=loc, type_info=infer_literal_type(start_val)),
                                end=LiteralIR(value=end_val, location=loc, type_info=infer_literal_type(end_val)),
                                location=loc,
                                type_info=UNKNOWN,
                            )
                        return r
            # Handle relational upper bound: ``offset + k * step < bound``
            # Derives k in 0..(bound - offset + step - 1) / step
            if isinstance(constraint, BinaryOpIR) and getattr(constraint, "operator", None) == BinaryOp.LT:
                upper = self._extract_linear_upper_bound(constraint.left, constraint.right, defid)
                if upper is not None:
                    return upper
        return None

    def _extract_linear_upper_bound(
        self, lhs: ExpressionIR, rhs: ExpressionIR, target_defid: DefId
    ) -> Optional[RangeIR]:
        """Solve ``offset + k * step < bound`` for *k*, returning ``0..ceil_div(bound - offset, step)``.

        Handles patterns emitted by slice_impl:
          - ``start + k * step < end``  →  k in 0 .. (end - start + step - 1) / step
        """
        coeff = self._extract_linear_coeff(lhs, target_defid)
        if coeff is None:
            return None
        offset_expr, step_expr = coeff
        bound_expr = rhs
        loc = getattr(lhs, 'location', None) or SourceLocation("", 0, 0)
        zero = LiteralIR(value=0, location=loc, type_info=infer_literal_type(0))
        one = LiteralIR(value=1, location=loc, type_info=infer_literal_type(1))
        # end = (bound - offset + step - 1) / step
        numerator = BinaryOpIR(
            operator=BinaryOp.ADD,
            left=BinaryOpIR(
                operator=BinaryOp.SUB,
                left=bound_expr,
                right=offset_expr,
                location=loc,
            ),
            right=BinaryOpIR(
                operator=BinaryOp.SUB,
                left=step_expr,
                right=one,
                location=loc,
            ),
            location=loc,
        )
        end_expr = BinaryOpIR(
            operator=BinaryOp.DIV,
            left=numerator,
            right=step_expr,
            location=loc,
        )
        return RangeIR(start=zero, end=end_expr, location=loc, type_info=UNKNOWN)

    def _extract_linear_coeff(
        self, expr: ExpressionIR, target_defid: DefId
    ) -> Optional[Tuple[ExpressionIR, ExpressionIR]]:
        """Match ``offset + var * step`` (or ``var * step + offset``) and return ``(offset, step)``.

        Returns None if *expr* is not a linear expression in *target_defid*.
        """
        if not isinstance(expr, BinaryOpIR) or expr.operator != BinaryOp.ADD:
            return None
        for addend, other in [(expr.left, expr.right), (expr.right, expr.left)]:
            if isinstance(addend, BinaryOpIR) and addend.operator == BinaryOp.MUL:
                for factor, coeff in [(addend.left, addend.right), (addend.right, addend.left)]:
                    if isinstance(factor, (IdentifierIR, IndexVarIR)) and getattr(factor, 'defid', None) == target_defid:
                        return (other, coeff)
        return None

    def infer_reduction_ranges_from_where(self, expr: ReductionExpressionIR) -> None:
        if not expr.where_clause or not getattr(expr.where_clause, 'constraints', None):
            return
        reduction_defids = set()
        for ident in (expr.loop_vars or []):
            d = getattr(ident, 'defid', None)
            if d is not None:
                reduction_defids.add(d)
        if not reduction_defids:
            return
        loc = getattr(expr, 'location', None)
        if loc is None:
            loc = SourceLocation("", 0, 0)
        for constraint in expr.where_clause.constraints:
            if not isinstance(constraint, BinaryOpIR):
                continue
            op = getattr(constraint, 'operator', None)
            if op not in (BinaryOp.LT, BinaryOp.LE):
                continue
            left = getattr(constraint, 'left', None)
            right = getattr(constraint, 'right', None)
            if not isinstance(left, IdentifierIR) or right is None:
                continue
            left_defid = getattr(left, 'defid', None)
            if left_defid is None or left_defid not in reduction_defids:
                continue
            if left_defid in expr.loop_var_ranges:
                continue
            right_defid = getattr(right, 'defid', None)
            if right_defid is not None and right_defid in reduction_defids:
                continue
            start_ir = LiteralIR(value=0, location=loc, type_info=infer_literal_type(0))
            range_ir = RangeIR(start=start_ir, end=right, location=loc, type_info=UNKNOWN)
            expr.loop_var_ranges[left_defid] = range_ir

    def find_reduction_in_value(self, expr: ExpressionIR) -> Optional[ReductionExpressionIR]:
        return self._find_reduction_in_expr(expr)

    def _get_output_var_index_from_reduction_body(
        self, red_expr: ReductionExpressionIR, target_defid: DefId
    ) -> Optional[ExpressionIR]:
        if not red_expr or not getattr(red_expr, "body", None):
            return None

        def contains_var(e: ExpressionIR) -> bool:
            if isinstance(e, (IdentifierIR, IndexVarIR)):
                return getattr(e, 'defid', None) == target_defid
            if isinstance(e, BinaryOpIR):
                return contains_var(e.left) or contains_var(e.right)
            if hasattr(e, "operand"):
                return contains_var(getattr(e, "operand"))
            return False

        def collect_indices(e: ExpressionIR) -> List[ExpressionIR]:
            out: List[ExpressionIR] = []
            if isinstance(e, RectangularAccessIR) and e.indices:
                for index in e.indices:
                    if isinstance(index, list):
                        for sub in index:
                            if contains_var(sub):
                                out.append(sub)
                    elif contains_var(index):
                        out.append(index)
            if isinstance(e, BinaryOpIR):
                out.extend(collect_indices(e.left))
                out.extend(collect_indices(e.right))
            if hasattr(e, "body"):
                out.extend(collect_indices(getattr(e, "body")))
            if hasattr(e, "arguments"):
                for arg in getattr(e, "arguments", []):
                    out.extend(collect_indices(arg))
            if hasattr(e, "value"):
                out.extend(collect_indices(getattr(e, "value")))
            if hasattr(e, "operand"):
                out.extend(collect_indices(getattr(e, "operand")))
            if isinstance(e, IfExpressionIR):
                out.extend(collect_indices(e.condition))
                out.extend(collect_indices(e.then_expr))
                if e.else_expr:
                    out.extend(collect_indices(e.else_expr))
            return out

        indices = collect_indices(red_expr.body)
        for index in indices:
            if isinstance(index, BinaryOpIR):
                return index
        for index in indices:
            if index is not None:
                return index
        return None

    def _solve_index_constraint_ir(
        self,
        index_expr: ExpressionIR,
        target_defid: DefId,
        dimension_bound: ExpressionIR,
        loc: Any,
        target_name: Optional[str] = None,
    ) -> Optional[ExpressionIR]:
        if loc is None and hasattr(dimension_bound, 'location'):
            loc = dimension_bound.location
        if loc is None:
            return None

        def _collect_defids(expr):
            """Collect all variable DefIds referenced by an expression."""
            out = set()
            if expr is None:
                return out
            if isinstance(expr, (IdentifierIR, IndexVarIR)):
                d = getattr(expr, 'defid', None)
                if d is not None:
                    out.add(d)
                return out
            if isinstance(expr, BinaryOpIR):
                out |= _collect_defids(expr.left)
                out |= _collect_defids(expr.right)
                return out
            if hasattr(expr, 'operand'):
                out |= _collect_defids(getattr(expr, 'operand'))
            if isinstance(expr, RectangularAccessIR):
                out |= _collect_defids(expr.array)
                for idx in (expr.indices or []):
                    out |= _collect_defids(idx)
            if isinstance(expr, MemberAccessIR):
                out |= _collect_defids(getattr(expr, 'object', None))
            return out

        # Defids that appear in index_expr but are NOT the target — these are
        # inner-scope variables (e.g. reduction vars).  If the solved result
        # still references any of them it means we moved an inner-scope var
        # into the range bound, which is invalid ("forbidden defids" check).
        other_defids = _collect_defids(index_expr) - {target_defid}

        def contains_target(expr: ExpressionIR) -> bool:
            if isinstance(expr, (IdentifierIR, IndexVarIR)):
                if getattr(expr, 'defid', None) == target_defid:
                    return True
                if target_name and getattr(expr, 'name', None) == target_name:
                    return True
                return False
            if isinstance(expr, BinaryOpIR):
                return contains_target(expr.left) or contains_target(expr.right)
            if hasattr(expr, 'operand'):
                return contains_target(getattr(expr, 'operand'))
            return False

        def solve(expr: ExpressionIR, bound: ExpressionIR) -> Optional[ExpressionIR]:
            if isinstance(expr, (IdentifierIR, IndexVarIR)):
                if getattr(expr, 'defid', None) == target_defid:
                    return bound
                if target_name and getattr(expr, 'name', None) == target_name:
                    return bound
                return None
            if not isinstance(expr, BinaryOpIR):
                return None
            left, right = expr.left, expr.right
            left_has = contains_target(left)
            right_has = contains_target(right)
            op = expr.operator
            if op == BinaryOp.ADD:
                if left_has and not right_has:
                    return solve(left, BinaryOpIR(operator=BinaryOp.SUB, left=bound, right=right, location=loc))
                if right_has and not left_has:
                    return solve(right, BinaryOpIR(operator=BinaryOp.SUB, left=bound, right=left, location=loc))
                return None
            if op == BinaryOp.SUB:
                if left_has and not right_has:
                    return solve(left, BinaryOpIR(operator=BinaryOp.ADD, left=bound, right=right, location=loc))
                return None
            if op == BinaryOp.MUL:
                if isinstance(left, (IdentifierIR, IndexVarIR)) and getattr(left, 'defid', None) == target_defid and not contains_target(right):
                    one = LiteralIR(value=1, location=loc, type_info=infer_literal_type(1))
                    den_minus = BinaryOpIR(operator=BinaryOp.SUB, left=right, right=one, location=loc)
                    num = BinaryOpIR(operator=BinaryOp.ADD, left=bound, right=den_minus, location=loc)
                    return BinaryOpIR(operator=BinaryOp.DIV, left=num, right=right, location=loc)
                if isinstance(right, (IdentifierIR, IndexVarIR)) and getattr(right, 'defid', None) == target_defid and not contains_target(left):
                    one = LiteralIR(value=1, location=loc, type_info=infer_literal_type(1))
                    den_minus = BinaryOpIR(operator=BinaryOp.SUB, left=left, right=one, location=loc)
                    num = BinaryOpIR(operator=BinaryOp.ADD, left=bound, right=den_minus, location=loc)
                    return BinaryOpIR(operator=BinaryOp.DIV, left=num, right=left, location=loc)
                return None
            if op == BinaryOp.DIV:
                if isinstance(left, (IdentifierIR, IndexVarIR)) and getattr(left, 'defid', None) == target_defid and not contains_target(right):
                    return BinaryOpIR(operator=BinaryOp.MUL, left=bound, right=right, location=loc)
                return None
            return None

        result = solve(index_expr, dimension_bound)
        if result is not None and other_defids:
            result_defids = _collect_defids(result)
            if result_defids & other_defids:
                return None
        return result

    def _build_output_range_end_from_constraint(
        self,
        shape_dim: ExpressionIR,
        red_expr: ReductionExpressionIR,
        idx: ExpressionIR,
        target_defid: DefId,
        location: Any,
        loop_var_ranges: Optional[Dict[DefId, Any]] = None,
        target_name: Optional[str] = None,
    ) -> Optional[ExpressionIR]:
        loc = location or (shape_dim.location if hasattr(shape_dim, "location") else None)
        if loc is None:
            return None
        constraint_idx = self._get_output_var_index_from_reduction_body(red_expr, target_defid) or idx
        return self._solve_index_constraint_ir(constraint_idx, target_defid, shape_dim, loc, target_name=target_name)

    def _compute_array_literal_dimension(self, array_lit: Any, dimension: int) -> Optional[int]:
        if not isinstance(array_lit, ArrayLiteralIR):
            return None
        current = array_lit
        for _ in range(dimension):
            if isinstance(current, ArrayLiteralIR) and len(current.elements) > 0:
                current = current.elements[0]
            else:
                return None
        if isinstance(current, ArrayLiteralIR):
            return len(current.elements)
        return None

    def _evaluate_constant_expression(self, expr: ExpressionIR) -> Optional[Any]:
        import numpy as np
        if isinstance(expr, LiteralIR):
            return expr.value
        if isinstance(expr, ArrayLiteralIR):
            elements = []
            for elem in expr.elements:
                if isinstance(elem, LiteralIR):
                    elements.append(elem.value)
                elif isinstance(elem, ArrayLiteralIR):
                    nested = self._evaluate_constant_expression(elem)
                    if nested is not None:
                        elements.append(nested)
            if elements:
                try:
                    return np.array(elements)
                except Exception:
                    return elements
        return None

    def infer_range_from_program_statements(
        self, tcx: Any, target_defid: DefId, einstein_node: Any, clause_indices: Optional[List] = None
    ) -> Optional[range]:
        if not tcx:
            return None
        try:
            from .shape_analysis import UnifiedShapeAnalysisPass
            shape_data = tcx.get_analysis(UnifiedShapeAnalysisPass)
        except RuntimeError:
            return None
        if not shape_data:
            return None
        expr_shapes = shape_data.get('expr_shapes', {}) if isinstance(shape_data, dict) else shape_data
        defid_shapes = shape_data.get('defid_shapes', {}) if isinstance(shape_data, dict) else {}
        _value = None
        if einstein_node:
            if is_einstein_binding(einstein_node) and einstein_node.clauses:
                _value = einstein_node.clauses[0].value
            else:
                _value = getattr(einstein_node, 'value', None)
        if not _value:
            return None

        def find_array_accesses(expr: Any) -> List[Any]:
            accesses = []
            if isinstance(expr, RectangularAccessIR):
                accesses.append(expr)
            elif hasattr(expr, 'left') and hasattr(expr, 'right'):
                accesses.extend(find_array_accesses(expr.left))
                accesses.extend(find_array_accesses(expr.right))
            elif hasattr(expr, 'operand'):
                accesses.extend(find_array_accesses(expr.operand))
            elif hasattr(expr, 'body'):
                accesses.extend(find_array_accesses(expr.body))
            elif hasattr(expr, 'value'):
                accesses.extend(find_array_accesses(expr.value))
            elif hasattr(expr, 'arguments'):
                for arg in expr.arguments:
                    accesses.extend(find_array_accesses(arg))
            return accesses

        array_accesses = find_array_accesses(_value)
        for access in array_accesses:
            if not isinstance(access, RectangularAccessIR):
                continue
            for idx_pos, idx in enumerate(access.indices or []):
                if isinstance(idx, (IdentifierIR, IndexVarIR)) and getattr(idx, 'defid', None) == target_defid:
                    array_expr = access.array
                    if hasattr(array_expr, 'defid') and array_expr.defid:
                        shape = defid_shapes.get(array_expr.defid)
                        if shape and isinstance(shape, (list, tuple)) and idx_pos < len(shape):
                            size = shape[idx_pos]
                            if isinstance(size, int):
                                return range(0, size)
                    for ex, shape in expr_shapes.items():
                        if hasattr(array_expr, 'defid') and hasattr(ex, 'defid') and array_expr.defid == ex.defid:
                            if isinstance(shape, (list, tuple)) and idx_pos < len(shape):
                                size = shape[idx_pos]
                                if isinstance(size, int):
                                    return range(0, size)
                        if array_expr is ex and isinstance(shape, (list, tuple)) and idx_pos < len(shape):
                            size = shape[idx_pos]
                            if isinstance(size, int):
                                return range(0, size)
        return None

    def infer_range_from_array_access(
        self,
        target_defid: DefId,
        expr: ExpressionIR,
        einstein_node: Any,
        loop_var_ranges: Optional[Dict[DefId, Any]],
        clause_indices: Optional[List],
        einstein_clause: Any,
        location: Any,
    ) -> Optional[range]:
        import numpy as np

        def _clause_for(node: Any) -> Any:
            if node is None:
                return None
            if is_einstein_binding(node):
                return node.clauses[0] if node.clauses else None
            return node

        _clause = _clause_for(einstein_node)
        _clause_indices = clause_indices if clause_indices is not None else (getattr(_clause, 'indices', None) or [])
        loop_var_ranges = loop_var_ranges if loop_var_ranges is not None else {}
        if _clause and hasattr(_clause, 'value') and isinstance(_clause.value, ReductionExpressionIR):
            loop_var_ranges = getattr(_clause.value, 'loop_var_ranges', {}) or loop_var_ranges

        reduction_var_max: Dict[DefId, int] = {}
        if _clause and hasattr(_clause, 'value') and isinstance(_clause.value, ReductionExpressionIR):
            red_expr = _clause.value
            for loop_var_defid, range_ir in loop_var_ranges.items():
                if not range_ir or not hasattr(range_ir, 'start') or not hasattr(range_ir, 'end'):
                    continue
                try:
                    if isinstance(range_ir.start, LiteralIR):
                        start_val = range_ir.start.value
                    else:
                        start_val = self._evaluate_constant_expression(range_ir.start)
                    if isinstance(range_ir.end, LiteralIR):
                        end_val = range_ir.end.value
                    else:
                        end_val = self._evaluate_constant_expression(range_ir.end)
                    if isinstance(start_val, (int, np.integer)) and isinstance(end_val, (int, np.integer)):
                        reduction_var_max[loop_var_defid] = int(end_val) - 1
                except Exception:
                    pass

        def find_array_access(expr_node: ExpressionIR) -> Optional[range]:
            if isinstance(expr_node, RectangularAccessIR):
                indices_list = expr_node.indices
                if not indices_list:
                    return None

                def expr_uses_defid(e: ExpressionIR) -> bool:
                    if isinstance(e, (IdentifierIR, IndexVarIR)):
                        return getattr(e, 'defid', None) == target_defid
                    if isinstance(e, BinaryOpIR):
                        return expr_uses_defid(e.left) or expr_uses_defid(e.right)
                    if hasattr(e, 'operand'):
                        return expr_uses_defid(getattr(e, 'operand'))
                    return False

                def _is_target(e: ExpressionIR) -> bool:
                    return isinstance(e, (IdentifierIR, IndexVarIR)) and getattr(e, 'defid', None) == target_defid

                def _extract_var_relation(expr: ExpressionIR):
                    """Extract how target variable relates to an index expression.
                    Returns (multiplier_ir, divisor_ir, offset) or None.
                      idx ≈ var * mult / div + offset_terms
                      ⇒ var < dim * div / mult   (offset ignored; conservative)
                    Returns None when var only appears inside % (no upper-bound constraint).
                    """
                    if _is_target(expr):
                        return (None, None, 0)
                    if not isinstance(expr, BinaryOpIR):
                        return None
                    if expr.operator == BinaryOp.DIV and _is_target(expr.left):
                        return (None, expr.right, 0)
                    if expr.operator == BinaryOp.MUL:
                        if _is_target(expr.left):
                            return (expr.right, None, 0)
                        if _is_target(expr.right):
                            return (expr.left, None, 0)
                    if expr.operator == BinaryOp.MOD:
                        return None
                    if expr.operator in (BinaryOp.ADD, BinaryOp.SUB):
                        left_has = expr_uses_defid(expr.left)
                        right_has = expr_uses_defid(expr.right)
                        if left_has and not right_has:
                            return _extract_var_relation(expr.left)
                        if right_has and not left_has and expr.operator == BinaryOp.ADD:
                            return _extract_var_relation(expr.right)
                    return None

                def _find_offset(e: ExpressionIR) -> int:
                    if isinstance(e, LiteralIR):
                        try:
                            val = e.value
                            if isinstance(val, (int, float)):
                                return abs(int(val))
                        except Exception:
                            pass
                        return 0
                    if isinstance(e, (IdentifierIR, IndexVarIR)):
                        ed = getattr(e, 'defid', None)
                        if ed is not None and ed in reduction_var_max:
                            return reduction_var_max[ed]
                        return 0
                    if isinstance(e, BinaryOpIR):
                        if e.operator == BinaryOp.ADD:
                            lo = _find_offset(e.left)
                            ro = _find_offset(e.right)
                            if _is_target(e.left):
                                return ro
                            if _is_target(e.right):
                                return lo
                            return max(lo, ro)
                        if e.operator == BinaryOp.SUB:
                            if isinstance(e.right, (IdentifierIR, IndexVarIR)):
                                rid = getattr(e.right, 'defid', None)
                                if rid is not None and rid in reduction_var_max:
                                    return reduction_var_max[rid]
                            elif isinstance(e.right, LiteralIR):
                                try:
                                    val = e.right.value
                                    if isinstance(val, (int, float)):
                                        return abs(int(val))
                                except Exception:
                                    pass
                        return 0
                    if hasattr(e, 'operand'):
                        return _find_offset(getattr(e, 'operand'))
                    return 0

                # Collect constraints from ALL matching index positions
                constraints = []  # (idx_pos, multiplier_ir, divisor_ir, offset)
                for idx_pos, idx in enumerate(indices_list):
                    if not (_is_target(idx) or (isinstance(idx, BinaryOpIR) and expr_uses_defid(idx))):
                        continue
                    rel = _extract_var_relation(idx)
                    if rel is not None:
                        mult_ir, div_ir, _ = rel
                        offset = _find_offset(idx) if (mult_ir is None and div_ir is None) else 0
                        constraints.append((idx_pos, mult_ir, div_ir, offset))

                if not constraints:
                    pass  # fall through to sub-expression traversal below
                else:
                    if isinstance(expr_node.array, IdentifierIR):
                        # Concrete shape_info path
                        if hasattr(expr_node.array, 'shape_info') and expr_node.array.shape_info:
                            shape = expr_node.array.shape_info
                            if isinstance(shape, (list, tuple)):
                                min_end = None
                                for (pos, mult_ir, div_ir, offset) in constraints:
                                    if pos >= len(shape) or not isinstance(shape[pos], int):
                                        continue
                                    end = shape[pos]
                                    if offset > 0:
                                        end = max(0, end - offset)
                                    if div_ir is not None and isinstance(div_ir, LiteralIR) and isinstance(div_ir.value, (int, float)):
                                        end = end * int(div_ir.value)
                                    if mult_ir is not None and isinstance(mult_ir, LiteralIR) and isinstance(mult_ir.value, (int, float)):
                                        mval = int(mult_ir.value)
                                        if mval > 0:
                                            end = (end + mval - 1) // mval
                                    if min_end is None or end < min_end:
                                        min_end = end
                                if min_end is not None:
                                    return range(0, min_end)

                        # Symbolic path: build end_ir for each constraint, intersect with min
                        loc = location or (getattr(einstein_node, 'location', None))
                        if loc is None:
                            loc = SourceLocation("", 0, 0)

                        end_irs = []
                        for (pos, mult_ir, div_ir, offset) in constraints:
                            shape_access = MemberAccessIR(
                                object=expr_node.array,
                                member='shape',
                                location=loc,
                            )
                            dim_lit = LiteralIR(value=pos, location=loc, type_info=infer_literal_type(pos))
                            shape_dim = RectangularAccessIR(
                                array=shape_access,
                                indices=[dim_lit],
                                location=loc,
                            )
                            end: ExpressionIR = shape_dim
                            if div_ir is not None:
                                end = BinaryOpIR(operator=BinaryOp.MUL, left=end, right=div_ir, location=loc)
                            if mult_ir is not None:
                                end = BinaryOpIR(operator=BinaryOp.DIV, left=end, right=mult_ir, location=loc)
                            end_irs.append(end)

                        # Intersect: min of all constraint ends
                        end_ir: ExpressionIR = end_irs[0]
                        for extra in end_irs[1:]:
                            end_ir = IfExpressionIR(
                                condition=BinaryOpIR(operator=BinaryOp.LT, left=end_ir, right=extra, location=loc),
                                then_expr=end_ir,
                                else_expr=extra,
                                location=loc,
                            )

                        if einstein_node and hasattr(einstein_node, "value"):
                            red_expr = self.find_reduction_in_value(_clause.value) if _clause else None
                            if red_expr is not None:
                                first_pos = constraints[0][0]
                                first_shape_access = MemberAccessIR(object=expr_node.array, member='shape', location=loc)
                                first_dim_lit = LiteralIR(value=first_pos, location=loc, type_info=infer_literal_type(first_pos))
                                first_shape_dim = RectangularAccessIR(array=first_shape_access, indices=[first_dim_lit], location=loc)
                                first_idx = indices_list[first_pos]
                                target_name = next(
                                    (getattr(i, "name", None) for i in _clause_indices if getattr(i, "defid", None) == target_defid),
                                    None,
                                )
                                constraint_end = self._build_output_range_end_from_constraint(
                                    first_shape_dim, red_expr, first_idx, target_defid, loc,
                                    loop_var_ranges=loop_var_ranges,
                                    target_name=target_name,
                                )
                                if constraint_end is not None:
                                    end_ir = constraint_end
                        if einstein_clause is not None:
                            if not hasattr(einstein_clause, 'variable_ranges'):
                                object.__setattr__(einstein_clause, 'variable_ranges', {})
                            start_lit = LiteralIR(value=0, location=loc, type_info=infer_literal_type(0))
                            range_ir = RangeIR(start=start_lit, end=end_ir, location=loc, type_info=UNKNOWN)
                            idx_with_var = next((i for i in _clause_indices if getattr(i, 'defid', None) == target_defid), None)
                            defid_to_set = idx_with_var.defid if (idx_with_var and getattr(idx_with_var, "defid", None)) else target_defid
                            if defid_to_set is not None:
                                einstein_clause.variable_ranges[defid_to_set] = range_ir
                            return range(0, 1000000)
                    if isinstance(expr_node.array, (MemberAccessIR, TupleAccessIR)):
                        base = getattr(expr_node.array, "object", None) or getattr(expr_node.array, "tuple_expr", None)
                        if base is not None and einstein_clause is not None:
                            loc = location or getattr(einstein_node, 'location', None)
                            if loc is None:
                                loc = SourceLocation("", 0, 0)
                            ma_end_irs = []
                            for (pos, mult_ir, div_ir, offset) in constraints:
                                shape_access = MemberAccessIR(object=base, member='shape', location=loc)
                                dim_lit = LiteralIR(value=pos, location=loc, type_info=infer_literal_type(pos))
                                dim_end: ExpressionIR = RectangularAccessIR(array=shape_access, indices=[dim_lit], location=loc)
                                if div_ir is not None:
                                    dim_end = BinaryOpIR(operator=BinaryOp.MUL, left=dim_end, right=div_ir, location=loc)
                                if mult_ir is not None:
                                    dim_end = BinaryOpIR(operator=BinaryOp.DIV, left=dim_end, right=mult_ir, location=loc)
                                ma_end_irs.append(dim_end)
                            end_ir = ma_end_irs[0]
                            for extra in ma_end_irs[1:]:
                                end_ir = IfExpressionIR(
                                    condition=BinaryOpIR(operator=BinaryOp.LT, left=end_ir, right=extra, location=loc),
                                    then_expr=end_ir, else_expr=extra, location=loc,
                                )
                            if not hasattr(einstein_clause, 'variable_ranges'):
                                object.__setattr__(einstein_clause, 'variable_ranges', {})
                            start_lit = LiteralIR(value=0, location=loc, type_info=infer_literal_type(0))
                            range_ir = RangeIR(start=start_lit, end=end_ir, location=loc, type_info=UNKNOWN)
                            idx_with_var = next((i for i in _clause_indices if getattr(i, 'defid', None) == target_defid), None)
                            defid_to_set = idx_with_var.defid if (idx_with_var and getattr(idx_with_var, "defid", None)) else target_defid
                            if defid_to_set is not None:
                                einstein_clause.variable_ranges[defid_to_set] = range_ir
                            return range(0, 1000000)
                    if isinstance(expr_node.array, ArrayLiteralIR):
                        first_pos = constraints[0][0]
                        size = self._compute_array_literal_dimension(expr_node.array, first_pos)
                        if size is not None:
                            return range(0, size)
                    try:
                        first_pos = constraints[0][0]
                        base_array = self._evaluate_constant_expression(expr_node.array)
                        if base_array is not None:
                            if isinstance(base_array, np.ndarray) and first_pos < len(base_array.shape):
                                return range(0, base_array.shape[first_pos])
                            if isinstance(base_array, list):
                                if first_pos == 0:
                                    return range(0, len(base_array))
                                if first_pos == 1 and len(base_array) > 0:
                                    first = base_array[0]
                                    if isinstance(first, list):
                                        return range(0, len(first))
                                    if isinstance(first, np.ndarray) and len(first.shape) > 0:
                                        return range(0, first.shape[0])
                    except Exception:
                        pass
            if isinstance(expr_node, ReductionExpressionIR) and expr_node.body:
                return find_array_access(expr_node.body)
            if isinstance(expr_node, IfExpressionIR):
                r = find_array_access(expr_node.condition)
                if r:
                    return r
                r = find_array_access(expr_node.then_expr)
                if r:
                    return r
                if expr_node.else_expr:
                    return find_array_access(expr_node.else_expr)
                return None
            if isinstance(expr_node, BinaryOpIR):
                r = find_array_access(expr_node.left)
                if r:
                    return r
                return find_array_access(expr_node.right)
            if hasattr(expr_node, 'operand'):
                return find_array_access(getattr(expr_node, 'operand'))
            if hasattr(expr_node, 'body'):
                return find_array_access(getattr(expr_node, 'body'))
            if hasattr(expr_node, 'value'):
                return find_array_access(getattr(expr_node, 'value'))
            if hasattr(expr_node, 'arguments'):
                for arg in getattr(expr_node, 'arguments', []):
                    r = find_array_access(arg)
                    if r:
                        return r
            if hasattr(expr_node, 'elements'):
                for el in getattr(expr_node, 'elements', []) or []:
                    if el is not None:
                        r = find_array_access(el)
                        if r:
                            return r
            if isinstance(expr_node, ArrayLiteralIR):
                try:
                    base_array = self._evaluate_constant_expression(expr_node)
                    if base_array is not None:
                        if isinstance(base_array, np.ndarray) and len(base_array.shape) > 0:
                            return range(0, base_array.shape[0])
                        if isinstance(base_array, list) and len(base_array) > 0:
                            return range(0, len(base_array))
                except Exception:
                    pass
            return None

        return find_array_access(expr)

class _ComplexityCounter(IRVisitor[int]):
    """Visitor to count IR nodes for complexity estimation"""
    def __init__(self):
        self.count = 0
    
    # Default visitor methods (no-op for other nodes)
    def visit_array_comprehension(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_array_pattern(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_arrow_expression(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_block_expression(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_builtin_call(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_binding(self, node) -> int:
        self.count += 1
        if not is_function_binding(node) and not is_einstein_binding(node):
            if hasattr(node, 'value') and node.value:
                node.value.accept(self)
        return self.count
    
    def visit_function_call(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_guard_pattern(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_identifier_pattern(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_if_expression(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_interpolated_string(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_jagged_access(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_lambda(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_literal_pattern(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_match_expression(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_module(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_pipeline_expression(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_program(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_range(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_rest_pattern(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_try_expression(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_tuple_access(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_tuple_expression(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_tuple_pattern(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_wildcard_pattern(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_array_literal(self, node) -> int:
        self.count += 1
        return self.count
    
    def visit_identifier(self, node: IdentifierIR) -> int:
        self.count += 1
        return self.count
    
    def visit_literal(self, node: LiteralIR) -> int:
        self.count += 1
        return self.count
    
    def visit_binary_op(self, node: BinaryOpIR) -> int:
        self.count += 1
        if node.left:
            node.left.accept(self)
        if node.right:
            node.right.accept(self)
        return self.count
    
    def visit_unary_op(self, node: UnaryOpIR) -> int:
        self.count += 1
        if node.operand:
            node.operand.accept(self)
        return self.count
    
    def visit_rectangular_access(self, node: RectangularAccessIR) -> int:
        self.count += 1
        if hasattr(node, 'array'):
            node.array.accept(self)
        for idx in (getattr(node, 'indices', None) or []):
            if idx is not None and hasattr(idx, 'accept'):
                idx.accept(self)
        return self.count
    
    def visit_member_access(self, node: MemberAccessIR) -> int:
        self.count += 1
        if hasattr(node, 'object'):
            node.object.accept(self)
        return self.count
    
    def visit_cast_expression(self, node: CastExpressionIR) -> int:
        self.count += 1
        if hasattr(node, 'expr'):
            node.expr.accept(self)
        return self.count
    
    def visit_reduction_expression(self, node: ReductionExpressionIR) -> int:
        self.count += 1
        if node.body:
            node.body.accept(self)
        return self.count
    
    def visit_where_expression(self, node: WhereExpressionIR) -> int:
        self.count += 1
        if node.expr:
            node.expr.accept(self)
        return self.count

