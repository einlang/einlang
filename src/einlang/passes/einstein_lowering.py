"""
Einstein Lowering Pass

Converts EinsteinDeclarationIR nodes to LoweredIteration structures.
This aligns with architecture where all ranges are computed at compile time
and execution uses pure loop structures.

Rust Pattern: rustc_mir::transform::MirPass

"""

from typing import Dict, List, Optional, Any, Tuple
from ..passes.base import BasePass, TyCtxt
from ..shared.defid import DefId
from ..passes.range_analysis import RangeAnalysisPass, Range
from ..shared.source_location import SourceLocation
from ..shared.types import BinaryOp, infer_literal_type, UNKNOWN, PrimitiveType
from ..ir.nodes import (
    ProgramIR, ExpressionIR, IdentifierIR, IndexVarIR, IndexRestIR, ReductionExpressionIR,
    WhereClauseIR, RangeIR, LiteralIR, EinsteinIR, EinsteinDeclarationIR,
    LoopStructure, BindingIR, GuardCondition,
    LoweredEinsteinClauseIR, LoweredEinsteinIR, LoweredReductionIR, LoweredComprehensionIR,
    IRVisitor, RectangularAccessIR, MemberAccessIR,
    ArrayComprehensionIR, VariableDeclarationIR, BinaryOpIR,
)


def _defid_of_var_in_expr(expr: Optional[ExpressionIR], name: str) -> Optional[DefId]:
    """Return defid of first IdentifierIR or IndexVarIR with given name in expr. Reuse body defid for loop vars."""
    if expr is None:
        return None
    if isinstance(expr, IdentifierIR) and expr.name == name:
        return getattr(expr, "defid", None)
    if isinstance(expr, IndexVarIR) and expr.name == name:
        return getattr(expr, "defid", None)
    children: List[Optional[ExpressionIR]] = []
    if hasattr(expr, "left") and hasattr(expr, "right"):
        children = [getattr(expr, "left"), getattr(expr, "right")]
    elif hasattr(expr, "operand"):
        children = [getattr(expr, "operand")]
    elif hasattr(expr, "array") and hasattr(expr, "indices"):
        children = [getattr(expr, "array")] + list(getattr(expr, "indices") or [])
    elif hasattr(expr, "body"):
        children = [getattr(expr, "body")]
    elif hasattr(expr, "object"):
        children = [getattr(expr, "object")]
    elif hasattr(expr, "arguments"):
        children = [getattr(expr, "callee_expr", None)] + list(getattr(expr, "arguments") or [])
    elif hasattr(expr, "expr"):
        children = [getattr(expr, "expr")]
    elif hasattr(expr, "statements") and hasattr(expr, "final_expr"):
        children = list(getattr(expr, "statements") or [])
        fe = getattr(expr, "final_expr", None)
        if fe is not None:
            children.append(fe)
    for c in children:
        if isinstance(c, ExpressionIR):
            out = _defid_of_var_in_expr(c, name)
            if out is not None:
                return out
    return None


class EinsteinLoweringPass(BasePass):
    """
    Einstein lowering pass.
    
    Converts EinsteinDeclarationIR nodes to LoweredIteration structures.
    All ranges are pre-computed by RangeAnalysisPass.
    """
    requires = [RangeAnalysisPass]  # Depends on range analysis
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """Lower Einstein declarations to loop structures"""
        # DefId from name resolution / body; einstein lowering copies from body
        # Get range analysis results (required)
        range_analyzer = tcx.get_analysis(RangeAnalysisPass)

        # Visit all statements and functions to lower Einstein declarations and reductions
        visitor = EinsteinLoweringVisitor(range_analyzer, tcx)

        new_statements = []
        for stmt in ir.statements:
            if stmt is None:
                raise ValueError("IR statement is None")
            result = stmt.accept(visitor)
            if result is None:
                raise ValueError("Einstein lowering returned None for statement")
            new_statements.append(result)
        ir.statements = new_statements

        from ..analysis.analysis_guard import should_analyze_function, is_generic_function
        function_ir_map = getattr(tcx, 'function_ir_map', None) or {}
        specialized_list = getattr(tcx, 'specialized_functions', []) or []
        def body_contains_lowerable(node, seen=None):
            seen = seen or set()
            if node is None or id(node) in seen:
                return False
            seen.add(id(node))
            if type(node).__name__ in ('ArrayComprehensionIR', 'EinsteinDeclarationIR', 'ReductionExpressionIR'):
                return True
            for attr in ('body', 'final_expr', 'then_expr', 'else_expr', 'condition', 'value', 'expr', 'array', 'left', 'right', 'object', 'scrutinee'):
                if hasattr(node, attr) and body_contains_lowerable(getattr(node, attr), seen):
                    return True
            binding = getattr(node, '_binding', None)
            if binding is not None and getattr(binding, 'expr', None) is not None and body_contains_lowerable(binding.expr, seen):
                return True
            for attr in ('statements', 'arguments', 'elements', 'arms', 'items', 'indices', 'parts'):
                if hasattr(node, attr):
                    for c in getattr(node, attr) or []:
                        if body_contains_lowerable(c, seen):
                            return True
            return False

        def lower_function_body(func, bucket: str):
            if not getattr(func, 'body', None):
                return
            visitor._current_function = func
            result = func.body.accept(visitor)
            if result is None:
                raise ValueError("Einstein lowering returned None for function body")
            func.body = result

        from ..ir.nodes import FunctionDefIR
        lowered_ids = set()
        if specialized_list:
            for func in specialized_list:
                if not isinstance(func, FunctionDefIR) or is_generic_function(func):
                    continue
                if id(func) in lowered_ids:
                    continue
                lowered_ids.add(id(func))
                lower_function_body(func, "specialized_list")


        if function_ir_map:
            for func in function_ir_map.values():
                if not isinstance(func, FunctionDefIR) or not getattr(func, 'body', None):
                    continue
                if id(func) in lowered_ids:
                    continue
                if is_generic_function(func):
                    continue
                lowered_ids.add(id(func))
                lower_function_body(func, "function_ir_map")

        for func in ir.functions:
            _body = getattr(func, 'body', None)
            if _body is None:
                continue
            if is_generic_function(func):
                continue
            if id(func) in lowered_ids:
                continue
            lowered_ids.add(id(func))
            lower_function_body(func, "ir.functions")

        return ir

class RestPatternReplacer(IRVisitor[ExpressionIR]):
    """
    Visitor to replace rest pattern identifiers with expanded variables.
    
    Transforms expressions like x[..batch, j] to x[batch.0, j]
    """
    
    def __init__(self, rest_var_mapping: Dict[str, str], tcx: TyCtxt):
        self.rest_var_mapping = rest_var_mapping  # Maps "..batch" -> "batch.0"
        self.tcx = tcx
    
    def visit_identifier(self, node: IdentifierIR) -> ExpressionIR:
        """Replace rest pattern identifiers with expanded variables"""
        if node.name in self.rest_var_mapping:
            expanded_name = self.rest_var_mapping[node.name]
            expanded_defid = self.tcx.resolver._rest_pattern_vars.get(expanded_name) if hasattr(self.tcx.resolver, '_rest_pattern_vars') else None
            new_node = IdentifierIR(
                name=expanded_name,
                location=node.location,
                defid=expanded_defid
            )
            # Copy type_info and shape_info from original node
            if hasattr(node, 'type_info'):
                new_node.type_info = node.type_info
            if hasattr(node, 'shape_info'):
                new_node.shape_info = node.shape_info
            return new_node
        return node
    
    def visit_rectangular_access(self, node: RectangularAccessIR) -> ExpressionIR:
        """Transform array access - recurse on array and indices. Indices are always flat (no nested groups)."""
        node.array = node.array.accept(self)
        raw = node.indices or []
        # Flatten: do not form nested; treat as single flat list
        indices_in = [x for g in raw for x in (g if isinstance(g, list) else [g])]
        indices = []
        for idx in indices_in:
            if idx is None:
                raise ValueError(
                    "Array access indices must not contain None; "
                    "invalid index slot (likely a bug in an earlier pass)"
                )
            if isinstance(idx, IndexRestIR):
                raise ValueError(
                    f"IndexRestIR (..{getattr(idx, 'name', '?')}) must not reach Einstein lowering; "
                    "rest patterns must be expanded in rest_pattern_preprocessing."
                )
            res = idx.accept(self) if hasattr(idx, "accept") else idx
            indices.append(res if res is not None else idx)
        node.indices = indices
        return node
    
    def visit_reduction_expression(self, node: ReductionExpressionIR) -> ExpressionIR:
        """Transform reduction body"""
        node.body = node.body.accept(self)
        return node
    
    def visit_binary_op(self, node) -> ExpressionIR:
        """Transform binary operation operands"""
        node.left = node.left.accept(self)
        node.right = node.right.accept(self)
        return node
    
    def visit_unary_op(self, node) -> ExpressionIR:
        """Transform unary operation operand"""
        node.operand = node.operand.accept(self)
        return node
    
    def visit_function_call(self, node) -> ExpressionIR:
        """Transform function call arguments"""
        node.arguments = [arg.accept(self) for arg in node.arguments]
        return node
    
    def visit_array_literal(self, node) -> ExpressionIR:
        """Transform array literal elements"""
        node.elements = [elem.accept(self) for elem in node.elements]
        return node
    
    def visit_tuple_literal(self, node) -> ExpressionIR:
        """Transform tuple literal elements"""
        node.elements = [elem.accept(self) for elem in node.elements]
        return node
    
    def visit_tuple_access(self, node) -> ExpressionIR:
        """Transform tuple access"""
        node.tuple_expr = node.tuple_expr.accept(self)
        return node
    
    def visit_block_expression(self, node) -> ExpressionIR:
        """Transform block expression"""
        if node.final_expr:
            node.final_expr = node.final_expr.accept(self)
        return node
    
    def visit_if_expression(self, node) -> ExpressionIR:
        """Transform if expression"""
        node.condition = node.condition.accept(self)
        node.then_expr = node.then_expr.accept(self)
        if node.else_expr:
            node.else_expr = node.else_expr.accept(self)
        return node
    
    def visit_lambda(self, node) -> ExpressionIR:
        """Transform lambda body"""
        node.body = node.body.accept(self)
        return node
    
    # For leaf nodes that don't contain expressions, return as-is
    def visit_literal(self, node) -> ExpressionIR:
        return node
    
    def visit_range(self, node) -> ExpressionIR:
        node.start = node.start.accept(self)
        node.end = node.end.accept(self)
        return node
    
    def visit_program(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit ProgramIR")
    
    def visit_function_def(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit FunctionDefIR")
    
    def visit_variable_declaration(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit VariableDeclarationIR")
    
    def visit_einstein_declaration(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit EinsteinDeclarationIR")

    def visit_einstein(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit EinsteinIR")
    
    def visit_constant_def(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit ConstantDefIR")
    
    def visit_module(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit Module")
    
    # Pattern visitors - not applicable for expressions
    def visit_literal_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit patterns")
    
    def visit_identifier_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit patterns")
    
    def visit_wildcard_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit patterns")
    
    def visit_tuple_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit patterns")
    
    def visit_array_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit patterns")
    
    def visit_rest_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit patterns")
    
    def visit_guard_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("RestPatternReplacer should not visit patterns")
    
    # Additional expression visitors - pass through
    def visit_array_comprehension(self, node) -> ExpressionIR:
        return node
    
    def visit_arrow_expression(self, node) -> ExpressionIR:
        node.body = node.body.accept(self)
        return node
    
    def visit_builtin_call(self, node) -> ExpressionIR:
        node.args = [arg.accept(self) for arg in node.args]
        return node
    
    def visit_cast_expression(self, node) -> ExpressionIR:
        node.expr = node.expr.accept(self)
        return node
    
    def visit_function_ref(self, node) -> ExpressionIR:
        return node
    
    def visit_interpolated_string(self, node) -> ExpressionIR:
        return node
    
    def visit_jagged_access(self, node) -> ExpressionIR:
        node.base = node.base.accept(self)
        chain = getattr(node, 'index_chain', None) or []
        new_chain = []
        for idx in chain:
            if idx is None:
                raise ValueError("IR index_chain slot is None")
            new_chain.append(idx.accept(self))
        node.index_chain = new_chain
        return node
    
    def visit_match_expression(self, node) -> ExpressionIR:
        node.scrutinee = node.scrutinee.accept(self)
        return node
    
    def visit_member_access(self, node) -> ExpressionIR:
        node.object = node.object.accept(self)
        return node
    
    def visit_pipeline_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_try_expression(self, node) -> ExpressionIR:
        node.expr = node.expr.accept(self)
        return node
    
    def visit_tuple_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_where_expression(self, node) -> ExpressionIR:
        return node

def has_unexpanded_rest_patterns(expr: ExpressionIR) -> bool:
    """Check if an expression contains any unexpanded rest patterns (IndexRestIR or .. identifiers)."""
    from ..ir.nodes import IdentifierIR, IndexRestIR, RectangularAccessIR, ReductionExpressionIR, BinaryOpIR, UnaryOpIR, FunctionCallIR

    def check_node(node) -> bool:
        if isinstance(node, IndexRestIR):
            return True
        if isinstance(node, IdentifierIR) and node.name.startswith(".."):
            return True
        elif isinstance(node, RectangularAccessIR):
            if node.array and check_node(node.array):
                return True
            if node.indices:
                for idx_group in (node.indices if node.indices and isinstance(node.indices[0], list) else [node.indices] if node.indices else []):
                    for idx in idx_group:
                        if check_node(idx):
                            return True
        elif isinstance(node, ReductionExpressionIR):
            if node.body and check_node(node.body):
                return True
        elif isinstance(node, BinaryOpIR):
            if node.left and check_node(node.left):
                return True
            if node.right and check_node(node.right):
                return True
        elif isinstance(node, UnaryOpIR):
            if node.operand and check_node(node.operand):
                return True
        elif isinstance(node, FunctionCallIR):
            if node.arguments:
                for arg in node.arguments:
                    if check_node(arg):
                        return True
        return False
    
    return check_node(expr)

class EinsteinLoweringVisitor(IRVisitor[None]):
    """Visitor to lower Einstein declarations to loop structures"""
    
    def __init__(self, range_accessor: Any, tcx: TyCtxt):
        self.range_analyzer = range_accessor  # RangeAccessor with get_range method
        self.tcx = tcx
        self._current_function = None  # Set during function traversal for param-based range fallback
        self._current_einstein_clause = None  # Enclosing clause when lowering; used for variable_ranges fallback

    def visit_einstein_declaration(self, node: EinsteinDeclarationIR) -> Optional[Any]:
        """Lower Einstein declaration (clauses list); returns LoweredEinsteinIR."""
        clauses = node.clauses or []
        if not clauses:
            return None
        if len(clauses) > 1:
            items = []
            all_shapes = []
            element_type = None
            loc = getattr(node, "location", None) or SourceLocation("", 0, 0)
            for clause in clauses:
                lowered = self._lower_einstein_clause_to_lowered(node, clause)
                if isinstance(lowered, LoweredEinsteinIR) and lowered.items:
                    items.extend(lowered.items)
                    if lowered.element_type and element_type is None:
                        element_type = lowered.element_type
                    if lowered.shape:
                        all_shapes.append(lowered.shape)
                    # Include LHS-declared extent so union is at least each clause's declared range (e.g. 0..10)
                    lhs_shape = self._shape_from_clause_indices(clause, loc)
                    if lhs_shape is not None:
                        all_shapes.append(lhs_shape)
            if items:
                # Prefer declaration's promoted element_type (from type inference) over first clause's type
                decl_element_type = getattr(node, 'element_type', None)
                if decl_element_type is not None and decl_element_type is not UNKNOWN:
                    element_type = decl_element_type
                # Prefer shape stored on IR by shape pass; else union of clause ranges
                tensor_shape = getattr(node, 'shape', None) if isinstance(getattr(node, 'shape', None), list) else None
                if tensor_shape is None and all_shapes:
                    tensor_shape = self._compute_shape_union(all_shapes, loc)
                if tensor_shape is None and items[0].loops:
                    tensor_shape = self._compute_shape(items[0].loops)
                return LoweredEinsteinIR(items=items, shape=tensor_shape, element_type=element_type)
            return None
        return self._lower_einstein_clause_to_lowered(node, clauses[0])

    def _lower_einstein_clause_to_lowered(self, decl: EinsteinDeclarationIR, clause: EinsteinIR) -> Optional[LoweredEinsteinIR]:
        """Lower one Einstein clause (EinsteinIR) to LoweredEinsteinIR. Uses decl for rest expansion key and shape."""
        import logging
        logger = logging.getLogger("einlang.passes.einstein_lowering")
        node = clause  # Use clause for indices, value, where_clause, variable_ranges
        prev_clause = self._current_einstein_clause
        self._current_einstein_clause = node
        # Rest patterns must be expanded by rest_pattern_preprocessing; we never see IndexRestIR here.
        for idx in node.indices:
            if isinstance(idx, IndexRestIR):
                raise ValueError(
                    f"IndexRestIR (..{idx.name}) must not reach Einstein lowering; "
                    "rest patterns must be expanded in rest_pattern_preprocessing. "
                    "Either expansion was deferred or a pass is missing."
                )

        if node.value and has_unexpanded_rest_patterns(node.value):
            raise ValueError(
                "Body must not contain unexpanded rest patterns (IndexRestIR or ..identifiers) at Einstein lowering; "
                "rest patterns must be expanded in rest_pattern_preprocessing."
            )
        
        # Extract loop variables from indices (IndexVarIR and LiteralIR only)
        loops = []
        loop_var_names = []
        
        # Rest patterns are expanded immediately in RestPatternPreprocessingPass; indices and body are already expanded.
        # Expand indices in place (one rest slot can become many)
        indices = list(node.indices)
        idx_pos = 0
        i = 0
        while i < len(indices):
            idx = indices[i]
            idx_pos = i
            if isinstance(idx, LiteralIR):
                logger.debug(f"  Processing literal index {idx_pos}: {idx.value}")
                i += 1
            elif isinstance(idx, IndexRestIR):
                raise ValueError(
                    f"IndexRestIR (..{idx.name}) must not reach Einstein lowering; "
                    "rest patterns must be expanded in rest_pattern_preprocessing."
                )
            elif isinstance(idx, IndexVarIR):
                var_name = idx.name
                logger.debug(f"  Processing index var {idx_pos}: {var_name}")
                loop_var_names.append(var_name)
                range_obj = None
                if getattr(idx, "range_ir", None) is not None:
                    range_obj = idx.range_ir
                if range_obj is None and getattr(idx, "defid", None):
                    r = (getattr(node, "variable_ranges", None) or {}).get(idx.defid)
                    if isinstance(r, RangeIR):
                        range_obj = r
                # Range only from clause/declaration variable_ranges or index.range_ir (no global dict).
                if getattr(node, "variable_ranges", None) and getattr(idx, "defid", None):
                    clause_r = node.variable_ranges.get(idx.defid)
                    if isinstance(clause_r, RangeIR):
                        range_obj = clause_r
                if range_obj:
                    iterable = self._range_to_iterable_ir(range_obj, node.location)
                    loc = idx.location if hasattr(idx, "location") and idx.location else node.location
                    ti = getattr(idx, 'type_info', None) or PrimitiveType("i32")
                    if idx.defid is None:
                        raise ValueError(
                            f"Einstein index variable '{var_name}' must have a defid (from name resolution). "
                            "Ensure NameResolutionPass runs before EinsteinLoweringPass."
                        )
                    var_node = IndexVarIR(var_name, loc or SourceLocation("", 0, 0), defid=idx.defid, type_info=ti)
                    loops.append(LoopStructure(variable=var_node, iterable=iterable))
                elif getattr(idx, "defid", None) is not None:
                    raise ValueError(
                        f"Range for index variable '{var_name}' (defid={idx.defid.krate}:{idx.defid.index}) could not be inferred. "
                        "Ensure RangeAnalysisPass runs before EinsteinLoweringPass and variable_ranges or array access provides a range."
                    )
                i += 1
            elif isinstance(idx, IdentifierIR):
                raise ValueError(
                    f"Einstein indices must be IndexVarIR or LiteralIR; got IdentifierIR ('{idx.name}'). "
                    "Convert index variables to IndexVarIR in an earlier pass (e.g. name resolution or ast_to_ir)."
                )
            else:
                i += 1

        node.indices = indices
        
        for loop in loops:
            v = getattr(loop, "variable", None)
            if v is not None and getattr(v, "defid", None) is None:
                raise ValueError(
                    "Loop variable has no defid; runtime cannot bind it. "
                    "Ensure index variables have defids from name resolution."
                )
        
        # Sync clause variable_ranges onto reduction.loop_var_ranges so _extract_reduction_ranges and
        # lower_reduction_expression see them (mono pipeline sets clause.variable_ranges; reduction may not have loop_var_ranges).
        if isinstance(node.value, ReductionExpressionIR) and getattr(node, 'variable_ranges', None):
            vr = node.variable_ranges
            for v in (node.value.loop_vars or []):
                did = getattr(v, 'defid', None)
                if not did:
                    continue
                if getattr(node.value, 'loop_var_ranges', None) and did in node.value.loop_var_ranges:
                    continue
                r = vr.get(did)
                if r is None and hasattr(did, 'krate') and hasattr(did, 'index'):
                    r = vr.get((did.krate, did.index))
                if r is None and hasattr(did, '__getitem__') and len(did) >= 2:
                    r = vr.get((did[0], did[1]))
                if r is not None:
                    if not getattr(node.value, 'loop_var_ranges', None):
                        node.value.loop_var_ranges = {}
                    node.value.loop_var_ranges[did] = r
        # Extract reduction_ranges from value BEFORE lowering: once value is replaced with LoweredReductionIR,
        # _extract_reduction_ranges would find no ReductionExpressionIR and return {}. Loop variables (e.g. j)
        # would then never be bound at runtime.
        reduction_ranges = self._extract_reduction_ranges(node.value, node.location)
        
        # Let value handle itself: recurse so reductions (and nested comprehensions) lower and replace
        if node.value:
            result = node.value.accept(self)
            if result is not None:
                node.value = result
                # When body is a LoweredReductionIR with empty .loops, set .loops from pre-extracted
                # reduction_ranges so runtime has loop vars (e.g. j) with correct DefIds.
                from ..ir.nodes import LoweredReductionIR
                if isinstance(result, LoweredReductionIR) and reduction_ranges and not result.loops:
                    result.loops = list(reduction_ranges.values())
        
        # Extract bindings and guards from where_clause
        bindings, guards = self._extract_bindings_and_guards(node.where_clause)
        
        # Allocation shape: when we have loops, derive from loops unless clause has literal indices.
        # Mixed literal + loop indices (e.g. [0, i]) need full rank from clause indices so backend writes at (0, i).
        if loops:
            shape = None
            if any(isinstance(idx, LiteralIR) for idx in (node.indices or [])):
                shape = self._shape_from_clause_indices(node, node.location)
            if shape is None:
                shape = self._compute_shape(loops)
        else:
            # Prefer shape stored on IR by shape pass (no lookup by identity)
            shape = getattr(decl, 'shape', None) if isinstance(getattr(decl, 'shape', None), list) else None
            if shape is None:
                shape = self._get_shape_from_analysis(decl)
            if shape is None and node.indices:
                shape = self._shape_from_literal_indices(node.indices, node.location)
        
        # Get element type (if available from type inference)
        # For arrays, type_info might contain element type information
        element_type = None
        if hasattr(node.value, 'type_info') and node.value.type_info:
            # Extract element type from type_info if it's an array type
            element_type = node.value.type_info
        
        # Create lowered Einstein clause (range/per-clause only)
        clause = LoweredEinsteinClauseIR(
            body=node.value,
            loops=loops,
            reduction_ranges=reduction_ranges,
            bindings=bindings,
            guards=guards,
            indices=indices,
        )
        # Wrap in group with shared shape, element_type
        self._current_einstein_clause = prev_clause
        return LoweredEinsteinIR(items=[clause], shape=shape, element_type=element_type)

    def lower_reduction_expression(self, node: ReductionExpressionIR) -> None:
        """Lower a reduction expression to LoweredIteration. loop_vars is the single source of truth."""
        loops = []
        for var_ident in node.loop_vars:
            var_name = var_ident.name
            # Get range from loop_var_ranges (explicit ranges in reduction)
            range_obj = None
            iterable = None
            if node.loop_var_ranges and getattr(var_ident, 'defid', None) in node.loop_var_ranges:
                range_ir = node.loop_var_ranges[var_ident.defid]
                # Convert RangeIR to iterable
                if isinstance(range_ir, RangeIR):
                    # Try to evaluate as constant
                    start_val = self._evaluate_constant(range_ir.start)
                    end_val = self._evaluate_constant(range_ir.end)
                    if start_val is not None and end_val is not None:
                        range_obj = range(start_val, end_val)
                        iterable = self._range_to_iterable_ir(range_obj, node.location)
                    else:
                        # Dynamic range - use RangeIR as iterable directly
                        iterable = range_ir
                elif isinstance(range_ir, LiteralIR) and isinstance(range_ir.value, range):
                    range_obj = range_ir.value
                    iterable = self._range_to_iterable_ir(range_obj, node.location)
                else:
                    # Unknown type - try to use as iterable directly
                    iterable = range_ir

            if not iterable and getattr(var_ident, 'defid', None) is not None:
                enc = getattr(self, '_current_einstein_clause', None)
                if enc and getattr(enc, 'variable_ranges', None):
                    vr = enc.variable_ranges
                    did = var_ident.defid
                    range_ir = vr.get(did) or (vr.get((did.krate, did.index)) if hasattr(did, 'krate') else None) or (vr.get((did[0], did[1])) if hasattr(did, '__getitem__') and len(did) >= 2 else None)
                    if isinstance(range_ir, RangeIR):
                        start_val = self._evaluate_constant(range_ir.start)
                        end_val = self._evaluate_constant(range_ir.end)
                        if start_val is not None and end_val is not None:
                            iterable = self._range_to_iterable_ir(range(start_val, end_val), node.location)
                        else:
                            iterable = range_ir
                    elif range_ir is not None:
                        iterable = range_ir
            if iterable:
                body_defid = _defid_of_var_in_expr(node.body, var_name) if node.body else None
                defid = body_defid or getattr(var_ident, "defid", None)
                if defid is None:
                    raise ValueError(
                        f"Reduction loop variable '{var_name}' has no defid; runtime cannot bind it. "
                        "Ensure reduction loop vars have defids from name resolution."
                    )
                loc = getattr(node, 'location', None) or getattr(node.body, 'location', None)
                ti = getattr(var_ident, 'type_info', None) or PrimitiveType("i32")
                var_ident = IdentifierIR(var_name, loc or SourceLocation('', 0, 0), defid=defid, type_info=ti)
                loops.append(LoopStructure(variable=var_ident, iterable=iterable))
            elif getattr(var_ident, 'defid', None) is not None:
                cur_func = getattr(self, '_current_function', None)
                func_name = getattr(cur_func, 'name', None) if cur_func else None
                spec_list = getattr(self.tcx, 'specialized_functions', [])
                is_specialized = cur_func is not None and cur_func in spec_list
                raise ValueError(
                    f"Range for reduction loop variable '{var_name}' (defid={var_ident.defid.krate}:{var_ident.defid.index}) could not be inferred. "
                    "Ensure RangeAnalysisPass runs before EinsteinLoweringPass and sets loop_var_ranges on the reduction."
                )
        
        # Extract bindings and guards from where_clause
        bindings, guards = self._extract_bindings_and_guards(node.where_clause)
        body = node.body
        if body is not None:
            lowered_body = body.accept(self)
            if lowered_body is not None:
                body = lowered_body
        return LoweredReductionIR(
            body=body,
            operation=node.operation,
            loops=loops,
            bindings=bindings,
            guards=guards,
            location=getattr(node, 'location', None),
            type_info=getattr(node, 'type_info', None),
        )
    
    def lower_array_comprehension(self, node: ArrayComprehensionIR) -> Optional[LoweredComprehensionIR]:
        """Lower array comprehension; returns new node (LoweredComprehensionIR) to replace it."""
        variables = getattr(node, 'variables', None) or []
        ranges_list = getattr(node, 'ranges', None) or []
        constraints = getattr(node, 'constraints', None) or []
        if len(variables) != len(ranges_list):
            return None
        # Lower body first (e.g. reductions or nested comprehensions inside body)
        body = node.body
        if body is not None:
            lowered_body = body.accept(self)
            if lowered_body is not None:
                body = lowered_body
        loops = []
        for i, (var_name, range_ir) in enumerate(zip(variables, ranges_list)):
            iterable = self._range_to_iterable_ir(range_ir, getattr(node, 'location', None))
            defid = None
            if getattr(node, 'variable_defids', None) and i < len(node.variable_defids):
                d = node.variable_defids[i]
                if d is not None:
                    defid = d if hasattr(d, 'krate') else getattr(d, 'defid', None)
            if defid is None and i < len(constraints):
                defid = self._defid_from_in_constraint(constraints[i], var_name)
            if defid is None:
                raise ValueError(
                    f"Array comprehension variable '{var_name}' has no defid; runtime cannot bind it. "
                    "Ensure variable_defids or constraint provides defid."
                )
            loc = getattr(node, 'location', None)
            var_ident = IdentifierIR(var_name, loc or SourceLocation('', 0, 0), defid=defid, type_info=PrimitiveType("i32"))
            loops.append(LoopStructure(variable=var_ident, iterable=iterable))
        # Split constraints: assignments (y = expr) -> bindings; rest -> guards.
        bindings = []
        guards = []
        for c in constraints or []:
            if isinstance(c, BinaryOpIR) and getattr(c, 'operator', None) == BinaryOp.ASSIGN:
                left, right = getattr(c, 'left', None), getattr(c, 'right', None)
                if isinstance(left, IdentifierIR):
                    defid = getattr(left, 'defid', None)
                    if defid is None:
                        raise ValueError(
                            f"Where-clause binding '{left.name}' must have defid. "
                            "Ensure name resolution sets defid on constraint left-hand side."
                        )
                    right_lowered = right.accept(self) if right is not None else right
                    bindings.append(BindingIR(name=left.name, expr=right_lowered if right_lowered is not None else right, defid=defid, location=getattr(c, 'location', None)))
                    continue
            cond_lowered = c.accept(self) if c is not None else c
            guards.append(GuardCondition(cond_lowered if cond_lowered is not None else c))
        return LoweredComprehensionIR(
            body=body,
            loops=loops,
            bindings=bindings,
            guards=guards,
            location=getattr(node, 'location', None),
        )
    
    def _defid_from_in_constraint(self, constraint, variable_name: str) -> Optional[Any]:
        """Get defid from constraint that binds var (e.g. 'var in range')."""
        if not hasattr(constraint, 'left') or not hasattr(constraint, 'operator'):
            return None
        from ..ir.nodes import IdentifierIR
        op = getattr(constraint, 'operator', None)
        is_in = op == 'in' or (getattr(op, 'value', None) == 'in')
        if is_in and isinstance(constraint.left, IdentifierIR) and constraint.left.name == variable_name:
            return getattr(constraint.left, 'defid', None)
        return None
    
    def _replace_rest_patterns_in_expr(self, expr: ExpressionIR, rest_var_mapping: Dict[str, str]) -> ExpressionIR:
        """
        Replace rest pattern identifiers in expression tree using visitor pattern.
        """
        replacer = RestPatternReplacer(rest_var_mapping, self.tcx)
        return expr.accept(replacer)
    
    def _range_to_iterable_ir(self, range_obj: Any, location) -> ExpressionIR:
        """Convert a range object to iterable IR (always RangeIR for ranges).
        
        Handles:
        - ExpressionIR (e.g. IdentifierIR for "row in nested") - return as-is for collection iteration
        - RangeInfo objects (StaticRange, DynamicRange, DependentRange)
        - Python range objects
        - RangeIR - already in IR form
        """
        from ..passes.range_info import StaticRange, DynamicRange, DependentRange
        
        # Comprehension "var in collection": range_obj is the collection expression (e.g. IdentifierIR('nested'))
        if isinstance(range_obj, ExpressionIR):
            return range_obj
        # If range_obj is already a RangeIR, return it
        if isinstance(range_obj, RangeIR):
            return range_obj
        
        # Handle RangeInfo objects (StaticRange/DynamicRange/DependentRange)
        if isinstance(range_obj, StaticRange):
            val = range_obj.to_python_range()
            return RangeIR(
                start=LiteralIR(val.start, location, type_info=infer_literal_type(val.start)),
                end=LiteralIR(val.stop, location, type_info=infer_literal_type(val.stop)),
                location=location,
            )
        
        if isinstance(range_obj, (DynamicRange, DependentRange)):
            # Dynamic/Dependent range - use to_range_ir() to get RangeIR with expressions
            return range_obj.to_range_ir(location)
        
        if isinstance(range_obj, range):
            return RangeIR(
                start=LiteralIR(range_obj.start, location, type_info=infer_literal_type(range_obj.start)),
                end=LiteralIR(range_obj.stop, location, type_info=infer_literal_type(range_obj.stop)),
                location=location,
            )
        
        # If range_obj is a Range (from RangeAnalyzer), convert to RangeIR
        if hasattr(range_obj, 'start') and hasattr(range_obj, 'end'):
            # Try to evaluate start and end as constants
            start_val = self._evaluate_constant(range_obj.start) if hasattr(range_obj, 'start') else None
            end_val = self._evaluate_constant(range_obj.end) if hasattr(range_obj, 'end') else None
            
            if start_val is not None and end_val is not None:
                val = range(start_val, end_val)
                return LiteralIR(
                    value=val, location=location, shape_info=None,
                    type_info=infer_literal_type(val)
                )
            else:
                start_ir = self._value_to_ir(range_obj.start, location)
                end_ir = self._value_to_ir(range_obj.end, location)
                return RangeIR(
                    start=start_ir, end=end_ir, location=location,
                    type_info=UNKNOWN
                )
        
        # Fallback: create a RangeIR with unknown bounds
        # This shouldn't happen if range analysis worked correctly
        return LiteralIR(
            value=range(0, 1),  # Default fallback
            location=location,
            shape_info=None,
            type_info=infer_literal_type(range(0, 1))
        )
    
    def _value_to_ir(self, value: Any, location) -> ExpressionIR:
        """Convert a Python value to IR expression. Set type_info for no info loss."""
        if isinstance(value, ExpressionIR):
            return value
        return LiteralIR(
            value=value, location=location, shape_info=None,
            type_info=infer_literal_type(value)
        )
    
    def _evaluate_constant(self, expr: Any) -> Optional[int]:
        """Try to evaluate an expression as a constant integer"""
        if isinstance(expr, int):
            return expr
        elif isinstance(expr, LiteralIR):
            if isinstance(expr.value, int):
                return expr.value
        # Could add more evaluation logic here
        return None
    
    def _infer_range_for_rest_expanded_var(
        self, rest_var_name: str, dim_idx: int, expr: ExpressionIR, location
    ) -> Optional[RangeIR]:
        """
        Infer range for rest-pattern expanded variable (e.g. batch.0, batch.1) from array accesses.
        After expansion the body has batch.0, batch.1 in indices; look for expanded name at position dim_idx.
        """
        from ..ir.nodes import RectangularAccessIR, IdentifierIR, LiteralIR, MemberAccessIR

        expanded_name = f"{rest_var_name[2:]}.{dim_idx}" if rest_var_name.startswith("..") else rest_var_name
        accesses = self._find_array_accesses(expr)
        for access in accesses:
            if not isinstance(access, RectangularAccessIR) or not access.indices:
                continue
            indices_flat = []
            for idx in (access.indices[0] if isinstance(access.indices[0], list) else access.indices):
                if isinstance(idx, list):
                    indices_flat.extend(idx)
                else:
                    indices_flat.append(idx)
            for pos, idx_expr in enumerate(indices_flat):
                if pos != dim_idx:
                    continue
                if isinstance(idx_expr, IdentifierIR) and (idx_expr.name == expanded_name or idx_expr.name == rest_var_name):
                    # Found expanded var or unexpanded rest pattern at this dimension - infer from array shape
                    shape_access = MemberAccessIR(
                        object=access.array, member="shape", location=location, type_info=UNKNOWN
                    )
                    dim_lit = LiteralIR(value=dim_idx, location=location, type_info=infer_literal_type(dim_idx))
                    shape_dim = RectangularAccessIR(
                        array=shape_access, indices=[dim_lit], location=location, type_info=PrimitiveType("i32")
                    )
                    return RangeIR(
                        start=LiteralIR(value=0, location=location, type_info=infer_literal_type(0)),
                        end=shape_dim,
                        location=location,
                        type_info=UNKNOWN
                    )
        return None

    def _infer_range_from_array_access(self, var_name: str, expr: ExpressionIR, location) -> Optional[RangeIR]:
        """
        Infer range for an index variable from how it's used in array accesses.
        
        Similar to RestPatternPreprocessingPass._infer_range_for_expanded_var, but for regular index variables.
        Finds array accesses that use this variable and infers range from the array shape.
        
        Note: The dimension index must account for rest pattern expansion. If we have matrix[..batch, i, j],
        after expansion the indices become [batch.0, i, j], but we need to find the original position
        of i and j in the access pattern to determine the correct array dimension.
        """
        from ..ir.nodes import RectangularAccessIR, IdentifierIR, IndexVarIR, LiteralIR, MemberAccessIR
        
        def _name_of(idx_expr):
            return getattr(idx_expr, "name", None) if (isinstance(idx_expr, (IdentifierIR, IndexVarIR))) else None

        def _index_expr_uses_var(idx_expr, vname: str) -> bool:
            if idx_expr is None:
                return False
            if isinstance(idx_expr, (IdentifierIR, IndexVarIR)):
                return getattr(idx_expr, "name", None) == vname
            if isinstance(idx_expr, BinaryOpIR):
                return _index_expr_uses_var(getattr(idx_expr, "left", None), vname) or _index_expr_uses_var(getattr(idx_expr, "right", None), vname)
            if hasattr(idx_expr, "operand"):
                return _index_expr_uses_var(getattr(idx_expr, "operand", None), vname)
            if hasattr(idx_expr, "object") and hasattr(idx_expr, "member"):
                return _index_expr_uses_var(getattr(idx_expr, "object", None), vname)
            return False
        
        # Find all array accesses in the expression
        accesses = self._find_array_accesses(expr)
        
        for access in accesses:
            # Check each dimension of the access
            if not isinstance(access, RectangularAccessIR):
                continue
            
            # Handle both flat and nested index lists
            indices_list = access.indices
            if not indices_list:
                continue
            
            # Count rest pattern dimensions to adjust dimension index
            # Rest patterns like ..batch expand to batch.0, batch.1, etc.
            # We need to count how many dimensions they occupy
            rest_pattern_dims = 0
            for idx_expr in (indices_list[0] if isinstance(indices_list[0], list) else indices_list):
                if isinstance(idx_expr, (IdentifierIR, IndexVarIR)):
                    # Check if this is an expanded rest pattern (e.g., batch.0, batch.1)
                    if '.' in idx_expr.name and idx_expr.name.split('.')[0] in getattr(self, '_rest_pattern_names', set()):
                        rest_pattern_dims += 1
                    elif idx_expr.name.startswith('..'):
                        # Unexpanded rest pattern - count as 1 for now (will be expanded later)
                        rest_pattern_dims += 1
            
            if isinstance(indices_list[0], list):
                # Nested lists (Einstein notation format)
                for idx_group in indices_list:
                    for dim_idx, idx_expr in enumerate(idx_group):
                        if _name_of(idx_expr) == var_name:
                            # Found usage: array[..., var_name, ...]
                            # Adjust dimension index: if rest patterns come before, we need to account for them
                            # For now, use the position in the current group
                            # The actual dimension will be determined at runtime from the array shape
                            array_expr = access.array
                            # Create array.shape[dim_idx] expression
                            # Note: dim_idx here is the position in the indices, which should match the array dimension
                            shape_access = MemberAccessIR(
                                object=array_expr,
                                member='shape',
                                location=location,
                                type_info=UNKNOWN
                            )
                            dim_literal = LiteralIR(value=dim_idx, location=location, type_info=infer_literal_type(dim_idx))
                            shape_dim = RectangularAccessIR(
                                array=shape_access,
                                indices=[dim_literal],
                                location=location,
                                type_info=PrimitiveType("i32")
                            )
                            start_lit = LiteralIR(value=0, location=location, type_info=infer_literal_type(0))
                            return RangeIR(start=start_lit, end=shape_dim, location=location, type_info=UNKNOWN)
            else:
                # Flat list (indices are often IndexVarIR, or compound e.g. j-k)
                for dim_idx, idx_expr in enumerate(indices_list):
                    if _name_of(idx_expr) == var_name or _index_expr_uses_var(idx_expr, var_name):
                        array_expr = access.array
                        shape_access = MemberAccessIR(
                            object=array_expr,
                            member='shape',
                            location=location,
                            type_info=UNKNOWN
                        )
                        dim_literal = LiteralIR(value=dim_idx, location=location, type_info=infer_literal_type(dim_idx))
                        shape_dim = RectangularAccessIR(
                            array=shape_access,
                            indices=[dim_literal],
                            location=location,
                            type_info=PrimitiveType("i32")
                        )
                        start_lit = LiteralIR(value=0, location=location, type_info=infer_literal_type(0))
                        return RangeIR(start=start_lit, end=shape_dim, location=location, type_info=UNKNOWN)
        return None

    def _infer_range_from_reduction_value(
        self, value: ExpressionIR, var_name: str, defid: DefId, location
    ) -> Optional[RangeIR]:
        """infer range for a reduction variable from reduction in value (loop_var_ranges or body)."""
        from ..ir.nodes import ReductionExpressionIR
        red = self._find_reduction_in_value(value)
        if red is None:
            return None
        if getattr(red, 'loop_var_ranges', None) and defid in red.loop_var_ranges:
            r = red.loop_var_ranges[defid]
            if isinstance(r, RangeIR):
                return r
        return self._infer_range_from_array_access(var_name, red.body, location)

    def _find_reduction_in_value(self, expr: ExpressionIR):
        """Find first ReductionExpressionIR in expression (e.g. inside BinaryOp)."""
        from ..ir.nodes import ReductionExpressionIR, BinaryOpIR
        if isinstance(expr, ReductionExpressionIR):
            return expr
        if isinstance(expr, BinaryOpIR):
            left = self._find_reduction_in_value(expr.left) if expr.left else None
            if left is not None:
                return left
            return self._find_reduction_in_value(expr.right) if expr.right else None
        if hasattr(expr, 'body'):
            return self._find_reduction_in_value(getattr(expr, 'body'))
        if hasattr(expr, 'operand'):
            return self._find_reduction_in_value(getattr(expr, 'operand'))
        return None
    
    def _find_array_accesses(self, expr: ExpressionIR) -> List[RectangularAccessIR]:
        """Find all array accesses in an expression tree"""
        from ..ir.nodes import RectangularAccessIR
        
        accesses = []
        
        class ArrayAccessFinder(IRVisitor[None]):
            def visit_rectangular_access(self, node: RectangularAccessIR) -> None:
                accesses.append(node)
                # Also visit the array expression (might be nested)
                if node.array:
                    node.array.accept(self)
            
            # Implement all required abstract methods with pass
            def visit_identifier(self, node): pass
            def visit_literal(self, node): pass
            def visit_binary_op(self, node): 
                if node.left: node.left.accept(self)
                if node.right: node.right.accept(self)
            def visit_unary_op(self, node): 
                if node.operand: node.operand.accept(self)
            def visit_function_call(self, node):
                # IR FunctionCallIR has no function_expr (callee is name/defid); AST has function_expr
                callee = getattr(node, 'function_expr', None)
                if callee:
                    callee.accept(self)
                for arg in node.arguments:
                    arg.accept(self)
            def visit_member_access(self, node):
                if node.object: node.object.accept(self)
            def visit_reduction_expression(self, node):
                if node.body:
                    node.body.accept(self)
            def visit_array_literal(self, node): pass
            def visit_tuple_expression(self, node):
                for el in getattr(node, 'elements', []) or []:
                    if el is not None and hasattr(el, 'accept'):
                        el.accept(self)
            def visit_block_expression(self, node):
                if hasattr(node, 'statements') and node.statements:
                    for stmt in node.statements:
                        stmt.accept(self)
                if hasattr(node, 'final_expr') and node.final_expr:
                    node.final_expr.accept(self)
            def visit_if_expression(self, node):
                if node.condition:
                    node.condition.accept(self)
                if node.then_expr:
                    node.then_expr.accept(self)
                if node.else_expr:
                    node.else_expr.accept(self)
            def visit_match_expression(self, node): pass
            def visit_range(self, node): pass
            def visit_where_expression(self, node):
                if getattr(node, 'expr', None):
                    node.expr.accept(self)
                for c in getattr(node, 'constraints', None) or []:
                    if c is not None and hasattr(c, 'accept'):
                        c.accept(self)
            def visit_lambda(self, node): pass
            def visit_cast_expression(self, node):
                if getattr(node, 'expr', None):
                    node.expr.accept(self)
            def visit_builtin_call(self, node): pass
            def visit_pipeline_expression(self, node): pass
            def visit_try_expression(self, node): pass
            def visit_interpolated_string(self, node): pass
            def visit_jagged_access(self, node): pass
            def visit_tuple_access(self, node):
                if getattr(node, 'tuple_expr', None) is not None:
                    node.tuple_expr.accept(self)
            def visit_arrow_expression(self, node): pass
            def visit_constant_def(self, node): pass
            def visit_variable_declaration(self, node): pass
            def visit_einstein_declaration(self, node): pass
            def visit_function_def(self, node): pass
            def visit_function_ref(self, node): pass
            def visit_program(self, node): pass
            def visit_module(self, node): pass
            def visit_identifier_pattern(self, node): pass
            def visit_literal_pattern(self, node): pass
            def visit_tuple_pattern(self, node): pass
            def visit_array_pattern(self, node): pass
            def visit_rest_pattern(self, node): pass
            def visit_guard_pattern(self, node): pass
            def visit_wildcard_pattern(self, node): pass
            def visit_array_comprehension(self, node): pass
        
        finder = ArrayAccessFinder()
        expr.accept(finder)
        return accesses

    def _extract_reduction_ranges(self, expr: ExpressionIR, location) -> Dict[DefId, LoopStructure]:
        """Extract reduction variables and their ranges from expression. Keys are variable DefIds."""
        reduction_ranges: Dict[DefId, LoopStructure] = {}
        
        class ReductionFinder(IRVisitor[None]):
            def __init__(self, parent: 'EinsteinLoweringVisitor'):
                self.parent = parent
                self.reductions = []
            
            
            # Required visitor methods (no-op for others)
            def visit_literal(self, node) -> None: pass
            def visit_identifier(self, node) -> None: pass
            def visit_binary_op(self, node) -> None:
                if node.left: node.left.accept(self)
                if node.right: node.right.accept(self)
            def visit_unary_op(self, node) -> None:
                if node.operand: node.operand.accept(self)
            def visit_rectangular_access(self, node) -> None:
                if node.array: node.array.accept(self)
                for idx in (node.indices or []):
                    idx.accept(self)
            def visit_jagged_access(self, node) -> None:
                if node.base: node.base.accept(self)
                for idx in (getattr(node, 'index_chain', None) or []):
                    idx.accept(self)
            def visit_function_call(self, node) -> None:
                for arg in node.arguments: arg.accept(self)
            def visit_array_comprehension(self, node) -> None: pass
            def visit_array_literal(self, node) -> None: pass
            def visit_tuple_expression(self, node) -> None: pass
            def visit_tuple_access(self, node) -> None: pass
            def visit_block_expression(self, node) -> None: pass
            def visit_if_expression(self, node) -> None:
                if node.condition: node.condition.accept(self)
                if node.then_expr: node.then_expr.accept(self)
                if node.else_expr: node.else_expr.accept(self)
            def visit_lambda(self, node) -> None: pass
            def visit_range(self, node) -> None: pass
            def visit_where_expression(self, node) -> None:
                if node.expr: node.expr.accept(self)
            def visit_arrow_expression(self, node) -> None: pass
            def visit_cast_expression(self, node) -> None:
                if node.expr: node.expr.accept(self)
            def visit_member_access(self, node) -> None:
                if node.object: node.object.accept(self)
            def visit_try_expression(self, node) -> None: pass
            def visit_match_expression(self, node) -> None: pass
            def visit_interpolated_string(self, node) -> None: pass
            def visit_function_def(self, node) -> None: pass
            def visit_constant_def(self, node) -> None: pass
            def visit_einstein_declaration(self, node) -> None: pass
            def visit_reduction_expression(self, node) -> None:
                self.reductions.append(node)
                if node.body:
                    node.body.accept(self)
            def visit_where_expression(self, node) -> None:
                if node.expr:
                    node.expr.accept(self)
            # Missing abstract methods
            def visit_module(self, node) -> None: pass
            def visit_program(self, node) -> None: pass
            def visit_pipeline_expression(self, node) -> None: pass
            def visit_builtin_call(self, node) -> None: pass
            def visit_function_ref(self, node) -> None: pass
            def visit_literal_pattern(self, node) -> None: pass
            def visit_identifier_pattern(self, node) -> None: pass
            def visit_wildcard_pattern(self, node) -> None: pass
            def visit_tuple_pattern(self, node) -> None: pass
            def visit_array_pattern(self, node) -> None: pass
            def visit_rest_pattern(self, node) -> None: pass
            def visit_guard_pattern(self, node) -> None: pass
            def visit_variable_declaration(self, node) -> None:
                # Visit the value expression to find reductions
                if node.value: node.value.accept(self)
        
        finder = ReductionFinder(self)
        expr.accept(finder)
        
        # Process each reduction found
        for reduction in finder.reductions:
            for var_ident in reduction.loop_vars:
                var_name = var_ident.name
                # Get range from loop_var_ranges (explicit ranges in reduction)
                range_obj = None
                if reduction.loop_var_ranges and getattr(var_ident, 'defid', None) in reduction.loop_var_ranges:
                    range_ir = reduction.loop_var_ranges[var_ident.defid]
                    # Convert RangeIR to iterable
                    if isinstance(range_ir, RangeIR):
                        # Try to evaluate as constant
                        start_val = self._evaluate_constant(range_ir.start)
                        end_val = self._evaluate_constant(range_ir.end)
                        if start_val is not None and end_val is not None:
                            range_obj = range(start_val, end_val)
                        else:
                            body_defid = _defid_of_var_in_expr(reduction.body, var_name) if reduction.body else None
                            defid = body_defid or getattr(var_ident, "defid", None)
                            if defid is None:
                                raise ValueError(
                                    f"Reduction loop var '{var_name}' must have defid. "
                                    "Ensure name resolution sets defid on reduction loop_vars."
                                )
                            ti = getattr(var_ident, 'type_info', None) or PrimitiveType("i32")
                            var_ident = IdentifierIR(var_name, location, defid=defid, type_info=ti)
                            reduction_ranges[defid] = LoopStructure(variable=var_ident, iterable=range_ir)
                            continue
                    elif isinstance(range_ir, LiteralIR) and isinstance(range_ir.value, range):
                        range_obj = range_ir.value
                
                if range_obj:
                    iterable = self._range_to_iterable_ir(range_obj, location)
                    body_defid = _defid_of_var_in_expr(reduction.body, var_name) if reduction.body else None
                    defid = body_defid or getattr(var_ident, "defid", None)
                    if defid is None:
                        raise ValueError(
                            f"Reduction loop var '{var_name}' must have defid. "
                            "Ensure name resolution sets defid on reduction loop_vars."
                        )
                    ti = getattr(var_ident, 'type_info', None) or PrimitiveType("i32")
                    var_ident = IdentifierIR(var_name, location, defid=defid, type_info=ti)
                    reduction_ranges[defid] = LoopStructure(variable=var_ident, iterable=iterable)
        
        return reduction_ranges
    
    def _extract_bindings_and_guards(self, where_clause: Optional[WhereClauseIR]) -> Tuple[List[BindingIR], List[GuardCondition]]:
        """Extract bindings and guards from where clause"""
        bindings = []
        guards = []
        
        if not where_clause:
            return bindings, guards
        
        # For now, we treat all constraints as guards
        # In a more sophisticated implementation, we'd distinguish between
        # bindings (let x = expr) and guards (x > 0)
        for constraint in where_clause.constraints:
            # Check if constraint is a binding (equality with identifier on left)
            # For now, treat all as guards
            guards.append(GuardCondition(condition=constraint))
        
        return bindings, guards
    
    def _shape_from_literal_indices(self, indices: List, location: SourceLocation) -> Optional[List[ExpressionIR]]:
        """When clause has only literal indices (e.g. fib[0]=3), extent per dim = index+1. Shape pass is responsible; this is fallback when lookup fails."""
        if not indices:
            return None
        out = []
        for idx in indices:
            if isinstance(idx, LiteralIR):
                v = getattr(idx, "value", None)
                try:
                    extent = int(v) + 1 if v is not None else 1
                except (TypeError, ValueError):
                    return None
                extent = max(1, extent)
                out.append(LiteralIR(
                    value=extent,
                    location=location,
                    shape_info=None,
                    type_info=infer_literal_type(extent),
                ))
            else:
                return None
        return out if out else None

    def _get_shape_from_analysis(self, node: EinsteinDeclarationIR) -> Optional[List[ExpressionIR]]:
        """
        Get shape from UnifiedShapeAnalysisPass results.
        
        This ensures we use static dimensions computed by shape analysis,
        rather than dynamic expressions that reference reduction variables.
        """
        from ..passes.shape_analysis import UnifiedShapeAnalysisPass
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            shape_analysis = self.tcx.get_analysis(UnifiedShapeAnalysisPass)
            if isinstance(shape_analysis, dict) and 'expr_shapes' in shape_analysis:
                expr_shapes = shape_analysis['expr_shapes']
                logger.debug(f"[EinsteinLowering] {node.name}: Checking shape in expr_shapes, found {len(expr_shapes)} entries")
                logger.debug(f"[EinsteinLowering] {node.name}: Node id={id(node)}, in dict? {node in expr_shapes}")
                if node in expr_shapes:
                    shape_tuple = expr_shapes[node]
                    logger.debug(f"[EinsteinLowering] {node.name}: Found shape: {shape_tuple}")
                    # Convert tuple of ints to List[LiteralIR]
                    shape_list = []
                    for dim in shape_tuple:
                        if isinstance(dim, int):
                            shape_list.append(LiteralIR(
                                value=dim,
                                location=node.location,
                                shape_info=None,
                                type_info=infer_literal_type(dim)
                            ))
                        else:
                            logger.debug(f"[EinsteinLowering] Non-static shape dimension: {dim}")
                            return None
                    return shape_list
                # No shape in analysis
        except RuntimeError:
            pass
        return None
    
    def _compute_shape(self, loops: List[LoopStructure]) -> Optional[List[ExpressionIR]]:
        """Compute output shape (memory allocation extent) from loop iterables.
        Uses range.stop (extent), not len (iteration count). This is the single source
        of truth when we have loops so allocation matches iteration."""
        shape = []
        for loop in loops:
            it = loop.iterable
            if isinstance(it, LiteralIR) and isinstance(getattr(it, "value", None), range):
                range_obj = it.value
                shape.append(LiteralIR(
                    value=range_obj.stop, location=it.location, shape_info=None,
                    type_info=infer_literal_type(range_obj.stop)
                ))
            elif isinstance(it, RangeIR):
                shape.append(it.end)
            elif getattr(it, "end", None) is not None:
                # Any iterable with .end (e.g. RangeIR from another module)
                shape.append(it.end)
            else:
                return None
        return shape if shape else None

    def _shape_from_clause_indices(self, clause: EinsteinIR, location: SourceLocation) -> Optional[List[ExpressionIR]]:
        """Shape from LHS indices: per-dimension end (exclusive) so union of consecutive clauses is max(ends)."""
        out = []
        var_ranges = getattr(clause, "variable_ranges", None) or {}
        for idx in (clause.indices or []):
            if isinstance(idx, LiteralIR):
                try:
                    k = int(idx.value)
                except (TypeError, ValueError):
                    return None
                out.append(LiteralIR(
                    value=k + 1, location=location, shape_info=None,
                    type_info=infer_literal_type(k + 1)
                ))
                continue
            if isinstance(idx, IndexVarIR):
                r = getattr(idx, "range_ir", None)
                if r is None and getattr(idx, "defid", None):
                    r = var_ranges.get(idx.defid)
                if isinstance(r, RangeIR):
                    out.append(r.end)
                elif isinstance(r, range):
                    out.append(LiteralIR(
                        value=r.stop, location=location, shape_info=None,
                        type_info=infer_literal_type(r.stop)
                    ))
                elif r is not None and hasattr(r, "start") and hasattr(r, "end"):
                    out.append(LiteralIR(
                        value=r.end, location=location, shape_info=None,
                        type_info=infer_literal_type(r.end)
                    ))
                else:
                    return None
            else:
                return None
        return out if out else None

    def _extent_as_int(self, shape_dim: ExpressionIR) -> Optional[int]:
        """Get extent (end) as int if shape dimension is a constant literal."""
        if isinstance(shape_dim, LiteralIR) and shape_dim.value is not None:
            try:
                return int(shape_dim.value)
            except (TypeError, ValueError):
                pass
        return None

    def _extent_from_shape_expr(self, expr: Any) -> Optional[int]:
        """Extract int from a shape dimension expr (LiteralIR or expr with .end)."""
        if expr is None:
            return None
        if isinstance(expr, LiteralIR) and expr.value is not None:
            try:
                return int(expr.value)
            except (TypeError, ValueError):
                return None
        if getattr(expr, 'end', None) is not None:
            return self._extent_from_shape_expr(expr.end)
        return None

    def _concrete_shape_tuple(self, shape_exprs: List[Any]) -> Optional[Tuple[int, ...]]:
        """If all shape elements yield an int, return (d0, d1, ...); else None."""
        if not shape_exprs:
            return None
        dims = []
        for e in shape_exprs:
            d = self._extent_from_shape_expr(e)
            if d is None:
                return None
            dims.append(d)
        return tuple(dims)

    def _compute_shape_union(self, shapes: List[List[ExpressionIR]], location: SourceLocation) -> Optional[List[ExpressionIR]]:
        """Union of clause shapes: for each dimension, use max of end (so allocation covers all clauses)."""
        if not shapes:
            return None
        rank = len(shapes[0])
        if not all(len(s) == rank for s in shapes):
            return None
        result = []
        for d in range(rank):
            dim_values = [s[d] for s in shapes]
            ints = [self._extent_as_int(x) for x in dim_values]
            if all(x is not None for x in ints):
                max_val = max(ints)
                result.append(LiteralIR(
                    value=max_val, location=location, shape_info=None,
                    type_info=infer_literal_type(max_val)
                ))
            else:
                # Prefer non-literal (variable) extent so allocation covers runtime size (e.g. seq_length+1)
                best = None
                best_int = -1
                for x in dim_values:
                    v = self._extent_as_int(x)
                    if v is None:
                        best = x
                        break
                    if v > best_int:
                        best_int = v
                        best = x
                if best is None:
                    best = dim_values[0]
                result.append(best)
        return result
    
    def visit_literal(self, node) -> Any:
        return node

    def visit_identifier(self, node) -> Any:
        return node

    def visit_index_var(self, node) -> Any:
        return node

    def visit_index_rest(self, node) -> Any:
        return node

    def visit_binary_op(self, node) -> Any:
        if node.left is not None:
            node.left = node.left.accept(self)
        if node.right is not None:
            node.right = node.right.accept(self)
        return node

    def visit_unary_op(self, node) -> Any:
        if node.operand is not None:
            node.operand = node.operand.accept(self)
        return node

    def visit_rectangular_access(self, node) -> Any:
        if node.array is not None:
            node.array = node.array.accept(self)
        indices = getattr(node, 'indices', None) or []
        for i, idx in enumerate(indices):
            if idx is None:
                raise ValueError("IR index slot is None")
            res = idx.accept(self)
            if res is None:
                raise ValueError("IR index slot became None after transform")
            node.indices[i] = res
        return node

    def visit_jagged_access(self, node) -> Any:
        if node.base is not None:
            node.base = node.base.accept(self)
        chain = getattr(node, 'index_chain', None) or []
        for i, idx in enumerate(chain):
            if idx is None:
                raise ValueError("IR index_chain slot is None")
            res = idx.accept(self)
            if res is None:
                raise ValueError("IR index_chain slot became None after transform")
            node.index_chain[i] = res
        return node

    def visit_function_call(self, node) -> Any:
        for i, arg in enumerate(node.arguments):
            if arg is None:
                raise ValueError("IR function call argument is None")
            node.arguments[i] = arg.accept(self)
        return node

    def visit_array_comprehension(self, node) -> Any:
        r = self.lower_array_comprehension(node)
        return r if r is not None else node

    def visit_array_literal(self, node) -> Any:
        for i, elem in enumerate(node.elements):
            if elem is None:
                raise ValueError("IR array literal element is None")
            node.elements[i] = elem.accept(self)
        return node

    def visit_tuple_expression(self, node) -> Any:
        for i, elem in enumerate(node.elements):
            if elem is None:
                raise ValueError("IR tuple element is None")
            node.elements[i] = elem.accept(self)
        return node

    def visit_tuple_access(self, node) -> Any:
        if node.tuple_expr is not None:
            node.tuple_expr = node.tuple_expr.accept(self)
        return node

    def visit_block_expression(self, node) -> Any:
        stmts = getattr(node, 'statements', None) or []
        for i, stmt in enumerate(stmts):
            if stmt is None:
                raise ValueError("IR block statement is None")
            result = stmt.accept(self)
            if isinstance(stmt, EinsteinDeclarationIR) and isinstance(result, LoweredEinsteinIR):
                node.statements[i] = VariableDeclarationIR(
                    getattr(stmt, "name", ""),
                    result,
                    location=getattr(stmt, "location", None),
                    defid=getattr(stmt, "defid", None),
                )
            else:
                node.statements[i] = result
        if node.final_expr is not None:
            node.final_expr = node.final_expr.accept(self)
        return node

    def visit_if_expression(self, node) -> Any:
        if node.condition is not None:
            node.condition = node.condition.accept(self)
        if node.then_expr is not None:
            node.then_expr = node.then_expr.accept(self)
        if node.else_expr is not None:
            node.else_expr = node.else_expr.accept(self)
        return node

    def visit_lambda(self, node) -> Any:
        if node.body is not None:
            node.body = node.body.accept(self)
        return node

    def visit_range(self, node) -> Any:
        return node

    def visit_reduction_expression(self, node) -> Any:
        return self.lower_reduction_expression(node)

    def visit_lowered_reduction(self, node) -> Any:
        if getattr(node, 'body', None) is not None:
            node.body = node.body.accept(self)
        for loop in getattr(node, 'loops', None) or []:
            if getattr(loop, 'iterable', None) is not None:
                loop.iterable = loop.iterable.accept(self)
        for loop in getattr(node, 'loops', None) or []:
            var = getattr(loop, 'variable', None)
            if var is None:
                continue
            body_defid = _defid_of_var_in_expr(node.body, var.name)
            if body_defid is not None and body_defid != getattr(var, 'defid', None):
                loc = getattr(var, 'location', None) or SourceLocation('', 0, 0)
                ti = getattr(var, 'type_info', None)
                if isinstance(var, IndexVarIR):
                    new_var = IndexVarIR(var.name, loc, defid=body_defid, range_ir=getattr(var, 'range_ir', None), type_info=ti)
                else:
                    new_var = IdentifierIR(var.name, loc, defid=body_defid, type_info=ti)
                object.__setattr__(loop, 'variable', new_var)
        for g in getattr(node, 'guards', None) or []:
            if getattr(g, 'condition', None) is not None:
                g.condition = g.condition.accept(self)
        return node

    def visit_lowered_comprehension(self, node) -> Any:
        if getattr(node, 'body', None) is not None:
            node.body = node.body.accept(self)
        for loop in getattr(node, 'loops', None) or []:
            if getattr(loop, 'iterable', None) is not None:
                loop.iterable = loop.iterable.accept(self)
        for g in getattr(node, 'guards', None) or []:
            if getattr(g, 'condition', None) is not None:
                g.condition = g.condition.accept(self)
        return node

    def visit_lowered_einstein_clause(self, node) -> Any:
        if getattr(node, 'body', None) is not None:
            node.body = node.body.accept(self)
        for loop in getattr(node, 'loops', None) or []:
            if getattr(loop, 'iterable', None) is not None:
                loop.iterable = loop.iterable.accept(self)
        for b in getattr(node, 'bindings', None) or []:
            if getattr(b, 'value', None) is not None:
                b.value = b.value.accept(self)
        for g in getattr(node, 'guards', None) or []:
            if getattr(g, 'condition', None) is not None:
                g.condition = g.condition.accept(self)
        return node

    def visit_lowered_einstein(self, node) -> Any:
        for item in getattr(node, 'items', None) or []:
            item.accept(self)
        if getattr(node, 'shape', None) is not None:
            if isinstance(node.shape, list):
                for s in node.shape:
                    if s is not None:
                        s.accept(self)
            else:
                node.shape.accept(self)
        return node

    def visit_where_expression(self, node) -> Any:
        from ..ir.nodes import WhereExpressionIR, ReductionExpressionIR, GuardCondition
        if isinstance(node, ReductionExpressionIR):
            return self.visit_reduction_expression(node)
        if not isinstance(node, WhereExpressionIR) or node.expr is None:
            return node
        if isinstance(node.expr, ReductionExpressionIR):
            lowered = self.lower_reduction_expression(node.expr)
            if lowered and getattr(node, 'constraints', None):
                outer_guards = [GuardCondition(condition=c) for c in node.constraints]
                lowered.guards = list(lowered.guards) + outer_guards
            return lowered
        node.expr = node.expr.accept(self)
        return node

    def visit_arrow_expression(self, node) -> Any:
        if node.left is not None:
            node.left = node.left.accept(self)
        if node.right is not None:
            node.right = node.right.accept(self)
        return node

    def visit_cast_expression(self, node) -> Any:
        if node.expr is not None:
            orig = node.expr
            node.expr = orig.accept(self)
            if getattr(node.expr, 'type_info', None) is None and getattr(orig, 'type_info', None) is not None:
                node.expr.type_info = orig.type_info
        return node

    def visit_member_access(self, node) -> Any:
        if node.object is not None:
            node.object = node.object.accept(self)
        return node

    def visit_try_expression(self, node) -> Any:
        if node.expr is not None:
            node.expr = node.expr.accept(self)
        return node

    def visit_match_expression(self, node) -> Any:
        if node.scrutinee is not None:
            node.scrutinee = node.scrutinee.accept(self)
        for arm in node.arms:
            if arm.body is not None:
                arm.body = arm.body.accept(self)
        return node

    def visit_interpolated_string(self, node) -> Any:
        for i, part in enumerate(node.parts):
            if isinstance(part, ExpressionIR):
                node.parts[i] = part.accept(self)
        return node

    def visit_function_def(self, node) -> Any:
        from ..analysis.analysis_guard import should_analyze_function
        if not should_analyze_function(node, tcx=self.tcx):
            import logging
            logger = logging.getLogger("einlang.passes.einstein_lowering")
            logger.debug(f"Skipping Einstein lowering for generic function: {node.name}")
            return node
        if node.body is not None:
            node.body = node.body.accept(self)
        return node

    def visit_constant_def(self, node) -> Any:
        if node.value is not None:
            node.value = node.value.accept(self)
        return node

    def _visit_statements(self, node) -> Any:
        if hasattr(node, 'statements'):
            for i, stmt in enumerate(node.statements):
                if stmt is None:
                    raise ValueError("IR statement is None")
                node.statements[i] = stmt.accept(self)
        return node

    def visit_module(self, node) -> Any:
        return self._visit_statements(node)

    def visit_program(self, node) -> Any:
        return self._visit_statements(node)

    def visit_pipeline_expression(self, node) -> Any:
        if hasattr(node, 'left') and node.left is not None:
            node.left = node.left.accept(self)
        if hasattr(node, 'right') and node.right is not None:
            node.right = node.right.accept(self)
        return node

    def visit_builtin_call(self, node) -> Any:
        if hasattr(node, 'args'):
            for i, arg in enumerate(node.args):
                if arg is None:
                    raise ValueError("IR builtin call argument is None")
                node.args[i] = arg.accept(self)
        return node

    def visit_function_ref(self, node) -> Any:
        return node

    def visit_literal_pattern(self, node) -> Any:
        return node

    def visit_identifier_pattern(self, node) -> Any:
        return node

    def visit_wildcard_pattern(self, node) -> Any:
        return node

    def visit_tuple_pattern(self, node) -> Any:
        if hasattr(node, 'patterns'):
            for i, p in enumerate(node.patterns):
                node.patterns[i] = p.accept(self)
        return node

    def visit_array_pattern(self, node) -> Any:
        if hasattr(node, 'patterns'):
            for i, p in enumerate(node.patterns):
                node.patterns[i] = p.accept(self)
        return node

    def visit_rest_pattern(self, node) -> Any:
        return node

    def visit_guard_pattern(self, node) -> Any:
        if hasattr(node, 'pattern') and node.pattern is not None:
            node.pattern = node.pattern.accept(self)
        if hasattr(node, 'guard') and node.guard is not None:
            node.guard = node.guard.accept(self)
        return node

    def visit_variable_declaration(self, node) -> Any:
        if hasattr(node, 'value') and node.value is not None:
            node.value = node.value.accept(self)
        return node

