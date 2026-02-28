"""
Rest Pattern Preprocessing Pass

Rust Pattern: N/A (Einlang-specific)
Reference: PASS_SYSTEM_DESIGN.md

Design Pattern: Visitor pattern for IR traversal (no isinstance/hasattr)

Expands rest patterns immediately in IR (clause indices and body). No expansion metadata stored.

Indices are flat from the start: the AST (einstein_indices, expr_list) and ast_to_ir produce
only flat index lists. No pass introduces nested index structure; we do not flatten afterwards.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any

from ..passes.base import BasePass, TyCtxt
from ..passes.einstein_grouping import EinsteinDeclarationGroupingPass
from ..ir.nodes import (
    ProgramIR, EinsteinIR, IRVisitor, IRNode, is_einstein_binding,
    FunctionDefIR, BlockExpressionIR, IfExpressionIR,
    IdentifierIR, IndexVarIR, IndexRestIR, RectangularAccessIR, ReductionExpressionIR,
    ExpressionIR, BinaryOpIR, UnaryOpIR, LiteralIR,
    BindingIR, RangeIR, MemberAccessIR
)
from ..ir.scoped_visitor import ScopedIRVisitor
from ..shared.defid import DefId
from ..shared.source_location import SourceLocation
from ..shared.types import infer_literal_type, UNKNOWN

logger = logging.getLogger("einlang.passes.rest_pattern_preprocessing")


class RestPatternPreprocessingPass(BasePass):
    """
    Preprocess rest patterns to determine matched index groups for static rank cases.
    
    Rust Pattern: N/A (Einlang-specific)
    
    Detects rest patterns in Einstein declarations and reduction expressions and
    expands them when rank can be inferred. Otherwise defers expansion to later passes.
    """
    requires = [EinsteinDeclarationGroupingPass]  # Depends on Einstein grouping
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """
        Preprocess rest patterns in Einstein declarations.
        
        Rust Pattern: Analysis pass stores results in TyCtxt
        """
        logger.debug("Starting rest pattern preprocessing")

        # Get Einstein declaration groups from previous pass
        try:
            einstein_groups = tcx.get_analysis(
                __import__('einlang.passes.einstein_grouping', fromlist=['EinsteinDeclarationGroupingPass']).EinsteinDeclarationGroupingPass
            )
        except RuntimeError:
            einstein_groups = {}
        
        # Collect Einstein declarations and process rest patterns
        visitor = RestPatternPreprocessor(tcx, einstein_groups, program=ir)
        
        for stmt in ir.statements:
            if isinstance(stmt, BindingIR) and getattr(stmt, 'defid', None) is not None:
                visitor.set_var(stmt.defid, stmt)
        
        for func in ir.functions:
            func.accept(visitor)
        for mod in getattr(ir, 'modules', None) or []:
            mod.accept(visitor)
        
        # Visit all constants
        for const in ir.constants:
            const.value.accept(visitor)
        
        for stmt in ir.statements:
            if is_einstein_binding(stmt):
                visitor.visit_einstein_declaration(stmt)
            elif hasattr(stmt, 'accept'):
                stmt.accept(visitor)
        
        tcx.set_analysis(RestPatternPreprocessingPass, {})
        
        logger.debug(
            f"Rest pattern preprocessing complete: "
            f"{visitor.patterns_preprocessed} patterns preprocessed, "
            f"{visitor.patterns_deferred} deferred"
        )
        
        return ir

    def process_specialized_functions(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """
        Run rest pattern preprocessing on specialized function bodies (incremental specialization).
        Type pass calls this after adding specialized functions so rest-pattern handles its part.
        """
        specialized_funcs = getattr(tcx, 'specialized_functions', [])
        if not specialized_funcs:
            return ir
        try:
            einstein_groups = tcx.get_analysis(
                __import__('einlang.passes.einstein_grouping', fromlist=['EinsteinDeclarationGroupingPass']).EinsteinDeclarationGroupingPass
            )
        except RuntimeError:
            einstein_groups = {}
        visitor = RestPatternPreprocessor(tcx, einstein_groups, program=ir)
        for stmt in ir.statements:
            if isinstance(stmt, BindingIR) and getattr(stmt, 'defid', None) is not None:
                visitor.set_var(stmt.defid, stmt)
        for func in specialized_funcs:
            try:
                logger.debug(f"[RestPatternPreprocessing] Processing specialized function {func.name}")
                func.accept(visitor)
            except Exception as e:
                logger.warning(f"Rest pattern preprocessing failed for specialized {func.name}: {e}")
        tcx.set_analysis(RestPatternPreprocessingPass, {})
        return ir

class RestPatternPreprocessor(ScopedIRVisitor[None]):
    """
    Rest pattern preprocessor - implements the actual preprocessing logic.
    
    Rust Pattern: rustc_hir::intravisit::Visitor
    Scope stack keyed by DefId (no name-based lookup).
    """
    
    def __init__(self, tcx: TyCtxt, einstein_groups: Dict[str, Any], program: Optional[ProgramIR] = None):
        super().__init__()
        self.tcx = tcx
        self.einstein_groups = einstein_groups
        self.program = program
        self.patterns_preprocessed = 0
        self.patterns_deferred = 0
        self._current_function: Optional[FunctionDefIR] = None
    
    def visit_program(self, node: ProgramIR) -> None:
        """Visit program and preprocess all Einstein declarations"""
        # Visit all statements
        for stmt in node.statements:
            stmt.accept(self)
        # Visit all functions
        for func in node.functions:
            func.accept(self)
    
    def visit_einstein_declaration(self, node: BindingIR) -> None:
        """
        Expand rest patterns in each clause. Uses first clause for rank/validation; applies to all clauses.
        """
        clauses = getattr(node, 'clauses', None) or []
        if not clauses:
            return
        first = clauses[0]
        node_name = getattr(node, 'name', '<unknown>')
        # Fail on rank mismatch: all clauses must have the same rank
        ranks = [len(c.indices) for c in clauses]
        if len(set(ranks)) > 1:
            self.tcx.reporter.report_error(
                f"Einstein declaration '{node_name}' has clauses with different ranks: {ranks}. All clauses must have the same rank.",
                location=node.location,
            )
            return
        output_rank = len(first.indices)  # Number of index positions (dimensions) for same-block lookup
        logger.debug(f"visit_einstein_declaration: {node_name}")
        
        # Check if indices contain rest patterns (from first clause)
        rest_pattern_indices = []
        output_rest_names = set()
        for i, idx_expr in enumerate(first.indices):
            logger.debug(f"  index {i}: {type(idx_expr).__name__} = {idx_expr.name if hasattr(idx_expr, 'name') else idx_expr}")
            if isinstance(idx_expr, IndexRestIR):
                rest_name = idx_expr.name
                rest_pattern_indices.append((i, rest_name))
                output_rest_names.add(rest_name)
                logger.debug(f"    -> rest pattern: {rest_name}")
        
        if not rest_pattern_indices:
            # No rest patterns, nothing to expand
            logger.debug(f"  no rest patterns found")
            return
        
        # Validate determination-first rule before expansion
        error = self._validate_rest_patterns(first, output_rest_names)
        if error:
            # Report error and skip expansion
            node_name = getattr(node, 'array_name', None) or getattr(node, 'name', None) or '<unknown>'
            self.tcx.reporter.report_error(
                error,
                location=node.location,
                code="E0005"  # Custom error code for rest pattern validation
            )
            return
        
        # Try to infer rank from body (first clause)
        logger.debug(f"  inferring rank from body...")
        rank = self._infer_rank_from_body(first)
        logger.debug(f"  inferred rank: {rank}")
        
        # Each pass handles partial specialization independently
        # If rank can't be determined from type_info, try to infer from access pattern
        if rank is None:
            # Partial specialization: infer rank from access pattern if possible
            accesses = self._find_array_accesses(first.value)
            if accesses:
                first_access = accesses[0]
                indices_list = (first_access.indices or [])
                if indices_list:
                    # Count non-rest indices
                    num_non_rest = sum(1 for idx in indices_list if not isinstance(idx, IndexRestIR))
                    # If all indices are non-rest (no rest patterns), rank = num_non_rest
                    # If there are rest patterns, we can't determine rank without type_info
                    num_rest = sum(1 for idx in indices_list if isinstance(idx, IndexRestIR))
                    if num_rest == 0:
                        # No rest patterns, rank = number of indices
                        rank = len(indices_list)
                        logger.debug(f"  inferred rank={rank} from access pattern (no rest patterns)")
                    else:
                        # All indices are rest patterns - infer per rest from access length in expansion loop
                        # (fallback: array_rank = len(indices_flat) for untyped/param RHS)
                        logger.debug(
                            f"All rest patterns in access; will infer per rest pattern from RHS in expansion loop"
                        )
            else:
                node_name = getattr(node, 'array_name', None) or getattr(node, 'name', None) or '<unknown>'
                self.tcx.reporter.report_error(
                    f"Rest pattern(s) {list(output_rest_names)} in '{node_name}' cannot be expanded: "
                    "no array accesses in body to infer rank from.",
                    location=node.location,
                    code="E0005",
                )
                return
        
        # Expand rest patterns from RHS only (: infer from body array rank, not global rank).
        # For each rest pattern, use the rank of the RHS array being accessed in a body access that uses it.
        expanded_names = []
        rest_dim_mapping = {}
        accesses = self._find_array_accesses(first.value)
        
        for i, (idx_pos, rest_name) in enumerate(rest_pattern_indices):
            # Find first body (RHS) access that uses this rest pattern; get rank from THAT array
            num_dims = None
            for access in accesses:
                indices_flat = (access.indices or [])
                rest_in_this = [idx for idx in indices_flat if isinstance(idx, IndexRestIR) and idx.name == rest_name]
                if not rest_in_this:
                    continue
                has_rest_in_flat = any(isinstance(idx, IndexRestIR) for idx in indices_flat)
                array_rank = len(indices_flat) if (indices_flat and not has_rest_in_flat) else None
                if array_rank is None:
                    array_rank = self._get_rank_for_array(access.array)
                if array_rank is None and indices_flat:
                    array_rank = len(indices_flat)
                    logger.debug(f"  rest pattern '{rest_name}': assuming array rank={array_rank} from access index count (param/untyped RHS)")
                if array_rank is None:
                    continue
                # Explicit = index slots that are NOT rest patterns (include expr indices e.g. i*stride-pad+m)
                explicit_count = sum(1 for idx in indices_flat if not isinstance(idx, IndexRestIR))
                num_dims = array_rank - explicit_count
                # Generic param type [f32; *] yields rank 1 from _get_rank_for_array, so num_dims=0 and we'd remove the rest slot.
                # When this access has both rest and explicit indices (e.g. x[..batch, j]), use access length as minimum rank.
                if num_dims <= 0 and explicit_count > 0 and len(indices_flat) > explicit_count:
                    array_rank = len(indices_flat)
                    num_dims = array_rank - explicit_count
                    logger.debug(f"  rest pattern '{rest_name}': using array rank={array_rank} from access (generic param would give num_dims<=0)")
                # Do not cap num_dims by rest_positions; a single ..pattern slot spans (array_rank - explicit_count) dimensions.
                if num_dims < 0:
                    num_dims = 0
                logger.debug(f"  rest pattern '{rest_name}': from RHS access array rank={array_rank}, explicit_count={explicit_count}, spans={num_dims}")
                break
            if num_dims is None:
                node_name = getattr(node, 'array_name', None) or getattr(node, 'name', None) or '<unknown>'
                self.tcx.reporter.report_error(
                    f"Rest pattern '..{rest_name}' in '{node_name}' cannot be expanded: "
                    "array rank could not be inferred from RHS (e.g. untyped parameter).",
                    location=node.location,
                    code="E0005",
                )
                return
            dim_indices = list(range(num_dims)) if num_dims > 0 else []
            rest_dim_mapping[rest_name] = dim_indices
            if num_dims > 0:
                for dim_idx in dim_indices:
                    expanded_names.append(f"{rest_name}.{dim_idx}")
            else:
                logger.debug(f"  rest pattern '{rest_name}' spans 0 dimensions - will be removed")
        
        for clause in clauses:
            new_indices: List[Any] = []
            rest_defid_to_expanded: Dict[Any, List[Any]] = {}
            for idx_expr in (clause.indices or []):
                if idx_expr is None:
                    raise ValueError("IR index slot is None")
                if isinstance(idx_expr, IndexRestIR):
                    rest_defid = getattr(idx_expr, "defid", None)
                    rest_name = idx_expr.name
                    if rest_name in rest_dim_mapping:
                        dim_indices = rest_dim_mapping[rest_name]
                        if dim_indices:
                            if rest_defid is None:
                                raise ValueError(
                                    f"IndexRestIR (..{rest_name}) must have defid. "
                                    "Ensure NameResolutionPass runs on AST before ASTToIRLoweringPass so defid is set on AST and copied to IR."
                                )
                            expanded_list: List[Any] = []
                            for dim_idx in dim_indices:
                                defid = self.tcx.resolver.allocate_for_local()
                                expanded_list.append(defid)
                                expanded_var = f"{rest_name}.{dim_idx}"
                                new_idx = IndexVarIR(
                                    name=expanded_var,
                                    location=idx_expr.location,
                                    defid=defid,
                                    range_ir=None
                                )
                                if hasattr(idx_expr, 'type_info'):
                                    new_idx.type_info = idx_expr.type_info
                                if hasattr(idx_expr, 'shape_info'):
                                    new_idx.shape_info = idx_expr.shape_info
                                new_indices.append(new_idx)
                                logger.debug(f"  replaced ..{rest_name} with {rest_name}.{dim_idx}")
                            if rest_defid is not None:
                                rest_defid_to_expanded[rest_defid] = expanded_list
                        else:
                            logger.debug(f"  removed ..{rest_name} (maps to 0 dimensions)")
                    else:
                        new_indices.append(idx_expr)
                else:
                    defid = getattr(idx_expr, "defid", None)
                    if isinstance(idx_expr, IndexVarIR):
                        if defid is None:
                            raise ValueError(
                                f"IndexVarIR '{idx_expr.name}' must have defid. "
                                "Ensure NameResolutionPass runs on AST before ASTToIRLoweringPass."
                            )
                        new_idx = IndexVarIR(
                            name=idx_expr.name,
                            location=idx_expr.location,
                            defid=defid,
                            range_ir=getattr(idx_expr, "range_ir", None),
                        )
                        if hasattr(idx_expr, 'type_info'):
                            new_idx.type_info = idx_expr.type_info
                        if hasattr(idx_expr, 'shape_info'):
                            new_idx.shape_info = idx_expr.shape_info
                        new_indices.append(new_idx)
                    else:
                        new_indices.append(idx_expr)
            clause.indices = new_indices
            logger.debug(f"  updated clause.indices: {[idx.name if hasattr(idx, 'name') else str(idx) for idx in clause.indices]}")
            if clause.value:
                transformer = RestPatternBodyTransformer(rest_dim_mapping, rest_defid_to_expanded, self.tcx)
                clause.value = clause.value.accept(transformer)
            if not hasattr(clause, 'variable_ranges') or clause.variable_ranges is None:
                object.__setattr__(clause, 'variable_ranges', {})
            accesses_clause = self._find_array_accesses(clause.value)
            for access in accesses_clause:
                for dim_idx, idx_expr in enumerate(access.indices or []):
                    defid = getattr(idx_expr, "defid", None)
                    if defid is None and isinstance(idx_expr, (IndexVarIR, IndexRestIR)):
                        name = getattr(idx_expr, "name", "?")
                        raise ValueError(
                            f"Index variable '{name}' must have defid in array access. "
                            "Ensure NameResolutionPass runs on AST before ASTToIRLoweringPass."
                        )
                    if defid is None:
                        continue
                    array_expr = access.array
                    shape_access = MemberAccessIR(
                        object=array_expr,
                        member='shape',
                        location=clause.location
                    )
                    dim_literal = LiteralIR(value=dim_idx, location=clause.location, type_info=infer_literal_type(dim_idx))
                    shape_dim = RectangularAccessIR(
                        array=shape_access,
                        indices=[dim_literal],
                        location=clause.location
                    )
                    start_lit = LiteralIR(value=0, location=clause.location, type_info=infer_literal_type(0))
                    range_ir = RangeIR(start=start_lit, end=shape_dim, location=clause.location, type_info=UNKNOWN)
                    clause.variable_ranges[defid] = range_ir
        if node.defid is not None:
            self.set_var(node.defid, len(clauses[0].indices) if clauses else 0)
        self.patterns_preprocessed += len(rest_pattern_indices)
    
    def _validate_rest_patterns(self, node: EinsteinIR, output_rest_names: set) -> Optional[str]:
        """
        Validate rest patterns follow the determination-first rule.
        
        Each rest pattern must appear alone in at least one array access
        before it can appear with other rest patterns. This ensures the compiler can
        determine how many dimensions each rest pattern spans.
        
        Returns: Error message if validation fails, None if valid
        """
        logger.debug(f"_validate_rest_patterns: output_rest_names={output_rest_names}")
        
        # Find all array accesses in body
        accesses = self._find_array_accesses(node.value)
        logger.debug(f"_validate_rest_patterns: found {len(accesses)} array accesses")
        if not accesses:
            # No array accesses - cannot validate
            return f"Rest pattern {list(output_rest_names)} in output but no array accesses in body"
        
        # Collect rest patterns that appear in body
        body_rest_names = set()
        for access in accesses:
            for idx in (access.indices or []):
                if isinstance(idx, IndexRestIR):
                    body_rest_names.add(idx.name)
        
        # Rule 1: Rest patterns in output must appear in body
        undetermined = output_rest_names - body_rest_names
        if undetermined:
            return f"Rest pattern {list(undetermined)} appears in output but not in body - cannot determine dimensions"
        
        # Rule 2: Determine which rest patterns appear alone (determination-first rule)
        # Also check for consistency - same rest pattern must span same number of dimensions
        determined_patterns = set()
        rest_pattern_dims = {}  # rest_name -> dimension_count
        
        for access in accesses:
            indices_flat = (access.indices or [])
            rest_in_access = []
            for idx in indices_flat:
                if isinstance(idx, IndexRestIR):
                    rest_in_access.append(idx.name)
            
            # If exactly one rest pattern in this access, it's determined
            if len(rest_in_access) == 1:
                rest_name = rest_in_access[0]
                determined_patterns.add(rest_name)
                
                # Check consistency - compute dimension count for this rest pattern.
                # Use same array_rank source as expansion (_get_rank_for_array); override only for
                # same-block previous Einstein output rank so we don't use wrong/incomplete type_info.
                logger.debug(f"  checking consistency for rest_name={rest_name}")
                array_rank = None
                arr_defid = getattr(access.array, 'defid', None) if isinstance(access.array, IdentifierIR) else None
                if arr_defid is not None:
                    rank_from_scope = self.get_var(arr_defid)
                    if isinstance(rank_from_scope, int):
                        array_rank = rank_from_scope
                        logger.debug(f"    array_rank={array_rank} from scope (defid={arr_defid})")
                if array_rank is None:
                    array_rank = self._get_rank_for_array(access.array)
                array_rank = array_rank or 0
                # Count explicit (non-rest) indices and how many dimensions this rest pattern spans
                explicit_count = sum(1 for idx in indices_flat if not isinstance(idx, IndexRestIR))
                # When we have rest + explicit indices, rest must span at least 1 dimension.
                # If array_rank inferred as 0 (e.g. param_type not yet set on specialized func),
                # use minimum: explicit_count + 1. If array_rank gives rest_dim_count <= 0
                # (e.g. param_type has wrong rank), clamp to 1 to avoid "0 vs 1" inconsistency.
                if array_rank == 0 and explicit_count > 0:
                    array_rank = explicit_count + 1
                    logger.debug(f"    array_rank inferred as 0 but rest+explicit indices present; using min rank={array_rank}")
                if array_rank == 0:
                    logger.debug(f"    array_rank could not be determined, skipping consistency check")
                    continue
                logger.debug(f"    array_rank={array_rank}")
                rest_dim_count = array_rank - explicit_count
                if rest_dim_count <= 0 and explicit_count > 0:
                    rest_dim_count = 1
                    logger.debug(f"    rest_dim_count was <=0 with explicit indices; clamped to 1")
                # Do not cap rest_dim_count by rest_positions (single ..pattern spans array_rank - explicit_count).
                
                logger.debug(f"    explicit_count={explicit_count}, rest_dim_count={rest_dim_count}")
                logger.debug(f"    rest_pattern_dims so far: {rest_pattern_dims}")
                
                if rest_name in rest_pattern_dims:
                    logger.debug(f"    rest_name '{rest_name}' already seen with {rest_pattern_dims[rest_name]} dims")
                    if rest_pattern_dims[rest_name] != rest_dim_count:
                        if arr_defid is not None and isinstance(self.get_var(arr_defid), int):
                            logger.debug(f"    previous Einstein output (defid={arr_defid}) has different rank; using canonical {rest_pattern_dims[rest_name]} (expansion will set correct rank)")
                            continue
                        error_msg = f"Rest pattern '..{rest_name}' has inconsistent dimensions: spans {rest_pattern_dims[rest_name]} dimensions in one array but {rest_dim_count} in another"
                        logger.debug(f"    INCONSISTENCY DETECTED: {error_msg}")
                        return error_msg
                else:
                    rest_pattern_dims[rest_name] = rest_dim_count
                    logger.debug(f"    set rest_pattern_dims['{rest_name}'] = {rest_dim_count}")
        
        # Rule 3: Check if multiple rest patterns appear together without all being determined
        undetermined_multi = output_rest_names - determined_patterns
        if undetermined_multi:
            for access in accesses:
                rest_in_access = []
                for idx in (access.indices or []):
                    if isinstance(idx, IndexRestIR):
                        rest_in_access.append(idx.name)
                
                # Multiple rest patterns in same access
                if len(rest_in_access) > 1:
                    overlapping = set(rest_in_access) & undetermined_multi
                    if overlapping:
                        return f"Rest patterns {sorted(rest_in_access)} appear together but not determined separately first. Each rest pattern must appear alone in at least one array access (determination-first rule)."
        
            return None
        
    def _access_uses_output_rest_patterns(self, access, output_rest_names: set) -> bool:
        """True if this access uses any of the rest pattern names that appear in the output (LHS). ."""
        for idx in (access.indices or []):
            if isinstance(idx, IndexRestIR) and idx.name in output_rest_names:
                return True
        return False

    def _infer_rank_from_body(self, node: BindingIR) -> Optional[int]:
        """
        Infer rank from body expression. prioritize arrays that use the same
        rest patterns as the output (LHS), then fall back to any array. Rank comes from
        that array's definition (type_info, param_type, variable), not from first access.
        """
        from ..shared.types import RectangularType, JaggedType
        
        accesses = self._find_array_accesses(node.value)
        logger.debug(f"_infer_rank_from_body: found {len(accesses)} array accesses")
        if not accesses:
            return None
        
        # Output rest pattern names (from LHS); never treat an index as a list
        output_rest_names = set()
        for idx in (node.indices or []):
            if isinstance(idx, IndexRestIR):
                output_rest_names.add(idx.name)
        
        def try_rank_for_access(acc) -> Optional[int]:
            return self._get_rank_for_array(acc.array)

        # First pass – prefer accesses that use the output's rest patterns
        if output_rest_names:
            for acc in accesses:
                if not self._access_uses_output_rest_patterns(acc, output_rest_names):
                    continue
                rank = try_rank_for_access(acc)
                if rank is not None:
                    logger.debug(f"_infer_rank_from_body: rank={rank} from prioritized access (uses output rest patterns) array={getattr(acc.array, 'name', None)}")
                    return rank
        # Second pass – any access
        for acc in accesses:
            rank = try_rank_for_access(acc)
            if rank is not None:
                logger.debug(f"_infer_rank_from_body: rank={rank} from access array={getattr(acc.array, 'name', None)}")
                return rank
        logger.debug(f"_infer_rank_from_body: cannot determine rank, deferring expansion")
        return None
    
    def _get_enclosing_function(self, node: IRNode) -> Optional[FunctionDefIR]:
        """Get the function that contains this node (for parameter lookup)"""
        # Simple approach: store current function during traversal
        # This is set in visit_function_def
        return getattr(self, '_current_function', None)
    
    def _get_rank_for_array(self, arr) -> Optional[int]:
        """
        Get rank of an array from scope stack (DefId-keyed).
        Priority: scope stack (Einstein output rank / var decl) → param type → type_info.
        """
        from ..shared.types import RectangularType, JaggedType
        from ..ir.nodes import FunctionDefIR, BindingIR
        if isinstance(arr, IdentifierIR):
            defid = getattr(arr, 'defid', None)
            if defid is not None:
                val = self.get_var(defid)
                if isinstance(val, int):
                    return val
                if isinstance(val, BindingIR):
                    r = self._try_infer_rank_from_var_def(arr.name, val)
                    if r is not None:
                        return r
            current_func = self._current_function
            if current_func and isinstance(current_func, FunctionDefIR):
                for param in current_func.parameters:
                    param_defid = getattr(param, 'defid', None)
                    if param_defid is not None and param_defid == defid and getattr(param, 'param_type', None):
                        pt = param.param_type
                        if isinstance(pt, RectangularType) and getattr(pt, 'shape', None):
                            return len(pt.shape)
                        n = 0
                        while isinstance(pt, (RectangularType, JaggedType)):
                            n += 1
                            pt = getattr(pt, 'element_type', None)
                        if n > 0:
                            return n
                        break
        if hasattr(arr, 'type_info') and arr.type_info:
            t = arr.type_info
            if isinstance(t, RectangularType) and getattr(t, 'shape', None):
                return len(t.shape)
            n = 0
            while isinstance(t, (RectangularType, JaggedType)):
                n += 1
                t = getattr(t, 'element_type', None)
            if n > 0:
                return n
        return None
    
    def _find_array_accesses(self, expr) -> List[RectangularAccessIR]:
        """Find all RectangularAccessIR nodes in expression"""
        collector = ArrayAccessCollector()
        accesses = expr.accept(collector)
        # Return the collected accesses from accept() return value, not collector.accesses
        return accesses if accesses else []
    
    def _try_infer_rank_from_var_def(self, var_name: str, var_def: BindingIR) -> Optional[int]:
        """
        Try to infer rank from a variable definition.
        
        Check array literal first (ground-truth structure), then type_info.
        Order: ArrayLiteral first, then parameter/type.
        """
        from ..shared.types import RectangularType
        from ..ir.nodes import ArrayLiteralIR
        
        # Case 1 (first): Infer from array literal value - primary for top-level declarations
        if hasattr(var_def, 'value') and isinstance(var_def.value, ArrayLiteralIR):
            rank = self._compute_array_literal_rank(var_def.value)
            if rank is not None and rank > 0:
                logger.debug(f"_try_infer_rank_from_var_def: inferred rank={rank} from array literal")
                return rank
        
        # Case 2: Check variable's type_info (if type inference has run)
        if hasattr(var_def, 'type_info') and var_def.type_info:
            var_type = var_def.type_info
            logger.debug(f"_try_infer_rank_from_var_def: variable {var_name} has type_info={var_type}")
            if isinstance(var_type, RectangularType) and hasattr(var_type, 'shape') and var_type.shape is not None:
                rank = len(var_type.shape)
                logger.debug(f"_try_infer_rank_from_var_def: inferred rank={rank} from type_info shape")
                return rank
            # Fallback for nested types
            from ..shared.types import JaggedType
            rank = 0
            current_type = var_type
            while isinstance(current_type, (RectangularType, JaggedType)):
                rank += 1
                current_type = current_type.element_type
            if rank > 0:
                logger.debug(f"_try_infer_rank_from_var_def: inferred rank={rank} from nested type")
                return rank
        
        return None
    
    def _compute_array_literal_rank(self, array_literal) -> Optional[int]:
        """
        Infer rank from array literal structure. standard algorithm
        _infer_rank_from_array_literal - count nested array levels with a while loop.

        Examples:
            [1, 2, 3] -> rank 1
            [[1, 2], [3, 4]] -> rank 2
            [[[1]]] -> rank 3
        """
        from ..ir.nodes import ArrayLiteralIR
        if not isinstance(array_literal, ArrayLiteralIR):
            return None
        if not hasattr(array_literal, 'elements') or not array_literal.elements:
            return None  # empty elements -> cannot infer rank
        rank = 1
        current = array_literal
        # Count nested array levels via first element chain
        while current.elements and isinstance(current.elements[0], ArrayLiteralIR):
            rank += 1
            current = current.elements[0]
        return rank
    
    # Visitor methods for traversing IR
    def visit_function_def(self, node: FunctionDefIR) -> None:
        """Visit function bodies and expand rest patterns in Einstein decls.
        Skip generic functions — rest patterns will be expanded on the specialized
        version where parameter types (and thus array ranks) are concrete."""
        from ..analysis.analysis_guard import is_generic_function
        if is_generic_function(node):
            logger.debug(f"Skipping rest pattern expansion for generic function '{node.name}'")
            return

        prev_function = self._current_function
        self._current_function = node

        with self.scope():
            if node.body:
                node.body.accept(self)

        self._current_function = prev_function
    
    def visit_variable_declaration(self, node: BindingIR) -> None:
        """Track variable declarations in scope stack by DefId for rank inference."""
        if node.defid is not None:
            self.set_var(node.defid, node)
            logger.debug(f"visit_variable_declaration: tracked '{node.pattern}' (defid={node.defid}) in scope")
        
        if hasattr(node, 'value') and node.value:
            node.value.accept(self)
    
    def visit_block_expression(self, node: BlockExpressionIR) -> None:
        """Visit block expressions with a new scope."""
        with self.scope():
            if hasattr(node, 'statements'):
                for stmt in node.statements:
                    if is_einstein_binding(stmt):
                        self.visit_einstein_declaration(stmt)
                    elif hasattr(stmt, 'accept'):
                        stmt.accept(self)
            if hasattr(node, 'final_expr') and node.final_expr:
                node.final_expr.accept(self)
    
    def visit_if_expression(self, node: IfExpressionIR) -> None:
        """Visit if expressions"""
        if node.condition:
            node.condition.accept(self)
        if node.then_expr:
            node.then_expr.accept(self)
        if node.else_expr:
            node.else_expr.accept(self)
    
    def visit_reduction_expression(self, node: ReductionExpressionIR) -> None:
        """Visit reduction expressions"""
        if node.body:
            node.body.accept(self)
        if node.where_clause:
            for constraint in node.where_clause.constraints:
                constraint.accept(self)
    
    # Default implementations for other nodes
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
            if idx is not None and hasattr(idx, "accept"):
                idx.accept(self)
    
    def visit_jagged_access(self, node) -> None:
        if node.array:
            node.array.accept(self)
        for idx in (getattr(node, "indices", None) or []):
            if idx is not None and hasattr(idx, "accept"):
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
        from ..ir.nodes import WhereExpressionIR, ReductionExpressionIR
        # Handle case where ReductionExpressionIR is incorrectly routed here
        if isinstance(node, ReductionExpressionIR):
            # Route to correct method
            self.visit_reduction_expression(node)
            return
        if not isinstance(node, WhereExpressionIR):
            return  # Skip if not a WhereExpressionIR
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
        for func in getattr(node, 'functions', None) or []:
            func.accept(self)
        for sub in getattr(node, 'submodules', None) or []:
            sub.accept(self)
    
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

    def visit_variable_declaration(self, node) -> None:
        """Visit variable declaration - recurse into value"""
        if hasattr(node, 'value') and node.value:
            node.value.accept(self)

class ArrayAccessCollector(IRVisitor[List[RectangularAccessIR]]):
    """Collect all RectangularAccessIR nodes from expression - visitor pattern"""
    
    def __init__(self):
        self.accesses: List[RectangularAccessIR] = []
    
    def visit_program(self, node: ProgramIR) -> List[RectangularAccessIR]:
        """Visit program - not used for array access collection"""
        return []
    
    def visit_rectangular_access(self, expr: RectangularAccessIR) -> List[RectangularAccessIR]:
        accesses = [expr]
        if expr.array:
            accesses.extend(expr.array.accept(self))
        for idx in (getattr(expr, "indices", None) or []):
            if idx is not None and hasattr(idx, "accept"):
                accesses.extend(idx.accept(self))
        return accesses
    
    def visit_binary_op(self, expr) -> List[RectangularAccessIR]:
        accesses = []
        if expr.left:
            accesses.extend(expr.left.accept(self))
        if expr.right:
            accesses.extend(expr.right.accept(self))
        return accesses
    
    def visit_function_call(self, expr) -> List[RectangularAccessIR]:
        accesses = []
        for arg in expr.arguments:
            accesses.extend(arg.accept(self))
        return accesses
    
    # Default: empty list
    def visit_literal(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_identifier(self, node) -> List[RectangularAccessIR]:
        return []

    def visit_index_var(self, node) -> List[RectangularAccessIR]:
        """Index slot (e.g. i in 0..N) - not an array access, return empty."""
        return []

    def visit_index_rest(self, node) -> List[RectangularAccessIR]:
        """Rest index slot (e.g. ..batch) - not an array access, return empty."""
        return []

    def visit_unary_op(self, node) -> List[RectangularAccessIR]:
        if node.operand:
            return node.operand.accept(self)
        return []
    
    def visit_jagged_access(self, node) -> List[RectangularAccessIR]:
        accesses = []
        if node.array:
            accesses.extend(node.array.accept(self))
        for idx in (getattr(node, "indices", None) or []):
            if idx is not None and hasattr(idx, "accept"):
                accesses.extend(idx.accept(self))
        return accesses
    
    def visit_block_expression(self, node) -> List[RectangularAccessIR]:
        accesses = []
        if hasattr(node, 'statements'):
            for stmt in node.statements:
                if hasattr(stmt, 'accept'):
                    accesses.extend(stmt.accept(self))
        if hasattr(node, 'final_expr') and node.final_expr:
            accesses.extend(node.final_expr.accept(self))
        return accesses
    
    def visit_if_expression(self, node) -> List[RectangularAccessIR]:
        accesses = []
        if node.condition:
            accesses.extend(node.condition.accept(self))
        if node.then_expr:
            accesses.extend(node.then_expr.accept(self))
        if node.else_expr:
            accesses.extend(node.else_expr.accept(self))
        return accesses
    
    def visit_lambda(self, node) -> List[RectangularAccessIR]:
        if node.body:
            return node.body.accept(self)
        return []
    
    def visit_array_literal(self, node) -> List[RectangularAccessIR]:
        accesses = []
        for elem in node.elements:
            accesses.extend(elem.accept(self))
        return accesses
    
    def visit_tuple_expression(self, node) -> List[RectangularAccessIR]:
        accesses = []
        for elem in node.elements:
            accesses.extend(elem.accept(self))
        return accesses
    
    def visit_match_expression(self, node) -> List[RectangularAccessIR]:
        accesses = []
        if node.scrutinee:
            accesses.extend(node.scrutinee.accept(self))
        for arm in node.arms:
            if arm.body:
                accesses.extend(arm.body.accept(self))
        return accesses
    
    def visit_where_expression(self, node) -> List[RectangularAccessIR]:
        accesses = []
        if node.expr:
            accesses.extend(node.expr.accept(self))
        for constraint in node.constraints:
            accesses.extend(constraint.accept(self))
        return accesses
    
    # Add stub implementations for all other IR node types
    def visit_range(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_array_comprehension(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_tuple_access(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_interpolated_string(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_cast_expression(self, node) -> List[RectangularAccessIR]:
        if node.expr:
            return node.expr.accept(self)
        return []
    
    def visit_member_access(self, node) -> List[RectangularAccessIR]:
        if node.object:
            return node.object.accept(self)
        return []
    
    def visit_try_expression(self, node) -> List[RectangularAccessIR]:
        if node.expr:
            return node.expr.accept(self)
        return []
    
    def visit_reduction_expression(self, node) -> List[RectangularAccessIR]:
        if node.body:
            return node.body.accept(self)
        return []
    
    def visit_arrow_expression(self, node) -> List[RectangularAccessIR]:
        accesses = []
        for comp in node.components:
            accesses.extend(comp.accept(self))
        return accesses
    
    def visit_pipeline_expression(self, node) -> List[RectangularAccessIR]:
        accesses = []
        if node.left:
            accesses.extend(node.left.accept(self))
        if node.right:
            accesses.extend(node.right.accept(self))
        return accesses
    
    def visit_builtin_call(self, node) -> List[RectangularAccessIR]:
        accesses = []
        for arg in node.args:
            accesses.extend(arg.accept(self))
        return accesses
    
    def visit_einstein_declaration(self, node) -> List[RectangularAccessIR]:
        accesses = []
        for clause in (getattr(node, 'clauses', None) or []):
            if clause.value:
                accesses.extend(clause.value.accept(self))
        return accesses
    
    def visit_constant_def(self, node) -> List[RectangularAccessIR]:
        if node.value:
            return node.value.accept(self)
        return []
    
    def visit_module(self, node) -> List[RectangularAccessIR]:
        return []
    
    # Pattern visitors (no-op)
    def visit_literal_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_identifier_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_wildcard_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_tuple_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_array_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_rest_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_guard_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_function_def(self, node) -> List[RectangularAccessIR]:
        if node.body:
            return node.body.accept(self)
        return []

    def visit_variable_declaration(self, node) -> List[RectangularAccessIR]:
        """Visit variable declaration - recurse into value"""
        if hasattr(node, 'value') and node.value:
            return node.value.accept(self)
        return []

class RestPatternBodyTransformer(IRVisitor[ExpressionIR]):
    """
    Transform body: replace rest patterns with expanded indices. DefIds keyed by IndexRestIR.defid only (per clause).
    """
    def __init__(self, rest_dim_mapping: Dict[str, List[int]], rest_defid_to_expanded: Dict[Any, List[Any]], tcx: 'TyCtxt'):
        self.rest_dim_mapping = rest_dim_mapping
        self.rest_defid_to_expanded = rest_defid_to_expanded
        self.tcx = tcx

    def visit_rectangular_access(self, node: RectangularAccessIR) -> ExpressionIR:
        new_array = node.array.accept(self) if node.array else node.array
        new_indices = []
        for idx_expr in (node.indices or []):
            if idx_expr is None:
                raise ValueError("IR index slot is None")
            if isinstance(idx_expr, IndexRestIR):
                rest_defid = getattr(idx_expr, "defid", None)
                if rest_defid is None:
                    raise ValueError(
                        f"IndexRestIR (..{idx_expr.name}) in body must have defid. "
                        "Ensure NameResolutionPass runs on AST before ASTToIRLoweringPass so defid is set and copied to IR."
                    )
                defids_list = self.rest_defid_to_expanded.get(rest_defid, [])
                rest_name = idx_expr.name
                dim_indices = self.rest_dim_mapping.get(rest_name, [])
                if dim_indices and len(defids_list) >= len(dim_indices):
                    for k, dim_idx in enumerate(dim_indices):
                        defid = defids_list[k]
                        expanded_var = f"{rest_name}.{dim_idx}"
                        new_idx = IndexVarIR(
                            name=expanded_var,
                            location=idx_expr.location,
                            defid=defid,
                            range_ir=None
                        )
                        if hasattr(idx_expr, 'type_info'):
                            new_idx.type_info = idx_expr.type_info
                        if hasattr(idx_expr, 'shape_info'):
                            new_idx.shape_info = idx_expr.shape_info
                        new_indices.append(new_idx)
                else:
                    new_indices.append(idx_expr)
            else:
                defid = getattr(idx_expr, "defid", None)
                if isinstance(idx_expr, IdentifierIR):
                    new_idx = IndexVarIR(
                        name=idx_expr.name,
                        location=idx_expr.location,
                        defid=defid,
                        range_ir=None
                    )
                    if hasattr(idx_expr, 'type_info'):
                        new_idx.type_info = idx_expr.type_info
                    if hasattr(idx_expr, 'shape_info'):
                        new_idx.shape_info = idx_expr.shape_info
                    new_indices.append(new_idx)
                elif isinstance(idx_expr, IndexVarIR):
                    new_idx = IndexVarIR(
                        name=idx_expr.name,
                        location=idx_expr.location,
                        defid=defid,
                        range_ir=getattr(idx_expr, "range_ir", None),
                    )
                    if hasattr(idx_expr, 'type_info'):
                        new_idx.type_info = idx_expr.type_info
                    if hasattr(idx_expr, 'shape_info'):
                        new_idx.shape_info = idx_expr.shape_info
                    new_indices.append(new_idx)
                else:
                    new_indices.append(idx_expr)

        new_node = RectangularAccessIR(
            array=new_array,
            indices=new_indices,
            location=node.location
        )
        if hasattr(node, 'type_info'):
            new_node.type_info = node.type_info
        if hasattr(node, 'shape_info'):
            new_node.shape_info = node.shape_info
        return new_node
    
    # Identity transformations for most nodes
    def visit_identifier(self, node: IdentifierIR) -> ExpressionIR:
        return node
    
    def visit_literal(self, node: LiteralIR) -> ExpressionIR:
        return node
    
    def visit_binary_op(self, node: BinaryOpIR) -> ExpressionIR:
        new_left = node.left.accept(self) if node.left else None
        new_right = node.right.accept(self) if node.right else None
        new_node = BinaryOpIR(
            operator=node.operator,
            left=new_left,
            right=new_right,
            location=node.location
        )
        if hasattr(node, 'type_info'):
            new_node.type_info = node.type_info
        return new_node
    
    def visit_unary_op(self, node: UnaryOpIR) -> ExpressionIR:
        new_operand = node.operand.accept(self) if node.operand else None
        new_node = UnaryOpIR(
            operator=node.operator,
            operand=new_operand,
            location=node.location
        )
        if hasattr(node, 'type_info'):
            new_node.type_info = node.type_info
        return new_node
    
    # Stub implementations for all other required methods
    def visit_program(self, node) -> ExpressionIR:
        raise NotImplementedError("Program transformation not supported")
    
    def visit_function_def(self, node) -> ExpressionIR:
        return node
    
    def visit_function_call(self, node) -> ExpressionIR:
        # Transform arguments (they may contain array accesses with rest patterns)
        from ..ir.nodes import FunctionCallIR
        new_arguments = [arg.accept(self) if hasattr(arg, 'accept') else arg for arg in node.arguments]
        new_node = FunctionCallIR(
            callee_expr=node.callee_expr,
            location=node.location,
            arguments=new_arguments,
            module_path=node.module_path if hasattr(node, 'module_path') else None
        )
        if hasattr(node, 'type_info'):
            new_node.type_info = node.type_info
        return new_node
    
    def visit_builtin_call(self, node) -> ExpressionIR:
        from ..ir.nodes import BuiltinCallIR
        from ..shared.defid import fixed_builtin_defid
        new_args = [arg.accept(self) if hasattr(arg, 'accept') else arg for arg in node.args]
        defid = getattr(node, 'defid', None) or fixed_builtin_defid(node.builtin_name)
        new_node = BuiltinCallIR(
            builtin_name=node.builtin_name,
            args=new_args,
            location=node.location,
            defid=defid,
        )
        if hasattr(node, 'type_info'):
            new_node.type_info = node.type_info
        return new_node
    
    def visit_reduction_expression(self, node) -> ExpressionIR:
        new_body = node.body.accept(self) if node.body else None
        # : preserve explicit reduction ranges (e.g. m in 0..kernel_w) so
        # einstein_lowering uses them; otherwise we lose them and infer wrong bound (shape(X)[2]).
        loop_var_ranges = getattr(node, 'loop_var_ranges', None)
        new_node = ReductionExpressionIR(
            operation=node.operation,
            loop_vars=node.loop_vars,
            body=new_body,
            where_clause=node.where_clause,
            loop_var_ranges=loop_var_ranges,
            location=node.location
        )
        if hasattr(node, 'type_info'):
            new_node.type_info = node.type_info
        return new_node
    
    def visit_jagged_access(self, node) -> ExpressionIR:
        return node
    
    def visit_tuple_access(self, node) -> ExpressionIR:
        return node
    
    def visit_member_access(self, node) -> ExpressionIR:
        return node
    
    def visit_cast_expression(self, node) -> ExpressionIR:
        from ..ir.nodes import CastExpressionIR
        new_expr = node.expr.accept(self) if node.expr else node.expr
        if new_expr is node.expr:
            return node
        return CastExpressionIR(
            expr=new_expr,
            target_type=node.target_type,
            location=node.location,
            type_info=getattr(node, 'type_info', None),
            shape_info=getattr(node, 'shape_info', None),
        )
    
    def visit_array_literal(self, node) -> ExpressionIR:
        return node
    
    def visit_range(self, node) -> ExpressionIR:
        return node
    
    def visit_block_expression(self, node: BlockExpressionIR) -> ExpressionIR:
        new_statements = [s.accept(self) if hasattr(s, 'accept') else s for s in (getattr(node, 'statements', None) or [])]
        new_final = node.final_expr.accept(self) if getattr(node, 'final_expr', None) else None
        return BlockExpressionIR(
            new_statements,
            node.location,
            final_expr=new_final,
            type_info=getattr(node, 'type_info', None),
            shape_info=getattr(node, 'shape_info', None),
        )
    
    def visit_if_expression(self, node) -> ExpressionIR:
        """Transform if/else branches - rest patterns can appear in condition and branches"""
        from ..ir.nodes import IfExpressionIR
        new_cond = node.condition.accept(self) if node.condition else node.condition
        new_then = node.then_expr.accept(self) if node.then_expr else node.then_expr
        new_else = node.else_expr.accept(self) if node.else_expr else node.else_expr
        return IfExpressionIR(
            condition=new_cond,
            then_expr=new_then,
            else_expr=new_else,
            location=node.location,
            type_info=getattr(node, 'type_info', None),
            shape_info=getattr(node, 'shape_info', None),
        )
    
    def visit_tuple_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_lambda(self, node) -> ExpressionIR:
        return node
    
    def visit_einstein_declaration(self, node) -> ExpressionIR:
        return node
    
    def visit_variable_declaration(self, node: BindingIR) -> ExpressionIR:
        new_value = node.expr.accept(self) if getattr(node, 'expr', None) else None
        return BindingIR(
            name=getattr(node, 'name', None) or getattr(node, 'pattern', ''),
            expr=new_value,
            type_info=getattr(node, 'type_info', None),
            location=getattr(node, 'location', None),
            defid=getattr(node, 'defid', None),
        )
    
    def visit_match_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_where_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_arrow_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_pipeline_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_array_comprehension(self, node) -> ExpressionIR:
        return node
    
    def visit_try_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_literal_pattern(self, node) -> ExpressionIR:
        return node
    
    def visit_identifier_pattern(self, node) -> ExpressionIR:
        return node
    
    def visit_wildcard_pattern(self, node) -> ExpressionIR:
        return node
    
    def visit_tuple_pattern(self, node) -> ExpressionIR:
        return node
    
    def visit_array_pattern(self, node) -> ExpressionIR:
        return node
    
    def visit_rest_pattern(self, node) -> ExpressionIR:
        return node
    
    def visit_guard_pattern(self, node) -> ExpressionIR:
        return node
    
    def visit_constant_def(self, node) -> ExpressionIR:
        return node
    
    def visit_module(self, node) -> ExpressionIR:
        return node
    
    def visit_interpolated_string(self, node) -> ExpressionIR:
        return node

