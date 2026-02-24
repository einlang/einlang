"""
Type Inference Pass

Rust Pattern: rustc_typeck
Reference: TYPE_SYSTEM_DESIGN.md
"""

import logging
import json
import os
from ..passes.base import BasePass, TyCtxt
from ..passes.range_analysis import RangeAnalysisPass
from ..ir.nodes import (
    ProgramIR, ExpressionIR, FunctionDefIR, ConstantDefIR,
    BlockExpressionIR, IRNode, IRVisitor, FunctionCallIR, LiteralIR,
    IdentifierIR, BinaryOpIR, UnaryOpIR, RectangularAccessIR, JaggedAccessIR,
    ArrayLiteralIR, TupleExpressionIR, ReductionExpressionIR, IfExpressionIR,
    MatchExpressionIR, CastExpressionIR, TryExpressionIR, LambdaIR, RangeIR,
    ArrayComprehensionIR, InterpolatedStringIR, TupleAccessIR, BuiltinCallIR,
    ArrowExpressionIR, PipelineExpressionIR, FunctionRefIR, MemberAccessIR,
    LiteralPatternIR, IdentifierPatternIR, WildcardPatternIR, TuplePatternIR,
    IndexVarIR, IndexRestIR,
    ArrayPatternIR, RestPatternIR, GuardPatternIR, MatchArmIR,
    EinsteinIR, EinsteinDeclarationIR,
    VariableDeclarationIR,
)
from ..shared.types import Type, FunctionType, PrimitiveType, RectangularType, UNKNOWN, I32, I64, F32, F64, BOOL, STR, RANGE, UNIT, infer_literal_type, TypeVisitor, Optional as TypeOptional, TypeKind, UnaryOp, BinaryOp
from ..shared.defid import DefId, assert_defid
from ..utils.config import DEFAULT_INT_TYPE, DEFAULT_FLOAT_TYPE
from typing import Optional, Tuple, List, Dict, Any, Set
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger("einlang.passes.type_inference")


def _builtin_return_first_arg(arg_types: List[Type]) -> Type:
    if len(arg_types) >= 1 and arg_types[0] not in (None, UNKNOWN):
        return arg_types[0]
    return UNKNOWN


def _builtin_return_sum(arg_types: List[Type]) -> Type:
    if len(arg_types) >= 1 and isinstance(arg_types[0], RectangularType):
        return arg_types[0].element_type
    return UNKNOWN


def _builtin_return_max_min(arg_types: List[Type]) -> Type:
    if len(arg_types) == 1 and isinstance(arg_types[0], RectangularType):
        return arg_types[0].element_type
    if len(arg_types) >= 1 and all(t in (I32, I64, F32, F64) for t in arg_types):
        if any(t in (F32, F64) for t in arg_types):
            return F64 if any(t == F64 for t in arg_types) else F32
        return I64 if any(t == I64 for t in arg_types) else I32
    return UNKNOWN


SHAPE_RETURN_TYPE = RectangularType(I32, None, is_dynamic_rank=True)

_BUILTIN_RETURN_TABLE = {
    "len": I32,
    "print": UNIT,
    "assert": UNIT,
    "typeof": STR,
    "shape": SHAPE_RETURN_TYPE,
    "array_append": _builtin_return_first_arg,
    "sum": _builtin_return_sum,
    "max": _builtin_return_max_min,
    "min": _builtin_return_max_min,
}


@dataclass
class FunctionSignature:
    """Function signature (ported from precision_engine.py)"""
    name: str
    parameter_types: Tuple[Type, ...]
    parameter_names: List[str]
    return_type: Optional[Type] = None

class TypeInferencePass(BasePass):
    """
    Type inference pass (Rust naming: rustc_typeck).
    
    Rust Pattern: rustc_typeck::check::typeck
    
    Implementation Alignment: Follows Rust's type checking:
    - Type inference on IR (not AST)
    - Type checking on IR
    - All types stored in IR nodes
    - Function types (arrow types) tracked
    
    Reference: `rustc_typeck::check::typeck` for type checking
    """
    requires = [RangeAnalysisPass]  # Depends on range analysis (needs type info from previous passes)
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """
        Infer types for all expressions.
        
        Rust Pattern: rustc_typeck::check::typeck()
        """
        inferencer = TypeInferencer(tcx)
        
        # ALIGNED: Store mono_service in TyCtxt for other passes to access
        tcx.monomorphization_service = inferencer.mono_service
        
        # Use visitor pattern: ir.accept(visitor) infers types in place
        ir.accept(inferencer)
        
        # Incremental mono service: each specialization is fully analyzed inside
        # MonomorphizationService._run_passes (rest_pattern, range, shape, type, einstein_lowering).
        # Loop only adds pending to IR and re-runs full program type inference to discover transitive specializations.
        while True:
            pending_funcs = inferencer.mono_service.get_pending_specialized_functions()
            if not pending_funcs:
                break
            for f in pending_funcs:
                if f not in ir.functions:
                    ir.functions.append(f)
                    if f.defid:
                        tcx.function_ir_map[f.defid] = f
                    logger.debug(f"Added specialized function {f.name} with DefId {f.defid} to program IR and function_ir_map")
            inferencer.mono_service.clear_pending_specialized_functions()
            ir.accept(inferencer)
        
        specialized_funcs = getattr(tcx, 'specialized_functions', [])
        if specialized_funcs:
            inferencer.mono_service.rewrite_calls_in_specialized_bodies()
        inferencer.mono_service.rewrite_calls_in_statements(ir.statements)
        inferencer.mono_service.unify_local_var_defids_in_program(ir)
        return ir


class TypeInferencer(IRVisitor[Type]):
    """
    Type inferencer (Rust naming: rustc_typeck).
    
    Rust Pattern: rustc_typeck::check
    
    Extends ScopedASTVisitor for scope management
    - Binds function parameters in scope before visiting body
    - Uses get_var/set_var for variable lookup
    """
    
    def __init__(self, tcx: TyCtxt):
        self.tcx = tcx
        # Use shared mono service from tcx (e.g. from _run_passes) so _monomorphizing is visible
        from ..analysis.monomorphization_service import MonomorphizationService
        self.mono_service = getattr(tcx, 'monomorphization_service', None) or MonomorphizationService(tcx)
        # Store program IR for function lookup
        self._current_program: Optional[ProgramIR] = None
        # Scope stack keyed by DefId (after name resolution we use defid, not names)
        self._scope_stack: List[Dict[DefId, Type]] = [{}]
        # Track call depth for recursion detection
        self._call_depth = 0
        # Track current function (for Einstein element_type fallback from params)
        self._current_function: Optional[FunctionDefIR] = None

    def visit_program(self, node: ProgramIR) -> Type:
        """Visit program and infer types in all statements and functions"""
        
        # Store program IR for function lookup
        self._current_program = node
        
        # Store program IR and function map in TyCtxt for monomorphization access
        self.tcx.program_ir = node
        # When this is the mini-program (single function, no top-level statements),
        # merge into existing function_ir_map so inner calls (e.g. max_pool1d) can be specialized.
        # Otherwise build DefId → FunctionDefIR from current program.
        is_mini_program = (len(node.functions) == 1 and len(node.statements) == 0 and
                          hasattr(self.tcx, 'function_ir_map') and self.tcx.function_ir_map is not None)
        if is_mini_program:
            for func in node.functions:
                if func.defid:
                    self.tcx.function_ir_map[func.defid] = func
        else:
            func_map = {}
            existing_fim = getattr(self.tcx, 'function_ir_map', None) or {}
            for defid, f in existing_fim.items():
                if defid and f is not None:
                    func_map[defid] = f
            for func in node.functions:
                if func.defid and func.defid not in existing_fim:
                    func_map[func.defid] = func
            for mod in getattr(node, 'modules', None) or []:
                for mfunc in self._collect_module_functions(mod):
                    if mfunc.defid and mfunc.defid not in existing_fim:
                        func_map[mfunc.defid] = mfunc
            self.tcx.function_ir_map = func_map

        # Visit all statements
        for stmt_idx, stmt in enumerate(node.statements):
            # Infer type of the statement/expression
            # Rust Pattern: visit_variable_declaration already binds variables to scope
            inferred_type = stmt.accept(self)
        
        # Visit all functions (bodies can look up callee via function_ir_map)
        for func in node.functions:
            func.accept(self)
        
        # Visit all constants
        for const in node.constants:
            const.accept(self)
        
        # Do not add/clear pending here; run() loop adds them and re-visits specialized bodies
        # so inner calls (e.g. row_values inside topk_2d) get inferred with concrete types.

        return UNKNOWN
    
    def _signature_from_function_ir(self, func: FunctionDefIR) -> FunctionSignature:
        """Build FunctionSignature from FunctionDefIR (single source of truth: IR)."""
        param_types: List[Type] = []
        param_names: List[str] = []
        if hasattr(func, 'parameters'):
            for param in func.parameters:
                param_names.append(getattr(param, 'name', f"param{len(param_names)}"))
                param_types.append(getattr(param, 'param_type', None) or UNKNOWN)
        return_type = getattr(func, 'return_type', None) or None
        return FunctionSignature(
            name=func.name,
            parameter_types=tuple(param_types),
            parameter_names=param_names,
            return_type=return_type
        )

    def _get_function(self, defid: Optional[DefId]) -> Optional[FunctionSignature]:
        """Look up function signature by DefId from function_ir_map (IR is source of truth)."""
        if defid is None:
            return None
        func_ir = getattr(self.tcx, 'function_ir_map', None)
        if func_ir is None:
            return None
        func = func_ir.get(defid)
        if func is None:
            return None
        return self._signature_from_function_ir(func)
    
    def visit_literal(self, expr: LiteralIR) -> Type:
        """Infer type of literal. Integer default i32, float default f32 (Rust-like)."""
        inferred_type = infer_literal_type(expr.value)
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        
        
        
        return inferred_type
    
    def visit_variable_declaration(self, stmt) -> Type:
        """
        Infer type of variable declaration and bind it in scope.
        
        Rust Pattern: Statements don't have types - only expressions do.
        We infer the type of the value expression and bind the variable name to that type.
        
        Rust Alignment: Check type annotation compatibility with literal coercion.
        """
        # Infer type from value expression
        value_type = stmt.value.accept(self)
        
        # Check type annotation compatibility (Rust pattern: strict checking with literal coercion)
        if stmt.type_annotation and stmt.type_annotation != UNKNOWN:
            expected_type = stmt.type_annotation
            
            # Rust special case: numeric literals can be coerced to any numeric type
            # In Rust: `let x: i64 = 42;` works (literal inference)
            # But: `let x: i32 = 42; let y: i64 = x;` fails (no variable coercion)
            is_literal_coercion = self._is_literal_coercion(stmt.value, value_type, expected_type)
            
            if not is_literal_coercion and not self._is_assignment_compatible(value_type, expected_type):
                # Create error message matching Rust style
                var_name = stmt.pattern if hasattr(stmt, 'pattern') else stmt.name
                error_msg = f"mismatched types: expected `{expected_type}`, found `{value_type}`"
                self.tcx.reporter.report_error(
                    message=error_msg,
                    location=stmt.value.location if hasattr(stmt, 'value') and hasattr(stmt.value, 'location') else None,
                    code="type_annotation_mismatch"
                )
            # Use annotated type (it's more specific, or coerced from literal)
            value_type = expected_type
        
        # Bind variable by DefId (VariableDeclarationIR stores defid on _binding)
        stmt_defid = getattr(stmt, 'defid', None) or (getattr(getattr(stmt, '_binding', None), 'defid', None))
        if stmt_defid is None:
            raise RuntimeError(
                f"Variable declaration has no DefId (pattern={getattr(stmt, 'pattern', getattr(stmt, 'name', '?'))}). "
                "Ensure NameResolutionPass runs before TypeInferencePass."
            )
        self._set_var(stmt_defid, value_type)
        # Rust Pattern: Don't set type_info on statements - only expressions have types
        
        return value_type
    
    def _is_literal_coercion(self, expr, value_type: Type, expected_type: Type) -> bool:
        """
        Check if this is a Rust-style literal coercion.
        
        Rust allows: let x: i64 = 42; (literal infers to i64)
        Rust rejects: let x: i32 = 42; let y: i64 = x; (no variable coercion)
        
        Returns True if expr is a numeric literal and both types are numeric.
        """
        from ..ir.nodes import LiteralIR
        
        # Must be a literal expression
        if not isinstance(expr, LiteralIR):
            return False
        
        # Must be numeric literal (int or float)
        if not isinstance(expr.value, (int, float)):
            return False
        
        # Both types must be numeric primitives
        numeric_types = {I32, I64, F32, F64}
        if value_type not in numeric_types or expected_type not in numeric_types:
            return False
        
        # Integer literals can coerce to any integer type
        if isinstance(expr.value, int):
            int_types = {I32, I64}
            return value_type in int_types and expected_type in int_types
        
        # Float literals can coerce to any float type
        if isinstance(expr.value, float):
            float_types = {F32, F64}
            return value_type in float_types and expected_type in float_types
        
        return False
    
    def visit_identifier(self, expr) -> Type:
        """Infer type by DefId. Name resolution must attach defid and bind in scope; type pass does not resolve names."""
        name = getattr(expr, 'name', '') or ''
        if getattr(expr, "defid", None) is None:
            raise RuntimeError(
                f"Identifier '{name}' has no DefId. Name resolution pass must attach DefIds to all identifiers."
            )
        inferred_type = self._get_var(expr.defid)
        if inferred_type is None:
            # Bind-on-first-use for iteration/reduction indices (e.g. in copied specialized IR)
            self._set_var(expr.defid, I32)
            inferred_type = I32
        expr.type_info = inferred_type
        return inferred_type

    def visit_index_var(self, node: IndexVarIR) -> Type:
        """Index variable slot is I32; delegate to range_ir if present."""
        if getattr(node, "range_ir", None) is not None:
            node.range_ir.accept(self)
        node.type_info = I32
        return I32

    def visit_index_rest(self, node: IndexRestIR) -> Type:
        """Rest index slot is I32."""
        node.type_info = I32
        return I32

    def _get_var(self, key: DefId) -> Optional[Type]:
        """Get variable from scope chain by DefId (inner → outer). key must not be None."""
        if key is None:
            raise RuntimeError("_get_var requires DefId; got None. Fail fast.")
        for scope in reversed(self._scope_stack):
            if key in scope:
                return scope[key]
        return None

    def _set_var(self, key: DefId, value: Type) -> None:
        """Set variable in current scope by DefId. key must not be None."""
        if key is None:
            raise RuntimeError("_set_var requires DefId; got None. Fail fast.")
        self._scope_stack[-1][key] = value
    
    @contextmanager
    def _scope(self):
        """Context manager for entering/exiting a scope (RAII)"""
        self._scope_stack.append({})
        try:
            yield
        finally:
            if len(self._scope_stack) > 1:
                self._scope_stack.pop()

    @contextmanager
    def _function_scope(self, node):
        """Scope for analyzing a function; restores _current_function on exit."""
        prev = self._current_function
        self._current_function = node
        try:
            yield
        finally:
            self._current_function = prev
    
    def _extract_base_element_type(self, type_obj: Type) -> Type:
        """
        Extract base element type from rectangular types.
        Unwrap RectangularType to get the element type.
        
        Examples:
            [f32; ?, ?] → f32  (2D array)
            [f32; ?] → f32     (1D array)
            f32 → f32          (scalar)
            UNKNOWN → None     (not inferrable)
        
        Note: RectangularType stores element_type directly,
        not nested RectangularTypes.
        """
        # Return None for UNKNOWN (preserves structure, indicates missing type)
        if type_obj == UNKNOWN or type_obj is None:
            return None
        
        # Unwrap rectangular types to get element
        if isinstance(type_obj, RectangularType):
            return type_obj.element_type  # Direct access, no recursion needed
        
        # Base case: primitive or other type
        return type_obj
    
    def _build_rectangular_array_type(self, element_type: Type, num_dimensions: int) -> Type:
        """
        Build a rectangular array type with the given element type and rank.
        Create RectangularType with dynamic dimensions.
        
        Examples:
            element_type=f32, num_dimensions=2 → [f32; ?, ?]
            element_type=None, num_dimensions=2 → [None; ?, ?]  (rank known, type unknown)
            element_type=f32, num_dimensions=1 → [f32; ?]
            element_type=f32, num_dimensions=0 → f32 (scalar, no wrapping)
            element_type=None, num_dimensions=0 → None (scalar, type unknown)
        
        Use None for element_type when unknown (preserves structure)
        """
        if num_dimensions == 0:
            # Scalar, no array wrapping
            return element_type  # Can be None if unknown
        
        # Allow None as element_type to preserve array structure (rank)
        # even when element type is not yet inferred
        # Create RectangularType with dynamic shape (represented as None in tuple)
        # shape tuple has length num_dimensions, each element is None for dynamic
        shape = tuple([None] * num_dimensions)
        
        # If element_type is None, use UNKNOWN as placeholder but preserve structure
        # This allows further analysis passes to fill in the type later
        actual_element_type = element_type if element_type is not None else UNKNOWN
        
        return RectangularType(element_type=actual_element_type, shape=shape)
    
    def visit_binary_op(self, expr) -> Type:
        """Infer type of binary operation with promotion rules"""
        left_type = expr.left.accept(self)
        right_type = expr.right.accept(self)
        
        # Check if this is a comparison operator
        from ..shared.types import BOOL, I32, I64, F32, F64
        comparison_ops_enum = {BinaryOp.LT, BinaryOp.GT, BinaryOp.LE, BinaryOp.GE, BinaryOp.EQ, BinaryOp.NE}
        
        if hasattr(expr, 'operator') and expr.operator in comparison_ops_enum:
            # Comparison operators return bool
            # RUST ALIGNMENT: Comparisons require same types (or widening within category)
            int_types = {I32, I64}
            float_types = {F32, F64}
            
            # Allow comparisons within same category (with widening)
            if (left_type in int_types and right_type in int_types):
                # i32 < i64 is allowed (widen to i64 for comparison)
                expr.type_info = BOOL
                return BOOL
            
            if (left_type in float_types and right_type in float_types):
                # f32 < f64 is allowed (widen to f64 for comparison)
                expr.type_info = BOOL
                return BOOL
            
            # Same types (including bool, etc.)
            if left_type == right_type:
                expr.type_info = BOOL
                return BOOL
            
            if (left_type in int_types and right_type in float_types) or \
               (left_type in float_types and right_type in int_types):
                from ..ir.nodes import LiteralIR
                left_is_int_lit = isinstance(expr.left, LiteralIR) and left_type in int_types
                right_is_int_lit = isinstance(expr.right, LiteralIR) and right_type in int_types
                if not (left_is_int_lit or right_is_int_lit):
                    self._promote_types(left_type, right_type, location=expr.location)
            
            expr.type_info = BOOL
            return BOOL
        
        # Type promotion rules for arithmetic operations
        # Preserve precision, promote when needed
        # RUST PATTERN: Integer literals can be promoted to float in arithmetic
        from ..ir.nodes import LiteralIR
        
        # Check if we're mixing int and float with a literal
        if (left_type in {I32, I64} and right_type in {F32, F64}) or \
           (left_type in {F32, F64} and right_type in {I32, I64}):
            left_is_int_literal = isinstance(expr.left, LiteralIR) and left_type in {I32, I64}
            right_is_int_literal = isinstance(expr.right, LiteralIR) and right_type in {I32, I64}
            
            if left_is_int_literal or right_is_int_literal:
                # Promote the literal to the float type
                target_float_type = right_type if left_is_int_literal else left_type
                
                if left_is_int_literal:
                    expr.left.type_info = target_float_type
                if right_is_int_literal:
                    expr.right.type_info = target_float_type
                
                expr.type_info = target_float_type
                return target_float_type
        
        # POW special case: float ** int → float (Rust's powi)
        if hasattr(expr, 'operator') and expr.operator == BinaryOp.POW:
            if left_type in {F32, F64} and right_type in {I32, I64}:
                expr.type_info = left_type
                return left_type
        
        inferred_type = self._promote_types(left_type, right_type, location=expr.location)
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def _promote_types(self, left: Type, right: Type, location=None) -> Type:
        """
        Type promotion rules - RUST PATTERN: No implicit cross-category conversion.
        
        Rules (using Type objects directly - NO STRING CONVERSIONS):
        - If either is UNKNOWN, return the other (or UNKNOWN if both are)
        - Same types -> no promotion
        - i32 + i64 -> i64 (widen within same category)
        - f32 + f64 -> f64 (widen within same category)
        - i32 + f32 -> ERROR (cross-category mixing not allowed - explicit cast required)
        
        Fail fast on incompatible types (like precision_engine.py)
        """
        from ..shared.types import I32, I64, F32, F64
        
        # Handle UNKNOWN
        if left == UNKNOWN:
            return right
        if right == UNKNOWN:
            return left
        
        # Same types - no promotion needed
        if left == right:
            return left
        
        # RUST PATTERN: Widening within same category only
        # Float widening (f32 <-> f64)
        if (left == F32 and right == F64) or (left == F64 and right == F32):
            return F64
        
        # Integer widening (i32 <-> i64)
        if (left == I32 and right == I64) or (left == I64 and right == I32):
            return I64
        
        # RUST PATTERN: Cross-category is ERROR
        # Check if mixing int and float
        int_types = {I32, I64}
        float_types = {F32, F64}
        
        if (left in int_types and right in float_types) or (left in float_types and right in int_types):
            # Determine which is which for correct error message
            if left in int_types:
                int_type, float_type = left, right
            else:
                int_type, float_type = right, left
            
            # Report type mismatch error with correct labels (use .name for clean output)
            int_name = int_type.name if hasattr(int_type, 'name') else str(int_type)
            float_name = float_type.name if hasattr(float_type, 'name') else str(float_type)
            self.tcx.reporter.report_error(
                message=f"Type mismatch: cannot mix integer type `{int_name}` and float type `{float_name}`. Use explicit cast.",
                location=location,
                code="E0308"  # Rust's type mismatch error code
            )
            return UNKNOWN
        
        # Default: incompatible types
        return left
    
    def _is_assignment_compatible(self, value_type: Type, expected_type: Type) -> bool:
        """
        Check if assignment is compatible (TRUE Rust semantics: NO implicit conversions).
        
        Rust Rules:
        - UNKNOWN: Always compatible (gradual typing for inference)
        - Same type: OK (i32 = i32, f32 = f32)
        - ANY type difference: ERROR (requires explicit cast)
        
        Rust does NOT allow implicit widening:
        - i32 → i64: ERROR (use `x as i64` or `x.into()`)
        - f32 → f64: ERROR (use `x as f64` or `x.into()`)
        - i32 → f32: ERROR (use `x as f32`)
        
        This enforces Rust's "explicit is better than implicit" philosophy.
        """
        value_t = value_type
        expected_t = expected_type
        
        # UNKNOWN is compatible with any type (gradual typing for inference)
        if value_t == UNKNOWN or expected_t == UNKNOWN:
            return True
        
        # Handle function types specially (for lambda expressions)
        # expected_type might be a Tree from parser for function type annotations
        if isinstance(value_t, FunctionType):
            # If expected is a Tree (parser artifact), accept it for now
            # The parser needs to convert function type annotations properly
            # TODO: Fix parser to convert function_type to FunctionType objects
            if not isinstance(expected_t, Type):
                return True  # Skip check for parser Tree objects
            # Both are types, check if they're both function types
            if isinstance(expected_t, FunctionType):
                # Function types must match exactly (Rust: no covariance/contravariance yet)
                if len(value_t.param_types) != len(expected_t.param_types):
                    return False
                for vp, ep in zip(value_t.param_types, expected_t.param_types):
                    if vp != ep:
                        return False
                return value_t.return_type == expected_t.return_type
            return False
        
        # Same type: OK (only allowed implicit assignment)
        if value_t == expected_t:
            return True
        
        # Handle array types - check if both are arrays
        from ..shared.types import TypeKind
        if hasattr(value_t, 'kind') and hasattr(expected_t, 'kind'):
            if value_t.kind == TypeKind.RECTANGULAR and expected_t.kind == TypeKind.RECTANGULAR:
                # Both are rectangular arrays
                # Element types must match exactly (Rust: no implicit conversions)
                if value_t.element_type != expected_t.element_type:
                    return False
                
                # Check if expected type is dynamic rank - accepts any concrete shape
                if hasattr(expected_t, 'is_dynamic_rank') and expected_t.is_dynamic_rank:
                    return True
                
                # Check shape compatibility with '?' wildcards
                if hasattr(expected_t, 'shape') and hasattr(value_t, 'shape'):
                    if expected_t.shape and value_t.shape:
                        # Shapes must have same rank
                        if len(expected_t.shape) != len(value_t.shape):
                            return False
                        # Each dimension must match or be '?' (wildcard)
                        for exp_dim, val_dim in zip(expected_t.shape, value_t.shape):
                            if exp_dim is not None and exp_dim != '?' and exp_dim != val_dim:
                                return False
                        return True
                
                # If not dynamic and shapes don't match, incompatible
                return False
        
        # Rust semantics: ALL other type differences require explicit cast
        # No implicit widening, no cross-category, no narrowing
        return False
    
    def _get_callee_param_count(self, expr: FunctionCallIR) -> Optional[int]:
        """Get expected parameter count for the callee (function or lambda), or None if unknown (e.g. module call)."""
        callee_expr = getattr(expr, 'callee_expr', None)
        if callee_expr is not None and isinstance(callee_expr, LambdaIR):
            return len(callee_expr.parameters)
        if expr.function_defid is None or self._current_program is None:
            return None
        if expr.module_path:
            return None  # Module call - arity not checked here
        prog = self._current_program
        for func in prog.functions:
            if func.defid == expr.function_defid:
                return len(func.parameters)
        for stmt in getattr(prog, 'statements', []) or []:
            if isinstance(stmt, VariableDeclarationIR):
                if getattr(stmt, 'defid', None) == expr.function_defid:
                    val = getattr(stmt, 'value', None)
                    if val is not None and hasattr(val, 'parameters'):
                        return len(val.parameters)
                    break
        return None

    def visit_function_call(self, expr: FunctionCallIR) -> Type:
        """
        Infer type of function call (ported from precision_engine_visitors.py).
        
        :
        1. Check call arity (compile-time) for lambdas and known functions
        2. Infer argument types first
        3. Try monomorphization (only if types are fully known)
        4. Look up function signature from registry (NOT by visiting definition)
        5. Return function's return type from signature
        """
        # Compile-time arity check for lambdas and callees resolvable from program
        expected_params = self._get_callee_param_count(expr)
        if expected_params is not None and len(expr.arguments) != expected_params:
            self.tcx.reporter.report_error(
                message=f"This call expects {expected_params} argument(s), got {len(expr.arguments)}",
                location=expr.location,
                code="E0061",
            )
            expr.type_info = UNKNOWN
            return UNKNOWN

        # Infer argument types first (needed for builtin return type resolution)
        arg_types_list: List[Type] = []
        for arg in expr.arguments:
            t = arg.accept(self)
            arg_types_list.append(t)
            try:
                object.__setattr__(arg, "type_info", t)
            except AttributeError:
                pass
        arg_types: Tuple[Type, ...] = tuple(arg_types_list)

        # Literal coercion: unsuffixed integer literals coerce to
        # the float type when all args share a type parameter (like Rust's
        # `fn pow<T>(a: T, b: T)`).  Without explicit type parameters we
        # approximate: only coerce when ALL args are scalar PrimitiveType (no
        # array/rectangular args), meaning the function is a pure scalar op
        # where all parameters likely share the same type constraint.
        called_function_defid = expr.function_defid
        if called_function_defid and self.mono_service._is_generic_function(called_function_defid):
            from ..shared.types import I32, I64, F32, F64, PrimitiveType as PrimT
            from ..ir.nodes import LiteralIR
            int_types = {I32, I64}
            float_types = {F32, F64}
            all_primitive = all(isinstance(t, PrimT) for t in arg_types)
            float_args = [t for t in arg_types if t in float_types]
            if all_primitive and float_args:
                target_float = F64 if F64 in float_args else F32
                coerced = list(arg_types)
                for i, (t, arg) in enumerate(zip(arg_types, expr.arguments)):
                    if t in int_types and isinstance(arg, LiteralIR):
                        coerced[i] = target_float
                        object.__setattr__(arg, 'type_info', target_float)
                        if isinstance(arg.value, int):
                            object.__setattr__(arg, 'value', float(arg.value))
                arg_types = tuple(coerced)

        # Only monomorphize if we have full type information
        # This prevents recursion - don't monomorphize during mutual recursion because types are UNKNOWN
        in_recursive_specialization = (
            called_function_defid
            and called_function_defid in getattr(self.mono_service, '_monomorphizing', set())
        )
        if in_recursive_specialization:
            # Recursive call (e.g. factorial(n-1) inside factorial): infer return type from arg
            # Single-param numeric functions often return same type (factorial, gcd, etc.)
            if len(arg_types) == 1 and arg_types[0] != UNKNOWN:
                from ..shared.types import PrimitiveType
                if isinstance(arg_types[0], PrimitiveType) and arg_types[0].name in ('i32', 'i64', 'f32', 'f64'):
                    inferred_type = arg_types[0]
                    expr.type_info = inferred_type
                    return inferred_type
        elif called_function_defid and self.mono_service._is_generic_function(called_function_defid):
            effective_arg_types = list(arg_types)
            if not all(t is not None and t != UNKNOWN for t in effective_arg_types):
                sig = self._get_function(called_function_defid)
                if sig and getattr(sig, 'parameter_types', None) and len(sig.parameter_types) == len(effective_arg_types):
                    for i, t in enumerate(effective_arg_types):
                        if t is None or t == UNKNOWN:
                            effective_arg_types[i] = sig.parameter_types[i] if i < len(sig.parameter_types) else UNKNOWN
                if not all(t is not None and t != UNKNOWN for t in effective_arg_types) and len(effective_arg_types) == 1 and self._current_function and hasattr(self._current_function, 'parameters') and self._current_function.parameters:
                    param_type = getattr(self._current_function.parameters[0], 'param_type', None)
                    if param_type is not None and param_type != UNKNOWN:
                        effective_arg_types[0] = param_type
            if all(t is not None and t != UNKNOWN for t in effective_arg_types):
                # Attempt incremental monomorphization (monomorphize_if_needed in same visit)
                specialized_func = self.mono_service.incremental_monomorphize(
                    expr,
                    tuple(effective_arg_types),
                    "type_inference",
                    required_passes=['range', 'type']
                )
            else:
                specialized_func = None
            if specialized_func and specialized_func.defid:
                assert_defid(specialized_func.defid, allow_none=False)
                expr.function_defid = specialized_func.defid
                logger.debug(f"Updated call {expr.function_name} to use specialized DefId {specialized_func.defid}")
                # Get return type from specialized function signature
                if hasattr(specialized_func, 'return_type') and specialized_func.return_type:
                    inferred_type = specialized_func.return_type
                else:
                    inferred_type = UNKNOWN
                expr.type_info = inferred_type
                return inferred_type
        
        # Look up user-defined function signature by DefId (name resolution sets expr.function_defid)
        signature = self._get_function(expr.function_defid) if expr.function_defid else None
        if signature is not None and expr.function_defid is not None and all(t is not None and t != UNKNOWN for t in arg_types):
            generic_defid = self.mono_service.get_generic_defid_for_specialized(expr.function_defid)
            if generic_defid is None:
                generic_defid = expr.function_defid
            spec_defid = self.mono_service.get_specialized_defid(generic_defid, arg_types)
            if spec_defid is None and self.mono_service._is_generic_function(generic_defid):
                if expr.function_defid != generic_defid:
                    assert_defid(generic_defid, allow_none=False)
                    expr.function_defid = generic_defid
                specialized_func = self.mono_service.incremental_monomorphize(
                    expr, arg_types, "type_inference", required_passes=['range', 'type']
                )
                if specialized_func and getattr(specialized_func, 'defid', None):
                    spec_defid = specialized_func.defid
                    assert_defid(spec_defid, allow_none=False)
                    expr.function_defid = spec_defid
            if spec_defid is not None:
                assert_defid(spec_defid, allow_none=False)
                spec_sig = self._get_function(spec_defid)
                if spec_sig is not None:
                    signature = spec_sig
                    expr.function_defid = spec_defid

        if signature is None:
            module_path = getattr(expr, "module_path", None) or ()
            if module_path and len(module_path) >= 1 and module_path[0] == "python":
                inferred_type = self._infer_python_module_call_type(expr.function_name, arg_types)
                expr.type_info = inferred_type
                return inferred_type
            logger.debug(f"Function DefId {expr.function_defid} not in function_ir_map")
            inferred_type = UNKNOWN
            expr.type_info = inferred_type
            return inferred_type

        # Never type-check a generic call against the generic's signature (may be wrong/unset).
        if expr.function_defid is not None and self.mono_service._is_generic_function(expr.function_defid):
            signature = FunctionSignature(
                name=signature.name,
                parameter_types=tuple(UNKNOWN for _ in signature.parameter_types),
                parameter_names=signature.parameter_names,
                return_type=signature.return_type,
            )

        # Type check: arity (number of arguments)
        if len(arg_types) != len(signature.parameter_types):
            self.tcx.reporter.report_error(
                message=f"Function '{signature.name}' expects {len(signature.parameter_types)} arguments, got {len(arg_types)}",
                location=expr.location,
                code="E0061"
            )
            expr.type_info = UNKNOWN
            return UNKNOWN
        
        # Type check each argument against parameter type
        # This ensures type safety - mismatches are compile-time errors
        for i, (arg_type, param_type) in enumerate(zip(arg_types, signature.parameter_types)):
            if param_type != UNKNOWN and arg_type != UNKNOWN:
                # Check type compatibility
                if not self._types_compatible(arg_type, param_type):
                    param_name = signature.parameter_names[i] if i < len(signature.parameter_names) else f"arg{i+1}"
                    # Report error and mark compilation as failed
                    self.tcx.reporter.report_error(
                        message=f"Type mismatch for argument '{param_name}' in call to '{expr.function_name}': expected {param_type}, got {arg_type}",
                        location=expr.location,
                        code="E0308"
                    )
                    # Set type to UNKNOWN after error (don't continue with wrong type)
                    expr.type_info = UNKNOWN
                    # Continue checking other arguments but return UNKNOWN
                    # Don't return early - check all arguments for better error messages
        
        # Return function's return type from signature
        # If return type is None/UNKNOWN, just return UNKNOWN (don't try to visit definition)
        # This prevents recursion - we don't visit function definitions during type inference
        if signature.return_type and signature.return_type != UNKNOWN:
            inferred_type = signature.return_type
        else:
            # Return type not yet known (function body not yet analyzed or has UNKNOWN return type)
            inferred_type = UNKNOWN
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        
        
        return inferred_type
    
    def _infer_python_module_call_type(self, func_name: str, arg_types: Tuple[Type, ...]) -> Type:
        """Infer return type for Python module calls (e.g. numpy.power, numpy.sqrt).
        
        strict type promotion (same rules as _promote_types).
        Callers must ensure arg types are compatible before reaching here.
        """
        if not arg_types:
            return UNKNOWN
        promoted = arg_types[0]
        for t in arg_types[1:]:
            promoted = self._promote_types(promoted, t)
        if promoted is not None and promoted != UNKNOWN:
            return promoted
        return UNKNOWN

    def _types_compatible(self, actual: Type, expected: Type) -> bool:
        """
        Check if actual type is compatible with expected type.
        
        Strict type checking - only exact matches and widening allowed.
        - Same types are compatible
        - Unknown types are compatible (runtime check needed)
        - Widening is allowed (i32 -> i64, f32 -> f64)
        - Cross-category (i32 vs f32) is NOT compatible
        - Dynamic rank types ([T; *]) match any rank of that element type
        
        Uses structured type comparison only - no string comparisons.
        """
        # Handle UnknownType - allow (runtime check needed)
        if actual == UNKNOWN or expected == UNKNOWN:
            return True
        
        # Exact match using Type.__eq__ (for PrimitiveType, compares .name)
        if actual == expected:
            return True
        
        # Widening rules for primitive types
        if isinstance(actual, PrimitiveType) and isinstance(expected, PrimitiveType):
            # f32 can widen to f64
            if actual.name == "f32" and expected.name == "f64":
                return True
            
            # i32 can widen to i64 (future support)
            if actual.name == "i32" and expected.name == "i64":
                return True
        
        # RectangularType: same element type and compatible rank (specialize by rank, not shape)
        # [f32; None] and [f32] (shape=None) both mean 1D array of f32; allow when element types match
        if isinstance(actual, RectangularType) and isinstance(expected, RectangularType):
            if not self._types_compatible(actual.element_type, expected.element_type):
                return False
            actual_rank = len(actual.shape) if actual.shape is not None else None
            expected_rank = len(expected.shape) if expected.shape is not None else None
            if actual_rank == expected_rank:
                return True
            if actual_rank is None or expected_rank is None:
                return True
            if expected.is_dynamic_rank:
                return True
            return False

        # Dynamic rank matching: [T; *] matches any rank of that element type
        # Example: [[f32]] matches [f32; *], [[[f32]]] matches [f32; *]
        # Extract the BASE element type from nested RectangularTypes for comparison
        if isinstance(expected, RectangularType) and expected.is_dynamic_rank:
            if isinstance(actual, RectangularType):
                # Extract base element type from actual (may be nested)
                actual_base = actual.element_type
                while isinstance(actual_base, RectangularType):
                    actual_base = actual_base.element_type
                
                # Extract base element type from expected (should be primitive)
                expected_base = expected.element_type
                while isinstance(expected_base, RectangularType):
                    expected_base = expected_base.element_type
                
                # Check if base element types are compatible
                return self._types_compatible(actual_base, expected_base)
        
        # Cross-category types are NOT compatible
        # i32 vs f32/f64: NOT compatible
        # f32 vs i32: NOT compatible
        # This ensures type safety - explicit casts required for cross-category conversions
        return False
    
    def visit_unary_op(self, expr) -> Type:
        """Infer type of unary operation"""
        inferred_type = expr.operand.accept(self)
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_rectangular_access(self, expr) -> Type:
        """Infer type of rectangular array access. Full indexing -> element type; partial indexing -> slice (RectangularType with remaining dims)."""
        array_type = expr.array.accept(self)
        indices = getattr(expr, 'indices', None) or []
        for idx in indices:
            if idx is not None and hasattr(idx, 'accept'):
                idx.accept(self)
        if not isinstance(array_type, RectangularType):
            inferred_type = UNKNOWN
        else:
            shape = array_type.shape
            rank = len(shape) if shape else 0
            num_indices = len(indices)
            if rank == 0:
                inferred_type = RectangularType(array_type.element_type, None, is_dynamic_rank=True) if num_indices > 0 else array_type.element_type
            elif num_indices >= rank:
                inferred_type = array_type.element_type
            else:
                remaining_shape = shape[num_indices:]
                inferred_type = RectangularType(
                    element_type=array_type.element_type,
                    shape=remaining_shape if remaining_shape else None,
                    is_dynamic_rank=getattr(array_type, 'is_dynamic_rank', False),
                )
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_jagged_access(self, expr) -> Type:
        """Infer type of jagged array access"""
        base_type = expr.base.accept(self)
        # Visit all indices to infer their types
        for idx in (getattr(expr, 'index_chain', None) or []):
            if idx is not None and hasattr(idx, 'accept'):
                idx.accept(self)
        # TODO: Extract element type from base type
        inferred_type = UNKNOWN
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_block_expression(self, expr: BlockExpressionIR) -> Type:
        """
        Infer type of block expression - type is final_expr type or unit.
        
        Rust Pattern: Block type is final expression type, or unit if no final_expr
        """
        # Enter new scope for block (RAII pattern)
        with self._scope():
            # Execute statements for side effects (type checking)
            for stmt in expr.statements:
                stmt.accept(self)  # Visitor pattern
            
            # Block's type is the type of final_expr, or unit if None
            if expr.final_expr is not None:
                inferred_type = expr.final_expr.accept(self)  # Visitor pattern
            else:
                # No final expression - returns unit type
                inferred_type = UNKNOWN  # TODO: Add unit type
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_if_expression(self, expr) -> Type:
        """
        Infer type of if expression.
        
        Rust Pattern: rustc validates if-without-else must return unit type
        """
        # Visit condition first to infer types for all nodes in condition
        condition_type = expr.condition.accept(self)
        then_type = expr.then_expr.accept(self)
        
        if expr.else_expr:
            else_type = expr.else_expr.accept(self)
            # TODO: Unify types
            inferred_type = then_type
        else:
            # If-without-else: validate that then branch returns unit
            # This is required because if condition is false, no value is produced
            if then_type != UNKNOWN and then_type is not None:
                # Check if then_type is unit/void (for now, we don't have explicit unit type)
                # In practice, this means if-without-else can only be used as a statement
                # Report error: if-expression without else must not produce a value
                self.tcx.reporter.report_error(
                    "If-expression without else branch can only be used as a statement "
                    "(when then branch returns unit). Use 'if ... else ...' for expressions.",
                    location=expr.location
                )
            inferred_type = then_type
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_lambda(self, expr) -> Type:
        """Infer type of lambda (returns FunctionType). Bind params in scope before visiting body."""
        param_types = tuple(
            param.param_type if param.param_type else UNKNOWN
            for param in expr.parameters
        )
        with self._scope():
            for i, param in enumerate(expr.parameters):
                param_type = param_types[i] if i < len(param_types) else UNKNOWN
                param_defid = getattr(param, 'defid', None)
                if param_defid is not None:
                    self._set_var(param_defid, param_type)
            return_type = expr.body.accept(self)
        inferred_type = FunctionType(param_types, return_type)
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_function_def(self, node: FunctionDefIR) -> Type:
        """
        Visit function definition (ported from precision_engine.py).
        
        Analyzes function body to infer return type, especially for generic functions.
        For functions like abs(x) = if x >= 0 { x } else { -x }, infers return type = parameter type.
        
        CRITICAL: Always visit the body so every IR node gets type_info (required by IR validation).
        When signature is missing we still traverse and set type_info (possibly UNKNOWN).
        """
        signature = self._get_function(node.defid) if node.defid else None

        with self._function_scope(node):
            # Enter new scope for function body (RAII pattern - RAII pattern)
            with self._scope():
                # Bind parameters in function scope BEFORE visiting body
                parameter_names: List[str] = []
                parameter_types: List[Type] = []
                if signature:
                    parameter_names = signature.parameter_names
                    parameter_types = signature.parameter_types

                if hasattr(node, 'parameters'):
                    for i, param in enumerate(node.parameters):
                        param_type = parameter_types[i] if i < len(parameter_types) else (
                            getattr(param, 'param_type', None) or UNKNOWN
                        )
                        param_defid = getattr(param, 'defid', None)
                        if param_defid is None:
                            raise RuntimeError(
                                f"Parameter has no DefId (name={getattr(param, 'name', '?')}). "
                                "Ensure NameResolutionPass runs before TypeInferencePass."
                            )
                        self._set_var(param_defid, param_type)
                        if signature:
                            logger.debug(f"Bound parameter {param_defid} → {param_type} in function {node.name}")

                # CRITICAL: Always visit body so all nodes get type_info (IR must be well-typed)
                body_type = node.body.accept(self)

                # Infer return type from function body (only when signature available)
                return_type = None
                if signature and isinstance(node.body, BlockExpressionIR) and node.body.final_expr:
                    final_expr = node.body.final_expr

                    # Check if final_expr is an if-expression
                    from ..ir.nodes import IfExpressionIR, IdentifierIR, UnaryOpIR
                    if isinstance(final_expr, IfExpressionIR):
                        then_expr = final_expr.then_expr
                        else_expr = final_expr.else_expr

                        # Extract identifiers from both branches
                        then_ident = None
                        else_ident = None

                        # Extract identifier from then branch (may be wrapped in BlockExpressionIR)
                        if isinstance(then_expr, BlockExpressionIR) and then_expr.final_expr:
                            then_expr = then_expr.final_expr
                        if isinstance(then_expr, IdentifierIR):
                            then_ident = then_expr

                        # Extract identifier from else branch (may be unary minus)
                        if isinstance(else_expr, BlockExpressionIR) and else_expr.final_expr:
                            else_expr = else_expr.final_expr
                        if isinstance(else_expr, IdentifierIR):
                            else_ident = else_expr
                        elif isinstance(else_expr, UnaryOpIR) and getattr(else_expr, "operator", None) == UnaryOp.NEG:
                            # Handle -x case
                            if isinstance(else_expr.operand, IdentifierIR):
                                else_ident = else_expr.operand

                        # If both branches reference the same parameter (by DefId), return type = parameter type
                        # Handles abs(x) = if x >= 0 { x } else { -x }
                        if (then_ident and else_ident and getattr(then_ident, 'defid', None) and
                                then_ident.defid == getattr(else_ident, 'defid', None)):
                            for param_idx, param in enumerate(node.parameters):
                                if getattr(param, 'defid', None) == then_ident.defid:
                                    param_type = parameter_types[param_idx] if param_idx < len(parameter_types) else UNKNOWN
                                    return_type = param_type
                                    logger.debug(f"Inferred return type for '{node.name}' from if-expr pattern (defid): {return_type}")
                                    break

                    # If final expression is an identifier, look up its type in scope by DefId
                    if return_type is None and isinstance(final_expr, IdentifierIR):
                        inferred_return_type = self._get_var(final_expr.defid)
                        if inferred_return_type is not None:
                            return_type = inferred_return_type
                            logger.debug(f"Inferred return type for '{node.name}' from final expr defid: {return_type}")

                    # Use the expression's type (visitor pattern handles all cases)
                    if return_type is None and hasattr(final_expr, 'type_info') and final_expr.type_info is not None:
                        return_type = final_expr.type_info
                        logger.debug(f"Inferred return type for '{node.name}' from final expr type_info: {return_type}")

                # Fallback: use body_type if return_type not inferred
                if return_type is None:
                    return_type = body_type if body_type != UNKNOWN else UNKNOWN

        is_generic = node.defid is not None and self.mono_service._is_generic_function(node.defid)
        if not is_generic and return_type is not None and return_type != UNKNOWN:
            if not getattr(node, 'return_type', None) or node.return_type == UNKNOWN:
                node.return_type = return_type
        
        param_types = tuple(
            param.param_type if param.param_type else UNKNOWN
            for param in node.parameters
        )
        
        return FunctionType(param_types, return_type if return_type is not None else UNKNOWN)
    
    def visit_constant_def(self, node: ConstantDefIR) -> Type:
        """Visit constant definition"""
        return node.value.accept(self)

    def _extract_int_from_shape_expr(self, expr: Any) -> Optional[int]:
        """Extract a single int from a shape dimension expression (LiteralIR, or RangeIR.end as literal)."""
        if expr is None:
            return None
        v = getattr(expr, 'value', None)
        if isinstance(v, int):
            return v
        if hasattr(expr, 'end'):
            return self._extract_int_from_shape_expr(expr.end)
        return None

    def _concrete_shape_from_exprs(self, shape_exprs: List[Any]) -> Optional[Tuple[int, ...]]:
        """If all shape elements are or wrap literal ints, return (d0, d1, ...); else None."""
        if not shape_exprs:
            return None
        dims = []
        for expr in shape_exprs:
            d = self._extract_int_from_shape_expr(expr)
            if d is None:
                return None
            dims.append(d)
        return tuple(dims)

    def visit_lowered_einstein(self, node: Any) -> Type:
        """Visit lowered Einstein: recurse into shape, then all clauses so every nested node gets type_info."""
        shape_exprs = getattr(node, 'shape', []) or []
        for expr in shape_exprs:
            if expr is not None:
                expr.accept(self)
        result_type = UNKNOWN
        for item in getattr(node, 'items', []) or []:
            result_type = item.accept(self) or result_type
        concrete = self._concrete_shape_from_exprs(shape_exprs)
        if concrete is not None:
            elem = result_type if result_type not in (None, UNKNOWN) else UNKNOWN
            return RectangularType(element_type=elem, shape=concrete)
        if shape_exprs:
            return self._build_rectangular_array_type(result_type if result_type not in (None, UNKNOWN) else UNKNOWN, len(shape_exprs))
        return result_type

    def visit_lowered_einstein_clause(self, node: Any) -> Type:
        """Visit lowered clause: recurse into body, loops, bindings, guards. Visit loop.variable so IdentifierIR gets type_info."""
        for loop in getattr(node, 'loops', []) or []:
            var = getattr(loop, 'variable', None)
            if var is not None and getattr(var, 'defid', None) is not None:
                self._set_var(var.defid, I32)
                var.accept(self)
            if getattr(loop, 'iterable', None):
                loop.iterable.accept(self)
        for loop in (getattr(node, 'reduction_ranges', None) or {}).values():
            var = getattr(loop, 'variable', None)
            if var is not None and getattr(var, 'defid', None) is not None:
                self._set_var(var.defid, I32)
                var.accept(self)
            if getattr(loop, 'iterable', None):
                loop.iterable.accept(self)
        if getattr(node, 'body', None):
            body_type = node.body.accept(self)
        else:
            body_type = UNKNOWN
        for b in getattr(node, 'bindings', []) or []:
            if getattr(b, 'expr', None):
                b.expr.accept(self)
        for g in getattr(node, 'guards', []) or []:
            if getattr(g, 'condition', None):
                g.condition.accept(self)
        return body_type if body_type is not None else UNKNOWN

    def visit_lowered_reduction(self, node: Any) -> Type:
        """Visit lowered reduction: recurse into loops (variable + iterable), body, and guards."""
        for loop in getattr(node, 'loops', []) or []:
            var = getattr(loop, 'variable', None)
            if var is not None and getattr(var, 'defid', None) is not None:
                self._set_var(var.defid, I32)
                var.accept(self)
            if getattr(loop, 'iterable', None):
                loop.iterable.accept(self)
        body_type = node.body.accept(self) if getattr(node, 'body', None) else UNKNOWN
        for g in getattr(node, 'guards', []) or []:
            if getattr(g, 'condition', None):
                g.condition.accept(self)
        inferred_type = body_type if body_type is not None else UNKNOWN
        object.__setattr__(node, 'type_info', inferred_type)
        return inferred_type

    def visit_lowered_comprehension(self, node: Any) -> Type:
        """Visit lowered comprehension: recurse into body and guards. Result is array of body element type."""
        body_type = node.body.accept(self) if getattr(node, 'body', None) else UNKNOWN
        for g in getattr(node, 'guards', []) or []:
            if getattr(g, 'condition', None):
                g.condition.accept(self)
        elem = self._get_base_element_type(body_type) if body_type and body_type is not UNKNOWN else (body_type if body_type is not None else UNKNOWN)
        inferred_type = RectangularType(element_type=elem, shape=None) if elem is not UNKNOWN else UNKNOWN
        object.__setattr__(node, 'type_info', inferred_type)
        return inferred_type

    def visit_unary_op(self, node) -> Type:
        """Infer type of unary operation"""
        inferred_type = node.operand.accept(self)
        
        # Set type_info on IR node (use object.__setattr__ for frozen nodes)
        object.__setattr__(node, 'type_info', inferred_type)
        return inferred_type
    
    def visit_range(self, node) -> Type:
        """Infer type of range expression (RANGE primitive)"""
        if node.start:
            node.start.accept(self)
        if node.end:
            node.end.accept(self)
        inferred_type = RANGE
        object.__setattr__(node, 'type_info', inferred_type)
        return inferred_type
    
    def _iterable_element_type(self, range_type: Type) -> Type:
        """Infer type of iteration variable from iterable type. E.g. [f32; N] -> f32; 0..n -> I32."""
        from ..shared.types import JaggedType
        if isinstance(range_type, RectangularType):
            return range_type.element_type
        if isinstance(range_type, JaggedType):
            return range_type.element_type
        # Range (0..n), RANGE, or unknown: iteration variable is an index -> I32
        return I32

    def visit_array_comprehension(self, node) -> Type:
        """Infer type of array comprehension. Bind iteration variables from iterable element type; visit body and constraints in same scope."""
        raw_defids = getattr(node, 'variable_defids', None) or []
        defids = raw_defids if isinstance(raw_defids, list) else [raw_defids] if raw_defids is not None else []
        var_names = set(getattr(node, 'variables', None) or [])
        ranges = getattr(node, 'ranges', None) or []
        with self._scope():
            collected = set()
            # Infer each iteration variable type from its range (iterable). E.g. x in data -> element type of data.
            for i, defid in enumerate(defids):
                if defid is None or defid in collected:
                    continue
                var_type = I32
                if i < len(ranges) and ranges[i] is not None:
                    range_type = ranges[i].accept(self)
                    var_type = self._iterable_element_type(range_type)
                collected.add(defid)
                self._set_var(defid, var_type)
            # Also bind any defids found in body/constraints/ranges (covers copied IR with None variable_defids)
            if var_names:
                if node.body:
                    for defid, _ in self._collect_identifier_defids_in_expr(node.body, var_names):
                        if defid not in collected:
                            collected.add(defid)
                            self._set_var(defid, I32)
                for c in (node.constraints or []):
                    for defid, _ in self._collect_identifier_defids_in_expr(c, var_names):
                        if defid not in collected:
                            collected.add(defid)
                            self._set_var(defid, I32)
                for r in (node.ranges or []):
                    if r:
                        for defid, _ in self._collect_identifier_defids_in_expr(r, var_names):
                            if defid not in collected:
                                collected.add(defid)
                                self._set_var(defid, I32)
            if node.body:
                node.body.accept(self)
            # Visit constraints in same scope so iteration vars (e.g. i in data[i] > 0) are in scope
            for constraint in node.constraints or []:
                constraint.accept(self)
        for range_expr in node.ranges or []:
            if range_expr:
                range_expr.accept(self)
        body_type = getattr(node.body, 'type_info', None) if node.body else None
        base_elem = self._get_base_element_type(body_type) if body_type and body_type is not UNKNOWN else I32
        inferred_type = RectangularType(element_type=base_elem, shape=None)
        object.__setattr__(node, 'type_info', inferred_type)
        return inferred_type

    def visit_array_literal(self, expr) -> Type:
        """Infer type of array literal (: infer concrete shapes with flat element types)"""
        if not expr.elements:
            inferred_type = UNKNOWN
        else:
            # Infer element types and unify
            element_types = [elem.accept(self) for elem in expr.elements]
            # Get the common element type (first element for now - could add unification logic)
            element_type = element_types[0] if element_types else UNKNOWN
            
            # Infer shape from array literal structure
            shape = self._infer_array_literal_shape(expr)
            
            # Flatten element type for multidimensional arrays
            # [[1, 2, 3], [4, 5, 6]] → [i32; 2, 3] with element_type=i32 (not [[i32; 3]; 2, 3])
            # This ensures compatibility with type annotations like [i32; 2, 3]
            base_element_type = self._get_base_element_type(element_type)
            
            inferred_type = RectangularType(element_type=base_element_type, shape=shape)
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def _get_base_element_type(self, elem_type: Type) -> Type:
        """Get the base (scalar) element type from a potentially nested array type"""
        if isinstance(elem_type, RectangularType):
            # Recursively unwrap nested RectangularTypes
            return self._get_base_element_type(elem_type.element_type)
        return elem_type
    
    def _infer_array_literal_shape(self, array_lit) -> Optional[Tuple[int, ...]]:
        """Infer shape from array literal recursively"""
        if not hasattr(array_lit, 'elements') or not array_lit.elements:
            return (0,)
        
        shape = [len(array_lit.elements)]
        
        # Check if first element is also an array literal (nested)
        from ..ir.nodes import ArrayLiteralIR
        if array_lit.elements and isinstance(array_lit.elements[0], ArrayLiteralIR):
            first_elem_shape = self._infer_array_literal_shape(array_lit.elements[0])
            if first_elem_shape:
                shape.extend(first_elem_shape)
        
        return tuple(shape)
    
    def visit_tuple_expression(self, expr) -> Type:
        """Infer type of tuple expression"""
        # Visit all elements to infer their types
        for elem in expr.elements:
            elem.accept(self)
        # Tuples have heterogeneous types - return UNKNOWN for now
        # TODO: Implement tuple type
        inferred_type = UNKNOWN
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_tuple_access(self, expr) -> Type:
        """Infer type of tuple access"""
        tuple_type = expr.tuple_expr.accept(self)
        # TODO: Extract element type at index
        inferred_type = UNKNOWN
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_interpolated_string(self, expr) -> Type:
        """Infer type of interpolated string"""
        # Visit all expression parts (strings are not visited)
        for part in expr.parts:
            if hasattr(part, 'accept'):
                part.accept(self)
        inferred_type = PrimitiveType("str")
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_cast_expression(self, expr) -> Type:
        """Infer type of cast expression"""
        # Visit expression being cast
        expr.expr.accept(self)
        # Cast type is the target type
        inferred_type = expr.target_type if expr.target_type else UNKNOWN
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_member_access(self, expr) -> Type:
        """Infer type of member access"""
        object_type = expr.object.accept(self)
        # TODO: Lookup member type from object type (struct field, module item)
        inferred_type = UNKNOWN

        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_try_expression(self, expr) -> Type:
        """Infer type of try expression"""
        # Try returns Result type - for now return operand type
        inferred_type = expr.operand.accept(self)
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_match_expression(self, expr) -> Type:
        """Infer type of match expression"""
        scrutinee_type = expr.scrutinee.accept(self)
        arm_types = []
        for arm in expr.arms:
            with self._scope():
                self._bind_pattern_vars(arm.pattern, scrutinee_type)
                if isinstance(arm.pattern, GuardPatternIR):
                    arm.pattern.guard_expr.accept(self)
                arm_types.append(arm.body.accept(self))
        inferred_type = arm_types[0] if arm_types else UNKNOWN
        expr.type_info = inferred_type
        return inferred_type

    def _bind_pattern_vars(self, pattern, scrutinee_type: Type) -> None:
        """Bind identifier variables in a pattern to their inferred types."""
        if isinstance(pattern, IdentifierPatternIR):
            did = getattr(pattern, 'defid', None)
            if did is not None:
                self._set_var(did, scrutinee_type)
        elif isinstance(pattern, GuardPatternIR):
            self._bind_pattern_vars(pattern.inner_pattern, scrutinee_type)
        elif isinstance(pattern, TuplePatternIR):
            for elem in pattern.patterns:
                self._bind_pattern_vars(elem, UNKNOWN)
        elif isinstance(pattern, ArrayPatternIR):
            for elem in pattern.patterns:
                self._bind_pattern_vars(elem, UNKNOWN)
        elif isinstance(pattern, RestPatternIR):
            if pattern.pattern is not None:
                did = getattr(pattern.pattern, 'defid', None)
                if did is not None:
                    self._set_var(did, UNKNOWN)
    
    def visit_reduction_expression(self, expr) -> Type:
        """Infer type of reduction. Bind loop var DefIds; visit range exprs so they get type_info; then body."""
        with self._scope():
            for loop_var in expr.loop_vars or []:
                if isinstance(loop_var, IdentifierIR) and getattr(loop_var, 'defid', None) is not None:
                    self._set_var(loop_var.defid, I32)
            # Visit range expressions in loop_var_ranges so RangeIR (and start/end) get type_info (fixes validation)
            for range_ir in (getattr(expr, 'loop_var_ranges', None) or {}).values():
                if range_ir is not None:
                    range_ir.accept(self)
            inferred_type = expr.body.accept(self)
            if expr.where_clause:
                for constraint in expr.where_clause.constraints:
                    constraint.accept(self)
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_where_expression(self, expr) -> Type:
        """Infer type of where expression"""
        from ..ir.nodes import WhereExpressionIR, ReductionExpressionIR
        # Handle case where ReductionExpressionIR is incorrectly routed here
        if isinstance(expr, ReductionExpressionIR):
            # Route to correct method
            return self.visit_reduction_expression(expr)
        if not isinstance(expr, WhereExpressionIR):
            raise TypeError(
                f"visit_where_expression expected WhereExpressionIR, got {type(expr).__name__}"
            )
        # E0303: Reduction cannot have iteration domain in where; use sum[k in 0..4](...) instead
        if isinstance(expr.expr, ReductionExpressionIR):
            reduction_loop_names = {getattr(ident, "name", None) for ident in (expr.expr.loop_vars or [])}
            reduction_loop_names.discard(None)
            op_name = getattr(expr.expr, "operation", "reduction")
            for constraint in expr.constraints:
                if isinstance(constraint, BinaryOpIR) and getattr(constraint, "operator", None) == "in":
                    left = getattr(constraint, "left", None)
                    if isinstance(left, IdentifierIR):
                        var_name = getattr(left, "name", None)
                        if var_name and var_name in reduction_loop_names:
                            self.tcx.reporter.report_error(
                                f"Reduction cannot have iteration domain '{var_name} in ...' in where clause. "
                                f"Use inline syntax: {op_name}[{var_name} in range](...).",
                                constraint.location,
                                code="E0303",
                            )
        # Visit expression
        inferred_type = expr.expr.accept(self)
        # Visit all constraints
        for constraint in expr.constraints:
            constraint.accept(self)
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_arrow_expression(self, expr) -> Type:
        """Infer type of arrow expression"""
        # Visit all components
        for component in expr.components:
            component.accept(self)
        # Arrow expression returns function type
        # For now, return UNKNOWN (complex composition)
        inferred_type = UNKNOWN
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_pipeline_expression(self, expr) -> Type:
        """Infer type of pipeline expression"""
        # Visit left expression
        expr.left.accept(self)
        # Pipeline type is the right expression's return type
        inferred_type = expr.right.accept(self)
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def visit_builtin_call(self, expr) -> Type:
        """Infer type of builtin call from _BUILTIN_RETURN_TABLE."""
        args = getattr(expr, "args", []) or []
        arg_types: List[Type] = []
        for arg in args:
            t = arg.accept(self)
            arg_types.append(t if t is not None else UNKNOWN)
        builtin_name = getattr(expr, "builtin_name", None)
        result = _BUILTIN_RETURN_TABLE.get(builtin_name, UNKNOWN)
        if callable(result):
            result = result(arg_types)
        expr.type_info = result
        return result
    
    def visit_function_ref(self, expr) -> Type:
        """Infer type of function reference"""
        # Function reference returns FunctionType
        # TODO: Lookup function type from DefId
        inferred_type = UNKNOWN
        
        # Set type_info on IR node
        expr.type_info = inferred_type
        return inferred_type
    
    def _collect_identifier_defids_in_expr(self, expr, names: Optional[Set[str]] = None):
        """Collect (defid, name) for all IdentifierIR in expr. If names is set, only include those names."""
        out = []
        if expr is None:
            return out
        if isinstance(expr, IdentifierIR) and expr.defid is not None:
            if names is None or (expr.name and expr.name in names):
                out.append((expr.defid, expr.name))
            return out
        for attr in ('left', 'right', 'operand', 'body', 'expr', 'condition', 'then_expr', 'else_expr', 'value'):
            if hasattr(expr, attr):
                out.extend(self._collect_identifier_defids_in_expr(getattr(expr, attr), names))
        for attr in ('arguments', 'indices', 'elements'):
            if hasattr(expr, attr):
                for sub in getattr(expr, attr) or []:
                    if isinstance(sub, list):
                        for s in sub:
                            out.extend(self._collect_identifier_defids_in_expr(s, names))
                    else:
                        out.extend(self._collect_identifier_defids_in_expr(sub, names))
        return out

    def visit_einstein(self, node: EinsteinIR) -> Type:
        """Infer type of one Einstein clause. Bind index DefIds; delegate to visit_index_var/visit_index_rest."""
        index_names = set()
        for idx in node.indices or []:
            if isinstance(idx, (IndexVarIR, IndexRestIR)):
                if idx.defid is None:
                    raise RuntimeError(
                        f"Einstein index slot has no DefId (name={idx.name}). "
                        "Ensure NameResolutionPass runs before TypeInferencePass."
                    )
                self._set_var(idx.defid, I32)
                if idx.name:
                    index_names.add(idx.name)
            idx.accept(self)
        # Also bind any identifiers in value/where that match index names (in case DefIds differ)
        if index_names and node.value:
            for defid, _ in self._collect_identifier_defids_in_expr(node.value, index_names):
                self._set_var(defid, I32)
        if node.where_clause:
            for c in (node.where_clause.constraints or []):
                for defid, _ in self._collect_identifier_defids_in_expr(c, index_names):
                    self._set_var(defid, I32)
        # Visit range expressions in variable_ranges so RangeIR (and start/end) get type_info
        for rng in (getattr(node, 'variable_ranges', None) or {}).values():
            if rng is not None and hasattr(rng, 'accept'):
                rng.accept(self)
        for idx in (node.indices or []):
            if idx is not None and hasattr(idx, 'accept'):
                idx.accept(self)
        if node.where_clause:
            for c in (node.where_clause.constraints or []):
                c.accept(self)
        if node.value:
            return node.value.accept(self)
        return UNKNOWN

    def visit_einstein_declaration(self, node) -> Type:
        """
        Visit Einstein declaration and infer array type.
        
        Einstein declarations create array variables.
        - Infer element type from value expression
        - Create RectangularType with element type and correct rank
        - Bind array name in scope (so D[i] lookups work)
        - Do NOT set type_info on node (EinsteinDeclarationIR is not an expression)
        
        Try multiple sources for element type (in priority order):
        1. Direct value expression (before lowering)
        2. Lowered body expression (after lowering)
        3. Lowered metadata element_type
        4. Node's element_type attribute
        """
        import json
        
        # Use None for element_type when unknown (instead of UNKNOWN)
        # This preserves array structure (rank) while indicating type is not yet inferred
        element_type = None
        source = "unknown"
        
        
        # Try multiple sources for element type
        
        def _is_unknown_type(t) -> bool:
            """True if type is UNKNOWN or TypeKind.UNKNOWN (treat as not inferred)."""
            if t is None:
                return True
            if t is UNKNOWN:
                return True
            if hasattr(t, "kind") and getattr(t, "kind", None) == TypeKind.UNKNOWN:
                return True
            return False

        # 1. Propose precision for each clause, then promote (reject if not compatible), then set back to each clause
        if node.clauses:
            proposed_per_clause = []  # (clause, proposed_element_type) for each clause
            for clause in node.clauses:
                array_type = clause.accept(self)
                et = self._extract_base_element_type(array_type)
                proposed_per_clause.append((clause, et))
            clause_element_types = [et for _, et in proposed_per_clause if not _is_unknown_type(et)]
            if clause_element_types:
                element_type = clause_element_types[0]
                for et in clause_element_types[1:]:
                    element_type = self._promote_types(element_type, et, node.location)
                if not _is_unknown_type(element_type):
                    source = "clauses_promoted"
        
        # 2. Last resort: Check node's element_type attribute (if set by shape analysis)
        if element_type is None and hasattr(node, 'element_type') and node.element_type:
            et = node.element_type
            if not _is_unknown_type(et):
                element_type = et
                source = "node_attribute"

        # 3. Fallback: Use first parameter's element type when body inference fails (e.g. softmax
        #    output where exp/sum return UNKNOWN). Ensures float arrays get f32, avoiding int32
        #    backend fallback that produces wrong output (integer division → one-hot).
        if element_type is None and self._current_function and hasattr(self._current_function, 'parameters'):
            from ..shared.types import JaggedType, F32, F64
            for param in self._current_function.parameters:
                pt = getattr(param, 'param_type', None)
                if pt is None:
                    continue
                if isinstance(pt, (RectangularType, JaggedType)):
                    et = getattr(pt, 'element_type', None)
                    if et in (F32, F64) or (hasattr(et, 'name') and getattr(et, 'name', '') in ('f32', 'f64')):
                        element_type = et
                        source = "first_float_param"
                        logger.debug(f"Einstein element_type fallback from param '{param.name}': {et}")
                        break
                elif pt in (F32, F64) or (hasattr(pt, 'name') and getattr(pt, 'name', '') in ('f32', 'f64')):
                    element_type = pt
                    source = "first_float_param"
                    break

        # All clauses must have the same rank; fail on mismatch
        if not node.clauses:
            num_dimensions = 0
        else:
            ranks = [len(c.indices) for c in node.clauses]
            if len(set(ranks)) > 1:
                self.tcx.reporter.report_error(
                    f"Einstein declaration '{getattr(node, 'name', '?')}' has clauses with different ranks: {ranks}. All clauses must have the same rank.",
                    location=node.location,
                )
            num_dimensions = ranks[0]
        shape_exprs = getattr(node, 'shape', []) or []
        concrete = self._concrete_shape_from_exprs(shape_exprs)
        if concrete is not None:
            elem = element_type if not _is_unknown_type(element_type) else UNKNOWN
            array_type = RectangularType(element_type=elem, shape=concrete)
        else:
            array_type = self._build_rectangular_array_type(element_type, num_dimensions)
        
        
        # Store in scope by DefId (fail fast if no defid)
        decl_defid = getattr(node, 'defid', None)
        if decl_defid is None:
            raise RuntimeError(
                f"Einstein declaration has no DefId (name={getattr(node, 'name', '?')}). "
                "Ensure NameResolutionPass runs before TypeInferencePass."
            )
        self._set_var(decl_defid, array_type)
        
        # Store element_type on the IR node so EinsteinLoweringPass can use it
        # This is what the pass ordering expects (TypeInference before EinsteinLowering)
        if not hasattr(node, 'element_type') or node.element_type is None:
            object.__setattr__(node, 'element_type', element_type)
        
        # Return the array type (statements don't have type_info, but visitor needs return value)
        return array_type
    
    # Pattern visitors (no-op, patterns don't have types)
    def visit_literal_pattern(self, node) -> Type:
        raise NotImplementedError("TypeInferencer only handles expressions, not patterns")
    
    def visit_identifier_pattern(self, node) -> Type:
        raise NotImplementedError("TypeInferencer only handles expressions, not patterns")
    
    def visit_wildcard_pattern(self, node) -> Type:
        raise NotImplementedError("TypeInferencer only handles expressions, not patterns")
    
    def visit_tuple_pattern(self, node) -> Type:
        raise NotImplementedError("TypeInferencer only handles expressions, not patterns")
    
    def visit_array_pattern(self, node) -> Type:
        raise NotImplementedError("TypeInferencer only handles expressions, not patterns")
    
    def visit_rest_pattern(self, node) -> Type:
        raise NotImplementedError("TypeInferencer only handles expressions, not patterns")
    
    def visit_guard_pattern(self, node) -> Type:
        raise NotImplementedError("TypeInferencer only handles expressions, not patterns")
    
    def _collect_module_functions(self, mod: Any) -> List[FunctionDefIR]:
        """Recursively collect all functions from a module and its submodules."""
        from ..ir.nodes import FunctionDefIR, ModuleIR
        if not isinstance(mod, ModuleIR):
            return []
        result = list(getattr(mod, "functions", None) or [])
        for sub in getattr(mod, "submodules", None) or []:
            result.extend(self._collect_module_functions(sub))
        return result

    def visit_module(self, node) -> Type:
        """Visit module - recurse into functions and submodules."""
        for func in self._collect_module_functions(node):
            func.accept(self)
        return UNKNOWN

