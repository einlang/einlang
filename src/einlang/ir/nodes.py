"""
IR Nodes

Rust Pattern: rustc_hir::Node
Reference: IR_DESIGN.md
"""

from typing import List, Optional, Any, Tuple, Dict, Union, TYPE_CHECKING
from ..shared.source_location import SourceLocation
from ..shared.defid import DefId, assert_defid
if TYPE_CHECKING:
    from ..shared.types import BinaryOp, UnaryOp


class IRNode:
    """
    Base class for all IR nodes.

    Rust Pattern: rustc_hir::Node

    Implementation Alignment: Follows Rust's `rustc_hir::Node` structure:
    - Every node has `Span` (source location) - we use `SourceLocation`
    - Nodes are mutable (can be modified in place by passes)
    - DefId is NOT on the base; add to subclasses that need it (FunctionValueIR, etc.)

    Reference: `rustc_hir::Node` has `span: Span` and optional `hir_id`

    Design: Regular class (not dataclass) to avoid inheritance issues with defaults
    """
    __slots__ = ('location',)

    def __init__(self, location: SourceLocation):
        self.location = location
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        """
        Accept visitor (Rust pattern: visitor pattern).
        
        Rust Pattern: rustc_hir::intravisit::Visitor
        """
        raise NotImplementedError(f"accept() not implemented for {self.__class__.__name__}")

    def _get_all_attributes(self):
        """Get all attribute values for equality/hashing (works with __slots__)."""
        attrs = {}
        # Walk MRO to collect all slots
        for cls in self.__class__.__mro__:
            if hasattr(cls, '__slots__'):
                slots = cls.__slots__
                if isinstance(slots, str):
                    slots = (slots,)
                for slot in slots:
                    if slot not in attrs:
                        attrs[slot] = getattr(self, slot, None)
        return attrs
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._get_all_attributes() == other._get_all_attributes()
    
    def __hash__(self):
        attrs = self._get_all_attributes()
        # CRITICAL: Exclude unhashable attributes (like shape_info, type_info which may be lists/dicts)
        # These are metadata and shouldn't affect hash/equality
        hashable_attrs = {}
        for key, value in attrs.items():
            # Skip metadata attributes that are unhashable
            if key in ('shape_info', 'type_info'):
                continue
            # Convert lists to tuples for hashability
            if isinstance(value, list):
                try:
                    hashable_attrs[key] = tuple(value)
                except TypeError:
                    # If list contains unhashable items, skip this attribute
                    continue
            elif isinstance(value, dict):
                # Convert dict to frozenset of items for hashability
                try:
                    hashable_attrs[key] = frozenset(value.items())
                except TypeError:
                    # If dict contains unhashable items, skip this attribute
                    continue
            else:
                hashable_attrs[key] = value
        return hash(tuple(sorted(hashable_attrs.items())))


class ExpressionIR(IRNode):
    """
    Expression in IR.

    Rust Pattern: rustc_hir::Expr

    Design Pattern: Visitor pattern handles dispatch - no kind field needed
    Regular class (not dataclass) to avoid inheritance issues
    """
    __slots__ = ('type_info', 'shape_info')

    def __init__(self, location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location)
        self.type_info = type_info
        self.shape_info = shape_info


class LiteralIR(ExpressionIR):
    """Literal expression"""
    __slots__ = ('value',)

    def __init__(self, value: Any, location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.value = value
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_literal(self)


class IdentifierIR(ExpressionIR):
    """
    Identifier expression. Variable identity is DefId only (global or local).
    Lookup: get_value(defid) (single scope stack).
    """
    __slots__ = ('name', 'defid')

    def __init__(self, name: str, location: SourceLocation, defid: Optional[DefId] = None,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        assert_defid(defid)
        self.name = name
        self.defid = defid

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_identifier(self)


class IndexVarIR(ExpressionIR):
    """
    Variable index slot (loop variable) in Einstein LHS or reduction.
    Symmetric with IndexRestIR. name + defid + optional range.
    """
    __slots__ = ("name", "defid", "range_ir")

    def __init__(
        self,
        name: str,
        location: SourceLocation,
        defid: Optional[DefId] = None,
        range_ir: Optional["RangeIR"] = None,
        type_info: Optional[Any] = None,
        shape_info: Optional[Any] = None,
    ):
        super().__init__(location, type_info, shape_info)
        assert_defid(defid)
        self.name = name
        self.defid = defid
        self.range_ir = range_ir

    def accept(self, visitor: "IRVisitor[T]") -> "T":
        return visitor.visit_index_var(self)


class IndexRestIR(ExpressionIR):
    """
    Rest index slot in Einstein LHS or reduction (e.g. ..batch).
    Symmetric with IndexVarIR. Resolved in name resolution with DefId.
    """
    __slots__ = ("name", "defid")

    def __init__(
        self,
        name: str,
        location: SourceLocation,
        defid: Optional[DefId] = None,
        type_info: Optional[Any] = None,
        shape_info: Optional[Any] = None,
    ):
        super().__init__(location, type_info, shape_info)
        assert_defid(defid)
        self.name = name
        self.defid = defid

    def accept(self, visitor: "IRVisitor[T]") -> "T":
        return visitor.visit_index_rest(self)


class BinaryOpIR(ExpressionIR):
    """Binary operation. Rust: expressions have no DefId."""
    __slots__ = ('operator', 'left', 'right')

    def __init__(self, operator: "BinaryOp", left: ExpressionIR, right: ExpressionIR,
                 location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.operator = operator
        self.left = left
        self.right = right
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_binary_op(self)


class UnaryOpIR(ExpressionIR):
    """Unary operation"""
    __slots__ = ('operator', 'operand')

    def __init__(self, operator: "UnaryOp", operand: ExpressionIR, location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.operator = operator
        self.operand = operand
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_unary_op(self)


class RectangularAccessIR(ExpressionIR):
    """
    Rectangular array access: A[i, j] (multi-dimensional indexing)
    
    Used for rectangular types (regular tensors), supports Einstein notation.
    """
    __slots__ = ('array', 'indices')
    
    def __init__(self, array: ExpressionIR, indices: List[ExpressionIR], location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.array = array
        self.indices = indices
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_rectangular_access(self)


class JaggedAccessIR(ExpressionIR):
    """
    Jagged array access: A[i][j] (chained indexing)
    Used for jagged types (ragged arrays), does NOT support Einstein notation.
    """
    __slots__ = ('base', 'index_chain')

    def __init__(self, base: ExpressionIR, index_chain: List[ExpressionIR], location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None,
                 **kwargs: Any):
        super().__init__(location, type_info, shape_info, **kwargs)
        self.base = base
        self.index_chain = index_chain
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_jagged_access(self)


class BlockExpressionIR(ExpressionIR):
    """
    Block expression { statements; final_expr }
    
    Rust Pattern: rustc_hir::Block - blocks have statements and optional final expression
    
    Semantics:
    - statements: List of statements (executed for side effects)
    - final_expr: Optional final expression (block's return value)
    - If final_expr is None, block returns unit type ()
    - Creates new scope for variable bindings
    """
    __slots__ = ('statements', 'final_expr')
    
    def __init__(self, statements: List[ExpressionIR], location: SourceLocation,
                 final_expr: Optional[ExpressionIR] = None,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.statements = statements
        self.final_expr = final_expr
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_block_expression(self)


class IfExpressionIR(ExpressionIR):
    """If expression"""
    __slots__ = ('condition', 'then_expr', 'else_expr')

    def __init__(self, condition: ExpressionIR, then_expr: ExpressionIR,
                 location: SourceLocation, else_expr: Optional[ExpressionIR] = None,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.condition = condition
        self.then_expr = then_expr
        self.else_expr = else_expr
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_if_expression(self)


class LambdaIR(ExpressionIR):
    """Lambda expression (rvalue). No defid; closure identity is at use/call site."""
    __slots__ = ('parameters', 'body')

    def __init__(self, parameters: List['ParameterIR'], body: ExpressionIR,
                 location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.parameters = parameters
        self.body = body

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_lambda(self)


class FunctionCallIR(ExpressionIR):
    """Function call. Uses function_defid for callee (DefId reference). Expression has no DefId."""
    __slots__ = ('function_name', 'function_defid', 'arguments', 'module_path', 'callee_expr')

    def __init__(self, function_name: str, location: SourceLocation,
                 function_defid: Optional[DefId] = None,
                 arguments: Optional[List[ExpressionIR]] = None,
                 module_path: Optional[Tuple[str, ...]] = None,
                 callee_expr: Optional[ExpressionIR] = None,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        assert_defid(function_defid)
        self.function_name = function_name
        self.function_defid = function_defid
        self.arguments = arguments if arguments is not None else []
        self.module_path = module_path  # For Python module calls (e.g., ('math',) for math::sqrt)
        self.callee_expr = callee_expr  # Inline callable, e.g. LambdaIR for (|x| x+1)(5)

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_function_call(self)


class RangeIR(ExpressionIR):
    """Range expression: start..end (exclusive) or start..=end (inclusive)"""
    __slots__ = ('start', 'end', 'inclusive')

    def __init__(self, start: ExpressionIR, end: ExpressionIR, location: SourceLocation,
                 inclusive: bool = False,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.start = start
        self.end = end
        self.inclusive = inclusive
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_range(self)


class ArrayComprehensionIR(ExpressionIR):
    """
    Array comprehension: [expr | var in range, ...]
    
    Supports multiple variables for cartesian product:
    - [expr | i in 0..3, j in 0..3] → cartesian product (flat array)
    - [[expr | j in 0..3] | i in 0..3] → nested (jagged array, body is another comprehension)
    
    Semantics:
    - Multiple variables in ONE comprehension = cartesian product = flat array
    - Nested comprehensions (body IS a comprehension) = jagged array
    """
    __slots__ = ('body', 'variables', 'variable_defids', 'ranges', 'constraints')
    
    def __init__(self, body: ExpressionIR, 
                 variables: Union[str, List[str]],  # Support single var (backward compat) or list
                 ranges: Union[RangeIR, List[RangeIR]],  # Support single range or list
                 location: SourceLocation, 
                 constraints: Optional[List[ExpressionIR]] = None,
                 variable_defids: Optional[Union[DefId, List[DefId]]] = None,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.body = body
        
        # Normalize to lists for multiple variables (backward compatible with single variable)
        if isinstance(variables, str):
            self.variables = [variables]
            # Single range/iterable: always store as list of one (ranges can be RangeIR or array expr e.g. ArrayLiteralIR)
            self.ranges = [ranges] if not isinstance(ranges, list) else ranges
            self.variable_defids = [variable_defids] if variable_defids and not isinstance(variable_defids, list) else (variable_defids if variable_defids else [])
        else:
            self.variables = variables
            self.ranges = ranges if isinstance(ranges, list) else [ranges]
            self.variable_defids = variable_defids if isinstance(variable_defids, list) else ([variable_defids] if variable_defids else [])
        
        # Ensure lists are same length
        if len(self.variables) != len(self.ranges):
            raise ValueError(f"Mismatch: {len(self.variables)} variables but {len(self.ranges)} ranges")
        if self.variable_defids and len(self.variable_defids) != len(self.variables):
            while len(self.variable_defids) < len(self.variables):
                self.variable_defids.append(None)
        for d in self.variable_defids:
            assert_defid(d)
        self.constraints = constraints if constraints is not None else []
    
    # Backward compatibility properties
    @property
    def variable(self) -> str:
        """Backward compatibility: return first variable if single, else raise"""
        if len(self.variables) == 1:
            return self.variables[0]
        raise AttributeError("ArrayComprehensionIR has multiple variables, use .variables")
    
    @property
    def variable_defid(self) -> Optional[DefId]:
        """Backward compatibility: return first variable_defid if single, else raise"""
        if len(self.variables) == 1:
            return self.variable_defids[0] if self.variable_defids else None
        raise AttributeError("ArrayComprehensionIR has multiple variables, use .variable_defids")
    
    @property
    def range_expr(self) -> RangeIR:
        """Backward compatibility: return first range if single, else raise"""
        if len(self.ranges) == 1:
            return self.ranges[0]
        raise AttributeError("ArrayComprehensionIR has multiple ranges, use .ranges")
    
    @property
    def expr(self) -> ExpressionIR:
        """Backward compatibility: return body (expr is alias for body)"""
        return self.body
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_array_comprehension(self)


class ArrayLiteralIR(ExpressionIR):
    """Array literal: [1, 2, 3]"""
    __slots__ = ('elements',)

    def __init__(self, elements: List[ExpressionIR], location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None,
                 **kwargs: Any):
        super().__init__(location, type_info, shape_info, **kwargs)
        self.elements = elements
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_array_literal(self)


class TupleExpressionIR(ExpressionIR):
    """Tuple expression: (a, b, c)"""
    __slots__ = ('elements',)
    
    def __init__(self, elements: List[ExpressionIR], location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None,
                 **kwargs: Any):
        super().__init__(location, type_info, shape_info, **kwargs)
        self.elements = elements

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_tuple_expression(self)


class TupleAccessIR(ExpressionIR):
    """Tuple access: tuple.0, tuple.1"""
    __slots__ = ('tuple_expr', 'index')

    def __init__(self, tuple_expr: ExpressionIR, index: int, location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None,
                 **kwargs: Any):
        super().__init__(location, type_info, shape_info, **kwargs)
        self.tuple_expr = tuple_expr
        self.index = index
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_tuple_access(self)


class InterpolatedStringIR(ExpressionIR):
    """Interpolated string: "Hello {name}" """
    __slots__ = ('parts',)
    
    def __init__(self, parts: List[Union[str, ExpressionIR]], location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None,
                 **kwargs: Any):
        super().__init__(location, type_info, shape_info, **kwargs)
        self.parts = parts
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_interpolated_string(self)


class CastExpressionIR(ExpressionIR):
    """Type cast: x as i64"""
    __slots__ = ('expr', 'target_type')

    def __init__(self, expr: ExpressionIR, target_type: Any, location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None,
                 **kwargs: Any):
        super().__init__(location, type_info, shape_info, **kwargs)
        self.expr = expr
        self.target_type = target_type
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_cast_expression(self)


class MemberAccessIR(ExpressionIR):
    """Member access: arr.size, arr.shape"""
    __slots__ = ('object', 'member')
    
    def __init__(self, object: ExpressionIR, member: str, location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None,
                 **kwargs: Any):
        super().__init__(location, type_info, shape_info, **kwargs)
        self.object = object
        self.member = member
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_member_access(self)


class TryExpressionIR(ExpressionIR):
    """Try expression: try expr"""
    __slots__ = ('operand',)

    def __init__(self, operand: ExpressionIR, location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None,
                 **kwargs: Any):
        super().__init__(location, type_info, shape_info, **kwargs)
        self.operand = operand
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_try_expression(self)


class ReductionExpressionIR(ExpressionIR):
    """Reduction expression: sum[i](A[i]). Loop vars are IndexVarIR or IdentifierIR (each has .name and .defid). No defid on node; identity comes from binding when reduction is RHS."""
    __slots__ = ('operation', 'loop_vars', 'body', 'where_clause', 'loop_var_ranges')

    def __init__(self, operation: str, loop_vars: Optional[List[Union['IndexVarIR', 'IdentifierIR']]], body: ExpressionIR,
                 location: SourceLocation, where_clause: Optional['WhereClauseIR'] = None,
                 loop_var_ranges: Optional[Dict[DefId, 'RangeIR']] = None,
                 type_info: Optional[Any] = None,
                 shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.operation = operation
        self.body = body
        self.where_clause = where_clause
        self.loop_var_ranges = loop_var_ranges if loop_var_ranges is not None else {}
        self.loop_vars = loop_vars if loop_vars is not None else []

    @property
    def loop_var_names(self) -> List[str]:
        return [getattr(ident, 'name', '') for ident in self.loop_vars]

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_reduction_expression(self)


class WhereExpressionIR(ExpressionIR):
    """Where expression: expr where constraint"""
    __slots__ = ('expr', 'constraints')
    
    def __init__(self, expr: ExpressionIR, constraints: List[ExpressionIR], location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.expr = expr
        self.constraints = constraints
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_where_expression(self)


class ArrowExpressionIR(ExpressionIR):
    """Arrow expression: f >>> g >>> h"""
    __slots__ = ('components', 'operator')
    
    def __init__(self, components: List[ExpressionIR], operator: str, location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.components = components
        self.operator = operator  # ">>>", "***", "&&&", "|||"
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_arrow_expression(self)


class PipelineExpressionIR(ExpressionIR):
    """Pipeline expression: x |> f |> g"""
    __slots__ = ('left', 'right', 'operator')
    
    def __init__(self, left: ExpressionIR, right: ExpressionIR, location: SourceLocation,
                 operator: str = "|>",
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.left = left
        self.right = right
        self.operator = operator
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_pipeline_expression(self)


class BuiltinCallIR(ExpressionIR):
    """Builtin function call. Rust: references builtin definition (DefId)."""
    __slots__ = ('builtin_name', 'args', 'defid')

    def __init__(self, builtin_name: str, args: List[ExpressionIR], location: SourceLocation,
                 defid: Optional[DefId] = None, type_info: Optional[Any] = None,
                 shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        assert_defid(defid)
        self.defid = defid
        self.builtin_name = builtin_name
        self.args = args

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_builtin_call(self)


class FunctionRefIR(ExpressionIR):
    """Function reference (first-class functions). Rust: references function DefId."""
    __slots__ = ('function_defid',)

    def __init__(self, function_defid: DefId, location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        assert_defid(function_defid, allow_none=False)
        self.function_defid = function_defid

    @property
    def defid(self) -> DefId:
        """Backend uses expr.defid for def_table; same as function_defid."""
        return self.function_defid
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_function_ref(self)



class ParameterIR(IRNode):
    """Function parameter. Variable identity is DefId."""
    __slots__ = ('name', 'param_type', 'defid')

    def __init__(self, name: str, location: SourceLocation, param_type: Optional[Any] = None,
                 defid: Optional[DefId] = None):
        super().__init__(location)
        assert_defid(defid)
        self.defid = defid
        self.name = name
        self.param_type = param_type


class FunctionDefIR(IRNode):
    """Function definition in IR"""
    __slots__ = ('name', 'parameters', 'return_type', 'body', 'shape_info', 'type_info', 'defid', '_is_partially_specialized', '_generic_defid')

    def __init__(self, name: str, parameters: List[ParameterIR], body: ExpressionIR,
                 location: SourceLocation, return_type: Optional[Any] = None,
                 shape_info: Optional[Any] = None, type_info: Optional[Any] = None,
                 defid: Optional[DefId] = None, _is_partially_specialized: bool = False,
                 _generic_defid: Optional[DefId] = None):
        super().__init__(location)
        assert_defid(defid)
        assert_defid(_generic_defid)
        self.defid = defid
        self.name = name
        self.parameters = parameters
        self.return_type = return_type
        self.body = body
        self.shape_info = shape_info
        self.type_info = type_info
        self._is_partially_specialized = _is_partially_specialized
        self._generic_defid = _generic_defid
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_function_def(self)


class ConstantDefIR(IRNode):
    """Constant definition in IR. Rust: DefId for items."""
    __slots__ = ('name', 'value', 'type_info', 'defid')

    def __init__(self, name: str, value: ExpressionIR, location: SourceLocation,
                 type_info: Optional[Any] = None, defid: Optional[DefId] = None):
        super().__init__(location)
        assert_defid(defid)
        self.defid = defid
        self.name = name
        self.value = value
        self.type_info = type_info

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_constant_def(self)


class ProgramIR:
    """
    Complete program in IR.
    
    Rust Pattern: rustc_hir::Crate
    """
    __slots__ = ('modules', 'functions', 'constants', 'statements', 'source_files', 'defid_to_name')
    
    def __init__(self, modules: List['ModuleIR'], functions: List[FunctionDefIR],
                 constants: List[ConstantDefIR], statements: List[ExpressionIR],
                 source_files: dict, defid_to_name: Optional[Dict[DefId, str]] = None):
        self.modules = modules
        self.functions = functions
        self.constants = constants
        self.statements = statements  # Top-level statements (variable declarations, etc.)
        self.source_files = source_files
        d2n = defid_to_name or {}
        for k in d2n:
            assert_defid(k, allow_none=False)
        self.defid_to_name = d2n

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        """Accept visitor (Rust pattern: visitor pattern)."""
        return visitor.visit_program(self)


class ModuleIR(IRNode):
    """Module in IR"""
    __slots__ = ('path', 'functions', 'constants', 'submodules', 'defid')

    def __init__(self, path: Tuple[str, ...], location: SourceLocation,
                 functions: List[FunctionDefIR], constants: List[ConstantDefIR],
                 submodules: List['ModuleIR'], defid: Optional[DefId] = None):
        super().__init__(location)
        assert_defid(defid)
        self.defid = defid
        self.path = path
        self.functions = functions
        self.constants = constants
        self.submodules = submodules
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_module(self)


# Pattern IR Nodes (for Match expressions)

class PatternIR(IRNode):
    """Base class for pattern IR nodes. Only IdentifierPatternIR has defid (for bindings)."""
    __slots__ = ()

    def __init__(self, location: SourceLocation):
        super().__init__(location)


class LiteralPatternIR(PatternIR):
    """Literal pattern: 42"""
    __slots__ = ('value',)

    def __init__(self, value: Union[int, float, bool, str], location: SourceLocation):
        super().__init__(location)
        self.value = value
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_literal_pattern(self)


class IdentifierPatternIR(PatternIR):
    """Identifier pattern: x (binds value to variable). Only pattern type with defid."""
    __slots__ = ('name', 'defid')

    def __init__(self, name: str, location: SourceLocation, defid: Optional[DefId] = None):
        super().__init__(location)
        self.name = name
        assert_defid(defid)
        self.defid = defid
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_identifier_pattern(self)


class WildcardPatternIR(PatternIR):
    """Wildcard pattern: _ (matches anything, no binding)"""
    __slots__ = ()

    def __init__(self, location: SourceLocation):
        super().__init__(location)
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_wildcard_pattern(self)


class TuplePatternIR(PatternIR):
    """Tuple pattern: (a, b, c)"""
    __slots__ = ('patterns',)

    def __init__(self, patterns: List[PatternIR], location: SourceLocation):
        super().__init__(location)
        self.patterns = patterns
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_tuple_pattern(self)


class ArrayPatternIR(PatternIR):
    """Array pattern: [a, b, ..rest]"""
    __slots__ = ('patterns',)

    def __init__(self, patterns: List[PatternIR], location: SourceLocation):
        super().__init__(location)
        self.patterns = patterns
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_array_pattern(self)


class RestPatternIR(PatternIR):
    """Rest pattern: ..rest"""
    __slots__ = ('pattern',)

    def __init__(self, pattern: IdentifierPatternIR, location: SourceLocation):
        super().__init__(location)
        self.pattern = pattern
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_rest_pattern(self)


class GuardPatternIR(PatternIR):
    """Guard pattern: x where x > 0"""
    __slots__ = ('inner_pattern', 'guard_expr')

    def __init__(self, inner_pattern: PatternIR, guard_expr: ExpressionIR, location: SourceLocation):
        super().__init__(location)
        self.inner_pattern = inner_pattern
        self.guard_expr = guard_expr
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_guard_pattern(self)


class OrPatternIR(PatternIR):
    """Or pattern: pat1 | pat2 | ..."""
    __slots__ = ('alternatives',)

    def __init__(self, alternatives: List[PatternIR], location: SourceLocation):
        super().__init__(location)
        self.alternatives = alternatives
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_or_pattern(self)


class ConstructorPatternIR(PatternIR):
    """Constructor pattern: Some(x), Circle(r)"""
    __slots__ = ('constructor_name', 'patterns', 'is_struct_literal')

    def __init__(self, constructor_name: str, patterns: List[PatternIR],
                 is_struct_literal: bool = False, location: SourceLocation = None):
        super().__init__(location or SourceLocation("", 0, 0))
        self.constructor_name = constructor_name
        self.patterns = patterns
        self.is_struct_literal = is_struct_literal
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_constructor_pattern(self)


class BindingPatternIR(PatternIR):
    """Binding pattern: identifier_pattern @ inner_pattern (name binding + sub-pattern)."""
    __slots__ = ('identifier_pattern', 'inner_pattern')

    def __init__(self, identifier_pattern: 'IdentifierPatternIR', inner_pattern: PatternIR, location: SourceLocation):
        super().__init__(location)
        self.identifier_pattern = identifier_pattern
        self.inner_pattern = inner_pattern

    @property
    def name(self) -> str:
        return self.identifier_pattern.name

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_binding_pattern(self)


class RangePatternIR(PatternIR):
    """Range pattern: start..end (exclusive) or start..=end (inclusive)"""
    __slots__ = ('start', 'end', 'inclusive')

    def __init__(self, start: Union[int, float], end: Union[int, float], inclusive: bool,
                 location: SourceLocation):
        super().__init__(location)
        self.start = start
        self.end = end
        self.inclusive = inclusive
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_range_pattern(self)


class MatchArmIR:
    """Match arm: pattern + body"""
    __slots__ = ('pattern', 'body')
    
    def __init__(self, pattern: PatternIR, body: ExpressionIR):
        self.pattern = pattern
        self.body = body


class MatchExpressionIR(ExpressionIR):
    """Match expression: match scrutinee { arms }"""
    __slots__ = ('scrutinee', 'arms')
    
    def __init__(self, scrutinee: ExpressionIR, arms: List[MatchArmIR], location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.scrutinee = scrutinee
        self.arms = arms
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_match_expression(self)


# Where Clause IR

class WhereClauseIR:
    """Where clause: constraints for filtering. ranges is keyed by DefId (index variable)."""
    __slots__ = ('constraints', 'ranges')
    
    def __init__(self, constraints: List[ExpressionIR], ranges: Optional[Dict[DefId, Any]] = None):
        self.constraints = constraints
        self.ranges = ranges if ranges is not None else {}


# Lowered iteration structures (aligned with LoopStructure + shared iteration shape)
# LoweredIteration has body, loops, bindings, guards, reduction_ranges, shape, element_type

class LoopStructure:
    """
    Loop iteration: for variable in iterable ().
    iterable is an expression (RangeIR, LiteralIR(range), etc.).
    variable is IndexVarIR for index loops (defid only; no name-based lookup) or IdentifierIR for reduction.
    """
    __slots__ = ('variable', 'iterable')

    def __init__(self, variable: "Union[IdentifierIR, IndexVarIR]", iterable: ExpressionIR):
        self.variable = variable
        self.iterable = iterable

    def __str__(self) -> str:
        return f"{self.variable.name} in {self.iterable}"


class BindingIR(IRNode):
    """Canonical binding (name = expr). Only LHS (defid/name) is the reference; expr is rvalue."""
    __slots__ = ('name', 'expr', 'type_info', 'defid')

    def __init__(self, name: str, expr: Any, type_info: Optional[Any] = None,
                 location: Optional[SourceLocation] = None,
                 defid: Optional[DefId] = None):
        super().__init__(location or SourceLocation('', 0, 0))
        assert_defid(defid)
        self.name = name
        self.expr = expr
        self.type_info = type_info
        self.defid = defid

    @property
    def pattern(self) -> str:
        return self.name

    @property
    def value(self) -> Any:
        return self.expr

    def get_defid_binding(self) -> Optional[tuple]:
        if self.defid is not None:
            return (self.defid, self.expr)
        return None

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_binding(self)

    def __str__(self) -> str:
        type_str = f": {self.type_info}" if self.type_info else ""
        return f"{self.name}{type_str} = {self.expr}"


class GuardCondition:
    """Runtime guard condition"""
    __slots__ = ('condition',)
    
    def __init__(self, condition: ExpressionIR):
        self.condition = condition
    
    def __str__(self) -> str:
        """Human-readable representation"""
        return f"guard {self.condition}"


class LoweredIteration:
    """
    Unified lowered representation for all iteration constructs.
    
    Used by EinsteinDeclarationIR via composition.
    
    Provides shared iteration structure:
    - body: The expression being iterated
    - loops: Index iteration structure
    - bindings: Local variable bindings (where clause)
    - guards: Runtime conditions (filters)
    - reduction_ranges: Reduction variable ranges (sum, prod, etc.) keyed by DefId.
    - shape/element_type: Type information
    """
    __slots__ = ('body', 'loops', 'bindings', 'guards', 'reduction_ranges', 'shape', 'element_type')
    
    def __init__(
        self,
        body: ExpressionIR,
        loops: Optional[List[LoopStructure]] = None,
        bindings: Optional[List['BindingIR']] = None,
        guards: Optional[List[GuardCondition]] = None,
        reduction_ranges: Optional[Dict[DefId, LoopStructure]] = None,
        shape: Optional[List[ExpressionIR]] = None,
        element_type: Optional[Any] = None
    ):
        self.body = body
        self.loops = loops if loops is not None else []
        self.bindings = bindings if bindings is not None else []
        self.guards = guards if guards is not None else []
        self.reduction_ranges = reduction_ranges if reduction_ranges is not None else {}
        self.shape = shape
        self.element_type = element_type
    
    def is_empty(self) -> bool:
        """Check if lowered iteration is empty"""
        return not self.loops and not self.bindings and not self.guards and not self.reduction_ranges
    
    def is_rectangular(self) -> bool:
        """Check if all loops are rectangular (no conditions)"""
        # We don't have loop.condition, so all loops are rectangular
        return True
    
    def __str__(self) -> str:
        """Human-readable representation"""
        parts = [f"body: {self.body}"]
        if self.loops:
            parts.append(f"loops: {[str(l) for l in self.loops]}")
        if self.reduction_ranges:
            parts.append(f"reduction_ranges: {list(self.reduction_ranges.keys())}")
        if self.bindings:
            parts.append(f"bindings: {[str(b) for b in self.bindings]}")
        if self.guards:
            parts.append(f"guards: {[str(g) for g in self.guards]}")
        if self.shape:
            parts.append(f"shape: {self.shape}")
        return "LoweredIteration(" + ", ".join(parts) + ")"


class LoweredEinsteinClauseIR:
    """Single lowered Einstein clause (body, loops, bindings, guards, indices). reduction_ranges keyed by DefId."""
    __slots__ = ('body', 'loops', 'reduction_ranges', 'bindings', 'guards', 'indices')

    def __init__(
        self,
        body: ExpressionIR,
        loops: Optional[List[LoopStructure]] = None,
        reduction_ranges: Optional[Dict[DefId, LoopStructure]] = None,
        bindings: Optional[List['BindingIR']] = None,
        guards: Optional[List[GuardCondition]] = None,
        indices: Optional[List[Any]] = None,
    ):
        self.body = body
        self.loops = loops if loops is not None else []
        self.reduction_ranges = reduction_ranges if reduction_ranges is not None else {}
        self.bindings = bindings if bindings is not None else []
        self.guards = guards if guards is not None else []
        self.indices = indices if indices is not None else []
    
    def __str__(self) -> str:
        parts = [f"body: {self.body}"]
        if self.loops:
            parts.append(f"loops: {[str(l) for l in self.loops]}")
        return "LoweredEinsteinClauseIR(" + ", ".join(parts) + ")"

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_lowered_einstein_clause(self)


class LoweredEinsteinIR:
    """Lowered Einstein declaration: one tensor, one shape. All clauses write to the same memory (same shape)."""
    __slots__ = ('items', 'shape', 'element_type')

    def __init__(self, items: List['LoweredEinsteinClauseIR'],
                 shape: Optional[List[ExpressionIR]] = None,
                 element_type: Optional[Any] = None):
        self.items = items
        self.shape = shape
        self.element_type = element_type

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_lowered_einstein(self)


class LoweredReductionIR(ExpressionIR):
    """
    Lowered reduction (LoweredIteration shape). Replaces ReductionExpressionIR.
    body, operation, loops, bindings, guards; reduction_ranges derived from loops.
    """
    __slots__ = ('body', 'operation', 'loops', 'bindings', 'guards')
    
    def __init__(
        self,
        body: ExpressionIR,
        operation: str,
        loops: Optional[List[LoopStructure]] = None,
        bindings: Optional[List['BindingIR']] = None,
        guards: Optional[List[GuardCondition]] = None,
        location: Optional[SourceLocation] = None,
        type_info: Optional[Any] = None,
        shape_info: Optional[Any] = None,
    ):
        loc = location or getattr(body, 'location', None)
        if loc is None:
            loc = SourceLocation(file='', line=0, column=0)
        super().__init__(loc, type_info=type_info, shape_info=shape_info)
        self.body = body
        self.operation = operation
        self.loops = loops if loops is not None else []
        self.bindings = bindings if bindings is not None else []
        self.guards = guards if guards is not None else []
    
    @property
    def reduction_ranges(self) -> Dict[DefId, LoopStructure]:
        """Reduction variable ranges keyed by variable DefId (for compatibility with execute_reduction_with_loops which uses .values())."""
        result: Dict[DefId, LoopStructure] = {}
        for loop in self.loops:
            d = getattr(loop.variable, 'defid', None)
            if d is not None:
                result[d] = loop
        return result
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_lowered_reduction(self)
    
    def __str__(self) -> str:
        parts = [f"body: {self.body}"]
        if self.loops:
            parts.append(f"loops: {[str(l) for l in self.loops]}")
        return "LoweredReductionIR(" + ", ".join(parts) + ")"


class LoweredComprehensionIR(ExpressionIR):
    """
    Lowered array comprehension (LoweredIteration shape). Replaces ArrayComprehensionIR.
    body, loops, bindings (y = expr), guards (filters).
    """
    __slots__ = ('body', 'loops', 'bindings', 'guards')
    
    def __init__(
        self,
        body: ExpressionIR,
        loops: Optional[List[LoopStructure]] = None,
        bindings: Optional[List['BindingIR']] = None,
        guards: Optional[List[GuardCondition]] = None,
        location: Optional[SourceLocation] = None,
        type_info: Optional[Any] = None,
        shape_info: Optional[Any] = None,
    ):
        loc = location or getattr(body, 'location', None)
        if loc is None:
            loc = SourceLocation(file='', line=0, column=0)
        super().__init__(loc, type_info=type_info, shape_info=shape_info)
        self.body = body
        self.loops = loops if loops is not None else []
        self.bindings = bindings if bindings is not None else []
        self.guards = guards if guards is not None else []
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_lowered_comprehension(self)
    
    def __str__(self) -> str:
        parts = [f"body: {self.body}"]
        if self.loops:
            parts.append(f"loops: {[str(l) for l in self.loops]}")
        return "LoweredComprehensionIR(" + ", ".join(parts) + ")"


# Variable Declaration IR (Statement)
# Under the hood shares BindingIR shape so execution/lowering can treat both the same.

class VariableDeclarationIR(IRNode):
    """
    Variable declaration: let x = expr  or  let x: i32 = expr

    Rust Pattern: rustc_hir::Local (let statement). Lookup by defid.
    """
    __slots__ = ('_binding',)

    def __init__(self, pattern: str, value: ExpressionIR,
                 type_annotation: Optional[Any] = None,
                 location: SourceLocation = None,
                 defid: Optional[DefId] = None):
        loc = location if location is not None else SourceLocation('', 0, 0)
        super().__init__(loc)
        self._binding = BindingIR(
            name=pattern,
            expr=value,
            type_info=type_annotation,
            location=loc,
            defid=defid
        )
    
    @property
    def pattern(self) -> str:
        return self._binding.name
    
    @pattern.setter
    def pattern(self, v: str) -> None:
        self._binding.name = v
    
    @property
    def value(self) -> ExpressionIR:
        return self._binding.expr
    
    @value.setter
    def value(self, v: ExpressionIR) -> None:
        self._binding.expr = v
    
    @property
    def type_annotation(self) -> Optional[Any]:
        return self._binding.type_info
    
    @type_annotation.setter
    def type_annotation(self, v: Optional[Any]) -> None:
        self._binding.type_info = v
    
    @property
    def defid(self) -> Optional[DefId]:
        return getattr(self._binding, 'defid', None)

    def to_binding(self) -> 'BindingIR':
        """Same shape as where-clause bindings; execution can treat uniformly."""
        return self._binding

    def get_defid_binding(self):
        """Return (defid, value) for scope registration, or None. Lets block register bindings without isinstance."""
        did = self.defid
        if did is not None:
            return (did, self.value)
        return None

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_variable_declaration(self)


# Einstein IR: one clause (indices, value, where_clause). Declaration holds list of these.

class EinsteinIR(IRNode):
    """One Einstein clause. Holds indices, value, where_clause, and variable_ranges.
    Ranges are only on the clause (not on EinsteinDeclarationIR). variable_ranges is keyed by DefId.
    Precision (element_type) is on the declaration; runtime receives it and passes it in."""
    __slots__ = ('indices', 'value', 'where_clause', 'variable_ranges')

    def __init__(self, indices: List[ExpressionIR], value: ExpressionIR,
                 location: SourceLocation, where_clause: Optional[WhereClauseIR] = None,
                 variable_ranges: Optional[Dict[DefId, Any]] = None):
        super().__init__(location)
        self.indices = indices
        self.value = value
        self.where_clause = where_clause
        self.variable_ranges = variable_ranges if variable_ranges is not None else {}

    @property
    def loop_vars(self) -> List[str]:
        """Derive loop variable names from indices (IndexVarIR, IndexRestIR; legacy IdentifierIR)."""
        out: List[str] = []
        for idx in self.indices or []:
            if isinstance(idx, (IndexVarIR, IndexRestIR, IdentifierIR)) and getattr(idx, "name", None):
                out.append(idx.name)
            elif isinstance(idx, (list, tuple)):
                for sub in idx:
                    if isinstance(sub, (IndexVarIR, IndexRestIR, IdentifierIR)) and getattr(sub, "name", None):
                        out.append(sub.name)
        return out

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_einstein(self)


class EinsteinDeclarationIR(IRNode):
    """Einstein declaration: name, defid, clauses, and type info only.
    No ranges on the declaration: only types (element_type, precision) and shapes (shape).
    Ranges (variable_ranges) live on each clause (EinsteinIR)."""
    __slots__ = ('name', 'defid', 'clauses', 'shape', 'element_type')

    def __init__(self, name: str, location: SourceLocation, defid: Optional[DefId] = None,
                 clauses: Optional[List[EinsteinIR]] = None,
                 shape: Optional[List[ExpressionIR]] = None,
                 element_type: Optional[Any] = None):
        super().__init__(location)
        assert_defid(defid)
        self.name = name
        self.defid = defid
        self.clauses = clauses if clauses is not None else []
        self.shape = shape
        self.element_type = element_type

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_einstein_declaration(self)


# Type variable for visitor pattern
from typing import TypeVar, Generic
from abc import ABC, abstractmethod
T = TypeVar('T')


class IRVisitor(ABC, Generic[T]):
    """
    Visitor for IR nodes (no isinstance needed).
    
    Rust Pattern: rustc_hir::intravisit::Visitor
    """
    
    @abstractmethod
    def visit_literal(self, node: LiteralIR) -> T:
        """Visit literal expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_identifier(self, node: IdentifierIR) -> T:
        """Visit identifier expression"""
        raise NotImplementedError

    def visit_index_var(self, node: "IndexVarIR") -> T:
        """Visit variable index slot. Default: no-op (subclasses recurse into node.range if needed)."""
        return None  # type: ignore[return-value]

    def visit_index_rest(self, node: "IndexRestIR") -> T:
        """Visit rest index slot (e.g. ..batch in Einstein indices). Default: no-op."""
        return None  # type: ignore[return-value]

    @abstractmethod
    def visit_binary_op(self, node: BinaryOpIR) -> T:
        """Visit binary operation"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_function_call(self, node: FunctionCallIR) -> T:
        """Visit function call"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_function_def(self, node: FunctionDefIR) -> T:
        """Visit function definition"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_constant_def(self, node: ConstantDefIR) -> T:
        """Visit constant definition"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_rectangular_access(self, node: RectangularAccessIR) -> T:
        """Visit rectangular array access"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_jagged_access(self, node: JaggedAccessIR) -> T:
        """Visit jagged array access"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_block_expression(self, node: BlockExpressionIR) -> T:
        """Visit block expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_if_expression(self, node: IfExpressionIR) -> T:
        """Visit if expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_lambda(self, node: LambdaIR) -> T:
        """Visit lambda expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_unary_op(self, node: UnaryOpIR) -> T:
        """Visit unary operation"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_range(self, node: RangeIR) -> T:
        """Visit range expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_array_comprehension(self, node: ArrayComprehensionIR) -> T:
        """Visit array comprehension"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_module(self, node: 'ModuleIR') -> T:
        """Visit module"""
        raise NotImplementedError
    
    # Missing expression visitors
    @abstractmethod
    def visit_array_literal(self, node: ArrayLiteralIR) -> T:
        """Visit array literal"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_tuple_expression(self, node: TupleExpressionIR) -> T:
        """Visit tuple expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_tuple_access(self, node: TupleAccessIR) -> T:
        """Visit tuple access"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_interpolated_string(self, node: InterpolatedStringIR) -> T:
        """Visit interpolated string"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_cast_expression(self, node: CastExpressionIR) -> T:
        """Visit cast expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_member_access(self, node: MemberAccessIR) -> T:
        """Visit member access"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_try_expression(self, node: TryExpressionIR) -> T:
        """Visit try expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_match_expression(self, node: MatchExpressionIR) -> T:
        """Visit match expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_reduction_expression(self, node: ReductionExpressionIR) -> T:
        """Visit reduction expression"""
        raise NotImplementedError
    
    def visit_lowered_reduction(self, node: 'LoweredReductionIR') -> T:
        """Visit lowered reduction (replaces ReductionExpressionIR in tree). Default: recurse into body."""
        return node.body.accept(self)

    def visit_lowered_comprehension(self, node: 'LoweredComprehensionIR') -> T:
        """Visit lowered array comprehension (replaces ArrayComprehensionIR in tree). Default: recurse into body."""
        return node.body.accept(self)

    def visit_lowered_einstein_clause(self, node: 'LoweredEinsteinClauseIR') -> T:
        """Visit single lowered Einstein clause. Default: recurse into body."""
        return node.body.accept(self)

    def visit_lowered_einstein(self, node: 'LoweredEinsteinIR') -> T:
        """Visit lowered Einstein (multiple clauses). Default: recurse into all items, return last result."""
        if not node.items:
            return None  # type: ignore[return-value]
        result = node.items[0].accept(self)
        for item in node.items[1:]:
            result = item.accept(self)
        return result

    @abstractmethod
    def visit_where_expression(self, node: WhereExpressionIR) -> T:
        """Visit where expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_arrow_expression(self, node: ArrowExpressionIR) -> T:
        """Visit arrow expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_pipeline_expression(self, node: PipelineExpressionIR) -> T:
        """Visit pipeline expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_builtin_call(self, node: BuiltinCallIR) -> T:
        """Visit builtin call"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_function_ref(self, node: FunctionRefIR) -> T:
        """Visit function reference"""
        raise NotImplementedError
    
    # Pattern visitors
    @abstractmethod
    def visit_literal_pattern(self, node: LiteralPatternIR) -> T:
        """Visit literal pattern"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_identifier_pattern(self, node: IdentifierPatternIR) -> T:
        """Visit identifier pattern"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_wildcard_pattern(self, node: WildcardPatternIR) -> T:
        """Visit wildcard pattern"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_tuple_pattern(self, node: TuplePatternIR) -> T:
        """Visit tuple pattern"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_array_pattern(self, node: ArrayPatternIR) -> T:
        """Visit array pattern"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_rest_pattern(self, node: RestPatternIR) -> T:
        """Visit rest pattern"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_guard_pattern(self, node: GuardPatternIR) -> T:
        """Visit guard pattern"""
        raise NotImplementedError
    
    def visit_or_pattern(self, node: 'OrPatternIR') -> T:
        """Visit or pattern: pat1 | pat2. Default: visit first alternative."""
        if node.alternatives:
            return node.alternatives[0].accept(self)
        return None  # type: ignore[return-value]
    
    def visit_constructor_pattern(self, node: 'ConstructorPatternIR') -> T:
        """Visit constructor pattern: Some(x). Default: visit sub-patterns."""
        for p in node.patterns:
            p.accept(self)
        return None  # type: ignore[return-value]
    
    def visit_binding_pattern(self, node: 'BindingPatternIR') -> T:
        """Visit binding pattern: name @ pat. Default: visit inner pattern."""
        return node.inner_pattern.accept(self)
    
    def visit_range_pattern(self, node: 'RangePatternIR') -> T:
        """Visit range pattern: start..=end. Default: no-op."""
        return None  # type: ignore[return-value]
    
    # Statement visitors
    @abstractmethod
    def visit_einstein_declaration(self, node: EinsteinDeclarationIR) -> T:
        """Visit Einstein declaration (clauses list). Default: recurse into each clause."""
        raise NotImplementedError

    def visit_einstein(self, node: EinsteinIR) -> T:
        """Visit one Einstein clause. Default: no-op."""
        return None  # type: ignore[return-value]

    def visit_variable_declaration(self, node: 'VariableDeclarationIR') -> T:
        """Visit variable declaration. Default: forward to visit_binding (same shape under the hood)."""
        return self.visit_binding(node.to_binding())

    def visit_binding(self, node: 'BindingIR') -> T:
        """Visit binding (name = expr). Default: no-op (return None)."""
        return None  # type: ignore[return-value]
    
    # Program visitor
    @abstractmethod
    def visit_program(self, node: 'ProgramIR') -> T:
        """Visit program"""
        raise NotImplementedError

