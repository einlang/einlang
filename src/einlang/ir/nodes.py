"""
IR Nodes

Rust Pattern: rustc_hir::Node
Reference: IR_DESIGN.md
"""

from typing import List, Optional, Any, Tuple, Dict, Union, TYPE_CHECKING
from ..shared.source_location import SourceLocation
from ..shared.defid import DefId, assert_defid
if TYPE_CHECKING:
    from ..shared.types import BinaryOp, UnaryOp, ReductionOp


def _t(x: Optional[list]) -> tuple:
    """Convert list (or None) to tuple for immutable slot storage."""
    return tuple(x) if x is not None else ()



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

    def __str__(self) -> str:
        return str(self.value)


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

    def __str__(self) -> str:
        return self.name or '?'


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

    def __str__(self) -> str:
        return self.name or '?'


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

    def __str__(self) -> str:
        return f"..{self.name}" if self.name else ".."


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

    def __str__(self) -> str:
        op = self.operator.value if hasattr(self.operator, 'value') else str(self.operator)
        return f"{self.left} {op} {self.right}"


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

    def __str__(self) -> str:
        op = self.operator.value if hasattr(self.operator, 'value') else str(self.operator)
        sep = ' ' if op.isalpha() else ''
        return f"{op}{sep}{self.operand}"


class DifferentialIR(ExpressionIR):
    """
    Differential of the value of operand (@expr). Same shape and dtype as operand.
    Differentials combine per math (e.g. dz = dx + dy); see AUTODIFF_DESIGN.md.
    """
    __slots__ = ('operand',)

    def __init__(self, operand: ExpressionIR, location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.operand = operand

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_differential(self)

    def __str__(self) -> str:
        return f"@{self.operand}"


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
        self.indices = _t(indices)
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_rectangular_access(self)

    def __str__(self) -> str:
        arr = str(self.array) if self.array is not None else '?'
        idx = ', '.join(str(i) for i in self.indices)
        return f"{arr}[{idx}]"


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
        self.index_chain = _t(index_chain)
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_jagged_access(self)

    def __str__(self) -> str:
        base = str(self.base) if self.base is not None else '?'
        chain = ''.join(f'[{i}]' for i in self.index_chain)
        return f"{base}{chain}"


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
        self.statements = _t(statements)
        self.final_expr = final_expr
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_block_expression(self)

    def __str__(self) -> str:
        if self.final_expr is not None:
            return f"{{ ...; {self.final_expr} }}"
        return "{ ... }"


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

    def __str__(self) -> str:
        s = f"if {self.condition} then {self.then_expr}"
        if self.else_expr is not None:
            s += f" else {self.else_expr}"
        return s


class LambdaIR(ExpressionIR):
    """Lambda expression (rvalue). No defid; closure identity is at use/call site."""
    __slots__ = ('parameters', 'body')

    def __init__(self, parameters: List['ParameterIR'], body: ExpressionIR,
                 location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.parameters = _t(parameters)
        self.body = body

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_lambda(self)

    def __str__(self) -> str:
        params = ', '.join(p.name for p in self.parameters)
        return f"|{params}| {self.body}"


class FunctionValueIR(ExpressionIR):
    """Function value (rvalue). Name and defid live on BindingIR; this holds parameters, body, return_type.
    custom_diff_body: optional @fn rule body; used only by AutodiffPass and cleared after it runs."""
    __slots__ = ('parameters', 'return_type', 'body', '_is_partially_specialized', '_generic_defid', 'custom_diff_body')

    def __init__(self, parameters: List['ParameterIR'], body: ExpressionIR,
                 location: SourceLocation, return_type: Optional[Any] = None,
                 shape_info: Optional[Any] = None, type_info: Optional[Any] = None,
                 _is_partially_specialized: bool = False,
                 _generic_defid: Optional[DefId] = None,
                 custom_diff_body: Optional['ExpressionIR'] = None):
        super().__init__(location, type_info, shape_info)
        assert_defid(_generic_defid)
        self.parameters = _t(parameters)
        self.return_type = return_type
        self.body = body
        self._is_partially_specialized = _is_partially_specialized
        self._generic_defid = _generic_defid
        self.custom_diff_body = custom_diff_body

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_function_value(self)

    def __str__(self) -> str:
        params = ', '.join(p.name for p in self.parameters)
        return f"fn({params}) {{ ... }}"


class FunctionCallIR(ExpressionIR):
    """Function call. Callee is IdentifierIR (has defid = binding we call) or other expr (lambda)."""
    __slots__ = ('callee_expr', 'arguments', 'module_path')

    def __init__(self, callee_expr: ExpressionIR, location: SourceLocation,
                 arguments: Optional[List[ExpressionIR]] = None,
                 module_path: Optional[Tuple[str, ...]] = None,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.callee_expr = callee_expr
        self.arguments = _t(arguments)
        self.module_path = module_path

    @property
    def function_name(self) -> str:
        if isinstance(self.callee_expr, IdentifierIR):
            return self.callee_expr.name or ''
        return '<callable>'

    @property
    def function_defid(self) -> Optional[DefId]:
        if isinstance(self.callee_expr, IdentifierIR):
            return self.callee_expr.defid
        return None

    def set_callee_defid(self, defid: DefId) -> None:
        if isinstance(self.callee_expr, IdentifierIR):
            self.callee_expr.defid = defid

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_function_call(self)

    def __str__(self) -> str:
        name = self.function_name
        args = ', '.join(str(a) for a in self.arguments)
        return f"{name}({args})"


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

    def __str__(self) -> str:
        op = '..=' if self.inclusive else '..'
        return f"{self.start}{op}{self.end}"


class ArrayComprehensionIR(ExpressionIR):
    """
    Array comprehension: [expr | var in range, ...]
    Loop vars are IndexVarIR or IdentifierIR (name + defid per variable).
    """
    __slots__ = ('body', 'loop_vars', 'ranges', 'constraints')

    def __init__(self, body: ExpressionIR,
                 loop_vars: List[Union['IndexVarIR', 'IdentifierIR']],
                 ranges: List[ExpressionIR],
                 location: SourceLocation,
                 constraints: Optional[List[ExpressionIR]] = None,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.body = body
        self.loop_vars = _t(loop_vars)
        self.ranges = _t(ranges)
        if len(self.loop_vars) != len(self.ranges):
            raise ValueError(f"Mismatch: {len(self.loop_vars)} loop_vars but {len(self.ranges)} ranges")
        self.constraints = _t(constraints)

    @property
    def variables(self) -> List[str]:
        return [v.name for v in self.loop_vars]

    @property
    def variable_defids(self) -> List[Optional[DefId]]:
        return [v.defid for v in self.loop_vars]

    @property
    def variable(self) -> str:
        if len(self.loop_vars) == 1:
            return self.loop_vars[0].name
        raise AttributeError("ArrayComprehensionIR has multiple variables, use .variables")

    @property
    def variable_defid(self) -> Optional[DefId]:
        if len(self.loop_vars) == 1:
            return self.loop_vars[0].defid
        raise AttributeError("ArrayComprehensionIR has multiple variables, use .variable_defids")

    @property
    def range_expr(self) -> ExpressionIR:
        if len(self.ranges) == 1:
            return self.ranges[0]
        raise AttributeError("ArrayComprehensionIR has multiple ranges, use .ranges")

    @property
    def expr(self) -> ExpressionIR:
        return self.body

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_array_comprehension(self)

    def __str__(self) -> str:
        vars_ranges = ', '.join(
            f"{v.name} in {r}" for v, r in zip(self.loop_vars, self.ranges)
        )
        return f"[{self.body} | {vars_ranges}]"


class ArrayLiteralIR(ExpressionIR):
    """Array literal: [1, 2, 3]"""
    __slots__ = ('elements',)

    def __init__(self, elements: List[ExpressionIR], location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None,
                 **kwargs: Any):
        super().__init__(location, type_info, shape_info, **kwargs)
        self.elements = _t(elements)
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_array_literal(self)

    def __str__(self) -> str:
        elems = ', '.join(str(e) for e in self.elements)
        return f"[{elems}]"


class TupleExpressionIR(ExpressionIR):
    """Tuple expression: (a, b, c)"""
    __slots__ = ('elements',)
    
    def __init__(self, elements: List[ExpressionIR], location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None,
                 **kwargs: Any):
        super().__init__(location, type_info, shape_info, **kwargs)
        self.elements = _t(elements)

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_tuple_expression(self)

    def __str__(self) -> str:
        elems = ', '.join(str(e) for e in self.elements)
        return f"({elems})"


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

    def __str__(self) -> str:
        return f"{self.tuple_expr}.{self.index}"


class InterpolatedStringIR(ExpressionIR):
    """Interpolated string: "Hello {name}" """
    __slots__ = ('parts',)
    
    def __init__(self, parts: List[Union[str, ExpressionIR]], location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None,
                 **kwargs: Any):
        super().__init__(location, type_info, shape_info, **kwargs)
        self.parts = _t(parts)
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_interpolated_string(self)

    def __str__(self) -> str:
        pieces = []
        for p in self.parts:
            if isinstance(p, str):
                pieces.append(p)
            else:
                pieces.append(f"{{{p}}}")
        return f'"{"".join(pieces)}"'


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

    def __str__(self) -> str:
        return f"{self.expr} as {self.target_type}"


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

    def __str__(self) -> str:
        return f"{self.object}.{self.member}"


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

    def __str__(self) -> str:
        return f"try {self.operand}"


class ReductionExpressionIR(ExpressionIR):
    """Reduction expression: sum[i](A[i]). Loop vars are IndexVarIR or IdentifierIR (each has .name and .defid)."""
    __slots__ = ('operation', 'loop_vars', 'body', 'where_clause', 'loop_var_ranges')

    def __init__(self, operation: "ReductionOp", loop_vars: Optional[List[Union['IndexVarIR', 'IdentifierIR']]], body: ExpressionIR,
                 location: SourceLocation, where_clause: Optional['WhereClauseIR'] = None,
                 loop_var_ranges: Optional[Dict[DefId, 'RangeIR']] = None,
                 type_info: Optional[Any] = None,
                 shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.operation = operation
        self.body = body
        self.where_clause = where_clause
        self.loop_var_ranges = loop_var_ranges if loop_var_ranges is not None else {}
        self.loop_vars = _t(loop_vars)

    @property
    def loop_var_names(self) -> List[str]:
        return [ident.name for ident in self.loop_vars]

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_reduction_expression(self)

    def __str__(self) -> str:
        op = self.operation.value if hasattr(self.operation, 'value') else str(self.operation)
        vars_str = ', '.join(v.name for v in self.loop_vars) if self.loop_vars else ''
        return f"{op}[{vars_str}]({self.body})"


class SelectAtArgmaxIR(ExpressionIR):
    """Autodiff: differential of max/min reduction. Represents d(ext_i body) = d_body at argmax/argmin(primal_body)."""
    __slots__ = ('primal_body', 'diff_body', 'loop_vars', 'loop_var_ranges', 'use_argmin')

    def __init__(
        self,
        primal_body: ExpressionIR,
        diff_body: ExpressionIR,
        loop_vars: Optional[List[Union['IndexVarIR', 'IdentifierIR']]],
        loop_var_ranges: Optional[Dict[DefId, 'RangeIR']] = None,
        location: Optional[SourceLocation] = None,
        type_info: Optional[Any] = None,
        shape_info: Optional[Any] = None,
        use_argmin: bool = False,
    ):
        loc = location or (primal_body.location if primal_body else None)
        if loc is None:
            loc = SourceLocation(file='', line=0, column=0)
        super().__init__(loc, type_info=type_info, shape_info=shape_info)
        self.primal_body = primal_body
        self.diff_body = diff_body
        self.loop_vars = _t(loop_vars)
        self.loop_var_ranges = loop_var_ranges if loop_var_ranges is not None else {}
        self.use_argmin = use_argmin

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_select_at_argmax(self)

    def __str__(self) -> str:
        fn = 'argmin' if self.use_argmin else 'argmax'
        vars_str = ', '.join(v.name for v in self.loop_vars) if self.loop_vars else ''
        return f"{self.diff_body} at {fn}[{vars_str}]({self.primal_body})"


class WhereExpressionIR(ExpressionIR):
    """Where expression: expr where constraint"""
    __slots__ = ('expr', 'constraints')
    
    def __init__(self, expr: ExpressionIR, constraints: List[ExpressionIR], location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.expr = expr
        self.constraints = _t(constraints)
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_where_expression(self)

    def __str__(self) -> str:
        constraints = ', '.join(str(c) for c in self.constraints)
        return f"{self.expr} where {constraints}"


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

    def __str__(self) -> str:
        return f"{self.left} {self.operator} {self.right}"


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
        self.args = _t(args)

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_builtin_call(self)

    def __str__(self) -> str:
        args = ', '.join(str(a) for a in self.args)
        return f"{self.builtin_name}({args})"


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


class DiffRuleIR(IRNode):
    """Custom autodiff rule for a user function: @fn f(params) { body }. Keyed by callee_defid; body uses @param -> DifferentialIR."""
    __slots__ = ('callee_defid', 'body')

    def __init__(self, callee_defid: Optional[DefId] = None, body: Optional[ExpressionIR] = None,
                 location: Optional[SourceLocation] = None):
        super().__init__(location or SourceLocation('', 0, 0))
        self.callee_defid = callee_defid
        self.body = body

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        if hasattr(visitor, 'visit_diff_rule'):
            return visitor.visit_diff_rule(self)
        return None  # type: ignore[return-value]


class ProgramIR(IRNode):
    """
    Complete program in IR. statements is the preserved list (includes BindingIR and may include other statement types).
    bindings/functions/constants are derived from statements.
    """
    __slots__ = ('modules', 'statements', 'source_files', 'bindings')

    def __init__(self, statements: List[Any],
                 source_files: Optional[dict] = None, modules: Optional[List['ModuleIR']] = None,
                 location: Optional[SourceLocation] = None, **_kw: Any):
        super().__init__(location or SourceLocation('', 0, 0))
        self.statements = statements
        self.bindings = [s for s in statements if isinstance(s, BindingIR)]
        self.source_files = source_files if source_files is not None else {}
        self.modules = _t(modules)

    @property
    def defid_to_name(self) -> Dict[DefId, str]:
        d2n: Dict[DefId, str] = {}
        for s in self.bindings:
            if s.defid is not None:
                d2n[s.defid] = s.name or ""
        return d2n

    @property
    def functions(self) -> List['BindingIR']:
        return [b for b in self.bindings if is_function_binding(b)]

    @property
    def constants(self) -> List['BindingIR']:
        return [b for b in self.bindings if is_constant_binding(b)]

    @property
    def diff_rules(self) -> List['DiffRuleIR']:
        return [s for s in (self.statements or []) if isinstance(s, DiffRuleIR)]

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_program(self)


def clear_autodiff_only_fields(program: 'ProgramIR') -> None:
    """Reset autodiff-only fields on the IR (e.g. custom_diff_body). No longer needed after AutodiffPass."""
    for b in (program.bindings or []):
        if isinstance(b.expr, FunctionValueIR) and getattr(b.expr, 'custom_diff_body', None) is not None:
            object.__setattr__(b.expr, 'custom_diff_body', None)
    for mod in (program.modules or []):
        _clear_autodiff_only_fields_module(mod)


def _clear_autodiff_only_fields_module(mod: 'ModuleIR') -> None:
    for b in (mod.functions or []):
        if isinstance(b.expr, FunctionValueIR) and getattr(b.expr, 'custom_diff_body', None) is not None:
            object.__setattr__(b.expr, 'custom_diff_body', None)
    for sub in (mod.submodules or []):
        _clear_autodiff_only_fields_module(sub)


class ModuleIR(IRNode):
    """Module in IR"""
    __slots__ = ('path', 'functions', 'constants', 'submodules', 'defid')

    def __init__(self, path: Tuple[str, ...], location: SourceLocation,
                 functions: List['BindingIR'], constants: List['BindingIR'],
                 submodules: List['ModuleIR'], defid: Optional[DefId] = None):
        super().__init__(location)
        assert_defid(defid)
        self.defid = defid
        self.path = path
        self.functions = _t(functions)
        self.constants = _t(constants)
        self.submodules = _t(submodules)
    
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
        self.patterns = _t(patterns)
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_tuple_pattern(self)


class ArrayPatternIR(PatternIR):
    """Array pattern: [a, b, ..rest]"""
    __slots__ = ('patterns',)

    def __init__(self, patterns: List[PatternIR], location: SourceLocation):
        super().__init__(location)
        self.patterns = _t(patterns)
    
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
        self.alternatives = _t(alternatives)
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_or_pattern(self)


class ConstructorPatternIR(PatternIR):
    """Constructor pattern: Some(x), Circle(r)"""
    __slots__ = ('constructor_name', 'patterns', 'is_struct_literal')

    def __init__(self, constructor_name: str, patterns: List[PatternIR],
                 is_struct_literal: bool = False, location: SourceLocation = None):
        super().__init__(location or SourceLocation("", 0, 0))
        self.constructor_name = constructor_name
        self.patterns = _t(patterns)
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


class MatchArmIR(IRNode):
    """Match arm: pattern + body"""
    __slots__ = ('pattern', 'body')
    
    def __init__(self, pattern: PatternIR, body: ExpressionIR,
                 location: Optional[SourceLocation] = None):
        super().__init__(location or SourceLocation('', 0, 0))
        self.pattern = pattern
        self.body = body


class MatchExpressionIR(ExpressionIR):
    """Match expression: match scrutinee { arms }"""
    __slots__ = ('scrutinee', 'arms')
    
    def __init__(self, scrutinee: ExpressionIR, arms: List[MatchArmIR], location: SourceLocation,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location, type_info, shape_info)
        self.scrutinee = scrutinee
        self.arms = _t(arms)
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_match_expression(self)

    def __str__(self) -> str:
        return f"match {self.scrutinee} {{ {len(self.arms)} arms }}"


# Where Clause IR

class WhereClauseIR(IRNode):
    """Where clause: constraints for filtering. ranges is keyed by DefId (index variable)."""
    __slots__ = ('constraints', 'ranges')
    
    def __init__(self, constraints: List[ExpressionIR], ranges: Optional[Dict[DefId, Any]] = None,
                 location: Optional[SourceLocation] = None):
        super().__init__(location or SourceLocation('', 0, 0))
        self.constraints = _t(constraints)
        self.ranges = ranges if ranges is not None else {}


# Lowered iteration structures (aligned with LoopStructure + shared iteration shape)
# LoweredIteration has body, loops, bindings, guards, reduction_ranges, shape, element_type

class LoopStructure(IRNode):
    """
    Loop iteration: for variable in iterable ().
    iterable is an expression (RangeIR, LiteralIR(range), etc.).
    variable is IndexVarIR for index loops (defid only; no name-based lookup) or IdentifierIR for reduction.
    """
    __slots__ = ('variable', 'iterable')

    def __init__(self, variable: "Union[IdentifierIR, IndexVarIR]", iterable: ExpressionIR,
                 location: Optional[SourceLocation] = None):
        super().__init__(location or SourceLocation('', 0, 0))
        self.variable = variable
        self.iterable = iterable

    def __str__(self) -> str:
        return f"{self.variable.name} in {self.iterable}"


class BindingIR(IRNode):
    """Canonical binding (name = expr). Only LHS (defid/name) is the reference; expr is rvalue. IR has only bindings."""
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

    @property
    def parameters(self) -> List['ParameterIR']:
        return self.expr.parameters if isinstance(self.expr, FunctionValueIR) else []

    @property
    def body(self):
        return self.expr.body if isinstance(self.expr, FunctionValueIR) else None

    @property
    def return_type(self):
        return self.expr.return_type if isinstance(self.expr, FunctionValueIR) else None

    @property
    def clauses(self) -> List[Any]:
        if isinstance(self.expr, EinsteinIR):
            return self.expr.clauses or []
        return []

    def get_defid_binding(self) -> Optional[tuple]:
        if self.defid is not None:
            return (self.defid, self.expr)
        return None

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_binding(self)

    def __str__(self) -> str:
        type_str = f": {self.type_info}" if self.type_info else ""
        return f"{self.name}{type_str} = {self.expr}"


def is_function_binding(binding: Any) -> bool:
    """True if binding is a function definition (expr is FunctionValueIR)."""
    if not isinstance(binding, BindingIR):
        return False
    return isinstance(binding.expr, FunctionValueIR)


def is_einstein_binding(binding: Any) -> bool:
    """True if binding is an Einstein declaration (expr is EinsteinIR)."""
    return isinstance(binding, BindingIR) and isinstance(binding.expr, EinsteinIR)


def is_constant_binding(binding: Any) -> bool:
    """True if binding has defid and is not a function (variable or constant)."""
    if not isinstance(binding, BindingIR):
        return False
    return binding.defid is not None and not is_function_binding(binding)


FunctionDefIR = BindingIR
ConstantDefIR = BindingIR


class GuardCondition(IRNode):
    """Runtime guard condition"""
    __slots__ = ('condition',)
    
    def __init__(self, condition: ExpressionIR, location: Optional[SourceLocation] = None):
        super().__init__(location or SourceLocation('', 0, 0))
        self.condition = condition
    
    def __str__(self) -> str:
        """Human-readable representation"""
        return f"guard {self.condition}"


class LoweredIteration(IRNode):
    """
    Unified lowered representation for all iteration constructs.
    
    Used by BindingIR + EinsteinIR via composition.
    
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
        element_type: Optional[Any] = None,
        location: Optional[SourceLocation] = None
    ):
        super().__init__(location or SourceLocation('', 0, 0))
        self.body = body
        self.loops = _t(loops)
        self.bindings = _t(bindings)
        self.guards = _t(guards)
        self.reduction_ranges = reduction_ranges if reduction_ranges is not None else {}
        self.shape = _t(shape) if shape is not None else None
        self.element_type = element_type
    
    def is_empty(self) -> bool:
        """Check if lowered iteration is empty"""
        return not self.loops and not self.bindings and not self.guards and not self.reduction_ranges
    
    def is_rectangular(self) -> bool:
        """Check if all loops are rectangular (no conditions)"""
        # We don't have loop.condition, so all loops are rectangular
        return True
    
    def __str__(self) -> str:
        vars_str = ', '.join(f"{l.variable.name} in {l.iterable}" for l in self.loops) if self.loops else ''
        return f"for {vars_str}: {self.body}" if vars_str else str(self.body)


class LoweredEinsteinClauseIR(IRNode):
    """Single lowered Einstein clause (body, loops, bindings, guards, indices). reduction_ranges keyed by DefId.
    recurrence_dims_override: set by RecurrenceOrderPass when clause has same-timestep dependency (e.g. reads u[t,0] when writing u[t,1])."""
    __slots__ = ('body', 'loops', 'reduction_ranges', 'bindings', 'guards', 'indices', 'recurrence_dims_override')

    def __init__(
        self,
        body: ExpressionIR,
        loops: Optional[List[LoopStructure]] = None,
        reduction_ranges: Optional[Dict[DefId, LoopStructure]] = None,
        bindings: Optional[List['BindingIR']] = None,
        guards: Optional[List[GuardCondition]] = None,
        indices: Optional[List[Any]] = None,
        recurrence_dims_override: Optional[List[int]] = None,
        location: Optional[SourceLocation] = None,
    ):
        super().__init__(location or SourceLocation('', 0, 0))
        self.body = body
        self.loops = _t(loops)
        self.reduction_ranges = reduction_ranges if reduction_ranges is not None else {}
        self.bindings = _t(bindings)
        self.guards = _t(guards)
        self.indices = _t(indices)
        self.recurrence_dims_override = recurrence_dims_override
    
    def __str__(self) -> str:
        idx = ', '.join(str(i) for i in self.indices) if self.indices else ''
        return f"[{idx}] = {self.body}"

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_lowered_einstein_clause(self)


class LoweredEinsteinIR(IRNode):
    """Lowered Einstein declaration: one tensor, one shape. All clauses write to the same memory (same shape)."""
    __slots__ = ('items', 'shape', 'element_type')

    def __init__(self, items: List['LoweredEinsteinClauseIR'],
                 shape: Optional[List[ExpressionIR]] = None,
                 element_type: Optional[Any] = None,
                 location: Optional[SourceLocation] = None):
        super().__init__(location or SourceLocation('', 0, 0))
        self.items = _t(items)
        self.shape = _t(shape) if shape is not None else None
        self.element_type = element_type

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_lowered_einstein(self)


class LoweredRecurrenceIR(ExpressionIR):
    """
    Recurrence loop isolated out of the Einstein clause.
    initial: run once (non-recurrence clauses). recurrence_loop: the timestep loop (e.g. t).
    body: recurrence clauses only; executed once per timestep with loop var in env.
    """
    __slots__ = ('initial', 'recurrence_loop', 'body')

    def __init__(
        self,
        initial: LoweredEinsteinIR,
        recurrence_loop: 'LoopStructure',
        body: LoweredEinsteinIR,
        location: Optional[SourceLocation] = None,
    ):
        super().__init__(location or SourceLocation('', 0, 0))
        self.initial = initial
        self.recurrence_loop = recurrence_loop
        self.body = body

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_lowered_recurrence(self)

    def __str__(self) -> str:
        loop_var = self.recurrence_loop.variable.name if self.recurrence_loop else '?'
        return f"recurrence[{loop_var}]({self.body})"


class LoweredReductionIR(ExpressionIR):
    """
    Lowered reduction (LoweredIteration shape). Replaces ReductionExpressionIR.
    body, operation, loops, bindings, guards; reduction_ranges derived from loops.
    """
    __slots__ = ('body', 'operation', 'loops', 'bindings', 'guards')
    
    def __init__(
        self,
        body: ExpressionIR,
        operation: "ReductionOp",
        loops: Optional[List[LoopStructure]] = None,
        bindings: Optional[List['BindingIR']] = None,
        guards: Optional[List[GuardCondition]] = None,
        location: Optional[SourceLocation] = None,
        type_info: Optional[Any] = None,
        shape_info: Optional[Any] = None,
    ):
        loc = location or body.location
        if loc is None:
            loc = SourceLocation(file='', line=0, column=0)
        super().__init__(loc, type_info=type_info, shape_info=shape_info)
        self.body = body
        self.operation = operation
        self.loops = _t(loops)
        self.bindings = _t(bindings)
        self.guards = _t(guards)
    
    @property
    def reduction_ranges(self) -> Dict[DefId, LoopStructure]:
        """Reduction variable ranges keyed by variable DefId (for compatibility with execute_reduction_with_loops which uses .values())."""
        result: Dict[DefId, LoopStructure] = {}
        for loop in self.loops:
            d = loop.variable.defid
            if d is not None:
                result[d] = loop
        return result
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_lowered_reduction(self)
    
    def __str__(self) -> str:
        op = self.operation.value if hasattr(self.operation, 'value') else str(self.operation)
        vars_str = ', '.join(l.variable.name for l in self.loops) if self.loops else ''
        return f"{op}[{vars_str}]({self.body})"


class LoweredSelectAtArgmaxIR(ExpressionIR):
    """Lowered select-at-argmax/argmin (autodiff of max/min). primal_body, diff_body, loops; result = diff at argmax/argmin(primal)."""
    __slots__ = ('primal_body', 'diff_body', 'loops', 'bindings', 'guards', 'use_argmin')

    def __init__(
        self,
        primal_body: ExpressionIR,
        diff_body: ExpressionIR,
        loops: Optional[List[LoopStructure]] = None,
        bindings: Optional[List['BindingIR']] = None,
        guards: Optional[List[GuardCondition]] = None,
        location: Optional[SourceLocation] = None,
        type_info: Optional[Any] = None,
        shape_info: Optional[Any] = None,
        use_argmin: bool = False,
    ):
        loc = location or (primal_body.location if primal_body else None)
        if loc is None:
            loc = SourceLocation(file='', line=0, column=0)
        super().__init__(loc, type_info=type_info, shape_info=shape_info)
        self.primal_body = primal_body
        self.diff_body = diff_body
        self.loops = _t(loops)
        self.bindings = _t(bindings)
        self.guards = _t(guards)
        self.use_argmin = use_argmin

    @property
    def reduction_ranges(self) -> Dict[DefId, LoopStructure]:
        result: Dict[DefId, LoopStructure] = {}
        for loop in self.loops or []:
            d = loop.variable.defid
            if d is not None:
                result[d] = loop
        return result

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_lowered_select_at_argmax(self)

    def __str__(self) -> str:
        fn = 'argmin' if self.use_argmin else 'argmax'
        vars_str = ', '.join(l.variable.name for l in self.loops) if self.loops else ''
        return f"{self.diff_body} at {fn}[{vars_str}]({self.primal_body})"


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
        loc = location or body.location
        if loc is None:
            loc = SourceLocation(file='', line=0, column=0)
        super().__init__(loc, type_info=type_info, shape_info=shape_info)
        self.body = body
        self.loops = _t(loops)
        self.bindings = _t(bindings)
        self.guards = _t(guards)
    
    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_lowered_comprehension(self)
    
    def __str__(self) -> str:
        vars_ranges = ', '.join(f"{l.variable.name} in {l.iterable}" for l in self.loops) if self.loops else ''
        return f"[{self.body} | {vars_ranges}]"


# Einstein IR: one clause (indices, value, where_clause). EinsteinIR holds list of EinsteinClauseIR.

class EinsteinClauseIR(IRNode):
    """One Einstein clause. Holds indices, value, where_clause, and variable_ranges.
    Ranges are only on the clause (not on the binding). variable_ranges is keyed by DefId.
    Precision (element_type) is on the declaration; runtime receives it and passes it in."""
    __slots__ = ('indices', 'value', 'where_clause', 'variable_ranges')

    def __init__(self, indices: List[ExpressionIR], value: ExpressionIR,
                 location: SourceLocation, where_clause: Optional[WhereClauseIR] = None,
                 variable_ranges: Optional[Dict[DefId, Any]] = None):
        super().__init__(location)
        self.indices = _t(indices)
        self.value = value
        self.where_clause = where_clause
        self.variable_ranges = variable_ranges if variable_ranges is not None else {}

    @property
    def loop_vars(self) -> List[str]:
        """Derive loop variable names from indices (IndexVarIR, IndexRestIR; legacy IdentifierIR)."""
        out: List[str] = []
        for idx in self.indices or []:
            if isinstance(idx, (IndexVarIR, IndexRestIR, IdentifierIR)) and idx.name:
                out.append(idx.name)
            elif isinstance(idx, (list, tuple)):
                for sub in idx:
                    if isinstance(sub, (IndexVarIR, IndexRestIR, IdentifierIR)) and sub.name:
                        out.append(sub.name)
        return out

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_einstein_clause(self)

    def __str__(self) -> str:
        idx = ', '.join(str(i) for i in self.indices) if self.indices else ''
        return f"[{idx}] = {self.value}"


class EinsteinIR(ExpressionIR):
    """Einstein value (rvalue). Name/defid on BindingIR; this holds clauses, shape, element_type."""
    __slots__ = ('clauses', 'shape', 'element_type')

    def __init__(self, clauses: Optional[List[EinsteinClauseIR]] = None,
                 shape: Optional[List[ExpressionIR]] = None,
                 element_type: Optional[Any] = None,
                 location: Optional[SourceLocation] = None,
                 type_info: Optional[Any] = None, shape_info: Optional[Any] = None):
        super().__init__(location or SourceLocation('', 0, 0), type_info, shape_info)
        self.clauses = _t(clauses)
        self.shape = _t(shape) if shape is not None else None
        self.element_type = element_type

    def accept(self, visitor: 'IRVisitor[T]') -> 'T':
        return visitor.visit_einstein(self)

    def __str__(self) -> str:
        clauses = '; '.join(str(c) for c in self.clauses) if self.clauses else ''
        return f"{{ {clauses} }}"



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

    def visit_differential(self, node: 'DifferentialIR') -> T:
        """Visit differential expression (@expr). Default: recurse into operand."""
        return node.operand.accept(self)

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

    def visit_select_at_argmax(self, node: 'SelectAtArgmaxIR') -> T:
        """Visit select-at-argmax (autodiff of max reduction). Default: recurse into bodies."""
        if node.primal_body is not None:
            node.primal_body.accept(self)
        if node.diff_body is not None:
            return node.diff_body.accept(self)
        return None  # type: ignore[return-value]

    def visit_lowered_select_at_argmax(self, node: 'LoweredSelectAtArgmaxIR') -> T:
        """Visit lowered select-at-argmax. Default: recurse into bodies."""
        if node.primal_body is not None:
            node.primal_body.accept(self)
        if node.diff_body is not None:
            return node.diff_body.accept(self)
        return None  # type: ignore[return-value]

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

    def visit_lowered_recurrence(self, node: 'LoweredRecurrenceIR') -> T:
        """Visit recurrence isolated out of Einstein. Default: recurse into initial, recurrence_loop.iterable, body."""
        result = None  # type: ignore[assignment]
        if node.initial:
            result = node.initial.accept(self)
        if node.recurrence_loop and node.recurrence_loop.iterable:
            result = node.recurrence_loop.iterable.accept(self)
        if node.body:
            result = node.body.accept(self)
        return result

    @abstractmethod
    def visit_where_expression(self, node: WhereExpressionIR) -> T:
        """Visit where expression"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_pipeline_expression(self, node: PipelineExpressionIR) -> T:
        """Visit pipeline expression"""
        raise NotImplementedError

    def visit_differential(self, node: 'DifferentialIR') -> T:
        """Visit differential expression (@expr). Default: recurse into operand."""
        return node.operand.accept(self)

    @abstractmethod
    def visit_builtin_call(self, node: BuiltinCallIR) -> T:
        """Visit builtin call"""
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
    
    def visit_function_value(self, node: 'FunctionValueIR') -> T:
        if node.body is not None:
            return node.body.accept(self)
        return None  # type: ignore[return-value]

    def visit_einstein(self, node: 'EinsteinIR') -> T:
        return None  # type: ignore[return-value]

    def visit_einstein_clause(self, node: EinsteinClauseIR) -> T:
        """Visit one Einstein clause. Default: no-op."""
        return None  # type: ignore[return-value]

    def visit_binding(self, node: 'BindingIR') -> T:
        """Visit binding (name = expr). Default: delegate to node.expr.accept(self)."""
        if node.expr is not None:
            return node.expr.accept(self)
        return None  # type: ignore[return-value]
    
    # Program visitor
    @abstractmethod
    def visit_program(self, node: 'ProgramIR') -> T:
        """Visit program"""
        raise NotImplementedError



