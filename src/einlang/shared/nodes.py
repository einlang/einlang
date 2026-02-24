"""
Einlang AST (Abstract Syntax Tree) Definitions
Clean, minimal AST nodes for our core functionality

Moved to shared module to eliminate circular imports between frontend and backend.
This follows industry best practices used by LLVM, Rust, and Clang compilers.

Visitor Pattern Support:
- All AST nodes now have accept() methods for polymorphic dispatch
- Follows same pattern as IR nodes (LLVM-style)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, TYPE_CHECKING, TypeVar
try:
    from typing import Final, Literal
except ImportError:
    # Python 3.7 compatibility
    from typing_extensions import Final, Literal
from enum import Enum
from .types import BinaryOp, UnaryOp

if TYPE_CHECKING:
    pass

T = TypeVar('T')

# Compile-time constants for constraint types
BINDING_CONSTRAINT: Final = "binding"
ITERATION_DOMAIN_CONSTRAINT: Final = "iteration_domain"
INDEX_RELATIONAL_CONSTRAINT: Final = "index_relational"
VALUE_RELATIONAL_CONSTRAINT: Final = "value_relational"

class ArrowOperator(Enum):
    """Arrow operators for ML/DL computation graph construction"""
    SEQUENTIAL = ">>>"  # A >>> B >>> C - Sequential composition
    PARALLEL = "***"    # A *** B *** C - Parallel composition
    FANOUT = "&&&"      # A &&& B &&& C - Fanout composition
    CHOICE = "|||"      # A ||| B ||| C - Choice composition

class NodeType(Enum):
    """AST node types"""
    PROGRAM = "program"
    FUNCTION_DEF = "function_def"
    VARIABLE_DECL = "variable_decl"
    EINSTEIN_DECL = "einstein_decl"  # Single or multiple clauses (array_name + clauses)
    EXPR_STMT = "expr_stmt"  # Expression used as statement
    REDUCTION_EXPR = "reduction_expr"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    CAST = "cast"
    RECTANGULAR_ACCESS = "rectangular_access"  # Renamed from rectangular_access
    JAGGED_ACCESS = "jagged_access"  # New for jagged types
    LITERAL = "literal"
    IDENTIFIER = "identifier"
    FUNCTION_CALL = "function_call"
    METHOD_CALL = "method_call"
    IF_EXPR = "if_expr"
    ARRAY_LITERAL = "array_literal"
    MEMBER_ACCESS = "member_access"
    MODULE_ACCESS = "module_access"
    USE_STMT = "use_stmt"
    MOD_STMT = "mod_stmt"
    INLINE_MOD_STMT = "inline_mod_stmt"
    WHERE_EXPR = "where_expr"
    TUPLE_EXPR = "tuple_expr"
    ARRAY_COMPREHENSION = "array_comprehension"
    INTERPOLATED_STRING = "interpolated_string"
    PIPELINE_EXPR = "pipeline_expr"
    LAMBDA_EXPR = "lambda_expr"
    TRY_EXPR = "try_expr"
    BLOCK_EXPR = "block_expr"
    ARROW_EXPR = "arrow_expr"  # Consolidated arrow expression
    TUPLE_DESTRUCTURE_PATTERN = "tuple_destructure_pattern"
    CONSTRAINT = "constraint"
    RANGE = "range"
    INDEX_VAR = "index_var"      # Variable index slot (LHS or reduction): name + optional range
    INDEX_REST = "index_rest"    # Rest index slot (LHS or reduction): ..name
    MATCH_EXPR = "match_expr"
    LITERAL_PATTERN = "literal_pattern"
    IDENTIFIER_PATTERN = "identifier_pattern"
    WILDCARD_PATTERN = "wildcard_pattern"
    TUPLE_PATTERN = "tuple_pattern"
    ARRAY_PATTERN = "array_pattern"
    REST_PATTERN = "rest_pattern"  # Rest pattern ..pattern
    GUARD_PATTERN = "guard_pattern"
    ENUM_DEF = "enum_def"
    STRUCT_DEF = "struct_def"
    CONSTRUCTOR_PATTERN = "constructor_pattern"

class ConstraintType(Enum):
    """
    Constraint classification for where clauses.
    
    Distinguishes between different constraint types for proper handling
    in analysis passes and runtime evaluation.
    """
    BINDING = "binding"                      # x = expr (introduces variable binding)
    ITERATION_DOMAIN = "iteration_domain"    # x in collection/range (defines what to iterate over: ranges, arrays, any collection)
    INDEX_RELATIONAL = "index_relational"    # i < j, i >= 0 (index constraints - allowed in Einstein)
    VALUE_RELATIONAL = "value_relational"    # arr[i] > 0 (value constraints - NOT allowed in Einstein)

@dataclass(frozen=True)
class SourceLocation:
    """Source code location for error reporting"""
    file: str
    line: int
    column: int
    start: int = 0
    end: int = 0
    
    def __str__(self) -> str:
        """Human-readable location representation"""
        return f"{self.file}:{self.line}:{self.column}"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"SourceLocation(file={self.file!r}, line={self.line}, column={self.column}, start={self.start}, end={self.end})"

class ASTNode:
    """
    Base class for all AST nodes
    
    Visitor Pattern Support (LLVM-style):
    - All nodes have accept() method for polymorphic dispatch
    - Subclasses must implement accept() to call appropriate visit_* method
    
    Compile-time optimization:
    - __slots__ for memory efficiency and compile-time attribute checking
    """
    __slots__ = ('node_type', 'location')
    
    def __init__(self, node_type: NodeType, location: SourceLocation):
        self.node_type = node_type
        self.location = location
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        """
        Accept a visitor (polymorphic dispatch).
        
        Subclasses must implement this to call the appropriate visit_* method.
        
        Example:
            class MyPass(ASTVisitor[str]):
                def visit_literal(self, node: Literal) -> str:
                    return f"Literal: {node.value}"
            
            node = Literal(42)
            result = node.accept(MyPass())  # Polymorphic call
        """
        raise NotImplementedError(f"accept() not implemented for {self.__class__.__name__}")
    

class Expression(ASTNode):
    """
    Base class for expressions
    
    Compile-time optimization:
    - __slots__ extended to include metadata fields for compile-time attribute checking
    - Pre-computed metadata populated by analysis passes
    """
    __slots__ = ('_is_reduction', '_reduction_vars', '_has_where_clause', '_constraint_dependencies',
                 '_type_info', '_range_info', '_shape_info')
    
    def __init__(self, node_type: NodeType = None, location: SourceLocation = None):
        super().__init__(node_type, location)
        
        # ✅ COMPILE-TIME OPTIMIZATION: Pre-computed expression metadata
        # These fields are populated by ExpressionStructurePass to eliminate runtime analysis
        self._is_reduction: Optional[bool] = None
        self._reduction_vars: Optional[List[str]] = None
        self._has_where_clause: Optional[bool] = None
        self._constraint_dependencies: Optional[Dict[str, set]] = None  # constraint_var -> set of vars it depends on
        
        # Additional metadata fields populated by various analysis passes
        self._type_info: Optional[Any] = None
        self._range_info: Optional[Any] = None
        self._shape_info: Optional[Any] = None
    
    def set_expression_metadata(self, is_reduction: bool, 
                               reduction_vars: List[str], 
                               has_where_clause: bool,
                               constraint_dependencies: Dict[str, set]):
        """Set pre-computed expression structure metadata (called by ExpressionStructurePass)"""
        self._is_reduction = is_reduction
        self._reduction_vars = reduction_vars
        self._has_where_clause = has_where_clause
        self._constraint_dependencies = constraint_dependencies
    
    def get_expression_metadata(self):
        """Get pre-computed expression metadata (used by runtime)"""
        return {
            'is_reduction': self._is_reduction,
            'reduction_vars': self._reduction_vars,
            'has_where_clause': self._has_where_clause,
            'constraint_dependencies': self._constraint_dependencies
        }

@dataclass
class Range(Expression):
    """Range expression for Einstein notation constraints and tensor operations"""
    start: 'Expression'
    end: 'Expression'
    
    def __init__(self, start: 'Expression', end: 'Expression', location: Optional['SourceLocation'] = None):
        super().__init__(NodeType.RANGE, location)
        self.start = start
        self.end = end
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_range(self)

class Statement(ASTNode):
    """
    Base class for statements
    
    Compile-time optimization:
    - __slots__ for memory efficiency and compile-time attribute checking
    """
    __slots__ = ('_type_info',)
    
    def __init__(self, node_type: NodeType = None, location: SourceLocation = None):
        super().__init__(node_type, location)
        # Type analysis metadata (populated by type analysis passes)
        self._type_info: Optional[Any] = None

@dataclass
class ExpressionStatement(Statement):
    """
    Expression used as a statement (evaluates expression, discards result).
    
    Used for:
    - Function calls with side effects: print(...);
    - If expressions used as statements: if (...) { ... };
    - Any expression in statement position
    
    This is the standard AST pattern (LLVM, Rust, Python, JavaScript ASTs all have this).
    
    Examples:
        print("hello");          # FunctionCall wrapped in ExpressionStatement
        if (x > 0) { ... };     # IfExpression wrapped in ExpressionStatement
    """
    expr: Expression
    
    def __init__(self, expr: Expression, location: SourceLocation = None):
        super().__init__(NodeType.EXPR_STMT, location or (expr.location if expr else None))
        self.expr = expr
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_expression_statement(self)

@dataclass
class Program(ASTNode):
    """Program root node"""
    statements: List[Statement]
    
    def __init__(self, statements: List[Statement], location: SourceLocation = None):
        super().__init__(NodeType.PROGRAM, location)
        self.statements = statements
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_program(self)

@dataclass
class FunctionDefinition(Statement):
    """Function definition with pub visibility"""
    name: str
    parameters: List['Parameter']
    return_type: Optional['Type']
    body: 'BlockExpression'  # Changed from List[Statement] + final_expr
    is_public: bool = False
    
    def __init__(self, name: str, parameters: List['Parameter'], return_type: Optional['Type'], body: 'BlockExpression', is_public: bool = False, location: SourceLocation = None):
        super().__init__(NodeType.FUNCTION_DEF, location)
        self.name = name
        self.parameters = parameters
        self.return_type = return_type
        self.body = body
        self.is_public = is_public
        
        # Analysis metadata (static attributes - always present)
        self.statement_metadata: Optional[Any] = None  # StatementMetadata from RangeAnalysisEngine
        self._defid: Optional[Any] = None  # DefId for this function (allocated by module pass)
        self._is_generic: bool = False  # True if function has generic parameters
        self._instantiation: Optional[Any] = None  # Instantiation info for specialized functions
        self._return_type_info: Optional[Any] = None  # Inferred return type from type analysis
        self._function_signature: Optional[Any] = None  # Function signature tracked by type analysis
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_function_definition(self)

@dataclass
class Parameter:
    """Function parameter"""
    name: str
    type_annotation: Optional['Type'] = None  # Type: set by parser if explicit, by specialization if inferred

@dataclass
class VariableDeclaration(Statement):
    """Variable declaration (let x = value or let (x, y) = tuple)"""
    pattern: Union[str, 'TupleDestructurePattern']
    value: Expression
    type_annotation: Optional['Type'] = None
    
    def __init__(self, pattern: Union[str, 'TupleDestructurePattern'], value: Expression, type_annotation: Optional['Type'] = None, location: SourceLocation = None):
        super().__init__(NodeType.VARIABLE_DECL, location)
        self.pattern = pattern
        self.value = value
        self.type_annotation = type_annotation
        
        # Analysis metadata (static attributes - always present)
        self.statement_metadata: Optional[Any] = None  # StatementMetadata from RangeAnalysisEngine
    
    @property
    def name(self) -> str:
        """Convenience property: returns first variable name (for simple patterns or tuple destructuring)"""
        if isinstance(self.pattern, str):
            return self.pattern
        else:
            return self.pattern.variables[0].name if self.pattern.variables else "unknown"
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_variable_declaration(self)

@dataclass
class Literal(Expression):
    """Literal value (number, string, boolean)"""
    value: Union[int, float, str, bool]
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __init__(self, value: Union[int, float, str, bool], location: SourceLocation = None):
        super().__init__(NodeType.LITERAL, location)
        self.value = value
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_literal(self)

@dataclass
class InterpolatedString(Expression):
    """String with variable interpolation and format specs"""
    parts: List[Union['Literal', 'InterpolationPart']]  # All parts are Expression nodes with accept()
    
    def __init__(self, parts: List[Union['Literal', 'InterpolationPart']], location: SourceLocation = None):
        super().__init__(NodeType.INTERPOLATED_STRING, location)
        self.parts = parts
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_interpolated_string(self)

@dataclass
class InterpolationPart(Expression):
    """Part of an interpolated string containing an expression and optional format spec"""
    expr: Expression
    format_spec: Optional[str] = None
    
    def __init__(self, expr: Expression, format_spec: Optional[str] = None, location: SourceLocation = None):
        # Use the expression's node type since this is a transparent wrapper
        super().__init__(NodeType.INTERPOLATED_STRING, location or expr.location)
        self.expr = expr
        self.format_spec = format_spec
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        # InterpolationPart is transparent - just visit the inner expression
        # Format spec is handled at runtime (in backend)
        return self.expr.accept(visitor)

@dataclass
class Identifier(Expression):
    """Identifier (variable name)"""
    name: str
    
    def __str__(self) -> str:
        return self.name
    
    def __init__(self, name: str, location: SourceLocation = None):
        super().__init__(NodeType.IDENTIFIER, location)
        self.name = name
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_identifier(self)

class IndexVar(Expression):
    """
    Variable index slot (introduces a loop variable) in Einstein LHS or reduction.
    Symmetric with IndexRest. name + optional range (e.g. i in 0..n).
    """
    name: str
    range_expr: Optional['Expression']

    def __init__(self, name: str, range_expr: Optional['Expression'] = None, location: SourceLocation = None):
        super().__init__(NodeType.INDEX_VAR, location)
        self.name = name
        self.range_expr = range_expr

    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_index_var(self)


class IndexRest(Expression):
    """
    Rest index slot (introduces rest dimensions) in Einstein LHS or reduction.
    Symmetric with IndexVar. ..name.
    """
    name: str

    def __init__(self, name: str, location: SourceLocation = None):
        super().__init__(NodeType.INDEX_REST, location)
        self.name = name

    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_index_rest(self)


@dataclass
class FunctionCall(Expression):
    """
    Function call - function expression is always an AST node (supports visitor pattern)
    
    Examples:
    - Simple call: print(x) -> function_expr=Identifier("print")
    - Member call: obj.method() -> function_expr=MemberAccess(...)
    - Module call: math::sqrt() -> function_expr=ModuleAccess(...)
    
    Metadata attached during ModulePass (scope-aware):
    - _resolved_module_key: Full module path (e.g., "std::math")
    - _resolved_function_name: Actual function name
    - _resolved_module_ref: Module object reference (set during lowering)
    """
    function_expr: Expression  # Always an AST node (Identifier, MemberAccess, ModuleAccess, etc.)
    arguments: List[Expression]
    
    def __init__(self, function_expr: Expression, arguments: List[Expression], 
                 location: SourceLocation = None):
        super().__init__(NodeType.FUNCTION_CALL, location)
        self.function_expr = function_expr
        self.arguments = arguments
        
        # Resolution metadata (attached during ModulePass)
        self._resolved_module_key: Optional[str] = None
        self._resolved_function_name: Optional[str] = None
        self._resolved_module_ref: Optional[Any] = None  # Module object (set during lowering)
        
        # Monomorphization metadata (attached during CallDiscovery for generic calls)
        self._defid: Optional[Any] = None  # DefId for resolved function (set by module pass or call discovery)
        self._resolved_scope_path: Optional[str] = None  # Scope path for resolution
        self._resolved_function_def: Optional[Any] = None  # Specialized function definition (set by CallRewriter)
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_function_call(self)

@dataclass
class MethodCall(Expression):
    """Method call (obj.method()) - method can be any expression for computed method calls"""
    object: Expression
    method_expr: Expression
    arguments: List[Expression]
    
    def __init__(self, object: Expression, method_expr: Expression, arguments: List[Expression], location: SourceLocation = None):
        super().__init__(NodeType.METHOD_CALL, location)
        self.object = object
        self.method_expr = method_expr
        self.arguments = arguments
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_method_call(self)

@dataclass
class MemberAccess(Expression):
    """Member access (obj.property or tuple.0)"""
    object: Expression
    property: Union[str, int]  # String for properties, int for tuple access
    
    def __init__(self, object: Expression, property: Union[str, int], location: SourceLocation = None):
        super().__init__(NodeType.MEMBER_ACCESS, location)
        self.object = object
        self.property = property
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_member_access(self)

@dataclass
class ModuleAccess(Expression):
    """Module access (module::function)"""
    object: Expression
    property: str
    
    def __init__(self, object: Expression, property: str, location: SourceLocation = None):
        super().__init__(NodeType.MODULE_ACCESS, location)
        self.object = object
        self.property = property
        
        # Module resolution metadata (set during ModulePass)
        self._resolved_module_key: Optional[str] = None  # Full module path
        self._defid: Optional[Any] = None  # DefId for module property dispatch
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_module_access(self)

@dataclass
class BinaryExpression(Expression):
    """Binary operation (a + b, a == b, etc.)"""
    left: Expression
    operator: BinaryOp  # Structured data, not string
    right: Expression
    
    def __str__(self) -> str:
        return f"({self.left} {self.operator.value} {self.right})"
    
    def __init__(self, left: Expression, operator: BinaryOp, right: Expression, location: SourceLocation = None):
        super().__init__(NodeType.BINARY_OP, location)
        self.left = left
        self.operator = operator
        self.right = right
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_binary_expression(self)

@dataclass
class UnaryExpression(Expression):
    """Unary operation (-x, !x)"""
    operator: UnaryOp  # Structured data, not string
    operand: Expression
    
    def __init__(self, operator: UnaryOp, operand: Expression, location: SourceLocation = None):
        super().__init__(NodeType.UNARY_OP, location)
        self.operator = operator
        self.operand = operand
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_unary_expression(self)


class CastExpression(Expression):
    """Type cast expression (x as T)"""
    expr: Expression
    target_type: 'Type'
    
    def __init__(self, expr: Expression, target_type: 'Type', location: SourceLocation = None):
        super().__init__(NodeType.CAST, location)
        self.expr = expr
        self.target_type = target_type
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_cast_expression(self)

@dataclass
class ArrayLiteral(Expression):
    """Array literal [1, 2, 3]"""
    elements: List[Expression]
    
    def __init__(self, elements: List[Expression], location: SourceLocation = None):
        super().__init__(NodeType.ARRAY_LITERAL, location)
        self.elements = elements
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_array_literal(self)


@dataclass
class UseStatement(Statement):
    """Use statement for imports and reexports"""
    path: List[str]
    is_function: bool = False
    is_wildcard: bool = False
    is_public: bool = False
    alias: Optional[str] = None
    
    def __init__(self, path: Union[str, List[str]], is_function: bool = False, is_wildcard: bool = False, is_public: bool = False, alias: Optional[str] = None, location: SourceLocation = None):
        super().__init__(NodeType.USE_STMT, location)
        if isinstance(path, str):
            self.path = path.split("::")
        else:
            self.path = path
        self.is_function = is_function
        self.is_wildcard = is_wildcard
        self.is_public = is_public
        self.alias = alias
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_use_statement(self)

@dataclass
class ModuleDeclaration(Statement):
    """Module declaration statement (mod name; or pub mod name;)"""
    name: str
    is_public: bool = False
    
    def __init__(self, name: str, is_public: bool = False, location: SourceLocation = None):
        super().__init__(NodeType.MOD_STMT, location)
        self.name = name
        self.is_public = is_public
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_module_declaration(self)

@dataclass
class InlineModule(Statement):
    """Inline module definition (mod name { ... })"""
    name: str
    body: List[Statement]
    is_public: bool = False
    
    def __init__(self, name: str, body: List[Statement], is_public: bool = False, location: SourceLocation = None):
        super().__init__(NodeType.INLINE_MOD_STMT, location)
        self.name = name
        self.body = body
        self.is_public = is_public
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_inline_module(self)

# Import proper Type from type system (no stub needed)
from .types import Type


class EinsteinClause:
    """
    One Einstein clause (indices + value + where + optional else to fill gap). Used by EinsteinDeclaration.
    When range is in LHS and where filters, else_expr is the value for indices that fail the guard.
    """
    __slots__ = ('indices', 'value', 'where_clause', 'else_expr', 'location', '_shape_info', '_range_info',
                 '_declaration_group', '_static_constraints', '_dynamic_constraints',
                 '_constraint_dependencies', '_iteration_constraints', '_reduction_constraints',
                 'ranges', 'metadata')

    def __init__(self, indices: List['Expression'], value: Expression,
                 where_clause: Optional['WhereClause'] = None,
                 else_expr: Optional['Expression'] = None,
                 location: Optional[SourceLocation] = None):
        self.indices = indices
        self.value = value
        self.where_clause = where_clause if where_clause is not None else WhereClause.empty()
        self.else_expr = else_expr
        self.location = location
        self._shape_info = None
        self._range_info = None
        self._declaration_group = None
        self._static_constraints = None
        self._dynamic_constraints = None
        self._constraint_dependencies = None
        self._iteration_constraints = None
        self._reduction_constraints = None
        self.ranges = None
        self.metadata = None

    @property
    def constraints(self) -> List['Expression']:
        return list(self.where_clause.constraints)

    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        """Visit clause – recurse into value and constraints."""
        self.value.accept(visitor)
        if self.where_clause:
            for c in self.where_clause.constraints:
                c.accept(visitor)


class EinsteinDeclaration(Statement):
    """
    Einstein notation array declaration. One or more clauses (same array).
    Grammar produces single clause; grouping merges multiples into one.
    """
    def __init__(self, array_name: str, clauses: List[EinsteinClause],
                 location: Optional[SourceLocation] = None):
        super().__init__(NodeType.EINSTEIN_DECL, location)
        self.array_name = array_name
        self.clauses = clauses
        self.statement_metadata: Optional[Any] = None
        self._shape_info: Optional['ShapeInfo'] = None
        self._range_info: Optional[object] = None
        self._coverage_info: Optional['CoverageResult'] = None
        self._execution_hints: Optional[Dict] = None
        self._declaration_group: Optional[object] = None
        self._static_constraints: Optional[List['Expression']] = None
        self._dynamic_constraints: Optional[List['Expression']] = None
        self._constraint_dependencies: Optional[Dict[str, set]] = None
        self._iteration_constraints: Optional[List['Expression']] = None
        self._reduction_constraints: Optional[List['Expression']] = None
        self.ranges: Optional[Dict[str, Any]] = None
        self.metadata: Optional[Dict[str, Any]] = None

    @property
    def indices(self):
        return self.clauses[0].indices if self.clauses else []

    @property
    def value(self):
        return self.clauses[0].value if self.clauses else None

    @property
    def where_clause(self):
        return self.clauses[0].where_clause if self.clauses else WhereClause.empty()

    @property
    def constraints(self) -> List['Expression']:
        return list(self.where_clause.constraints) if self.clauses else []

    def get_shape_info(self) -> 'ShapeInfo':
        if self._shape_info is None:
            raise RuntimeError("Shape analysis not run yet")
        return self._shape_info

    def set_shape_info(self, shape_info: 'ShapeInfo'):
        self._shape_info = shape_info

    def get_range_info(self) -> Optional[object]:
        return self._range_info

    def set_range_info(self, range_info: object):
        self._range_info = range_info

    def get_coverage_info(self) -> 'CoverageResult':
        if self._coverage_info is None:
            raise RuntimeError("Coverage analysis not run yet")
        return self._coverage_info

    def set_coverage_info(self, coverage_info: 'CoverageResult'):
        self._coverage_info = coverage_info

    def get_declaration_group(self) -> Optional[object]:
        return self._declaration_group

    def set_declaration_group(self, group: object):
        self._declaration_group = group

    def get_execution_hints(self) -> Optional[Dict]:
        return self._execution_hints

    def set_execution_hints(self, hints: Dict):
        self._execution_hints = hints

    def set_constraint_analysis(self, static_constraints: List['Expression'],
                               dynamic_constraints: List['Expression'],
                               constraint_dependencies: Dict[str, set]):
        self._static_constraints = static_constraints
        self._dynamic_constraints = dynamic_constraints
        self._constraint_dependencies = constraint_dependencies

    def set_separated_constraints(self, iteration_constraints: List['Expression'],
                                  reduction_constraints: List['Expression']):
        self._iteration_constraints = iteration_constraints
        self._reduction_constraints = reduction_constraints

    def get_constraint_analysis(self):
        return {
            'static': self._static_constraints,
            'dynamic': self._dynamic_constraints,
            'dependencies': self._constraint_dependencies,
            'iteration': self._iteration_constraints,
            'reduction': self._reduction_constraints
        }

    def has_analysis_info(self) -> bool:
        return self._shape_info is not None and self._coverage_info is not None

    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_einstein_declaration(self)


@dataclass
class EnumVariant:
    """Enum variant definition"""
    name: str
    fields: List[Union[str, 'Type']]  # Field names or types (for tuple-style variants)
    location: Optional[SourceLocation] = None


@dataclass
class EnumDefinition(Statement):
    """Enum definition (Algebraic Data Type)"""
    name: str
    variants: List[EnumVariant]
    generic_params: List[str] = None  # Generic type parameters: <T, E>
    is_public: bool = False
    
    def __init__(self, name: str, variants: List[EnumVariant], generic_params: Optional[List[str]] = None, is_public: bool = False, location: Optional[SourceLocation] = None):
        super().__init__(NodeType.ENUM_DEF, location)
        self.name = name
        self.variants = variants
        self.generic_params = generic_params if generic_params is not None else []
        self.is_public = is_public
        self._defid: Optional[Any] = None  # DefId for this enum
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_enum_definition(self)


@dataclass
class StructField:
    """Struct field definition"""
    name: str
    field_type: 'Type'
    location: Optional[SourceLocation] = None


@dataclass
class StructDefinition(Statement):
    """Struct definition (Product Type)"""
    name: str
    fields: List[StructField]  # For named structs: { x: f32, y: f32 }
    is_tuple_struct: bool = False  # True for tuple structs: (f32, f32)
    generic_params: List[str] = None  # Generic type parameters: <T>
    is_public: bool = False
    
    def __init__(self, name: str, fields: List[StructField], is_tuple_struct: bool = False, generic_params: Optional[List[str]] = None, is_public: bool = False, location: Optional[SourceLocation] = None):
        super().__init__(NodeType.STRUCT_DEF, location)
        self.name = name
        self.fields = fields
        self.is_tuple_struct = is_tuple_struct
        self.generic_params = generic_params if generic_params is not None else []
        self.is_public = is_public
        self._defid: Optional[Any] = None  # DefId for this struct
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_struct_definition(self)

@dataclass 
class RectangularAccess(Expression):
    """Rectangular array element access with tensor-style indices [i,j,k]"""
    base_expr: Expression
    indices: List[Expression]

    def __init__(self, base_expr: Expression, indices: List[Expression], location: SourceLocation = None):
        super().__init__(NodeType.RECTANGULAR_ACCESS, location)
        self.base_expr = base_expr
        self.indices = indices
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_rectangular_access(self)

@dataclass 
class JaggedAccess(Expression):
    """Jagged array element access with nested-style indices [i][j][k]"""
    base_expr: Expression
    index_chain: List[Expression]  # Chain of single indices for nested access

    def __init__(self, base_expr: Expression, index_chain: List[Expression], location: SourceLocation = None):
        super().__init__(NodeType.JAGGED_ACCESS, location)
        self.base_expr = base_expr
        self.index_chain = index_chain
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_jagged_access(self)

@dataclass
class OverClause:
    """Over clause for reductions"""
    range_groups: List['RangeGroup']

@dataclass
class RangeGroup:
    """Range group within over clause"""
    range_expr: Optional[Range]
    variables: List[str]

@dataclass(frozen=True)
class WhereClause:
    """
    Where clause containing constraint expressions - always present, never None.
    
    Constraints are arbitrary boolean expressions:
    - Binary relations: i < 10, x = 5, j in 1..10  (BinaryExpression)
    - Function calls: isPrime(x), isValid(y)        (FunctionCall)
    - Unary ops: !(x in excluded)                   (UnaryExpression)
    - Complex: (i + j) > 10 && isPrime(i)          (any Expression)
    
    LIFECYCLE:
    1. Parser: Creates with expressions
    2. Analysis Passes: Classify and annotate (store metadata on parent node)
    3. Executor: Reads generated IR (never sees WhereClause)
    
    Design Principle:
    - WhereClause is STRUCTURE ONLY (list of constraint expressions)
    - All analysis results go in StatementMetadata on parent statement
    - Passes iterate over constraints and classify them as needed
    
    Note: Not an ASTNode subclass, so doesn't inherit location.
    """
    constraints: tuple['Expression', ...] = ()
    
    @staticmethod
    def empty() -> 'WhereClause':
        """Create empty where clause (no constraints)"""
        return WhereClause(constraints=())
    
    @staticmethod
    def from_list(constraints: List['Expression']) -> 'WhereClause':
        """Create from list of constraint expressions"""
        return WhereClause(constraints=tuple(constraints))
    
    def is_empty(self) -> bool:
        """Check if there are no constraints"""
        return len(self.constraints) == 0
    
    def has_constraints(self) -> bool:
        """Check if there are constraints"""
        return len(self.constraints) > 0
    
    def count(self) -> int:
        """Number of constraints"""
        return len(self.constraints)
    
    def __iter__(self):
        """Allow iteration: for expr in where_clause"""
        return iter(self.constraints)
    
    def __len__(self):
        """Allow len(where_clause)"""
        return len(self.constraints)
    
    def __repr__(self):
        if self.is_empty():
            return "WhereClause(empty)"
        return f"WhereClause({len(self.constraints)} constraints)"

# Constraints are now just Expression objects (no wrapper needed).
# Where clauses and array comprehensions use List[Expression] directly.
#
# Typical constraint expressions:
# - BinaryExpression: i < 10, x = 5, j in 1..10
# - FunctionCall: isPrime(x), isValid(y)
# - UnaryExpression: !(x in excluded)
# - Complex: any boolean expression
#
# Constraint classification is now stored in pass-specific data structures
# or as metadata on the parent statement, not on individual constraint nodes.
# ============================================================================

@dataclass
class WhereExpression(Expression):
    """Expression with where clause"""
    expr: Expression
    where_clause: WhereClause
    
    def __init__(self, expr: Expression, where_clause: WhereClause, location: SourceLocation = None):
        super().__init__(NodeType.WHERE_EXPR, location)
        self.expr = expr
        self.where_clause = where_clause
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_where_expression(self)

@dataclass
class TupleExpression(Expression):
    """Tuple expression"""
    elements: List[Expression]
    
    def __init__(self, elements: List[Expression], location: SourceLocation = None):
        super().__init__(NodeType.TUPLE_EXPR, location)
        self.elements = elements
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_tuple_expression(self)

@dataclass
class ArrayComprehension(Expression):
    """
    Array comprehension expression.
    
    Constraints are arbitrary boolean expressions:
    - Binary relations: x in 1..10, i < j, x = 5
    - Function calls: isPrime(x), isValid(y)
    - Unary ops: !(x in excluded)
    - Complex: (i + j) > 10
    
    Examples:
    [x | x in 1..10, x > 5]           # Both BinaryExpression
    [x | x in 1..10, isPrime(x)]      # BinaryExpression + FunctionCall
    [x | x in 1..10, !(x in arr)]     # BinaryExpression + UnaryExpression
    """
    expr: Expression
    constraints: List['Expression'] = None
    
    def __init__(self, expr: Expression, constraints: List['Expression'], location: SourceLocation = None):
        super().__init__(NodeType.ARRAY_COMPREHENSION, location)
        self.expr = expr
        self.constraints = constraints
        
        # Analysis metadata (static attributes - always present)
        self.statement_metadata: Optional[Any] = None  # StatementMetadata from ConstraintClassifier
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_array_comprehension(self)

@dataclass
class ReductionExpression(Expression):
    """Reduction operation with Einstein notation"""
    function_name: str
    body: Expression
    over_clause: OverClause
    where_clause: WhereClause  # Always present, use WhereClause.empty() if no constraints

    def __init__(self, function_name: str, body: Expression, over_clause: OverClause, 
                 where_clause: Optional[WhereClause] = None, location: SourceLocation = None):
        super().__init__(NodeType.REDUCTION_EXPR, location)
        self.function_name = function_name
        self.body = body
        self.over_clause = over_clause
        self.where_clause = where_clause or WhereClause.empty()  # Never None!
        
        # Where constraints attached from outer scope (set by type analysis)
        self._where_constraints: List[Expression] = []
    
        # Metadata for analysis passes (e.g., rest pattern preprocessing)
        self.metadata: Optional[Dict[str, Any]] = None
    
    def has_where_clause(self) -> bool:
        """Check if has non-empty where clause"""
        return self.where_clause.has_constraints()
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_reduction_expression(self)

@dataclass 
class IfExpression(Expression):
    """If expression that returns a value
    
    Uses BlockExpression for then/else branches to unify the statements+expr pattern.
    """
    condition: Expression
    then_block: 'BlockExpression'
    else_block: Optional['BlockExpression']

    def __init__(self, condition: Expression, then_block: 'BlockExpression',
                 else_block: Optional['BlockExpression'] = None,
                 location: SourceLocation = None):
        super().__init__(NodeType.IF_EXPR, location)
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_if_expression(self)

@dataclass
class PipelineExpression(Expression):
    """Pipeline expression (x |> f |> g)"""
    left: Expression
    operator: 'BinaryOp'  # "|>", "?>", "!>" - type-safe enum
    right: Expression
    else_clause: Optional[Expression] = None  # Optional "else" fallback for Option pipelines
    catch_clause: Optional[Expression] = None  # Optional "catch" handler for Result pipelines
    
    def __init__(self, left: Expression, operator: 'BinaryOp', right: Expression, 
                 else_clause: Optional[Expression] = None,
                 catch_clause: Optional[Expression] = None,
                 location: SourceLocation = None):
        super().__init__(NodeType.PIPELINE_EXPR, location)
        self.left = left
        self.operator = operator
        self.right = right
        self.else_clause = else_clause
        self.catch_clause = catch_clause
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_pipeline_expression(self)

@dataclass
class BlockExpression(Expression):
    """Block expression that can be used as a value: { statements; final_expr }"""
    statements: List[Statement]
    final_expr: Optional[Expression]
    
    def __init__(self, statements: List[Statement], final_expr: Optional[Expression], location: SourceLocation = None):
        super().__init__(NodeType.BLOCK_EXPR, location)
        self.statements = statements
        self.final_expr = final_expr
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_block_expression(self)

@dataclass
class LambdaExpression(Expression):
    """Lambda expression (|x| x * 2), multiple params (|x, y| x + y), or parameterless (|| expr)"""
    parameters: List[str]  # Empty list for parameterless lambdas
    body: Expression  # Can be a simple expression or a BlockExpression
    
    def __init__(self, parameters: List[str], body: Expression, location: SourceLocation = None):
        super().__init__(NodeType.LAMBDA_EXPR, location)
        self.parameters = parameters
        self.body = body
        
        self._defid: Optional[Any] = None  # DefId for this lambda
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_lambda_expression(self)

@dataclass
class TryExpression(Expression):
    """Try expression: try operation"""
    operand: Expression
    
    def __init__(self, operand: Expression, location: SourceLocation = None):
        super().__init__(NodeType.TRY_EXPR, location)
        self.operand = operand
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_try_expression(self)

@dataclass
class AnnotatedVariable:
    """Variable with optional type annotation: x or x: i32"""
    name: str
    type_annotation: Optional['Type'] = None

@dataclass
class TupleDestructurePattern(ASTNode):
    """Tuple destructuring pattern: (x, y, z) or (x: i32, y: str)"""
    variables: List[AnnotatedVariable]
    
    def __init__(self, variables: List[AnnotatedVariable], location: SourceLocation = None):
        super().__init__(NodeType.TUPLE_DESTRUCTURE_PATTERN, location)
        self.variables = variables
    

# =====================================================================
# ARROW EXPRESSIONS FOR DL GRAPH CONSTRUCTION
# =====================================================================

@dataclass
class ArrowExpression(Expression):
    """
    Arrow operator expression for ML/DL computation graphs.
    
    Note: Arrows compose FUNCTIONS only, not data. The composed function is then applied to data separately.
    
    Supports four types of composition:
    - SEQUENTIAL (>>>): A >>> B >>> C - sequential composition (output of A -> input of B)
    - PARALLEL (***): A *** B *** C - parallel composition (different inputs to each: (a, b, c) -> (A(a), B(b), C(c)))
    - FANOUT (&&&): A &&& B &&& C - fanout composition (same input to all: x -> (A(x), B(x), C(x)))
    - CHOICE (|||): A ||| B ||| C - choice composition (conditional routing)
    """
    operator: ArrowOperator
    components: List[Expression]
    
    def __init__(self, operator: ArrowOperator, components: List[Expression], location: SourceLocation = None):
        super().__init__(NodeType.ARROW_EXPR, location)
        self.operator = operator
        self.components = components
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_arrow_expression(self)

# =====================================================================
# PATTERN MATCHING - PATTERNS AND MATCH EXPRESSIONS
# =====================================================================

class Pattern(ASTNode):
    """Base class for match patterns"""
    pass

@dataclass
class LiteralPattern(Pattern):
    """Literal pattern: matches exact value (1, "hello", true)"""
    value: Literal  # Use existing Literal node
    
    def __init__(self, value: Literal, location: SourceLocation = None):
        super().__init__(NodeType.LITERAL_PATTERN, location)
        self.value = value
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_literal_pattern(self)

@dataclass
class IdentifierPattern(Pattern):
    """Identifier pattern: binds value to variable (x, name)"""
    name: str
    
    def __init__(self, name: str, location: SourceLocation = None):
        super().__init__(NodeType.IDENTIFIER_PATTERN, location)
        self.name = name
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_identifier_pattern(self)

@dataclass
class WildcardPattern(Pattern):
    """Wildcard pattern: matches anything (_)"""
    
    def __init__(self, location: SourceLocation = None):
        super().__init__(NodeType.WILDCARD_PATTERN, location)
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_wildcard_pattern(self)

@dataclass
class TuplePattern(Pattern):
    """Tuple pattern: matches and destructures tuples ((a, b), (x, y, z))"""
    patterns: List[Pattern]
    
    def __init__(self, patterns: List[Pattern], location: SourceLocation = None):
        super().__init__(NodeType.TUPLE_PATTERN, location)
        self.patterns = patterns
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_tuple_pattern(self)

@dataclass
class RestPattern(Pattern):
    """Rest pattern: ..pattern (binds remaining elements to pattern)"""
    pattern: Pattern  # Sub-pattern (usually IdentifierPattern, but could be wildcard, etc.)
    
    def __init__(self, pattern: Pattern, location: SourceLocation = None):
        super().__init__(NodeType.REST_PATTERN, location)
        self.pattern = pattern
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_rest_pattern(self)

@dataclass
class ArrayPattern(Pattern):
    """
    Array pattern: matches arrays ([], [x], [first, ..rest], [..rest, last], [first, ..rest, last])
    
    Rest patterns are part of the patterns list, not a separate field.
    The patterns list can contain RestPattern nodes at any position (beginning, middle, or end).
    """
    patterns: List[Pattern]  # Element patterns (can include RestPattern at any position)
    
    def __init__(self, patterns: List[Pattern], location: SourceLocation = None):
        super().__init__(NodeType.ARRAY_PATTERN, location)
        self.patterns = patterns
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_array_pattern(self)

@dataclass
class GuardPattern(Pattern):
    """Guard pattern: pattern with where clause (x where x > 0)"""
    pattern: Pattern
    guard: Expression  # Guard condition
    
    def __init__(self, pattern: Pattern, guard: Expression, location: SourceLocation = None):
        super().__init__(NodeType.GUARD_PATTERN, location)
        self.pattern = pattern
        self.guard = guard
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_guard_pattern(self)


@dataclass
class ConstructorPattern(Pattern):
    """Constructor pattern: matches enum/struct constructors (Circle(r), Point { x, y })"""
    constructor_name: str
    patterns: List[Pattern]  # Patterns for constructor fields
    is_struct_literal: bool = False  # True for struct literals: Point { x, y }
    
    def __init__(self, constructor_name: str, patterns: List[Pattern], is_struct_literal: bool = False, location: SourceLocation = None):
        super().__init__(NodeType.CONSTRUCTOR_PATTERN, location)
        self.constructor_name = constructor_name
        self.patterns = patterns
        self.is_struct_literal = is_struct_literal
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_constructor_pattern(self)

@dataclass
class MatchArm(ASTNode):
    """Match arm: pattern => expression"""
    pattern: Pattern
    body: Expression  # Can be BlockExpression
    
    def __init__(self, pattern: Pattern, body: Expression, location: SourceLocation = None):
        super().__init__(NodeType.MATCH_EXPR, location)  # Reuse MATCH_EXPR for now
        self.pattern = pattern
        self.body = body

@dataclass
class MatchExpression(Expression):
    """Match expression: match expr { arms }"""
    scrutinee: Expression  # Value to match against
    arms: List[MatchArm]
    
    def __init__(self, scrutinee: Expression, arms: List[MatchArm], location: SourceLocation = None):
        super().__init__(NodeType.MATCH_EXPR, location)
        self.scrutinee = scrutinee
        self.arms = arms
    
    def accept(self, visitor: 'ASTVisitor[T]') -> 'T':
        return visitor.visit_match_expression(self)
