"""
Type System

Rust Pattern: rustc_middle::ty::Ty
Reference: TYPE_SYSTEM_DESIGN.md

Convention: type_info on IR expressions carries only precision (e.g. i32, f32) and,
for arrays, rank (number of dimensions). It must NOT carry concrete shapes; shapes
come from shape analysis and lowered IR (see IR_DESIGN.md).
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Generic, TypeVar, Any
from abc import ABC, abstractmethod
from enum import Enum


class TypeKind(Enum):
    """
    Type kind (Rust pattern: rustc_middle::ty::TyKind).
    
    Rust Pattern: rustc_middle::ty::TyKind enum
    """
    PRIMITIVE = "primitive"  # i32, f32, bool, etc.
    ARRAY = "array"
    RECTANGULAR = "rectangular"  # Regular tensor: [T; N, M, K]
    JAGGED = "jagged"            # Ragged/jagged arrays: jagged[T; depth]
    TUPLE = "tuple"
    FUNCTION = "function"  # Arrow type (function type)
    UNKNOWN = "unknown"


T = TypeVar('T')


@dataclass(frozen=True)
class Type:
    """
    Type representation (Rust pattern: rustc_middle::ty::Ty).
    
    Rust Pattern: rustc_middle::ty::Ty
    
    Implementation Alignment: Follows Rust's type representation:
    - Immutable (frozen dataclass)
    - All types tracked (including function types)
    - Type information stored in IR nodes
    - Visitor pattern support (accept method for type matching)
    
    Reference: `rustc_middle::ty::Ty` structure with visitor pattern
    """
    kind: TypeKind
    
    def accept(self, visitor: 'TypeVisitor[T]') -> T:
        """
        Accept type visitor (Rust pattern: rustc_middle::ty::TypeVisitor).
        
        Rust Pattern: Visitor pattern for type matching (no isinstance)
        
        Implementation Alignment: Follows Rust's type visitor pattern:
        - Type-safe dispatch based on kind
        - No isinstance checks needed
        - Visitor pattern for all type operations
        
        Reference: `rustc_middle::ty::TypeVisitor` pattern
        """
        # Dictionary dispatch for type visitor (visitor pattern, replaces if/elif chain)
        _type_visitor_dispatch = {
            TypeKind.FUNCTION: lambda: visitor.visit_function_type(self),  # type: ignore
            TypeKind.PRIMITIVE: lambda: visitor.visit_primitive_type(self),  # type: ignore
            TypeKind.ARRAY: lambda: visitor.visit_array_type(self),  # type: ignore
            TypeKind.TUPLE: lambda: visitor.visit_tuple_type(self),  # type: ignore
        }
        
        handler = _type_visitor_dispatch.get(self.kind)
        if handler:
            return handler()
        else:
            return visitor.visit_unknown_type(self)  # type: ignore


@dataclass(frozen=True)
class PrimitiveType(Type):
    """Primitive type (i32, f32, bool, etc.)"""
    name: str  # "i32", "f32", "bool", etc.
    
    def __init__(self, name: str):
        super().__init__(kind=TypeKind.PRIMITIVE)
        object.__setattr__(self, 'name', name)
    
    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other):
        """Compare primitive types by name"""
        if not isinstance(other, PrimitiveType):
            return False
        return self.name == other.name
    
    def __hash__(self):
        """Hash primitive types by name"""
        return hash(('PrimitiveType', self.name))


@dataclass(frozen=True)
class FunctionType(Type):
    """
    Function type (arrow type) - Rust pattern: rustc_middle::ty::FnSig.
    
    Rust Pattern: rustc_middle::ty::FnSig (function signature)
    
    Implementation Alignment: Follows Rust's function type representation:
    - Parameter types tracked
    - Return type tracked
    - Function types preserved through type inference
    - Arrow expressions return FunctionType
    
    Reference: `rustc_middle::ty::FnSig` for function signatures
    
    Note: This is the "arrow type" - functions have FunctionType
    """
    param_types: Tuple[Type, ...]  # Parameter types
    return_type: Type  # Return type
    
    def __init__(self, param_types: Tuple[Type, ...], return_type: Type):
        super().__init__(kind=TypeKind.FUNCTION)
        object.__setattr__(self, 'param_types', param_types)
        object.__setattr__(self, 'return_type', return_type)
    
    def __str__(self) -> str:
        """Format as arrow type: (T1, T2) -> T3"""
        params = ", ".join(str(t) for t in self.param_types)
        return f"({params}) -> {self.return_type}"


@dataclass(frozen=True)
class TupleType(Type):
    """Tuple type"""
    element_types: Tuple[Type, ...]
    
    def __init__(self, element_types: Tuple[Type, ...]):
        super().__init__(kind=TypeKind.TUPLE)
        object.__setattr__(self, 'element_types', element_types)


@dataclass(frozen=True)
class RectangularType(Type):
    """
    Regular tensor type: [T; N, M, K]
    
    Supports Einstein notation.
    Access pattern: A[i, j, k]
    """
    element_type: Type
    shape: Optional[Tuple] = None
    is_dynamic_rank: bool = False
    
    def __init__(self, element_type: Type, 
                 shape: Optional[Tuple] = None, 
                 is_dynamic_rank: bool = False):
        super().__init__(kind=TypeKind.RECTANGULAR)
        object.__setattr__(self, 'element_type', element_type)
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'is_dynamic_rank', is_dynamic_rank)
    
    def __str__(self):
        if self.is_dynamic_rank:
            return f"[{self.element_type}; *]"
        elif self.shape is None:
            return f"[{self.element_type}]"
        else:
            dims = ', '.join('?' if d is None else str(d) for d in self.shape)
            return f"[{self.element_type}; {dims}]"
    
    def __eq__(self, other):
        if not isinstance(other, RectangularType):
            return NotImplemented
        return (self.element_type == other.element_type and 
                self.shape == other.shape and 
                self.is_dynamic_rank == other.is_dynamic_rank)
    
    def __hash__(self):
        return hash((self.kind, self.element_type, self.shape, self.is_dynamic_rank))


@dataclass(frozen=True)
class JaggedType(Type):
    """
    Jagged/ragged array type: jagged[T; depth]
    
    Does NOT support Einstein notation.
    Access pattern: A[i][j][k]
    """
    element_type: Type
    nesting_depth: Optional[int] = 1
    is_dynamic_depth: bool = False
    
    def __init__(self, element_type: Type,
                 nesting_depth: Optional[int] = 1, 
                 is_dynamic_depth: bool = False):
        super().__init__(kind=TypeKind.JAGGED)
        object.__setattr__(self, 'element_type', element_type)
        object.__setattr__(self, 'nesting_depth', nesting_depth)
        object.__setattr__(self, 'is_dynamic_depth', is_dynamic_depth)
    
    def __str__(self):
        if self.is_dynamic_depth:
            return f"jagged[{self.element_type}; ?]"
        elif self.nesting_depth == 1:
            return f"jagged[{self.element_type}]"
        else:
            return f"jagged[{self.element_type}; {self.nesting_depth}d]"
    
    def __eq__(self, other):
        if not isinstance(other, JaggedType):
            return NotImplemented
        return (self.element_type == other.element_type and
                self.nesting_depth == other.nesting_depth and
                self.is_dynamic_depth == other.is_dynamic_depth)
    
    def __hash__(self):
        return hash((self.kind, self.element_type, self.nesting_depth, self.is_dynamic_depth))


class TypeVisitor(ABC, Generic[T]):
    """
    Type visitor pattern (Rust pattern: rustc_middle::ty::TypeVisitor).
    
    Rust Pattern: rustc_middle::ty::TypeVisitor
    
    Implementation Alignment: Follows Rust's type visitor pattern:
    - Visitor pattern for type matching (no isinstance)
    - Type-safe dispatch
    - Extensible (add new visitors without changing Type)
    
    Reference: `rustc_middle::ty::TypeVisitor` for type operations
    """
    
    @abstractmethod
    def visit_function_type(self, ty: FunctionType) -> T:
        """Visit function type"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_primitive_type(self, ty: PrimitiveType) -> T:
        """Visit primitive type"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_rectangular_type(self, ty: RectangularType) -> T:
        """Visit rectangular array type"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_jagged_type(self, ty: JaggedType) -> T:
        """Visit jagged array type"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_tuple_type(self, ty: TupleType) -> T:
        """Visit tuple type"""
        raise NotImplementedError
    
    @abstractmethod
    def visit_unknown_type(self, ty: Type) -> T:
        """Visit unknown type"""
        raise NotImplementedError


# Common primitive types (All standard types as constants)
I32 = PrimitiveType("i32")
I64 = PrimitiveType("i64")
F32 = PrimitiveType("f32")
F64 = PrimitiveType("f64")
BOOL = PrimitiveType("bool")
STR = PrimitiveType("str")
UNKNOWN = Type(kind=TypeKind.UNKNOWN)

# Literal type for range objects (iterable, not a value type - used for loop bounds)
RANGE = PrimitiveType("range")

# Unit type (void-like; for print, assert)
UNIT = PrimitiveType("unit")


def infer_literal_type(value: Any) -> Type:
    """
    Infer compile-time type for a literal value.
    Integer default i32, float default f32 (Rust-like).
    All literals must have type_info at compile time.
    """
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, int):
        return I32
    if isinstance(value, float):
        return F32
    if isinstance(value, str):
        return STR
    if isinstance(value, range):
        return RANGE
    if value is None:
        return UNKNOWN
    return UNKNOWN


# ============================================================================
# AST Operator Enums (for AST nodes, not type system)
# ============================================================================

class BinaryOp(Enum):
    """Binary operators - compile-time checked enum"""
    # Arithmetic
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    POW = "**"
    
    # Comparison
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    
    # Logical
    AND = "&&"
    OR = "||"
    
    # Assignment
    ASSIGN = "="
    
    # Range
    IN = "in"
    
    # Pipeline
    PIPE = "|>"
    PIPE_OPTIONAL = "?>"
    PIPE_ERROR = "!>"


class UnaryOp(Enum):
    """Unary operators - compile-time checked enum"""
    NOT = "not"
    BOOL_NOT = "!"
    NEG = "-"
    POS = "+"


class PipelineClauseType(Enum):
    """Pipeline clause types - for else/catch clauses"""
    ELSE = "else"
    CATCH = "catch"

