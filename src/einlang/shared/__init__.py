"""
Shared components for architecture.

Rust Pattern: Shared foundational types and utilities
"""

from .defid import DefId, Resolver
from .source_location import SourceLocation
from .errors import Error, ErrorReporter, EinlangError, EinlangSourceError, EinlangImplementationError
from .types import (
    Type, TypeKind, PrimitiveType, FunctionType, RectangularType, JaggedType, TupleType, TypeVisitor,
    I32, F32, F64, I64, BOOL, STR, UNKNOWN, UNIT,
    BinaryOp, UnaryOp, PipelineClauseType,
)
from .nodes import (
    ASTNode, Expression, Statement, Program, NodeType,
    FunctionDefinition, VariableDeclaration, ExpressionStatement,
    UseStatement, ModuleDeclaration, InlineModule, EinsteinDeclaration,
    Literal, Identifier, IndexVar, IndexRest, FunctionCall, MethodCall,
    MemberAccess, ModuleAccess, BinaryExpression, UnaryExpression, CastExpression,
    ArrayLiteral, InterpolatedString, InterpolationPart,
    RectangularAccess, JaggedAccess, ArrayComprehension, ReductionExpression,
    WhereExpression, TupleExpression, IfExpression, PipelineExpression,
    LambdaExpression, TryExpression, BlockExpression, Range,
    MatchExpression, MatchArm,
    Parameter, TupleDestructurePattern, AnnotatedVariable,
    Pattern, LiteralPattern, IdentifierPattern, WildcardPattern,
    TuplePattern, ArrayPattern, RestPattern, GuardPattern,
    WhereClause, OverClause, RangeGroup,
)
from .ast_visitor import ASTVisitor

# Type aliases for compatibility
from typing import Any
ExpressionValue = Any  # Simplified type for utils/base.py compatibility
