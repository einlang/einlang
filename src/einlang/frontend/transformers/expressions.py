"""
Expression Parser - Extracted from EinlangTransformer
Handles parsing of binary expressions and operators
"""

from typing import Any, Callable
from typing_extensions import TypeAlias
from lark.lexer import Token
from ...shared import BinaryExpression, BinaryOp, SourceLocation, ASTNode

# Type aliases for better clarity
LarkMeta: TypeAlias = Any  # Lark's internal Meta object
LocationExtractor: TypeAlias = Callable[[LarkMeta], SourceLocation]

class BinaryExpressionParser:
    """Dedicated parser for binary expressions"""
    
    def __init__(self, location_extractor: LocationExtractor) -> None:
        self.extract_location = location_extractor
    
    def parse_logical_or(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Parse logical OR expression"""
        return self._create_binary_expression(meta, left, operator, right)
    
    def parse_logical_and(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Parse logical AND expression"""
        return self._create_binary_expression(meta, left, operator, right)
    
    def parse_equality(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Parse equality expression (==, !=)"""
        return self._create_binary_expression(meta, left, operator, right)
    
    def parse_relational(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Parse relational expression (<, >, <=, >=)"""
        return self._create_binary_expression(meta, left, operator, right)
    
    def parse_additive(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Parse additive expression (+, -)"""
        return self._create_binary_expression(meta, left, operator, right)
    
    def parse_multiplicative(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Parse multiplicative expression (*, /, %)"""
        return self._create_binary_expression(meta, left, operator, right)
    
    def parse_power(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Parse power expression (**)"""
        return self._create_binary_expression(meta, left, operator, right)
    
    def _create_binary_expression(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Create a binary expression with location information"""
        location = self.extract_location(meta)
        # Trust: Lark Token has column attribute
        location = SourceLocation(
                file=location.file,
                line=location.line,
                column=operator.column
            )
        # Parser converts token to enum directly
        operator_enum = BinaryOp(str(operator))
        return BinaryExpression(
            left=left,
            operator=operator_enum,
            right=right,
            location=location
        )
