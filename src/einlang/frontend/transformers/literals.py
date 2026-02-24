"""
Literal Parser - Extracted from EinlangTransformer
Handles parsing of all literal types (numbers, strings, booleans)
"""

from typing import Union
from ...shared import Literal, InterpolatedString, SourceLocation
from ...utils.config import (
    STRING_QUOTE_CHAR, BOOLEAN_TRUE_LITERAL, BOOLEAN_FALSE_LITERAL,
    DECIMAL_SEPARATOR, SCIENTIFIC_NOTATION_INDICATOR
)
from .string_interpolation import StringInterpolationParser
from ...utils.base import handle_token

class LiteralParser:
    """Dedicated parser for literal values using polymorphic token handling"""
    
    @staticmethod
    def parse(token, location: SourceLocation) -> Union[Literal, InterpolatedString]:
        """Parse literal token into appropriate AST node"""
        parser = LiteralParser()
        
        # Use polymorphic token dispatch
        token_info = handle_token(token)
        if token_info.get('is_terminal') and 'type' in token_info:
            return parser._parse_typed_token_info(token, token_info, location)
        
        # Handle string tokens
        return parser._parse_string_token(str(token), location)
    
    def _parse_typed_token_info(self, token, token_info: dict, location: SourceLocation) -> Union[Literal, InterpolatedString]:
        """Parse tokens using polymorphic token information"""
        token_type = token_info.get('type', 'unknown')
        token_value = token_info.get('value', str(token))
        
        if token_type == 'INTEGER_OR_FLOAT':
            return self._parse_number(token_value, location)
        elif token_type == 'STRING':
            return self._parse_string(token_value, location)
        elif token_type == 'TRUE':
            return Literal(value=True, location=location)
        elif token_type == 'FALSE':
            return Literal(value=False, location=location)
        else:
            return Literal(value=token_value, location=location)
    
    def _parse_string_token(self, token_str: str, location: SourceLocation) -> Union[Literal, InterpolatedString]:
        """Parse string token without type information"""
        # Handle boolean literals
        if token_str == BOOLEAN_TRUE_LITERAL:
            return Literal(value=True, location=location)
        elif token_str == BOOLEAN_FALSE_LITERAL:
            return Literal(value=False, location=location)
        
        # Handle quoted strings
        if token_str.startswith(STRING_QUOTE_CHAR) and token_str.endswith(STRING_QUOTE_CHAR):
            return self._parse_string(token_str, location)
        
        # Try to parse as number
        try:
            return self._parse_number(token_str, location)
        except ValueError:
            return Literal(value=token_str, location=location)
    
    def _parse_number(self, value_str: str, location: SourceLocation) -> Literal:
        """Parse numeric literal"""
        if DECIMAL_SEPARATOR in value_str or SCIENTIFIC_NOTATION_INDICATOR in value_str.lower():
            return Literal(value=float(value_str), location=location)
        else:
            return Literal(value=int(value_str), location=location)
    
    def _parse_string(self, quoted_str: str, location: SourceLocation) -> Union[Literal, InterpolatedString]:
        """Parse string literal (potentially with interpolation)"""
        # Remove quotes
        clean_value = quoted_str[1:-1] if quoted_str.startswith(STRING_QUOTE_CHAR) and quoted_str.endswith(STRING_QUOTE_CHAR) else quoted_str
        
        # Use dedicated string interpolation parser
        return StringInterpolationParser.parse(clean_value, location)
