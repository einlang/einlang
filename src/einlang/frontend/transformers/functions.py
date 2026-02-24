"""
Function Definition Parser - Extracted from EinlangTransformer
Handles parsing of function definitions and parameters
"""

from typing import List, Optional, Dict, Any, Tuple, Union, Callable
from typing_extensions import TypeAlias
from lark.lexer import Token
from ...shared import FunctionDefinition, Parameter, SourceLocation, ASTNode, BlockExpression
from ...utils.base import handle_token

# Type aliases for better clarity
LarkMeta: TypeAlias = Any  # Lark's internal Meta object
TypeAnnotation: TypeAlias = Union['PrimitiveType', 'RectangularType', 'ListType', ASTNode]
LocationExtractor: TypeAlias = Callable[[LarkMeta], SourceLocation]

class FunctionDefinitionParser:
    """Dedicated parser for function definitions with polymorphic token handling"""
    
    def __init__(self, location_extractor: LocationExtractor) -> None:
        self.extract_location = location_extractor
    
    def parse_function_definition(self, meta: LarkMeta, name: Token, lpar: Token, *args: Union[List[Parameter], Dict[str, Any], TypeAnnotation, Token], is_public: bool = False) -> FunctionDefinition:
        """Parse function definition with common logic for pub and private functions"""
        location = self.extract_location(meta)
        
        # Parse function components
        params, return_type, body_block = self._parse_function_args(args)
        
        # Extract function body as BlockExpression
        body = self._extract_function_body(body_block)
        
        return FunctionDefinition(
            name=str(name),
            parameters=params,
            return_type=return_type,
            body=body,
            is_public=is_public,
            location=location
        )
    
    def _parse_function_args(self, args: Tuple[Union[List[Parameter], 'BlockExpression', TypeAnnotation, Token], ...]) -> Tuple[List[Parameter], Optional[TypeAnnotation], Optional['BlockExpression']]:
        """Parse function arguments: param_list?, RPAR, (return_type)?, block"""
        from ...shared.nodes import BlockExpression
        params: List[Parameter] = []
        return_type: Optional[TypeAnnotation] = None
        body_block: Optional[BlockExpression] = None
        
        for item in args:
            # Use explicit type checking with protocols instead of hasattr/isinstance
            if self._is_rpar_token(item):
                continue  # Skip RPAR token
            elif self._is_parameter_list(item, params):
                # First list should be parameters
                params = item
            elif self._is_type_annotation(item):
                # Type annotation for return type
                return_type = item
            elif self._is_body_block(item):
                # This is the block
                body_block = item
        
        return params, return_type, body_block
    
    def _extract_function_body(self, body_block: Optional['BlockExpression']) -> 'BlockExpression':
        """Extract function body as BlockExpression"""
        from ...shared.nodes import BlockExpression
        if not body_block:
            return BlockExpression([], None)
        
        # body_block is already a BlockExpression
        return body_block
    
    def _is_rpar_token(self, item: Any) -> bool:
        """Check if item is an RPAR token using polymorphic dispatch"""
        token_info = handle_token(item)
        return token_info.get('type') == 'RPAR'
    
    def _is_parameter_list(self, item: Any, current_params: List[Parameter]) -> bool:
        """Check if item is a parameter list"""
        return len(current_params) == 0 and type(item) is list
    
    def _is_body_block(self, item: Any) -> bool:
        """Check if item is a BlockExpression"""
        from ...shared.nodes import BlockExpression
        return isinstance(item, BlockExpression)
    
    def _is_type_annotation(self, item: Any) -> bool:
        """Check if item is a type annotation - explicit type checking"""
        # Fast fail approach - explicit checks only
        item_type = type(item)
        if item_type in (list, dict, str, int, float, bool):
            return False
        
        # Check class name for type annotations
        class_name = item_type.__name__
        return 'Type' in class_name

class ParameterParser:
    """Dedicated parser for function parameters"""
    
    def __init__(self, location_extractor: LocationExtractor) -> None:
        self.extract_location = location_extractor
    
    def parse_parameter(self, meta: LarkMeta, name: Token, type_annotation: Optional[TypeAnnotation] = None) -> Parameter:
        """Parse a single function parameter"""
        return Parameter(name=str(name), type_annotation=type_annotation)
