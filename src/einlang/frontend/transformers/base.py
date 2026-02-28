"""
Einlang Parser - Applying Lark Best Practices (Working Version)

Key improvements applied:
1. ✅ Native caching (cache=True) instead of manual caching  
2. ✅ Standard terminals (%import common.ESCAPED_STRING)
3. ✅ Grammar aliases for key rules (-> alias_name)
4. ✅ Removed unnecessary terminal methods
5. ✅ Simplified grammar where possible
"""

from lark import Transformer, v_args
from lark.lexer import Token
from typing import List, Optional, Union, Any, Dict, Tuple
from typing_extensions import TypeAlias
from dataclasses import dataclass
import logging

from ...shared import *
from ...shared import BlockExpression  # Ensure BlockExpression is available
from ...shared.types import BinaryOp, PipelineClauseType
from ...utils.base import handle_token, extract_location_info

@dataclass
class ImportPathInfo:
    """Internal type for import path information"""
    path: List[str]
    is_wildcard: bool = False
    is_function: bool = False
    function_names: Optional[List[str]] = None  # For {func1, func2} syntax (legacy)
    function_items: Optional[List[Tuple[str, Optional[str]]]] = None  # For {func1 as f1, func2} syntax with aliases


@dataclass
class ElseClauseResult:
    """Result of else_clause parsing - use instead of dict"""
    type: str  # 'else_block' or 'else_if'
    statements: Optional[List[ASTNode]] = None
    final_expr: Optional[ASTNode] = None
    if_expr: Optional[Any] = None  # IfExpression when type=='else_if'

from .literals import LiteralParser
from .functions import FunctionDefinitionParser, ParameterParser
from .expressions import BinaryExpressionParser

# Precise types for better type safety
ParserToken: TypeAlias = Union[str, int, float, bool]
ParseResult: TypeAlias = Union[ASTNode, List[ASTNode]]
# Lark Meta object contains location information - use more specific type
LarkMeta: TypeAlias = Union[None, object]  # More specific than Any
TypeAnnotation: TypeAlias = Union['PrimitiveType', 'RectangularType']  # Updated for unified type system
# Forward references for the unified type system

# Module-level type annotations for clarity
logger: logging.Logger = logging.getLogger(__name__)

"""
Einlang AST Transformer
Converts Lark parse tree to Einlang AST nodes
"""


@v_args(inline=True, meta=True)
class EinlangTransformer(Transformer):
    """
    Einlang AST Transformer with improved error reporting
    
    Catches missing transformer methods and provides clear error messages
    instead of cryptic Tree object errors.
    """
    
    def _call_userfunc(self, tree, new_children=None):
        """Override to provide better error messages for missing transformer methods and signature mismatches"""
        try:
            return super()._call_userfunc(tree, new_children)
        except AttributeError as e:
            if f"'{self.__class__.__name__}' object has no attribute" in str(e):
                method_name = str(e).split("'")[3]  # Extract method name from error
                # Trust: Lark Tree objects always have .data
                rule_name = tree.data
                
                from ..shared.errors import EinlangSourceError
                from ..utils.diagnostics import EinlangErrorCode
                
                raise EinlangSourceError(
                    message=f"Missing transformer method '{method_name}' for grammar rule '{rule_name}'",
                    error_code=EinlangErrorCode.SYNTAX_ERROR.value,
                    category="syntax"
                )
        except TypeError as e:
            # Catch signature mismatches like "takes 5 arguments but 7 were given"
            error_str = str(e)
            if "takes" in error_str and "arguments but" in error_str and "were given" in error_str:
                # Trust: Lark Tree objects always have .data
                rule_name = tree.data
                
                # Extract argument counts
                import re
                match = re.search(r'takes (\d+) .*arguments but (\d+) were given', error_str)
                if match:
                    expected, actual = match.groups()
                    
                    from ..shared.errors import EinlangSourceError
                    from ..utils.diagnostics import EinlangErrorCode
                    
                    raise EinlangSourceError(
                        message=f"Transformer method signature mismatch for rule '{rule_name}': expected {expected} arguments but grammar provides {actual}",
                        error_code=EinlangErrorCode.SYNTAX_ERROR.value,
                        category="syntax"
                    )
            raise  # Re-raise if it's a different TypeError
        except Exception:
            # Re-raise any other exceptions
            raise
    
    def __init__(self) -> None:
        super().__init__()
        # Initialize specialized parsers
        self.function_parser: FunctionDefinitionParser = FunctionDefinitionParser(self._extract_location)
        self.parameter_parser: ParameterParser = ParameterParser(self._extract_location)
        self.expression_parser: BinaryExpressionParser = BinaryExpressionParser(self._extract_location)
        self.current_file: str = ""  # Must be set by parser before use
    
    def _extract_location(self, meta: LarkMeta) -> SourceLocation:
        """Extract location from Lark meta object"""
        # Extract location info directly
        if meta is None:
            location_info = {'has_location': False, 'line': 0, 'column': 0, 'start_pos': 0, 'end_pos': 0}
        else:
            location_info = extract_location_info(meta)
        
        # Fast fail: current_file must be set by parser
        if not self.current_file:
            raise RuntimeError(
                "Parser bug: current_file not set. "
                "Parser must call parse_file() or set current_file before parsing."
            )
        
        if location_info['has_location']:
            return SourceLocation(
                file=self.current_file,
                line=location_info['line'],
                column=location_info['column'], 
                start=location_info['start_pos'],
                end=location_info['end_pos'],
                end_line=location_info.get('end_line', 0),
                end_column=location_info.get('end_column', 0),
            )
        return SourceLocation(
            file=self.current_file,
            line=0,
            column=0,
            start=0,
            end=0
        )
    

    
    # =========================================================================
    # PROGRAM STRUCTURE
    # =========================================================================
    
    def program(self, meta: LarkMeta, *statements: ASTNode) -> Program:
        """✅ Clean meta-based location tracking"""
        location = self._extract_location(meta)
        # Flatten lists (from grouped imports like pub use mod::{a, b})
        flattened = []
        for stmt in statements:
            if isinstance(stmt, list):
                flattened.extend(stmt)
            else:
                flattened.append(stmt)
        return Program(statements=flattened, location=location)

    # =========================================================================
    # FUNCTION DEFINITIONS - USING ALIASES
    # =========================================================================
    
    def function_def(self, meta: LarkMeta, name: Token, lpar: Token, *args: Union[List[Parameter], Dict[str, Any], TypeAnnotation]) -> FunctionDefinition:
        """✅ Grammar: 'fn' NAME LPAR param_list? RPAR ('->' type)? block - 'fn' and '->' filtered"""
        return self.function_parser.parse_function_definition(meta, name, lpar, *args, is_public=False)
    
    def pub_function_def(self, meta: LarkMeta, name: Token, lpar: Token, *args: Union[List[Parameter], Dict[str, Any], TypeAnnotation]) -> FunctionDefinition:
        """✅ Grammar: 'pub' 'fn' NAME LPAR param_list? RPAR ('->' type)? block - 'pub', 'fn' and '->' filtered"""
        return self.function_parser.parse_function_definition(meta, name, lpar, *args, is_public=True)
    
    def parameter(self, meta: LarkMeta, name: Token, type_annotation: Optional[TypeAnnotation] = None) -> Parameter:
        """✅ Uses grammar alias with meta location"""
        return self.parameter_parser.parse_parameter(meta, name, type_annotation)

    # =========================================================================
    # VARIABLE DECLARATIONS - USING ALIASES
    # =========================================================================
    
    def variable_declaration(self, meta: LarkMeta, name: Token, value: ASTNode, type_annotation: Optional[TypeAnnotation] = None, where_clause: Optional[WhereClause] = None) -> VariableDeclaration:
        """✅ Perfect meta-based signature!"""
        # Grammar: "let" NAME (":" type)? "=" expr ("where" where_constraints)? ";"
        location = self._extract_location(meta)
        return VariableDeclaration(
            name=str(name),
            type_annotation=type_annotation,
            value=value,
            location=location
        )
    
    def var_decl(self, meta: LarkMeta, *args: Union[Token, ASTNode, TypeAnnotation]) -> VariableDeclaration:
        """✅ Handle variable declarations with optional type annotations"""
        # @v_args gives us: name, [type_annotation], value
        # Precise type annotations for local variables
        name: ParserToken
        value: ParseResult
        type_annotation: Optional[ParseResult] = None
        
        if len(args) == 2:  # let name = value;
            name, value = args
            type_annotation = None
        elif len(args) == 3:  # let name: type = value;
            name, type_annotation, value = args
        else:
            arg_count: int = len(args)
            raise ValueError(f"Unexpected var_decl args: {arg_count}")
            
        location: SourceLocation = self._extract_location(meta)
        
        # Convert pattern to appropriate form
        if isinstance(name, TupleDestructurePattern):
            pattern_obj = name
        else:
            pattern_obj = str(name)  # Simple identifier
            
        return VariableDeclaration(
            pattern=pattern_obj,
            type_annotation=type_annotation,
            value=value,
            location=location
        )
    
    def const_decl(self, meta: LarkMeta, *args: Union[Token, ASTNode, TypeAnnotation]) -> VariableDeclaration:
        """✅ Handle constant declarations - same as var_decl but marked as constant"""
        # Constants are VariableDeclaration nodes (no separate ConstantDef in AST)
        # Reuse var_decl logic
        var_decl = self.var_decl(meta, *args)
        # Mark as constant (if VariableDeclaration supports it)
        # For now, just return as VariableDeclaration - constants handled in IR lowering
        return var_decl
    

    
    
    def expr_stmt(self, meta: LarkMeta, expr: ASTNode) -> ExpressionStatement:
        """✅ Grammar: expr ';' - Wraps expression in ExpressionStatement for proper AST structure"""
        location = self._extract_location(meta)
        return ExpressionStatement(expr=expr, location=location)

    def einstein_indices(self, meta: LarkMeta, *indices) -> List:
        """✅ Grammar: einstein_index ("," einstein_index)* """
        return list(indices)
    
    def einstein_index(self, meta: LarkMeta, first_arg, *args) -> ASTNode:
        """✅ Grammar: (NAME ("in" expr)?) | literal | named_rest
        
        Returns IndexVar (variable slot), IndexRest (rest slot), or Literal.
        """
        from ...shared.nodes import IndexVar, IndexRest, Literal
        location = self._extract_location(meta)
        if isinstance(first_arg, Literal):
            return first_arg
        if isinstance(first_arg, IndexRest):
            return first_arg
        # NAME token: variable slot with optional range
        name = first_arg
        if not args:
            return IndexVar(name=str(name), range_expr=None, location=location)
        range_expr = args[0] if len(args) == 1 else args[1]
        return IndexVar(name=str(name), range_expr=range_expr, location=location)
    
    def named_rest(self, meta: LarkMeta, dotdot: Token, name: Token) -> 'IndexRest':
        """✅ Grammar: DOTDOT NAME - rest index slot ..name (returns IndexRest)."""
        from ...shared.nodes import IndexRest
        location = self._extract_location(meta)
        return IndexRest(name=str(name), location=location)
    
    def expr_item(self, meta: LarkMeta, item) -> ASTNode:
        """✅ Grammar: expr | named_rest - allows named rest patterns in expr_list"""
        # If it's already an ASTNode (from named_rest or expr), return it directly
        if isinstance(item, ASTNode):
            return item
        # Otherwise, it should be an expr which is already an ASTNode
        return item
    
    def enum_def(self, meta: LarkMeta, name: Token, *args) -> 'EnumDefinition':
        """Transform enum definition: enum Name { Variant1, Variant2(f32) }"""
        from ...shared.nodes import EnumDefinition, EnumVariant
        from lark.lexer import Token
        location = self._extract_location(meta)
        
        # Parse args: generic_params (optional), lbrace, variant_list (optional), rbrace
        # With @v_args(inline=True), tokens are filtered, so we get: generic_params?, variant_list?
        # Lark automatically transforms Trees, so args are already transformed
        # Check if name is actually a Token (should be)
        # If name is a list, it means args shifted - name is actually variant_list
        if not isinstance(name, Token):
            # name is not a Token, it's the variant_list (args shifted)
            variant_list = name
            name = None  # Will need to extract from tree or handle differently
        else:
            variant_list = None
        
        generic_params = None
        
        for arg in args:
            if isinstance(arg, list):
                # Check if it's generic_params (list of strings) or variant_list (list of EnumVariant)
                if arg and isinstance(arg[0], str):
                    generic_params = arg
                elif arg and (isinstance(arg[0], EnumVariant) or (hasattr(arg[0], 'name') and hasattr(arg[0], 'fields'))):
                    variant_list = arg
        
        variant_list_result = variant_list if variant_list else []
        generic_param_list = generic_params if generic_params else []
        
        # Get name from Token or use a default
        if name is None:
            # Name was shifted - try to get from meta or use default
            name_str = "Unknown"
        else:
            name_str = str(name)
        
        return EnumDefinition(
            name=name_str,
            variants=variant_list_result,
            generic_params=generic_param_list,
            is_public=False,
            location=location
        )
    
    def pub_enum_def(self, meta: LarkMeta, pub_kw: Token, enum_kw: Token, name: Token, generic_params: Optional[List[str]] = None, lbrace: Token = None, variant_list: Optional[List] = None, rbrace: Token = None) -> 'EnumDefinition':
        """Transform pub enum definition"""
        enum_def = self.enum_def(meta, enum_kw, name, generic_params, lbrace, variant_list, rbrace)
        enum_def.is_public = True
        return enum_def
    
    def enum_variant_list(self, meta: LarkMeta, *variants) -> List:
        """Transform enum variant list"""
        return list(variants)
    
    def enum_variant(self, meta: LarkMeta, name: Token, fields: Optional[List] = None) -> 'EnumVariant':
        """Transform enum variant: Name or Name(f32, str)"""
        from ...shared.nodes import EnumVariant
        location = self._extract_location(meta)
        field_list = fields if fields else []
        return EnumVariant(
            name=str(name),
            fields=field_list,
            location=location
        )
    
    def enum_variant_field_list(self, meta: LarkMeta, *fields) -> List:
        """Transform enum variant field list"""
        return list(fields)
    
    def enum_variant_field(self, meta: LarkMeta, *args) -> Union[str, 'Type', Tuple]:
        """Transform enum variant field: name: type or type"""
        # Can be: type, or name: type
        if len(args) == 1:
            return args[0]  # Just type
        elif len(args) == 2:
            # name: type
            return (str(args[0]), args[1])  # Return tuple (name, type)
        else:
            return args[0] if args else None  # Fallback
    
    def struct_def(self, meta: LarkMeta, name: Token, *args) -> 'StructDefinition':
        """Transform struct definition: struct Name { x: f32, y: f32 } or struct Name(f32, f32)"""
        from ...shared.nodes import StructDefinition, StructField
        location = self._extract_location(meta)
        
        # Parse args: generic_params (optional), lbrace/lparen, field_list, rbrace/rparen
        # With @v_args(inline=True), tokens are filtered
        generic_params = None
        is_tuple_struct = False
        fields = []
        
        # Parse args to find fields and generic_params
        for arg in args:
            if isinstance(arg, str) and arg in ['(', ')', '{', '}']:
                if arg == '(':
                    is_tuple_struct = True
                continue
            elif isinstance(arg, list):
                # Check if it's generic_params (list of strings) or field_list (list of StructField)
                if arg and isinstance(arg[0], str):
                    generic_params = arg
                else:
                    # Field list
                    for field in arg:
                        if isinstance(field, StructField):
                            fields.append(field)
                        elif isinstance(field, tuple) and len(field) == 2:
                            # (name, type) tuple
                            fields.append(StructField(name=field[0], field_type=field[1], location=location))
            elif isinstance(arg, StructField):
                fields.append(arg)
        
        generic_param_list = generic_params if generic_params else []
        return StructDefinition(
            name=str(name),
            fields=fields,
            is_tuple_struct=is_tuple_struct,
            generic_params=generic_param_list,
            is_public=False,
            location=location
        )
    
    def pub_struct_def(self, meta: LarkMeta, pub_kw: Token, name: Token, *args) -> 'StructDefinition':
        """Transform pub struct definition"""
        struct_def = self.struct_def(meta, struct_kw, name, generic_params, *args)
        struct_def.is_public = True
        return struct_def
    
    def struct_field_list(self, meta: LarkMeta, *fields) -> List:
        """Transform struct field list"""
        return list(fields)
    
    def struct_field(self, meta: LarkMeta, name: Token, field_type: 'Type') -> 'StructField':
        """Transform struct field: name: type (colon is filtered out by grammar)"""
        from ...shared.nodes import StructField
        location = self._extract_location(meta)
        return StructField(
            name=str(name),
            field_type=field_type,
            location=location
        )
    
    def generic_params(self, meta: LarkMeta, langle: Token, params: List[str], rangle: Token) -> List[str]:
        """Transform generic parameters: <T, E>"""
        return params
    
    def generic_param_list(self, meta: LarkMeta, *params) -> List[str]:
        """Transform generic parameter list"""
        param_list = []
        for param in params:
            if isinstance(param, str):
                param_list.append(param)
            elif isinstance(param, Token):
                param_list.append(str(param))
        return param_list
    
    def generic_param(self, meta: LarkMeta, name: Token, type_constraint: Optional['Type'] = None) -> str:
        """Transform generic parameter: T or T: Trait"""
        # For now, just return the name (constraints handled later)
        return str(name)
    
    def named_type(self, meta: LarkMeta, *name_parts) -> 'Type':
        """Transform named type: Color, Shape, Point"""
        from ..analysis.types.types import NamedType
        location = self._extract_location(meta)
        # name_parts is list of identifiers separated by ::
        type_name = '::'.join(str(part) for part in name_parts if not isinstance(part, str) or part != '::')
        return NamedType(name=type_name, location=location)
    
    def generic_type(self, meta: LarkMeta, *args) -> 'Type':
        """Transform generic type: Option<T>, Result<T, E>"""
        from ..analysis.types.types import GenericType, NamedType
        location = self._extract_location(meta)
        # Parse: name < type_list >
        type_name = None
        type_args = []
        in_type_args = False
        for arg in args:
            if isinstance(arg, str) and arg == '<':
                in_type_args = True
                continue
            elif isinstance(arg, str) and arg == '>':
                in_type_args = False
                continue
            elif in_type_args:
                if isinstance(arg, list):
                    type_args.extend(arg)
                else:
                    type_args.append(arg)
            else:
                if isinstance(arg, Token):
                    type_name = str(arg)
                elif isinstance(arg, NamedType):
                    type_name = arg.name
        
        if type_name:
            base_type = NamedType(name=type_name, location=location)
            return GenericType(base_type=base_type, type_arguments=type_args, location=location)
        else:
            # Fallback
            return NamedType(name="Unknown", location=location)
    
    def einstein_decl(self, meta: LarkMeta, name: Token, lsqb: Token, einstein_indices: List[ASTNode], rsqb: Token, value: ASTNode, where_constraints: Optional[WhereClause] = None) -> EinsteinDeclaration:
        """✅ Grammar: "let" NAME LSQB einstein_indices RSQB "=" expr ("where" where_constraints)? ";" """
        location = self._extract_location(meta)
        indices = einstein_indices
        where_clause = None
        if where_constraints:
            where_clause = where_constraints
        elif value.node_type == NodeType.WHERE_EXPR and value.where_clause.has_constraints():
            where_clause = value.where_clause
            value = value.expr
        from ...shared.nodes import EinsteinClause
        clause = EinsteinClause(indices, value, where_clause, location)
        return EinsteinDeclaration(array_name=str(name), clauses=[clause], location=location)
    
    # =========================================================================  
    # EXPRESSIONS - SINGLE BINARY HANDLER ✅
    # =========================================================================
    
    def logical_or_binary(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Handle logical OR binary expressions"""
        return self.expression_parser.parse_logical_or(meta, left, operator, right)
        
    def logical_and_binary(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Handle logical AND binary expressions"""
        return self.expression_parser.parse_logical_and(meta, left, operator, right)
        
    def equality_binary(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Handle equality binary expressions"""
        return self.expression_parser.parse_equality(meta, left, operator, right)
        
    def relational_binary(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Handle relational binary expressions"""
        return self.expression_parser.parse_relational(meta, left, operator, right)
        
    def additive_binary(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Handle additive binary expressions"""
        return self.expression_parser.parse_additive(meta, left, operator, right)
        
    def multiplicative_binary(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Handle multiplicative binary expressions"""
        return self.expression_parser.parse_multiplicative(meta, left, operator, right)
        
    def power_binary(self, meta: LarkMeta, left: ASTNode, operator: Token, right: ASTNode) -> BinaryExpression:
        """Handle power binary expressions"""
        return self.expression_parser.parse_power(meta, left, operator, right)
    

    
    # =========================================================================
    # LITERALS - NO TERMINAL METHODS NEEDED! ✅
    # =========================================================================
    
    # String interpolation parsing delegated to specialized parser
    
    def literal(self, meta: LarkMeta, token: Token) -> Union[Literal, InterpolatedString]:
        """✅ Handle all literal types - receives INTEGER_OR_FLOAT, STRING, or string tokens"""
        location = self._extract_location(meta)
        return LiteralParser.parse(token, location)
    
    def identifier(self, meta: LarkMeta, name: Token) -> Identifier:
        """✅ Clean meta-based identifier"""
        location = self._extract_location(meta)
        return Identifier(name=str(name), location=location)

    # =========================================================================
    # OPERATIONS - USING ALIASES
    # =========================================================================
    
    def callable_primary(self, meta: LarkMeta, *children: ASTNode) -> ASTNode:
        """Alias: return the transformed expression (one child for identifier/lambda/etc., or middle child for LPAR expr RPAR)."""
        if len(children) == 1:
            return children[0]
        if len(children) == 3:
            return children[1]  # LPAR expr RPAR
        raise RuntimeError(f"callable_primary expected 1 or 3 children, got {len(children)}")

    def postfix_call(self, meta: LarkMeta, callee: ASTNode, lpar: Token, *args_and_rpar: Union[List[ASTNode], Token]) -> FunctionCall:
        """Any expression followed by ( args ) is a call."""
        location = self._extract_location(meta)
        arguments: List[ASTNode] = []
        if len(args_and_rpar) == 2:
            args, rpar = args_and_rpar
            arguments = args if isinstance(args, list) else [args]
        else:
            arguments = []
        return FunctionCall(
            function_expr=callee,
            arguments=arguments,
            location=location
        )
    
    def rectangular_access(self, meta: LarkMeta, name: Token, lsqb: Token, indices: List[ASTNode], rsqb: Token) -> RectangularAccess:
        """✅ Grammar: NAME LSQB index_list RSQB - rectangular access [i,j] style"""
        location = self._extract_location(meta)
        return RectangularAccess(
            base_expr=Identifier(str(name)),
            indices=indices,
            location=location
        )
    
    def jagged_access(self, meta: LarkMeta, base: ASTNode, *indices: ASTNode):
        """Handle jagged array access with nested [i][j] syntax"""
        location = self._extract_location(meta)
        return JaggedAccess(
            base_expr=base,
            indices=list(indices),
            location=location
        )
    
    def reduction_expression(self, meta: LarkMeta, name: Token, lsqb: Token, indices: List[ASTNode], rsqb: Token, lpar: Token, body: ASTNode, rpar: Token) -> ReductionExpression:
        """✅ Grammar: NAME LSQB expr_list RSQB LPAR expr RPAR - terminals passed, ignore brackets
        
        Supports three forms:
        1. sum[k](...)  - simple variable
        2. sum[k in 0..i+1](...)  - variable with range
        3. sum[..batch](...)  - named rest pattern (will be expanded later)
        """
        from ...shared.nodes import IndexRest
        
        location = self._extract_location(meta)
        
        # Process indices: IndexVar, IndexRest, Identifier, BinaryExpression
        range_groups = []
        for idx in indices:
            if isinstance(idx, Identifier):
                range_groups.append(RangeGroup(range_expr=None, variables=[idx.name]))
            elif isinstance(idx, IndexRest):
                range_groups.append(RangeGroup(range_expr=None, variables=[idx.name]))
            elif isinstance(idx, BinaryExpression) and idx.operator == BinaryOp.IN:
                if isinstance(idx.left, Identifier):
                    var_name = idx.left.name
                    range_expr = idx.right
                    range_groups.append(RangeGroup(range_expr=range_expr, variables=[var_name]))
                elif isinstance(idx.left, IndexRest):
                    # Rest pattern with range: sum[..batch in 0..N](...)
                    # This is more complex - defer to analysis passes
                    # Store rest pattern name without ".." prefix (parser normalizes this)
                    range_groups.append(RangeGroup(
                        range_expr=idx.right,
                        variables=[idx.left.name]
                    ))
                else:
                    # Fallback - shouldn't happen with valid syntax
                    raise ValueError(f"Invalid reduction variable: {idx}")
            else:
                # Fallback - shouldn't happen with valid syntax
                raise ValueError(f"Invalid reduction index: {idx}")
        
        over_clause = OverClause(range_groups=range_groups)
        
        return ReductionExpression(
            function_name=str(name),
            body=body,
            over_clause=over_clause,
            where_clause=None,  # Where clause is now at the expression level
            location=location
        )
    
    # =========================================================================
    # ARRAY AND TUPLE CONSTRUCTS - USING ALIASES
    # =========================================================================
    
    def array_construct_literal(self, meta: LarkMeta, lsqb_token: Token, array_elements: List[ASTNode], rsqb_token: Token) -> ArrayLiteral:
        """✅ Uses grammar alias: LSQB array_elements RSQB"""
        location = self._extract_location(meta)
        # array_elements is the actual list of expressions
        return ArrayLiteral(elements=array_elements, location=location)
    
    def array_literal(self, meta: LarkMeta, array_element_list: Optional[List[ASTNode]] = None) -> ArrayLiteral:
        """✅ Grammar: '[' array_element_list? ']' - brackets filtered by @v_args(inline=True)"""
        location = self._extract_location(meta)
        return ArrayLiteral(elements=array_element_list or [], location=location)
    
    def empty_array_literal(self, meta: LarkMeta, lsqb_token: Token, rsqb_token: Token) -> ArrayLiteral:
        """✅ Uses grammar alias: LSQB RSQB"""
        location = self._extract_location(meta)
        return ArrayLiteral(elements=[], location=location)
    
    def array_construct_comprehension(self, meta: LarkMeta, lsqb_token: Token, expr: ASTNode, where_clause: WhereClause, rsqb_token: Token) -> ArrayComprehension:
        """✅ Uses grammar alias: LSQB expr where_clause RSQB"""
        # Trust: WhereClause has constraints as static attribute
        location = self._extract_location(meta)
        return ArrayComprehension(
            expr=expr,
            constraints=where_clause.constraints,
            location=location
        )
    
    def tuple_construct(self, meta: LarkMeta, *items: ASTNode) -> TupleExpression:
        """✅ Uses grammar alias with meta-based location"""
        location = self._extract_location(meta)
        return TupleExpression(elements=list(items), location=location)

    # =========================================================================
    # CONTROL FLOW - USING ALIASES
    # =========================================================================
    
    def if_expr(self, meta: LarkMeta, condition: ASTNode, then_block: BlockExpression, else_clause: Optional[ElseClauseResult] = None) -> IfExpression:
        """✅ Grammar: 'if' expr block else_clause? - 'if' filtered by @v_args(inline=True)"""
        from ...shared.nodes import BlockExpression
        location = self._extract_location(meta)
        
        # then_block is already a BlockExpression
        then_block_expr = then_block
        
        # Extract else block info if present and create BlockExpression
        else_block_expr: Optional[BlockExpression] = None
        
        if else_clause:
            if else_clause.type == 'else_block':
                else_statements = else_clause.statements or []
                else_expr = else_clause.final_expr
                else_block_expr = BlockExpression(
                    statements=else_statements,
                    final_expr=else_expr,
                    location=location
                )
            elif else_clause.type == 'else_if':
                # Handle else if as else expression - wrap in BlockExpression
                else_if_expr = else_clause.if_expr
                else_block_expr = BlockExpression(
                    statements=[],
                    final_expr=else_if_expr,
                    location=location
                )
        
        return IfExpression(
            condition=condition,
            then_block=then_block_expr,
            else_block=else_block_expr,
            location=location
        )
    
    def block(self, meta: LarkMeta, *body_items: ASTNode) -> BlockExpression:
        """✅ Grammar: '{' statement* expr? '}' - braces filtered by @v_args(inline=True)"""
        statements: List[ASTNode] = []
        final_expr: Optional[ASTNode] = None
        
        # No need for a for loop; just check body_items[-1] for a final expression, and the rest are statements.
        if body_items:
            # Check if the last item is an expression (final return value)
            last_item = body_items[-1]
            # Use proper isinstance check instead of string matching
            if isinstance(last_item, Expression):
                final_expr = last_item
                statements = list(body_items[:-1])  # All but the last item
            else:
                # Last item is also a statement
                statements = list(body_items)
        
        # Return a BlockExpression AST node (not a dict)
        return BlockExpression(
            statements=statements,
            final_expr=final_expr,
            location=self._extract_location(meta)
        )
    
    def else_clause(self, meta: LarkMeta, block_or_if: Union[BlockExpression, 'IfExpression']) -> ElseClauseResult:
        """✅ Grammar: 'else' block | 'else' if_expr - 'else' filtered by @v_args(inline=True)"""
        from ...shared.nodes import BlockExpression
        if isinstance(block_or_if, BlockExpression):
            return ElseClauseResult(
                type='else_block',
                statements=block_or_if.statements,
                final_expr=block_or_if.final_expr
            )
        else:
            return ElseClauseResult(type='else_if', if_expr=block_or_if)
    
    # =========================================================================
    # STATEMENTS - USING ALIASES
    # =========================================================================
    
    def use_stmt(self, meta: LarkMeta, import_path: ImportPathInfo, alias: Optional[Token] = None) -> Union[UseStatement, List[UseStatement]]:
        """✅ Grammar: 'use' import_path ('as' NAME)? ';' """
        location = self._extract_location(meta)
        
        # Handle function list imports with aliases: use std::math::{min, max as maximum};
        if import_path.function_items:
            # Create separate UseStatement for each function (with optional alias)
            statements = []
            for func_name, func_alias in import_path.function_items:
                func_path = import_path.path + [func_name]
                statements.append(UseStatement(
                    path=func_path,
                    is_wildcard=False,
                    is_function=True,
                    is_public=False,
                    alias=func_alias,  # May be None
                    location=location
                ))
            return statements
        
        return UseStatement(path=import_path.path, is_wildcard=import_path.is_wildcard, is_function=import_path.is_function, is_public=False, alias=str(alias) if alias else None, location=location)
    
    def pub_use_stmt(self, meta: LarkMeta, use_stmt: Union[UseStatement, List[UseStatement]]) -> Union[UseStatement, List[UseStatement]]:
        """✅ Grammar: 'pub' use_stmt """
        location = self._extract_location(meta)
        
        # Handle both single UseStatement and list of UseStatements (from grouped imports)
        if isinstance(use_stmt, list):
            # Grouped import: mark each statement as public
            return [UseStatement(path=stmt.path, is_wildcard=stmt.is_wildcard, is_function=stmt.is_function, 
                               is_public=True, alias=stmt.alias, location=location) for stmt in use_stmt]
        else:
            # Single import: mark as public
            return UseStatement(path=use_stmt.path, is_wildcard=use_stmt.is_wildcard, is_function=use_stmt.is_function, 
                              is_public=True, alias=use_stmt.alias, location=location)
    
    def mod_stmt(self, meta: LarkMeta, name: Token) -> ModuleDeclaration:
        """✅ Grammar: 'mod' NAME ';' """
        location = self._extract_location(meta)
        return ModuleDeclaration(name=str(name), is_public=False, location=location)
    
    def pub_mod_stmt(self, meta: LarkMeta, name: Token) -> ModuleDeclaration:
        """✅ Grammar: 'pub' 'mod' NAME ';' """
        location = self._extract_location(meta)
        return ModuleDeclaration(name=str(name), is_public=True, location=location)
    
    def inline_mod_stmt(self, meta: LarkMeta, name: Token, body: BlockExpression) -> InlineModule:
        """✅ Grammar: 'mod' NAME block """
        location = self._extract_location(meta)
        statements = body.statements if hasattr(body, 'statements') else []
        return InlineModule(name=str(name), body=statements, is_public=False, location=location)
    
    def pub_inline_mod_stmt(self, meta: LarkMeta, name: Token, body: BlockExpression) -> InlineModule:
        """✅ Grammar: 'pub' 'mod' NAME block """
        location = self._extract_location(meta)
        statements = body.statements if hasattr(body, 'statements') else []
        return InlineModule(name=str(name), body=statements, is_public=True, location=location)
    
    def import_path(self, meta: LarkMeta, path_info: ImportPathInfo) -> ImportPathInfo:
        """✅ Grammar: module_path | wildcard_import | function_list_import """
        return path_info
    
    def module_path(self, meta: LarkMeta, *parts: Token) -> ImportPathInfo:
        """✅ Grammar: NAME ("::" NAME)* """
        path_list = [str(part) for part in parts]
        return ImportPathInfo(path_list, is_wildcard=False, is_function=False)
    
    def wildcard_import(self, meta: LarkMeta, *parts: Token) -> ImportPathInfo:
        """✅ Grammar: NAME ("::" NAME)* "::" MULTIPLY """
        # All parts except the last one (which is "*") form the module path
        path_parts = [str(part) for part in parts if str(part) != "*"]
        return ImportPathInfo(path_parts, is_wildcard=True, is_function=False)
    
    def function_name_list(self, meta: LarkMeta, *names: Token) -> List[str]:
        """✅ Grammar: NAME ("," NAME)* """
        return [str(name) for name in names]
    
    def import_item(self, meta: LarkMeta, *args) -> Tuple[str, Optional[str]]:
        """✅ Grammar: NAME ('as' NAME)? """
        if len(args) == 1:
            return (str(args[0]), None)  # (name, no alias)
        # Has alias: NAME as NAME
        return (str(args[0]), str(args[1]))  # (name, alias)
    
    def import_item_list(self, meta: LarkMeta, *items: Tuple[str, Optional[str]]) -> List[Tuple[str, Optional[str]]]:
        """✅ Grammar: import_item ("," import_item)* ","? """
        return list(items)
    
    def function_list_import(self, meta: LarkMeta, *args) -> ImportPathInfo:
        """✅ Grammar: NAME ("::" NAME)* "::" "{" import_item_list "}" """
        # Parse: all args except last are path parts, last is list of (name, alias) tuples
        import_items = args[-1]  # List[Tuple[str, Optional[str]]]
        path_parts = [str(part) for part in args[:-1]]
        
        # Store both legacy function_names and new function_items with aliases
        function_names = [name for name, alias in import_items]
        
        return ImportPathInfo(
            path_parts,
            is_wildcard=False,
            is_function=False,
            function_names=function_names,
            function_items=import_items  # List of (name, alias) tuples
        )

    # =========================================================================
    # MEMBER ACCESS - USING ALIASES  
    # =========================================================================
    
    def member_access(self, meta: LarkMeta, object_expr: ASTNode, dot: Token, property_name: Token) -> MemberAccess:
        """✅ Grammar: primary_expr DOT (NAME | INTEGER_OR_FLOAT) - supports both properties and tuple access"""
        location = self._extract_location(meta)
        
        # Determine if this is a property access (string) or tuple access (number)
        property_str = str(property_name)
        if property_str.isdigit():
            # Tuple access: tuple.0, tuple.1, etc.
            property_obj = int(property_str)
        else:
            # Property access: obj.property
            property_obj = property_str
            
        return MemberAccess(
            object=object_expr,
            property=property_obj,
            location=location
        )
    
    def module_access(self, meta: LarkMeta, module_expr: ASTNode, doublecolon: Token, function_name: Token) -> ModuleAccess:
        """✅ Grammar: primary_expr DOUBLECOLON NAME - module function reference"""
        location = self._extract_location(meta)
        # Create a proper ModuleAccess node  
        return ModuleAccess(
            object=module_expr,
            property=str(function_name),
            location=location
        )
    
    def chained_access(self, meta: LarkMeta, member_access: MemberAccess, lsqb: Token, indices: List[ASTNode], rsqb: Token) -> RectangularAccess:
        """✅ Uses grammar alias with meta-based location"""
        location = self._extract_location(meta)
        return RectangularAccess(
            base_expr=member_access,
            indices=indices,
            location=location
        )
    
    def array_access(self, meta: LarkMeta, primary_expr: ASTNode, lsqb: Token, indices: List[ASTNode], rsqb: Token) -> RectangularAccess:
        """✅ Grammar: primary_expr LSQB expr_list RSQB - single array access, ignore brackets"""
        location = self._extract_location(meta)
        return RectangularAccess(
            base_expr=primary_expr,
            indices=indices,
            location=location
        )
    
    def chained_array_access(self, meta: LarkMeta, array_access_expr: RectangularAccess, lsqb: Token, second_indices: List[ASTNode], rsqb: Token) -> RectangularAccess:
        """✅ Grammar: array_access LSQB expr_list RSQB - handles array[1][0] syntax, ignore brackets"""
        location = self._extract_location(meta)
        
        # array_access_expr is already a RectangularAccess, create another level
        return RectangularAccess(
            base_expr=array_access_expr,
            indices=second_indices,
            location=location
        )
    
    # =========================================================================
    # WHERE CLAUSES AND CONSTRAINTS
    # =========================================================================
    
    def range_expr(self, meta: LarkMeta, *args: Union[ASTNode, Token]) -> Union[ASTNode, Range]:
        """Handle range expression: additive_expr ((DOTDOT | DOTDOTEQ) additive_expr)?"""
        if len(args) == 1:
            return args[0]
        elif len(args) == 3:
            location = self._extract_location(meta)
            inclusive = str(args[1]) == '..='
            return Range(
                start=args[0],
                end=args[2],
                inclusive=inclusive,
                location=location
            )
        else:
            raise ValueError(f"Unexpected range_expr args: {args}")

    # =========================================================================
    # TYPES - USING ALIASES
    # =========================================================================
    
    def primitive_type(self, meta: LarkMeta, type_token: Token):
        """✅ Direct rule match - receives actual type token from terminals
        
        Returns PrimitiveTypeEnum constants.
        """
        from ...shared.types import I32, I64, F32, F64, BOOL, STR
        
        # Extract the actual type name from the token
        type_name = str(type_token)
        
        # Remove PTYPE_ prefix if present  
        if type_name.startswith('PTYPE_'):
            type_name = type_name[6:].lower()
        
        # Map to type constants
        type_obj_mapping = {
            'i32': I32,
            'i64': I64,
            'f32': F32,
            'f64': F64,
            'bool': BOOL,
            'str': STR,
            'int': I32,    # Generic int maps to i32
            'float': F32,  # Generic float maps to f32 (Rust-like default)
        }
        return type_obj_mapping.get(type_name, I32)  # Default to I32 if unknown
    
    def shape_spec(self, meta: LarkMeta, *args) -> Union[List, str]:
        """Transform shape_spec: explicit_shape | dynamic_rank"""
        if len(args) == 1 and str(args[0]) == "*":
            return "*"  # Dynamic rank
        return list(args)  # Explicit shape dimensions
    
    def explicit_shape(self, meta: LarkMeta, *dimensions) -> List:
        """Transform explicit_shape: shape_dimension ("," shape_dimension)*"""
        return list(dimensions)
    
    def shape_dimension(self, meta: LarkMeta, dim: Union[Token, ASTNode]) -> str:
        """Transform shape_dimension: INTEGER | NAME | "?" """
        return str(dim)
    
    def dynamic_rank(self, meta: LarkMeta, *args) -> str:
        """Transform dynamic_rank: "*" """
        return "*"
    
    def jagged_depth(self, meta: LarkMeta, depth: Union[Token, ASTNode]) -> str:
        """Transform jagged_depth: dimension_suffix | dynamic_depth"""
        return str(depth)
    
    def dynamic_depth(self, meta: LarkMeta, *args) -> str:
        """Transform dynamic_depth: "?" """
        # Handle variable arguments to avoid missing argument errors
        if args:
            return str(args[0])
        return "?"  # Default to dynamic
    
    def dimension_suffix(self, meta: LarkMeta, *args) -> str:
        """Transform dimension_suffix: "2d" | "3d" | "4d" """
        # Handle variable arguments to avoid missing argument errors
        if args:
            return str(args[0])
        return "2d"  # Default to 2d
    
    def shape_spec(self, meta: LarkMeta, *args):
        """Transform shape_spec: explicit_shape | dynamic_rank"""
        if len(args) == 1 and args[0] == "*":
            return "*"
        # Return the first argument (which should be the explicit_shape or dynamic_rank)
        return args[0] if args else None
    
    def explicit_shape(self, meta: LarkMeta, *dimensions):
        """Transform explicit_shape: shape_dimension ("," shape_dimension)*"""
        return list(dimensions)
    
    def shape_dimension(self, meta: LarkMeta, *args):
        """Transform shape_dimension: INTEGER_OR_FLOAT | NAME | "?" """
        if not args:
            return None
        dimension = args[0]
        dim_str = str(dimension)
        if dim_str == '?':
            return None
        try:
            return int(dim_str)
        except ValueError:
            return dim_str
    
    
    def rectangular_type(self, meta: LarkMeta, *args):
        """Transform rectangular type: LSQB primitive_type (SEMICOLON shape_spec)? RSQB"""
        from ...shared.types import RectangularType
        
        location = self._extract_location(meta)  # For error messages only
        
        # Filter out tokens to get the actual data
        element_type = None
        shape_spec = None
        
        for arg in args:
            # Skip bracket tokens
            if isinstance(arg, str) and arg in ['[', ']', ';']:
                continue
            # First non-token is element_type
            elif element_type is None:
                element_type = arg
            # Second non-token is shape_spec
            else:
                shape_spec = arg
                break
        
        if shape_spec is None:
            # [T] - 1D with unknown size
            return RectangularType(element_type=element_type, shape=None, is_dynamic_rank=False)
        
        # Handle different shape_spec types
        if isinstance(shape_spec, str) and shape_spec == "*":
            # [T; *] - dynamic rank
            return RectangularType(element_type=element_type, shape=None, is_dynamic_rank=True)
        
        # Parse explicit shape: [T; 3, 4] or [T; ?, ?]
        if isinstance(shape_spec, (list, tuple)):
            # Keep dimensions as int/str (shape_dimension already normalizes them)
            dimensions = tuple(shape_spec)
            return RectangularType(element_type=element_type, shape=dimensions, is_dynamic_rank=False)
        
        # Single dimension: [T; ?] or [T; n]
        # Keep dimension as int/str (shape_dimension already normalizes it)
        shape = (shape_spec,)  # Tuple of shape dimensions
        return RectangularType(element_type=element_type, shape=shape, is_dynamic_rank=False)
    
    def jagged_type(self, meta: LarkMeta, *args):
        """Transform jagged type: "jagged" LSQB primitive_type (SEMICOLON jagged_depth)? RSQB"""
        from ...shared.types import JaggedType
        
        location = self._extract_location(meta)
        
        # Filter out tokens to get the actual data
        element_type = None
        jagged_depth = None
        
        for arg in args:
            # Skip jagged keyword and bracket tokens
            if isinstance(arg, str) and arg in ['jagged', '[', ']', ';']:
                continue
            # First non-token is element_type
            elif element_type is None:
                element_type = arg
            # Second non-token is jagged_depth
            else:
                jagged_depth = arg
                break
        
        if jagged_depth is None:
            # jagged[T] - 1D jagged (default)
            return JaggedType(element_type=element_type, nesting_depth=1, is_dynamic_depth=False)
        
        depth_str = str(jagged_depth)
        
        if depth_str == "?":
            # jagged[T; ?] - dynamic depth
            return JaggedType(element_type=element_type, nesting_depth=None, is_dynamic_depth=True)
        
        # Parse static depth: "2d", "3d", etc.
        if depth_str.endswith('d'):
            try:
                depth = int(depth_str[:-1])
                return JaggedType(element_type=element_type, nesting_depth=depth, is_dynamic_depth=False)
            except ValueError:
                # Fallback to 1D if parsing fails
                return JaggedType(element_type=element_type, nesting_depth=1, is_dynamic_depth=False)
        
        # Default to 1D jagged
        return JaggedType(element_type=element_type, nesting_depth=1, is_dynamic_depth=False)
    
    def tuple_type(self, meta: LarkMeta, *args):
        """Transform tuple type: LPAR type_list RPAR"""
        from ...shared.types import TupleType
        
        location = self._extract_location(meta)
        
        # Filter out parenthesis tokens and extract type list
        element_types = []
        
        for arg in args:
            # Skip parenthesis tokens
            if isinstance(arg, str) and arg in ['(', ')']:
                continue
            # Trust: type_list is a list/tuple from Lark
            elif isinstance(arg, (list, tuple)):
                # arg is the type_list - extract the individual types
                element_types.extend(arg)
            else:
                # Single type argument
                element_types.append(arg)
        
        # Semantic validation: tuples must have at least 2 elements (best practice)
        if len(element_types) < 2:
            from ...shared.errors import EinlangSyntaxError
            raise EinlangSyntaxError(
                f"Tuple type must have at least 2 elements, got {len(element_types)}", 
                location=location
            )
        
        return TupleType(tuple(element_types))
    
    def type_list(self, meta: LarkMeta, *types):
        """Transform type_list: type ("," type)* - unified for both tuple and function types"""
        # Return list of types for both tuple_type and function_type to consume
        return list(types)

    # =========================================================================
    # PASS-THROUGH METHODS (keep existing structure working)
    # =========================================================================
    

    
    def expr(self, meta: LarkMeta, logical_or_expr: ASTNode, where_clause: Optional[WhereClause] = None) -> Union[ASTNode, WhereExpression]:
        """✅ Trust @v_args(inline=True) to handle optional where clause"""
        if where_clause is None:
            return logical_or_expr
        # Create a where expression wrapper with proper location
        location = self._extract_location(meta)
        return WhereExpression(
            expr=logical_or_expr,
            where_clause=where_clause,
            location=location
        )
    

    

    
    def unary_operation(self, meta: LarkMeta, operator: Token, operand: ASTNode) -> UnaryExpression:
        """Handle unary operations"""
        location = self._extract_location(meta)
        return UnaryExpression(
            operator=UnaryOp(str(operator)),
            operand=operand,
            location=location
        )
    
    # Passthrough methods for single-element cases
    def logical_or_expr(self, meta: LarkMeta, *args: Union[ASTNode, Token]) -> ASTNode:
        """Handle logical OR expressions with repetition pattern"""
        if len(args) == 1:
            return args[0]
        result = args[0]
        location = self._extract_location(meta)
        for operator, right in zip(args[1::2], args[2::2]):
            result = BinaryExpression(
                left=result,
                operator=BinaryOp(str(operator)),
                right=right,
                location=location
            )
        return result
    
    def logical_and_expr(self, meta: LarkMeta, *args: Union[ASTNode, Token]) -> ASTNode:
        """Handle logical AND expressions with repetition pattern"""
        if len(args) == 1:
            return args[0]
        result = args[0]
        location = self._extract_location(meta)
        for operator, right in zip(args[1::2], args[2::2]):
            result = BinaryExpression(
                left=result,
                operator=BinaryOp(str(operator)),
                right=right,
                location=location
            )
        return result
        
    def equality_expr(self, meta: LarkMeta, *args: Union[ASTNode, Token]) -> ASTNode:
        """Handle equality expressions with repetition pattern"""
        if len(args) == 1:
            return args[0]
        result = args[0]
        location = self._extract_location(meta)
        for operator, right in zip(args[1::2], args[2::2]):
            result = BinaryExpression(
                left=result,
                operator=BinaryOp(str(operator)),
                right=right,
                location=location
            )
        return result
        
    def relational_expr(self, meta: LarkMeta, *args: Union[ASTNode, Token]) -> ASTNode:
        """Handle relational expressions with repetition pattern"""
        if len(args) == 1:
            return args[0]
        result = args[0]
        location = self._extract_location(meta)
        for operator, right in zip(args[1::2], args[2::2]):
            result = BinaryExpression(
                left=result,
                operator=BinaryOp(str(operator)),
                right=right,
                location=location
            )
        return result
        
    def additive_expr(self, meta: LarkMeta, *args: Union[ASTNode, Token]) -> ASTNode:
        """Handle additive expressions with repetition pattern"""
        if len(args) == 1:
            return args[0]
        result = args[0]
        location = self._extract_location(meta)
        for operator, right in zip(args[1::2], args[2::2]):
            result = BinaryExpression(
                left=result,
                operator=BinaryOp(str(operator)),
                right=right,
                location=location
            )
        return result
        
    def multiplicative_expr(self, meta: LarkMeta, *args: Union[ASTNode, Token]) -> ASTNode:
        """Handle multiplicative expressions with repetition pattern"""
        if len(args) == 1:
            return args[0]
        result = args[0]
        location = self._extract_location(meta)
        for operator, right in zip(args[1::2], args[2::2]):
            result = BinaryExpression(
                left=result,
                operator=BinaryOp(str(operator)),
                right=right,
                location=location
            )
        return result
        
    def power_expr(self, meta: LarkMeta, *args: Union[ASTNode, Token]) -> ASTNode:
        """Handle power expressions with right-associativity (a**b**c = a**(b**c))"""
        if len(args) == 1:
            return args[0]
        
        location = self._extract_location(meta)
        
        # For right-associativity, build the tree from right to left
        # Extract operands and operators
        operands = args[::2]  # args[0], args[2], args[4], ...
        operators = args[1::2]  # args[1], args[3], args[5], ...
        
        # Start with the rightmost operand
        result = operands[-1]
        
        # Build the tree from right to left
        for i in range(len(operators) - 1, -1, -1):
            result = BinaryExpression(
                left=operands[i],
                operator=BinaryOp(str(operators[i])),
                right=result,
                location=location
            )
        
        return result
        
    def cast_expr(self, meta: LarkMeta, *args: Union[ASTNode, Token]) -> ASTNode:
        """Handle cast expressions (expr as type)"""
        if len(args) == 1:
            # No cast, just the expression
            return args[0]
        
        # args = [expression, 'as' token, type]
        expression = args[0]
        target_type = args[2]  # Skip the 'as' token at args[1]
        location = self._extract_location(meta)
        
        return CastExpression(
            expr=expression,
            target_type=target_type,
            location=location
        )
    
    def try_expr(self, meta: LarkMeta, operand: ASTNode = None) -> TryExpression:
        """Handle try expressions: 'try expr'
        
        Grammar: try_expr: "try" expr
        
        Captures the entire following expression for error handling.
        
        Examples:
            try 10 / 2    -> TryExpression(BinaryExpression(10, /, 2))
            try -x        -> TryExpression(UnaryExpression(-, x))
            try x.foo()   -> TryExpression(FunctionCall("x.foo"))
        
        Note: operand is optional to handle edge cases with @v_args(inline=True)
        """
        location = self._extract_location(meta)
        
        if operand is None:
            raise ValueError(
                "try_expr received None operand. This indicates a grammar or "
                "transformer issue where the expression child is not being passed."
            )
        
        return TryExpression(operand, location)
    
    def unary_expr(self, meta: LarkMeta, *args: Union[ASTNode, Token]) -> ASTNode:
        """Handle unary expressions with zero or more unary operators - needs *args for unary_op*"""
        if len(args) == 1:
            # No unary operators, just the primary expression
            return args[0]
        
        # Build nested unary expressions from right to left
        result = args[-1]  # Start with the primary expression
        location = self._extract_location(meta)
        
        # Apply unary operators from right to left
        for operator in reversed(args[:-1]):
            result = UnaryExpression(
                operator=UnaryOp(str(operator)),
                operand=result,
                location=location
            )
        
        return result
    

        

    

    
    def array_element_list(self, meta: LarkMeta, *elements: ASTNode) -> List[ASTNode]:
        """✅ Grammar: range_expr (',' range_expr)*"""
        return list(elements)
    
    def comprehension_constraints(self, meta: LarkMeta, *constraints: ASTNode) -> List[ASTNode]:
        """✅ Grammar: comprehension_constraint (',' comprehension_constraint)* - commas filtered"""
        # Return list of constraint expressions
        return list(constraints)
    
    def array_comprehension(self, meta: LarkMeta, range_expr: ASTNode, constraints: List[ASTNode]) -> ArrayComprehension:
        """✅ Grammar: '[' range_expr '|' comprehension_constraints ']' - brackets and pipe filtered by @v_args(inline=True)"""
        location = self._extract_location(meta)
        
        # Constraints come directly from comprehension_constraints rule
        return ArrayComprehension(
            expr=range_expr,
            constraints=constraints,
            location=location
        )
    

    
    def tuple_expr(self, meta: LarkMeta, *args: Union[ASTNode, Token]) -> TupleExpression:
        """✅ Grammar: LPAR expr ',' expr (',' expr)* RPAR"""
        location = self._extract_location(meta)
        
        # With @v_args(inline=True), string literals like "," are filtered out
        # So we should get: LPAR, expr, expr, ..., RPAR
        
        # Filter out LPAR and RPAR tokens using polymorphic token handling
        elements: List[ASTNode] = []
        for arg in args:
            token_info = handle_token(arg)
            if token_info.get('type') in ('LPAR', 'RPAR', 'COMMA'):
                continue  # Skip bracket and comma tokens
            else:
                elements.append(arg)  # Keep expressions
        
        return TupleExpression(elements=elements, location=location)
    

        
    def argument_list(self, meta: LarkMeta, *arguments: ASTNode) -> List[ASTNode]:
        """✅ Clean signature for argument lists"""
        # Filter out any Meta objects that might have leaked through
        clean_args: List[ASTNode] = []
        for arg in arguments:
            if self._is_meta_object(arg):
                continue  # Skip Meta objects
            clean_args.append(arg)
        return clean_args
    
    def primary_expr(self, meta: LarkMeta, *args: Union[ASTNode, Token]) -> ASTNode:
        """✅ Handle primary expressions including parenthesized expressions"""
        # For parenthesized expressions: LPAR expr RPAR using polymorphic token handling
        if len(args) == 3:
            first_token_info = handle_token(args[0])
            if first_token_info.get('type') == 'LPAR':
                return args[1]  # Return the expression inside parentheses
        
        if len(args) == 1:
            return args[0]  # Return single element directly
        else:
            # This should never happen with correct grammar
            raise ValueError(f"Unexpected primary_expr arguments: {len(args)} args: {args}")
        

        

    
    def expr_list(self, meta: LarkMeta, *expressions: ASTNode) -> List[ASTNode]:
        # Filter out any Meta objects that might have leaked through  
        clean_exprs: List[ASTNode] = []  # Precise type annotation for accumulator
        for expr in expressions:
            if self._is_meta_object(expr):
                continue  # Skip Meta objects
            clean_exprs.append(expr)
        return clean_exprs
        
    def param_list(self, meta: LarkMeta, *params: Parameter) -> List[Parameter]:
        """✅ Clean signature for parameter lists with Meta filtering"""
        # Filter out any Meta objects that might have leaked through
        clean_params: List[Parameter] = []
        for param in params:
            if self._is_meta_object(param):
                continue  # Skip Meta objects
            clean_params.append(param)
        return clean_params
        
    def shape_list(self, meta: LarkMeta, *shapes: Union[Token, ASTNode]) -> List[Union[Token, ASTNode]]:
        """✅ Clean signature for shape lists with Meta filtering"""
        # Filter out any Meta objects that might have leaked through
        clean_shapes: List[Union[Token, ASTNode]] = []
        for shape in shapes:
            if self._is_meta_object(shape):
                continue  # Skip Meta objects
            clean_shapes.append(shape)
        return clean_shapes
    
    def where_clause(self, meta: LarkMeta, *bindings: ASTNode) -> WhereClause:
        """Handle where clause - bindings are Expression objects (typically BinaryExpression)"""
        return WhereClause.from_list(list(bindings))
    
    def _is_type_annotation(self, obj: Any) -> bool:
        """Polymorphic check for type annotations"""
        # Trust: Lark type annotations have 'Type' in their class name
        obj_type = type(obj).__name__
        return 'Type' in obj_type
    
    def _is_meta_object(self, obj: Any) -> bool:
        """Polymorphic check for Meta objects"""
        # Trust: Lark Meta objects have 'Meta' in their class name
        obj_type = type(obj).__name__
        return 'Meta' in obj_type
        
    def where_constraints(self, meta: LarkMeta, where_clause_result: WhereClause) -> WhereClause:
        """✅ Passthrough for where_clause alias (grammar: where_constraints: where_clause)"""
        return where_clause_result
    

        
    def value_binding(self, meta: LarkMeta, name: Identifier, value: ASTNode) -> BinaryExpression:
        """Create BinaryExpression for value binding (x = expr)"""
        return BinaryExpression(left=name, operator=BinaryOp.ASSIGN, right=value, location=self._extract_location(meta))
        
    def range_membership(self, meta: LarkMeta, name: Identifier, range_expr: ASTNode) -> BinaryExpression:
        """Handle range membership: identifier in range_expr (only for comprehensions)"""
        # Create BinaryExpression for range membership (i in 1..10)
        # Not allowed in Einstein where clauses - domain definitions must be inline: let A[i in 0..N] = ...
        return BinaryExpression(left=name, operator=BinaryOp.IN, right=range_expr, location=self._extract_location(meta))
    
    def einstein_binding(self, meta: LarkMeta, binding: ASTNode) -> ASTNode:
        """Handle einstein_binding - passthrough since it's just value_binding | logical_or_expr"""
        # einstein_binding excludes range_membership - domain definitions must be inline
        return binding

    
    # Condition expression methods (delegate to main expression methods)

        

        

    # =========================================================================
    # OPERATOR METHODS - Transform terminal tokens to strings
    # =========================================================================
    
    def additive_op(self, meta, op):
        """Convert additive operator token to string"""
        return str(op)
        
    def multiplicative_op(self, meta, op):
        """Convert multiplicative operator token to string"""  
        return str(op)
        
    def power_op(self, meta, op):
        """Convert power operator token to string"""
        return str(op)
        
    def equality_op(self, meta, op):
        """Convert equality operator token to string"""
        return str(op)
        
    def relational_op(self, meta, op):
        """Convert relational operator token to string"""
        return str(op)
        
    def logical_and_op(self, meta, op):
        """Convert logical and operator token to string"""
        return str(op)
        
    def logical_or_op(self, meta, op):
        """Convert logical or operator token to string"""
        return str(op)
        
    def unary_op(self, meta, op):
        """Convert unary operator token to string"""
        return str(op)

    # =========================================================================
    # PIPELINE AND LAMBDA EXPRESSIONS
    # =========================================================================
    
    def pipeline_expr(self, meta, left, *ops_and_rights):
        """Handle pipeline expressions with multiple operators and optional else/catch clause"""
        location = self._extract_location(meta)
        
        # Separate operators/rights from the optional else/catch clause at the end
        clause_expr = None
        clause_type = None
        actual_ops_and_rights = list(ops_and_rights)
        
        # Check for else or catch clause at the end - trust Lark Tree has data attribute
        if actual_ops_and_rights and hasattr(actual_ops_and_rights[-1], 'data'):
            clause_node = actual_ops_and_rights.pop()
            if clause_node.data == 'pipeline_else_clause':
                clause_expr = clause_node.expr
                clause_type = PipelineClauseType.ELSE
            elif clause_node.data == 'pipeline_catch_clause':
                clause_expr = clause_node.handler
                clause_type = PipelineClauseType.CATCH
        
        # TODO: Add type-based validation later
        # - else clause should be used with Option types
        # - catch clause should be used with Result types
        # For now, allow flexible usage based on actual types rather than operators
        
        # Build left-associative chain of pipeline expressions
        result = left
        for i in range(0, len(actual_ops_and_rights), 2):
            if i + 1 < len(actual_ops_and_rights):
                op = actual_ops_and_rights[i]
                right = actual_ops_and_rights[i + 1]
                # op is already a BinaryOp enum from pipeline_op transformer method
                # If this is the last operation and we have a clause, add it now
                is_last = (i + 2 >= len(actual_ops_and_rights))
                else_clause = clause_expr if (is_last and clause_type == PipelineClauseType.ELSE) else None
                catch_clause = clause_expr if (is_last and clause_type == PipelineClauseType.CATCH) else None
                result = PipelineExpression(result, op, right, 
                                           else_clause=else_clause,
                                           catch_clause=catch_clause,
                                           location=location)
        
        return result
    
    def lambda_expr(self, meta, *args):
        """Handle lambda expressions: ||expr, |x|expr, |x,y|expr, etc."""
        location = self._extract_location(meta)
        # Args structure: PIPE, [param_list], PIPE, body
        
        if len(args) == 3:
            # Parameterless lambda: || body
            # Args: PIPE, PIPE, body
            body = args[2]
            # If body is a block dict, convert it to an expression
            body = self._convert_block_to_expr(body, location)
            return LambdaExpression([], body, location)
        elif len(args) >= 4:
            # Lambda with parameters: |param_list| body  
            # Args: PIPE, param_list, PIPE, body
            param_list = args[1]  # This is the param_list (list of Parameter objects)
            body = args[3]
            # If body is a block dict, convert it to an expression
            body = self._convert_block_to_expr(body, location)
            # Extract parameter names from Parameter objects
            if isinstance(param_list, list):
                param_names = [param.name for param in param_list]
            else:
                param_names = []
            return LambdaExpression(param_names, body, location)
        else:
            # Fallback for malformed input
            fallback_body = args[-1] if args else None
            if fallback_body:
                fallback_body = self._convert_block_to_expr(fallback_body, location)
            return LambdaExpression([], fallback_body, location)
    
    def _convert_block_to_expr(self, body, location):
        """Return body as-is (block and expr both valid). No dict conversion."""
        return body
    
    def atom(self, meta, *args):
        """Handle atom expressions - extract content from parentheses"""
        # If we have LPAR expr RPAR, return just the expr
        if len(args) == 3 and str(args[0]) == '(' and str(args[2]) == ')':
            return args[1]  # Return the middle expression
        # Otherwise, return the single argument (literal, identifier, etc.)
        elif len(args) == 1:
            return args[0]
        else:
            # Fallback - this shouldn't happen with current grammar
            return args[0] if args else None
    
    def pipeline_op(self, meta, op):
        """Convert pipeline operator token to BinaryOp enum"""
        return BinaryOp(str(op))
    
    def pipeline_else_clause(self, meta, expression):
        """Handle else clauses in pipeline expressions"""
        # Wrap in a simple container to identify it as an else clause
        class PipelineElseClause:
            def __init__(self, expression):
                self.expr = expression
                self.data = 'pipeline_else_clause'
        
        return PipelineElseClause(expression)
    
    def pipeline_catch_clause(self, meta, handler):
        """Handle catch clauses in pipeline expressions"""
        # Handler can be either a lambda expression or an identifier
        # Wrap in a simple container to identify it as a catch clause
        class PipelineCatchClause:
            def __init__(self, handler):
                self.handler = handler
                self.data = 'pipeline_catch_clause'
        
        return PipelineCatchClause(handler)
    
    def match_expr(self, meta, expr, arms=None):
        """Handle match expressions"""
        from ...shared.nodes import MatchExpression, MatchArm
        location = self._extract_location(meta)
        arm_list = arms if arms else []
        # Flatten if arms is a list of lists
        if arm_list and isinstance(arm_list, list):
            arm_list = [arm for arm in arm_list]
        
        # Convert arm tuples to MatchArm nodes
        match_arms = []
        for arm in arm_list:
            if isinstance(arm, MatchArm):
                match_arms.append(arm)
            elif isinstance(arm, tuple) and len(arm) == 2:
                # Old format: (pattern, body)
                pattern, body = arm
                match_arms.append(MatchArm(pattern=pattern, body=body, location=location))
        
        return MatchExpression(scrutinee=expr, arms=match_arms, location=location)
    
    def match_arm_list(self, meta, *arms):
        """Handle list of match arms"""
        # Just return the list of arms
        return list(arms)
    
    def match_arm(self, meta, *args):
        """Handle match arms: pattern (| pattern)* (where guard)? => body
        
        Lark filters anonymous terminals ("where", "=>"), so args contains only
        patterns (Pattern), PIPE tokens, and expressions. If a guard is present
        there are 2 trailing expressions (guard, body); otherwise just 1 (body).
        """
        from ...shared.nodes import MatchArm, GuardPattern, OrPattern, Pattern
        from lark import Token
        location = self._extract_location(meta)
        
        patterns = []
        exprs = []
        for arg in args:
            if isinstance(arg, Token) and str(arg) == '|':
                continue
            if isinstance(arg, Pattern):
                patterns.append(arg)
            else:
                exprs.append(arg)
        
        if len(exprs) >= 2:
            guard_expr, body = exprs[-2], exprs[-1]
        else:
            guard_expr, body = None, exprs[0] if exprs else None
        
        pat = patterns[0] if len(patterns) == 1 else OrPattern(alternatives=patterns, location=location)
        
        if guard_expr is not None:
            pat = GuardPattern(pattern=pat, guard=guard_expr, location=location)
        
        return MatchArm(pattern=pat, body=body, location=location)
    
    def literal_pattern(self, meta, literal):
        """Handle literal patterns"""
        from ...shared.nodes import LiteralPattern
        location = self._extract_location(meta)
        # literal is already a Literal node
        return LiteralPattern(value=literal, location=location)
    
    def identifier_pattern(self, meta, name):
        """Handle identifier patterns"""
        from ...shared.nodes import IdentifierPattern
        location = self._extract_location(meta)
        return IdentifierPattern(name=str(name), location=location)
    
    def wildcard_pattern(self, meta):
        """Handle wildcard patterns"""
        from ...shared.nodes import WildcardPattern
        location = self._extract_location(meta)
        return WildcardPattern(location=location)
    
    def tuple_pattern(self, meta, *patterns):
        """Handle tuple patterns: (a, b) or (a, b, c)"""
        from ...shared.nodes import TuplePattern
        location = self._extract_location(meta)
        # Filter out parentheses and commas
        pattern_list = [p for p in patterns if not (isinstance(p, str) and p in ('(', ')', ','))]
        return TuplePattern(patterns=pattern_list, location=location)
    
    def array_pattern(self, meta, elements=None):
        """Handle array patterns: [], [x], [first, ..rest], [..rest, last], [first, ..rest, last]"""
        from ...shared.nodes import ArrayPattern, Pattern, RestPattern
        location = self._extract_location(meta)
        
        if elements is None:
            # Empty array pattern []
            return ArrayPattern(patterns=[], location=location)
        
        # Parse all patterns as flat list
        # elements is a list from array_pattern_elements
        if isinstance(elements, list):
            pattern_list = elements
        elif isinstance(elements, Pattern):
            # Single pattern: [x] or [..rest]
            pattern_list = [elements]
        else:
            pattern_list = [elements] if elements else []
        
        # Filter out any non-Pattern items (shouldn't happen, but be safe)
        pattern_list = [p for p in pattern_list if isinstance(p, Pattern)]
        
        # Validate: only one RestPattern allowed (following Rust)
        rest_count = sum(1 for p in pattern_list if isinstance(p, RestPattern))
        if rest_count > 1:
            raise ValueError(f"Array pattern can have at most one rest pattern, found {rest_count}")
        
        return ArrayPattern(patterns=pattern_list, location=location)
    
    def array_pattern_elements(self, meta, *items):
        """Handle array pattern elements: collects all items into a list"""
        # With inline=True, items are passed as separate arguments
        # Filter out commas, keep only Pattern items
        from ...shared.nodes import Pattern
        pattern_list = [p for p in items if isinstance(p, Pattern)]
        return pattern_list
    
    def array_pattern_item(self, meta, item):
        """Handle array pattern item: pattern or rest_pattern"""
        # Just pass through - item is already a Pattern or RestPattern
        return item
    
    def rest_pattern(self, meta, pattern):
        """Handle rest pattern: ..pattern"""
        from ...shared.nodes import RestPattern
        location = self._extract_location(meta)
        return RestPattern(pattern=pattern, location=location)
    
    def array_pattern_with_rest(self, meta, *items):
        """Handle array pattern with rest at any position: [..rest], [first, ..rest], [..rest, last], [first, ..rest, last]"""
        # Following Rust's approach: return flat list with RestPattern in it
        # Filter out commas and other string tokens, keep all patterns (including RestPattern)
        from ...shared.nodes import Pattern
        pattern_items = [item for item in items if isinstance(item, Pattern)]
        return pattern_items
    
    def array_pattern_list(self, meta, *patterns):
        """Handle array pattern list: [x, y, z]"""
        # Filter out brackets and commas, return as list
        pattern_list = [p for p in patterns if not (isinstance(p, str) and p in ('[', ']', ','))]
        return pattern_list
    
    def binding_pattern(self, meta, name, pattern):
        """Handle binding patterns: name @ pattern"""
        from ...shared.nodes import BindingPattern
        location = self._extract_location(meta)
        return BindingPattern(name=str(name), pattern=pattern, location=location)
    
    def range_pattern_inclusive(self, meta, start_lit, _dotdoteq, end_lit):
        """Handle inclusive range patterns: start..=end"""
        from ...shared.nodes import RangePattern
        location = self._extract_location(meta)
        return RangePattern(start=start_lit, end=end_lit, inclusive=True, location=location)
    
    def range_pattern_exclusive(self, meta, start_lit, _dotdot, end_lit):
        """Handle exclusive range patterns: start..end"""
        from ...shared.nodes import RangePattern
        location = self._extract_location(meta)
        return RangePattern(start=start_lit, end=end_lit, inclusive=False, location=location)
    
    def pattern_list(self, meta, *patterns):
        """Handle pattern lists in constructor patterns"""
        # Just return the list of patterns
        return list(patterns)
    
    def constructor_pattern(self, meta, name, patterns=None):
        """Handle constructor patterns like Circle(r) or Point { x, y }"""
        from ...shared.nodes import ConstructorPattern
        location = self._extract_location(meta)
        pattern_list = patterns if patterns else []
        # Flatten if patterns is a list
        if pattern_list and isinstance(pattern_list, list):
            pattern_list = [p for p in pattern_list]
        elif pattern_list and not isinstance(pattern_list, list):
            pattern_list = [pattern_list]
        else:
            pattern_list = []
        # Check if this is a struct literal pattern (has braces in grammar)
        is_struct_literal = False  # Will be determined by grammar structure
        return ConstructorPattern(
            constructor_name=str(name),
            patterns=pattern_list,
            is_struct_literal=is_struct_literal,
            location=location
        )
    
    def struct_pattern(self, meta, name, lbrace, fields=None, rbrace=None):
        """Handle struct patterns like Point { x, y }"""
        from ...shared.nodes import ConstructorPattern
        location = self._extract_location(meta)
        pattern_list = fields if fields else []
        # Flatten if patterns is a list
        if pattern_list and isinstance(pattern_list, list):
            pattern_list = [p for p in pattern_list]
        elif pattern_list and not isinstance(pattern_list, list):
            pattern_list = [pattern_list]
        else:
            pattern_list = []
        return ConstructorPattern(
            constructor_name=str(name),
            patterns=pattern_list,
            is_struct_literal=True,
            location=location
        )
    
    def struct_field_pattern_list(self, meta, *fields):
        """Transform struct field pattern list"""
        return list(fields)
    
    def struct_field_pattern(self, meta, name, pattern=None):
        """Transform struct field pattern: x or x: pattern"""
        # Return tuple (name, pattern) or just name
        if pattern:
            return (str(name), pattern)
        return str(name)
    
    def tuple_destructure_pattern(self, meta: LarkMeta, *annotated_vars) -> TupleDestructurePattern:
        """Handle tuple destructuring patterns like (x, y, z) or (x: i32, y: str)"""
        from ...shared import AnnotatedVariable
        location = self._extract_location(meta)
        
        # Filter out parentheses and comma tokens, keep only AnnotatedVariable instances
        variables = []
        for item in annotated_vars:
            # Skip string tokens like parentheses and commas
            if isinstance(item, str) and item in ('(', ')', ','):
                continue
            # Add AnnotatedVariable instances
            elif isinstance(item, AnnotatedVariable):
                variables.append(item)
        
        return TupleDestructurePattern(variables=variables, location=location)
    
    def annotated_var(self, meta: LarkMeta, *args) -> 'AnnotatedVariable':
        """Transform annotated variable: NAME or NAME COLON type"""
        from ...shared import AnnotatedVariable
        
        # Parse args: either [name] or [name, colon_token, type]
        variable_name = None
        type_annotation = None
        
        for arg in args:
            # Skip colon token
            if isinstance(arg, str) and arg == ':':
                continue
            # First non-token argument is the variable name
            elif variable_name is None:
                variable_name = str(arg)
            # Second non-token argument is the type annotation
            else:
                type_annotation = arg
                break
        
        return AnnotatedVariable(name=variable_name, type_annotation=type_annotation)

    # =========================================================================
    # NO TERMINAL METHODS NEEDED
    # =========================================================================
    # Lark automatically handles NAME, INTEGER, ESCAPED_STRING tokens

