"""
AST to IR Lowering

Rust Pattern: rustc_hir::lowering
Reference: PASS_SYSTEM_DESIGN.md, IR_DESIGN.md

Design Pattern: Visitor pattern for AST traversal (no isinstance/hasattr)
"""

import logging
from typing import Optional, List, Union, Dict, Tuple, Any

logger = logging.getLogger(__name__)
from ..passes.base import BasePass, TyCtxt
from ..ir.nodes import (
    ProgramIR, ExpressionIR, BindingIR, FunctionDefIR, FunctionValueIR, ConstantDefIR, EinsteinIR,
    LiteralIR, IdentifierIR, BinaryOpIR, UnaryOpIR, FunctionCallIR,
    ParameterIR, BlockExpressionIR, IfExpressionIR, LambdaIR,
    ModuleIR, IRNode, RectangularAccessIR, JaggedAccessIR,
    ArrayLiteralIR, TupleExpressionIR, TupleAccessIR, InterpolatedStringIR,
    CastExpressionIR, MemberAccessIR, TryExpressionIR, MatchExpressionIR,
    ReductionExpressionIR, WhereExpressionIR,
    PipelineExpressionIR, BuiltinCallIR,
    LiteralPatternIR, IdentifierPatternIR, WildcardPatternIR,
    TuplePatternIR, ArrayPatternIR, RestPatternIR, GuardPatternIR,
    OrPatternIR, ConstructorPatternIR, BindingPatternIR, RangePatternIR,
    MatchArmIR, WhereClauseIR, EinsteinClauseIR, PatternIR,
    RangeIR, ArrayComprehensionIR,
    IndexVarIR,
)
from ..shared.source_location import SourceLocation
from ..shared.defid import DefId, DefType
import sys
import os


def _defid_of_identifier_in_expr(expr: Optional[ExpressionIR], name: str) -> Optional[DefId]:
    """Return defid of first IdentifierIR or IndexVarIR with given name in expr tree. Used when building ReductionExpressionIR."""
    if expr is None:
        return None
    if isinstance(expr, IdentifierIR) and expr.name == name:
        return getattr(expr, 'defid', None)
    if isinstance(expr, IndexVarIR) and expr.name == name:
        return getattr(expr, 'defid', None)
    children: List[Optional[ExpressionIR]] = []
    if hasattr(expr, 'left') and hasattr(expr, 'right'):
        children = [getattr(expr, 'left'), getattr(expr, 'right')]
    elif hasattr(expr, 'operand'):
        children = [getattr(expr, 'operand')]
    elif hasattr(expr, 'array') and hasattr(expr, 'indices'):
        children = [getattr(expr, 'array')] + list(getattr(expr, 'indices') or [])
    elif hasattr(expr, 'body'):
        children = [getattr(expr, 'body')]
    elif hasattr(expr, 'object'):
        children = [getattr(expr, 'object')]
    elif hasattr(expr, 'arguments'):
        children = [getattr(expr, 'callee_expr', None)] + list(getattr(expr, 'arguments') or [])
    elif hasattr(expr, 'expr'):
        children = [getattr(expr, 'expr')]
    for c in children:
        if isinstance(c, ExpressionIR):
            out = _defid_of_identifier_in_expr(c, name)
            if out is not None:
                return out
    return None


# Import existing AST nodes and visitor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..shared.nodes import (
    Program as ASTProgram,
    FunctionDefinition as ASTFunctionDef,
    Expression as ASTExpression,
    Literal as ASTLiteral,
    Identifier as ASTIdentifier,
    BinaryExpression as ASTBinaryOp,
    UnaryExpression as ASTUnaryOp,
    FunctionCall as ASTFunctionCall,
    Parameter as ASTParameter,
    BlockExpression as ASTBlock,
    IfExpression as ASTIfExpr,
    LambdaExpression as ASTLambda,
    ArrayLiteral as ASTArrayLiteral,
    ArrayComprehension as ASTArrayComprehension,
    RectangularAccess as ASTRectangularAccess,
    JaggedAccess as ASTJaggedAccess,
    TupleExpression as ASTTupleExpression,
    InterpolatedString as ASTInterpolatedString,
    CastExpression as ASTCastExpression,
    MemberAccess as ASTMemberAccess,
    MethodCall as ASTMethodCall,
    PipelineExpression as ASTPipelineExpression,
    TryExpression as ASTTryExpression,
    MatchExpression as ASTMatchExpression,
    ReductionExpression as ASTReductionExpression,
    WhereExpression as ASTWhereExpression,
    Range as ASTRange,
    ExpressionStatement as ASTExpressionStatement,
    EinsteinDeclaration as ASTEinsteinDeclaration,
    InlineModule as ASTInlineModule
)
# Note: ConstantDef may not exist in AST - constants might be handled differently
# Check if it exists, otherwise we'll handle constants via a different mechanism
try:
    from ..shared.nodes import ConstantDef as ASTConstantDef
except ImportError:
    ASTConstantDef = None  # Constants handled differently
from ..shared.ast_visitor import ASTVisitor


def _defid_of_identifier_in_expr(expr: Optional[ExpressionIR], name: str) -> Optional[DefId]:
    """Return defid of first IdentifierIR or IndexVarIR with given name in expr tree. Used when building ReductionExpressionIR."""
    if expr is None:
        return None
    if isinstance(expr, IdentifierIR) and expr.name == name:
        return getattr(expr, 'defid', None)
    if isinstance(expr, IndexVarIR) and expr.name == name:
        return getattr(expr, 'defid', None)
    children: List[Optional[ExpressionIR]] = []
    if hasattr(expr, 'left') and hasattr(expr, 'right'):
        children = [getattr(expr, 'left'), getattr(expr, 'right')]
    elif hasattr(expr, 'operand'):
        children = [getattr(expr, 'operand')]
    elif hasattr(expr, 'array') and hasattr(expr, 'indices'):
        children = [getattr(expr, 'array')] + list(getattr(expr, 'indices') or [])
    elif hasattr(expr, 'body'):
        children = [getattr(expr, 'body')]
    elif hasattr(expr, 'object'):
        children = [getattr(expr, 'object')]
    elif hasattr(expr, 'arguments'):
        children = [getattr(expr, 'callee_expr', None)] + list(getattr(expr, 'arguments') or [])
    elif hasattr(expr, 'expr'):
        children = [getattr(expr, 'expr')]
    for c in children:
        if isinstance(c, ExpressionIR):
            out = _defid_of_identifier_in_expr(c, name)
            if out is not None:
                return out
    return None


class ASTToIRLoweringPass(BasePass):
    """
    AST to IR lowering pass (Rust naming: rustc_hir::lowering).
    
    Rust Pattern: rustc_hir::lowering
    
    Design Pattern: Visitor pattern for AST traversal (no isinstance/hasattr)
    
    Implementation Alignment: Follows Rust's HIR lowering:
    - AST → IR (early lowering, immediately after name resolution)
    - IR is desugared AST with DefIds
    - Source locations preserved
    - All metadata preserved
    - Desugars syntactic sugar (like Rust's HIR lowering)
    
    Reference: `rustc_hir::lowering` for HIR lowering
    """
    requires = []  # Depends on name resolution (DefIds needed)
    
    def run(self, ir_or_ast: Union[ProgramIR, ASTProgram], tcx: TyCtxt) -> ProgramIR:
        """
        Lower AST to IR (desugared, normalized).
        
        Rust Pattern: rustc_hir::lowering::lower_crate()
        
        Design Pattern: Type-safe dispatch (no isinstance)
        """
        # Type-safe check: if already IR, return as-is
        if isinstance(ir_or_ast, ProgramIR):
            return ir_or_ast
        
        # It's AST, lower it using visitor pattern
        if not isinstance(ir_or_ast, ASTProgram):
            raise TypeError(f"Expected ASTProgram or ProgramIR, got {type(ir_or_ast)}")
        
        lowerer = ASTToIRLowerer(tcx)
        return lowerer.lower_program(ir_or_ast)

class ASTToIRLowerer(ASTVisitor[Optional[IRNode]]):
    """
    AST to IR lowerer (Rust naming: rustc_hir::lowering).
    
    Rust Pattern: rustc_hir::lowering::LoweringContext
    
    Design Pattern: Visitor pattern for AST traversal (no isinstance/hasattr)
    
    Implementation Alignment: Follows Rust's lowering:
    - Converts AST nodes to IR nodes using visitor pattern
    - Preserves source locations
    - Attaches DefIds from name resolution
    - Desugars syntactic sugar
    
    Reference: `rustc_hir::lowering::LoweringContext` for lowering
    """
    
    def __init__(self, tcx: TyCtxt):
        self.tcx = tcx
        self._all_functions: List[FunctionDefIR] = []
        self._all_bindings: List[BindingIR] = []
    
    def lower_program(self, ast: ASTProgram) -> ProgramIR:
        """Lower entire program"""
        # AST Program has statements, not separate functions/constants lists
        # Extract functions and constants from statements
        # Reset nested function tracking for this program
        self._all_functions = []
        self._all_bindings = []
        self._resolver_defid_to_lowered = {}
        self._build_resolver_defid_to_lowered()
        statements: List[Any] = []

        from ..shared.nodes import FunctionDefinition
        from ..shared.nodes import ExpressionStatement, EinsteinDeclaration as ASTEinsteinDeclaration

        for stmt_idx, stmt in enumerate(ast.statements):
            stmt_ir = stmt.accept(self)

            if isinstance(stmt_ir, list):
                for sub_stmt in stmt_ir:
                    if isinstance(sub_stmt, BindingIR):
                        statements.append(sub_stmt)
                continue

            if isinstance(stmt_ir, (FunctionDefIR, ConstantDefIR, BindingIR)):
                statements.append(stmt_ir)
            elif isinstance(stmt_ir, ExpressionIR):
                statements.append(stmt_ir)

        modules: List[Any] = []
        module_functions = self._lower_module_functions()
        statements.extend(module_functions)
        nested_functions = [f for f in self._all_functions if f not in statements]
        statements.extend(nested_functions)

        return ProgramIR(
            statements=statements,
            source_files=self.tcx.source_files,
            modules=modules,
        )
    
    def _lower_module_functions(self) -> List[FunctionDefIR]:
        """
        Lower all module functions to IR (Rust pattern: all module functions are part of the program).
        
        Following Rust's pattern - all modules (stdlib and user) are treated the same.
        They're all part of the crate and should be available at runtime.
        
        This method finds all module functions (stdlib and user) registered in the resolver and lowers them to IR.
        
        Rust alignment: Uses resolver's _def_registry to get function definitions (already parsed
        during name resolution), avoiding re-parsing. This matches Rust's precompiled stdlib model.
        """
        module_functions = []
        
        if not self.tcx.resolver:
            return module_functions
        
        from ..shared.nodes import FunctionDefinition
        from ..shared.defid import DefType
        
        # Collect all module functions (both stdlib and user modules)
        # Rust alignment: All modules are treated the same - they're all part of the crate
        # NOTE: Functions are stored in _def_registry, not _symbol_table (to support overloading)
        module_function_defids = {}
        stdlib_count = 0
        user_count = 0
        
        # Iterate through _def_registry to find all functions
        for defid, (def_type, definition) in self.tcx.def_registry.items():
            # Only process functions
            if def_type != DefType.FUNCTION:
                continue
            
            # Extract module path from definition if available
            # Functions have their module path encoded in their symbol key
            # We need to find it by looking at the definition's attributes or the defid lookup
            if not isinstance(definition, FunctionDefinition):
                continue
            
            # Get function name from definition
            func_name = definition.name if hasattr(definition, 'name') else None
            if not func_name:
                continue
            
            # Get module_path from function definition (set during import)
            module_path = getattr(definition, 'module_path', None)
            
            # Skip if module_path couldn't be determined or it's a top-level function
            if module_path is None or module_path == ():
                continue
            
            module_function_defids[defid] = (module_path, func_name)
            if module_path[0] == 'std':
                stdlib_count += 1
            else:
                user_count += 1
        
        # Lower each module function
        for defid, (module_path, func_name) in module_function_defids.items():
            registry_entry = self.tcx.get_definition(defid)
            if not registry_entry:
                raise ValueError(
                    f"DefId {defid.krate}:{defid.index} not found in def registry for {module_path}::{func_name}. "
                    "Ensure name resolution has run and the definition is registered."
                )
            
            def_type, definition = registry_entry
            if def_type != DefType.FUNCTION:
                continue
            
            # Get function definition (already parsed AST node)
            if not isinstance(definition, FunctionDefinition):
                continue
            
            # Lower function to IR (Rust pattern: reuse parsed AST, don't re-parse)
            # CRITICAL: Before lowering, ensure the function body is fully resolved
            # The function definition should have been resolved during name resolution,
            # but we need to make sure all identifiers in the body have DefIds
            try:
                # Ensure function body identifiers are resolved before lowering
                # This is a safety check - the body should already be resolved during name resolution
                from ..shared.nodes import Identifier as ASTIdentifier
                def check_and_resolve_identifiers(node, scope_manager, resolver):
                    """Recursively check and resolve identifiers in AST"""
                    if isinstance(node, ASTIdentifier):
                        if not hasattr(node, 'defid') or node.defid is None:
                            # Try to resolve it
                            scope = scope_manager.current_scope()
                            if scope:
                                binding = scope.lookup(node.name)
                                if binding and binding.defid:
                                    object.__setattr__(node, 'defid', binding.defid)
                                    object.__setattr__(node, '_defid', binding.defid)
                    # Recursively visit children
                    # Handle different node types properly
                    from ..shared.nodes import EinsteinDeclaration as ASTEinsteinDeclaration
                    if isinstance(node, ASTEinsteinDeclaration):
                        # For Einstein declarations, visit value and where_clause, but don't access .name (use .array_name)
                        if hasattr(node, 'value') and node.value:
                            check_and_resolve_identifiers(node.value, scope_manager, resolver)
                        if hasattr(node, 'where_clause') and node.where_clause:
                            if hasattr(node.where_clause, 'constraints'):
                                for constraint in node.where_clause.constraints:
                                    check_and_resolve_identifiers(constraint, scope_manager, resolver)
                        # Visit indices
                        if hasattr(node, 'indices'):
                            for idx in node.indices:
                                check_and_resolve_identifiers(idx, scope_manager, resolver)
                    else:
                        # For other nodes, recursively visit all attributes
                        # Skip 'name' attribute for EinsteinDeclaration (use 'array_name' instead)
                        from ..shared.nodes import EinsteinDeclaration as ASTEinsteinDeclaration
                        skip_attrs = set()
                        if isinstance(node, ASTEinsteinDeclaration):
                            skip_attrs.add('name')  # EinsteinDeclaration uses 'array_name', not 'name'
                        
                        for attr_name in dir(node):
                            if attr_name.startswith('_') or attr_name in skip_attrs:
                                continue
                            try:
                                attr_value = getattr(node, attr_name)
                                if attr_value and not callable(attr_value):
                                    if isinstance(attr_value, list):
                                        for item in attr_value:
                                            if item:
                                                check_and_resolve_identifiers(item, scope_manager, resolver)
                                    elif hasattr(attr_value, '__class__'):
                                        check_and_resolve_identifiers(attr_value, scope_manager, resolver)
                            except AttributeError:
                                # Skip attributes that don't exist (e.g., 'name' on EinsteinDeclaration)
                                pass
                            except Exception:
                                # Skip other errors during recursive traversal
                                pass
                
                # For user module functions, ensure identifiers are resolved before lowering
                # For stdlib functions, skip body resolution - let monomorphization service handle it
                # NOTE: This check is disabled because scope_manager is not available in TyCtxt during lowering.
                # Name resolution should have already resolved all identifiers, so this check is redundant.
                # if module_path[0] != 'std':
                #     # User module function - resolve identifiers in body before lowering
                #     from ..shared.nodes import BlockExpression
                #     if hasattr(definition, 'body') and isinstance(definition.body, BlockExpression):
                #         # Resolve identifiers in function body
                #         check_and_resolve_identifiers(definition.body, self.tcx.scope_manager, self.tcx.resolver)
                
                # SKIP body resolution for stdlib functions - let monomorphization service handle it
                # Rationale: stdlib functions will be specialized by the mono service when actually called.
                # During specialization, the mono service will properly set up parameter scope and resolve identifiers.
                # This avoids the complexity of resolving function bodies with Python module calls during lowering.
                # The function definition is registered with its AST, and the mono service will handle specialization.
                # Safely get function name (FunctionDefinition has 'name', not 'array_name')
                func_name = getattr(definition, 'name', None) or getattr(definition, 'array_name', None) or func_name

                func_ir = definition.accept(self)
                if isinstance(func_ir, FunctionDefIR):
                    from ..shared.defid import assert_defid
                    assert_defid(defid)
                    func_ir.defid = defid
                    module_functions.append(func_ir)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Lowering skipped for %s (DefId %s): %s",
                    func_name, defid, e,
                    exc_info=True,
                )
                continue

        return module_functions

    def _build_resolver_defid_to_lowered(self) -> None:
        """
        Build map from resolver DefId to lowered DefId (Rust: use definition DefId at call sites).
        Call before lowering statements so calls resolve to the DefId we actually lower.
        """
        self._resolver_defid_to_lowered = {}
        if not self.tcx.resolver:
            return
        from ..shared.nodes import FunctionDefinition
        from ..shared.defid import DefType
        for defid, (def_type, definition) in self.tcx.def_registry.items():
            if def_type != DefType.FUNCTION or not isinstance(definition, FunctionDefinition):
                continue
            func_name = getattr(definition, 'name', None)
            if not func_name:
                continue
            module_path = getattr(definition, 'module_path', None)
            if module_path is None or module_path == ():
                continue
            other = self.tcx.resolver.get_defid(module_path, func_name, DefType.FUNCTION)
            if other is not None and other != defid:
                self._resolver_defid_to_lowered[other] = defid
    
    def _file_path_to_module_path(self, file_path: str) -> Optional[Tuple[str, ...]]:
        """
        Convert file path to module path (same logic as name_resolution).
        
        Examples:
        - '/path/to/stdlib/math/sqrt.ein' → ('std', 'math', 'sqrt')
        - 'stdlib/math/basic.ein' → ('std', 'math', 'basic')
        """
        from pathlib import Path
        path_parts = Path(file_path).parts
        
        # Find stdlib in path
        if 'stdlib' in path_parts:
            stdlib_idx = path_parts.index('stdlib')
            after_stdlib = path_parts[stdlib_idx + 1:]
            if not after_stdlib:
                return None
            
            module_name = Path(after_stdlib[-1]).stem
            if len(after_stdlib) > 1:
                subdirs = after_stdlib[:-1]
                return ('std',) + subdirs + (module_name,)
            else:
                return ('std', module_name)
        
        # If file is not in stdlib, return None (legitimate - not all files are in stdlib)
        return None
    
    def visit_function_definition(self, ast_func: ASTFunctionDef) -> Optional[FunctionDefIR]:
        """Lower function definition - allocate at creation"""
        location = self._get_source_location(ast_func)
        defid = getattr(ast_func, 'defid', None)
        logger.debug(f"[ast_to_ir] Lowering function {ast_func.name} with DefId {defid}")
        parameters = []
        for param in ast_func.parameters:
            param_location = self._get_source_location(ast_func)
            if hasattr(param, 'location') and param.location:
                param_location = self._get_source_location_from_ast_location(param.location)
            param_defid = getattr(param, 'defid', None)
            if param_defid is None:
                raise RuntimeError(
                    f"Parameter '{getattr(param, 'name', '?')}' of function '{ast_func.name}' has no defid. "
                    "Ensure NameResolutionPass runs on AST before ASTToIRLoweringPass and allocates defids for parameters."
                )
            param_type = None
            if hasattr(param, 'type_annotation') and param.type_annotation:
                param_type = param.type_annotation
            param_ir = ParameterIR(
                name=param.name,
                param_type=param_type,
                location=param_location,
                defid=param_defid
            )
            parameters.append(param_ir)

        # Lower body using visitor pattern
        body_ir = ast_func.body.accept(self)
        if not isinstance(body_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower function body for '{ast_func.name}': expected ExpressionIR, got {type(body_ir).__name__} at {location}"
            )
        
        # Extract return type from AST function definition
        # Type annotations are PrimitiveType objects from parser
        return_type = None
        if hasattr(ast_func, 'return_type') and ast_func.return_type:
            return_type = ast_func.return_type
        
        func_value = FunctionValueIR(
            parameters=parameters,
            body=body_ir,
            location=location,
            return_type=return_type,
        )
        func_ir = FunctionDefIR(
            name=ast_func.name,
            expr=func_value,
            location=location,
            defid=defid,
        )
        self._all_functions.append(func_ir)
        return func_ir

    def visit_constant_def(self, ast_const: ASTConstantDef) -> Optional[ConstantDefIR]:
        """Lower constant definition - visitor pattern (no isinstance)"""
        location = self._get_source_location(ast_const)
        defid = getattr(ast_const, 'defid', None)  # Trust infrastructure
        
        value_ir = ast_const.value.accept(self)  # Visitor pattern
        if not isinstance(value_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower constant value for '{ast_const.name}': expected ExpressionIR, got {type(value_ir).__name__} at {location}"
            )
        
        return ConstantDefIR(
            name=ast_const.name,
            expr=value_ir,
            location=location,
            defid=defid,
        )
    
    def visit_parameter(self, ast_param: ASTParameter) -> Optional[ParameterIR]:
        """Lower parameter - visitor pattern (no isinstance)"""
        location = self._get_source_location(ast_param)
        defid = getattr(ast_param, 'defid', None)  # Trust infrastructure
        
        return ParameterIR(
            name=ast_param.name,
            param_type=None,
            location=location,
            defid=defid
        )
    
    def visit_literal(self, ast_lit: ASTLiteral) -> LiteralIR:
        """Lower literal. Type pass annotates types."""
        location = self._get_source_location(ast_lit)
        return LiteralIR(value=ast_lit.value, location=location)
    
    def visit_identifier(self, ast_id: ASTIdentifier) -> IdentifierIR:
        """Lower identifier - defid from name resolution only (no name handling here)."""
        location = self._get_source_location(ast_id)
        name = ast_id.name if isinstance(ast_id.name, str) else getattr(ast_id.name, 'value', str(ast_id.name))
        defid = getattr(ast_id, 'defid', None)
        return IdentifierIR(name=name, location=location, defid=defid)
    
    def visit_binary_expression(self, ast_op: ASTBinaryOp) -> Optional[BinaryOpIR]:
        """Lower binary operation - visitor pattern (no isinstance)"""
        location = self._get_source_location(ast_op)
        
        left = ast_op.left.accept(self)  # Visitor pattern
        right = ast_op.right.accept(self)  # Visitor pattern
        
        if not isinstance(left, ExpressionIR) or not isinstance(right, ExpressionIR):
            raise ValueError(f"Invalid expression type: left={type(left).__name__}, right={type(right).__name__}")
        
        return BinaryOpIR(
            operator=ast_op.operator,
            left=left,
            right=right,
            location=location
        )
    
    def visit_unary_expression(self, ast_op: ASTUnaryOp) -> Optional[UnaryOpIR]:
        """Lower unary operation - visitor pattern (no isinstance)"""
        location = self._get_source_location(ast_op)
        
        operand = ast_op.operand.accept(self)  # Visitor pattern
        if not isinstance(operand, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower unary operation operand: expected ExpressionIR, got {type(operand).__name__} at {location}"
            )
        
        return UnaryOpIR(
            operator=ast_op.operator,
            operand=operand,
            location=location
        )
    
    def visit_function_call(self, ast_call: ASTFunctionCall) -> Optional[FunctionCallIR]:
        """Lower function call - visitor pattern. Callee and args always lower themselves."""
        location = self._get_source_location(ast_call)
        from ..shared.nodes import ModuleAccess

        # Let function_expr and arguments handle themselves (visitor pattern)
        callee_ir = ast_call.function_expr.accept(self)
        arguments = []
        for arg in ast_call.arguments:
            arg_ir = arg.accept(self)
            if isinstance(arg_ir, ExpressionIR):
                arguments.append(arg_ir)

        # Expression callee (lambda, etc.) -> use callee_expr; name/module callee -> use name/defid path
        if not isinstance(ast_call.function_expr, (ASTIdentifier, ModuleAccess)):
            if not isinstance(callee_ir, ExpressionIR):
                return None
            return FunctionCallIR(
                callee_expr=callee_ir,
                location=location,
                arguments=arguments,
            )

        # Extract module path FIRST (for Python module support)
        # Try to get from name resolution first (stored as _resolved_module_path on function_expr)
        # Fallback to extracting from ModuleAccess if not available
        module_path = None
        
        
        
        # First, try to get from function_expr if it's a ModuleAccess (stored during name resolution)
        if isinstance(ast_call.function_expr, ModuleAccess):
            module_path = getattr(ast_call.function_expr, '_resolved_module_path', None)
            
        
        # If not found, extract from ModuleAccess structure
        if module_path is None and isinstance(ast_call.function_expr, ModuleAccess):
            # Extract module path from ModuleAccess
            path_parts = []
            current = ast_call.function_expr
            while isinstance(current, ModuleAccess):
                path_parts.insert(0, current.property)
                current = current.object
            if isinstance(current, ASTIdentifier):
                path_parts.insert(0, current.name)
            
            if len(path_parts) >= 2:
                # Module path is all but last component, function name is last
                module_path = tuple(path_parts[:-1])
                
                
                # Resolve alias if the first component is an alias (e.g., 'math' -> 'python::math')
                if self.tcx.resolver and module_path:
                    alias_info = self.tcx.resolver.lookup_alias(module_path[0])
                    
                    if alias_info:
                        # Resolve alias: ('math',) -> ('std', 'math') or ('python', 'math')
                        resolved_module_path = alias_info
                        # If there are more components, append them
                        if len(module_path) > 1:
                            module_path = resolved_module_path + module_path[1:]
                        else:
                            module_path = resolved_module_path
                        
                        # Check if resolved path is an Einlang stdlib module (starts with 'std')
                        # If so, try to get DefId for the function instead of treating as Python module
                        if module_path and len(module_path) > 0 and module_path[0] == 'std':
                            # This is an Einlang stdlib function - try to get DefId
                            func_name = path_parts[-1]  # Function name is last component
                            if self.tcx.resolver:
                                stdlib_defid = self.tcx.resolver.get_defid(module_path, func_name, DefType.FUNCTION)
                                if stdlib_defid:
                                    # Found stdlib function - use DefId and clear module_path
                                    function_defid = stdlib_defid
                                    module_path = None  # Clear to indicate stdlib function, not Python module
                                    logger.debug(f"[ast_to_ir] Resolved stdlib function {module_path}::{func_name} with DefId {stdlib_defid}")
                                else:
                                    # Function not found - module may not be loaded yet
                                    # This shouldn't happen if name resolution worked, but log for debugging
                                    logger.warning(f"[ast_to_ir] Stdlib function {module_path}::{func_name} not found in resolver - may not be loaded")
            else:
                pass
        
        # arguments already lowered above (callee and args handle themselves)
        # Extract function name from function_expr (AST FunctionCall uses function_expr, not function_name)
        # function_expr can be Identifier, MemberAccess, ModuleAccess, etc.
        function_name = "unknown"
        function_defid = getattr(ast_call, 'function_defid', None)  # Trust infrastructure
        
        # Also check function_expr.defid if function_defid is None (for Identifier function calls)
        if function_defid is None and isinstance(ast_call.function_expr, ASTIdentifier):
            function_defid = getattr(ast_call.function_expr, 'defid', None)
        
        
        # If function_expr is an Identifier, get the name directly
        if isinstance(ast_call.function_expr, ASTIdentifier):
            function_name = ast_call.function_expr.name
        else:
            # ModuleAccess or other expression types
            if isinstance(ast_call.function_expr, ModuleAccess):
                # Get function name from module access (property field)
                function_name = ast_call.function_expr.property
                # Also try to get DefId from module access node (set by visit_module_access)
                if function_defid is None:
                    module_access_defid = getattr(ast_call.function_expr, 'defid', None)
                    if module_access_defid:
                        function_defid = module_access_defid
                    elif module_path and self.tcx.resolver:
                        # Module access wasn't resolved - try to resolve it now using extracted module_path
                        func_name = function_name  # Already extracted from property
                        
                        # Check if this is an Einlang stdlib module (starts with 'std')
                        # If so, try to get DefId for the function
                        if module_path and len(module_path) > 0 and module_path[0] == 'std':
                            stdlib_defid = self.tcx.resolver.get_defid(module_path, func_name, DefType.FUNCTION)
                            if stdlib_defid:
                                function_defid = stdlib_defid
                                function_name = func_name
                                module_path = None  # Clear to indicate stdlib function, not Python module
                        else:
                            # CRITICAL: Never treat python:: paths as stdlib functions
                            # python:: paths should always be Python module calls
                            if module_path and len(module_path) > 0 and module_path[0] == 'python':
                                # This is a Python module call - preserve module_path, don't try to resolve as stdlib
                                # function_defid should remain None for Python module calls
                                pass
                            else:
                                # Try to resolve through resolver (may be a Python module or alias)
                                # Check if it's an alias first
                                alias_info = self.tcx.resolver.lookup_alias(module_path[0] if module_path else "")
                                if alias_info:
                                    # Resolve through alias
                                    resolved_module_path = alias_info
                                    # Check if resolved path is stdlib (but NOT if original was python::)
                                    if resolved_module_path and len(resolved_module_path) > 0 and resolved_module_path[0] == 'std':
                                        # This is an Einlang stdlib function - try to get DefId
                                        stdlib_defid = self.tcx.resolver.get_defid(resolved_module_path, func_name, DefType.FUNCTION)
                                        if stdlib_defid:
                                            function_defid = stdlib_defid
                                            function_name = func_name
                                            module_path = None  # Clear to indicate stdlib function
                                    elif resolved_module_path and len(resolved_module_path) > 0 and resolved_module_path[0] == 'python':
                                        # Resolved to python:: - keep as Python module call (don't resolve DefId)
                                        module_path = resolved_module_path
                                        # function_defid should remain None for Python module calls
                                    else:
                                        func_defid = self.tcx.resolver.get_defid(resolved_module_path, func_name, DefType.FUNCTION)
                                        if func_defid:
                                            function_defid = func_defid
                                            function_name = func_name
            elif hasattr(ast_call, '_resolved_function_name'):
                # Use resolved function name if available (from name resolution)
                function_name = ast_call._resolved_function_name
            else:
                # Fallback: try to get name from function_expr
                function_name = getattr(ast_call.function_expr, 'name', None) or getattr(ast_call.function_expr, 'property', None) or getattr(ast_call.function_expr, 'member', 'unknown')
        
        # Check if this is a builtin function call (should use BuiltinCallIR, not FunctionCallIR)
        # Builtins are identified by having DefId with krate=1 (Rust pattern)
        from ..shared.defid import DefId, _BUILTIN_CRATE, DefType, FIXED_BUILTIN_ORDER, fixed_builtin_defid
        is_builtin = False
        builtin_defid = None
        if function_defid:
            # Check if DefId is in builtin crate (Rust pattern: krate=1)
            if function_defid.krate == _BUILTIN_CRATE:
                # Verify it's actually a builtin by checking resolver
                if self.tcx.resolver:
                    def_type, _ = self.tcx.get_definition(function_defid) or (None, None)
                    if def_type == DefType.BUILTIN:
                        is_builtin = True
                        builtin_defid = function_defid
        
        # Also check by name as fallback (for builtins not yet resolved)
        if not is_builtin and function_name in ('assert', 'print', 'len', 'typeof'):
            # This is a builtin - should use BuiltinCallIR
            # Try to get DefId from resolver
            if self.tcx.resolver:
                builtin_defid = self.tcx.resolver.get_defid((), function_name, DefType.BUILTIN)
                if builtin_defid:
                    def_type, _ = self.tcx.get_definition(builtin_defid) or (None, None)
                    if def_type == DefType.BUILTIN:
                        is_builtin = True
        
        
        
        # If function_defid is None and we have a resolver, try only builtin lookup.
        # Do NOT resolve get_defid((), name, FUNCTION) here - that would bypass use-statement
        # scoping (e.g. function only imported inside another scope would still resolve).
        if not is_builtin and function_defid is None and self.tcx.resolver:
            looked_up_defid = self.tcx.resolver.get_defid((), function_name, DefType.BUILTIN)
            if looked_up_defid:
                is_builtin = True
                builtin_defid = looked_up_defid
        if is_builtin and builtin_defid is None and function_name in FIXED_BUILTIN_ORDER:
            builtin_defid = fixed_builtin_defid(function_name)
        if is_builtin:
            from ..ir.nodes import BuiltinCallIR
            return BuiltinCallIR(
                builtin_name=function_name,
                args=arguments,
                location=location,
                defid=builtin_defid
            )
        
        # Regular function call (module_path already extracted at the beginning)
        # CRITICAL FIX: For python:: paths, function_defid MUST be None
        if module_path and len(module_path) > 0 and module_path[0] == 'python':
            function_defid = None

        if function_defid is not None:
            function_defid = getattr(self, '_resolver_defid_to_lowered', {}).get(function_defid, function_defid)

        callee_expr = IdentifierIR(name=function_name, location=location, defid=function_defid)
        return FunctionCallIR(
            callee_expr=callee_expr,
            location=location,
            arguments=arguments,
            module_path=module_path,
        )
    
    def visit_block_expression(self, ast_block: ASTBlock) -> Optional[BlockExpressionIR]:
        """
        Lower block
        """
        location = self._get_source_location(ast_block)
        statements = []
        for stmt in ast_block.statements:
            # Visitor pattern: dispatch to appropriate visit_* method
            stmt_ir = stmt.accept(self)
            if stmt_ir is not None:
                # Handle tuple destructuring: visit_variable_declaration may return a list
                if isinstance(stmt_ir, list):
                    statements.extend(stmt_ir)
                else:
                    statements.append(stmt_ir)
        
        final_expr = None
        if ast_block.final_expr is not None:
            final_expr_ir = ast_block.final_expr.accept(self)
            if isinstance(final_expr_ir, ExpressionIR):
                final_expr = final_expr_ir

        return BlockExpressionIR(
            statements=statements,
            final_expr=final_expr,
            location=location
        )
    
    def visit_if_expression(self, ast_if: ASTIfExpr) -> Optional[IfExpressionIR]:
        """Lower if expression - visitor pattern (no isinstance)"""
        location = self._get_source_location(ast_if)
        
        # Lower condition
        condition = ast_if.condition.accept(self)  # Visitor pattern
        if not isinstance(condition, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower if condition: expected ExpressionIR, got {type(condition).__name__} at {location}"
            )
        
        # Lower then_block (BlockExpression) to BlockExpressionIR
        # Keep the entire BlockExpressionIR (not just final_expr) to preserve statements
        then_block_ir = ast_if.then_block.accept(self)  # Visitor pattern
        if not isinstance(then_block_ir, BlockExpressionIR):
            raise RuntimeError(
                f"Failed to lower if then_block: expected BlockExpressionIR, got {type(then_block_ir).__name__} at {location}"
            )
        # Use the entire block as then_expr (BlockExpressionIR is an ExpressionIR)
        then_expr = then_block_ir
        
        # Lower else_block if present
        # Keep the entire BlockExpressionIR (not just final_expr) to preserve statements
        else_expr = None
        if ast_if.else_block:
            else_block_ir = ast_if.else_block.accept(self)  # Visitor pattern
            if isinstance(else_block_ir, BlockExpressionIR):
                # Use the entire block as else_expr (BlockExpressionIR is an ExpressionIR)
                else_expr = else_block_ir
            elif isinstance(else_block_ir, ExpressionIR):
                else_expr = else_block_ir
        
        return IfExpressionIR(
            condition=condition,
            then_expr=then_expr,
            else_expr=else_expr,
            location=location
        )
    
    def visit_lambda_expression(self, ast_lambda: ASTLambda) -> Optional[LambdaIR]:
        """Lower lambda - allocate at creation"""
        location = self._get_source_location(ast_lambda)

        parameters = []
        param_defids = getattr(ast_lambda, '_param_defids', {}) or {}
        for param_name in ast_lambda.parameters:
            param_defid = param_defids.get(param_name)
            parameters.append(ParameterIR(
                name=param_name,
                param_type=None,
                location=location,
                defid=param_defid
            ))

        body_ir = ast_lambda.body.accept(self)
        logger.debug(f"[visit_lambda_expression] Lambda body type: {type(body_ir).__name__}, is ExpressionIR: {isinstance(body_ir, ExpressionIR) if body_ir else False}")
        if not isinstance(body_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower lambda body: expected ExpressionIR, got {type(body_ir).__name__} at {location}"
            )
        
        return LambdaIR(
            parameters=parameters,
            body=body_ir,
            location=location
        )
    
    def _get_source_location(self, node: ASTExpression) -> SourceLocation:
        """Extract source location from AST node - trust infrastructure"""
        # Trust infrastructure: AST nodes have location
        loc = node.location
        
        # Handle None location (fallback)
        if loc is None:
            # Use default location from tcx if available
            default_file = list(self.tcx.source_files.keys())[0] if self.tcx.source_files else "unknown"
            return SourceLocation(
                file=default_file,
                line=1,
                column=1,
                end_line=1,
                end_column=1
            )
        
        return self._get_source_location_from_ast_location(loc)
    
    def _get_source_location_from_ast_location(self, loc) -> SourceLocation:
        """Extract SourceLocation from AST SourceLocation object"""
        return SourceLocation(
            file=loc.file,
            line=loc.line,
            column=loc.column,
            end_line=getattr(loc, 'end_line', loc.line),
            end_column=getattr(loc, 'end_column', loc.column)
        )
    
    # Required visitor methods (ASTVisitor interface) - Expression visitors
    def visit_array_literal(self, node: ASTArrayLiteral) -> Optional[ExpressionIR]:
        """Lower array literal - visitor pattern"""
        location = self._get_source_location(node)
        elements = []
        for elem in node.elements:
            elem_ir = elem.accept(self)
            if isinstance(elem_ir, ExpressionIR):
                elements.append(elem_ir)
        return ArrayLiteralIR(
            elements=elements,
            location=location
        )
    
    def visit_array_comprehension(self, node: ASTArrayComprehension) -> Optional[ExpressionIR]:
        """Lower array comprehension - visitor pattern"""
        from ..shared.nodes import BinaryExpression, Identifier, Range as ASTRange
        from ..shared.types import BinaryOp
        from ..ir.nodes import ArrayComprehensionIR, RangeIR
        
        location = self._get_source_location(node)
        
        # Lower body expression
        body_ir = node.expr.accept(self)
        if not isinstance(body_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower array comprehension body: expected ExpressionIR, got {type(body_ir).__name__} at {location}"
            )
        
        # Extract variable and range/array from constraints
        # Pattern: [expr | var in start..end, condition] or [expr | var in array, condition]
        # Support multiple iteration variables by creating nested comprehensions
        iteration_variables = []  # List of (variable_name, range_expr, variable_defid)
        constraints_ir = []
        
        if node.constraints:
            for constraint in node.constraints:
                # Check if constraint is "var in range" or "var in array"
                if isinstance(constraint, BinaryExpression) and constraint.operator == BinaryOp.IN:
                    # Left side should be an identifier (the variable)
                    if isinstance(constraint.left, Identifier):
                        variable_name = constraint.left.name
                        
                        # If this variable is already an iteration variable, treat this as a filter
                        # Example: [x | x in array_a, x in array_b] - first is iteration, second is filter
                        existing_var = any(var_name == variable_name for var_name, _, _ in iteration_variables)
                        if existing_var:
                            # This is a filter condition, not another iteration variable
                            constraint_ir = constraint.accept(self)
                            if isinstance(constraint_ir, ExpressionIR):
                                constraints_ir.append(constraint_ir)
                            continue
                        
                        # Get the variable's DefId (attached during name resolution)
                        variable_defid = getattr(constraint.left, 'defid', None)
                        range_expr_for_var = None
                        
                        # Right side can be a Range, Identifier (array), or ArrayLiteral
                        from ..shared.nodes import ArrayLiteral as ASTArrayLiteral
                        
                        if isinstance(constraint.right, ASTRange):
                            start_ir = constraint.right.start.accept(self)
                            end_ir = constraint.right.end.accept(self)
                            if isinstance(start_ir, ExpressionIR) and isinstance(end_ir, ExpressionIR):
                                range_expr_for_var = RangeIR(
                                    start=start_ir,
                                    end=end_ir,
                                    location=self._get_source_location(constraint.right),
                                    inclusive=getattr(constraint.right, 'inclusive', False)
                                )
                        elif isinstance(constraint.right, (Identifier, ASTArrayLiteral)):
                            # Right side is an array - use array expression as iterable so we bind var to each element
                            array_expr_ir = constraint.right.accept(self)
                            if isinstance(array_expr_ir, ExpressionIR):
                                range_expr_for_var = array_expr_ir
                        
                        if range_expr_for_var:
                            # This is an iteration variable
                            iteration_variables.append((variable_name, range_expr_for_var, variable_defid))
                            # Don't add to general constraints
                            continue
                
                # Lower other constraints (e.g., filter conditions)
                constraint_ir = constraint.accept(self)
                if isinstance(constraint_ir, ExpressionIR):
                    constraints_ir.append(constraint_ir)
        
        if not iteration_variables:
            self.tcx.reporter.report_error(
                message="Array comprehension must have an iteration variable and range/array (e.g., `x in 0..5` or `x in arr`)",
                location=location
            )
            return None
        
        # Create a single ArrayComprehensionIR with multiple variables (cartesian product)
        # multiple variables in ONE comprehension = flat array
        # Nested comprehensions (body IS another comprehension) are handled separately
        variables = [var_name for var_name, _, _ in iteration_variables]
        ranges = [range_expr for _, range_expr, _ in iteration_variables]
        variable_defids = [var_defid for _, _, var_defid in iteration_variables]
        
        # Check if body is already a comprehension (true nesting)
        # If so, we'll preserve the nested structure
        is_nested = isinstance(body_ir, ArrayComprehensionIR)
        
        if is_nested:
            if len(variables) != 1:
                self.tcx.reporter.report_error(
                    message="Nested comprehension should have single outer variable",
                    location=location
                )
                return None
            loop_vars = [IdentifierIR(name=variables[0], location=location, defid=variable_defids[0])]
            return ArrayComprehensionIR(
                body=body_ir,
                loop_vars=loop_vars,
                ranges=[ranges[0]],
                constraints=constraints_ir,
                location=location
            )
        else:
            loop_vars = [IdentifierIR(name=var_name, location=location, defid=did) for var_name, did in zip(variables, variable_defids or [None] * len(variables))]
            return ArrayComprehensionIR(
                body=body_ir,
                loop_vars=loop_vars,
                ranges=ranges,
                constraints=constraints_ir,
                location=location
            )
    
    def visit_rectangular_access(self, node: ASTRectangularAccess) -> Optional[RectangularAccessIR]:
        """Lower rectangular array access - visitor pattern"""
        location = self._get_source_location(node)
        array_ir = node.base_expr.accept(self)
        if not isinstance(array_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower rectangular access base expression: expected ExpressionIR, got {type(array_ir).__name__} at {location}"
            )
        indices = []
        for idx in node.indices:
            if idx is None:
                raise ValueError(
                    "Array access indices must not contain None; "
                    "invalid index slot from AST (parser/transformer bug)"
                )
            idx_ir = idx.accept(self)
            if isinstance(idx_ir, ExpressionIR):
                if isinstance(idx_ir, IdentifierIR):
                    idx_ir = IndexVarIR(
                        name=idx_ir.name,
                        location=idx_ir.location,
                        defid=getattr(idx_ir, "defid", None),
                        range_ir=None,
                    )
                indices.append(idx_ir)
        return RectangularAccessIR(
            array=array_ir,
            indices=indices,
            location=location
        )
    
    def visit_jagged_access(self, node: ASTJaggedAccess) -> Optional[JaggedAccessIR]:
        """Lower jagged array access - visitor pattern"""
        location = self._get_source_location(node)
        base_ir = node.base_expr.accept(self)
        if not isinstance(base_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower jagged access base expression: expected ExpressionIR, got {type(base_ir).__name__} at {location}"
            )
        index_chain = []
        for idx in node.index_chain:
            if idx is None:
                raise ValueError(
                    "Jagged access index_chain must not contain None; "
                    "invalid index slot from AST (parser/transformer bug)"
                )
            idx_ir = idx.accept(self)
            if isinstance(idx_ir, ExpressionIR):
                if isinstance(idx_ir, IdentifierIR):
                    idx_ir = IndexVarIR(
                        name=idx_ir.name,
                        location=idx_ir.location,
                        defid=getattr(idx_ir, "defid", None),
                        range_ir=None,
                    )
                index_chain.append(idx_ir)
        return JaggedAccessIR(
            base=base_ir,
            index_chain=index_chain,
            location=location
        )
    
    def visit_tuple_expression(self, node: ASTTupleExpression) -> Optional[ExpressionIR]:
        """Lower tuple expression - visitor pattern"""
        location = self._get_source_location(node)
        elements = []
        for elem in node.elements:
            elem_ir = elem.accept(self)
            if isinstance(elem_ir, ExpressionIR):
                elements.append(elem_ir)
        return TupleExpressionIR(
            elements=elements,
            location=location
        )
    
    def visit_interpolated_string(self, node: ASTInterpolatedString) -> Optional[ExpressionIR]:
        """Lower interpolated string - visitor pattern"""
        location = self._get_source_location(node)
        parts = []
        for part in node.parts:
            if isinstance(part, str):
                parts.append(part)
            else:
                part_ir = part.accept(self)
                if isinstance(part_ir, ExpressionIR):
                    parts.append(part_ir)
        return InterpolatedStringIR(
            parts=parts,
            location=location
        )
    
    def visit_cast_expression(self, node: ASTCastExpression) -> Optional[ExpressionIR]:
        """Lower cast expression - visitor pattern"""
        location = self._get_source_location(node)
        expr_ir = node.expr.accept(self)
        if not isinstance(expr_ir, ExpressionIR):
            return None
        # Extract target_type from AST node
        # Pass type object directly (no string conversion - backend uses type objects)
        target_type = None
        if hasattr(node, 'target_type') and node.target_type:
            target_type = node.target_type
        elif hasattr(node, 'type_name') and node.type_name:
            target_type = node.type_name
        elif hasattr(node, 'type') and node.type:
            # Pass the type object directly (e.g., PrimitiveType)
            target_type = node.type
        return CastExpressionIR(
            expr=expr_ir,
            target_type=target_type,
            location=location
        )
    
    def visit_member_access(self, node: ASTMemberAccess) -> Optional[ExpressionIR]:
        """Lower member access - visitor pattern"""
        location = self._get_source_location(node)
        object_ir = node.object.accept(self)
        if not isinstance(object_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower member access object: expected ExpressionIR, got {type(object_ir).__name__} at {location}"
            )
        # AST MemberAccess uses 'property', not 'member'
        return MemberAccessIR(
            object=object_ir,
            member=node.property,  # AST uses 'property' attribute
            location=location
        )
    
    def visit_method_call(self, node: ASTMethodCall) -> Optional[FunctionCallIR]:
        """Lower method call - visitor pattern (uses FunctionCallIR)"""
        location = self._get_source_location(node)
        # Lower object and method
        object_ir = node.object.accept(self)
        method_ir = node.method_expr.accept(self)
        if not isinstance(object_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower method call object: expected ExpressionIR, got {type(object_ir).__name__} at {location}"
            )
        if not isinstance(method_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower method call method expression: expected ExpressionIR, got {type(method_ir).__name__} at {location}"
            )
        # Lower arguments
        arguments = [object_ir]  # Method calls include object as first argument
        for arg in node.arguments:
            arg_ir = arg.accept(self)
            if isinstance(arg_ir, ExpressionIR):
                arguments.append(arg_ir)
        # Get method name from method_expr if it's an identifier
        method_name = getattr(method_ir, 'name', 'method') if isinstance(method_ir, IdentifierIR) else 'method'
        callee_expr = IdentifierIR(name=method_name, location=location, defid=None)
        return FunctionCallIR(
            callee_expr=callee_expr,
            location=location,
            arguments=arguments,
        )
    
    def visit_pipeline_expression(self, node: ASTPipelineExpression) -> Optional[ExpressionIR]:
        """Lower pipeline expression - visitor pattern"""
        location = self._get_source_location(node)
        left_ir = node.left.accept(self)
        right_ir = node.right.accept(self)
        if not isinstance(left_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower pipeline left expression: expected ExpressionIR, got {type(left_ir).__name__} at {location}"
            )
        if not isinstance(right_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower pipeline right expression: expected ExpressionIR, got {type(right_ir).__name__} at {location}"
            )
        op = getattr(node.operator, 'value', str(node.operator)) if hasattr(node, 'operator') and node.operator else "|>"
        return PipelineExpressionIR(
            left=left_ir,
            right=right_ir,
            location=location,
            operator=op
        )
    
    def visit_try_expression(self, node: ASTTryExpression) -> Optional[ExpressionIR]:
        """Lower try expression - visitor pattern"""
        location = self._get_source_location(node)
        operand_ir = node.operand.accept(self)
        if not isinstance(operand_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower try expression operand: expected ExpressionIR, got {type(operand_ir).__name__} at {location}"
            )
        return TryExpressionIR(
            operand=operand_ir,
            location=location
        )
    
    def visit_match_expression(self, node: ASTMatchExpression) -> Optional[ExpressionIR]:
        """Lower match expression - visitor pattern"""
        location = self._get_source_location(node)
        scrutinee_ir = node.scrutinee.accept(self)
        if not isinstance(scrutinee_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower match scrutinee: expected ExpressionIR, got {type(scrutinee_ir).__name__} at {location}"
            )
        # Lower arms (patterns and expressions)
        arms = []
        for arm in node.arms:
            # Lower pattern
            pattern_ir = arm.pattern.accept(self) if hasattr(arm.pattern, 'accept') else None
            if not isinstance(pattern_ir, PatternIR):
                # Try to lower pattern manually
                pattern_ir = self._lower_pattern(arm.pattern)
            if pattern_ir is None:
                continue
            
            # Lower body expression - MatchArm uses 'body', not 'expr'
            body_ir = arm.body.accept(self)
            if not isinstance(body_ir, ExpressionIR):
                continue
            
            arms.append(MatchArmIR(pattern=pattern_ir, body=body_ir))
        
        return MatchExpressionIR(
            scrutinee=scrutinee_ir,
            arms=arms,
            location=location
        )

    def visit_reduction_expression(self, node: ASTReductionExpression) -> Optional[ExpressionIR]:
        """Lower reduction expression - visitor pattern"""
        location = self._get_source_location(node)
        body_ir = node.body.accept(self)
        if not isinstance(body_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower reduction expression body: expected ExpressionIR, got {type(body_ir).__name__} at {location}"
            )
        
        # Extract loop variables and ranges from over_clause
        loop_vars = []
        loop_var_ranges = {}  # var_name -> RangeIR
        if hasattr(node, 'over_clause') and node.over_clause:
            if hasattr(node.over_clause, 'range_groups'):
                for group in node.over_clause.range_groups:
                    if hasattr(group, 'variables'):
                        # Extract range expression if present
                        range_ir = None
                        if hasattr(group, 'range_expr') and group.range_expr:
                            # The range_expr is an AST Range node, need to lower it
                            from ..shared.nodes import Range as ASTRange
                            if isinstance(group.range_expr, ASTRange):
                                # Lower the Range AST node explicitly
                                range_ir = self.visit_range(group.range_expr)
                            else:
                                # Try accept() as fallback
                                range_ir = group.range_expr.accept(self)
                            
                            if not isinstance(range_ir, RangeIR):
                                range_ir = None
                        
                        # Add variables and their ranges
                        for var_name in group.variables:
                            v = var_name if isinstance(var_name, str) else getattr(var_name, 'value', str(var_name))
                            loop_vars.append(v)
                            if range_ir:
                                loop_var_ranges[v] = range_ir
        
        # Lower where clause if present
        where_clause_ir = None
        if hasattr(node, 'where_clause') and node.where_clause:
            constraints_ir = []
            for constraint in node.where_clause.constraints:
                constraint_ir = constraint.accept(self)
                if isinstance(constraint_ir, ExpressionIR):
                    constraints_ir.append(constraint_ir)
            if constraints_ir:
                where_clause_ir = WhereClauseIR(constraints=constraints_ir)
        
        # Get operation name from function_name (AST uses function_name, IR uses operation)
        operation = getattr(node, 'operation', None)
        if operation is None:
            # Fallback to function_name if operation doesn't exist
            function_name = getattr(node, 'function_name', 'sum')
            if hasattr(function_name, 'value'):
                function_name = function_name.value
            operation = str(function_name).lower()  # Normalize to lowercase
        elif hasattr(operation, 'value'):
            operation = operation.value

        # DefId: copy from AST (name resolution sets _reduction_loop_var_defids and body identifiers' defid); one-to-one in visit_identifier.
        reduction_loop_var_defids = getattr(node, '_reduction_loop_var_defids', None) or {}
        loop_var_idents = [
            IdentifierIR(
                name,
                location,
                defid=reduction_loop_var_defids.get(name) or _defid_of_identifier_in_expr(body_ir, name),
            )
            for name in loop_vars
        ]
        loop_var_ranges_by_defid = {}
        for name, ident in zip(loop_vars, loop_var_idents):
            if getattr(ident, 'defid', None) is not None and name in loop_var_ranges:
                loop_var_ranges_by_defid[ident.defid] = loop_var_ranges[name]
        return ReductionExpressionIR(
            operation=operation,
            loop_vars=loop_var_idents,
            body=body_ir,
            where_clause=where_clause_ir,
            loop_var_ranges=loop_var_ranges_by_defid,
            location=location
        )
    
    def visit_where_expression(self, node: ASTWhereExpression) -> Optional[ExpressionIR]:
        """Lower where expression - visitor pattern"""
        location = self._get_source_location(node)
        expr_ir = node.expr.accept(self)
        if not isinstance(expr_ir, ExpressionIR):
            return None
        # Lower constraints
        constraints_ir = []
        if hasattr(node, 'where_clause') and node.where_clause:
            for constraint in node.where_clause.constraints:
                constraint_ir = constraint.accept(self)
                if isinstance(constraint_ir, ExpressionIR):
                    constraints_ir.append(constraint_ir)
        return WhereExpressionIR(
            expr=expr_ir,
            constraints=constraints_ir,
            location=location
        )
    
    def visit_range(self, node: ASTRange) -> Optional[ExpressionIR]:
        """Lower range expression - visitor pattern"""
        from ..ir.nodes import RangeIR
        
        location = self._get_source_location(node)
        start_ir = node.start.accept(self) if node.start else None
        end_ir = node.end.accept(self) if node.end else None
        
        if not isinstance(start_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower range start: expected ExpressionIR, got {type(start_ir).__name__} at {location}"
            )
        if not isinstance(end_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower range end: expected ExpressionIR, got {type(end_ir).__name__} at {location}"
            )
        
        return RangeIR(
            start=start_ir,
            end=end_ir,
            location=location,
            inclusive=getattr(node, 'inclusive', False),
        )
    
    def visit_index_var(self, node) -> Optional[ExpressionIR]:
        """Lower variable index slot to IndexVarIR. Delegate to range_expr for range."""
        from ..shared.nodes import IndexVar
        from ..ir.nodes import IndexVarIR, RangeIR
        if not isinstance(node, IndexVar):
            return None
        location = self._get_source_location(node)
        range_expr = getattr(node, "range_expr", None)
        range_ir = range_expr.accept(self) if range_expr else None
        if range_ir is not None and not isinstance(range_ir, RangeIR):
            range_ir = None
        # When accept returned None or non-RangeIR, build RangeIR from AST start/end if present
        if range_ir is None and range_expr is not None and hasattr(range_expr, "start") and hasattr(range_expr, "end"):
            start_ir = range_expr.start.accept(self) if range_expr.start else None
            end_ir = range_expr.end.accept(self) if range_expr.end else None
            if isinstance(start_ir, ExpressionIR) and isinstance(end_ir, ExpressionIR):
                range_ir = RangeIR(start=start_ir, end=end_ir, location=location or getattr(range_expr, "location", None))
        return IndexVarIR(
            name=node.name,
            location=location,
            defid=getattr(node, "defid", None),
            range_ir=range_ir,
        )

    def visit_index_rest(self, node) -> Optional[ExpressionIR]:
        """Lower rest index slot to IndexRestIR."""
        from ..shared.nodes import IndexRest
        from ..ir.nodes import IndexRestIR
        if not isinstance(node, IndexRest):
            return None
        defid = getattr(node, "defid", None)
        if defid is None:
            raise ValueError(
                f"IndexRest (..{node.name}) must have defid. "
                "Ensure NameResolutionPass runs before ASTToIRLoweringPass."
            )
        location = self._get_source_location(node)
        return IndexRestIR(name=node.name, location=location, defid=defid)

    # Statement visitors
    def visit_expression_statement(self, node: ASTExpressionStatement) -> Optional[ExpressionIR]:
        """Lower expression statement - visitor pattern"""
        return node.expr.accept(self)  # Just lower the expression
    
    def visit_einstein_declaration(self, node: ASTEinsteinDeclaration) -> Optional[BindingIR]:
        """Lower Einstein declaration to BindingIR with expr=EinsteinIR(clauses=[...])."""
        array_defid = getattr(node, 'defid', None)
        clause_irs: List[IRNode] = []
        for clause in node.clauses:
            one = self._lower_einstein_clause(node.array_name, clause, node, array_defid)
            if one:
                clause_irs.append(one)
        if not clause_irs:
            return None
        loc = self._get_source_location(node)
        einstein_expr = EinsteinIR(clauses=clause_irs, location=loc)
        return BindingIR(
            name=node.array_name,
            expr=einstein_expr,
            location=loc,
            defid=array_defid,
        )

    def _lower_einstein_clause(self, array_name: str, clause, node: ASTEinsteinDeclaration, array_defid: Optional[Any]) -> Optional[IRNode]:
        """Lower one Einstein clause to EinsteinClauseIR."""
        location = self._get_source_location(clause) or self._get_source_location(node)
        value_ir = clause.value.accept(self)
        if not isinstance(value_ir, ExpressionIR):
            raise RuntimeError(
                f"Failed to lower Einstein declaration value: expected ExpressionIR, got {type(value_ir).__name__} at {location}"
            )
        
        # Lower indices: each slot delegates to visit_index_var, visit_index_rest, or visit_literal
        indices_ir = []
        variable_ranges = {}
        from ..ir.nodes import IndexVarIR
        for idx in clause.indices:
            if idx is None:
                raise ValueError(
                    "Einstein clause indices must not contain None; "
                    "invalid index slot from AST (parser/transformer bug)"
                )
            idx_ir = idx.accept(self)
            if isinstance(idx_ir, ExpressionIR):
                if isinstance(idx_ir, IdentifierIR):
                    idx_ir = IndexVarIR(
                        name=idx_ir.name,
                        location=idx_ir.location,
                        defid=getattr(idx_ir, "defid", None),
                        range_ir=None,
                    )
                indices_ir.append(idx_ir)
                if isinstance(idx_ir, IndexVarIR) and idx_ir.range_ir is not None and getattr(idx_ir, "defid", None) is not None:
                    variable_ranges[idx_ir.defid] = idx_ir.range_ir
        
        # Lower where clause if present (from clause or declaration)
        where_clause_ir = None
        where_src = getattr(clause, 'where_clause', None) or getattr(node, 'where_clause', None)
        if where_src and getattr(where_src, 'constraints', None):
            constraints_ir = []
            for constraint in where_src.constraints:
                constraint_ir = constraint.accept(self)
                if isinstance(constraint_ir, ExpressionIR):
                    constraints_ir.append(constraint_ir)
            if constraints_ir:
                where_clause_ir = WhereClauseIR(constraints=constraints_ir)

        return EinsteinClauseIR(
            indices=indices_ir,
            value=value_ir,
            location=location,
            where_clause=where_clause_ir,
            variable_ranges=variable_ranges if variable_ranges else None
        )
    
    def visit_inline_module(self, node: ASTInlineModule) -> Optional[IRNode]:
        """Lower inline module - visitor pattern"""
        location = self._get_source_location(node)
        # Lower body statements
        for stmt in node.body:
            stmt.accept(self)
        # For now, return None - inline modules handled separately
        return None
    
    # Pattern visitors
    def visit_literal_pattern(self, node) -> Optional[PatternIR]:
        """Lower literal pattern - visitor pattern"""
        location = self._get_source_location(node) if hasattr(node, 'location') else self._get_default_location()
        value = node.value if hasattr(node, 'value') else None
        # If value is a Literal AST node, extract its value
        if value is not None and hasattr(value, 'value'):
            value = value.value
        return LiteralPatternIR(value=value, location=location)
    
    def visit_identifier_pattern(self, node) -> Optional[PatternIR]:
        """Lower identifier pattern - visitor pattern"""
        location = self._get_source_location(node) if hasattr(node, 'location') else self._get_default_location()
        name = node.name if hasattr(node, 'name') else None
        if name is None:
            raise RuntimeError(
                f"Failed to lower identifier pattern: pattern has no name attribute at {location}"
            )
        # Transfer DefId from AST to IR (allocated during name resolution)
        defid = getattr(node, 'defid', None)
        pattern = IdentifierPatternIR(name=name, location=location, defid=defid)
        if defid:
            logger.debug(f"[ast_to_ir] Transferred DefId {defid} from AST identifier pattern {name} to IR")
        return pattern
    
    def visit_wildcard_pattern(self, node) -> Optional[PatternIR]:
        """Lower wildcard pattern - visitor pattern"""
        location = self._get_source_location(node) if hasattr(node, 'location') else self._get_default_location()
        return WildcardPatternIR(location=location)
    
    def visit_tuple_pattern(self, node) -> Optional[PatternIR]:
        """Lower tuple pattern - visitor pattern"""
        location = self._get_source_location(node) if hasattr(node, 'location') else self._get_default_location()
        patterns = []
        if hasattr(node, 'patterns'):
            for pattern in node.patterns:
                pattern_ir = pattern.accept(self) if hasattr(pattern, 'accept') else self._lower_pattern(pattern)
                if isinstance(pattern_ir, PatternIR):
                    patterns.append(pattern_ir)
        return TuplePatternIR(patterns=patterns, location=location)
    
    def visit_array_pattern(self, node) -> Optional[PatternIR]:
        """Lower array pattern - visitor pattern"""
        location = self._get_source_location(node) if hasattr(node, 'location') else self._get_default_location()
        patterns = []
        if hasattr(node, 'patterns'):
            for pattern in node.patterns:
                pattern_ir = pattern.accept(self) if hasattr(pattern, 'accept') else self._lower_pattern(pattern)
                if isinstance(pattern_ir, PatternIR):
                    patterns.append(pattern_ir)
        return ArrayPatternIR(patterns=patterns, location=location)
    
    def visit_rest_pattern(self, node) -> Optional[PatternIR]:
        """Lower rest pattern - visitor pattern"""
        location = self._get_source_location(node) if hasattr(node, 'location') else self._get_default_location()
        # Rest pattern has an inner pattern (usually IdentifierPatternIR)
        inner_pattern = None
        if hasattr(node, 'pattern'):
            inner_pattern_ir = node.pattern.accept(self) if hasattr(node.pattern, 'accept') else self._lower_pattern(node.pattern)
            if isinstance(inner_pattern_ir, IdentifierPatternIR):
                inner_pattern = inner_pattern_ir
        if inner_pattern is None:
            # Create default identifier pattern
            inner_pattern = IdentifierPatternIR(name='rest', location=location)
        return RestPatternIR(pattern=inner_pattern, location=location)
    
    def visit_guard_pattern(self, node) -> Optional[PatternIR]:
        """Lower guard pattern - visitor pattern"""
        location = self._get_source_location(node) if hasattr(node, 'location') else self._get_default_location()
        # Guard pattern has inner pattern and guard expression
        inner_pattern = None
        if hasattr(node, 'pattern'):
            inner_pattern_ir = node.pattern.accept(self) if hasattr(node.pattern, 'accept') else self._lower_pattern(node.pattern)
            if isinstance(inner_pattern_ir, PatternIR):
                inner_pattern = inner_pattern_ir
        if inner_pattern is None:
            raise RuntimeError(
                f"Failed to lower guard pattern: inner pattern lowering failed at {location}"
            )
        
        guard_expr = None
        if hasattr(node, 'guard') and node.guard is not None:
            guard_expr_ir = node.guard.accept(self) if hasattr(node.guard, 'accept') else None
            if isinstance(guard_expr_ir, ExpressionIR):
                guard_expr = guard_expr_ir
        if guard_expr is None:
            raise RuntimeError(
                f"Failed to lower guard pattern: guard expression lowering failed at {location}"
            )
        
        return GuardPatternIR(inner_pattern=inner_pattern, guard_expr=guard_expr, location=location)
    
    def visit_or_pattern(self, node) -> Optional[PatternIR]:
        """Lower or pattern: pat1 | pat2 | ..."""
        location = self._get_source_location(node) if hasattr(node, 'location') else self._get_default_location()
        alternatives = []
        if hasattr(node, 'alternatives'):
            for alt in node.alternatives:
                alt_ir = alt.accept(self) if hasattr(alt, 'accept') else self._lower_pattern(alt)
                if isinstance(alt_ir, PatternIR):
                    alternatives.append(alt_ir)
        if len(alternatives) < 2:
            raise RuntimeError(f"Or pattern requires at least 2 alternatives at {location}")
        return OrPatternIR(alternatives=alternatives, location=location)
    
    def visit_constructor_pattern(self, node) -> Optional[PatternIR]:
        """Lower constructor pattern: Some(x), Circle(r)"""
        location = self._get_source_location(node) if hasattr(node, 'location') else self._get_default_location()
        patterns = []
        if hasattr(node, 'patterns'):
            for p in node.patterns:
                p_ir = p.accept(self) if hasattr(p, 'accept') else self._lower_pattern(p)
                if isinstance(p_ir, PatternIR):
                    patterns.append(p_ir)
        is_struct = getattr(node, 'is_struct_literal', False)
        return ConstructorPatternIR(
            constructor_name=node.constructor_name,
            patterns=patterns,
            is_struct_literal=is_struct,
            location=location,
        )
    
    def visit_binding_pattern(self, node) -> Optional[PatternIR]:
        """Lower binding pattern: name @ pattern"""
        location = self._get_source_location(node) if hasattr(node, 'location') else self._get_default_location()
        inner_ir = node.pattern.accept(self) if hasattr(node.pattern, 'accept') else self._lower_pattern(node.pattern)
        if not isinstance(inner_ir, PatternIR):
            raise RuntimeError(f"Failed to lower binding pattern inner pattern at {location}")
        defid = getattr(node, 'defid', None)
        ident_pat = IdentifierPatternIR(name=node.name, location=location, defid=defid)
        return BindingPatternIR(identifier_pattern=ident_pat, inner_pattern=inner_ir, location=location)
    
    def visit_range_pattern(self, node) -> Optional[PatternIR]:
        """Lower range pattern: start..end or start..=end"""
        location = self._get_source_location(node) if hasattr(node, 'location') else self._get_default_location()
        start_val = node.start.value if hasattr(node.start, 'value') else node.start
        end_val = node.end.value if hasattr(node.end, 'value') else node.end
        return RangePatternIR(start=start_val, end=end_val, inclusive=node.inclusive, location=location)
    
    def _lower_pattern(self, pattern_node) -> Optional[PatternIR]:
        """Helper to lower pattern node (fallback if accept() not available)"""
        if hasattr(pattern_node, 'node_type'):
            node_type = pattern_node.node_type
            if node_type == 'literal_pattern':
                return self.visit_literal_pattern(pattern_node)
            elif node_type == 'identifier_pattern':
                return self.visit_identifier_pattern(pattern_node)
            elif node_type == 'wildcard_pattern':
                return self.visit_wildcard_pattern(pattern_node)
            elif node_type == 'tuple_pattern':
                return self.visit_tuple_pattern(pattern_node)
            elif node_type == 'array_pattern':
                return self.visit_array_pattern(pattern_node)
            elif node_type == 'rest_pattern':
                return self.visit_rest_pattern(pattern_node)
            elif node_type == 'guard_pattern':
                return self.visit_guard_pattern(pattern_node)
            elif node_type == 'or_pattern':
                return self.visit_or_pattern(pattern_node)
            elif node_type == 'constructor_pattern':
                return self.visit_constructor_pattern(pattern_node)
            elif node_type == 'binding_pattern':
                return self.visit_binding_pattern(pattern_node)
            elif node_type == 'range_pattern':
                return self.visit_range_pattern(pattern_node)
        location = self._get_source_location(pattern_node) if hasattr(pattern_node, 'location') else self._get_default_location()
        raise RuntimeError(
            f"Unknown pattern type: {getattr(pattern_node, 'node_type', '?')} at {location}"
        )
    
    def _get_default_location(self) -> SourceLocation:
        """Get default source location"""
        default_file = list(self.tcx.source_files.keys())[0] if self.tcx.source_files else "unknown"
        return SourceLocation(
            file=default_file,
            line=1,
            column=1,
            end_line=1,
            end_column=1
        )
    
    # Required visitor methods (ASTVisitor interface)
    def visit_program(self, node: ASTProgram) -> Optional[IRNode]:
        """Visit program - handled by lower_program"""
        return None
    
    def visit_variable_declaration(self, node) -> Optional[Union[BindingIR, List[BindingIR]]]:
        """
        Lower variable declaration to BindingIR.
        For tuple destructuring returns list of BindingIR.
        """
        from ..shared.nodes import TupleDestructurePattern
        from ..ir.nodes import TupleAccessIR, IdentifierIR

        value_ir = node.value.accept(self) if node.value else None
        if not isinstance(value_ir, ExpressionIR):
            return None

        location = self._get_source_location(node)

        if isinstance(node.pattern, TupleDestructurePattern):
            statements: List[BindingIR] = []
            if not hasattr(self, '_tuple_tmp_counter'):
                self._tuple_tmp_counter = 0
            self._tuple_tmp_counter += 1
            temp_name = f"__tuple_tmp_{self._tuple_tmp_counter}"
            temp_defid = None
            if self.tcx.resolver:
                temp_defid = self.tcx.resolver.allocate_for_local()

            statements.append(BindingIR(
                name=temp_name,
                expr=value_ir,
                location=location,
                defid=temp_defid,
            ))

            for index, annotated_var in enumerate(node.pattern.variables):
                var_name = annotated_var.name
                var_defid = getattr(annotated_var, 'defid', None)
                temp_identifier = IdentifierIR(
                    name=temp_name,
                    location=location,
                    defid=temp_defid
                )
                tuple_access = TupleAccessIR(
                    tuple_expr=temp_identifier,
                    index=index,
                    location=location
                )
                statements.append(BindingIR(
                    name=var_name,
                    expr=tuple_access,
                    type_info=annotated_var.type_annotation,
                    location=location,
                    defid=var_defid,
                ))
            return statements
        defid = getattr(node, 'defid', None)
        return BindingIR(
            name=node.name,
            expr=value_ir,
            type_info=node.type_annotation,
            location=location,
            defid=defid,
        )
    
    def visit_use_statement(self, node) -> Optional[IRNode]:
        """Use statements not lowered to IR (handled in name resolution)"""
        return None
    
    def visit_module_declaration(self, node) -> Optional[IRNode]:
        """Module declarations not lowered to IR (handled separately)"""
        return None
    
    def visit_module_access(self, node) -> Optional[ExpressionIR]:
        """
        Lower standalone module access expression (e.g., math::pi, np::e).
        
        For property access (constants), returns FunctionCallIR with zero arguments.
        This allows properties to be accessed as zero-arg function calls.
        
        Pattern from visit_module_access returns FunctionCallIR for properties.
        """
        from ..shared.nodes import ModuleAccess, Identifier
        from ..ir.nodes import FunctionCallIR
        
        location = self._get_source_location(node)
        
        # Extract module path and property name
        path_parts = []
        current = node
        while isinstance(current, ModuleAccess):
            path_parts.insert(0, current.property)
            current = current.object
        
        if isinstance(current, Identifier):
            path_parts.insert(0, current.name)
        
        if len(path_parts) < 2:
            raise RuntimeError(
                f"Invalid module access: path must have at least 2 components (module and property) at {location}"
            )
        
        # Module path is all but last component, property name is last
        module_path = tuple(path_parts[:-1])
        property_name = path_parts[-1]
        
        # Resolve alias if the first component is an alias (e.g., 'math' -> 'std::math' or 'python::math')
        if self.tcx.resolver and module_path:
            alias_info = self.tcx.resolver.lookup_alias(module_path[0])
            if alias_info:
                # Resolve alias: ('math',) -> ('std', 'math') or ('python', 'math')
                resolved_module_path = alias_info
                if len(module_path) > 1:
                    module_path = resolved_module_path + module_path[1:]
                else:
                    module_path = resolved_module_path
        
        # CRITICAL: Never treat python:: paths as stdlib functions
        # python:: paths should always be Python module calls
        function_defid = None
        if module_path and len(module_path) > 0 and module_path[0] == 'python':
            
            # This is a Python module call - preserve module_path, don't try to resolve as stdlib
            # function_defid should remain None for Python module calls
            pass
        elif module_path and len(module_path) > 0 and module_path[0] == 'std':
            # This is an Einlang stdlib function - try to get DefId
            # First check if name resolution already set a defid on the node
            if hasattr(node, 'defid') and node.defid is not None:
                function_defid = node.defid
                module_path = None  # Clear module_path to indicate stdlib function
            elif self.tcx.resolver:
                # Try to get DefId from resolver (may work if module was loaded)
                function_defid = self.tcx.resolver.get_defid(module_path, property_name, DefType.FUNCTION)
                if function_defid:
                    # Found stdlib function - use it instead of Python module call
                    module_path = None  # Clear module_path to indicate stdlib function
                else:
                    # Cannot resolve stdlib function - this is an error
                    raise RuntimeError(
                        f"Cannot resolve stdlib function {module_path}::{property_name}. "
                        f"The stdlib module may not be loaded or the function may not exist at {location}"
                    )
        
        # For property access, create FunctionCallIR with zero arguments
        # This will be handled by the backend as a property lookup or stdlib function call
        
        callee_expr = IdentifierIR(name=property_name, location=location, defid=function_defid)
        return FunctionCallIR(
            callee_expr=callee_expr,
            location=location,
            arguments=[],
            module_path=module_path,
        )
