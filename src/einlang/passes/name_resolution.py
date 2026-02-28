"""
Name Resolution Pass

Rust Pattern: rustc_resolve::Resolver
Reference: NAME_RESOLUTION_DESIGN.md
"""

import logging
from typing import Union, Dict, Tuple, Optional, Any, List
from contextlib import contextmanager
from ..passes.base import BasePass, TyCtxt
from ..ir.nodes import ProgramIR, BindingIR
from ..shared.defid import DefType, Resolver, DefId, FIXED_BUILTIN_ORDER, fixed_builtin_defid
from ..shared.scope import ScopeManager, ScopeKind, Binding, BindingType, ScopeRedefinitionError
from ..analysis.module_system.path_resolver import MODULE_SEPARATOR
from ..analysis.module_system.module_loader import _read_file_cached
from ..utils.io_utils import is_temp_path
import sys
import os
from pathlib import Path

logger = logging.getLogger("einlang.passes.name_resolution")

# -----------------------------------------------------------------------------
# Scope define validation (call-site semantics)
# -----------------------------------------------------------------------------
# _define_in_scope() only performs scope.define(); each call site enforces rules.
#
# USE/IMPORT (visit_use_statement: wildcard, function import, module alias):
#   - Duplicate import (error): same name already in scope from a *different*
#     module path. Example: use std::math::abs; use std::array::abs;  → error.
#   - Same path or same module tree (allow): re-import from same path or
#     submodule is idempotent. Example: use std::math::{abs, exp}; then
#     use std::math::sign in same scope; or pub use basic::* then special
#     does use std::math::{abs, exp} (abs from basic, re-import from math).
#   - Use after local (error): name already defined as local/item. Example:
#     let x = 1; use foo::x;  → error.
#
# FUNCTION/CONST (pre-pass, _resolve_function_ast, _resolve_constant_ast):
#   - Redefinition (error): same name in same scope and existing is not an
#     import. Example: fn f() {} fn f() {}  → error.
#   - Item shadows import (allow): existing is import, new is function/const.
#     Example: use std::math::pi; const pi = 3;  → OK (local shadows).
#
# VARIABLE/LET (visit_variable_declaration, _resolve_variable_declaration_ast):
#   - Rust semantics: `let` always shadows previous bindings in the same scope.
#     Example: let x = 1; let x = x + 1;  → OK (x is 2, RHS sees old x).
#   - RHS is resolved BEFORE the new binding is introduced (Rust scoping).
#
# PARAMETER (function/lambda params):
#   - Duplicate parameter (error): same param name twice. Example: fn f(x, x) {}
# -----------------------------------------------------------------------------


def _define_in_scope(scope, name: str, binding: Binding, node=None, reporter=None) -> bool:
    """Define name in scope (overwrites if present). Call sites must validate semantics before calling."""
    scope.define(name, binding)
    return True


def _group_einstein_declarations_on_ast(ast: "ASTProgram") -> None:
    """Merge consecutive EinsteinDeclaration (same array_name) into one with multiple clauses."""
    _EinsteinGroupReplacementVisitor().visit_program(ast)


class _EinsteinGroupReplacementVisitor:
    """Visitor that merges EinsteinDeclaration groups in each statement list."""

    def visit_program(self, node) -> None:
        _replace_einstein_groups_with_blocks(node.statements)
        for stmt in node.statements:
            stmt.accept(self)

    def visit_function_definition(self, node) -> None:
        if node.body:
            node.body.accept(self)

    def visit_block_expression(self, node) -> None:
        _replace_einstein_groups_with_blocks(node.statements)
        for stmt in node.statements:
            stmt.accept(self)
        if node.final_expr:
            node.final_expr.accept(self)

    def visit_if_expression(self, node) -> None:
        node.condition.accept(self)
        if node.then_block:
            node.then_block.accept(self)
        if node.else_block:
            node.else_block.accept(self)

    def visit_expression_statement(self, node) -> None:
        node.expr.accept(self)

    def visit_einstein_declaration(self, node) -> None:
        for clause in node.clauses:
            clause.accept(self)

    def visit_variable_declaration(self, node) -> None:
        if node.value:
            node.value.accept(self)

    def visit_use_statement(self, node) -> None:
        pass

    def visit_module_declaration(self, node) -> None:
        pass

    def visit_inline_module(self, node) -> None:
        for stmt in node.body:
            stmt.accept(self)

    def visit_literal(self, node) -> None:
        pass

    def visit_identifier(self, node) -> None:
        pass

    def visit_index_var(self, node) -> None:
        if getattr(node, "range_expr", None) is not None:
            node.range_expr.accept(self)

    def visit_index_rest(self, node) -> None:
        pass

    def visit_enum_definition(self, node) -> None:
        pass

    def visit_struct_definition(self, node) -> None:
        pass

    # Forward remaining expression/pattern visits
    def visit_binary_expression(self, node) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_unary_expression(self, node) -> None:
        node.operand.accept(self)

    def visit_function_call(self, node) -> None:
        node.function_expr.accept(self)
        for arg in node.arguments:
            arg.accept(self)

    def visit_array_literal(self, node) -> None:
        for elem in node.elements:
            elem.accept(self)

    def visit_rectangular_access(self, node) -> None:
        node.base_expr.accept(self)
        for idx in (node.indices or []):
            if idx is not None and hasattr(idx, "accept"):
                idx.accept(self)

    def visit_jagged_access(self, node) -> None:
        node.base_expr.accept(self)
        for idx in (getattr(node, "index_chain", None) or []):
            if idx is not None and hasattr(idx, "accept"):
                idx.accept(self)

    def visit_array_comprehension(self, node) -> None:
        node.expr.accept(self)
        for c in node.constraints or []:
            c.accept(self)

    def visit_reduction_expression(self, node) -> None:
        node.body.accept(self)
        if node.where_clause:
            for c in node.where_clause.constraints:
                c.accept(self)

    def visit_where_expression(self, node) -> None:
        node.expr.accept(self)
        for c in node.where_clause.constraints:
            c.accept(self)

    def visit_lambda_expression(self, node) -> None:
        node.body.accept(self)

    def visit_cast_expression(self, node) -> None:
        node.expr.accept(self)

    def visit_member_access(self, node) -> None:
        node.object.accept(self)

    def visit_module_access(self, node) -> None:
        node.object.accept(self)

    def visit_method_call(self, node) -> None:
        node.object.accept(self)
        node.method_expr.accept(self)
        for arg in node.arguments:
            arg.accept(self)

    def visit_tuple_expression(self, node) -> None:
        for elem in node.elements:
            elem.accept(self)

    def visit_pipeline_expression(self, node) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_try_expression(self, node) -> None:
        node.operand.accept(self)

    def visit_interpolated_string(self, node) -> None:
        for part in node.parts:
            part.accept(self)

    def visit_range(self, node) -> None:
        if node.start:
            node.start.accept(self)
        if node.end:
            node.end.accept(self)

    def visit_arrow_expression(self, node) -> None:
        for c in node.components:
            c.accept(self)

    def visit_match_expression(self, node) -> None:
        node.scrutinee.accept(self)
        for arm in node.arms:
            arm.pattern.accept(self)
            arm.body.accept(self)

    def visit_literal_pattern(self, node) -> None:
        pass

    def visit_identifier_pattern(self, node) -> None:
        pass

    def visit_wildcard_pattern(self, node) -> None:
        pass

    def visit_tuple_pattern(self, node) -> None:
        for p in node.patterns:
            p.accept(self)

    def visit_array_pattern(self, node) -> None:
        for p in node.patterns:
            p.accept(self)

    def visit_rest_pattern(self, node) -> None:
        node.pattern.accept(self)

    def visit_guard_pattern(self, node) -> None:
        node.pattern.accept(self)
        node.guard.accept(self)

    def visit_or_pattern(self, node) -> None:
        for alt in node.alternatives:
            alt.accept(self)

    def visit_binding_pattern(self, node) -> None:
        node.pattern.accept(self)

    def visit_range_pattern(self, node) -> None:
        pass

    def visit_constructor_pattern(self, node) -> None:
        for p in node.patterns:
            p.accept(self)


def _replace_einstein_groups_with_blocks(statements: list) -> None:
    """Merge consecutive EinsteinDeclaration groups (same array_name) into one with multiple clauses."""
    from ..shared.nodes import EinsteinDeclaration as ASTEinsteinDeclaration, EinsteinClause
    new_list: list = []
    i = 0
    while i < len(statements):
        stmt = statements[i]
        if not isinstance(stmt, ASTEinsteinDeclaration):
            new_list.append(stmt)
            i += 1
            continue
        # Collect consecutive Einstein decls for the same array_name
        run: list = [stmt]
        j = i + 1
        while j < len(statements):
            s = statements[j]
            if not isinstance(s, ASTEinsteinDeclaration) or s.array_name != stmt.array_name:
                break
            run.append(s)
            j += 1
        if len(run) < 2:
            new_list.append(stmt)
        else:
            run.sort(key=lambda d: (d.location.line if d.location else 0, d.location.column if d.location else 0))
            clauses = [
                EinsteinClause(d.indices, d.value, d.where_clause, d.location)
                for d in run
            ]
            loc = run[0].location if run[0].location else None
            new_list.append(ASTEinsteinDeclaration(stmt.array_name, clauses, loc))
        i = j
    statements.clear()
    statements.extend(new_list)


# Cache path discovery (stdlib_root, crate_root) per root_path to avoid repeated I/O (exists/is_dir) across compilations
_path_discovery_cache: Dict[Tuple[Path, ...], Tuple[Optional[Path], Optional[Path]]] = {}
_MAX_PATH_CACHE_SIZE = 32


def clear_path_discovery_cache() -> None:
    """Clear the path discovery cache. Used by tests for parallel-safe runs (-n auto)."""
    _path_discovery_cache.clear()


def _discover_stdlib_and_crate_root(root_path: Path, stdlib_root_from_system: Optional[Path] = None) -> Tuple[Path, Path]:
    """Find stdlib and crate root; used when cache misses."""
    current = root_path.resolve()
    stdlib_root = stdlib_root_from_system
    if stdlib_root is None:
        for _ in range(5):
            sp = current / "stdlib"
            if sp.exists() and sp.is_dir():
                stdlib_root = sp
                break
            parent = current.parent
            if parent == current:
                break
            current = parent
        if stdlib_root is None:
            stdlib_root = Path("stdlib")
    current = root_path.resolve()
    crate_root = None
    for _ in range(5):
        demos = current / "examples" / "demos"
        examples = current / "examples"
        stdlib_p = current / "stdlib"
        if demos.exists() and demos.is_dir():
            crate_root = demos
            break
        if examples.exists() and examples.is_dir():
            crate_root = examples
            break
        if stdlib_p.exists() and stdlib_p.is_dir():
            crate_root = (current / "examples" / "demos") if (current / "examples" / "demos").exists() else current
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    if crate_root is None:
        crate_root = root_path
    return (stdlib_root, crate_root)

# Import AST nodes for name resolution on AST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..shared.nodes import (
    Program as ASTProgram, 
    FunctionDefinition,
    Identifier as ASTIdentifier,
    FunctionCall as ASTFunctionCall,
    VariableDeclaration,
    UseStatement
)
from ..shared.ast_visitor import ASTVisitor
# Note: ConstantDef may not exist in AST - constants might be handled differently
try:
    from ..shared.nodes import ConstantDef
except ImportError:
    ConstantDef = None  # Constants handled differently

class NameResolutionPass(BasePass):
    """
    Name resolution pass (Rust naming: rustc_resolve).
    
    Rust Pattern: rustc_resolve::Resolver
    
    Implementation Alignment: Follows Rust's name resolution:
    - DefId allocation (ONLY place)
    - Symbol table creation
    - Module resolution
    - All definitions get DefIds
    
    Reference: `rustc_resolve::Resolver` for name resolution
    
    CRITICAL: This is the ONLY place where DefIds are allocated!
    """
    requires = []  # No dependencies (first pass after parsing)
    
    def run(self, ast: ASTProgram, tcx: TyCtxt) -> ASTProgram:
        """
        Resolve names and allocate DefIds.
        
        Rust Pattern: rustc_resolve::Resolver::resolve_crate()
        
        This pass:
        1. Allocates DefIds for all definitions (via Resolver)
        2. Resolves all names to DefIds
        3. Attaches DefIds to IR nodes
        4. Builds symbol table
        """
        # Each compilation should have a fresh resolver
        # Since each compilation creates a new TyCtxt, tcx.resolver should be a fresh Resolver instance
        # But to be absolutely safe, clear its state to prevent any leaks
        resolver = tcx.resolver
        tcx.def_registry.clear()
        tcx.symbol_table.clear()
        tcx.alias_table.clear()
        resolver._builtin_next_index = 0
        resolver._local_next_index = 0
        
        from ..backends.numpy import (
            builtin_assert, builtin_print, builtin_len, builtin_typeof, builtin_array_append,
            builtin_shape, builtin_sum, builtin_max, builtin_min,
        )
        _builtin_impls = (
            builtin_assert, builtin_print, builtin_len, builtin_typeof, builtin_array_append,
            builtin_shape, builtin_sum, builtin_max, builtin_min,
        )
        for name, definition in zip(FIXED_BUILTIN_ORDER, _builtin_impls):
            resolver.allocate_for_item((), name, definition, DefType.BUILTIN)

        try:
            scope_manager = ScopeManager()
            
            # Initialize new module system components
            from ..analysis.module_system import PathResolver, ModuleLoader, SymbolLinker
            from ..analysis.module_system.path_resolver import MODULE_SEPARATOR
            from pathlib import Path
            
            # Path discovery: cache to avoid repeated exists()/is_dir() I/O across compilations
            root_path = tcx.module_system.root_path if (hasattr(tcx, 'module_system') and tcx.module_system) else Path.cwd()
            root_key = (root_path.resolve(),)
            stdlib_from_system = getattr(tcx.module_system, 'stdlib_root', None) if (hasattr(tcx, 'module_system') and tcx.module_system) else None
            if root_key in _path_discovery_cache:
                cached_stdlib, cached_crate = _path_discovery_cache[root_key]
                path_resolver = PathResolver(stdlib_root=cached_stdlib)
                path_resolver.set_crate_root(cached_crate)
            else:
                stdlib_root, crate_root = _discover_stdlib_and_crate_root(root_path, stdlib_from_system)
                path_resolver = PathResolver(stdlib_root=stdlib_root)
                path_resolver.set_crate_root(crate_root)
                if len(_path_discovery_cache) < _MAX_PATH_CACHE_SIZE:
                    _path_discovery_cache[root_key] = (stdlib_root, crate_root)
            if hasattr(tcx, 'discovered_modules') and tcx.discovered_modules:
                for _mp, file_path in tcx.discovered_modules.items():
                    if isinstance(file_path, Path) and is_temp_path(file_path):
                        if file_path.parent.exists():
                            path_resolver.set_crate_root(file_path.parent)
                        break
            
            # Initialize ModuleLoader (source_overlay avoids I/O when provided)
            source_overlay = getattr(tcx, "source_overlay", None) or {}
            module_loader = ModuleLoader(path_resolver, source_overlay=source_overlay)
            # Clear stateful caches at start of each compilation
            # Only stateless, time-consuming operations (like parser grammar cache) should persist
            # ModuleInfo contains AST nodes with DefIds, which are stateful and must be fresh
            module_loader.loaded_modules.clear()
            
            # Initialize SymbolLinker
            symbol_linker = SymbolLinker()
            
            # Store in tcx for access by visitor
            tcx.path_resolver = path_resolver
            tcx.module_loader = module_loader
            tcx.symbol_linker = symbol_linker
            
            # Rust pattern: NO eager discovery
            # Modules are loaded on-demand when referenced via `use` statements
            # The module_loader will resolve paths and load modules lazily
            # This is true tree-shaking - only load what's actually used

            with scope_manager.scope(ScopeKind.MODULE) as module_scope:
                # Name resolution operates on AST only (IR is produced after lowering)
                if not isinstance(ast, ASTProgram):
                    raise TypeError(
                        f"NameResolutionPass requires ASTProgram, got {type(ast).__name__}. "
                        "Name resolution runs on AST before IR lowering."
                    )

                # Einstein grouping (part of name resolution): group decls by (scope, name) for Fibonacci pattern
                _group_einstein_declarations_on_ast(ast)

                # Pre-pass: register all top-level functions so mutual recursion resolves (is_even/is_odd etc.)
                from ..shared.nodes import FunctionDefinition, VariableDeclaration
                for stmt in ast.statements:
                    if isinstance(stmt, FunctionDefinition):
                        if module_scope.defined_in_this_scope(stmt.name):
                            loc = getattr(stmt, "location", None)
                            tcx.reporter.report_error(f"redefinition of '{stmt.name}' in same scope", loc)
                            continue
                        defid = resolver.allocate_for_item((), stmt.name, stmt, DefType.FUNCTION)
                        object.__setattr__(stmt, "defid", defid)
                        _define_in_scope(
                            module_scope,
                            stmt.name,
                            Binding(
                                name=stmt.name,
                                binding_type=BindingType.FUNCTION,
                                definition=stmt,
                                defid=defid,
                                scope=module_scope,
                            ),
                            stmt,
                            tcx.reporter,
                        )

                # Single visitor: resolves uses and function bodies (DefIds already set by pre-pass for top-level functions)
                resolver_visitor = NameResolverVisitor(resolver, scope_manager, tcx)
                resolver_visitor.path_resolver = tcx.path_resolver
                resolver_visitor.module_loader = tcx.module_loader
                resolver_visitor.symbol_linker = tcx.symbol_linker
                resolver_visitor._parsed_modules.clear()
                for stmt in ast.statements:
                    stmt.accept(resolver_visitor)

                result = ast  # AST with DefIds attached

            return result
        finally:
            pass
    
    def _resolve_function(
        self,
        func: BindingIR,
        resolver: Resolver,
        scope_manager: ScopeManager,
        tcx: TyCtxt,
        module_path: tuple
    ) -> None:
        """Resolve function definition and allocate DefId (IR version)"""
        defid = resolver.allocate_for_item(module_path, func.name, func, DefType.FUNCTION)
        
        # Add to scope
        scope = scope_manager.current_scope()
        if scope:
            if scope.defined_in_this_scope(func.name):
                existing = scope._get_binding_in_this_scope(func.name)
                if getattr(existing, "module_path", None) is None:
                    loc = getattr(func, "location", None)
                    tcx.reporter.report_error(f"redefinition of '{func.name}' in same scope", loc)
                    object.__setattr__(func, 'defid', defid)
                    return
            binding = Binding(
                name=func.name,
                binding_type=BindingType.FUNCTION,
                definition=func,
                defid=defid,
                scope=scope
            )
            _define_in_scope(scope, func.name, binding, func, tcx.reporter)

        # Attach DefId to function (IR nodes use 'defid' not '_defid')
        # Rust pattern: Trust your infrastructure, no defensive checks
        object.__setattr__(func, 'defid', defid)
        if module_path:
            object.__setattr__(func, 'module_path', module_path)

    def _resolve_function_ast(
        self,
        func: FunctionDefinition,
        resolver: Resolver,
        scope_manager: ScopeManager,
        tcx: TyCtxt,
        module_path: tuple
    ) -> None:
        """Resolve function definition and allocate DefId (AST version)"""
        defid = resolver.allocate_for_item(module_path, func.name, func, DefType.FUNCTION)

        # Add to scope
        scope = scope_manager.current_scope()
        if scope:
            if scope.defined_in_this_scope(func.name):
                existing = scope._get_binding_in_this_scope(func.name)
                if getattr(existing, "module_path", None) is None:
                    loc = getattr(func, "location", None)
                    tcx.reporter.report_error(f"redefinition of '{func.name}' in same scope", loc)
                    object.__setattr__(func, 'defid', defid)
                    return
            binding = Binding(
                name=func.name,
                binding_type=BindingType.FUNCTION,
                definition=func,
                defid=defid,
                scope=scope
            )
            _define_in_scope(scope, func.name, binding, func, tcx.reporter)

        # Attach DefId to function (IR nodes use 'defid' not '_defid')
        # Rust pattern: Trust your infrastructure, no defensive checks
        object.__setattr__(func, 'defid', defid)
        if module_path:
            object.__setattr__(func, 'module_path', module_path)

        # Enter function scope and resolve parameters (allocate_for_local; no def_registry)
        with scope_manager.scope(ScopeKind.FUNCTION) as func_scope:
            for param in func.parameters:
                if func_scope:
                    if func_scope.defined_in_this_scope(param.name):
                        loc = getattr(param, "location", None)
                        tcx.reporter.report_error(f"duplicate parameter '{param.name}'", loc)
                    else:
                        param_defid = resolver.allocate_for_local()
                        object.__setattr__(param, 'defid', param_defid)
                        param_binding = Binding(
                            name=param.name,
                            binding_type=BindingType.PARAMETER,
                            definition=param,
                            defid=param_defid,
                            scope=func_scope
                        )
                        _define_in_scope(func_scope, param.name, param_binding, param, tcx.reporter)
            # Body is walked by NameResolverVisitor when it recurses from visit_function_definition
    
    def _resolve_constant(
        self,
        const: BindingIR,
        resolver: Resolver,
        scope_manager: ScopeManager,
        tcx: TyCtxt,
        module_path: tuple
    ) -> None:
        """Resolve constant definition and allocate DefId (IR version)"""
        defid = resolver.allocate_for_item(module_path, const.name, const, DefType.CONSTANT)

        # Add to scope
        scope = scope_manager.current_scope()
        if scope:
            if scope.defined_in_this_scope(const.name):
                existing = scope._get_binding_in_this_scope(const.name)
                if getattr(existing, "module_path", None) is None:
                    loc = getattr(const, "location", None)
                    tcx.reporter.report_error(f"redefinition of '{const.name}' in same scope", loc)
                    object.__setattr__(const, 'defid', defid)
                    return
            binding = Binding(
                name=const.name,
                binding_type=BindingType.CONSTANT,
                definition=const,
                defid=defid,
                scope=scope
            )
            _define_in_scope(scope, const.name, binding, const, tcx.reporter)
        
        # Attach DefId to constant (IR nodes use 'defid' not '_defid')
        # Rust pattern: Trust your infrastructure, no defensive checks
        object.__setattr__(const, 'defid', defid)
    
    def _resolve_constant_ast(
        self,
        const: ConstantDef,
        resolver: Resolver,
        scope_manager: ScopeManager,
        tcx: TyCtxt,
        module_path: tuple
    ) -> None:
        """Resolve constant definition and allocate DefId (AST version)"""
        defid = resolver.allocate_for_item(module_path, const.name, const, DefType.CONSTANT)

        # Add to scope
        scope = scope_manager.current_scope()
        if scope:
            if scope.defined_in_this_scope(const.name):
                existing = scope._get_binding_in_this_scope(const.name)
                if getattr(existing, "module_path", None) is None:
                    loc = getattr(const, "location", None)
                    tcx.reporter.report_error(f"redefinition of '{const.name}' in same scope", loc)
                    object.__setattr__(const, 'defid', defid)
                    return
            binding = Binding(
                name=const.name,
                binding_type=BindingType.CONSTANT,
                definition=const,
                defid=defid,
                scope=scope
            )
            _define_in_scope(scope, const.name, binding, const, tcx.reporter)
        
        # Attach DefId to constant (IR nodes use 'defid' not '_defid')
        # Rust pattern: Trust your infrastructure, no defensive checks
        object.__setattr__(const, 'defid', defid)

    
    def _resolve_variable_declaration_ast(
        self,
        var_decl: VariableDeclaration,
        resolver: Resolver,
        scope_manager: ScopeManager,
        tcx: TyCtxt,
        module_path: tuple
    ) -> None:
        """Resolve variable declaration (AST version). All vars get DefId."""
        from ..shared.nodes import TupleDestructurePattern

        scope = scope_manager.current_scope()
        if isinstance(var_decl.pattern, TupleDestructurePattern):
            for annotated_var in var_decl.pattern.variables:
                var_name = annotated_var.name
                defid = resolver.allocate_for_local()
                if scope:
                    _define_in_scope(
                        scope, var_name,
                        Binding(name=var_name, binding_type=BindingType.VARIABLE,
                                definition=var_decl, defid=defid, scope=scope),
                        annotated_var, tcx.reporter,
                    )
                object.__setattr__(annotated_var, 'defid', defid)
            var_decl_defid = defid
        else:
            var_name = var_decl.pattern
            defid = resolver.allocate_for_local()
            if scope:
                _define_in_scope(
                    scope, var_name,
                    Binding(name=var_name, binding_type=BindingType.VARIABLE,
                            definition=var_decl, defid=defid, scope=scope),
                    var_decl, tcx.reporter,
                )
            object.__setattr__(var_decl, 'defid', defid)
            var_decl_defid = defid

        object.__setattr__(var_decl, 'defid', var_decl_defid)
        
        # NOTE: Do NOT attach the variable's DefId to identifiers in the value expression!
        # The identifier's DefId will be resolved correctly in Phase 2 (uses resolution)
        # when visit_variable_declaration calls node.value.accept(self), which will
        # call visit_identifier and look up the correct DefId from the scope.


# NOTE: IR-level name resolution classes were removed - they were never used.
# Name resolution should ONLY handle AST (as per design comment on line 151-155).
# The lowering pass (ASTToIRLoweringPass) preserves DefIds from AST to IR.
# If DefIds are missing in IR, it's a lowering bug, not a name resolution bug.

class NameResolverVisitor(ASTVisitor[None]):
    """
    Single visitor for name resolution: allocates at definition sites and resolves uses.
    Rust Pattern: rustc_resolve::Resolver::resolve_name()
    """
    def __init__(self, resolver: Resolver, scope_manager: ScopeManager, tcx: TyCtxt):
        self.resolver = resolver
        self.scope_manager = scope_manager
        self.tcx = tcx
        # Cache parsed modules (within-compilation only, cleared at start of each compilation)
        # Only stateless, time-consuming operations should be cached across compilations
        # This cache is for avoiding re-parsing the same module multiple times within a single compilation
        self._parsed_modules: Dict[Tuple[str, ...], ASTProgram] = {}
        
        # New module system components (initialized in NameResolutionPass.run)
        self.path_resolver = None
        self.module_loader = None
        self.symbol_linker = None
    
        # Current module context for resolving relative use statements
        self.current_module_path: Tuple[str, ...] = ()

    def visit_identifier(self, node: ASTIdentifier) -> None:
        """
        Resolve identifier from scope stack (innermost to outermost). Set defid from binding.
        All resolution failures are reported as errors.
        """
        name = node.name if isinstance(node.name, str) else getattr(node.name, 'value', str(node.name))
        name = str(name)
        if not self.scope_manager:
            loc = getattr(node, "location", None)
            msg = f"name resolution failed: no scope for '{name}'"
            if self.tcx and self.tcx.reporter:
                self.tcx.reporter.report_error(msg, loc)
            raise ValueError(msg)
        binding = self.scope_manager.lookup(name)
        if binding and getattr(binding, "defid", None) is not None:
            object.__setattr__(node, "defid", binding.defid)
        else:
            loc = getattr(node, "location", None)
            msg = f"cannot find value `{name}` in this scope"
            if self.tcx and self.tcx.reporter:
                self.tcx.reporter.report_error(msg, loc, code="E0425", label="not found in this scope")
            raise ValueError(msg)
    
    def visit_function_call(self, node: ASTFunctionCall) -> None:
        """
        Resolve function call to DefId with qualified resolution priority.
        
        Resolution priority:
        1. Qualified module calls (std::math::sin) - PRIORITY 1
        2. Builtins (len, print, assert) - PRIORITY 2
        3. Lambdas (first-class functions) - PRIORITY 3
        4. User-defined functions - PRIORITY 4
        5. Aliased module functions (use statements) - PRIORITY 5
        6. Unqualified module functions (stdlib) - PRIORITY 6
        """
        from ..shared.nodes import ModuleAccess
        from ..shared.defid import DefType
        
        # PRIORITY 1: Qualified module calls (std::math::sin, math::sqrt)
        if isinstance(node.function_expr, ModuleAccess):
            # Module access like std::array::concatenate
            # Resolve the module path and member
            node.function_expr.accept(self)
            
            module_access_defid = getattr(node.function_expr, 'defid', None)
            resolved_path = getattr(node.function_expr, '_resolved_module_path', None) or ()
            is_python = bool(resolved_path and len(resolved_path) > 0 and resolved_path[0] == 'python')
            if module_access_defid:
                object.__setattr__(node, 'function_defid', module_access_defid)
                function_name = getattr(node.function_expr, '_resolved_function_name', None) or node.function_expr.property
                object.__setattr__(node, '_resolved_function_name', function_name)
            elif is_python:
                function_name = getattr(node.function_expr, 'property', None) or '?'
                object.__setattr__(node, '_resolved_function_name', function_name)
            else:
                qual = getattr(node.function_expr, 'property', None) or '?'
                loc = getattr(node.function_expr, "location", None)
                msg = f"qualified name could not be resolved: {qual}"
                if self.tcx and self.tcx.reporter:
                    self.tcx.reporter.report_error(msg, loc)
                raise ValueError(msg)
            for arg in node.arguments:
                arg.accept(self)
            return
        
        # For unqualified calls (Identifier), check in priority order
        # Rust semantics: local scope wins over builtins (builtins = prelude = outermost scope)
        if isinstance(node.function_expr, ASTIdentifier):
            func_name = node.function_expr.name
            
            # PRIORITY 2: Scope lookup (lambdas and user functions). Use full scope chain.
            binding = self.scope_manager.lookup(func_name) if self.scope_manager else None
            if binding:
                # Check if this is an aliased function (from use statement) without DefId yet
                if not binding.defid and hasattr(binding, '_alias_path'):
                    # This is a lazy-loaded function from use statement
                    # Rust pattern: use std::math::abs → alias_path is ('std', 'math', 'abs')
                    # We need to load the parent module and look for the function
                    alias_path = binding._alias_path
                    
                    # Rust pattern: Load the module containing the function
                    # If alias_path is ('std', 'math', 'abs'), the function is either:
                    # 1. In module ('std', 'math') exported via pub use
                    # 2. In module ('std', 'math', 'abs') as a nested module
                    # Try parent module first (most common pattern with pub use)
                    parent_module_path = alias_path[:-1] if len(alias_path) > 1 else alias_path
                    
                    module_defid = self._resolve_function_from_module(parent_module_path, func_name)
                    if not module_defid and len(alias_path) > 1:
                        # Try the full path as a module (less common)
                        module_defid = self._resolve_function_from_module(alias_path, func_name)
                    
                    if module_defid:
                        object.__setattr__(node.function_expr, 'defid', module_defid)
                        object.__setattr__(node, 'function_defid', module_defid)
                        # Resolve arguments and return
                        for arg in node.arguments:
                            arg.accept(self)
                        return
                elif binding.defid:
                    object.__setattr__(node.function_expr, 'defid', binding.defid)
                    object.__setattr__(node, 'function_defid', binding.defid)
                    for arg in node.arguments:
                        arg.accept(self)
                    return
            
            # PRIORITY 5: Aliased module functions (use statements)
            # First try alias lookup (for function imports like "use std::math::abs;")
            alias_path = self.resolver.lookup_alias(func_name)
            if alias_path:
                # Resolve function through alias (e.g., 'abs' → ('std', 'math') for "use std::math::abs;")
                # The function should be registered with key (alias_path, func_name) in symbol table
                module_defid = self.resolver.get_defid(alias_path, func_name, DefType.FUNCTION)
                if module_defid:
                    object.__setattr__(node.function_expr, 'defid', module_defid)
                    object.__setattr__(node, 'function_defid', module_defid)
                    # Resolve arguments and return
                    for arg in node.arguments:
                        arg.accept(self)
                    return
            
            # PRIORITY 5c: When resolving inside a loaded module (not the main program), allow
            # lookup in the current module's symbol table (e.g. pi from constants::* in std::math).
            # Do NOT use get_defid((), ...) - that would bypass use-statement scoping in the main program.
            current_module = getattr(self, "current_module_path", ()) or ()
            if current_module:
                module_defid = self.resolver.get_defid(current_module, func_name, DefType.FUNCTION) or self.resolver.get_defid(current_module, func_name, DefType.CONSTANT)
                if module_defid:
                    object.__setattr__(node.function_expr, "defid", module_defid)
                    object.__setattr__(node, "function_defid", module_defid)
                    for arg in node.arguments:
                        arg.accept(self)
                    return
            
            # PRIORITY 6: Unqualified module functions (stdlib - on-demand loading)
            module_defid = self._resolve_stdlib_function(func_name)
            if module_defid:
                object.__setattr__(node.function_expr, 'defid', module_defid)
                object.__setattr__(node, 'function_defid', module_defid)
                # Resolve arguments and return
                for arg in node.arguments:
                    arg.accept(self)
                return
            
            # PRIORITY 7: Builtins (prelude — outermost scope, Rust semantics)
            if func_name in FIXED_BUILTIN_ORDER:
                builtin_defid = fixed_builtin_defid(func_name)
                if builtin_defid:
                    object.__setattr__(node.function_expr, 'defid', builtin_defid)
                    object.__setattr__(node, 'function_defid', builtin_defid)
                    for arg in node.arguments:
                        arg.accept(self)
                    return
            if self.tcx and self.tcx.reporter:
                loc = getattr(node, "location", None)
                self.tcx.reporter.report_error(
                    f"Undefined function '{func_name}'. "
                    "Use a 'use' statement to import it (e.g. use my_module::{func_name};) or call with module path (e.g. my_module::{func_name}(...)).".format(func_name=func_name),
                    loc,
                )
        
        # Callable expression (e.g. inline lambda (|x| x+1)(5)) - resolve callee then args
        if not isinstance(node.function_expr, (ASTIdentifier, ModuleAccess)):
            node.function_expr.accept(self)
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_function_definition(self, node: FunctionDefinition) -> None:
        """Resolve params and body. DefId/scope define may already be done by main-program pre-pass (mutual recursion)."""
        scope = self.scope_manager.current_scope()
        defid = getattr(node, "defid", None)
        if defid is None:
            defid = self.resolver.allocate_for_item((), node.name, node, DefType.FUNCTION)
            object.__setattr__(node, "defid", defid)
        if scope and not scope.defined_in_this_scope(node.name):
            _define_in_scope(
                scope,
                node.name,
                Binding(
                    name=node.name,
                    binding_type=BindingType.FUNCTION,
                    definition=node,
                    defid=defid,
                    scope=scope,
                ),
                node,
                self.tcx.reporter,
            )
        with self.scope_manager.scope(ScopeKind.FUNCTION) as func_scope:
            for param in node.parameters:
                param_name = param.name if isinstance(param.name, str) else getattr(param.name, 'value', str(param.name))
                param_name = str(param_name)
                if func_scope and func_scope.defined_in_this_scope(param_name):
                    loc = getattr(param, "location", None)
                    if self.tcx and self.tcx.reporter:
                        self.tcx.reporter.report_error(f"duplicate parameter '{param_name}'", loc)
                    raise ValueError(
                        f"duplicate parameter '{param_name}' in function '{node.name}'. "
                        "Duplicate parameters are not allowed."
                    )
                else:
                    param_defid = self.resolver.allocate_for_local()
                    object.__setattr__(param, "defid", param_defid)
                    if func_scope:
                        _define_in_scope(
                            func_scope,
                            param_name,
                            Binding(
                                name=param_name,
                                binding_type=BindingType.PARAMETER,
                                definition=param,
                                defid=param_defid,
                                scope=func_scope
                            ),
                            param,
                            self.tcx.reporter,
                        )
            if node.body:
                node.body.accept(self)
    
    def visit_block_expression(self, node) -> None:
        """Allocate for block-local defs, then resolve statements and final_expr."""
        with self.scope_manager.scope(ScopeKind.BLOCK):
            from ..shared.nodes import EinsteinDeclaration as ASTEinsteinDeclaration
            scope = self.scope_manager.current_scope()
            if scope:
                for stmt in node.statements:
                    if isinstance(stmt, ASTEinsteinDeclaration):
                        name = getattr(stmt, "array_name", None)
                        if name and not scope.defined_in_this_scope(name):
                            defid = self.resolver.allocate_for_local()
                            object.__setattr__(stmt, "defid", defid)
                            _define_in_scope(
                                scope,
                                name,
                                Binding(
                                    name=name,
                                    binding_type=BindingType.VARIABLE,
                                    definition=stmt,
                                    defid=defid,
                                    scope=scope,
                                ),
                                stmt,
                                self.tcx.reporter,
                            )
            for stmt in node.statements:
                stmt.accept(self)
            if node.final_expr:
                node.final_expr.accept(self)
    
    def visit_variable_declaration(self, node: VariableDeclaration) -> None:
        """Resolve value FIRST (Rust: RHS sees pre-binding scope), then define the variable."""
        if node.value:
            node.value.accept(self)
        scope = self.scope_manager.current_scope()
        pattern = node.pattern
        if hasattr(pattern, "variables"):
            for annotated_var in pattern.variables:
                var_name = annotated_var.name
                defid = self.resolver.allocate_for_local()
                object.__setattr__(annotated_var, "defid", defid)
                self._define_variable_in_scope(scope, var_name, node, defid, annotated_var)
            object.__setattr__(node, "defid", defid)  # last defid for node
        else:
            var_name = pattern
            defid = self.resolver.allocate_for_local()
            object.__setattr__(node, "defid", defid)
            self._define_variable_in_scope(scope, var_name, node, defid, node)

    def _define_variable_in_scope(self, scope, var_name, node, defid, loc_node):
        """Rust semantics: `let` always shadows previous bindings in the same scope."""
        if not scope:
            return
        binding = Binding(
            name=var_name,
            binding_type=BindingType.VARIABLE,
            definition=node,
            defid=defid,
            scope=scope,
        )
        _define_in_scope(scope, var_name, binding, loc_node, self.tcx.reporter)
    
    # Default implementations for other nodes - just traverse children
    def visit_binary_expression(self, node) -> None:
        node.left.accept(self)
        node.right.accept(self)
    
    def visit_unary_expression(self, node) -> None:
        node.operand.accept(self)
    
    def visit_literal(self, node) -> None:
        pass  # Leaf node
    
    def visit_lambda_expression(self, node) -> None:
        with self.scope_manager.scope(ScopeKind.FUNCTION) as lambda_scope:
            param_defids: Dict[str, DefId] = {}
            for param_name in node.parameters:
                pn = str(param_name)
                if lambda_scope:
                    if lambda_scope.defined_in_this_scope(pn):
                        if self.tcx and self.tcx.reporter:
                            loc = getattr(node, "location", None)
                            self.tcx.reporter.report_error(f"duplicate parameter '{pn}'", loc)
                    param_defid = self.resolver.allocate_for_local()
                    param_defids[pn] = param_defid
                    _define_in_scope(
                        lambda_scope,
                        pn,
                        Binding(
                            name=pn,
                            binding_type=BindingType.PARAMETER,
                            definition=node,
                            defid=param_defid,
                            scope=lambda_scope
                        ),
                        node,
                        self.tcx.reporter,
                    )
            object.__setattr__(node, '_param_defids', param_defids)
            lambda_defid = self.resolver.allocate_for_local()
            object.__setattr__(node, 'defid', lambda_defid)
            if node.body:
                node.body.accept(self)
    
    def visit_if_expression(self, node) -> None:
        node.condition.accept(self)
        node.then_block.accept(self)  # AST uses then_block, not then_expr
        if node.else_block:
            node.else_block.accept(self)  # AST uses else_block, not else_expr
    
    def visit_program(self, node) -> None:
        for stmt in node.statements:
            stmt.accept(self)
    
    # Add other visit methods as needed - default to traversing children
    def visit_array_literal(self, node) -> None:
        for elem in node.elements:
            elem.accept(self)
    
    def visit_cast_expression(self, node) -> None:
        """Resolve names in cast expression"""
        if hasattr(node, 'expr') and node.expr:
            node.expr.accept(self)
    
    def visit_tuple_expression(self, node) -> None:
        """Resolve names in tuple expression"""
        if hasattr(node, 'elements'):
            for elem in node.elements:
                if hasattr(elem, 'accept'):
                    elem.accept(self)
    
    def visit_tuple_access(self, node) -> None:
        """Resolve names in tuple access"""
        if hasattr(node, 'tuple_expr') and node.tuple_expr:
            node.tuple_expr.accept(self)
    
    def visit_rectangular_access(self, node) -> None:
        """Resolve names in rectangular access (AST: base_expr; IR: array)."""
        base = getattr(node, 'base_expr', None) or getattr(node, 'array', None)
        if base:
            base.accept(self)
        for idx in (getattr(node, 'indices', None) or []):
            if idx is not None and hasattr(idx, 'accept'):
                idx.accept(self)
    
    def visit_member_access(self, node) -> None:
        """Resolve names in member access"""
        if hasattr(node, 'object') and node.object:
            node.object.accept(self)
    
    def visit_range(self, node) -> None:
        """Resolve names in range expression"""
        if hasattr(node, 'start') and node.start:
            node.start.accept(self)
        if hasattr(node, 'end') and node.end:
            node.end.accept(self)
        if hasattr(node, 'step') and node.step:
            node.step.accept(self)
    
    def visit_einstein_declaration(self, node) -> None:
        """Resolve names in Einstein declaration (iterate clauses)."""
        for clause in (node.clauses or []):
            clause.accept(self)

    def visit_einstein(self, node) -> None:
        """Resolve names in one Einstein clause."""
        for idx in (node.indices or []):
            if idx is not None and hasattr(idx, 'accept'):
                idx.accept(self)
        if node.value:
            node.value.accept(self)

    def visit_where_expression(self, node) -> None:
        """Resolve names in where expression (AST: expr, where_clause.constraints)."""
        if hasattr(node, 'expr') and node.expr:
            node.expr.accept(self)
        if hasattr(node, 'where_clause') and node.where_clause:
            for c in (getattr(node.where_clause, 'constraints', None) or []):
                if c is not None and hasattr(c, 'accept'):
                    c.accept(self)
    
    def visit_match_expression(self, node) -> None:
        """Resolve names in match expression (scrutinee, each arm pattern and body)."""
        scrutinee = getattr(node, 'scrutinee', None) or getattr(node, 'expr', None)
        if scrutinee:
            scrutinee.accept(self)
        if hasattr(node, 'arms'):
            for arm in node.arms:
                if hasattr(arm, 'pattern') and arm.pattern:
                    arm.pattern.accept(self)
                if hasattr(arm, 'body') and arm.body:
                    arm.body.accept(self)

    def visit_try_expression(self, node) -> None:
        """Resolve names in try expression."""
        operand = getattr(node, 'operand', None) or getattr(node, 'expr', None)
        if operand:
            operand.accept(self)

    def visit_arrow_expression(self, node) -> None:
        """Resolve names in arrow expression (all components)."""
        components = getattr(node, 'components', None)
        if components:
            for c in components:
                if c is not None and hasattr(c, 'accept'):
                    c.accept(self)
        elif hasattr(node, 'body') and node.body:
            node.body.accept(self)
    
    def visit_array_comprehension(self, node) -> None:
        """
        Resolve names in array comprehension.
        
        For [expr | var in range], we need to:
        1. Create a new scope for the comprehension
        2. Allocate DefId for the iteration variable
        3. Add it to the comprehension scope
        4. Resolve the body expression (which uses the variable)
        5. Resolve the range expression
        6. Pop the comprehension scope
        """
        from ..shared.nodes import BinaryExpression, Identifier, Range as ASTRange
        from ..shared.types import BinaryOp
        from ..shared.scope import ScopeKind
        
        # Create a new scope for the array comprehension (prevents iteration variables from leaking)
        def _do_resolve() -> None:
            iteration_variables = []  # List of (variable_name, variable_defid)
            seen_variables = {}  # Track variables we've already allocated to avoid duplicates
            if node.constraints:
                for constraint in node.constraints:
                    # Check if constraint is "var in range" or "var in array"
                    if isinstance(constraint, BinaryExpression) and constraint.operator == BinaryOp.IN:
                        if isinstance(constraint.left, Identifier):
                            variable_name = constraint.left.name
                            
                            # Check if we've already allocated this variable (e.g., x in 1..10, x in 5..15)
                            if variable_name in seen_variables:
                                # Reuse existing DefId
                                variable_defid = seen_variables[variable_name]
                                object.__setattr__(constraint.left, 'defid', variable_defid)
                                continue
                            
                            variable_defid = self.resolver.allocate_for_local() if getattr(self, 'resolver', None) else None
                            if variable_defid is not None:
                                object.__setattr__(constraint.left, 'defid', variable_defid)
                                if self.scope_manager:
                                    from ..shared.scope import Binding, BindingType
                                    binding = Binding(
                                        name=variable_name,
                                        binding_type=BindingType.VARIABLE,
                                        definition=constraint.left,
                                        defid=variable_defid,
                                        scope=self.scope_manager.current_scope()
                                    )
                                    _define_in_scope(self.scope_manager.current_scope(), variable_name, binding, constraint.left, self.tcx.reporter)
                            seen_variables[variable_name] = variable_defid
                            iteration_variables.append((variable_name, variable_defid))
            
            # Process bindings FIRST (before resolving body expression)
            # This allows the body expression to reference variables bound in where clauses
            # e.g., [result | x in data, y = x * 2, result = y + 1]
            if node.constraints:
                for constraint in node.constraints:
                    # Skip iteration variable constraints - they've already been processed above
                    if isinstance(constraint, BinaryExpression) and constraint.operator == BinaryOp.IN:
                        # This is an iteration variable constraint - already processed, just resolve the range
                        if hasattr(constraint, 'right'):
                            constraint.right.accept(self)
                        continue
                    
                    # Check if constraint is a binding (y = x * 2, result = y + 1)
                    if isinstance(constraint, BinaryExpression) and constraint.operator == BinaryOp.ASSIGN:
                        if isinstance(constraint.left, Identifier):
                            binding_var_name = constraint.left.name
                            if self.scope_manager:
                                from ..shared.scope import Binding, BindingType
                                binding_defid = self.resolver.allocate_for_local()
                                binding = Binding(
                                    name=binding_var_name,
                                    binding_type=BindingType.VARIABLE,
                                    definition=constraint.left,
                                    defid=binding_defid,
                                    scope=self.scope_manager.current_scope()
                                )
                                _define_in_scope(self.scope_manager.current_scope(), binding_var_name, binding, constraint.left, self.tcx.reporter)
                                object.__setattr__(constraint.left, 'defid', binding_defid)
                            # Resolve the binding expression (e.g., y + 1 in result = y + 1)
                            if hasattr(constraint, 'right'):
                                constraint.right.accept(self)
                        continue
                    
                    # This is a filter constraint (not a binding, not an iteration variable)
                    # Resolve the constraint - identifiers will resolve via scope lookup
                    constraint.accept(self)
            
            # NOW resolve body expression (which may use iteration variables AND bindings)
            # The body expression will resolve identifiers using the scope we just updated
            # No synchronization needed - if name resolution works correctly, identifiers should
            # resolve to the same DefIds as the iteration variables and bindings automatically
            node.expr.accept(self)

        if self.scope_manager:
            with self.scope_manager.scope(ScopeKind.BLOCK):
                _do_resolve()
        else:
            _do_resolve()
    
    def visit_match_expression(self, node) -> None:
        """
        Resolve match expression - allocate DefIds for identifier patterns.
        
        Rust Pattern: rustc_resolve::Resolver::resolve_match_expr()
        - Pattern variables are bound in the match arm scope
        - Each arm has its own scope for pattern variables
        - Pattern variables are available in the arm body
        """
        # Visit scrutinee first (outside match scope)
        if hasattr(node, 'scrutinee'):
            node.scrutinee.accept(self)
        elif hasattr(node, 'expr'):
            node.expr.accept(self)
        
        # Process each arm (Rust: each arm has its own scope for pattern variables)
        if hasattr(node, 'arms'):
            for arm in node.arms:
                # Enter scope for this arm's pattern variables (Rust pattern: per-arm scope)
                with self.scope_manager.scope(ScopeKind.BLOCK):
                    # Visit pattern first - this will call visit_identifier_pattern if it's an identifier pattern
                    # and allocate DefId and add to scope
                    if hasattr(arm, 'pattern') and arm.pattern:
                        arm.pattern.accept(self)
                    # Now visit the arm body (which will resolve identifiers in the body)
                    if hasattr(arm, 'body'):
                        arm.body.accept(self)
                    elif hasattr(arm, 'expr'):
                        arm.expr.accept(self)
    
    def visit_tuple_expression(self, node) -> None:
        for elem in node.elements:
            elem.accept(self)
    
    def visit_cast_expression(self, node) -> None:
        node.expr.accept(self)
    
    def visit_member_access(self, node) -> None:
        node.object.accept(self)
    
    def visit_method_call(self, node) -> None:
        node.object.accept(self)
        node.method_expr.accept(self)
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_arrow_expression(self, node) -> None:
        for component in node.components:
            component.accept(self)
    
    def visit_pipeline_expression(self, node) -> None:
        node.left.accept(self)
        node.right.accept(self)
    
    def visit_try_expression(self, node) -> None:
        node.operand.accept(self)
    
    def visit_reduction_expression(self, node, extra_constraint_lists=None) -> None:
        """Resolve names in reduction expression (AST only). Allocate DefIds for loop variables.
        extra_constraint_lists: optional list of constraint lists to resolve in reduction scope (e.g. outer WhereExpression constraints)."""
        from ..shared.scope import Binding, BindingType

        if not hasattr(node, 'over_clause') or not node.over_clause:
            return

        with self.scope_manager.scope(ScopeKind.REDUCTION) as reduction_scope:
            if hasattr(node.over_clause, 'range_groups'):
                for group in node.over_clause.range_groups:
                    if hasattr(group, 'range_expr') and group.range_expr:
                        group.range_expr.accept(self)
            loop_var_defids = {}
            if hasattr(node.over_clause, 'range_groups'):
                for group in node.over_clause.range_groups:
                    if hasattr(group, 'variables'):
                        for var_name in group.variables:
                            var_name = var_name if isinstance(var_name, str) else getattr(var_name, 'value', str(var_name))
                            var_name = str(var_name)
                            if not self.resolver:
                                continue
                            var_defid = self.resolver.allocate_for_local()
                            loop_var_defids[var_name] = var_defid
                            if reduction_scope:
                                binding = Binding(
                                    name=var_name,
                                    binding_type=BindingType.VARIABLE,
                                    definition=node,
                                    defid=var_defid,
                                    scope=reduction_scope,
                                )
                                _define_in_scope(reduction_scope, var_name, binding, node, self.tcx.reporter)
            object.__setattr__(node, '_reduction_loop_var_defids', loop_var_defids)

            node.body.accept(self)
            if node.where_clause:
                for constraint in node.where_clause.constraints:
                    constraint.accept(self)
            if extra_constraint_lists:
                for constraint_list in extra_constraint_lists:
                    for constraint in constraint_list:
                        constraint.accept(self)
    
    def visit_where_expression(self, node) -> None:
        from ..shared.nodes import WhereExpression as ASTWhereExpression
        from ..shared.nodes import ReductionExpression as ASTReductionExpression

        if not isinstance(node, ASTWhereExpression):
            return

        if isinstance(node.expr, ASTReductionExpression):
            outer = getattr(node, 'where_clause', None)
            from ..shared.types import BinaryOp
            from ..shared.nodes import BinaryExpression, Identifier
            reduction_loop_names = set()
            over = getattr(node.expr, 'over_clause', None)
            if over and getattr(over, 'range_groups', None):
                for group in over.range_groups:
                    for v in getattr(group, 'variables', []) or []:
                        name = v if isinstance(v, str) else getattr(v, 'value', getattr(v, 'name', str(v)))
                        if name:
                            reduction_loop_names.add(name)
            if outer and getattr(outer, 'constraints', None) and reduction_loop_names:
                op_name = getattr(node.expr, 'function_name', 'reduction')
                for constraint in outer.constraints:
                    if isinstance(constraint, BinaryExpression) and getattr(constraint, 'operator', None) == BinaryOp.IN:
                        left = getattr(constraint, 'left', None)
                        if isinstance(left, Identifier):
                            var_name = getattr(left, 'name', None)
                            if var_name is not None and not isinstance(var_name, str):
                                var_name = getattr(var_name, 'value', str(var_name))
                            if var_name and var_name in reduction_loop_names:
                                from ..shared.source_location import SourceLocation
                                loc = getattr(constraint, 'location', None) or getattr(node, 'location', None)
                                if loc is None:
                                    loc = SourceLocation(file="<test>", line=0, column=0, end_line=0, end_column=0)
                                self.tcx.reporter.report_error(
                                    f"Reduction cannot have iteration domain '{var_name} in ...' in where clause. "
                                    f"Use inline syntax: {op_name}[{var_name} in range](...).",
                                    loc,
                                    code="E0303",
                                )
            extra = [outer.constraints] if outer and getattr(outer, 'constraints', None) else None
            self.visit_reduction_expression(node.expr, extra_constraint_lists=extra)
        else:
            node.expr.accept(self)
            if node.where_clause:
                for constraint in node.where_clause.constraints:
                    constraint.accept(self)
    
    def visit_range(self, node) -> None:
        if node.start:
            node.start.accept(self)
        if node.end:
            node.end.accept(self)
        if getattr(node, 'step', None):
            node.step.accept(self)

    def visit_interpolated_string(self, node) -> None:
        for part in node.parts:
            part.accept(self)
    
    def visit_rectangular_access(self, node) -> None:
        node.base_expr.accept(self)
        for idx in (node.indices or []):
            if idx is not None and hasattr(idx, 'accept'):
                idx.accept(self)
    
    def visit_jagged_access(self, node) -> None:
        node.base_expr.accept(self)
        for idx in node.index_chain:
            idx.accept(self)
    
    def visit_expression_statement(self, node) -> None:
        node.expr.accept(self)
    
    def visit_einstein_declaration(self, node) -> None:
        """Allocate for Einstein (array name), then index vars and value."""
        module_path: tuple = ()
        scope = self.scope_manager.current_scope()
        is_module = scope is not None and scope.kind == ScopeKind.MODULE
        if is_module:
            existing = scope.lookup(node.array_name) if scope else None
            if existing and getattr(existing, "binding_type", None) == BindingType.VARIABLE and getattr(existing, "defid", None):
                if getattr(existing, "definition", None) is not node:
                    loc = getattr(node, "location", None)
                    if self.tcx and self.tcx.reporter:
                        self.tcx.reporter.report_error(
                            f"redefinition of '{node.array_name}' in same scope (use consecutive let for same array to merge, or use a different name)",
                            loc,
                        )
                defid = existing.defid
            else:
                defid = self.resolver.allocate_for_local()
            if scope:
                _define_in_scope(
                    scope,
                    node.array_name,
                    Binding(
                        name=node.array_name,
                        binding_type=BindingType.VARIABLE,
                        definition=node,
                        defid=defid,
                        scope=scope
                    ),
                    node,
                    self.tcx.reporter,
                )
            object.__setattr__(node, "defid", defid)
        else:
            existing_binding = scope.lookup(node.array_name) if scope else None
            if existing_binding and getattr(existing_binding, "binding_type", None) == BindingType.VARIABLE and getattr(existing_binding, "defid", None):
                if getattr(existing_binding, "definition", None) is not node:
                    loc = getattr(node, "location", None)
                    if self.tcx and self.tcx.reporter:
                        self.tcx.reporter.report_error(
                            f"redefinition of '{node.array_name}' in same scope (use consecutive let for same array to merge, or use a different name)",
                            loc,
                        )
                defid = existing_binding.defid
            else:
                defid = self.resolver.allocate_for_local()
                if scope:
                    _define_in_scope(
                        scope,
                        node.array_name,
                        Binding(
                            name=node.array_name,
                            binding_type=BindingType.VARIABLE,
                            definition=node,
                            defid=defid,
                            scope=scope
                        ),
                        node,
                        self.tcx.reporter,
                    )
            object.__setattr__(node, "defid", defid)
        from ..shared.nodes import Identifier, BinaryExpression
        from ..shared.types import BinaryOp
        from ..shared.defid import DefType

        with self.scope_manager.scope(ScopeKind.EINSTEIN) as einstein_scope:
            # Visit all indices from all clauses (index vars get DefIds in EINSTEIN scope)
            for clause in (getattr(node, "clauses", None) or []):
                for idx in (getattr(clause, "indices", None) or []):
                    idx.accept(self)
            from ..shared.nodes import ReductionExpression as ASTReductionExpression
            for clause in (getattr(node, "clauses", None) or []):
                if getattr(clause, "value", None):
                    val = clause.value
                    where = getattr(clause, "where_clause", None)
                    extra = [where.constraints] if where and getattr(where, "constraints", None) else None
                    if isinstance(val, ASTReductionExpression):
                        self.visit_reduction_expression(val, extra_constraint_lists=extra)
                    else:
                        val.accept(self)
            
            # Bind the variable name (e.g., "max_val" in "let max_val[..batch] = ...")
            var_name = getattr(node, 'array_name', None)
            if var_name:
                parent_scope = None
                if len(self.scope_manager._stack) > 1:
                    parent_scope = self.scope_manager._stack[-2]
                current_scope = parent_scope or self.scope_manager.current_scope()
                is_module = current_scope is not None and current_scope.kind == ScopeKind.MODULE

                if is_module:
                    existing = current_scope.lookup(var_name) if current_scope else None
                    if existing and getattr(existing, 'defid', None):
                        var_defid = existing.defid
                    else:
                        var_defid = getattr(node, 'defid', None) or self.resolver.allocate_for_local()
                        object.__setattr__(node, 'defid', var_defid)
                    if current_scope and not current_scope.defined_in_this_scope(var_name):
                        binding = Binding(
                            name=var_name,
                            binding_type=BindingType.VARIABLE,
                            definition=node,
                            defid=var_defid,
                            scope=current_scope
                        )
                        _define_in_scope(current_scope, var_name, binding, node, self.tcx.reporter)
                else:
                    var_defid = getattr(node, 'defid', None)
                    if not var_defid:
                        var_defid = self.resolver.allocate_for_local()
                        object.__setattr__(node, 'defid', var_defid)
                    existing_binding = self.scope_manager.lookup(var_name) if self.scope_manager else None
                    if current_scope and (not existing_binding or existing_binding.defid != var_defid) and not current_scope.defined_in_this_scope(var_name):
                        binding = Binding(
                            name=var_name,
                            binding_type=BindingType.VARIABLE,
                            definition=node,
                            defid=var_defid,
                            scope=current_scope
                        )
                        _define_in_scope(current_scope, var_name, binding, node, self.tcx.reporter)
            
            for clause in (getattr(node, "clauses", None) or []):
                where = getattr(clause, "where_clause", None)
                if not where or not getattr(where, "constraints", None):
                    continue
                if isinstance(getattr(clause, "value", None), ASTReductionExpression):
                    continue
                for constraint in where.constraints:
                    constraint.accept(self)
    
    def visit_inline_module(self, node) -> None:
        for stmt in node.body:
            stmt.accept(self)
    
    def visit_use_statement(self, node: UseStatement) -> None:
        """
        Process use statement (Rust pattern: use statement creates alias).
        
        Rust Edition 2018+ semantics:
        - use std::math;           → alias 'math', absolute (stdlib)
        - use std::math as m;      → alias 'm', absolute (stdlib)
        - use crate::helper;       → alias 'helper', explicit crate-relative
        - use helper;              → alias 'helper', implicit crate-relative (from root)
        - use super::sibling;      → relative (within modules)
        - use std::math::*;        → wildcard import (loads all exports)
        
        For non-wildcard imports: lazy loading (tree-shaking)
        - Only creates alias, module loaded when actually used
        
        For wildcard imports: immediate loading
        - Loads module NOW and imports all exports into scope
        """
        if not node.path or len(node.path) == 0:
            return
        
        # Convert AST's List[str] to tuple for internal use (Rust pattern: immutable paths)
        path_tuple = tuple(node.path)
        
        # Resolve path to absolute module path (Rust pattern: make_absolute)
        # Use the current_module_path attribute for context (set when loading stdlib modules)
        current_module = self.current_module_path if hasattr(self, 'current_module_path') else ()
        resolved_path = self._make_absolute_path(path_tuple, current_module)
        
        # Handle wildcard imports: use std::math::*
        # Rust pattern: Immediately load module and import all exports into scope
        if node.is_wildcard:
            
            # Load the module to get all its exports
            if self.module_loader:
                try:
                    module_info = self.module_loader.load_module(resolved_path)
                    self._ensure_module_resolved(module_info, resolved_path)
                    current_scope = self.scope_manager.current_scope()
                    
                    # Import all exported functions into current scope
                    for func_name in module_info.exports:
                        if func_name not in module_info.functions:
                            continue
                        func_def = module_info.functions[func_name]
                        # Only import public functions
                        if not getattr(func_def, 'is_public', False):
                            continue
                        # Store module path on function definition for later lowering
                        object.__setattr__(func_def, 'module_path', resolved_path)
                        # Allocate DefId for the imported function if not already done
                        if not hasattr(func_def, 'defid') or func_def.defid is None:
                            defid = self.resolver.allocate_for_item(resolved_path, func_name, func_def, DefType.FUNCTION)
                            object.__setattr__(func_def, 'defid', defid)
                        if current_scope:
                            if current_scope.defined_in_this_scope(func_name):
                                existing = current_scope._get_binding_in_this_scope(func_name)
                                existing_path = getattr(existing, "module_path", None)
                                if existing_path is not None:
                                    same_or_submodule = (existing_path == resolved_path or
                                        (len(resolved_path) <= len(existing_path) and existing_path[:len(resolved_path)] == resolved_path) or
                                        (len(existing_path) <= len(resolved_path) and resolved_path[:len(existing_path)] == existing_path))
                                    if not same_or_submodule and self.tcx and self.tcx.reporter:
                                        self.tcx.reporter.report_error(f"redefinition of '{func_name}' in same scope (duplicate import)", node.location if hasattr(node, 'location') else None)
                                    elif same_or_submodule:
                                        binding = Binding(
                                            name=func_name,
                                            binding_type=BindingType.FUNCTION,
                                            definition=func_def,
                                            defid=func_def.defid,
                                            scope=current_scope,
                                            module_path=resolved_path
                                        )
                                        _define_in_scope(current_scope, func_name, binding, None, self.tcx.reporter)
                                else:
                                    if self.tcx and self.tcx.reporter:
                                        self.tcx.reporter.report_error(f"use of '{func_name}' after local definition", node.location if hasattr(node, 'location') else None)
                            else:
                                binding = Binding(
                                    name=func_name,
                                    binding_type=BindingType.FUNCTION,
                                    definition=func_def,
                                    defid=func_def.defid,
                                    scope=current_scope,
                                    module_path=resolved_path
                                )
                                _define_in_scope(current_scope, func_name, binding, None, self.tcx.reporter)

                    logger.info(f"Wildcard import complete: {len(module_info.exports)} functions from {MODULE_SEPARATOR.join(resolved_path)}")
                except Exception as e:
                    import traceback
                    logger.warning(f"Failed to load module for wildcard import {resolved_path}: {e}")
            else:
                logger.warning("Module loader not initialized, skipping wildcard import")
            
            return
        
        # Rust pattern: Distinguish function imports from module imports
        # - use std::math::abs;  → function import (load immediately)
        # - use std::math;       → module import (register alias)
        # - use python::numpy;   → module import (Python modules have no .ein files)
        
        # Try to determine if this is a function import by attempting to load the module
        is_function_import = False
        func_name = None
        module_path = None
        
        # Python modules are ALWAYS module imports (they don't have physical .ein files)
        is_python_module = resolved_path and resolved_path[0] == 'python'
        
        if len(resolved_path) >= 2 and self.module_loader and not is_python_module:
            # Try interpreting last component as function name
            potential_func_name = resolved_path[-1]
            potential_module_path = resolved_path[:-1]
            
            try:
                # Try to load the parent module
                module_info = self.module_loader.load_module(potential_module_path)
                
                # Check if the last component is a function in this module
                if potential_func_name in module_info.functions:
                    is_function_import = True
                    func_name = potential_func_name
                    module_path = potential_module_path
            except Exception as e:
                # Not a function import, treat as module import
                pass
        
        # Handle function imports: use std::math::abs
        if is_function_import and func_name and module_path and self.module_loader:
            try:
                module_info = self.module_loader.load_module(module_path)
                # Run name resolution on the module so Einstein/reduction variables get correct DefIds.
                # _ensure_module_resolved pushes a scope and runs all statements (including use), so
                # we do not call _process_module_use_statements here (that would run use in current scope and pollute the caller).
                self._ensure_module_resolved(module_info, module_path)
                
                func_def = module_info.functions[func_name]
                
                # Check visibility - only public functions can be imported
                if not getattr(func_def, 'is_public', False):
                    error_msg = (
                        f"Cannot import private function '{func_name}' from module '{MODULE_SEPARATOR.join(module_path)}'. "
                        f"Only functions marked with 'pub' can be imported."
                    )
                    if self.tcx and self.tcx.reporter:
                        self.tcx.reporter.report_error(
                            error_msg,
                            node.location if hasattr(node, 'location') else None,
                        )
                    logger.error(error_msg)
                    return
                
                # Store module path on function definition for later lowering
                object.__setattr__(func_def, 'module_path', module_path)
                
                # Allocate DefId for the imported function
                if not hasattr(func_def, 'defid') or func_def.defid is None:
                    defid = self.resolver.allocate_for_item(module_path, func_name, func_def, DefType.FUNCTION)
                    object.__setattr__(func_def, 'defid', defid)
                else:
                    defid = func_def.defid
                
                define_name = getattr(node, 'alias', None) or func_name
                current_scope = self.scope_manager.current_scope()
                if current_scope:
                    if current_scope.defined_in_this_scope(define_name):
                        existing = current_scope._get_binding_in_this_scope(define_name)
                        existing_path = getattr(existing, "module_path", None)
                        if existing_path is not None:
                            same_or_submodule = (existing_path == module_path or
                                (len(module_path) <= len(existing_path) and existing_path[:len(module_path)] == module_path) or
                                (len(existing_path) <= len(module_path) and module_path[:len(existing_path)] == existing_path))
                            if not same_or_submodule and self.tcx and self.tcx.reporter:
                                self.tcx.reporter.report_error(f"redefinition of '{define_name}' in same scope (duplicate import)", node.location if hasattr(node, 'location') else None)
                            elif same_or_submodule:
                                binding = Binding(
                                    name=define_name,
                                    binding_type=BindingType.FUNCTION,
                                    definition=func_def,
                                    defid=defid,
                                    scope=current_scope,
                                    module_path=module_path
                                )
                                _define_in_scope(current_scope, define_name, binding, node, self.tcx.reporter)
                                logger.info(f"Function import: {define_name} from {MODULE_SEPARATOR.join(module_path)} with DefId {defid}")
                        else:
                            if self.tcx and self.tcx.reporter:
                                self.tcx.reporter.report_error(f"use of '{define_name}' after local definition", node.location if hasattr(node, 'location') else None)
                    else:
                        binding = Binding(
                            name=define_name,
                            binding_type=BindingType.FUNCTION,
                            definition=func_def,
                            defid=defid,
                            scope=current_scope,
                            module_path=module_path
                        )
                        _define_in_scope(current_scope, define_name, binding, node, self.tcx.reporter)
                        logger.info(f"Function import: {define_name} from {MODULE_SEPARATOR.join(module_path)} with DefId {defid}")
                
                return
            except Exception as e:
                import traceback
                logger.warning(f"Failed to load function {func_name} from {module_path}: {e}")
        
        # Handle module imports: use std::math (or use std::math as m)
        # Register alias for lazy loading - the actual loading happens when the name is used
        
        # Rust pattern: use std::math as m; → alias_name = 'm'
        #               use std::math;      → alias_name = 'math'
        if hasattr(node, 'alias') and node.alias:
            alias_name = node.alias  # Explicit alias: use python::random as rand
        else:
            alias_name = node.path[-1] if node.path else None  # Implicit: use python::random → 'random'
        
        if alias_name:
            # Register alias in resolver so lowering/backend can resolve np -> ('python', 'numpy')
            # Without this, ast_to_ir uses module_path = ('np',) and backend tries import_module('np') and fails
            self.resolver.register_alias(alias_name, resolved_path)
            # Also register in current scope for lexical lookup during name resolution
            current_scope = self.scope_manager.current_scope()
            if current_scope:
                if current_scope.defined_in_this_scope(alias_name):
                    existing = current_scope._get_binding_in_this_scope(alias_name)
                    existing_path = getattr(existing, "module_path", None)
                    if existing_path is not None:
                        same_or_submodule = (existing_path == resolved_path or
                            (len(resolved_path) <= len(existing_path) and existing_path[:len(resolved_path)] == resolved_path) or
                            (len(existing_path) <= len(resolved_path) and resolved_path[:len(existing_path)] == existing_path))
                        if not same_or_submodule and self.tcx and self.tcx.reporter:
                            self.tcx.reporter.report_error(f"redefinition of '{alias_name}' in same scope (duplicate import)", node.location if hasattr(node, 'location') else None)
                        elif same_or_submodule:
                            binding = Binding(
                                name=alias_name,
                                binding_type=BindingType.MODULE,
                                definition=None,
                                defid=None,
                                scope=current_scope,
                                module_path=resolved_path
                            )
                            _define_in_scope(current_scope, alias_name, binding, node, self.tcx.reporter)
                            logger.info(f"Registered module alias in scope: {alias_name} -> {MODULE_SEPARATOR.join(resolved_path)} (scope: {current_scope.kind})")
                    else:
                        if self.tcx and self.tcx.reporter:
                            self.tcx.reporter.report_error(f"use of '{alias_name}' after local definition", node.location if hasattr(node, 'location') else None)
                else:
                    binding = Binding(
                        name=alias_name,
                        binding_type=BindingType.MODULE,
                        definition=None,
                        defid=None,
                        scope=current_scope,
                        module_path=resolved_path
                    )
                    _define_in_scope(current_scope, alias_name, binding, node, self.tcx.reporter)
                    logger.info(f"Registered module alias in scope: {alias_name} -> {MODULE_SEPARATOR.join(resolved_path)} (scope: {current_scope.kind})")
        return
    
    def _make_absolute_path(self, path: Tuple[str, ...], current_module: Tuple[str, ...]) -> Tuple[str, ...]:
        """
        Make path absolute (Rust pattern: resolve relative paths).
        
        Args:
            path: Path to resolve (may be relative)
            current_module: Current module path for relative resolution
        
        Returns:
            Absolute module path
        """
        if not path:
            return current_module
        
        # Check if absolute (starts with 'std', 'crate', etc.)
        first = path[0]
        if first == 'std' or first == 'crate':
            # Absolute path
            return path
        elif first == 'super':
            # Relative: go up one level
            if not current_module:
                raise ValueError(f"Cannot use 'super' from crate root: {path}")
            return current_module[:-1] + path[1:] if len(current_module) > 0 else path[1:]
        else:
            # Implicit crate-relative: prepend current module
            return current_module + path if current_module else path
    
    def visit_module_declaration(self, node) -> None:
        pass  # Module declarations handled separately
    
    def visit_module_access(self, node) -> None:
        """
        Resolve module access (Rust pattern: std::math::sin).
        
        Handles qualified module access expressions:
        - std::math::sin → function reference
        - math::pi → property/constant access
        
        Rust Pattern: rustc_resolve::Resolver::resolve_path()
        """
        from ..shared.nodes import ModuleAccess, Identifier
        from ..shared.defid import DefType
        
        # Extract module path from node (e.g., std::math::sin)
        # ModuleAccess has: object (Identifier or ModuleAccess) and property (str)
        path_parts = []
        current = node
        while isinstance(current, ModuleAccess):
            path_parts.insert(0, current.property)
            current = current.object
        
        # Add the base identifier
        if isinstance(current, Identifier):
            path_parts.insert(0, current.name)
        else:
            # Complex expression - just visit it
            current.accept(self)
            return
        
        # Resolve module path (e.g., ('std', 'math', 'sin'))
        module_path = tuple(path_parts[:-1])  # All but last component
        item_name = path_parts[-1]  # Last component is the function/property name
        
        # Resolve alias if the first component is an alias (e.g., 'math' -> 'python::math')
        # Look up alias from scope (not global resolver) to respect lexical scoping
        if module_path and self.scope_manager:
            alias_binding = self.scope_manager.lookup(module_path[0])
            
            if alias_binding and alias_binding.binding_type == BindingType.MODULE:
                # Resolve alias: ('math',) -> ('python', 'math')
                resolved_module_path = alias_binding.module_path
                # If there are more components, append them
                if len(module_path) > 1:
                    module_path = resolved_module_path + module_path[1:]
                else:
                    module_path = resolved_module_path
                
        
        # Store resolved module path on the node for lowering (_resolved_module_key)
        # This allows lowering to extract module_path even if function_expr is transformed
        object.__setattr__(node, '_resolved_module_path', module_path)
        
        
        
        
        # CRITICAL: Never resolve python:: paths as Einlang functions
        # Python modules should never get DefIds assigned
        if module_path and len(module_path) > 0 and module_path[0] == 'python':
            # Store resolved module path but DO NOT assign DefId
            object.__setattr__(node, '_resolved_module_path', module_path)
            return  # Early return - don't try to resolve as Einlang function
        
        # 1. Use cache when existing
        func_defid = self.resolver.get_defid(module_path, item_name, DefType.FUNCTION)
        if func_defid:
            object.__setattr__(node, 'defid', func_defid)
            object.__setattr__(node, '_resolved_function_name', item_name)
            return

        # 2. std::: load via module_loader, register (cache), then use cache
        is_std = module_path and len(module_path) > 0 and module_path[0] == 'std'
        has_loader = bool(getattr(self, 'module_loader', None))
        if is_std and has_loader:
            try:
                self._load_and_register_module_functions(module_path, '', '', item_name)
                func_defid = self.resolver.get_defid(module_path, item_name, DefType.FUNCTION)
                if func_defid:
                    object.__setattr__(node, 'defid', func_defid)
                    object.__setattr__(node, '_resolved_function_name', item_name)
                    return
            except Exception as e:
                logger.debug(f"Stdlib load/cache failed for {module_path}::{item_name}: {e}")

        # 3. Non-std: file-based discovery, then load and register if source found
        file_path = None
        module_source = None
        actual_module_path = module_path
        if not is_std:
            discovered_modules = getattr(self.tcx, 'discovered_modules', None)
            if not discovered_modules and hasattr(self.tcx, 'module_system') and self.tcx.module_system:
                discovered_modules = self.tcx.module_system._discover_stdlib_modules()
            if discovered_modules:
                submodule_path = module_path + (item_name,)
                file_path = discovered_modules.get(submodule_path)
                if file_path and isinstance(file_path, Path) and file_path.exists():
                    module_source = _read_file_cached(file_path)
                    actual_module_path = submodule_path
                else:
                    file_path = discovered_modules.get(module_path)
                    if file_path and isinstance(file_path, Path) and file_path.exists():
                        module_source = _read_file_cached(file_path)
                        actual_module_path = module_path
            elif hasattr(self.tcx, 'module_system') and self.tcx.module_system:
                stdlib_modules = self.tcx.module_system._discover_stdlib_modules()
                if stdlib_modules:
                    submodule_path = module_path + (item_name,)
                    file_path = stdlib_modules.get(submodule_path)
                    if file_path and file_path.exists():
                        module_source = _read_file_cached(file_path)
                        actual_module_path = submodule_path
                    else:
                        file_path = stdlib_modules.get(module_path)
                        if file_path and file_path.exists():
                            module_source = _read_file_cached(file_path)
                            actual_module_path = module_path
        
        # Method 1: Use module_loader if available (handles path resolution, submodules, etc.)
        if not module_source and hasattr(self, 'module_loader') and self.module_loader:
            try:
                # Try submodule path first (e.g., ('std', 'math', 'abs') for std::math::abs)
                submodule_path = module_path + (item_name,)
                try:
                    module_info = self.module_loader.load_module(submodule_path)
                    if module_info and module_info.program:
                        # Get source from overlay (no I/O) or from path
                        if getattr(module_info, 'source_code', None):
                            module_source = module_info.source_code
                            actual_module_path = submodule_path
                            if discovered_modules:
                                file_path = discovered_modules.get(submodule_path)
                        elif hasattr(module_info, 'path') and module_info.path and not str(module_info.path).startswith('<overlay:'):
                            p = module_info.path
                            if p.exists():
                                module_source = _read_file_cached(p)
                                file_path = p
                                actual_module_path = submodule_path
                except Exception as e1:
                    # Fallback: Try parent module
                    try:
                        module_info = self.module_loader.load_module(module_path)
                        if module_info and module_info.program:
                            # Get source from overlay (no I/O) or from path
                            if getattr(module_info, 'source_code', None):
                                module_source = module_info.source_code
                                actual_module_path = module_path
                                if discovered_modules:
                                    file_path = discovered_modules.get(module_path)
                            elif hasattr(module_info, 'path') and module_info.path:
                                p = module_info.path
                                if not str(p).startswith('<overlay:'):
                                    if p.exists():
                                        module_source = _read_file_cached(p)
                                        file_path = p
                                        actual_module_path = module_path
                            if module_source:
                                pass
                    except Exception as e2:
                        pass
            except Exception as e:
                pass
        
        # Method 2: Use path_resolver (fallback)
        if not module_source and hasattr(self, 'path_resolver') and self.path_resolver:
            try:
                # Try submodule path FIRST (e.g., std::math::abs → stdlib/math/abs.ein)
                # This handles cases where the function is in a submodule file
                submodule_path = module_path + (item_name,)
                try:
                    resolved_submodule_path = self.path_resolver.resolve(submodule_path)
                    if resolved_submodule_path and resolved_submodule_path.exists() and resolved_submodule_path.is_file():
                        file_path = resolved_submodule_path
                        module_source = _read_file_cached(file_path)
                        actual_module_path = submodule_path
                except Exception as e2:
                    pass
                # If submodule not found, try parent module path
                if not module_source:
                    try:
                        resolved_path = self.path_resolver.resolve(module_path)
                        if resolved_path and resolved_path.exists():
                            if resolved_path.is_file():
                                file_path = resolved_path
                                module_source = _read_file_cached(file_path)
                                actual_module_path = module_path
                            elif resolved_path.is_dir():
                                mod_file = resolved_path / "mod.ein"
                                if mod_file.exists():
                                    file_path = mod_file
                                    module_source = _read_file_cached(file_path)
                                    actual_module_path = module_path
                    except Exception as e:
                        pass
            except Exception as e:
                pass
        
        # Method 3: Direct file system lookup (last resort fallback)
        # Try user modules (non-std) before Python fallback
        if not module_source and module_path and len(module_path) > 0 and module_path[0] != 'std' and module_path[0] != 'python':
            # Try to resolve as user module (Einlang module, not Python)
            if hasattr(self, 'path_resolver') and self.path_resolver:
                # Ensure crate_root is set before resolving user modules
                # Use stdlib_root's parent to find project root (not cwd which might be temp dir)
                if not self.path_resolver.crate_root and self.path_resolver.stdlib_root:
                    # Use stdlib_root's parent as project root
                    project_root = self.path_resolver.stdlib_root.parent
                    demos_path = project_root / "examples" / "demos"
                    if demos_path.exists() and demos_path.is_dir():
                        self.path_resolver.set_crate_root(demos_path)
                    else:
                        # Fallback to examples or project root
                        examples_path = project_root / "examples"
                        if examples_path.exists() and examples_path.is_dir():
                            self.path_resolver.set_crate_root(examples_path)
                        else:
                            self.path_resolver.set_crate_root(project_root)
                elif not self.path_resolver.crate_root:
                    # Fallback: search from stdlib_root if available, otherwise from cwd
                    from pathlib import Path as PathLib
                    start_path = self.path_resolver.stdlib_root.parent if self.path_resolver.stdlib_root else PathLib.cwd()
                    for _ in range(5):
                        demos_path = start_path / "examples" / "demos"
                        if demos_path.exists() and demos_path.is_dir():
                            self.path_resolver.set_crate_root(demos_path)
                            break
                        stdlib_path = start_path / "stdlib"
                        if stdlib_path.exists() and stdlib_path.is_dir():
                            if (start_path / "examples" / "demos").exists():
                                self.path_resolver.set_crate_root(start_path / "examples" / "demos")
                            else:
                                self.path_resolver.set_crate_root(start_path)
                            break
                        parent = start_path.parent
                        if parent == start_path:
                            break
                        start_path = parent
                
                try:
                    # Try to resolve user module path (e.g., math_utils → examples/demos/math_utils.ein)
                    resolved_path = self.path_resolver.resolve(module_path)
                    if resolved_path and resolved_path.exists():
                        if resolved_path.is_file():
                            file_path = resolved_path
                            module_source = _read_file_cached(file_path)
                            actual_module_path = module_path
                        elif resolved_path.is_dir():
                            mod_file = resolved_path / "mod.ein"
                            if mod_file.exists():
                                file_path = mod_file
                                module_source = _read_file_cached(file_path)
                                actual_module_path = module_path
                except Exception as e:
                    pass
        
        # Method 4: Direct file system lookup for stdlib (last resort fallback)
        # This handles cases where discovery/module_loader/path_resolver all fail
        if not module_source and module_path and len(module_path) > 0 and module_path[0] == 'std':
            # Try to find stdlib root and construct file path directly
            stdlib_root = None
            if hasattr(self.tcx, 'module_system') and self.tcx.module_system:
                stdlib_root = getattr(self.tcx.module_system, 'stdlib_root', None)
            elif hasattr(self, 'path_resolver') and self.path_resolver:
                stdlib_root = getattr(self.path_resolver, 'stdlib_root', None)
            
            if stdlib_root and stdlib_root.exists():
                # Try submodule file: stdlib/math/abs.ein
                submodule_path = module_path + (item_name,)
                submodule_file = stdlib_root
                for part in submodule_path[1:]:  # Skip 'std'
                    submodule_file = submodule_file / part
                submodule_file = submodule_file.with_suffix('.ein')
                
                if submodule_file.exists():
                    file_path = submodule_file
                    module_source = _read_file_cached(file_path)
                    actual_module_path = submodule_path
                else:
                    # Try parent module: stdlib/math/mod.ein or stdlib/math.ein
                    parent_module_file = stdlib_root
                    for part in module_path[1:]:  # Skip 'std'
                        parent_module_file = parent_module_file / part
                    # Try mod.ein first
                    mod_file = parent_module_file / "mod.ein"
                    if mod_file.exists():
                        file_path = mod_file
                        module_source = _read_file_cached(file_path)
                        actual_module_path = module_path
                    elif parent_module_file.with_suffix('.ein').exists():
                        file_path = parent_module_file.with_suffix('.ein')
                        module_source = _read_file_cached(file_path)
                        actual_module_path = module_path
        
        # If we have source, load and register the module
        if module_source:
            # Store in tcx.source_files for consistency
            if file_path:
                self.tcx.source_files[str(file_path)] = module_source
            
            # Handle submodule files (e.g., stdlib/math/abs.ein for std::math::abs)
            # If we found a submodule file, the function name should match the submodule name
            # Load the submodule and look for a function with that name
            if actual_module_path != module_path:
                # We found a submodule file - load it and look for function matching submodule name
                # Example: std::math::abs where abs.ein is the module file
                # The function in abs.ein should be named 'abs'
                func_defid = self._load_and_register_module_functions(
                    actual_module_path,
                    str(file_path) if file_path else "",
                    module_source,
                    item_name  # Function name should match submodule name
                )
                
                if func_defid and self.resolver.get_defid(module_path, item_name, DefType.FUNCTION) is None:
                    self.resolver.register_item(module_path, item_name, DefType.FUNCTION, func_defid)
                else:
                    pass
            else:
                # Regular module file - load normally
                func_defid = self._load_and_register_module_functions(
                    actual_module_path,
                    str(file_path) if file_path else "",
                    module_source,
                    item_name  # Load specific function (tree-shaking)
                )
            
            if func_defid:
                # Attach DefId to module access node
                object.__setattr__(node, 'defid', func_defid)
                object.__setattr__(node, '_resolved_function_name', item_name)
                return
        
        # If still not found, report error
        # But first check if this is a user module that should be resolved via crate_root
        if not module_source and module_path and len(module_path) > 0 and module_path[0] != 'std' and module_path[0] != 'python':
            # Try one more time with crate_root lookup for user modules
            if hasattr(self, 'path_resolver') and self.path_resolver and self.path_resolver.crate_root:
                try:
                    # Try examples/demos/math_utils.ein
                    user_module_file = self.path_resolver.crate_root / f"{module_path[0]}.ein"
                    if user_module_file.exists():
                        file_path = user_module_file
                        module_source = _read_file_cached(file_path)
                        actual_module_path = module_path
                        
                        # Load and register the module
                        func_defid = self._load_and_register_module_functions(
                            actual_module_path,
                            str(file_path),
                            module_source,
                            item_name
                        )
                        
                        if func_defid:
                            object.__setattr__(node, 'defid', func_defid)
                            object.__setattr__(node, '_resolved_function_name', item_name)
                            return
                except Exception as e:
                    pass
        
        if self.tcx and self.tcx.reporter and getattr(node, 'defid', None) is None:
            from ..analysis.module_system.path_resolver import MODULE_SEPARATOR
            if module_path and len(module_path) > 0 and module_path[0] == 'std':
                self.tcx.reporter.report_error(
                    f"Could not resolve stdlib function '{MODULE_SEPARATOR.join(module_path)}::{item_name}'.",
                    location=node.location if hasattr(node, 'location') else None,
                    code="E0433"
                )
            elif module_path and len(module_path) > 0 and module_path[0] != 'python':
                self.tcx.reporter.report_error(
                    f"Module '{MODULE_SEPARATOR.join(module_path)}' not found. "
                    f"Function '{item_name}' could not be resolved.",
                    location=node.location if hasattr(node, 'location') else None,
                    code="E0432"
                )
            else:
                self.tcx.reporter.report_error(
                    f"could not resolve name '{item_name}'",
                    location=node.location if hasattr(node, 'location') else None,
                )

    def visit_index_var(self, node) -> None:
        """Allocate DefId for variable index slot and bind in scope. Delegate to range_expr."""
        from ..shared.scope import Binding, BindingType
        from ..shared.nodes import IndexVar
        current_scope = self.scope_manager.current_scope() if self.scope_manager else None
        if not isinstance(node, IndexVar):
            return
        if not current_scope:
            loc = getattr(node, "location", None)
            msg = f"name resolution failed: no scope for index var '{getattr(node, 'name', '?')}'"
            if self.tcx and self.tcx.reporter:
                self.tcx.reporter.report_error(msg, loc)
            raise ValueError(msg)
        name = node.name
        existing = current_scope.lookup(name)
        if existing and getattr(existing, "defid", None):
            object.__setattr__(node, "defid", existing.defid)
        else:
            defid = self.resolver.allocate_for_local()
            binding = Binding(
                name=name,
                binding_type=BindingType.VARIABLE,
                definition=node,
                defid=defid,
                scope=current_scope,
            )
            _define_in_scope(current_scope, name, binding, node, self.tcx.reporter)
            object.__setattr__(node, "defid", defid)
        if getattr(node, "range_expr", None) is not None:
            node.range_expr.accept(self)

    def visit_index_rest(self, node) -> None:
        """Allocate DefId for rest index slot and bind in scope (AST only)."""
        from ..shared.scope import Binding, BindingType
        from ..shared.nodes import IndexRest
        current_scope = self.scope_manager.current_scope() if self.scope_manager else None
        if not isinstance(node, IndexRest):
            return
        if not current_scope:
            loc = getattr(node, "location", None)
            msg = f"name resolution failed: no scope for index rest '{getattr(node, 'name', '?')}'"
            if self.tcx and self.tcx.reporter:
                self.tcx.reporter.report_error(msg, loc)
            raise ValueError(msg)
        name = node.name
        existing = current_scope.lookup(name)
        if existing and getattr(existing, "defid", None):
            object.__setattr__(node, "defid", existing.defid)
        else:
            defid = self.resolver.allocate_for_local()
            binding = Binding(
                name=name,
                binding_type=BindingType.VARIABLE,
                definition=node,
                defid=defid,
                scope=current_scope,
            )
            _define_in_scope(current_scope, name, binding, node, self.tcx.reporter)
            object.__setattr__(node, "defid", defid)
    
    # Pattern visitors (no-op for name resolution)
    def visit_literal_pattern(self, node) -> None:
        pass
    
    def visit_identifier_pattern(self, node) -> None:
        """
        Resolve identifier pattern - bind in match arm scope.
        """
        from ..shared.scope import Binding, BindingType
        current_scope = self.scope_manager.current_scope()
        if current_scope:
            defid = self.resolver.allocate_for_local()
            binding = Binding(
                name=node.name,
                binding_type=BindingType.VARIABLE,
                definition=node,
                defid=defid,
                scope=current_scope
            )
            _define_in_scope(current_scope, node.name, binding, node, self.tcx.reporter)
            object.__setattr__(node, 'defid', defid)
    
    def visit_wildcard_pattern(self, node) -> None:
        pass
    
    def visit_tuple_pattern(self, node) -> None:
        """Resolve tuple pattern - recursively visit nested patterns"""
        if hasattr(node, 'patterns'):
            for pattern in node.patterns:
                if hasattr(pattern, 'accept'):
                    pattern.accept(self)
    
    def visit_array_pattern(self, node) -> None:
        """Resolve array pattern - recursively visit nested patterns"""
        if hasattr(node, 'patterns'):
            for pattern in node.patterns:
                if hasattr(pattern, 'accept'):
                    pattern.accept(self)
    
    def visit_rest_pattern(self, node) -> None:
        if hasattr(node, 'pattern') and node.pattern:
            node.pattern.accept(self)

    def visit_guard_pattern(self, node) -> None:
        if hasattr(node, 'pattern') and node.pattern:
            node.pattern.accept(self)
        if hasattr(node, 'guard') and node.guard:
            node.guard.accept(self)

    def visit_constructor_pattern(self, node) -> None:
        if hasattr(node, 'patterns'):
            for p in node.patterns:
                if p is not None and hasattr(p, 'accept'):
                    p.accept(self)

    def visit_or_pattern(self, node) -> None:
        """Resolve or pattern — each alternative must bind the same set of names."""
        if hasattr(node, 'alternatives'):
            for alt in node.alternatives:
                if hasattr(alt, 'accept'):
                    alt.accept(self)

    def visit_binding_pattern(self, node) -> None:
        """Resolve binding pattern: name @ inner (AST) or identifier_pattern @ inner_pattern (IR)."""
        from ..shared.scope import Binding, BindingType
        current_scope = self.scope_manager.current_scope()
        if current_scope:
            defid = self.resolver.allocate_for_local()
            binding = Binding(
                name=node.name,
                binding_type=BindingType.VARIABLE,
                definition=node,
                defid=defid,
                scope=current_scope
            )
            _define_in_scope(current_scope, node.name, binding, node, self.tcx.reporter)
            target = getattr(node, 'identifier_pattern', node)
            object.__setattr__(target, 'defid', defid)
        inner = getattr(node, 'inner_pattern', None) or getattr(node, 'pattern', None)
        if inner and hasattr(inner, 'accept'):
            inner.accept(self)

    def visit_range_pattern(self, node) -> None:
        pass

    def _resolve_function_from_module(self, module_path: Tuple[str, ...], function_name: str) -> Optional[DefId]:
        """
        Resolve function from a specific module (Rust pattern: on-demand module loading).
        
        Args:
            module_path: Module path tuple, e.g. ('std', 'math')
            function_name: Function name to find, e.g. 'abs'
            
        Returns:
            DefId if function found, None otherwise
        """
        
        if not (hasattr(self, 'module_loader') and self.module_loader):
            return None
        
        try:
            # Load the module using module_loader (handles pub use, submodules, etc.)
            module_info = self.module_loader.load_module(module_path)
            if not module_info or not module_info.program:
                return None
            
            
            # Parse and register functions from this module
            ast = module_info.program
            if module_path not in self._parsed_modules:
                self._parsed_modules[module_path] = ast
            
            # Use in-memory source when from overlay to avoid I/O; else read from path
            source = getattr(module_info, 'source_code', None)
            if source is None and hasattr(module_info, 'path') and module_info.path:
                path_str = str(module_info.path)
                if not path_str.startswith('<overlay:'):
                    source = _read_file_cached(module_info.path)
                else:
                    source = ''
            func_defid = self._load_and_register_module_functions(
                module_path,
                str(module_info.path) if hasattr(module_info, 'path') else '<module>',
                source,
                function_name
            )
            
            if func_defid:
                return func_defid
            
            return None
            
        except Exception as e:
            pass
        return None
    
    def _resolve_stdlib_function(self, function_name: str) -> Optional[DefId]:
        """
        Resolve function through module system (deprecated - use _resolve_function_from_module).
        
        This is a fallback for unqualified function calls without use statements.
        Following Rust pattern, such calls should fail - functions must be imported or qualified.
        """
        
        # Rust pattern: Unqualified stdlib calls should not work
        # User must either: 1) use std::math::abs; or 2) std::math::abs(x)
        # Returning None to enforce this pattern
        return None
    
    def _load_and_register_module_functions(
        self, 
        module_path: Tuple[str, ...], 
        file_path: str,
        module_source: str,
        function_name: Optional[str]  # None means load all functions
    ) -> Optional[DefId]:
        """
        Load module, parse it, and register all functions with DefIds.
        
        Uses new ModuleLoader for proper module loading with submodule discovery and pub use processing.
        
        Rust pattern: On-demand module loading and function registration with recursive dependencies.
        
        Returns DefId for the requested function, or None if not found.
        """
        # Try to use new ModuleLoader if available
        if self.module_loader and self.path_resolver:
            try:
                # Use ModuleLoader to load module (handles submodules, pub use, etc.)
                module_info = self.module_loader.load_module(module_path)
                ast = module_info.program
                _group_einstein_declarations_on_ast(ast)
                self._parsed_modules[module_path] = ast
                from ..shared.nodes import UseStatement, FunctionDefinition
                from ..shared.scope import Binding, BindingType, ScopeKind
                saved_module_path = self.current_module_path if hasattr(self, 'current_module_path') else ()
                self.current_module_path = module_path
                try:
                    with self.scope_manager.scope(ScopeKind.MODULE) as module_scope:
                        if not hasattr(self, '_resolved_stdlib_functions'):
                            self._resolved_stdlib_functions = set()
                        if not hasattr(self, '_registered_source_modules'):
                            self._registered_source_modules = set()

                        for _fn_name, func_def in list(module_info.functions.items()):
                            if hasattr(func_def, '_source_module') and func_def._source_module:
                                source_module_path = func_def._source_module
                                if source_module_path not in self._registered_source_modules:
                                    self._registered_source_modules.add(source_module_path)
                                    self._load_and_register_module_functions(
                                        source_module_path,
                                        str(module_info.path) if hasattr(module_info, 'path') else '<source>',
                                        getattr(module_info, 'source_code', None) or '',
                                        None,
                                    )

                        registered_functions = {}
                        for func_name_in_module, func_def in module_info.functions.items():
                            func_key = (module_path, func_name_in_module)
                            if func_key in self._resolved_stdlib_functions:
                                continue
                            
                            if hasattr(func_def, '_source_module') and func_def._source_module:
                                source_module_path = func_def._source_module
                                source_defid = self.resolver.get_defid(source_module_path, func_name_in_module, DefType.FUNCTION)
                                if source_defid:
                                    func_defid = source_defid
                                    object.__setattr__(func_def, 'defid', func_defid)
                                else:
                                    func_defid = self.resolver.allocate_for_item(module_path, func_name_in_module, func_def, DefType.FUNCTION)
                                    object.__setattr__(func_def, 'defid', func_defid)
                                self.resolver.register_item(module_path, func_name_in_module, DefType.FUNCTION, func_defid)
                            else:
                                if hasattr(func_def, 'defid') and func_def.defid is not None:
                                    func_defid = func_def.defid
                                else:
                                    func_defid = self.resolver.allocate_for_item(module_path, func_name_in_module, func_def, DefType.FUNCTION)
                                    object.__setattr__(func_def, 'defid', func_defid)
                                self.resolver.register_item(module_path, func_name_in_module, DefType.FUNCTION, func_defid)
                            
                            if not hasattr(func_def, 'module_path'):
                                object.__setattr__(func_def, 'module_path', module_path)
                            registered_functions[func_name_in_module] = (func_def, func_defid)
                        
                        local_item_names = {stmt.name for stmt in ast.statements if isinstance(stmt, FunctionDefinition)}
                        for func_name_in_module, (func_stmt, func_defid) in registered_functions.items():
                            if func_name_in_module not in local_item_names:
                                continue
                            func_binding = Binding(
                                name=func_name_in_module,
                                binding_type=BindingType.FUNCTION,
                                definition=func_stmt,
                                defid=func_defid,
                                scope=module_scope
                            )
                            if not module_scope.defined_in_this_scope(func_name_in_module):
                                _define_in_scope(module_scope, func_name_in_module, func_binding, func_stmt, self.tcx.reporter)

                        for stmt in ast.statements:
                            if isinstance(stmt, UseStatement):
                                stmt.accept(self)
                            elif isinstance(stmt, FunctionDefinition):
                                func_key = (module_path, stmt.name)
                                if func_key not in self._resolved_stdlib_functions:
                                    self._resolved_stdlib_functions.add(func_key)
                                    func_stmt = stmt
                                    with self.scope_manager.scope(ScopeKind.FUNCTION) as func_scope:
                                        for param in func_stmt.parameters:
                                            param_defid = self.resolver.allocate_for_local()
                                            if func_scope:
                                                param_binding = Binding(
                                                    name=param.name,
                                                    binding_type=BindingType.PARAMETER,
                                                    definition=param,
                                                    defid=param_defid,
                                                    scope=func_scope
                                                )
                                                _define_in_scope(func_scope, param.name, param_binding, param, self.tcx.reporter)
                                            object.__setattr__(param, 'defid', param_defid)
                                        
                                        from ..shared.nodes import BlockExpression
                                        if isinstance(func_stmt.body, BlockExpression):
                                            with self.scope_manager.scope(ScopeKind.BLOCK):
                                                pass
                                        
                                        if func_stmt.body:
                                            func_stmt.body.accept(self)
                
                finally:
                    self.current_module_path = saved_module_path
                
                if function_name:
                    return self.resolver.get_defid(module_path, function_name, DefType.FUNCTION)
                return None
                
            except Exception as e:
                logger.warning(f"Failed to load module {module_path} with ModuleLoader: {e}, falling back to old method")
                # Fall through to old method
        
        # Fallback to old method (for backward compatibility)
        # Track loading state to prevent circular dependencies
        if not hasattr(self, '_loading_modules'):
            self._loading_modules = set()
        
        if module_path in self._loading_modules:
            logger.warning(f"Circular dependency detected: {module_path} is already being loaded")
            return None
        
        self._loading_modules.add(module_path)
        
        try:
            # Parse module (cache parsed AST)
            if module_path not in self._parsed_modules:
                from ..frontend.parser import Parser
                from ..analysis.module_system.module_loader import _parse_cached
                try:
                    ast = _parse_cached(Parser(), module_source, file_path)
                    self._parsed_modules[module_path] = ast
                except Exception as e:
                    return None
            
            ast = self._parsed_modules[module_path]
            
            # Recursive dependency loading: Process use statements first
            # This ensures dependencies are loaded before we try to resolve function calls
            from ..shared.nodes import UseStatement
            for stmt in ast.statements:
                if isinstance(stmt, UseStatement):
                    # Process use statement to load dependencies
                    self._load_module_dependency(stmt, module_path)
            
            # Process pub use statements to re-export functions from submodules
            # This must happen after dependencies are loaded so submodules are available
            self._process_pub_use_reexports(ast, module_path)
            
            # Register all functions in the module with DefIds (Rust pattern: register all definitions)
            from ..shared.nodes import FunctionDefinition
            from ..shared.scope import Binding, BindingType, ScopeKind
            
            # Track which functions have been resolved to avoid infinite recursion
            # Use a set of (module_path, function_name) tuples
            if not hasattr(self, '_resolved_stdlib_functions'):
                self._resolved_stdlib_functions = set()
            
            # First pass: Register all functions (so they're available for cross-references)
            registered_functions = {}
            for stmt in ast.statements:
                if isinstance(stmt, FunctionDefinition):
                    # Check if already resolved
                    func_key = (module_path, stmt.name)
                    if func_key in self._resolved_stdlib_functions:
                        continue
                    
                    # Allocate DefId for function
                    func_defid = self.resolver.allocate_for_item(module_path, stmt.name, stmt, DefType.FUNCTION)
                    object.__setattr__(stmt, 'defid', func_defid)
                    object.__setattr__(stmt, 'module_path', module_path)
                    registered_functions[stmt.name] = (stmt, func_defid)
            
            # Second pass: Resolve function bodies (now all functions are registered and can reference each other)
            # Rust pattern: stdlib is a normal module, but we resolve it in isolation to avoid polluting current scope
            if registered_functions:
                # Enter module scope for this stdlib module (Rust pattern: module creates scope)
                with self.scope_manager.scope(ScopeKind.MODULE) as module_scope:
                    # Add all registered functions to module scope (so they can call each other)
                    # Rust pattern: Functions in same module can call each other without qualification
                    for func_name, (func_stmt, func_defid) in registered_functions.items():
                        func_binding = Binding(
                            name=func_name,
                            binding_type=BindingType.FUNCTION,
                            definition=func_stmt,
                            defid=func_defid,
                            scope=module_scope
                        )
                        if not module_scope.defined_in_this_scope(func_name):
                            _define_in_scope(module_scope, func_name, func_binding, func_stmt, self.tcx.reporter)
                    
                    # Now resolve each function body
                    for func_name, (func_stmt, func_defid) in registered_functions.items():
                        func_key = (module_path, func_name)
                        # Mark as being resolved (before body resolution to prevent cycles)
                        self._resolved_stdlib_functions.add(func_key)
                        
                        with self.scope_manager.scope(ScopeKind.FUNCTION) as func_scope:
                            for param in func_stmt.parameters:
                                param_defid = self.resolver.allocate_for_local()
                                if func_scope:
                                    param_binding = Binding(
                                        name=param.name,
                                        binding_type=BindingType.PARAMETER,
                                        definition=param,
                                        defid=param_defid,
                                        scope=func_scope
                                    )
                                    _define_in_scope(func_scope, param.name, param_binding, param, self.tcx.reporter)
                                object.__setattr__(param, 'defid', param_defid)
                            
                            # Allocate DefIds for local variables in function body
                            # (identifiers will be resolved later during monomorphization)
                            from ..shared.nodes import BlockExpression
                            if isinstance(func_stmt.body, BlockExpression):
                                with self.scope_manager.scope(ScopeKind.BLOCK):
                                    name_resolver = NameResolverVisitor(self.resolver, self.scope_manager, self.tcx)
                                    name_resolver.path_resolver = self.path_resolver
                                    name_resolver.module_loader = self.module_loader
                                    name_resolver.symbol_linker = self.symbol_linker
                                    func_stmt.body.accept(name_resolver)
            
            # Return DefId for requested function (if specified)
            if function_name:
                return self.resolver.get_defid(module_path, function_name, DefType.FUNCTION)
            # If function_name is None, we loaded all functions - return None
            return None
        finally:
            # Remove from loading set
            self._loading_modules.discard(module_path)
    
    def _process_pub_use_reexports(
        self,
        program,
        parent_module_path: Tuple[str, ...]
    ) -> None:
        """
        Process pub use re-exports (re-export functions from submodules).
        
        Handles:
        - pub use constants::*; → re-export all functions from constants submodule
        - pub use basic::sqrt; → re-export specific function
        
        This registers re-exported functions in the parent module's namespace.
        """
        from ..shared.nodes import UseStatement
        
        for stmt in program.statements:
            if not isinstance(stmt, UseStatement):
                continue
            
            # Only process public use statements (pub use)
            if not hasattr(stmt, 'is_public') or not stmt.is_public:
                continue
            
            # Resolve the submodule path
            if not stmt.path or len(stmt.path) == 0:
                continue
            
            # Resolve relative to parent module
            current_module = parent_module_path
            resolved_path = self._make_absolute_path(tuple(stmt.path), current_module)
            
            # For wildcard re-exports: pub use constants::*
            if stmt.is_wildcard:
                # Load the submodule to get its functions
                submodule_path = resolved_path
                if self.tcx.module_system and self.tcx.discovered_modules:
                    file_path = self.tcx.discovered_modules.get(submodule_path)
                    if file_path and file_path.exists():
                        module_source = _read_file_cached(file_path)
                        self.tcx.source_files[str(file_path)] = module_source
                        # Load and register submodule functions
                        self._load_and_register_module_functions(
                            submodule_path,
                            str(file_path),
                            module_source,
                            None  # Load all functions
                        )
                        
                        # Re-register all functions from submodule in parent namespace
                        for (registered_path, func_name, def_type), func_defid in list(self.tcx.symbol_table.items()):
                            if def_type == DefType.FUNCTION and registered_path == submodule_path:
                                if self.resolver.get_defid(parent_module_path, func_name, DefType.FUNCTION) is None:
                                    self.resolver.register_item(parent_module_path, func_name, DefType.FUNCTION, func_defid)
            
            # For specific function re-exports: pub use basic::sqrt [as alias]
            elif stmt.is_function and len(stmt.path) > 0:
                func_name = stmt.path[-1]
                submodule_path = resolved_path[:-1] if len(resolved_path) > 1 else resolved_path
                
                # Load submodule and get function DefId
                if self.tcx.module_system and self.tcx.discovered_modules:
                    file_path = self.tcx.discovered_modules.get(submodule_path)
                    if file_path and file_path.exists():
                        module_source = _read_file_cached(file_path)
                        self.tcx.source_files[str(file_path)] = module_source
                        func_defid = self._load_and_register_module_functions(
                            submodule_path,
                            str(file_path),
                            module_source,
                            func_name
                        )
                        
                        if func_defid:
                            alias_name = stmt.alias if stmt.alias else func_name
                            if self.resolver.get_defid(parent_module_path, alias_name, DefType.FUNCTION) is None:
                                self.resolver.register_item(parent_module_path, alias_name, DefType.FUNCTION, func_defid)
    
    def _ensure_module_resolved(self, module_info, module_path: Tuple[str, ...]) -> None:
        """
        Run name resolution (Phase 1 + Phase 2) on a loaded module's program so that
        all definitions get DefIds and identifiers (including Einstein/reduction variables)
        resolve correctly. Required for stdlib and other loaded modules whose AST is
        never part of the entry-point name resolution run. Resolves submodules recursively.
        """
        if not hasattr(self, '_resolved_modules'):
            self._resolved_modules = set()
        if module_path in self._resolved_modules:
            return
        self._resolved_modules.add(module_path)
        program = getattr(module_info, 'program', None)
        # Resolve submodules first (e.g. std::ml::indexing) so their ASTs get DefIds
        submodules = getattr(module_info, 'submodules', None) or {}
        for sub_name, sub_info in submodules.items():
            sub_path = module_path + (sub_name,)
            self._ensure_module_resolved(sub_info, sub_path)
        if not program or not getattr(program, 'statements', None):
            return
        _group_einstein_declarations_on_ast(program)
        from ..shared.scope import ScopeKind
        from ..shared.nodes import FunctionDefinition
        saved_module_path = self.current_module_path
        self.current_module_path = module_path
        try:
            with self.scope_manager.scope(ScopeKind.MODULE) as module_scope:
                for stmt in program.statements:
                    if isinstance(stmt, FunctionDefinition):
                        if module_scope.defined_in_this_scope(stmt.name):
                            continue
                        defid = self.resolver.allocate_for_item(module_path, stmt.name, stmt, DefType.FUNCTION)
                        object.__setattr__(stmt, "defid", defid)
                        object.__setattr__(stmt, "module_path", module_path)
                        _define_in_scope(
                            module_scope,
                            stmt.name,
                            Binding(
                                name=stmt.name,
                                binding_type=BindingType.FUNCTION,
                                definition=stmt,
                                defid=defid,
                                scope=module_scope,
                            ),
                            stmt,
                            self.tcx.reporter,
                        )
                for stmt in program.statements:
                    stmt.accept(self)
        finally:
            self.current_module_path = saved_module_path

    def _process_module_use_statements(self, program, module_path: Tuple[str, ...]) -> None:
        """
        Process use statements within a loaded module to bring imported functions into scope.
        
        This is needed when a module is loaded via ModuleLoader but hasn't had its use statements processed yet.
        """
        from ..shared.nodes import UseStatement
        # Check if we've already processed this module's use statements
        if not hasattr(self, '_processed_module_use_statements'):
            self._processed_module_use_statements = set()
        
        if module_path in self._processed_module_use_statements:
            return  # Already processed
        
        self._processed_module_use_statements.add(module_path)
        
        # Save and set current module context
        saved_module_path = self.current_module_path
        self.current_module_path = module_path
        try:
            for stmt in program.statements:
                if isinstance(stmt, UseStatement):
                    # Process the use statement to bring functions into scope
                    stmt.accept(self)
        finally:
            self.current_module_path = saved_module_path
    
    def _load_module_dependency(self, use_stmt: UseStatement, current_module_path: Tuple[str, ...]) -> None:
        """
        Load a module dependency recursively (Rust pattern: recursive dependency loading).
        
        When a module has a use statement, load that dependency and its dependencies recursively.
        This ensures all transitive dependencies are available.
        
        Args:
            use_stmt: Use statement from the current module
            current_module_path: Path of the module containing the use statement
        """
        if not use_stmt.path or len(use_stmt.path) == 0:
            return
        
        # Resolve dependency path to absolute module path
        path_tuple = tuple(use_stmt.path)
        resolved_path = self._make_absolute_path(path_tuple, current_module_path)
        
        # Check if dependency is already loaded
        if resolved_path in self._parsed_modules:
            return
        
        # Load dependency on-demand (recursive loading)
        if self.tcx.module_system and self.tcx.discovered_modules:
            file_path = self.tcx.discovered_modules.get(resolved_path)
            if file_path and file_path.exists():
                try:
                    module_source = _read_file_cached(file_path)
                    # Store in tcx.source_files
                    self.tcx.source_files[str(file_path)] = module_source
                    
                    # Recursively load and register dependency
                    # Pass None to load all functions in the dependency
                    self._load_and_register_module_functions(
                        resolved_path,
                        str(file_path),
                        module_source,
                        None  # Load all functions, not just one
                    )
                except Exception as e:
                    logger.warning(f"Failed to load dependency {resolved_path}: {e}")


