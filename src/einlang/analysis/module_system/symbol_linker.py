"""
Symbol Linking and Import Resolution

Pure symbol resolution algorithm for Einlang imports.
Follows Rust's name resolution and symbol linking.

Rust Pattern: rustc_resolve::UseTree resolution


This class is stateless and can be shared/reused.
"""

import logging
from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass

# Import AST types
try:
    from ...shared.nodes import Program, UseStatement
except ImportError:
    from typing import Any
    Program = Any
    UseStatement = Any

from .module_info import ModuleInfo

logger = logging.getLogger(__name__)

# Module separator 
MODULE_SEPARATOR = "::"


class SymbolNotFoundError(Exception):
    """Raised when a symbol cannot be resolved"""
    pass


@dataclass
class ResolvedImport:
    """Represents a successfully resolved import"""
    use_statement: UseStatement
    module_key: str  # Module key as string (for compatibility)
    imported_names: Set[str]
    is_wildcard: bool = False
    alias: Optional[str] = None
    
    def __repr__(self) -> str:
        return f"ResolvedImport(module={self.module_key}, names={self.imported_names}, wildcard={self.is_wildcard})"


@dataclass
class _ResolveResult:
    """Internal result of resolving a single use statement"""
    function_mappings: Dict[str, str]  # function_name -> module_key (string)
    import_info: ResolvedImport
    module_alias: Optional[Dict[str, str]] = None  # alias -> full_module_key (string)


class SymbolLinker:
    """
    Pure symbol resolution for Einlang imports.
    
    Resolves use statements to actual module symbols:
    - use std::math::sin; → maps 'sin' to 'std::math'
    - use python::numpy::*; → maps all numpy functions
    - use my_module as m; → creates module alias
    
    This class is stateless and can be shared/reused.
    All state is passed explicitly or returned.
    
    Rust Pattern: rustc_resolve::UseTree resolution
    """
    
    def resolve_imports(
        self,
        program: Program,
        loaded_modules: Dict[str, ModuleInfo]  # module_key (string) -> ModuleInfo
    ) -> Tuple[Dict[str, str], Dict[str, str], List[ResolvedImport], List[str]]:
        """
        Resolve all use statements in a program.
        
        Args:
            program: AST program containing use statements
            loaded_modules: Dict of module_key (string) -> ModuleInfo
        
        Returns:
            Tuple of:
            - function_to_module: Dict mapping function names to module keys (strings)
            - module_aliases: Dict mapping aliases to full module keys (strings)
            - resolved_imports: List of successfully resolved imports
            - errors: List of error messages
        
        Examples:
            After "use std::math::sin;":
            function_to_module = {'sin': 'std::math'}
            
            After "use std::math::constants;":
            module_aliases = {'constants': 'std::math::constants'}
        """
        function_to_module: Dict[str, str] = {}
        module_aliases: Dict[str, str] = {}
        resolved_imports: List[ResolvedImport] = []
        errors: List[str] = []
        
        use_statements = [stmt for stmt in program.statements if isinstance(stmt, UseStatement)]
        logger.debug(f"Found {len(use_statements)} use statements to resolve")
        
        for stmt in use_statements:
            try:
                logger.debug(f"Resolving use statement: {stmt.path}, is_function={stmt.is_function}, is_wildcard={stmt.is_wildcard}")
                result = self._resolve_use_statement(stmt, loaded_modules)
                if result:
                    # Merge function mappings
                    function_to_module.update(result.function_mappings)
                    # Add module alias if present
                    if result.module_alias:
                        module_aliases.update(result.module_alias)
                    resolved_imports.append(result.import_info)
                    logger.debug(f"Resolved: added {len(result.function_mappings)} function mappings")
            except Exception as e:
                error_msg = f"Failed to resolve import {stmt.path}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        logger.debug(f"Total function mappings: {len(function_to_module)}, module aliases: {len(module_aliases)}")
        return function_to_module, module_aliases, resolved_imports, errors
    
    def _resolve_use_statement(
        self,
        use_stmt: UseStatement,
        loaded_modules: Dict[str, ModuleInfo]
    ) -> Optional[_ResolveResult]:
        """Resolve a single use statement"""
        
        if use_stmt.is_wildcard:
            return self._resolve_wildcard(use_stmt, loaded_modules)
        elif use_stmt.is_function:
            return self._resolve_function(use_stmt, loaded_modules)
        else:
            return self._resolve_module(use_stmt, loaded_modules)
    
    def _resolve_wildcard(
        self,
        use_stmt: UseStatement,
        loaded_modules: Dict[str, ModuleInfo]
    ) -> _ResolveResult:
        """
        Resolve wildcard import: use std::math::*;
        
        Imports all exported functions from the module.
        """
        module_key = MODULE_SEPARATOR.join(use_stmt.path)
        module_info = loaded_modules.get(module_key)
        
        if not module_info:
            raise SymbolNotFoundError(
                f"Module '{module_key}' not found for wildcard import. "
                f"Available modules: {', '.join(sorted(loaded_modules.keys()))}"
            )
        
        # Import all exported functions
        function_mappings = {}
        imported_names = set()
        for func_name in module_info.exports:
            # Convert module name tuple to string for compatibility
            module_key_str = MODULE_SEPARATOR.join(module_info.name)
            function_mappings[func_name] = module_key_str
            imported_names.add(func_name)
        
        import_info = ResolvedImport(
            use_statement=use_stmt,
            module_key=module_key,
            imported_names=imported_names,
            is_wildcard=True
        )
        
        logger.debug(f"Resolved wildcard: {module_key} → {len(imported_names)} functions")
        return _ResolveResult(function_mappings, import_info)
    
    def _resolve_function(
        self,
        use_stmt: UseStatement,
        loaded_modules: Dict[str, ModuleInfo]
    ) -> _ResolveResult:
        """
        Resolve function import: use std::math::sin;
        
        Imports a specific function from a module.
        """
        if len(use_stmt.path) < 2:
            raise SymbolNotFoundError(f"Invalid function import path: {use_stmt.path}")
        
        module_path = use_stmt.path[:-1]  # ['std', 'math']
        func_name = use_stmt.path[-1]     # 'sin'
        
        module_key = MODULE_SEPARATOR.join(module_path)
        module_info = loaded_modules.get(module_key)
        
        if not module_info:
            raise SymbolNotFoundError(
                f"Module '{module_key}' not found for function '{func_name}'. "
                f"Available modules: {', '.join(sorted(loaded_modules.keys()))}"
            )
        
        # Check if function exists in module
        if func_name not in module_info.exports:
            raise SymbolNotFoundError(
                f"Function '{func_name}' not exported from '{module_key}'. "
                f"Available: {', '.join(sorted(module_info.exports))}"
            )
        
        # Map function name to module (convert tuple to string)
        module_key_str = MODULE_SEPARATOR.join(module_info.name)
        function_mappings = {func_name: module_key_str}
        
        import_info = ResolvedImport(
            use_statement=use_stmt,
            module_key=module_key,
            imported_names={func_name},
            is_wildcard=False
        )
        
        logger.debug(f"Resolved function: {func_name} from {module_key}")
        return _ResolveResult(function_mappings, import_info)
    
    def _resolve_module(
        self,
        use_stmt: UseStatement,
        loaded_modules: Dict[str, ModuleInfo]
    ) -> _ResolveResult:
        """
        Resolve module import: use std::math; or use std::math as m;
        
        Imports the module itself (for qualified calls like math::sin).
        
        Rust behavior:
        - use python::math; → alias "math" → "python::math"
        - use python::math as pm; → alias "pm" → "python::math"
        """
        module_key = MODULE_SEPARATOR.join(use_stmt.path)
        module_info = loaded_modules.get(module_key)
        
        # AUTO-DETECT: Prioritize function imports over module imports
        # If parent module exports a function with the same name, treat as function import
        # Example: use std::math::sum; → if std::math exports 'sum', import the function
        if len(use_stmt.path) >= 2:
            parent_module_key = MODULE_SEPARATOR.join(use_stmt.path[:-1])
            potential_func_name = use_stmt.path[-1]
            parent_module = loaded_modules.get(parent_module_key)
            
            if parent_module and potential_func_name in parent_module.exports:
                # This is a function import! Resolve it as such
                logger.debug(f"Auto-detected function import: {potential_func_name} from {parent_module_key}")
                # Create a modified use statement with is_function=True
                func_stmt = UseStatement(
                    path=use_stmt.path,
                    is_function=True,
                    is_wildcard=False,
                    is_public=use_stmt.is_public,
                    alias=use_stmt.alias,
                    location=use_stmt.location
                )
                return self._resolve_function(func_stmt, loaded_modules)
        
        if not module_info:
            raise SymbolNotFoundError(
                f"Module '{module_key}' not found for module import. "
                f"Available modules: {', '.join(sorted(loaded_modules.keys()))}"
            )
        
        # Create module alias mapping for qualified function calls
        if use_stmt.alias:
            # use python::math as pm; -> "pm" maps to "python::math"
            module_alias_name = use_stmt.alias
        else:
            # use std::math::constants; -> "constants" maps to "std::math::constants"
            module_alias_name = use_stmt.path[-1]  # Last part of path
        
        module_alias = {module_alias_name: module_key}
        
        import_info = ResolvedImport(
            use_statement=use_stmt,
            module_key=module_key,
            imported_names=set(),  # No direct function imports
            is_wildcard=False,
            alias=use_stmt.alias
        )
        
        logger.debug(f"Resolved module: {module_key} → alias '{module_alias_name}'")
        return _ResolveResult({}, import_info, module_alias)
    
    def check_symbol_exists(
        self,
        symbol_name: str,
        function_to_module: Dict[str, str],
        loaded_modules: Dict[str, ModuleInfo]
    ) -> bool:
        """
        Check if a symbol is accessible after import resolution.
        
        Args:
            symbol_name: Function name to check
            function_to_module: Resolved function mappings
            loaded_modules: All loaded modules
        
        Returns:
            True if symbol exists and is accessible
        """
        return symbol_name in function_to_module
    
    def get_module_for_symbol(
        self,
        symbol_name: str,
        function_to_module: Dict[str, str]
    ) -> Optional[str]:
        """Get the module key that exports a given symbol"""
        return function_to_module.get(symbol_name)

