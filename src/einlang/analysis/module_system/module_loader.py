"""
Module Loader

Loads modules and builds module tree with proper submodule discovery and pub use processing.

Rust Pattern: rustc_metadata::loader


This class handles:
- Module loading from filesystem
- Submodule discovery from mod declarations
- pub use re-export processing
- Building proper ModuleInfo structure with hierarchy
"""

import copy
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from .path_resolver import PathResolver, ModuleNotFoundError
from .module_info import ModuleInfo, ModuleDeclaration
from ...utils.io_utils import read_source_file

# Import AST types 
try:
    from ...shared.nodes import Program, UseStatement, FunctionDefinition, InlineModule
except ImportError:
    Program = Any
    UseStatement = Any
    FunctionDefinition = Any
    InlineModule = Any

logger = logging.getLogger(__name__)

# Constants
MODULE_SEPARATOR = "::"
MODULE_FILE_EXTENSION = ".ein"

# Cache file contents to avoid repeated I/O across compilations (e.g. same stdlib modules)
_file_content_cache: Dict[Path, str] = {}
_FILE_CACHE_MAX = 256


def _read_file_cached(file_path: Path) -> str:
    """Read file content with process-wide cache to reduce I/O."""
    key = file_path.resolve()
    if key in _file_content_cache:
        return _file_content_cache[key]
    content = read_source_file(file_path)
    if len(_file_content_cache) < _FILE_CACHE_MAX:
        _file_content_cache[key] = content
    return content


def clear_file_content_cache() -> None:
    """Clear the file content cache. Used by tests for parallel-safe runs (-n auto)."""
    _file_content_cache.clear()
    _parse_source.cache_clear()


@lru_cache(maxsize=256)
def _parse_source(source_code: str, source_file: str):
    """Parse source string to AST, cached across compilations."""
    from ...frontend.parser import Parser
    return Parser().parse(source_code, source_file)


def _parse_cached(parser, source_code: str, source_file: str):
    """Return a deep copy of the cached parse result (callers mutate ASTs)."""
    return copy.deepcopy(_parse_source(source_code, source_file))


class CircularImportError(Exception):
    """Raised when circular imports are detected"""
    pass


class ModuleLoader:
    """
    Module loader that builds proper module tree with submodule discovery.
    
    Rust Pattern: rustc_metadata::loader
    
    This class:
    - Loads modules from filesystem
    - Discovers submodules from mod declarations
    - Processes pub use re-exports
    - Builds ModuleInfo structure with proper hierarchy
    """
    
    def __init__(
        self,
        path_resolver: PathResolver,
        parser: Optional[Any] = None,
        source_overlay: Optional[Dict[Tuple[str, ...], str]] = None,
    ):
        """
        Args:
            path_resolver: PathResolver instance for path resolution
            parser: Parser instance (auto-created if None)
            source_overlay: Optional in-memory module sources (module_path -> source);
                           when set, load from overlay instead of disk (avoids I/O).
        """
        self.path_resolver = path_resolver
        self.source_overlay = source_overlay or {}
        self.loaded_modules: Dict[Tuple[str, ...], ModuleInfo] = {}
        self.loading_stack: List[Tuple[str, ...]] = []

        # Support dependency injection or use shared parser
        if parser is None:
            from ...frontend.parser import Parser
            self.parser = Parser()
        else:
            self.parser = parser
    
    def load_module(self, module_path: Tuple[str, ...]) -> ModuleInfo:
        """
        Load a module and build its complete hierarchy.
        
        Args:
            module_path: Module path as tuple (e.g., ('std', 'math'))
        
        Returns:
            ModuleInfo with complete hierarchy (submodules, re-exports, etc.)
        
        Raises:
            ModuleNotFoundError: If module cannot be found
            CircularImportError: If circular import detected
        """
        # Return cached module if already loaded
        if module_path in self.loaded_modules:
            return self.loaded_modules[module_path]
        
        # Check for circular imports
        if module_path in self.loading_stack:
            chain = " -> ".join(MODULE_SEPARATOR.join(k) for k in self.loading_stack + [module_path])
            raise CircularImportError(f"Circular import detected: {chain}")
        
        try:
            self.loading_stack.append(module_path)
            if len(self.loading_stack) > 20:
                stack_str = " -> ".join(str(p) for p in self.loading_stack[-10:])
                raise RuntimeError(f"Deep recursion detected (depth {len(self.loading_stack)}): ...{stack_str}")
            
            # Prefer in-memory overlay to avoid I/O on critical path
            if module_path in self.source_overlay:
                source_code = self.source_overlay[module_path]
                path_for_info = Path(f"<overlay:{MODULE_SEPARATOR.join(module_path)}>")
            else:
                file_path = self.path_resolver.resolve(module_path)
                if self.path_resolver.is_python_module(module_path):
                    return self._load_python_module(module_path, file_path)
                source_code = _read_file_cached(file_path)
                path_for_info = file_path

            program = _parse_cached(self.parser, source_code, str(path_for_info))
            
            # Parse module declarations (mod name;)
            declarations = self._parse_module_declarations(source_code)
            
            # Parse inline modules from AST
            inline_declarations = self._parse_inline_modules(program)
            declarations.extend(inline_declarations)
            
            # Extract functions and exports
            functions, exports = self._extract_functions_and_exports(program)
            
            # Load declared submodules recursively
            submodules = self._load_declared_submodules(module_path, declarations)
            
            # Process pub use re-exports (requires submodules to be loaded first)
            self._process_pub_use_reexports(program, submodules, functions, exports, module_path)
            
            # Create module info (source_code set when from overlay to avoid I/O in name_resolution)
            module_info = ModuleInfo(
                name=module_path,
                path=path_for_info,
                source_code=source_code if module_path in self.source_overlay else None,
                program=program,
                declarations=declarations,
                functions=functions,
                exports=exports,
                submodules=submodules
            )
            
            # Cache and return
            self.loaded_modules[module_path] = module_info
            logger.debug(f"Loaded module {MODULE_SEPARATOR.join(module_path)}: {len(functions)} functions, {len(submodules)} submodules")
            return module_info
            
        except Exception as e:
            if isinstance(e, (ModuleNotFoundError, CircularImportError)):
                raise
            else:
                raise RuntimeError(f"Failed to load module {module_path}: {e}")
        finally:
            self.loading_stack.pop()
    
    def _load_python_module(self, module_path: Tuple[str, ...], virtual_path: Path) -> ModuleInfo:
        """Load a Python module (virtual module)"""
        # Create minimal ModuleInfo for Python module
        from ...shared.nodes import Program
        program = Program(statements=[])
        
        module_info = ModuleInfo(
            name=module_path,
            path=virtual_path,
            program=program,
            declarations=[],
            functions={},
            exports=set(),
            submodules={}
        )
        
        # Cache and return
        self.loaded_modules[module_path] = module_info
        return module_info
    
    def _parse_module_declarations(self, source: str) -> List[ModuleDeclaration]:
        """Parse 'mod name;' and 'pub mod name;' declarations from source"""
        declarations = []
        
        for line_num, line in enumerate(source.split('\n'), 1):
            stripped = line.strip()
            
            # Skip comments and empty lines
            if not stripped or stripped.startswith('#'):
                continue
            
            # Match: 'mod name;' or 'pub mod name;'
            match = re.match(r'^(pub\s+)?mod\s+(\w+);', stripped)
            if match:
                is_public = bool(match.group(1))
                name = match.group(2)
                declarations.append(ModuleDeclaration(
                    name=name,
                    is_public=is_public,
                    location=(line_num, 1)
                ))
        
        return declarations
    
    def _parse_inline_modules(self, program: Program) -> List[ModuleDeclaration]:
        """Parse inline modules from AST"""
        declarations = []
        
        for stmt in program.statements:
            if isinstance(stmt, InlineModule):
                declarations.append(ModuleDeclaration(
                    name=stmt.name,
                    is_public=stmt.is_public,
                    location=(stmt.location.line, stmt.location.column) if hasattr(stmt, 'location') and stmt.location else None
                ))
        
        return declarations
    
    def _extract_functions_and_exports(self, program: Program) -> Tuple[Dict[str, FunctionDefinition], Set[str]]:
        """
        Extract functions and determine exports from parsed program.
        
        Following Rust's visibility rules:
        - Only pub functions are exported
        - Private functions are only accessible within the module
        """
        functions = {}
        exports = set()
        
        for stmt in program.statements:
            if isinstance(stmt, FunctionDefinition):
                functions[stmt.name] = stmt
                # Rust semantics: Only pub functions are exported
                if stmt.is_public:
                    exports.add(stmt.name)
            elif isinstance(stmt, InlineModule):
                # Extract functions from inline modules
                inline_functions, inline_exports = self._extract_functions_from_inline_module(stmt)
                functions.update(inline_functions)
                exports.update(inline_exports)
        
        return functions, exports
    
    def _extract_functions_from_inline_module(self, inline_module: InlineModule) -> Tuple[Dict[str, FunctionDefinition], Set[str]]:
        """Extract functions and exports from an inline module"""
        functions = {}
        exports = set()
        
        # Extract statements from the body (BlockExpression.statements or List[Statement])
        body = inline_module.body
        if hasattr(body, 'statements'):
            body_statements = body.statements
        elif isinstance(body, list):
            body_statements = body
        else:
            body_statements = []
        
        for stmt in body_statements:
            if isinstance(stmt, FunctionDefinition):
                # Prefix function names with module name to avoid conflicts
                prefixed_name = f"{inline_module.name}::{stmt.name}"
                functions[prefixed_name] = stmt
                if stmt.is_public:
                    exports.add(prefixed_name)
        
        return functions, exports
    
    def _load_declared_submodules(
        self,
        parent_path: Tuple[str, ...],
        declarations: List[ModuleDeclaration]
    ) -> Dict[str, ModuleInfo]:
        """
        Load all declared submodules recursively.
        
        Args:
            parent_path: Parent module path
            declarations: List of mod declarations
        
        Returns:
            Dict mapping submodule name to ModuleInfo
        """
        submodules = {}
        
        for decl in declarations:
            # Load all declared submodules (both public and private for internal access)
            try:
                submodule_path = parent_path + (decl.name,)
                submodule_info = self.load_module(submodule_path)
                submodules[decl.name] = submodule_info
                logger.debug(f"Loaded submodule {decl.name} for parent {MODULE_SEPARATOR.join(parent_path)}")
            except Exception as e:
                logger.warning(f"Failed to load submodule {decl.name}: {e}")
        
        return submodules
    
    def _process_pub_use_reexports(
        self,
        program: Program,
        submodules: Dict[str, ModuleInfo],
        functions: Dict[str, FunctionDefinition],
        exports: Set[str],
        current_module_path: Tuple[str, ...]
    ) -> None:
        """
        Process pub use re-exports after submodules are loaded.
        
        This must be called after _load_declared_submodules() because we need
        the submodules to be available before we can re-export their functions.
        
        Handles two forms:
        1. pub use module::* - re-export all public functions from module
        2. pub use module::function [as alias] - re-export specific function
        
        Args:
            program: Parsed AST
            submodules: Dict of loaded submodules
            functions: Dict to update with re-exported functions
            exports: Set to update with re-exported function names
            current_module_path: Current module path (for logging)
        """
        for stmt in program.statements:
            if not isinstance(stmt, UseStatement):
                continue
            
            # Only process public use statements (pub use)
            if not stmt.is_public:
                continue
            
            # Wildcard re-export: pub use module::*
            if stmt.is_wildcard and len(stmt.path) == 1:
                submodule_name = stmt.path[0]
                if submodule_name in submodules:
                    submodule_info = submodules[submodule_name]
                    # Re-export all public functions from submodule
                    for func_name in submodule_info.exports:
                        if func_name in submodule_info.functions:
                            functions[func_name] = submodule_info.functions[func_name]
                            exports.add(func_name)
                            logger.debug(f"Re-exported {func_name} from submodule {submodule_name} via pub use {submodule_name}::*")
                else:
                    logger.warning(f"Submodule {submodule_name} not found for wildcard re-export in {MODULE_SEPARATOR.join(current_module_path)}")
            
            # Specific function re-export: pub use module::function [as alias]
            elif not stmt.is_wildcard and len(stmt.path) >= 2:
                func_name = stmt.path[-1]
                export_name = stmt.alias if stmt.alias else func_name
                
                # Check if this is a submodule re-export (e.g., pub use submod::func)
                submodule_name = stmt.path[0]
                if submodule_name in submodules:
                    submodule_info = submodules[submodule_name]
                    if func_name in submodule_info.functions:
                        # Add the function to parent module with export name
                        func_def = submodule_info.functions[func_name]
                        # Mark source module so name resolution can register all functions from it
                        submodule_path = current_module_path + (submodule_name,)
                        object.__setattr__(func_def, '_source_module', submodule_path)
                        functions[export_name] = func_def
                        exports.add(export_name)
                        logger.debug(f"Re-exported {export_name} from submodule {submodule_name} via pub use {submodule_name}::{func_name}")
                    else:
                        logger.warning(f"Function {func_name} not found in submodule {submodule_name} for re-export")
                else:
                    # External module re-export (e.g., pub use std::math::tanh)
                    # or relative submodule re-export (e.g., pub use clamp::clamp in std::math)
                    # Extract the module path (everything except the function name)
                    relative_module_path = tuple(stmt.path[:-1])
                    
                    # If path looks relative (doesn't start with 'std'), make it absolute
                    if relative_module_path and relative_module_path[0] != 'std':
                        # Relative to current module
                        external_module_path = current_module_path + relative_module_path
                    else:
                        # Absolute path
                        external_module_path = relative_module_path
                    
                    # Rust pattern: Skip external module loading if it would cause circular dependency
                    # Check if external_module_path is an ancestor or self
                    if external_module_path == current_module_path or external_module_path in [current_module_path[:i] for i in range(1, len(current_module_path))]:
                        logger.warning(f"Skipping circular pub use: {MODULE_SEPARATOR.join(current_module_path)} -> {MODULE_SEPARATOR.join(external_module_path)}")
                        continue
                    
                    try:
                        # Load the external module
                        external_module_info = self.load_module(external_module_path)
                        
                        # Check if the function exists in the external module
                        if func_name in external_module_info.functions:
                            # Add the function to current module with export name
                            import copy
                            func_def = copy.copy(external_module_info.functions[func_name])
                            # Mark source module so name resolution can register all functions from it
                            # (e.g. std::ml re-exports max_pool from pool_ops; pool_ops also has max_pool1d, etc.)
                            object.__setattr__(func_def, '_source_module', external_module_path)
                            functions[export_name] = func_def
                            exports.add(export_name)
                            logger.debug(f"Re-exported {export_name} from external module {MODULE_SEPARATOR.join(external_module_path)}")
                        else:
                            logger.warning(f"Function {func_name} not found in external module {MODULE_SEPARATOR.join(external_module_path)}")
                    except Exception as e:
                        logger.warning(f"Failed to load external module {MODULE_SEPARATOR.join(external_module_path)} for re-export: {e}")
    
    def get_module(self, module_path: Tuple[str, ...]) -> Optional[ModuleInfo]:
        """Get a loaded module by path"""
        return self.loaded_modules.get(module_path)
    
    def is_module_loaded(self, module_path: Tuple[str, ...]) -> bool:
        """Check if a module is already loaded"""
        return module_path in self.loaded_modules

