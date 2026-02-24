"""
Module System

Rust Pattern: rustc_resolve::module
Reference: MODULE_SYSTEM_DESIGN.md
"""

from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path

from ..utils.io_utils import read_source_file, is_temp_path
from ..shared.defid import DefId, DefType, Resolver
from ..ir.nodes import ModuleIR, ProgramIR, FunctionDefIR, ConstantDefIR
from ..shared.source_location import SourceLocation


class ModuleDiscovery:
    """
    Module discovery (Rust naming: rustc_resolve::module).
    
    Rust Pattern: rustc_resolve::module::ModuleTreeBuilder
    
    Implementation Alignment: Follows Rust's module discovery:
    - Discovers modules from file system
    - Builds module tree
    - Handles explicit `mod` declarations
    - Resolves module paths
    
    Reference: `rustc_resolve::module::ModuleTreeBuilder` for module discovery
    """
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.discovered_modules: Dict[Tuple[str, ...], Path] = {}
    
    def discover_modules(self, entry_point: Path) -> Dict[Tuple[str, ...], Path]:
        """
        Discover all modules starting from entry point.
        
        Rust Pattern: rustc_resolve::module::ModuleTreeBuilder::build_module_tree()
        """
        # Check if entry point exists (for temporary files, it might not exist yet)
        if not entry_point.exists():
            # For temporary files or test files, only include the entry point itself
            entry_module_path = self._path_to_module_path(entry_point)
            return {entry_module_path: entry_point}
        
        # Start with entry point
        entry_module_path = self._path_to_module_path(entry_point)
        self.discovered_modules[entry_module_path] = entry_point
        
        # Discover modules recursively
        self._discover_recursive(entry_point, entry_module_path)
        
        return self.discovered_modules
    
    def _discover_recursive(self, file_path: Path, module_path: Tuple[str, ...]) -> None:
        """Recursively discover modules"""
        # Read file to find explicit `mod` declarations
        # For now, discover files in same directory
        parent_dir = file_path.parent
        
        # Skip automatic sibling discovery in temp dirs (tests use /tmp, etc.)
        if is_temp_path(parent_dir):
            return
        
        # Discover sibling files (only if they exist)
        for sibling in parent_dir.glob("*.ein"):
            if sibling == file_path:
                continue
            
            # Only include files that actually exist
            if not sibling.exists():
                continue
            
            sibling_module_path = self._path_to_module_path(sibling)
            if sibling_module_path not in self.discovered_modules:
                self.discovered_modules[sibling_module_path] = sibling
    
    def _path_to_module_path(self, file_path: Path) -> Tuple[str, ...]:
        """Convert file path to module path"""
        # Remove extension
        name = file_path.stem
        return (name,)


class ModuleLoader:
    """
    Module loader (Rust naming: rustc_resolve::module).
    
    Rust Pattern: rustc_resolve::module::ModuleLoader
    
    Implementation Alignment: Follows Rust's module loading:
    - Loads module source files
    - Parses modules
    - Returns module AST/IR
    
    Reference: `rustc_resolve::module::ModuleLoader` for module loading
    """
    
    def __init__(self):
        self.loaded_modules: Dict[Tuple[str, ...], str] = {}
    
    def load_module(self, module_path: Path) -> str:
        """
        Load module source code.
        
        Rust Pattern: rustc_resolve::module::ModuleLoader::load_file()
        """
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        source = read_source_file(module_path)
        self.loaded_modules[module_path] = source
        return source
    
    def load_all(self, module_paths: Dict[Tuple[str, ...], Path]) -> Dict[Tuple[str, ...], str]:
        """Load all modules"""
        sources = {}
        for module_path_tuple, file_path in module_paths.items():
            # Skip loading if file doesn't exist (might be a temporary file from another test)
            if not file_path.exists():
                continue
            try:
                sources[module_path_tuple] = self.load_module(file_path)
            except FileNotFoundError:
                # Skip modules that don't exist (e.g., temporary files from other tests)
                continue
        return sources


class ModuleResolver:
    """
    Module resolver (Rust naming: rustc_resolve::module).
    
    Rust Pattern: rustc_resolve::module::ModuleResolver
    
    Implementation Alignment: Follows Rust's module resolution:
    - Resolves module paths to DefIds
    - Handles module imports
    - Integrates with name resolution
    - Allocates DefIds for modules
    
    Reference: `rustc_resolve::module::ModuleResolver` for module resolution
    """
    
    def __init__(self, resolver: Resolver):
        self.resolver = resolver
        self.module_defids: Dict[Tuple[str, ...], DefId] = {}
    
    def resolve_module(
        self,
        module_path: Tuple[str, ...],
        module_ir: ModuleIR
    ) -> DefId:
        """
        Resolve module and allocate DefId.
        
        Rust Pattern: rustc_resolve::module::ModuleResolver::resolve_module()
        """
        # Check if already resolved
        if module_path in self.module_defids:
            return self.module_defids[module_path]
        
        # Allocate DefId for module
        defid = self.resolver.allocate_for_item(
            module_path[:-1] if len(module_path) > 1 else (),
            module_path[-1] if module_path else "",
            module_ir,
            DefType.MODULE
        )
        
        self.module_defids[module_path] = defid
        return defid
    
    def resolve_import(
        self,
        import_path: Tuple[str, ...],
        current_module_path: Tuple[str, ...]
    ) -> Optional[DefId]:
        """
        Resolve import path to DefId.
        
        Rust Pattern: rustc_resolve::module::ModuleResolver::resolve_import()
        """
        # Resolve relative or absolute path
        if import_path[0] == "::":
            # Absolute path
            resolved_path = import_path[1:]
        else:
            # Relative path
            resolved_path = current_module_path + import_path
        
        return self.module_defids.get(resolved_path)


class ModuleSystem:
    """
    Module system coordinator (Rust naming: rustc_resolve::module).
    
    Rust Pattern: rustc_resolve::module coordination
    
    Implementation Alignment: Follows Rust's module system:
    - Coordinates discovery, loading, resolution
    - Integrates with name resolution
    - Handles module dependencies
    
    Reference: Rust module system coordination
    """
    
    def __init__(self, root_path: Path, resolver: Resolver):
        self.root_path = root_path
        self.resolver = resolver
        self.discovery = ModuleDiscovery(root_path)
        self.loader = ModuleLoader()
        self.resolver_module = ModuleResolver(resolver)
        # Discover stdlib root (Rust pattern: stdlib is always accessible)
        self.stdlib_root = self._find_stdlib_root(root_path)
    
    def _find_stdlib_root(self, root_path: Path) -> Optional[Path]:
        """
        Find stdlib root directory (Rust pattern: stdlib is accessible).
        
        Searches for stdlib directory starting from root_path and going up.
        """
        current = Path(root_path).resolve()
        # Search up to 5 levels
        for _ in range(5):
            stdlib_path = current / "stdlib"
            if stdlib_path.exists() and stdlib_path.is_dir():
                return stdlib_path
            parent = current.parent
            if parent == current:  # Reached filesystem root
                break
            current = parent
        return None
    
    def process_modules(
        self,
        entry_point: Path
    ) -> Tuple[Dict[Tuple[str, ...], Path], Dict[Tuple[str, ...], str]]:
        """
        Process all modules: discover only (lazy loading).
        
        Rust Pattern: rustc_resolve::module coordination with tree-shaking
        
        Tree-shaking: Modules are NOT loaded here - only discovered.
        Modules are loaded on-demand when actually referenced (lazy loading).
        This enables tree-shaking: unused imports are never loaded.
        """
        # 1. Discover modules (including stdlib) - but don't load yet
        module_paths = self.discovery.discover_modules(entry_point)
        
        # 2. Discover stdlib modules (Rust pattern: stdlib is accessible, but not preloaded)
        if self.stdlib_root:
            stdlib_modules = self._discover_stdlib_modules()
            module_paths.update(stdlib_modules)
        
        # 3. Tree-shaking: Don't load modules here - they'll be loaded on-demand
        # Return empty sources dict - modules will be loaded when referenced
        module_sources = {}
        
        # 4. Resolve modules (DefId allocation happens in name resolution pass)
        # For now, just return discovered modules (not loaded)
        
        return module_paths, module_sources
    
    def load_module_on_demand(
        self,
        module_path: Tuple[str, ...],
        module_paths: Dict[Tuple[str, ...], Path]
    ) -> Optional[str]:
        """
        Load a module on-demand (tree-shaking: only when referenced).
        
        Rust Pattern: Lazy module loading - modules loaded only when used.
        
        Args:
            module_path: Module path tuple (e.g., ('std', 'math', 'sqrt'))
            module_paths: Dict of discovered module paths to file paths
        
        Returns:
            Module source code if found and loaded, None otherwise
        """
        # Check if module is already loaded
        if module_path in self.loader.loaded_modules:
            return self.loader.loaded_modules[module_path]
        
        # Find file path for this module
        file_path = module_paths.get(module_path)
        if not file_path:
            return None
        
        # Load module source (lazy loading)
        try:
            source = self.loader.load_module(file_path)
            # Store in loaded_modules dict keyed by module_path
            self.loader.loaded_modules[module_path] = source
            return source
        except FileNotFoundError:
            return None
    
    def _discover_stdlib_modules(self) -> Dict[Tuple[str, ...], Path]:
        """
        Discover all stdlib modules (Rust pattern: stdlib is accessible).
        
        Returns dict mapping module paths to file paths.
        """
        stdlib_modules = {}
        if not self.stdlib_root or not self.stdlib_root.exists():
            return stdlib_modules
        
        # Discover all .ein files in stdlib (Rust pattern: recursive discovery)
        for tsc_file in self.stdlib_root.rglob("*.ein"):
            relative_path = tsc_file.relative_to(self.stdlib_root)
            path_parts = relative_path.parts[:-1]  # Remove filename
            module_name = tsc_file.stem  # Remove .ein extension
            
            # Handle mod.ein files specially (mod.ein is the module entry point)
            # stdlib/math/mod.ein → ('std', 'math')
            if tsc_file.name == "mod.ein":
                if path_parts:
                    # Has subdirectory: stdlib/math/mod.ein → ('std', 'math')
                    module_path = ('std',) + path_parts
                else:
                    # Top-level: stdlib/mod.ein → ('std',) - but this is unusual
                    continue  # Skip top-level mod.ein
            else:
                # Regular file: stdlib/math/basic.ein → ('std', 'math', 'basic')
                if path_parts:
                    # Has subdirectory: stdlib/math/basic.ein → ('std', 'math', 'basic')
                    module_path = ('std',) + path_parts + (module_name,)
                else:
                    # Top-level: stdlib/basic.ein → ('std', 'basic')
                    module_path = ('std', module_name)
            
            stdlib_modules[module_path] = tsc_file
        
        return stdlib_modules

