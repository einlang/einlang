"""
Module System Types

Module system types - shared between passes and runtime.
These types are pure data structures with no business logic.

Rust Pattern: rustc_resolve::module::Module

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Import AST types (will be available at runtime)
try:
    from ...shared.nodes import FunctionDefinition, Program
except ImportError:
    # Fallback for type hints only
    FunctionDefinition = Any
    Program = Any


@dataclass
class ModuleDeclaration:
    """
    Represents a 'mod name;' or 'pub mod name;' declaration.
    
    Rust Pattern: rustc_resolve::module::ModuleDeclaration
    """
    name: str
    is_public: bool = False
    location: Optional[Tuple[int, int]] = None  # (line, column)
    
    def __str__(self) -> str:
        """Human-readable representation"""
        visibility = "pub " if self.is_public else ""
        return f"{visibility}mod {self.name};"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"ModuleDeclaration(name={self.name!r}, is_public={self.is_public}, location={self.location})"


@dataclass
class ModuleInfo:
    """
    Information about a loaded module.
    
    Rust Pattern: rustc_resolve::module::Module
    
    This structure represents a complete module with its hierarchy:
    - name: Module path as tuple (e.g., ('std', 'math'))
    - path: File path (None for inline modules)
    - program: Parsed AST
    - declarations: mod declarations found in the module
    - functions: All functions (including re-exports from submodules)
    - exports: Public function names (what can be imported)
    - submodules: Declared submodules (for pub use processing)
    - lazy_loader: For Python modules and other lazy-loaded modules
    """
    name: Tuple[str, ...]  # Module path as tuple (e.g., ('std', 'math')) - Rust-style structured data
    path: Optional[Path]  # None for inline modules (like Rust)
    program: Program
    declarations: List[ModuleDeclaration]
    functions: Dict[str, FunctionDefinition]  # All functions (including re-exports)
    exports: Set[str]  # Public function names
    submodules: Dict[str, 'ModuleInfo']  # Declared submodules
    lazy_loader: Optional[Any] = None  # For Python modules and other lazy-loaded modules
    source_code: Optional[str] = None  # Set when loaded from overlay (avoids path.read_text() I/O)
    
    def __str__(self) -> str:
        """Human-readable representation"""
        name_str = '::'.join(self.name)  # Format tuple as string for display
        return f"Module({name_str}, {len(self.functions)} functions, {len(self.submodules)} submodules)"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return (f"ModuleInfo(name={self.name!r}, path={self.path}, "
                f"functions={list(self.functions.keys())}, "
                f"exports={self.exports}, "
                f"submodules={list(self.submodules.keys())})")

