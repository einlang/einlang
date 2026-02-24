"""Module system: path resolution, symbol linking, module loading."""

from .path_resolver import PathResolver, ModuleNotFoundError
from .module_info import ModuleInfo, ModuleDeclaration
from .symbol_linker import SymbolLinker, ResolvedImport, SymbolNotFoundError
from .module_loader import ModuleLoader

# Import old ModuleSystem for backward compatibility
import sys
from pathlib import Path
_parent_module = Path(__file__).parent.parent
_old_module_system_path = _parent_module / "module_system.py"
if _old_module_system_path.exists():
    # Import from parent directory
    import importlib.util
    spec = importlib.util.spec_from_file_location("einlang.analysis.module_system_old", _old_module_system_path)
    old_module = importlib.util.module_from_spec(spec)
    sys.modules["einlang.analysis.module_system_old"] = old_module
    spec.loader.exec_module(old_module)
    ModuleSystem = old_module.ModuleSystem
else:
    # Fallback: create a dummy class
    class ModuleSystem:
        """Placeholder for old ModuleSystem (should be migrated to new components)"""
        pass

__all__ = [
    'PathResolver',
    'ModuleNotFoundError',
    'ModuleInfo',
    'ModuleDeclaration',
    'SymbolLinker',
    'ResolvedImport',
    'SymbolNotFoundError',
    'ModuleLoader',
    'ModuleSystem',  # For backward compatibility
]

