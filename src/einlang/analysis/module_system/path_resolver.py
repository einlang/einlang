"""
Module Path Resolution

Pure path resolution algorithm for Einlang modules.
Follows Rust's module resolution rules.

Rust Pattern: rustc_resolve::module::PathResolution


This class is stateless and can be shared/reused.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Module system constants 
MODULE_SEPARATOR = "::"
MODULE_FILE_EXTENSION = ".ein"
STD_MODULE_PREFIX = "std"
PYTHON_MODULE_PREFIX = "python"


class ModuleNotFoundError(Exception):
    """Raised when a module path cannot be resolved"""
    pass


class PathResolver:
    """
    Pure path resolution for Einlang modules.
    
    Resolves module paths to filesystem paths following Rust's rules:
    - std::math → stdlib/math.ein or stdlib/math/mod.ein
    - python::numpy → virtual (handled by Python importlib)
    - my_module → crate_root/my_module.ein or crate_root/my_module/mod.ein
    
    This class is stateless and can be shared/reused.
    
    Rust Pattern: rustc_resolve::module::PathResolution
    """
    
    def __init__(self, stdlib_root: Optional[Path] = None):
        """
        Args:
            stdlib_root: Path to standard library root (auto-discovered if None)
        """
        if stdlib_root is None:
            # Auto-discover stdlib root (search up from current directory)
            stdlib_root = self._find_stdlib_root(Path.cwd())
        self.stdlib_root = stdlib_root
        self.crate_root: Optional[Path] = None
        self.external_crates: Dict[str, Path] = {}
    
    def _find_stdlib_root(self, start_path: Path) -> Path:
        """Find stdlib root directory by searching up from start_path"""
        current = start_path.resolve()
        for _ in range(5):
            stdlib_path = current / "stdlib"
            if stdlib_path.exists() and stdlib_path.is_dir():
                return stdlib_path
            parent = current.parent
            if parent == current:  # Reached filesystem root
                break
            current = parent
        # Default fallback
        return Path("stdlib")
    
    def set_crate_root(self, crate_root: Path) -> None:
        """Set the crate root for internal module resolution"""
        if not isinstance(crate_root, Path):
            crate_root = Path(crate_root)
        self.crate_root = crate_root.resolve()
        logger.debug(f"PathResolver: Set crate root to {self.crate_root}")
    
    def register_external_crate(self, crate_name: str, crate_path: Path) -> None:
        """Register an external crate for resolution"""
        if not isinstance(crate_path, Path):
            crate_path = Path(crate_path)
        self.external_crates[crate_name] = crate_path.resolve()
        logger.debug(f"PathResolver: Registered crate '{crate_name}' at {crate_path}")
    
    def resolve(
        self, 
        path_parts: Tuple[str, ...],
        current_module_path: Tuple[str, ...] = ()
    ) -> Path:
        """
        Resolve a module path to a filesystem path.
        
        Rust Pattern: rustc_resolve::path resolution with prefixes.
        
        Args:
            path_parts: Module path as tuple (e.g., ('std', 'math') for 'std::math')
            current_module_path: Current module path for relative resolution (e.g., ('std', 'math'))
        
        Returns:
            Resolved Path to the module file
        
        Raises:
            ModuleNotFoundError: If path cannot be resolved
        
        Examples:
            resolve(('std', 'math')) → Path('stdlib/math.ein')
            resolve(('crate', 'my_module'), ()) → Path('crate_root/my_module.ein')
            resolve(('self', 'helper'), ('std', 'math')) → Path('stdlib/math/helper.ein')
            resolve(('super', 'parent'), ('std', 'math', 'basic')) → Path('stdlib/math.ein')
        """
        if not path_parts:
            raise ModuleNotFoundError("Empty module path")
        
        # Convert tuple to list for compatibility with interface
        path_list = list(path_parts)
        
        # Handle Rust path prefixes (Rust pattern: crate::, self::, super::)
        if path_list[0] == "crate":
            # Absolute path from crate root
            if len(path_list) == 1:
                raise ModuleNotFoundError("crate:: must specify a module name")
            return self._resolve_internal(path_list[1:])
        
        if path_list[0] == "self":
            # Relative path from current module
            if len(path_list) == 1:
                # self:: refers to current module - resolve current module path
                if not current_module_path:
                    raise ModuleNotFoundError("self:: used but no current module path provided")
                # Resolve current module
                return self.resolve(current_module_path, ())
            # self::helper → resolve relative to current module
            combined_path = current_module_path + tuple(path_list[1:])
            return self.resolve(combined_path, ())
        
        if path_list[0] == "super":
            # Relative path from parent module
            if not current_module_path:
                raise ModuleNotFoundError("super:: used but no current module path provided")
            if len(current_module_path) == 0:
                raise ModuleNotFoundError("super:: used at crate root (no parent module)")
            # Get parent module path
            parent_module_path = current_module_path[:-1]
            if len(path_list) == 1:
                # super:: refers to parent module
                return self.resolve(parent_module_path, ())
            # super::helper → resolve relative to parent module
            combined_path = parent_module_path + tuple(path_list[1:])
            return self.resolve(combined_path, ())
        
        # Handle standard library modules
        if path_list[0] == STD_MODULE_PREFIX:
            return self._resolve_stdlib(path_list[1:])
        
        # Handle Python modules (return virtual path)
        if path_list[0] == PYTHON_MODULE_PREFIX:
            return self._resolve_python(path_list)
        
        # Handle external crates
        if path_list[0] in self.external_crates:
            return self._resolve_external_crate(path_list)
        
        # Handle internal crate modules (relative to current module if provided)
        if current_module_path:
            # Try resolving relative to current module first
            try:
                combined_path = current_module_path + tuple(path_list)
                return self.resolve(combined_path, ())
            except ModuleNotFoundError:
                pass  # Fall through to absolute resolution
        
        # Handle internal crate modules (absolute)
        return self._resolve_internal(path_list)
    
    def _resolve_stdlib(self, path_parts: List[str]) -> Path:
        """
        Resolve standard library module path.
        
        std::math → stdlib/math.ein or stdlib/math/mod.ein
        std::math::abs → stdlib/math/abs.ein (submodule file)
        """
        if not path_parts:
            raise ModuleNotFoundError(
                f"Invalid std module path: std:: (must specify module name)"
            )
        
        # Use Path.joinpath for proper filesystem path joining (not MODULE_SEPARATOR)
        # path_parts is like ['math', 'basic'] or ['math', 'abs'] - join with / for filesystem
        path_obj = self.stdlib_root
        for part in path_parts:
            path_obj = path_obj / part
        
        # Try single file first (handles submodule files like stdlib/math/abs.ein)
        # This handles cases where the last component is a function name in a submodule file
        single_file = path_obj.with_suffix(MODULE_FILE_EXTENSION)
        if single_file.exists():
            return single_file
        
        # Try directory with mod.ein: stdlib/math/basic/mod.ein
        dir_mod_file = path_obj / f"mod{MODULE_FILE_EXTENSION}"
        if dir_mod_file.exists():
            return dir_mod_file
        
        # If path has multiple parts, try treating last part as submodule file
        # Example: std::math::abs where abs.ein is a file in math/ directory
        if len(path_parts) > 1:
            # Try: stdlib/math/abs.ein (where abs is the last part)
            parent_path = self.stdlib_root
            for part in path_parts[:-1]:
                parent_path = parent_path / part
            submodule_file = (parent_path / path_parts[-1]).with_suffix(MODULE_FILE_EXTENSION)
            if submodule_file.exists():
                return submodule_file
        
        raise ModuleNotFoundError(
            f"Standard library module '{MODULE_SEPARATOR.join(['std'] + path_parts)}' not found. "
            f"Searched: {single_file}, {dir_mod_file}"
            + (f", {submodule_file}" if len(path_parts) > 1 else "")
        )
    
    def _resolve_python(self, path_parts: List[str]) -> Path:
        """
        Resolve Python module path (returns virtual path).
        
        python::math → Path('python::math') (virtual)
        
        Actual Python module loading is handled by the loader.
        """
        if len(path_parts) < 2:
            raise ModuleNotFoundError(
                "Cannot import 'python' directly. "
                "Use specific Python modules like 'python::math'"
            )
        
        # Return virtual path - actual loading handled elsewhere
        module_path = MODULE_SEPARATOR.join(path_parts)
        return Path(module_path)  # Virtual path like 'python::math'
    
    def _resolve_external_crate(self, path_parts: List[str]) -> Path:
        """
        Resolve external crate module path.
        
        mycrate::utils → external_crates['mycrate']/utils.ein
        """
        crate_name = path_parts[0]
        crate_root = self.external_crates[crate_name]
        remaining_path = path_parts[1:]
        
        if not remaining_path:
            # Reference to crate root itself
            lib_file = crate_root / f"lib{MODULE_FILE_EXTENSION}"
            if lib_file.exists():
                return lib_file
            raise ModuleNotFoundError(
                f"External crate '{crate_name}' has no lib{MODULE_FILE_EXTENSION}"
            )
        
        # Try single file: mycrate/utils.ein (use Path.joinpath for proper filesystem paths)
        path_obj = crate_root
        for part in remaining_path:
            path_obj = path_obj / part
        single_file = path_obj.with_suffix(MODULE_FILE_EXTENSION)
        if single_file.exists():
            return single_file
        
        # Try directory: mycrate/utils/mod.ein
        dir_mod_file = path_obj / f"mod{MODULE_FILE_EXTENSION}"
        if dir_mod_file.exists():
            return dir_mod_file
        
        raise ModuleNotFoundError(
            f"Module '{MODULE_SEPARATOR.join(path_parts)}' not found in external crate '{crate_name}'"
        )
    
    def _resolve_internal(self, path_parts: List[str]) -> Path:
        """
        Resolve internal crate module path.
        
        my_module → crate_root/my_module.ein or crate_root/my_module/mod.ein
        """
        if not self.crate_root:
            raise ModuleNotFoundError(
                f"Cannot resolve internal module '{MODULE_SEPARATOR.join(path_parts)}': "
                f"no crate root set. Call set_crate_root() first."
            )
        
        # Try single file: crate_root/my_module.ein (use Path.joinpath for proper filesystem paths)
        path_obj = self.crate_root
        for part in path_parts:
            path_obj = path_obj / part
        single_file = path_obj.with_suffix(MODULE_FILE_EXTENSION)
        if single_file.exists():
            return single_file
        
        # Try directory: crate_root/my_module/mod.ein
        dir_mod_file = path_obj / f"mod{MODULE_FILE_EXTENSION}"
        if dir_mod_file.exists():
            return dir_mod_file
        
        raise ModuleNotFoundError(
            f"Internal module '{MODULE_SEPARATOR.join(path_parts)}' not found. "
            f"Searched: {single_file}, {dir_mod_file}"
        )
    
    def is_python_module(self, path_parts: Tuple[str, ...]) -> bool:
        """Check if path refers to a Python module"""
        return path_parts and path_parts[0] == PYTHON_MODULE_PREFIX
    
    def is_stdlib_module(self, path_parts: Tuple[str, ...]) -> bool:
        """Check if path refers to a stdlib module"""
        return path_parts and path_parts[0] == STD_MODULE_PREFIX
    
    def is_external_crate(self, path_parts: Tuple[str, ...]) -> bool:
        """Check if path refers to an external crate"""
        return path_parts and path_parts[0] in self.external_crates

