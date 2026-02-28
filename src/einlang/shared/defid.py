"""
DefId System

Rust Pattern: rustc_hir::def_id::DefId, rustc_resolve::Resolver

Design (learned from Rust):
- DefId = (krate, index). Items (fn, const, module, builtin) get DefId and are
  stored in symbol table keyed by (module_path, name, def_type) so same name
  for different kinds never collides. No global cache by name alone.
- Locals (params, lets) get DefId via allocate_for_local only: just allocate;
  no symbol table, no def_registry. Caller attaches defid to the node.
  See DEFID_SYSTEM_DESIGN.md "Learning from Rust: Symbol Table and DefId Allocation".
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DefType(Enum):
    """
    Definition type (Rust pattern: rustc_hir::def::DefKind).
    
    Rust Pattern: rustc_hir::def::DefKind enum
    """
    FUNCTION = "function"
    CONSTANT = "constant"
    VARIABLE = "variable"
    PARAMETER = "parameter"
    LAMBDA = "lambda"
    TYPE = "type"
    MODULE = "module"
    BUILTIN = "builtin"


@dataclass(frozen=True)
class DefId:
    """
    Definition Identifier (Rust pattern: rustc_hir::def_id::DefId).
    
    Rust Pattern: rustc_hir::def_id::DefId
    
    Implementation Alignment: Follows Rust's DefId structure:
    - krate: Crate number (0 = current compilation unit)
    - index: Sequential index within crate
    - Immutable (frozen dataclass)
    - Deterministic allocation order
    
    Reference: `rustc_hir::def_id::DefId` structure
    """
    krate: int  # Crate number (0 = current compilation unit)
    index: int  # Sequential index within crate
    
    def __str__(self) -> str:
        """Format as krate:index (Rust pattern)"""
        return f"{self.krate}:{self.index}"

    def __deepcopy__(self, memo: Any) -> "DefId":
        return self


# ---------------------------------------------------------------------------
# DefId for all items including variables
# ---------------------------------------------------------------------------
# All definitions get a DefId: functions, constants, modules, builtins,
# and variables (module-level and local: params, lets, einstein, etc.).
# Local vars' DefIds are only used in local scopes (attached to identifier
# use sites); we do not put local vars in the symbol table (symbol table
# has builtins, module functions, constants, modules only).

# Backward compatibility: runtime may still use Local type alias until it switches to DefId
Local = int  # Deprecated: variable identity is DefId; kept for runtime/environment during migration


# Crate ids: user code is crate 0, builtins are a separate crate (Rust pattern)
_LOCAL_CRATE = 0
BUILTIN_CRATE = 1
_BUILTIN_CRATE = BUILTIN_CRATE
RUNTIME_CRATE = -1

FIXED_BUILTIN_ORDER = (
    "assert", "print", "len", "typeof", "array_append", "shape", "sum", "max", "min",
)


def fixed_builtin_defid(name: str) -> Optional[DefId]:
    """Return fixed DefId for a builtin name, or None if not in fixed set."""
    if name in FIXED_BUILTIN_ORDER:
        return DefId(krate=_BUILTIN_CRATE, index=FIXED_BUILTIN_ORDER.index(name))
    return None


def assert_defid(value: Any, *, allow_none: bool = True) -> None:
    """Raise if value is not DefId (and not None when allow_none). Compiler/IR must use DefId only, not tuple."""
    if value is None and allow_none:
        return
    if not isinstance(value, DefId):
        raise TypeError(f"defid must be DefId or None, got {type(value).__name__}: {value}")


class Resolver:
    """
    Name resolver and DefId allocator (Rust naming: rustc_resolve::Resolver).
    
    Rust Pattern: rustc_resolve::Resolver with Definitions table
    
    Implementation Alignment: Follows Rust's `rustc_resolve::Resolver` implementation:
    - Single allocation point (in name resolution)
    - Symbol table: (module_path, name, def_type) → DefId (no global name collision)
    - Registry: DefId → (DefType, definition)
    - Sequential allocation starting after builtins
    
    DefId guarantee (crate 0): Indices are globally increasing and never conflict.
    Both allocate_for_item (functions, etc.) and allocate_for_local use the same
    _local_next_index counter, so every new DefId in crate 0 gets a strictly
    greater index than any previous one. No two distinct definitions receive
    the same DefId. (CONSTANT/MODULE reuse by symbol_key is intentional.)
    
    Name map (_symbol_table, _alias_table, _def_registry) is only used by pre-name-resolution
    and name-resolution passes.
    
    Reference: `rustc_resolve::Resolver` with `Definitions` table
    
    Note: Using Rust naming - "Resolver" instead of "DefIdAllocator" to match rustc
    """
    
    def __init__(self, tcx: Any):
        self._tcx = tcx
        self._builtin_next_index = 0
        self._local_next_index = 0

    @property
    def _def_registry(self) -> Dict[DefId, Tuple[DefType, Any]]:
        return self._tcx.def_registry

    @property
    def _symbol_table(self) -> Dict[Tuple[Tuple[str, ...], str, DefType], DefId]:
        return self._tcx.symbol_table

    @property
    def _alias_table(self) -> Dict[str, Tuple[str, ...]]:
        return self._tcx.alias_table

    def allocate_for_item(
        self,
        module_path: Tuple[str, ...],
        name: str,
        definition: Any,
        def_type: DefType,
    ) -> DefId:
        """
        Allocate DefId for an item (function, constant, module, builtin).
        Writes into tcx.def_registry and tcx.symbol_table (scoped to compilation).
        For CONSTANT/MODULE, reuses existing if present.
        """
        symbol_key = (module_path, name, def_type)
        reg = self._tcx.def_registry
        sym = self._tcx.symbol_table
        if def_type == DefType.BUILTIN:
            if name in FIXED_BUILTIN_ORDER:
                idx = FIXED_BUILTIN_ORDER.index(name)
                defid = DefId(krate=_BUILTIN_CRATE, index=idx)
            else:
                idx = self._builtin_next_index
                defid = DefId(krate=_BUILTIN_CRATE, index=idx)
                self._builtin_next_index = idx + 1
        else:
            if def_type in (DefType.CONSTANT, DefType.MODULE) and symbol_key in sym:
                return sym[symbol_key]
            idx = self._local_next_index
            defid = DefId(krate=_LOCAL_CRATE, index=idx)
            self._local_next_index = idx + 1
            assert self._local_next_index > idx
        sym[symbol_key] = defid
        reg[defid] = (def_type, definition)
        if definition is not None:
            if hasattr(definition, '_defid'):
                definition._defid = defid
            elif hasattr(definition, 'defid'):
                object.__setattr__(definition, 'defid', defid)
        return defid

    def allocate_for_local(self) -> DefId:
        """
        Allocate DefId for a local (variable, parameter, lambda, etc.).
        No symbol table, no def_registry. Caller attaches defid to the definition. query() will not find locals.
        Guarantee: returned index is strictly greater than any previously returned local index (monotonically increasing).
        """
        idx = self._local_next_index
        defid = DefId(krate=_LOCAL_CRATE, index=idx)
        self._local_next_index = idx + 1
        assert self._local_next_index > idx
        return defid

    def query(self, defid: DefId) -> Optional[Tuple[DefType, Any]]:
        """Look up definition by DefId from tcx-scoped registry."""
        return self._tcx.def_registry.get(defid)

    def get_defid(self, module_path: Tuple[str, ...], name: str, def_type: DefType) -> Optional[DefId]:
        """Lookup DefId from tcx-scoped symbol table."""
        symbol_key = (module_path, name, def_type)
        return self._tcx.symbol_table.get(symbol_key)

    def register_item(self, module_path: Tuple[str, ...], name: str, def_type: DefType, defid: DefId) -> None:
        """Register an existing DefId in tcx-scoped symbol table."""
        self._tcx.symbol_table[(module_path, name, def_type)] = defid

    def register_alias(self, alias_name: str, module_path: Tuple[str, ...]) -> None:
        """Register module alias in tcx-scoped alias table."""
        self._tcx.alias_table[alias_name] = module_path

    def lookup_alias(self, alias_name: str) -> Optional[Tuple[str, ...]]:
        """Lookup module alias from tcx-scoped alias table."""
        return self._tcx.alias_table.get(alias_name)
    

