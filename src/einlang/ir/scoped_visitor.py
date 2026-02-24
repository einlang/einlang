"""
Scoped IR Visitor - aligned pattern for scope management in IR passes.

This provides automatic scope tracking for function parameters and local variables,
eliminating the need for manual var_definitions dict building.

ScopedASTVisitor automatically tracks variables in scopes.
Alignment: ScopedIRVisitor does the same for IR nodes.
"""

from typing import TypeVar, Generic, Dict, List, Optional, Any
from contextlib import contextmanager
from .nodes import IRVisitor
from ..shared.defid import DefId

T = TypeVar('T')


class ScopedIRVisitor(IRVisitor[T], Generic[T]):
    """
    IR visitor with automatic scope management (aligned pattern).

    Scope key is DefId — name-based lookup is deliberately not supported.
    """

    def __init__(self):
        self._scope_stack: List[Dict[DefId, Any]] = [{}]

    # =========================================================================
    # Scope Management API
    # =========================================================================

    @contextmanager
    def scope(self):
        """Context manager for entering/exiting a scope (RAII pattern)."""
        self._scope_stack.append({})
        try:
            yield
        finally:
            if len(self._scope_stack) > 1:
                self._scope_stack.pop()

    def set_var(self, defid: DefId, value: Any) -> None:
        """Set variable in current scope."""
        self._scope_stack[-1][defid] = value

    def get_var(self, defid: DefId) -> Optional[Any]:
        """Get variable from current scope or any parent scope (shadowing)."""
        for scope in reversed(self._scope_stack):
            if defid in scope:
                return scope[defid]
        return None

    def get_var_current_scope(self, defid: DefId) -> Optional[Any]:
        """Get variable from current scope only (no parent lookup)."""
        return self._scope_stack[-1].get(defid)

    def get_all_vars(self) -> Dict[DefId, Any]:
        """Get all variables from all scopes (innermost scope wins)."""
        result: Dict[DefId, Any] = {}
        for scope in self._scope_stack:
            result.update(scope)
        return result
