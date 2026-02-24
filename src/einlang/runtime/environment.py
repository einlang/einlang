"""
Execution Environment

Single scope stack for all variables (global and local). Every binding is keyed by DefId.
Function call = push scope; return = pop scope. Lookup walks stack from innermost to outermost.

Rule: After name resolution, no code shall rely on variable/index name for binding or lookup;
DefId is the only semantic identifier. Name is for debug/errors only.
"""

from typing import Dict, List, Any, Optional, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from ..shared.defid import DefId


@dataclass(frozen=True)
class FunctionValue:
    """First-class function value (DefId + optional closure env)."""
    defid: DefId
    closure_env: Optional['ExecutionEnvironment'] = None


class ExecutionEnvironment:
    """
    Single scope stack: all variables (global and local) keyed by DefId.
    - enter_scope(): push new scope (block or function)
    - exit_scope(): pop
    - set_value(defid, value, name=None): store in current (top) scope
    - get_value(defid): lookup from top scope outward
    """
    _scope_stack: List[Dict[DefId, Any]]
    _defid_names: Dict[DefId, str]

    def __init__(self):
        self._scope_stack = [{}]  # Initial scope (global/top-level)
        self._defid_names = {}    # DefId → human-readable name (debug only)

    def enter_scope(self) -> None:
        """Push a new scope (e.g. on function entry or block)."""
        self._scope_stack.append({})

    def exit_scope(self) -> None:
        """Pop current scope."""
        if not self._scope_stack:
            raise RuntimeError("Cannot exit scope: no active scope")
        self._scope_stack.pop()

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Context manager: enter scope on enter, exit scope on exit (always, including on exception)."""
        self.enter_scope()
        try:
            yield
        finally:
            self.exit_scope()

    def set_value(self, defid: DefId, value: Any, name: str = None) -> None:
        """Store value for DefId in current (top) scope. Optional name for debug."""
        if defid is None:
            raise RuntimeError("set_value: defid must not be None.")
        if not self._scope_stack:
            raise RuntimeError("Cannot set value: no active scope")
        self._scope_stack[-1][defid] = value
        if name is not None:
            self._defid_names[defid] = name

    def get_value(self, defid: DefId) -> Optional[Any]:
        """Lookup DefId from current scope outward (innermost to outermost)."""
        if defid is None:
            raise RuntimeError("get_value: defid must not be None.")
        for scope in reversed(self._scope_stack):
            if defid in scope:
                return scope[defid]
        return None

    def get_current_scope(self) -> Dict[DefId, Any]:
        """Return the current (top) scope. For collecting outputs before exit_scope."""
        if not self._scope_stack:
            return {}
        return self._scope_stack[-1]

    def has_scope(self) -> bool:
        """True if there is at least one scope."""
        return len(self._scope_stack) > 0
