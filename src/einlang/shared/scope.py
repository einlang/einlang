"""
Scope resolution — scope resolution system.


    Value = (type, value, defid) e.g. ('function', func_def, defid), ('alias', path, defid).
Same idea — stack of scopes, each scope is name → Binding. define = set_var, lookup = get_var.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple

from .defid import DefId


# -----------------------------------------------------------------------------
# Errors (caller may raise when duplicate item in same scope)
# -----------------------------------------------------------------------------


class ScopeRedefinitionError(ValueError):
    """Raised when a name is already defined in this scope and redefinition is not allowed."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"redefinition of '{name}' in same scope")


# -----------------------------------------------------------------------------
# Scope kind (categorizes scope types)
# -----------------------------------------------------------------------------


class ScopeKind(Enum):
    MODULE = "module"
    FUNCTION = "function"
    BLOCK = "block"
    EINSTEIN = "einstein"
    REDUCTION = "reduction"
    CLOSURE = "closure"
    LOOP = "loop"


# -----------------------------------------------------------------------------
# Binding kind (first element of tuple — 'function', 'lambda', 'alias')
# -----------------------------------------------------------------------------


class BindingType(Enum):
    FUNCTION = "function"
    CONSTANT = "constant"
    VARIABLE = "variable"
    LAMBDA = "lambda"
    PARAMETER = "parameter"
    MODULE = "module"


# -----------------------------------------------------------------------------
# Binding (tuple (type, value, defid); structured)
# -----------------------------------------------------------------------------


@dataclass
class Binding:
    """One name binding (value in scope dict)."""
    name: str
    binding_type: BindingType
    definition: Any
    defid: Optional[DefId]
    scope: Scope
    module_path: Optional[Tuple[str, ...]] = None


# -----------------------------------------------------------------------------
# Scope (one dict in _scope_stack — name → value)
# -----------------------------------------------------------------------------


@dataclass
class Scope:
    """
    One scope level (one Dict[str, T] in _scope_stack).
    Single map: name → Binding. define() overwrites (shadow); lookup() inner→outer.
    """

    parent: Optional[Scope]
    kind: ScopeKind
    _bindings: Dict[str, Binding] = field(default_factory=dict)

    def lookup(self, name: str) -> Optional[Binding]:
        """Get variable from scope chain, innermost to outermost (get_var)."""
        if name in self._bindings:
            return self._bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None

    def lookup_all(self, name: str) -> List[Binding]:
        """All bindings for name along the scope chain, innermost first."""
        out: List[Binding] = []
        if name in self._bindings:
            out.append(self._bindings[name])
        if self.parent is not None:
            out.extend(self.parent.lookup_all(name))
        return out

    def defined_in_this_scope(self, name: str) -> bool:
        """True if name is in this scope’s map (name in _scope_stack[-1])."""
        return name in self._bindings

    def _get_binding_in_this_scope(self, name: str) -> Optional[Binding]:
        """Binding for name in this scope only."""
        return self._bindings.get(name)

    def define(self, name: str, binding: Binding) -> None:
        """Set variable in current scope; overwrites if present (set_var)."""
        self._bindings[name] = binding


# -----------------------------------------------------------------------------
# Scope manager (_scope_stack, scope() push/pop, set_var/get_var on current)
# -----------------------------------------------------------------------------


class ScopeManager:
    """
    Scope stack (_scope_stack). enter_scope = push, exit_scope = pop.
    lookup/define operate on current (innermost) scope.
    """

    def __init__(self) -> None:
        self._stack: List[Scope] = []
        self._current: Optional[Scope] = None

    def enter_scope(self, kind: ScopeKind, parent: Optional[Scope] = None) -> Scope:
        """Push a new scope (scope() appends {})."""
        p = parent if parent is not None else self._current
        scope = Scope(parent=p, kind=kind)
        self._stack.append(scope)
        self._current = scope
        return scope

    def exit_scope(self) -> None:
        """Pop current scope (scope() exit pops)."""
        if not self._stack:
            raise RuntimeError("Cannot exit scope: no active scope")
        self._stack.pop()
        self._current = self._stack[-1] if self._stack else None

    @contextmanager
    def scope(self, kind: ScopeKind, parent: Optional[Scope] = None) -> Generator[Scope, None, None]:
        """Context manager: enter on __enter__, exit on __exit__ (with self.scope())."""
        s = self.enter_scope(kind, parent)
        try:
            yield s
        finally:
            self.exit_scope()

    def current_scope(self) -> Optional[Scope]:
        """Innermost scope (_scope_stack[-1])."""
        return self._current

    def get_enclosing_scope(self, kind: ScopeKind) -> Optional[Scope]:
        """Nearest enclosing scope of the given kind ."""
        for s in reversed(self._stack):
            if s.kind == kind:
                return s
        return None

    def lookup(self, name: str) -> Optional[Binding]:
        """Resolve name in current scope chain (get_var)."""
        if self._current is None:
            return None
        return self._current.lookup(name)
