"""Compiler-generated IndexVarIR names: leading ``$`` is not valid in user ``NAME`` (Lark CNAME).

Allocate with ``Resolver.allocate_internal_iv_name()`` for a monotonic ``$0``, ``$1``, … sequence scoped
to the compilation (same ``Resolver`` / tcx as local DefIds).
"""

from typing import Optional


def is_compiler_internal_iv_name(name: Optional[str]) -> bool:
    """True if *name* matches ``allocate_internal_iv_name`` output (``$`` + decimal digits)."""
    if not name or name[0] != "$":
        return False
    return len(name) > 1 and name[1:].isdigit()
