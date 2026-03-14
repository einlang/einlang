"""
Optional attribute access for IR/AST nodes.

Many passes and backends handle values that can be different node types (e.g. an
index expression can be IdentifierIR, IndexVarIR, or BinaryOpIR). Only some have
attributes like .defid or .name. Using direct attribute access raises when the
node does not have the attribute.

Use the helpers here instead of getattr(..., "attr", default) at call sites where
the node type is not statically known, so we have a single place to document and
adjust behavior. The trust_ir_nodes script does not replace these.
"""

from typing import Any, Optional


def opt_defid(node: Any) -> Optional[Any]:
    """Return node.defid if present, else None. Use for any node that might be AST or IR."""
    return getattr(node, "defid", None)


def opt_name(node: Any, default: str = "") -> str:
    """Return node.name if present, else default. Use when node may be ArrayLiteralIR or other type without .name."""
    return getattr(node, "name", None) or default


def opt_attr(node: Any, attr: str, default: Any = None) -> Any:
    """Return getattr(node, attr, default). Use for one-off optional attributes."""
    return getattr(node, attr, default)
