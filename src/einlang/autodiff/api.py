"""Stable runtime autodiff API for backends and other runtime consumers."""

from .intrinsics import (
    AutodiffBuiltinKind,
    autodiff_builtin_defid,
    autodiff_builtin_kind,
    autodiff_builtin_name,
)
from ..autodiff_runtime import (
    jacobian_value_for_defids,
    symbolic_jacobian_relation,
    symbolic_tangent_for_defid,
    tangent_value_for_defid,
)

__all__ = [
    "AutodiffBuiltinKind",
    "autodiff_builtin_defid",
    "autodiff_builtin_kind",
    "autodiff_builtin_name",
    "tangent_value_for_defid",
    "jacobian_value_for_defids",
    "symbolic_tangent_for_defid",
    "symbolic_jacobian_relation",
]
