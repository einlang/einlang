"""NumPy backend expression visitors. Compatibility shim."""

from .numpy_expressions_support import _safe_oob_ndarray_access
from .numpy_expressions_mixin import ExpressionVisitorMixin

__all__ = [
    "ExpressionVisitorMixin",
    "_safe_oob_ndarray_access",
]
