"""NumPy backend Einstein execution. Compatibility shim."""

from ..shared.recurrence_analysis import (
    _BodyReferencesDefidVisitor,
    _collect_lhs_read_index_lists,
    _index_expr_is_loop_var,
    _index_expr_is_loop_var_or_offset,
    _loop_dims_from_clause_indices,
    _recurrence_dims,
    _recurrence_dims_for_hybrid,
    _reduction_var_bounded_by_loop_var,
)
from .numpy_einstein_mixin_setup import EinsteinExecutionSetupMixin
from .numpy_einstein_mixin_recurrence import EinsteinExecutionRecurrenceMixin
from .numpy_einstein_mixin_clause import EinsteinExecutionClauseMixin

class EinsteinExecutionMixin(
    EinsteinExecutionSetupMixin,
    EinsteinExecutionRecurrenceMixin,
    EinsteinExecutionClauseMixin,
):
    pass

__all__ = [
    "EinsteinExecutionMixin",
    "_BodyReferencesDefidVisitor",
    "_collect_lhs_read_index_lists",
    "_index_expr_is_loop_var",
    "_index_expr_is_loop_var_or_offset",
    "_loop_dims_from_clause_indices",
    "_recurrence_dims",
    "_recurrence_dims_for_hybrid",
    "_reduction_var_bounded_by_loop_var",
]
