"""NumPy backend, env is DefId-keyed."""

from typing import Any
from ..backends.base import Backend
from ..ir.nodes import IRVisitor

from .numpy_core import CoreExecutionMixin
from .numpy_expressions import ExpressionVisitorMixin
from .numpy_einstein import EinsteinExecutionMixin
from .numpy_helpers import (
    builtin_assert,
    builtin_print,
    builtin_len,
    builtin_typeof,
    builtin_array_append,
    builtin_shape,
    builtin_sum,
    builtin_max,
    builtin_min,
)


class NumPyBackend(
    CoreExecutionMixin,
    ExpressionVisitorMixin,
    EinsteinExecutionMixin,
    Backend,
    IRVisitor[Any],
):
    """
    NumPy backend (facade). Composes:
    - CoreExecutionMixin: execute, env, _call_function, resolve, program/module/function_def/constant_def
    - ExpressionVisitorMixin: expression visit_* (literal through builtin_call, match, where, arrow, pipeline)
    - EinsteinExecutionMixin: lowered einstein, reduction, comprehension
    Env is DefId-keyed.
    """

    def __init__(self):
        CoreExecutionMixin.__init__(self)
