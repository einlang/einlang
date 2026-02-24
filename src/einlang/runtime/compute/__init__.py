"""
Runtime Compute Module

Pure execution patterns for lowered iteration structures.
"""

from .lowered_execution import (
    execute_lowered_loops,
    execute_lowered_bindings,
    check_lowered_guards,
    execute_full_lowered_iteration,
    execute_reduction_with_loops,
    has_dynamic_bounds,
    count_loop_iterations
)

__all__ = [
    'execute_lowered_loops',
    'execute_lowered_bindings',
    'check_lowered_guards',
    'execute_full_lowered_iteration',
    'execute_reduction_with_loops',
    'has_dynamic_bounds',
    'count_loop_iterations'
]

