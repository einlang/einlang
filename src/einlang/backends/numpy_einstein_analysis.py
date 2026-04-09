"""NumPy backend Einstein helpers."""

"""NumPy backend Einstein execution: variable decl, lowered einstein/clause; env only."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ir.nodes import (
    LiteralIR, RangeIR, LoweredEinsteinIR, LoweredEinsteinClauseIR, LoweredRecurrenceIR,
    LoweredReductionIR, ReductionExpressionIR, BinaryOpIR, UnaryOpIR, RectangularAccessIR, IndexVarIR,
    FunctionCallIR, IdentifierIR, IfExpressionIR, BlockExpressionIR,
    is_function_binding, is_einstein_binding,
    IRVisitor, BindingIR,
)
from ..shared.defid import DefId, RUNTIME_CRATE
from ..shared.optional_attr import opt_defid
from ..shared.types import (
    BF16,
    BOOL,
    F16,
    F32,
    F64,
    F8E4M3,
    I8,
    I32,
    I64,
    BinaryOp,
    PrimitiveType,
    RectangularType,
    TupleType,
    Type,
    TypeKind,
)
from ..utils.config import DEFAULT_EINSTEIN_LOOP_MAX
from .numpy_helpers import _reject_non_lowered

_TYPE_NAME_TO_NUMPY_DTYPE = {
    I8: np.int8,
    I32: np.int32,
    I64: np.int64,
    F16: np.float16,
    F32: np.float32,
    F64: np.float64,
    BOOL: np.bool_,
}
try:
    import ml_dtypes as _ml_dtypes
except ImportError:
    _ml_dtypes = None
else:
    _TYPE_NAME_TO_NUMPY_DTYPE[BF16] = _ml_dtypes.bfloat16
    _TYPE_NAME_TO_NUMPY_DTYPE[F8E4M3] = _ml_dtypes.float8_e4m3fn


def _einlang_vectorize_debug_detail_enabled() -> bool:
    """Per-clause ``[vectorize] detail …`` lines (not only the summary)."""
    v = os.environ.get("EINLANG_DEBUG_VECTORIZE", "").strip().lower()
    return v in ("2", "verbose", "all", "detail")


def _einlang_recurrence_block_vectorized_binding_enabled() -> bool:
    """Recurrence clauses with a block body use broadcast index tensors and one body evaluation
    per timestep (same as non-block recurrence). Set EINLANG_VECTORIZE_RECURRENCE_BLOCK=0 to use
    scalar iteration over non-recurrence dims (debug / legacy)."""
    v = os.environ.get("EINLANG_VECTORIZE_RECURRENCE_BLOCK", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _einlang_debug_recurrence_block_enabled() -> bool:
    v = os.environ.get("EINLANG_DEBUG_RECURRENCE_BLOCK", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _allocate_numpy_output(output_shape: List[int], dtype: Any) -> np.ndarray:
    """Allocate backend output storage.

    Tuple-valued Einstein/recurrence outputs use ``dtype=object`` so each cell can
    hold an arbitrary Python tuple (possibly containing ndarrays).
    """
    if dtype is object:
        output = np.empty(output_shape, dtype=object)
        output.fill(None)
        return output
    return np.zeros(output_shape, dtype=dtype)


def _evaluate_shape_dim(shape_dim: Any, backend: Any) -> Any:
    """Shape entries may already be materialized ints or still be IR nodes."""
    if isinstance(shape_dim, (int, np.integer)):
        return int(shape_dim)
    if hasattr(shape_dim, "accept"):
        return shape_dim.accept(backend)
    return shape_dim


def _lowered_clause_loop_axis_names(lowered: Any) -> str:
    """Comma-separated Einstein clause loop index names (order = loop nesting)."""
    parts: List[str] = []
    for lp in lowered.loops or []:
        v = lp.variable
        if v is not None and getattr(v, "name", None):
            parts.append(v.name)
    return ",".join(parts)


def _vectorize_axes_scalar_vs_vector(lowered: Any, scalar_dim_indices: List[int]) -> str:
    """Human-readable which loop axes are scalar-iterated vs vectorized (broadcast arrays)."""
    loops = lowered.loops or []
    scalar_set = set(scalar_dim_indices)
    sn = [loops[d].variable.name for d in scalar_dim_indices if 0 <= d < len(loops) and loops[d].variable and loops[d].variable.name]
    vn = [
        loops[k].variable.name
        for k in range(len(loops))
        if k not in scalar_set and loops[k].variable and loops[k].variable.name
    ]
    return f"scalar_axes={','.join(sn)} vector_axes={','.join(vn)}"


class _BodyReferencesDefidVisitor(IRVisitor[bool]):
    """Visitor that returns True iff the tree contains an IdentifierIR or IndexVarIR with defid == target_defid."""

    def __init__(self, target_defid: Any) -> None:
        self._target = target_defid

    def references(self, expr: Any) -> bool:
        """True if expr (IR with accept) contains any node with defid == self._target."""
        if expr is None or self._target is None:
            return False
        return expr.accept(self)

    def _any(self, *nodes: Any) -> bool:
        for n in nodes:
            if n is not None and n.accept(self):
                return True
        return False

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_identifier(self, node: Any) -> bool:
        return node.defid == self._target

    def visit_index_var(self, node: Any) -> bool:
        return node.defid == self._target

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_binary_op(self, node: Any) -> bool:
        return self._any(node.left, node.right)

    def visit_function_call(self, node: Any) -> bool:
        args = node.arguments or []
        return self._any(node.callee_expr, *args)

    def visit_rectangular_access(self, node: Any) -> bool:
        if self._any(node.array):
            return True
        for idx in (node.indices or []):
            if self._any(idx):
                return True
        return False

    def visit_jagged_access(self, node: Any) -> bool:
        return self._any(node.base, *((node.index_chain or [])))

    def visit_block_expression(self, node: Any) -> bool:
        for stmt in (node.statements or []):
            if self._any(stmt):
                return True
        return self._any(node.final_expr)

    def visit_if_expression(self, node: Any) -> bool:
        return self._any(
            node.condition,
            node.then_expr,
            node.else_expr,
        )

    def visit_lambda(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_unary_op(self, node: Any) -> bool:
        return self._any(node.operand)

    def visit_range(self, node: Any) -> bool:
        return self._any(node.start, node.end)

    def visit_array_comprehension(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return self._any(*(node.elements or []))

    def visit_tuple_expression(self, node: Any) -> bool:
        return self._any(*(node.elements or []))

    def visit_tuple_access(self, node: Any) -> bool:
        return self._any(node.tuple_expr)

    def visit_interpolated_string(self, node: Any) -> bool:
        return self._any(*(node.parts or []))

    def visit_cast_expression(self, node: Any) -> bool:
        return self._any(node.expr)

    def visit_member_access(self, node: Any) -> bool:
        return self._any(node.object)

    def visit_try_expression(self, node: Any) -> bool:
        return self._any(node.operand)

    def visit_match_expression(self, node: Any) -> bool:
        if self._any(node.scrutinee):
            return True
        for arm in (node.arms or []):
            if self._any(arm.body):
                return True
        return False

    def visit_reduction_expression(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_lowered_reduction(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_where_expression(self, node: Any) -> bool:
        return self._any(node.expr, *(node.constraints or []))

    def visit_pipeline_expression(self, node: Any) -> bool:
        return self._any(node.left, node.right)

    def visit_builtin_call(self, node: Any) -> bool:
        return self._any(*(node.args or []))

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return self._any(*(node.patterns or []))

    def visit_array_pattern(self, node: Any) -> bool:
        return self._any(*(node.patterns or []))

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return self._any(node.inner_pattern, node.guard_expr)

    def visit_or_pattern(self, node: Any) -> bool:
        for alt in (node.alternatives or []):
            if self._any(alt):
                return True
        return False

    def visit_constructor_pattern(self, node: Any) -> bool:
        return self._any(*(node.patterns or []))

    def visit_binding_pattern(self, node: Any) -> bool:
        return self._any(node.inner_pattern)

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_function_value(self, node: Any) -> bool:
        return self._any(node.body)

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False

    def visit_binding(self, node: Any) -> bool:
        """Recurse into binding RHS (e.g. let z_cell = ... * state[t-1,...] + ...)."""
        expr = node.expr
        return self._any(expr) if expr is not None else False

    def visit_program(self, node: Any) -> bool:
        return self._any(*(node.statements or []))


class _ReductionDimsCounter(IRVisitor[int]):
    """Visitor that returns the max number of reduction dimensions in any LoweredReductionIR in the tree (0 if none)."""

    def visit_lowered_reduction(self, node: Any) -> int:
        loops = node.loops or []
        body_count = node.body.accept(self) or 0
        return max(len(loops), body_count)

    def visit_binary_op(self, node: Any) -> int:
        left = node.left.accept(self) or 0
        right = node.right.accept(self) or 0
        return max(left, right)

    def visit_rectangular_access(self, node: Any) -> int:
        n = node.array.accept(self) or 0
        for idx in node.indices:
            val = idx.accept(self)
            if val is None:
                val = 0
            n = max(n, val)
        return n

    def visit_function_call(self, node: Any) -> int:
        n = node.callee_expr.accept(self) or 0
        for a in node.arguments:
            val = a.accept(self)
            if val is None:
                val = 0
            n = max(n, val)
        return n

    def visit_unary_op(self, node: Any) -> int:
        return node.operand.accept(self) or 0

    def visit_if_expression(self, node: Any) -> int:
        c = node.condition.accept(self) or 0
        t = node.then_expr.accept(self) or 0
        e = node.else_expr.accept(self) if node.else_expr is not None else 0
        if e is None:
            e = 0
        return max(c, t, e)

    def visit_block_expression(self, node: Any) -> int:
        n = 0
        for stmt in node.statements:
            val = stmt.accept(self)
            if val is None:
                val = 0
            n = max(n, val)
        final = node.final_expr.accept(self) if node.final_expr is not None else 0
        if final is None:
            final = 0
        return max(n, final)

    def visit_lambda(self, node: Any) -> int:
        return node.body.accept(self) or 0

    def visit_range(self, node: Any) -> int:
        start = node.start.accept(self) if node.start is not None else 0
        end = node.end.accept(self) if node.end is not None else 0
        if start is None:
            start = 0
        if end is None:
            end = 0
        return max(start, end)

    def visit_array_comprehension(self, node: Any) -> int:
        return node.body.accept(self) or 0

    def visit_where_expression(self, node: Any) -> int:
        n = node.expr.accept(self) or 0
        for c in node.constraints:
            val = c.accept(self)
            if val is None:
                val = 0
            n = max(n, val)
        return n

    def visit_pipeline_expression(self, node: Any) -> int:
        left = node.left.accept(self)
        right = node.right.accept(self)
        if left is None:
            left = 0
        if right is None:
            right = 0
        return max(left, right)

    def visit_builtin_call(self, node: Any) -> int:
        n = 0
        for a in node.args:
            val = a.accept(self)
            if val is None:
                val = 0
            n = max(n, val)
        return n

    def visit_literal(self, node: Any) -> int:
        return 0

    def visit_identifier(self, node: Any) -> int:
        return 0

    def visit_index_var(self, node: Any) -> int:
        if node.range_ir is None:
            return 0
        val = node.range_ir.accept(self)
        return val if val is not None else 0

    def visit_index_rest(self, node: Any) -> int:
        return 0

    def visit_jagged_access(self, node: Any) -> int:
        n = node.base.accept(self) or 0
        for idx in node.index_chain:
            val = idx.accept(self)
            if val is None:
                val = 0
            n = max(n, val)
        return n

    def visit_module(self, node: Any) -> int:
        return 0

    def visit_array_literal(self, node: Any) -> int:
        n = 0
        for e in node.elements:
            val = e.accept(self)
            if val is None:
                val = 0
            n = max(n, val)
        return n

    def visit_tuple_expression(self, node: Any) -> int:
        n = 0
        for e in node.elements:
            val = e.accept(self)
            if val is None:
                val = 0
            n = max(n, val)
        return n

    def visit_tuple_access(self, node: Any) -> int:
        return node.tuple_expr.accept(self) or 0

    def visit_interpolated_string(self, node: Any) -> int:
        n = 0
        for p in node.parts:
            val = p.accept(self)
            if val is None:
                val = 0
            n = max(n, val)
        return n

    def visit_cast_expression(self, node: Any) -> int:
        return node.expr.accept(self) or 0

    def visit_member_access(self, node: Any) -> int:
        return node.object.accept(self) or 0

    def visit_try_expression(self, node: Any) -> int:
        return node.operand.accept(self) or 0

    def visit_match_expression(self, node: Any) -> int:
        n = node.scrutinee.accept(self) or 0
        for arm in node.arms:
            val = arm.body.accept(self)
            if val is None:
                val = 0
            n = max(n, val)
        return n

    def visit_reduction_expression(self, node: Any) -> int:
        return node.body.accept(self)

    def visit_literal_pattern(self, node: Any) -> int:
        return 0

    def visit_identifier_pattern(self, node: Any) -> int:
        return 0

    def visit_wildcard_pattern(self, node: Any) -> int:
        return 0

    def visit_tuple_pattern(self, node: Any) -> int:
        n = 0
        for e in node.elements:
            n = max(n, e.accept(self))
        return n

    def visit_array_pattern(self, node: Any) -> int:
        n = 0
        for e in node.elements:
            n = max(n, e.accept(self))
        return n

    def visit_rest_pattern(self, node: Any) -> int:
        return 0

    def visit_guard_pattern(self, node: Any) -> int:
        return max(node.inner_pattern.accept(self), node.guard_expr.accept(self))

    def visit_binding(self, node: Any) -> int:
        return node.expr.accept(self)

    def visit_program(self, node: Any) -> int:
        n = 0
        for stmt in node.statements:
            n = max(n, stmt.accept(self))
        return n


def _count_reduction_dims_in_expr(expr: Any) -> int:
    """Return max number of reduction dimensions in any LoweredReductionIR in expr (0 if none)."""
    if expr is None:
        return 0
    return expr.accept(_ReductionDimsCounter())



class _ReductionUsesClauseVarVisitor(IRVisitor[bool]):
    """True if any LoweredReductionIR has a loop whose iterable references a clause loop var (dynamic bounds)."""

    def __init__(self, clause_loop_defids: List[Any]) -> None:
        self._clause_loop_defids = clause_loop_defids

    def _any(self, *nodes: Any) -> bool:
        for n in nodes:
            if n is not None and n.accept(self):
                return True
        return False

    def visit_lowered_reduction(self, node: Any) -> bool:
        for loop in node.loops:
            it = loop.iterable
            if it is not None and any(
                _BodyReferencesDefidVisitor(d).references(it) for d in self._clause_loop_defids
            ):
                return True
        return node.body.accept(self)

    def visit_binary_op(self, node: Any) -> bool:
        return node.left.accept(self) or node.right.accept(self)

    def visit_rectangular_access(self, node: Any) -> bool:
        if node.array.accept(self):
            return True
        for idx in node.indices:
            if idx.accept(self):
                return True
        return False

    def visit_function_call(self, node: Any) -> bool:
        if node.callee_expr.accept(self):
            return True
        for a in node.arguments:
            if a.accept(self):
                return True
        return False

    def visit_unary_op(self, node: Any) -> bool:
        return node.operand.accept(self)

    def visit_if_expression(self, node: Any) -> bool:
        return (
            node.condition.accept(self)
            or node.then_expr.accept(self)
            or (node.else_expr.accept(self) if node.else_expr is not None else False)
        )

    def visit_block_expression(self, node: Any) -> bool:
        for stmt in node.statements:
            if stmt.accept(self):
                return True
        return (
            node.final_expr.accept(self)
            if node.final_expr is not None
            else False
        )

    def visit_lambda(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_range(self, node: Any) -> bool:
        return node.start.accept(self) or node.end.accept(self)

    def visit_array_comprehension(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_where_expression(self, node: Any) -> bool:
        if node.expr.accept(self):
            return True
        for c in node.constraints:
            if c.accept(self):
                return True
        return False

    def visit_pipeline_expression(self, node: Any) -> bool:
        return node.left.accept(self) or node.right.accept(self)

    def visit_builtin_call(self, node: Any) -> bool:
        return any(a.accept(self) for a in node.args)

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_identifier(self, node: Any) -> bool:
        return False

    def visit_index_var(self, node: Any) -> bool:
        return (
            node.range_ir.accept(self)
            if node.range_ir is not None
            else False
        )

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_jagged_access(self, node: Any) -> bool:
        if node.base.accept(self):
            return True
        return any(idx.accept(self) for idx in node.index_chain)

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_tuple_expression(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_tuple_access(self, node: Any) -> bool:
        return node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> bool:
        return any(p.accept(self) for p in node.parts)

    def visit_cast_expression(self, node: Any) -> bool:
        return node.expr.accept(self)

    def visit_member_access(self, node: Any) -> bool:
        return node.object.accept(self)

    def visit_try_expression(self, node: Any) -> bool:
        return node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> bool:
        if node.scrutinee.accept(self):
            return True
        return any(arm.body.accept(self) for arm in node.arms)

    def visit_reduction_expression(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_array_pattern(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return node.inner_pattern.accept(self) or node.guard_expr.accept(self)

    def visit_binding(self, node: Any) -> bool:
        return node.expr.accept(self)

    def visit_program(self, node: Any) -> bool:
        return any(stmt.accept(self) for stmt in node.statements)

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_einstein(self, node: Any) -> bool:
        return any(item.accept(self) for item in node.items)

    def visit_lowered_recurrence(self, node: Any) -> bool:
        if node.initial is not None and node.initial.accept(self):
            return True
        rloop = node.recurrence_loop
        if rloop is not None and rloop.iterable is not None:
            if rloop.iterable.accept(self):
                return True
        return (
            node.body is not None
            and node.body.accept(self)
        )

    def visit_or_pattern(self, node: Any) -> bool:
        return any(alt.accept(self) for alt in (node.alternatives or []))

    def visit_constructor_pattern(self, node: Any) -> bool:
        return any(p.accept(self) for p in node.patterns)

    def visit_binding_pattern(self, node: Any) -> bool:
        return node.inner_pattern.accept(self)

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_function_value(self, node: Any) -> bool:
        return node.body.accept(self) if node.body is not None else False

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False


def _reduction_uses_clause_var_in_bounds(expr: Any, clause_loop_defids: List[Any]) -> bool:
    """True if any LoweredReductionIR in expr has a loop whose iterable references a clause loop var (dynamic bounds)."""
    if expr is None or not clause_loop_defids:
        return False
    return expr.accept(_ReductionUsesClauseVarVisitor(clause_loop_defids))


def _merge_defids_by_name(a: Dict[str, List[Any]], b: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Merge two name->defids dicts (a and b) into one. Modifies and returns a."""
    for k, v in b.items():
        a.setdefault(k, []).extend(v)
    return a


class _DefidsByNameCollector(IRVisitor[Dict[str, List[Any]]]):
    """Collect all (name -> list of defids) from IdentifierIR/IndexVarIR in the tree."""

    def _empty(self) -> Dict[str, List[Any]]:
        return {}

    def _one(self, name: str, defid: Any) -> Dict[str, List[Any]]:
        if name is not None and defid is not None:
            return {name: [defid]}
        return self._empty()

    def visit_literal(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_identifier(self, node: Any) -> Dict[str, List[Any]]:
        return self._one(node.name, node.defid)

    def visit_index_var(self, node: Any) -> Dict[str, List[Any]]:
        out = self._one(node.name, node.defid)
        if node.range_ir is not None:
            _merge_defids_by_name(out, node.range_ir.accept(self))
        return out

    def visit_index_rest(self, node: Any) -> Dict[str, List[Any]]:
        return self._one(node.name, node.defid)

    def visit_binary_op(self, node: Any) -> Dict[str, List[Any]]:
        out = node.left.accept(self)
        _merge_defids_by_name(out, node.right.accept(self))
        return out

    def visit_unary_op(self, node: Any) -> Dict[str, List[Any]]:
        return node.operand.accept(self)

    def visit_rectangular_access(self, node: Any) -> Dict[str, List[Any]]:
        out = node.array.accept(self)
        for idx in node.indices:
            _merge_defids_by_name(out, idx.accept(self))
        return out

    def visit_function_call(self, node: Any) -> Dict[str, List[Any]]:
        out = node.callee_expr.accept(self)
        for a in node.arguments:
            _merge_defids_by_name(out, a.accept(self))
        return out

    def visit_if_expression(self, node: Any) -> Dict[str, List[Any]]:
        out = node.condition.accept(self)
        _merge_defids_by_name(out, node.then_expr.accept(self))
        if node.else_expr is not None:
            _merge_defids_by_name(out, node.else_expr.accept(self))
        return out

    def visit_block_expression(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for stmt in node.statements:
            _merge_defids_by_name(out, stmt.accept(self))
        if node.final_expr is not None:
            _merge_defids_by_name(out, node.final_expr.accept(self))
        return out

    def visit_lambda(self, node: Any) -> Dict[str, List[Any]]:
        return node.body.accept(self)

    def visit_range(self, node: Any) -> Dict[str, List[Any]]:
        out = node.start.accept(self)
        _merge_defids_by_name(out, node.end.accept(self))
        return out

    def visit_lowered_reduction(self, node: Any) -> Dict[str, List[Any]]:
        out = node.body.accept(self)
        for lp in node.loops:
            if lp.variable is not None:
                _merge_defids_by_name(out, lp.variable.accept(self))
            if lp.iterable is not None:
                _merge_defids_by_name(out, lp.iterable.accept(self))
        return out

    def visit_jagged_access(self, node: Any) -> Dict[str, List[Any]]:
        out = node.base.accept(self)
        for idx in node.index_chain:
            _merge_defids_by_name(out, idx.accept(self))
        return out

    def visit_module(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_array_literal(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for e in node.elements:
            _merge_defids_by_name(out, e.accept(self))
        return out

    def visit_tuple_expression(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for e in node.elements:
            _merge_defids_by_name(out, e.accept(self))
        return out

    def visit_tuple_access(self, node: Any) -> Dict[str, List[Any]]:
        return node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for p in node.parts:
            _merge_defids_by_name(out, p.accept(self))
        return out

    def visit_cast_expression(self, node: Any) -> Dict[str, List[Any]]:
        return node.expr.accept(self)

    def visit_member_access(self, node: Any) -> Dict[str, List[Any]]:
        return node.object.accept(self)

    def visit_try_expression(self, node: Any) -> Dict[str, List[Any]]:
        return node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> Dict[str, List[Any]]:
        out = node.scrutinee.accept(self)
        for arm in node.arms:
            _merge_defids_by_name(out, arm.body.accept(self))
        return out

    def visit_reduction_expression(self, node: Any) -> Dict[str, List[Any]]:
        return node.body.accept(self)

    def visit_where_expression(self, node: Any) -> Dict[str, List[Any]]:
        out = node.expr.accept(self)
        for c in node.constraints:
            _merge_defids_by_name(out, c.accept(self))
        return out

    def visit_pipeline_expression(self, node: Any) -> Dict[str, List[Any]]:
        out = node.left.accept(self)
        _merge_defids_by_name(out, node.right.accept(self))
        return out

    def visit_builtin_call(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for a in node.args:
            _merge_defids_by_name(out, a.accept(self))
        return out

    def visit_array_comprehension(self, node: Any) -> Dict[str, List[Any]]:
        return node.body.accept(self)

    def visit_literal_pattern(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_identifier_pattern(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_wildcard_pattern(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_tuple_pattern(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for e in node.elements:
            _merge_defids_by_name(out, e.accept(self))
        return out

    def visit_array_pattern(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for e in node.elements:
            _merge_defids_by_name(out, e.accept(self))
        return out

    def visit_rest_pattern(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_guard_pattern(self, node: Any) -> Dict[str, List[Any]]:
        out = node.inner_pattern.accept(self)
        _merge_defids_by_name(out, node.guard_expr.accept(self))
        return out

    def visit_binding(self, node: Any) -> Dict[str, List[Any]]:
        return node.expr.accept(self)

    def visit_program(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for stmt in node.statements:
            _merge_defids_by_name(out, stmt.accept(self))
        return out

    def visit_lowered_comprehension(self, node: Any) -> Dict[str, List[Any]]:
        return node.body.accept(self)

    def visit_lowered_einstein_clause(self, node: Any) -> Dict[str, List[Any]]:
        out = node.body.accept(self)
        for lp in node.loops or []:
            if lp.variable is not None:
                _merge_defids_by_name(out, lp.variable.accept(self))
            if lp.iterable is not None:
                _merge_defids_by_name(out, lp.iterable.accept(self))
        for idx in node.indices or []:
            if idx is not None and hasattr(idx, "accept"):
                _merge_defids_by_name(out, idx.accept(self))
        return out

    def visit_lowered_einstein(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for item in node.items:
            _merge_defids_by_name(out, item.accept(self))
        return out

    def visit_lowered_recurrence(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        if node.initial is not None:
            _merge_defids_by_name(out, node.initial.accept(self))
        rloop = node.recurrence_loop
        if rloop is not None and rloop.iterable is not None:
            _merge_defids_by_name(out, rloop.iterable.accept(self))
        if node.body is not None:
            _merge_defids_by_name(out, node.body.accept(self))
        return out

    def visit_lowered_select_at_argmax(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        if node.primal_body is not None:
            _merge_defids_by_name(out, node.primal_body.accept(self))
        if node.diff_body is not None:
            _merge_defids_by_name(out, node.diff_body.accept(self))
        return out

    def visit_or_pattern(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for alt in (node.alternatives or []):
            _merge_defids_by_name(out, alt.accept(self))
        return out

    def visit_constructor_pattern(self, node: Any) -> Dict[str, List[Any]]:
        out = self._empty()
        for p in node.patterns:
            _merge_defids_by_name(out, p.accept(self))
        return out

    def visit_binding_pattern(self, node: Any) -> Dict[str, List[Any]]:
        return node.inner_pattern.accept(self)

    def visit_range_pattern(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_function_value(self, node: Any) -> Dict[str, List[Any]]:
        return node.body.accept(self) if node.body is not None else self._empty()

    def visit_einstein(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()

    def visit_einstein_clause(self, node: Any) -> Dict[str, List[Any]]:
        return self._empty()
