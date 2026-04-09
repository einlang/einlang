"""NumPy backend expression visitors. All lookup via env (no global table)."""

from typing import Any, Dict, List, Optional, Tuple, cast
import time
import warnings

import numpy as np

from ..shared.types import BinaryOp, UnaryOp, TypeKind
from ..shared.optional_attr import opt_defid
from ..ir.nodes import (
    LiteralIR, IdentifierIR, IndexVarIR, BinaryOpIR, UnaryOpIR, FunctionCallIR,
    BlockExpressionIR, RangeIR, ArrayComprehensionIR, RectangularAccessIR, JaggedAccessIR,
    ArrayLiteralIR, TupleExpressionIR, TupleAccessIR, InterpolatedStringIR, CastExpressionIR,
    MemberAccessIR, TryExpressionIR, MatchExpressionIR, ReductionExpressionIR, WhereExpressionIR,
    PipelineExpressionIR, BuiltinCallIR,
    MatchArmIR, ExpressionIR, LoweredComprehensionIR, LoweredReductionIR,
    LoweredSelectAtArgmaxIR,
    LoweredEinsteinIR,
    DifferentialIR,
    IRVisitor,
)
from ..runtime.environment import FunctionValue
from .numpy_helpers import (
    _reject_non_lowered, _PatternMatcher,
    builtin_assert, builtin_print, builtin_len, builtin_shape, builtin_typeof,
    builtin_sum, builtin_max, builtin_min, builtin_array_append,
)


def _first_parallel_index_defid(
    idx_expr: ExpressionIR, reduction_body_defids: Any
) -> Any:
    if idx_expr is None:
        return None
    if isinstance(idx_expr, IdentifierIR):
        d = idx_expr.defid
        if d is not None and d not in reduction_body_defids:
            return d
        return None
    if isinstance(idx_expr, IndexVarIR):
        d = idx_expr.defid
        if d is not None and d not in reduction_body_defids:
            return d
        return None
    if isinstance(idx_expr, BinaryOpIR):
        left = _first_parallel_index_defid(idx_expr.left, reduction_body_defids)
        if left is not None:
            return left
        return _first_parallel_index_defid(idx_expr.right, reduction_body_defids)
    if isinstance(idx_expr, UnaryOpIR):
        return _first_parallel_index_defid(idx_expr.operand, reduction_body_defids)
    if isinstance(idx_expr, CastExpressionIR):
        return _first_parallel_index_defid(idx_expr.expr, reduction_body_defids)
    return None


def _is_scalar_like(x: Any) -> bool:
    if x is None:
        return True
    if np.isscalar(x):
        return True
    if isinstance(x, np.ndarray):
        return x.ndim == 0 or x.size == 1
    return False


def _is_scalar_or_0d_array(x: Any) -> bool:
    return np.isscalar(x) or (isinstance(x, np.ndarray) and getattr(x, "ndim", -1) == 0)


def _normalize_literal_sequence(value: Any) -> Any:
    """Convert Python list literals to backend-friendly ndarrays when possible."""
    if not isinstance(value, list):
        return value
    try:
        arr = np.asarray(value)
    except Exception:
        return value
    if arr.dtype.kind == "f":
        return arr.astype(np.float32, copy=False)
    if arr.dtype.kind in ("i", "u"):
        return arr.astype(np.int32, copy=False)
    if arr.dtype.kind == "b":
        return arr.astype(bool, copy=False)
    return arr


def _invoke_runtime_builtin(fn: Any, args: List[Any]) -> Any:
    """Invoke a runtime builtin with the same argument rules across call sites."""
    if fn == builtin_assert:
        if len(args) == 0:
            raise RuntimeError("assert() called with no arguments")
        return builtin_assert(args[0], args[1] if len(args) > 1 else "Assertion failed")
    return fn(*args)


def _apply_optional_bias(
    result: Any,
    bias: Any,
    backend: Any,
    *,
    last_dim_row: bool = False,
) -> Any:
    if bias is None:
        return result
    bias_val = bias.accept(backend)
    if isinstance(bias_val, np.ndarray) and isinstance(result, np.ndarray):
        if last_dim_row and bias_val.size == result.shape[-1]:
            return result + np.reshape(bias_val, (1, -1))
        return result + np.broadcast_to(bias_val, result.shape)
    if _is_scalar_or_0d_array(bias_val):
        return result + bias_val
    return result


def _sliding_window_axis_view(
    array: np.ndarray,
    axis: int,
    window_size: int,
    stride: int = 1,
) -> Optional[np.ndarray]:
    """Return a generic strided sliding-window view along a single axis.

    The window axis is appended at the end, matching NumPy's
    ``sliding_window_view`` behavior. This is the backend building block for
    lowering Einstein local-window patterns into vectorized NumPy kernels.
    """
    if (
        not isinstance(array, np.ndarray)
        or window_size <= 0
        or stride <= 0
        or array.ndim == 0
    ):
        return None
    ndim = array.ndim
    axis = axis % ndim
    if window_size > array.shape[axis]:
        return None
    try:
        windows = np.lib.stride_tricks.sliding_window_view(
            array,
            window_shape=window_size,
            axis=axis,
        )
    except Exception:
        return None
    if stride != 1:
        index = [slice(None)] * windows.ndim
        index[axis] = slice(None, None, stride)
        windows = windows[tuple(index)]
    return windows


def _sliding_window_nd_view(
    array: np.ndarray,
    axes: Tuple[int, ...],
    window_shape: Tuple[int, ...],
    strides: Tuple[int, ...],
) -> Optional[np.ndarray]:
    """Sliding-window view over multiple axes; subsample output with per-axis strides.

    Same role as ``_sliding_window_axis_view`` but for 2D/3D spatial conv (valid, dilation 1).
    """
    if (
        not isinstance(array, np.ndarray)
        or len(axes) != len(window_shape)
        or len(axes) != len(strides)
        or array.ndim == 0
    ):
        return None
    ndim = array.ndim
    axes_norm = tuple(a % ndim for a in axes)
    if len(set(axes_norm)) != len(axes_norm):
        return None
    for ax, ws, st in zip(axes_norm, window_shape, strides):
        if ws <= 0 or st <= 0 or array.shape[ax] < ws:
            return None
    try:
        windows = np.lib.stride_tricks.sliding_window_view(
            array, window_shape=window_shape, axis=axes_norm
        )
    except Exception:
        return None
    index = [slice(None)] * windows.ndim
    for ax, st in zip(axes_norm, strides):
        if st != 1:
            index[ax] = slice(None, None, st)
    try:
        return windows[tuple(index)]
    except Exception:
        return None


def _conv_spatial_stride_from_index(
    spatial_idx: Any,
    kernel_red_defid: Any,
    *,
    _add: BinaryOp,
    _mul: BinaryOp,
) -> Optional[int]:
    """Stride along output for ``out * stride + kernel`` or ``out + kernel`` (same rules as 1D fast path)."""
    if isinstance(spatial_idx, BinaryOpIR) and spatial_idx.operator == _add:
        left, right = spatial_idx.left, spatial_idx.right
        if _expr_contains_defid(left, kernel_red_defid) and _expr_contains_defid(right, kernel_red_defid):
            return None
        if _expr_contains_defid(right, kernel_red_defid):
            left, right = right, left
        if not _expr_contains_defid(left, kernel_red_defid):
            return None
        if isinstance(right, (IdentifierIR, IndexVarIR)):
            return 1
        if isinstance(right, BinaryOpIR) and right.operator == _mul:
            rL, rR = right.left, right.right
            if isinstance(rR, LiteralIR):
                stride = int(rR.value)
            elif isinstance(rL, LiteralIR):
                stride = int(rL.value)
            else:
                return None
            if stride not in (1, 2):
                return None
            return stride
        return None
    if isinstance(spatial_idx, (IdentifierIR, IndexVarIR)) and spatial_idx.defid == kernel_red_defid:
        return 1
    return None


def _windowed_einsum_reduction(
    input_arr: np.ndarray,
    weight_arr: np.ndarray,
    *,
    window_axis: int,
    window_size: int,
    stride: int,
    equation: str,
) -> Optional[np.ndarray]:
    """Evaluate an Einstein-style local-window reduction via sliding windows.

    This is intentionally conv-agnostic: callers provide the source tensor,
    the reduction tensor, the windowed axis metadata, and the einsum equation
    that lowers the original Einstein sum-of-products into NumPy.
    """
    windows = _sliding_window_axis_view(
        input_arr,
        axis=window_axis,
        window_size=window_size,
        stride=stride,
    )
    if windows is None:
        return None
    try:
        return np.einsum(equation, windows, weight_arr, optimize=True)
    except Exception:
        return None


def _safe_oob_ndarray_access(array: np.ndarray, indices: List[Any]) -> Any:
    """Advanced ndarray indexing with zero-fill for out-of-bounds positions.

    This is used only during speculative vectorized clause evaluation, where an
    array-valued `if` may eagerly evaluate a branch that is semantically masked
    away. It preserves the in-bounds values and substitutes zero for masked-out
    OOB positions so the clause can still vectorize.
    """
    if not indices:
        return array
    if len(indices) > array.ndim:
        raise IndexError(
            f"too many indices for array: expected at most {array.ndim}, got {len(indices)}"
        )

    normalized: List[np.ndarray] = []
    for idx in indices:
        idx_arr = np.asarray(idx)
        if idx_arr.dtype.kind not in ("i", "u", "b"):
            raise TypeError(
                "safe vectorized rectangular access requires integer-like ndarray indices"
            )
        normalized.append(idx_arr.astype(np.intp, copy=False))
    broadcast = np.broadcast_arrays(*normalized)

    # The common vectorized case is already in bounds; avoid building masks/clipped
    # arrays when we can safely dispatch to NumPy's normal advanced indexing.
    all_in_bounds = True
    for axis, idx_arr in enumerate(broadcast):
        axis_size = array.shape[axis]
        if axis_size <= 0:
            raise IndexError(f"indexing axis {axis} with size {axis_size} is not supported")
        if idx_arr.size == 0:
            continue
        try:
            idx_min = int(idx_arr.min())
            idx_max = int(idx_arr.max())
        except ValueError:
            idx_min = 0
            idx_max = -1
        if idx_min < 0 or idx_max >= axis_size:
            all_in_bounds = False
            break
    if all_in_bounds:
        return array[tuple(broadcast)]

    valid = np.ones(broadcast[0].shape, dtype=bool)
    clipped: List[np.ndarray] = []
    for axis, idx_arr in enumerate(broadcast):
        axis_size = array.shape[axis]
        if axis_size <= 0:
            raise IndexError(f"indexing axis {axis} with size {axis_size} is not supported")
        in_bounds = (idx_arr >= 0) & (idx_arr < axis_size)
        valid &= in_bounds
        clipped.append(idx_arr.clip(0, axis_size - 1))

    gathered = array[tuple(clipped)]
    if gathered.shape != valid.shape:
        extra_ndim = gathered.ndim - valid.ndim
        valid = valid.reshape(valid.shape + (1,) * max(extra_ndim, 0))
    zero = np.zeros((), dtype=gathered.dtype)
    return np.where(valid, gathered, zero)


class _ExprContainsDefidVisitor(IRVisitor[bool]):
    """Returns True iff the expression tree contains IdentifierIR or IndexVarIR with defid == target_defid."""

    def __init__(self, target_defid: Any) -> None:
        self._target = target_defid

    def visit_identifier(self, node: IdentifierIR) -> bool:
        return node.defid == self._target

    def visit_index_var(self, node: IndexVarIR) -> bool:
        return node.defid == self._target

    def visit_binary_op(self, node: BinaryOpIR) -> bool:
        return node.left.accept(self) or node.right.accept(self)

    def visit_unary_op(self, node: UnaryOpIR) -> bool:
        return node.operand.accept(self)

    def visit_rectangular_access(self, node: RectangularAccessIR) -> bool:
        if node.array.accept(self):
            return True
        for idx in node.indices or []:
            if idx.accept(self):
                return True
        return False

    def visit_function_call(self, node: FunctionCallIR) -> bool:
        if node.callee_expr is not None and node.callee_expr.accept(self):
            return True
        for a in (node.arguments or []):
            if a.accept(self):
                return True
        return False

    def visit_jagged_access(self, node: JaggedAccessIR) -> bool:
        if node.base.accept(self):
            return True
        for idx in (node.index_chain or []):
            if idx is not None and idx.accept(self):
                return True
        return False

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_block_expression(self, node: Any) -> bool:
        return False

    def visit_if_expression(self, node: Any) -> bool:
        return False

    def visit_lambda(self, node: Any) -> bool:
        return False

    def visit_range(self, node: Any) -> bool:
        return False

    def visit_array_comprehension(self, node: Any) -> bool:
        return False

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return False

    def visit_tuple_expression(self, node: Any) -> bool:
        return False

    def visit_tuple_access(self, node: Any) -> bool:
        return False

    def visit_interpolated_string(self, node: Any) -> bool:
        return False

    def visit_cast_expression(self, node: Any) -> bool:
        return False

    def visit_member_access(self, node: Any) -> bool:
        return False

    def visit_try_expression(self, node: Any) -> bool:
        return False

    def visit_match_expression(self, node: Any) -> bool:
        return False

    def visit_reduction_expression(self, node: Any) -> bool:
        return False

    def visit_where_expression(self, node: Any) -> bool:
        return False

    def visit_pipeline_expression(self, node: Any) -> bool:
        return False

    def visit_builtin_call(self, node: Any) -> bool:
        return False

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return False

    def visit_array_pattern(self, node: Any) -> bool:
        return False

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return False

    def visit_or_pattern(self, node: Any) -> bool:
        return False

    def visit_constructor_pattern(self, node: Any) -> bool:
        return False

    def visit_binding_pattern(self, node: Any) -> bool:
        return False

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_function_value(self, node: Any) -> bool:
        return False

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False

    def visit_binding(self, node: Any) -> bool:
        return False

    def visit_program(self, node: Any) -> bool:
        return False

    def visit_lowered_reduction(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_einstein(self, node: Any) -> bool:
        for item in node.items or []:
            if item.accept(self):
                return True
        return False

    def visit_lowered_recurrence(self, node: Any) -> bool:
        if node.initial is not None and node.initial.accept(self):
            return True
        rec_loop = node.recurrence_loop
        if rec_loop is not None and rec_loop.iterable is not None and rec_loop.iterable.accept(self):
            return True
        if node.body is not None and node.body.accept(self):
            return True
        return False


def _expr_contains_defid(expr: Any, target_defid: Any) -> bool:
    if expr is None or target_defid is None:
        return False
    if hasattr(expr, "accept"):
        return expr.accept(_ExprContainsDefidVisitor(target_defid))
    return False


def _reduction_axes_in_access(
    backend: Any, indices: List[Any], reduction_defids: List[Any]
) -> Optional[Tuple[int, ...]]:
    """Return the array axis index for each reduction defid (position in indices = axis in array)."""
    axes: List[Optional[int]] = [None] * len(reduction_defids)
    for axis_in_array, idx in enumerate(indices):
        idx_defid = opt_defid(idx)
        if idx_defid is not None and idx_defid in reduction_defids:
            pos = reduction_defids.index(idx_defid)
            axes[pos] = axis_in_array
        else:
            for pos, rd in enumerate(reduction_defids):
                if axes[pos] is None and _expr_contains_defid(idx, rd):
                    axes[pos] = axis_in_array
                    break
        try:
            idx.accept(backend)
        except Exception:
            return None
    if any(a is None for a in axes):
        return None
    return tuple(axes)


def _infer_reduction_axes_from_shape(
    shape: Tuple[int, ...], reduction_sizes: List[int]
) -> Optional[Tuple[int, ...]]:
    used: set = set()
    axes: List[int] = []
    for rs in reduction_sizes:
        found = None
        for i, s in enumerate(shape):
            if i not in used and int(s) == int(rs):
                found = i
                break
        if found is None:
            return None
        axes.append(found)
        used.add(found)
    return tuple(axes)


def _try_matmul_reduction(expr: LoweredReductionIR, backend: Any, plan: Any) -> Optional[Any]:
    from ..shared.types import ReductionOp
    op = expr.operation
    if op != ReductionOp.SUM:
        return None
    if expr.guards or expr.bindings:
        return None
    loops = list(expr.loops or [])
    if not loops:
        return None
    reduction_defids: List[Any] = []
    reduction_sizes: List[int] = []
    for loop in loops:
        loop_var = loop.variable
        if loop_var is None:
            return None
        loop_defid = loop_var.defid
        if loop_defid is None:
            return None
        reduction_defids.append(loop_defid)
        try:
            iterable = loop.iterable.accept(backend)
            if iterable is None:
                return None
            reduction_sizes.append(int(len(iterable)))
        except Exception:
            return None
    mul_left = getattr(plan, "left", None)
    mul_right = getattr(plan, "right", None)
    bias = getattr(plan, "bias", None)
    scale = getattr(plan, "scale", None)
    if mul_left is None or mul_right is None:
        return None
    if not isinstance(mul_left, RectangularAccessIR) or not isinstance(mul_right, RectangularAccessIR):
        return None
    indices_left = mul_left.indices or []
    indices_right = mul_right.indices or []
    from ..ir.nodes import IdentifierIR, IndexVarIR
    for _idx in indices_left + indices_right:
        for _rd in reduction_defids:
            if _expr_contains_defid(_idx, _rd):
                if not (isinstance(_idx, (IdentifierIR, IndexVarIR)) and _idx.defid in reduction_defids):
                    return None
    n_red = len(reduction_sizes)
    # GEMM-style: 1 or 2 reduction dims (QKV / batched matmul, or full matmul). Recurrence clauses skip this path.
    if n_red not in (1, 2):
        return None
    # Evaluate base arrays (not indexed) so we get correct shapes for BLAS when parallel
    # loop vars are already set to broadcast arrays (avoids huge intermediate in vectorized path).
    left_arr = mul_left.array
    right_arr = mul_right.array
    try:
        with backend.env.scope():
            for i, (defid, N) in enumerate(zip(reduction_defids, reduction_sizes)):
                if n_red == 1:
                    backend.env.set_value(defid, np.arange(N, dtype=np.intp))
                else:
                    shape = [1] * n_red
                    shape[i] = N
                    backend.env.set_value(defid, np.arange(N, dtype=np.intp).reshape(shape))
            axes_left = _reduction_axes_in_access(backend, indices_left, reduction_defids)
            axes_right = _reduction_axes_in_access(backend, indices_right, reduction_defids)
            if left_arr is not None:
                left_val = left_arr.accept(backend)
            else:
                left_val = mul_left.accept(backend)
            if right_arr is not None:
                right_val = right_arr.accept(backend)
            else:
                right_val = mul_right.accept(backend)
    except Exception:
        return None
    if not isinstance(left_val, np.ndarray) or not isinstance(right_val, np.ndarray):
        return None
    left_val, kept_left = _slice_array_at_scalar_indices(
        left_val, indices_left, reduction_defids, backend
    )
    right_val, kept_right = _slice_array_at_scalar_indices(
        right_val, indices_right, reduction_defids, backend
    )
    axes_left = _remap_axes_after_scalar_slicing(axes_left, kept_left)
    axes_right = _remap_axes_after_scalar_slicing(axes_right, kept_right)
    if axes_left is None:
        axes_left = _infer_reduction_axes_from_shape(left_val.shape, reduction_sizes)
    if axes_right is None:
        axes_right = _infer_reduction_axes_from_shape(right_val.shape, reduction_sizes)
    if axes_left is None or axes_right is None:
        return None
    # np.matmul supports batch dims: 2D @ 3D -> (batch, m, p), 3D @ 2D -> (batch, m, p), 3D @ 3D -> (batch, m, p).
    # Contraction: last dim of left, first "matrix" dim of right (0 if 2D, 1 if 3D).
    right_contract_axis = 0 if right_val.ndim == 2 else 1
    use_matmul = (
        len(axes_left) == 1
        and len(axes_right) == 1
        and axes_left[0] == left_val.ndim - 1
        and axes_right[0] == right_contract_axis
    )
    try:
        if use_matmul and (
            (left_val.ndim == 2 and right_val.ndim == 2)
            or (left_val.ndim == 2 and right_val.ndim == 3)
            or (left_val.ndim == 3 and right_val.ndim == 2)
            or (left_val.ndim == 3 and right_val.ndim == 3 and left_val.shape[0] == right_val.shape[0])
        ):
            result = np.matmul(left_val, right_val)
            # Index out scalar batch dims so result matches parallel_shape (e.g. fc1: (4,1500,1536) -> [L,:,:] -> (1500,1536)).
            if result.ndim > 2:
                batch_indices = (
                    indices_left[: left_val.ndim - 2]
                    if left_val.ndim == 3
                    else indices_right[: right_val.ndim - 2]
                )
                key: List[Any] = []
                for idx in batch_indices:
                    try:
                        v = idx.accept(backend)
                        if _is_scalar_or_0d_array(v):
                            key.append(int(v))
                        else:
                            key.append(slice(None))
                    except Exception:
                        key.append(slice(None))
                if len(key) == result.ndim - 2 and all(isinstance(k, int) for k in key):
                    key.extend([slice(None)] * 2)
                    result = result[tuple(key)]
        elif use_matmul:
            result = np.tensordot(left_val, right_val, axes=(axes_left, axes_right))
        else:
            result = np.tensordot(left_val, right_val, axes=(axes_left, axes_right))
    except Exception:
        return None
    if bias is not None:
        try:
            result = _apply_optional_bias(result, bias, backend)
        except Exception:
            return None
    if scale is not None:
        result = result * scale
    return result


def _try_windowed_sumprod_einsum_spatial_nd(
    expr: LoweredReductionIR,
    backend: Any,
    loops: List[Any],
    mul_left: RectangularAccessIR,
    mul_right: RectangularAccessIR,
    bias: Optional[Any],
    *,
    _add: BinaryOp,
    _mul: BinaryOp,
) -> Optional[Any]:
    """2D/3D valid conv: sum over input channel and kernel axes; dilation 1; strides 1 or 2."""
    n_red = len(loops)
    if n_red not in (3, 4):
        return None
    n_spatial = n_red - 1
    red_vars = [loops[i].variable for i in range(n_red)]
    if any(v is None or v.defid is None for v in red_vars):
        return None
    reduction_defids = [v.defid for v in red_vars]
    try:
        n_ci = int(len(loops[0].iterable.accept(backend)))
        kern_sizes = [int(len(loops[i].iterable.accept(backend))) for i in range(1, n_red)]
    except Exception:
        return None
    il, ir = mul_left.indices or [], mul_right.indices or []
    if len(il) != n_red and len(il) != n_red + 1:
        il, ir = ir, il
        mul_left, mul_right = mul_right, mul_left
    if len(il) != n_red and len(il) != n_red + 1:
        return None
    if len(ir) != n_red + 1:
        il, ir = ir, il
        mul_left, mul_right = mul_right, mul_left
    if len(ir) != n_red + 1:
        return None
    for rd in reduction_defids:
        if rd is not None and _expr_contains_defid(ir[0], rd):
            return None
    n_batch = len(il) - n_red
    if n_batch not in (0, 1):
        return None
    if n_batch == 1:
        for rd in reduction_defids:
            if rd is not None and _expr_contains_defid(il[0], rd):
                return None
    ch_idx = il[n_batch]
    if not _expr_contains_defid(ch_idx, reduction_defids[0]):
        return None
    for k in range(1, n_red + 1):
        if not _expr_contains_defid(ir[k], reduction_defids[k - 1]):
            return None
    strides: List[int] = []
    for s in range(n_spatial):
        sp_idx = il[n_batch + 1 + s]
        st = _conv_spatial_stride_from_index(
            sp_idx, reduction_defids[s + 1], _add=_add, _mul=_mul
        )
        if st is None:
            return None
        strides.append(st)
    try:
        input_arr = mul_left.array
        weight_arr = mul_right.array
        if input_arr is not None:
            input_arr = input_arr.accept(backend)
        if weight_arr is not None:
            weight_arr = weight_arr.accept(backend)
    except Exception:
        return None
    if not isinstance(input_arr, np.ndarray) or not isinstance(weight_arr, np.ndarray):
        return None
    if input_arr.ndim != n_batch + n_red or weight_arr.ndim != n_red + 1:
        return None
    c_in_axis = n_batch
    co, c_w = weight_arr.shape[0], weight_arr.shape[1]
    if c_w != n_ci or tuple(weight_arr.shape[2:]) != tuple(kern_sizes):
        return None
    if input_arr.shape[c_in_axis] != n_ci:
        return None
    axes = tuple(range(n_batch + 1, n_batch + 1 + n_spatial))
    window_shape = tuple(kern_sizes)
    strides_t = tuple(strides)
    for ax, ws, insz, st in zip(axes, window_shape, [input_arr.shape[a] for a in axes], strides_t):
        if insz < ws:
            return None
        out_d = (insz - ws) // st + 1
        if out_d < 1:
            return None
    out_spatial = tuple(
        (input_arr.shape[axes[d]] - window_shape[d]) // strides_t[d] + 1
        for d in range(n_spatial)
    )
    out_axis_letters = ("i", "j", "k")
    kernel_axis_letters = ("m", "n", "p")
    if n_spatial > len(out_axis_letters) or n_spatial > len(kernel_axis_letters):
        return None
    win_sub_parts: List[str] = []
    w_sub_parts: List[str] = ["o", "c"]
    out_sub_parts: List[str] = []
    if n_batch:
        win_sub_parts.append("b")
    win_sub_parts.append("c")
    for d in range(n_spatial):
        win_sub_parts.append(out_axis_letters[d])
    for d in range(n_spatial):
        kc = kernel_axis_letters[d]
        win_sub_parts.append(kc)
        w_sub_parts.append(kc)
    if n_batch:
        out_sub_parts.append("b")
    out_sub_parts.append("o")
    for d in range(n_spatial):
        out_sub_parts.append(out_axis_letters[d])
    win_sub = "".join(win_sub_parts)
    w_sub = "".join(w_sub_parts)
    out_sub = "".join(out_sub_parts)
    eq = f"{win_sub},{w_sub}->{out_sub}"
    try:
        windows = _sliding_window_nd_view(
            input_arr, axes=axes, window_shape=window_shape, strides=strides_t
        )
        if windows is None:
            return None
        result = np.einsum(eq, windows, weight_arr, optimize=True)
    except Exception:
        return None
    if n_batch:
        exp_shape = (input_arr.shape[0], co) + out_spatial
    else:
        exp_shape = (co,) + out_spatial
    if result.shape != exp_shape:
        return None
    if bias is not None:
        try:
            result = _apply_optional_bias(result, bias, backend, last_dim_row=False)
        except Exception:
            pass
    return result


def _try_windowed_sumprod_einsum(expr: LoweredReductionIR, backend: Any, plan: Any) -> Optional[Any]:
    """Fast path for Einstein sum-of-products over sliding windows.

    Recognizes 1D (two reduction loops) and 2D/3D conv (three/four loops): channel plus
    spatial kernel axes, valid convolution, dilation 1, strides 1 or 2 per spatial dim.
    """
    from ..shared.types import ReductionOp
    op = expr.operation
    if op != ReductionOp.SUM:
        return None
    if expr.guards or expr.bindings:
        return None
    loops = list(expr.loops or [])
    n_red = len(loops)
    if n_red not in (2, 3, 4):
        return None
    red0_var = loops[0].variable
    if red0_var is None or red0_var.defid is None:
        return None
    red0_defid = red0_var.defid
    if n_red >= 2 and (loops[1].variable is None or loops[1].variable.defid is None):
        return None
    red1_defid = loops[1].variable.defid if n_red >= 2 else None
    try:
        n_ci = int(len(loops[0].iterable.accept(backend)))
        n_k = int(len(loops[1].iterable.accept(backend))) if n_red == 2 else 0
    except Exception:
        return None
    _add = BinaryOp.ADD
    _mul = BinaryOp.MUL
    mul_left = getattr(plan, "left", None)
    mul_right = getattr(plan, "right", None)
    bias = getattr(plan, "bias", None)
    if not isinstance(mul_left, RectangularAccessIR) or not isinstance(mul_right, RectangularAccessIR):
        return None
    if n_red in (3, 4):
        return _try_windowed_sumprod_einsum_spatial_nd(
            expr, backend, loops, mul_left, mul_right, bias, _add=_add, _mul=_mul
        )
    il, ir = mul_left.indices or [], mul_right.indices or []
    if len(il) != 2 or len(ir) != 3:
        il, ir = ir, il
        mul_left, mul_right = mul_right, mul_left
    if len(il) != 2 or len(ir) != 3:
        return None
    if not (_expr_contains_defid(il[0], red0_defid) and _expr_contains_defid(ir[1], red0_defid) and _expr_contains_defid(ir[2], red1_defid)):
        return None
    stride = _conv_spatial_stride_from_index(cast(Any, il[1]), cast(Any, red1_defid), _add=_add, _mul=_mul)
    if stride is None:
        return None
    # Use full arrays (no parallel/reduction indexing) so we get 2D input and 3D weight for im2col + BLAS.
    try:
        input_arr = mul_left.array
        weight_arr = mul_right.array
        if input_arr is not None:
            input_arr = input_arr.accept(backend)
        if weight_arr is not None:
            weight_arr = weight_arr.accept(backend)
    except Exception:
        return None
    if not isinstance(input_arr, np.ndarray) or not isinstance(weight_arr, np.ndarray):
        return None
    if input_arr.ndim != 2 or weight_arr.ndim != 3:
        return None
    C_in, L_in = input_arr.shape
    Co, Cig, K = weight_arr.shape
    if Cig != C_in or K != n_k:
        return None
    L_out = (L_in - K) // stride + 1
    if L_out < 1:
        return None
    try:
        result = _windowed_einsum_reduction(
            input_arr,
            weight_arr,
            window_axis=1,
            window_size=K,
            stride=stride,
            equation="ctk,ock->ot",
        )
        if result is None or result.shape != (Co, L_out):
            return None
    except Exception:
        return None
    if bias is not None:
        try:
            result = _apply_optional_bias(result, bias, backend, last_dim_row=True)
        except Exception:
            pass
    return result
def _index_to_reduction_position(idx: Any, reduction_defids: List[Any]) -> Optional[int]:
    """If index is a simple reduction variable, return its position in reduction_defids; else None."""
    if idx is None or not isinstance(idx, (IdentifierIR, IndexVarIR)):
        return None
    did = idx.defid
    if did is None or did not in reduction_defids:
        return None
    return reduction_defids.index(did)


def _free_key_for_index(idx: Any, reduction_defids: List[Any], side: str, pos: int) -> Any:
    """Return a hashable key for a free index so same variable in left/right gets same key."""
    if idx is None or _index_to_reduction_position(idx, reduction_defids) is not None:
        return None
    if isinstance(idx, (IdentifierIR, IndexVarIR)):
        did = idx.defid
        if did is not None:
            return ("defid", did)
    return (side, pos)


def _slice_array_at_scalar_indices(
    arr: np.ndarray,
    indices: List[Any],
    reduction_defids: List[Any],
    backend: Any,
) -> Tuple[np.ndarray, List[int]]:
    """Slice array at indices that evaluate to scalar; keep dimensions for array-valued indices.
    Returns (sliced_array, kept_positions) where kept_positions are index positions that were not
    sliced to a scalar (so subscript letters for those positions are still needed).
    E.g. ln1[0, s, d] with s array, d reduction -> slice at 0 only -> (197, 192), kept=[1,2]."""
    if arr.ndim != len(indices):
        return arr, list(range(arr.ndim))
    key: List[Any] = []
    kept: List[int] = []
    for pos, idx in enumerate(indices):
        if _index_to_reduction_position(idx, reduction_defids) is not None:
            key.append(slice(None))
            kept.append(pos)
        else:
            try:
                v = idx.accept(backend)
                if _is_scalar_or_0d_array(v):
                    key.append(int(v))
                else:
                    key.append(slice(None))
                    kept.append(pos)
            except Exception:
                return arr, list(range(arr.ndim))
    try:
        return arr[tuple(key)], kept
    except Exception:
        return arr, list(range(arr.ndim))


def _remap_axes_after_scalar_slicing(
    axes: Optional[Tuple[int, ...]],
    kept_positions: List[int],
) -> Optional[Tuple[int, ...]]:
    """Remap original array axes after scalar-index slicing removes dimensions."""
    if axes is None:
        return None
    axis_map = {old_axis: new_axis for new_axis, old_axis in enumerate(kept_positions)}
    remapped: List[int] = []
    for axis in axes:
        if axis not in axis_map:
            return None
        remapped.append(axis_map[axis])
    return tuple(remapped)


def _try_einsum_reduction(expr: LoweredReductionIR, backend: Any, plan: Any) -> Optional[Any]:
    """Generic sum-of-product fast path: sum over (left * right [+ bias]) lowered to np.einsum.
    Supports any number of reduction dims; indices must be simple (IdentifierIR/IndexVarIR) on reduction dims.
    NumPy einsum uses BLAS where applicable (e.g. matrix multiply)."""
    from ..shared.types import ReductionOp
    op = expr.operation
    if op != ReductionOp.SUM:
        return None
    if expr.guards or expr.bindings:
        return None
    loops = list(expr.loops or [])
    if not loops:
        return None
    reduction_defids: List[Any] = []
    reduction_sizes: List[int] = []
    for loop in loops:
        loop_var = loop.variable
        if loop_var is None:
            return None
        loop_defid = loop_var.defid
        if loop_defid is None:
            return None
        reduction_defids.append(loop_defid)
        try:
            iterable = loop.iterable.accept(backend)
            if iterable is None:
                return None
            reduction_sizes.append(int(len(iterable)))
        except Exception:
            return None
    mul_left = getattr(plan, "left", None)
    mul_right = getattr(plan, "right", None)
    bias = getattr(plan, "bias", None)
    if mul_left is None or mul_right is None:
        return None
    if not isinstance(mul_left, RectangularAccessIR) or not isinstance(mul_right, RectangularAccessIR):
        return None
    indices_left = mul_left.indices or []
    indices_right = mul_right.indices or []
    for _idx in indices_left + indices_right:
        for _rd in reduction_defids:
            if _expr_contains_defid(_idx, _rd):
                if not (isinstance(_idx, (IdentifierIR, IndexVarIR)) and _idx.defid in reduction_defids):
                    return None
                break
    n_red = len(reduction_defids)
    reduction_letters = [chr(ord("a") + i) for i in range(min(n_red, 26))]
    if n_red > 26:
        return None
    # Build subscript so same free variable (by defid) in left and right gets same letter (e.g. batched score: hid,hjd->hij).
    free_key_to_letter: Dict[Any, str] = {}
    output_order: List[Any] = []
    next_letter_idx = [0]

    def letter_for_free(key: Any) -> str:
        if key not in free_key_to_letter:
            free_key_to_letter[key] = chr(ord("i") + (next_letter_idx[0] % 26))
            next_letter_idx[0] += 1
            output_order.append(key)
        return free_key_to_letter[key]

    def sub_for_indices(indices: List[Any], side: str) -> List[str]:
        sub: List[str] = []
        for pos, idx in enumerate(indices):
            red_pos = _index_to_reduction_position(idx, reduction_defids)
            if red_pos is not None:
                sub.append(reduction_letters[red_pos])
            else:
                key = _free_key_for_index(idx, reduction_defids, side, pos)
                sub.append(letter_for_free(key))
        return sub

    left_sub_list = sub_for_indices(indices_left, "L")
    right_sub_list = sub_for_indices(indices_right, "R")
    left_sub = "".join(left_sub_list)
    right_sub = "".join(right_sub_list)
    out_sub = "".join(free_key_to_letter[k] for k in output_order)
    # Evaluate base arrays so we get correct shapes when parallel vars are broadcast (saves memory).
    left_arr = mul_left.array
    right_arr = mul_right.array
    try:
        with backend.env.scope():
            for i, (defid, N) in enumerate(zip(reduction_defids, reduction_sizes)):
                if n_red == 1:
                    backend.env.set_value(defid, np.arange(N, dtype=np.intp))
                else:
                    shape = [1] * n_red
                    shape[i] = N
                    backend.env.set_value(defid, np.arange(N, dtype=np.intp).reshape(shape))
            if left_arr is not None:
                left_val = left_arr.accept(backend)
            else:
                left_val = mul_left.accept(backend)
            if right_arr is not None:
                right_val = right_arr.accept(backend)
            else:
                right_val = mul_right.accept(backend)
    except Exception:
        return None
    if not isinstance(left_val, np.ndarray) or not isinstance(right_val, np.ndarray):
        return None
    left_val, kept_left = _slice_array_at_scalar_indices(left_val, indices_left, reduction_defids, backend)
    right_val, kept_right = _slice_array_at_scalar_indices(right_val, indices_right, reduction_defids, backend)
    left_sub = "".join(left_sub_list[i] for i in kept_left)
    right_sub = "".join(right_sub_list[i] for i in kept_right)
    out_sub = "".join(
        free_key_to_letter[k]
        for k in output_order
        if free_key_to_letter[k] in left_sub or free_key_to_letter[k] in right_sub
    )
    try:
        result = np.einsum(f"{left_sub},{right_sub}->{out_sub}", left_val, right_val, optimize=True)
    except Exception:
        return None
    if bias is not None:
        try:
            result = _apply_optional_bias(result, bias, backend)
        except Exception:
            return None
    return result


def _binary_and(visitor: "ExpressionVisitorMixin", left: Any, right: Any) -> Any:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.logical_and(left, right)
    return visitor._to_bool(left) and visitor._to_bool(right)


def _binary_or(visitor: "ExpressionVisitorMixin", left: Any, right: Any) -> Any:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.logical_or(left, right)
    return visitor._to_bool(left) or visitor._to_bool(right)


def _unary_not(visitor: "ExpressionVisitorMixin", operand: Any) -> Any:
    if isinstance(operand, np.ndarray):
        return np.logical_not(operand)
    return not visitor._to_bool(operand)


def _safe_true_divide(l, r):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.true_divide(l, r)


def _safe_mod(l, r):
    with np.errstate(divide="ignore", invalid="ignore"):
        return l % r


def _safe_eq(v, l, r):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=DeprecationWarning)
        try:
            return l == r
        except (DeprecationWarning, TypeError):
            return False


def _safe_ne(v, l, r):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=DeprecationWarning)
        try:
            return l != r
        except (DeprecationWarning, TypeError):
            return True


def _both_integer(l, r):
    def _is_int(x):
        if isinstance(x, (int, np.integer)):
            return True
        if isinstance(x, np.ndarray):
            return np.issubdtype(x.dtype, np.integer)
        return False
    return _is_int(l) and _is_int(r)


_BINARY_OP_MAP = {
    BinaryOp.ADD: lambda v, l, r: l + r,
    BinaryOp.SUB: lambda v, l, r: l - r,
    BinaryOp.MUL: lambda v, l, r: l * r,
    BinaryOp.DIV: lambda v, l, r: l // r if _both_integer(l, r) else _safe_true_divide(l, r),
    BinaryOp.MOD: lambda v, l, r: _safe_mod(l, r),
    BinaryOp.POW: lambda v, l, r: l ** r,
    BinaryOp.EQ: _safe_eq,
    BinaryOp.NE: _safe_ne,
    BinaryOp.LT: lambda v, l, r: l < r,
    BinaryOp.LE: lambda v, l, r: l <= r,
    BinaryOp.GT: lambda v, l, r: l > r,
    BinaryOp.GE: lambda v, l, r: l >= r,
    BinaryOp.AND: _binary_and,
    BinaryOp.OR: _binary_or,
    BinaryOp.IN: lambda v, l, r: l in r,
}

_UNARY_OP_MAP = {
    UnaryOp.NEG: lambda v, o: -o,
    UnaryOp.POS: lambda v, o: o,
    UnaryOp.NOT: _unary_not,
    UnaryOp.BOOL_NOT: _unary_not,
}
