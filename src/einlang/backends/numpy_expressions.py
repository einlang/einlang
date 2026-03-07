"""NumPy backend expression visitors. All lookup via env (no global table)."""

from typing import Any, Dict, List, Optional, Tuple
import time
import warnings

import numpy as np

from ..shared.types import BinaryOp, UnaryOp, TypeKind
from ..ir.nodes import (
    LiteralIR, IdentifierIR, IndexVarIR, BinaryOpIR, UnaryOpIR, FunctionCallIR,
    BlockExpressionIR, RangeIR, ArrayComprehensionIR, RectangularAccessIR, JaggedAccessIR,
    ArrayLiteralIR, TupleExpressionIR, TupleAccessIR, InterpolatedStringIR, CastExpressionIR,
    MemberAccessIR, TryExpressionIR, MatchExpressionIR, ReductionExpressionIR, WhereExpressionIR,
    PipelineExpressionIR, BuiltinCallIR,
    MatchArmIR, ExpressionIR, LoweredComprehensionIR, LoweredReductionIR,
)
from ..runtime.environment import FunctionValue
from .numpy_helpers import (
    _reject_non_lowered, _PatternMatcher, _extract_binding,
    builtin_assert, builtin_print, builtin_len, builtin_shape, builtin_typeof,
    builtin_sum, builtin_max, builtin_min, builtin_array_append,
)


def _is_scalar_like(x: Any) -> bool:
    if x is None:
        return True
    if np.isscalar(x):
        return True
    if isinstance(x, np.ndarray):
        return x.ndim == 0 or x.size == 1
    return False


def _reduction_axis_in_access(backend: Any, access: Any, reduction_defid: Any, indices: List[Any]) -> Optional[int]:
    axis = 0
    for idx in indices:
        idx_defid = getattr(idx, "defid", None)
        if idx_defid is not None and idx_defid == reduction_defid:
            return axis
        try:
            v = idx.accept(backend)
            if not _is_scalar_like(v):
                axis += 1
        except Exception:
            return None
    return None


def _expr_contains_defid(expr: Any, target_defid: Any) -> bool:
    if expr is None or target_defid is None:
        return False
    if isinstance(expr, (IdentifierIR, IndexVarIR)) and getattr(expr, "defid", None) == target_defid:
        return True
    if hasattr(expr, "left") and hasattr(expr, "right"):
        if _expr_contains_defid(getattr(expr, "left", None), target_defid):
            return True
        if _expr_contains_defid(getattr(expr, "right", None), target_defid):
            return True
    if hasattr(expr, "operand"):
        return _expr_contains_defid(getattr(expr, "operand"), target_defid)
    if hasattr(expr, "array") and hasattr(expr, "indices"):
        if _expr_contains_defid(getattr(expr, "array"), target_defid):
            return True
        for i in getattr(expr, "indices", []) or []:
            if _expr_contains_defid(i, target_defid):
                return True
    if hasattr(expr, "arguments"):
        for a in getattr(expr, "arguments", []) or []:
            if _expr_contains_defid(a, target_defid):
                return True
    if hasattr(expr, "callee_expr") and _expr_contains_defid(getattr(expr, "callee_expr"), target_defid):
        return True
    return False


def _reduction_axes_in_access(
    backend: Any, indices: List[Any], reduction_defids: List[Any]
) -> Optional[Tuple[int, ...]]:
    """Return the array axis index for each reduction defid (position in indices = axis in array)."""
    axes: List[Optional[int]] = [None] * len(reduction_defids)
    for axis_in_array, idx in enumerate(indices):
        idx_defid = getattr(idx, "defid", None)
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


def _try_matmul_reduction(expr: LoweredReductionIR, backend: Any) -> Optional[Any]:
    op = getattr(expr, "operation", None)
    if op != "sum":
        return None
    if getattr(expr, "guards", None) or getattr(expr, "bindings", None):
        return None
    from ..passes.einstein_lowering import _defid_of_var_in_expr
    reduction_ranges = getattr(expr, "reduction_ranges", None) or {}
    loops = list(reduction_ranges.values()) if isinstance(reduction_ranges, dict) else []
    if not loops:
        return None
    reduction_defids: List[Any] = []
    reduction_sizes: List[int] = []
    for loop in loops:
        loop_var = getattr(loop, "variable", None)
        if loop_var is None:
            return None
        loop_defid = getattr(loop_var, "defid", None)
        if loop_defid is None:
            return None
        body_defid = _defid_of_var_in_expr(expr.body, getattr(loop_var, "name", "") or "") or loop_defid
        reduction_defids.append(body_defid)
        try:
            iterable = loop.iterable.accept(backend) if hasattr(loop.iterable, "accept") else None
            if iterable is None:
                return None
            reduction_sizes.append(int(len(iterable)))
        except Exception:
            return None
    body = getattr(expr, "body", None)
    if body is None:
        return None
    mul_left: Optional[Any] = None
    mul_right: Optional[Any] = None
    bias: Optional[Any] = None
    scale: Optional[float] = None
    _add = getattr(BinaryOp, "ADD", None) or "+"
    _mul = getattr(BinaryOp, "MUL", None) or "*"
    _div = getattr(BinaryOp, "DIV", None) or "/"
    body_op = getattr(body, "operator", None)
    if isinstance(body, BinaryOpIR) and body_op in (_add, "+"):
        add_left = getattr(body, "left", None)
        add_right = getattr(body, "right", None)
        if isinstance(add_left, BinaryOpIR) and getattr(add_left, "operator", None) in (_mul, "*"):
            mul_left = getattr(add_left, "left", None)
            mul_right = getattr(add_left, "right", None)
            bias = add_right
        elif isinstance(add_right, BinaryOpIR) and getattr(add_right, "operator", None) in (_mul, "*"):
            mul_left = getattr(add_right, "left", None)
            mul_right = getattr(add_right, "right", None)
            bias = add_left
    elif isinstance(body, BinaryOpIR) and body_op in (_mul, "*"):
        bl = getattr(body, "left", None)
        br = getattr(body, "right", None)
        if isinstance(bl, RectangularAccessIR) and isinstance(br, RectangularAccessIR):
            mul_left, mul_right = bl, br
        elif isinstance(bl, BinaryOpIR) and getattr(bl, "operator", None) in (_mul, "*"):
            al, ar = getattr(bl, "left", None), getattr(bl, "right", None)
            if isinstance(al, RectangularAccessIR) and isinstance(ar, RectangularAccessIR) and isinstance(br, LiteralIR):
                mul_left, mul_right = al, ar
                try:
                    scale = float(getattr(br, "value", None))
                except (TypeError, ValueError):
                    scale = None
        elif isinstance(br, BinaryOpIR) and getattr(br, "operator", None) in (_mul, "*"):
            al, ar = getattr(br, "left", None), getattr(br, "right", None)
            if isinstance(al, RectangularAccessIR) and isinstance(ar, RectangularAccessIR) and isinstance(bl, LiteralIR):
                mul_left, mul_right = al, ar
                try:
                    scale = float(getattr(bl, "value", None))
                except (TypeError, ValueError):
                    scale = None
    elif isinstance(body, BinaryOpIR) and body_op in (_div, "/"):
        div_left = getattr(body, "left", None)
        div_right = getattr(body, "right", None)
        if isinstance(div_left, BinaryOpIR) and getattr(div_left, "operator", None) in (_mul, "*"):
            al = getattr(div_left, "left", None)
            ar = getattr(div_left, "right", None)
            if isinstance(al, RectangularAccessIR) and isinstance(ar, RectangularAccessIR):
                mul_left, mul_right = al, ar
                try:
                    if isinstance(div_right, LiteralIR):
                        v = float(getattr(div_right, "value", None))
                        if v != 0.0:
                            scale = 1.0 / v
                except (TypeError, ValueError):
                    pass
    if mul_left is None or mul_right is None:
        return None
    if not isinstance(mul_left, RectangularAccessIR) or not isinstance(mul_right, RectangularAccessIR):
        return None
    indices_left = getattr(mul_left, "indices", None) or []
    indices_right = getattr(mul_right, "indices", None) or []
    from ..ir.nodes import IdentifierIR, IndexVarIR
    for _idx in indices_left + indices_right:
        for _rd in reduction_defids:
            if _expr_contains_defid(_idx, _rd):
                if not (isinstance(_idx, (IdentifierIR, IndexVarIR)) and getattr(_idx, "defid", None) in reduction_defids):
                    return None
    n_red = len(reduction_sizes)
    # GEMM-style: 1 or 2 reduction dims (QKV / batched matmul, or full matmul). Recurrence clauses skip this path.
    if n_red not in (1, 2):
        return None
    # Evaluate base arrays (not indexed) so we get correct shapes for BLAS when parallel
    # loop vars are already set to broadcast arrays (avoids huge intermediate in vectorized path).
    left_arr = getattr(mul_left, "array", None)
    right_arr = getattr(mul_right, "array", None)
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
            if left_arr is not None and hasattr(left_arr, "accept"):
                left_val = left_arr.accept(backend)
            else:
                left_val = mul_left.accept(backend)
            if right_arr is not None and hasattr(right_arr, "accept"):
                right_val = right_arr.accept(backend)
            else:
                right_val = mul_right.accept(backend)
    except Exception:
        return None
    if not isinstance(left_val, np.ndarray) or not isinstance(right_val, np.ndarray):
        return None
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
                        if np.isscalar(v) or (isinstance(v, np.ndarray) and getattr(v, "ndim", -1) == 0):
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
            bias_val = bias.accept(backend)
            if isinstance(bias_val, np.ndarray) and isinstance(result, np.ndarray):
                result = result + np.broadcast_to(bias_val, result.shape)
            elif np.isscalar(bias_val) or (isinstance(bias_val, np.ndarray) and bias_val.ndim == 0):
                result = result + bias_val
        except Exception:
            return None
    if scale is not None:
        result = result * scale
    return result


def _try_conv_im2col_einsum(expr: LoweredReductionIR, backend: Any) -> Optional[Any]:
    """Strict 1D conv fast path: sum[ci,k](input[ci, stride*t+k] * weight[co,ci,k]) + bias, stride 1 or 2.
    Uses im2col unfold + np.einsum. Returns None if pattern does not match."""
    op = getattr(expr, "operation", None)
    if op != "sum":
        return None
    if getattr(expr, "guards", None) or getattr(expr, "bindings", None):
        return None
    from ..passes.einstein_lowering import _defid_of_var_in_expr
    reduction_ranges = getattr(expr, "reduction_ranges", None) or {}
    loops = list(reduction_ranges.values()) if isinstance(reduction_ranges, dict) else []
    if len(loops) != 2:
        return None
    red0_var = getattr(loops[0], "variable", None)
    red1_var = getattr(loops[1], "variable", None)
    if red0_var is None or red1_var is None:
        return None
    red0_defid = getattr(red0_var, "defid", None)
    red1_defid = getattr(red1_var, "defid", None)
    if red0_defid is None or red1_defid is None:
        return None
    try:
        n_ci = int(len(loops[0].iterable.accept(backend)))
        n_k = int(len(loops[1].iterable.accept(backend)))
    except Exception:
        return None
    body = getattr(expr, "body", None)
    if body is None:
        return None
    _add = getattr(BinaryOp, "ADD", None) or "+"
    _mul = getattr(BinaryOp, "MUL", None) or "*"
    mul_left: Optional[RectangularAccessIR] = None
    mul_right: Optional[RectangularAccessIR] = None
    bias: Optional[Any] = None
    body_op = getattr(body, "operator", None)
    if isinstance(body, BinaryOpIR) and body_op in (_add, "+"):
        add_left = getattr(body, "left", None)
        add_right = getattr(body, "right", None)
        if isinstance(add_left, BinaryOpIR) and getattr(add_left, "operator", None) in (_mul, "*"):
            mul_left = getattr(add_left, "left", None)
            mul_right = getattr(add_left, "right", None)
            bias = add_right
        elif isinstance(add_right, BinaryOpIR) and getattr(add_right, "operator", None) in (_mul, "*"):
            mul_left = getattr(add_right, "left", None)
            mul_right = getattr(add_right, "right", None)
            bias = add_left
    elif isinstance(body, BinaryOpIR) and body_op in (_mul, "*"):
        mul_left = getattr(body, "left", None)
        mul_right = getattr(body, "right", None)
    if not isinstance(mul_left, RectangularAccessIR) or not isinstance(mul_right, RectangularAccessIR):
        return None
    il, ir = getattr(mul_left, "indices", None) or [], getattr(mul_right, "indices", None) or []
    if len(il) != 2 or len(ir) != 3:
        il, ir = ir, il
        mul_left, mul_right = mul_right, mul_left
    if len(il) != 2 or len(ir) != 3:
        return None
    if not (_expr_contains_defid(il[0], red0_defid) and _expr_contains_defid(ir[1], red0_defid) and _expr_contains_defid(ir[2], red1_defid)):
        return None
    second_idx = il[1]
    stride = 1
    if isinstance(second_idx, BinaryOpIR):
        add_op = getattr(second_idx, "operator", None)
        if add_op in (_add, "+"):
            left, right = getattr(second_idx, "left", None), getattr(second_idx, "right", None)
            if _expr_contains_defid(left, red1_defid) and _expr_contains_defid(right, red1_defid):
                return None
            if _expr_contains_defid(right, red1_defid):
                left, right = right, left
            if not _expr_contains_defid(left, red1_defid):
                return None
            # Stride from the t part (right): t*stride -> stride 1 or 2; plain t -> stride 1
            if isinstance(right, (IdentifierIR, IndexVarIR)):
                stride = 1
            elif isinstance(right, BinaryOpIR) and getattr(right, "operator", None) in (_mul, "*"):
                try:
                    rL, rR = getattr(right, "left", None), getattr(right, "right", None)
                    if isinstance(rR, LiteralIR):
                        stride = int(rR.value)
                    elif isinstance(rL, LiteralIR):
                        stride = int(rL.value)
                    else:
                        return None
                    if stride not in (1, 2):
                        return None
                except (TypeError, ValueError):
                    return None
            else:
                return None
    else:
        if not (isinstance(second_idx, (IdentifierIR, IndexVarIR)) and getattr(second_idx, "defid", None) == red1_defid):
            return None
    # Use full arrays (no parallel/reduction indexing) so we get 2D input and 3D weight for im2col + BLAS.
    try:
        input_arr = getattr(mul_left, "array", None)
        weight_arr = getattr(mul_right, "array", None)
        if input_arr is not None and hasattr(input_arr, "accept"):
            input_arr = input_arr.accept(backend)
        if weight_arr is not None and hasattr(weight_arr, "accept"):
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
        unfolded = np.empty((C_in, L_out, K), dtype=input_arr.dtype)
        for t in range(L_out):
            start = t * stride
            unfolded[:, t, :] = input_arr[:, start : start + K]
        # BLAS-friendly: einsum "ctk,ock->ot" is batched matmul on last two dims
        result = np.einsum("ctk,ock->ot", unfolded, weight_arr, optimize=True)
    except Exception:
        return None
    if bias is not None:
        try:
            bias_val = bias.accept(backend)
            if isinstance(bias_val, np.ndarray) and bias_val.size == result.shape[-1]:
                result = result + np.reshape(bias_val, (1, -1))
            elif np.isscalar(bias_val) or (isinstance(bias_val, np.ndarray) and bias_val.ndim == 0):
                result = result + bias_val
        except Exception:
            pass
    return result


def _index_to_reduction_position(idx: Any, reduction_defids: List[Any]) -> Optional[int]:
    """If index is a simple reduction variable, return its position in reduction_defids; else None."""
    if idx is None or not isinstance(idx, (IdentifierIR, IndexVarIR)):
        return None
    did = getattr(idx, "defid", None)
    if did is None or did not in reduction_defids:
        return None
    return reduction_defids.index(did)


def _free_key_for_index(idx: Any, reduction_defids: List[Any], side: str, pos: int) -> Any:
    """Return a hashable key for a free index so same variable in left/right gets same key."""
    if idx is None or _index_to_reduction_position(idx, reduction_defids) is not None:
        return None
    if isinstance(idx, (IdentifierIR, IndexVarIR)):
        did = getattr(idx, "defid", None)
        if did is not None:
            return ("defid", did)
    return (side, pos)


def _slice_array_at_scalar_indices(
    arr: np.ndarray,
    indices: List[Any],
    reduction_defids: List[Any],
    backend: Any,
) -> Tuple[np.ndarray, List[int]]:
    """Slice array at any non-reduction index that evaluates to a scalar (e.g. W[L,d,k] with L scalar -> W[L,:,:]).
    Returns (sliced_array, kept_positions) where kept_positions are index positions that were not sliced (for subscript rebuild).
    If any non-reduction index is non-scalar, returns (arr, list(range(arr.ndim))) unchanged."""
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
                if np.isscalar(v) or (isinstance(v, np.ndarray) and getattr(v, "ndim", -1) == 0):
                    key.append(int(v))
                else:
                    return arr, list(range(arr.ndim))
            except Exception:
                return arr, list(range(arr.ndim))
    try:
        return arr[tuple(key)], kept
    except Exception:
        return arr, list(range(arr.ndim))


def _try_einsum_reduction(expr: LoweredReductionIR, backend: Any) -> Optional[Any]:
    """Generic sum-of-product fast path: sum over (left * right [+ bias]) lowered to np.einsum.
    Supports any number of reduction dims; indices must be simple (IdentifierIR/IndexVarIR) on reduction dims.
    NumPy einsum uses BLAS where applicable (e.g. matrix multiply)."""
    op = getattr(expr, "operation", None)
    if op != "sum":
        return None
    if getattr(expr, "guards", None) or getattr(expr, "bindings", None):
        return None
    from ..passes.einstein_lowering import _defid_of_var_in_expr
    reduction_ranges = getattr(expr, "reduction_ranges", None) or {}
    loops = list(reduction_ranges.values()) if isinstance(reduction_ranges, dict) else []
    if not loops:
        return None
    reduction_defids: List[Any] = []
    reduction_sizes: List[int] = []
    for loop in loops:
        loop_var = getattr(loop, "variable", None)
        if loop_var is None:
            return None
        loop_defid = getattr(loop_var, "defid", None)
        if loop_defid is None:
            return None
        body_defid = _defid_of_var_in_expr(expr.body, getattr(loop_var, "name", "") or "") or loop_defid
        reduction_defids.append(body_defid)
        try:
            iterable = loop.iterable.accept(backend) if hasattr(loop.iterable, "accept") else None
            if iterable is None:
                return None
            reduction_sizes.append(int(len(iterable)))
        except Exception:
            return None
    body = getattr(expr, "body", None)
    if body is None:
        return None
    _add = getattr(BinaryOp, "ADD", None) or "+"
    _mul = getattr(BinaryOp, "MUL", None) or "*"
    mul_left: Optional[RectangularAccessIR] = None
    mul_right: Optional[RectangularAccessIR] = None
    bias: Optional[Any] = None
    body_op = getattr(body, "operator", None)
    if isinstance(body, BinaryOpIR) and body_op in (_add, "+"):
        add_left = getattr(body, "left", None)
        add_right = getattr(body, "right", None)
        if isinstance(add_left, BinaryOpIR) and getattr(add_left, "operator", None) in (_mul, "*"):
            mul_left = getattr(add_left, "left", None)
            mul_right = getattr(add_left, "right", None)
            bias = add_right
        elif isinstance(add_right, BinaryOpIR) and getattr(add_right, "operator", None) in (_mul, "*"):
            mul_left = getattr(add_right, "left", None)
            mul_right = getattr(add_right, "right", None)
            bias = add_left
    elif isinstance(body, BinaryOpIR) and body_op in (_mul, "*"):
        mul_left = getattr(body, "left", None)
        mul_right = getattr(body, "right", None)
    if mul_left is None or mul_right is None:
        return None
    if not isinstance(mul_left, RectangularAccessIR) or not isinstance(mul_right, RectangularAccessIR):
        return None
    indices_left = getattr(mul_left, "indices", None) or []
    indices_right = getattr(mul_right, "indices", None) or []
    for _idx in indices_left + indices_right:
        for _rd in reduction_defids:
            if _expr_contains_defid(_idx, _rd):
                if not (isinstance(_idx, (IdentifierIR, IndexVarIR)) and getattr(_idx, "defid", None) in reduction_defids):
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
    left_arr = getattr(mul_left, "array", None)
    right_arr = getattr(mul_right, "array", None)
    try:
        with backend.env.scope():
            for i, (defid, N) in enumerate(zip(reduction_defids, reduction_sizes)):
                if n_red == 1:
                    backend.env.set_value(defid, np.arange(N, dtype=np.intp))
                else:
                    shape = [1] * n_red
                    shape[i] = N
                    backend.env.set_value(defid, np.arange(N, dtype=np.intp).reshape(shape))
            if left_arr is not None and hasattr(left_arr, "accept"):
                left_val = left_arr.accept(backend)
            else:
                left_val = mul_left.accept(backend)
            if right_arr is not None and hasattr(right_arr, "accept"):
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
            bias_val = bias.accept(backend)
            if isinstance(bias_val, np.ndarray) and isinstance(result, np.ndarray):
                result = result + np.broadcast_to(bias_val, result.shape)
            elif np.isscalar(bias_val) or (isinstance(bias_val, np.ndarray) and bias_val.ndim == 0):
                result = result + bias_val
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

class ExpressionVisitorMixin:
    """Expression visit_*; function/builtin lookup via env only."""

    def _raise_here(self, exc: Exception, expr) -> None:
        """Re-raise *exc* as an EinlangSourceError pinned to *expr*.location (or exc.clause_location if set)."""
        from ..shared.errors import EinlangSourceError
        if isinstance(exc, EinlangSourceError):
            raise
        clause_loc = getattr(exc, "clause_location", None)
        loc = clause_loc if (clause_loc and (getattr(clause_loc, "line", 0) or getattr(clause_loc, "file", ""))) else getattr(expr, "location", None)
        source_code = None
        tcx = getattr(self, "_tcx", None)
        if tcx and loc:
            sf = getattr(tcx, "source_files", None)
            if sf and getattr(loc, "file", None):
                source_code = sf.get(loc.file)
        raise EinlangSourceError(
            message=str(exc),
            location=loc,
            error_code="E0007",
            category="runtime",
            source_code=source_code,
        ) from exc

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, np.ndarray):
            return bool(value.all())
        return bool(value)

    def visit_literal(self, expr: LiteralIR) -> Any:
        val = expr.value
        if isinstance(val, range):
            size = len(val)
            if size > 1_000_000:
                raise RuntimeError(f"Loop range too large: size={size}")
            return val
        return val

    def visit_identifier(self, expr) -> Any:
        from ..shared.defid import DefId
        defid = getattr(expr, "defid", None)
        if defid is None:
            raise RuntimeError(f"Variable not found (defid=None). Name (log): {getattr(expr, 'name', '?')}")
        value = self.env.get_value(defid)
        if value is None:
            raise RuntimeError(f"Variable not found (defid={defid}). Name (log): {getattr(expr, 'name', '?')}")
        return value

    def visit_index_var(self, expr) -> Any:
        defid = getattr(expr, "defid", None)
        if defid is None:
            raise RuntimeError(f"Index variable has no DefId. Name (log): {getattr(expr, 'name', '?')}")
        value = self.env.get_value(defid)
        if value is None:
            raise RuntimeError(f"Index variable not found (defid={defid}). Name (log): {getattr(expr, 'name', '?')}")
        return value

    def visit_binary_op(self, expr: BinaryOpIR) -> Any:
        left = expr.left.accept(self)
        right = expr.right.accept(self)
        op = getattr(expr, "operator", None)
        fn = _BINARY_OP_MAP.get(op) if isinstance(op, BinaryOp) else None
        if fn is None:
            raise RuntimeError(f"Unknown operator: {getattr(expr, 'operator', None)}")
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            if op != BinaryOp.IN:
                if left.ndim != right.ndim:
                    if left.ndim < right.ndim:
                        left = np.reshape(left, left.shape + (1,) * (right.ndim - left.ndim))
                    else:
                        right = np.reshape(right, right.shape + (1,) * (left.ndim - right.ndim))
                if left.shape != right.shape:
                    left, right = np.broadcast_arrays(left, right)
        try:
            return fn(self, left, right)
        except (ZeroDivisionError, FloatingPointError) as e:
            self._raise_here(e, getattr(expr, "right", expr))
        except Exception as e:
            self._raise_here(e, expr)

    def visit_unary_op(self, expr: UnaryOpIR) -> Any:
        operand = expr.operand.accept(self)
        op = getattr(expr, "operator", None)
        fn = _UNARY_OP_MAP.get(op) if isinstance(op, UnaryOp) else None
        if fn is None:
            raise RuntimeError(f"Unknown unary operator: {getattr(expr, 'operator', None)}")
        try:
            return fn(self, operand)
        except Exception as e:
            self._raise_here(e, expr)

    def visit_function_call(self, expr: FunctionCallIR) -> Any:
        try:
            return self._visit_function_call_inner(expr)
        except Exception as e:
            self._raise_here(e, expr)

    def _visit_function_call_inner(self, expr: FunctionCallIR) -> Any:
        module_path = getattr(expr, "module_path", None) or ()
        if expr.function_defid is None and module_path and len(module_path) > 0 and module_path[0] == "python":
            args = [arg.accept(self) for arg in expr.arguments]
            return self._call_python_module(module_path, getattr(expr, "function_name", ""), args)
        callee_expr = getattr(expr, "callee_expr", None)
        if callee_expr is not None:
            callee_value = callee_expr.accept(self)
            if isinstance(callee_value, FunctionValue):
                effective_defid = callee_value.defid
            elif hasattr(callee_value, "parameters") and hasattr(callee_value, "body"):
                effective_defid = getattr(callee_value, "defid", None)
                if effective_defid is not None and self.env.get_value(effective_defid) is None:
                    self.env.set_value(effective_defid, callee_value)
            else:
                raise RuntimeError(f"Callable did not evaluate to function (got {type(callee_value).__name__})")
            if effective_defid is None:
                raise RuntimeError("Callable has no DefId")
            args = [arg.accept(self) for arg in expr.arguments]
            func_def = self.env.get_value(effective_defid)
            if func_def is None:
                raise RuntimeError(f"Function (DefId: {effective_defid}) not found")
            return self._call_function(func_def, args)
        if expr.function_defid is None:
            raise RuntimeError("Function call has no DefId")
        args = [arg.accept(self) for arg in expr.arguments]
        callee = self.env.get_value(expr.function_defid)
        if callee is None:
            raise RuntimeError(f"Function not found (DefId: {expr.function_defid})")
        if isinstance(callee, FunctionValue):
            callee = self.env.get_value(callee.defid)
            if callee is None:
                raise RuntimeError(f"Lambda/function (DefId: {expr.function_defid}) not found")
        if hasattr(callee, "body") and hasattr(callee, "parameters"):
            return self._call_function(callee, args)
        if callee == builtin_assert:
            if len(args) == 0:
                raise RuntimeError("assert() called with no arguments")
            if len(args) == 1:
                return builtin_assert(args[0])
            return builtin_assert(args[0], args[1])
        return callee(*args)

    def visit_rectangular_access(self, expr: RectangularAccessIR) -> Any:
        array = expr.array.accept(self)
        indices = [idx.accept(self) for idx in (expr.indices or []) if idx is not None]
        try:
            if isinstance(array, np.ndarray):
                return array[tuple(indices)]
            if isinstance(array, (list, tuple, str)):
                idx = indices[0] if indices else 0
                return array[int(idx)]
        except (IndexError, KeyError) as e:
            self._raise_here(e, expr)
        raise RuntimeError(f"rectangular_access: expected ndarray, list, or str, got {type(array).__name__}")

    def visit_jagged_access(self, expr: JaggedAccessIR) -> Any:
        array = expr.base.accept(self)
        for idx in (getattr(expr, 'index_chain', None) or []):
            array = array[idx.accept(self)]
        return array

    def visit_block_expression(self, expr: BlockExpressionIR) -> Any:
        with self.env.scope():
            for stmt in getattr(expr, "statements", []) or []:
                result_value = stmt.accept(self)
                binding = getattr(stmt, "_binding", None)
                variable_defid = getattr(binding, "defid", None) if binding else None
                if variable_defid is not None:
                    var_name = getattr(binding, "name", None) or getattr(stmt, "name", None)
                    self.env.set_value(variable_defid, result_value, name=var_name)
            if getattr(expr, "final_expr", None):
                return expr.final_expr.accept(self)
        return None

    def visit_if_expression(self, expr) -> Any:
        cond = expr.condition.accept(self)
        if isinstance(cond, np.ndarray) and cond.ndim > 0:
            then_val = expr.then_expr.accept(self)
            else_val = expr.else_expr.accept(self) if getattr(expr, "else_expr", None) else None
            return np.where(cond, then_val, else_val)
        if self._to_bool(cond):
            return expr.then_expr.accept(self)
        if getattr(expr, "else_expr", None):
            return expr.else_expr.accept(self)
        return None

    def visit_lambda(self, expr) -> Any:
        defid = getattr(expr, "defid", None)
        if defid is None:
            resolver = getattr(self, "resolver", None)
            if resolver is not None:
                defid = resolver.allocate_for_local()
            else:
                raise RuntimeError("Lambda has no DefId and backend has no resolver to allocate one")
        self.env.set_value(defid, expr)
        return FunctionValue(defid=defid, closure_env=self.env)

    def visit_range(self, expr: RangeIR) -> Any:
        start = expr.start.accept(self)
        end = expr.end.accept(self)
        end_int = int(end) + 1 if getattr(expr, 'inclusive', False) else int(end)
        return range(int(start), end_int)

    def visit_array_comprehension(self, expr: ArrayComprehensionIR) -> Any:
        _reject_non_lowered(type(expr).__name__)

    def visit_lowered_comprehension(self, expr: LoweredComprehensionIR) -> Any:
        from ..runtime.compute.lowered_execution import execute_lowered_loops, check_lowered_guards
        results = []
        def ev(e): return e.accept(self)
        for context in execute_lowered_loops(expr.loops, {}, ev):
            with self.env.scope():
                for defid, val in context.items():
                    if defid is not None:
                        self.env.set_value(defid, val)
                full = {}
                for binding in getattr(expr, "bindings", []) or []:
                    defid = getattr(binding, "defid", None)
                    if defid is not None:
                        val = binding.expr.accept(self)
                        full[defid] = val
                        self.env.set_value(defid, val)
                if not (expr.guards and not check_lowered_guards(expr.guards, full, lambda c: self._to_bool(c.accept(self)))):
                    results.append(expr.body.accept(self))
        return np.array(results) if results else np.array([])

    def visit_array_literal(self, expr: ArrayLiteralIR) -> Any:
        type_info = getattr(expr, "type_info", None)
        if type_info is not None and getattr(type_info, "kind", None) == TypeKind.JAGGED:
            evaluated = [e.accept(self) for e in expr.elements]
            return list(evaluated)
        dtype = None
        if type_info is not None:
            converter = getattr(self, "_type_info_to_numpy_dtype", None)
            if callable(converter):
                el = getattr(type_info, "element_type", None) or type_info
                dtype = converter(el)
        if dtype is None and expr.elements:
            for e in expr.elements:
                v = getattr(e, "value", None)
                if v is not None and isinstance(v, (float, np.floating)):
                    dtype = np.float32
                    break
        evaluated = [e.accept(self) for e in expr.elements]
        if dtype is None and evaluated:
            if isinstance(evaluated[0], (float, np.floating)):
                dtype = np.float32
            elif isinstance(evaluated[0], (int, np.integer)) and not isinstance(evaluated[0], (bool, np.bool_)):
                if any(isinstance(x, (float, np.floating)) for x in evaluated):
                    dtype = np.float32
        if dtype is not None:
            return np.array(evaluated, dtype=dtype)
        return np.array(evaluated)

    def visit_tuple_expression(self, expr: TupleExpressionIR) -> Any:
        return tuple(e.accept(self) for e in expr.elements)

    def visit_tuple_access(self, expr: TupleAccessIR) -> Any:
        t = expr.tuple_expr.accept(self)
        return t[expr.index]

    def visit_interpolated_string(self, expr: InterpolatedStringIR) -> Any:
        parts = []
        for part in getattr(expr, "parts", []) or []:
            parts.append(str(part.accept(self)) if hasattr(part, "accept") else str(part))
        return "".join(parts)

    def visit_cast_expression(self, expr: CastExpressionIR) -> Any:
        inner = getattr(expr, "expr", None) or getattr(expr, "operand", None)
        if inner is None:
            raise RuntimeError("CastExpressionIR has no expr or operand")
        val = inner.accept(self)
        target = getattr(expr, "target_type", None)
        if target is None:
            return val
        name = getattr(target, "name", None) or (target if isinstance(target, str) else None)
        _cast_dtype = self._resolve_cast_dtype(name)
        if _cast_dtype is not None:
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                return val.astype(_cast_dtype)
            if name == "bool":
                return bool(val)
            if name in ("i8", "i32", "i64"):
                return int(val)
            return float(val)
        elem_type = getattr(target, "element_type", None)
        if elem_type is not None and val is not None:
            elem_name = getattr(elem_type, "name", None) or (elem_type if isinstance(elem_type, str) else None)
            _elem_dtype = self._resolve_cast_dtype(elem_name)
            if _elem_dtype is not None:
                return np.asarray(val, dtype=_elem_dtype)
        return val

    @staticmethod
    def _resolve_cast_dtype(name):
        _CAST_DTYPES = {
            "i8": np.int8, "i32": np.int32, "i64": np.int64,
            "f16": np.float16, "f32": np.float32, "f64": np.float64,
            "bool": np.bool_,
        }
        dt = _CAST_DTYPES.get(name)
        if dt is not None:
            return dt
        try:
            import ml_dtypes
            _ML_DTYPES = {"bf16": ml_dtypes.bfloat16, "f8e4m3": ml_dtypes.float8_e4m3fn}
            return _ML_DTYPES.get(name)
        except ImportError:
            return None

    def _call_python_module(
        self,
        module_path: tuple,
        function_name: str,
        args: List[Any],
    ) -> Any:
        import importlib
        parts = list(module_path)
        if parts and parts[0] == "python":
            parts = parts[1:]
        if not parts:
            raise RuntimeError(f"Invalid module path for Python module call: {module_path}")
        module_name = ".".join(parts)
        if module_path == ("python", "builtins") or module_path == ("builtins",):
            import builtins
            callable_func = getattr(builtins, function_name, None)
            if callable_func is None:
                raise RuntimeError(f"Python builtin '{function_name}' not found")
            return callable_func(*args)
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise RuntimeError(f"Failed to import Python module '{module_name}': {e}")
        value = getattr(module, function_name, None)
        if value is None:
            raise RuntimeError(
                f"Function/property '{function_name}' not found in Python module '{module_name}'"
            )
        if not callable(value):
            if len(args) == 0:
                return value
            raise RuntimeError(
                f"Cannot call non-callable property {module_name}::{function_name}: {type(value)}"
            )
        with np.errstate(all="ignore"):
            return value(*args)

    def visit_member_access(self, expr: MemberAccessIR) -> Any:
        obj = expr.object.accept(self)
        member = getattr(expr, "member", None)
        if member is None:
            return None
        if isinstance(member, str):
            if member == "length" and isinstance(obj, (np.ndarray, list, tuple, str)):
                return len(obj)
            if member == "size" and isinstance(obj, np.ndarray):
                return obj.size
            if member == "shape" and isinstance(obj, np.ndarray):
                return obj.shape
            return getattr(obj, member, None)
        if isinstance(member, int):
            return obj[member]
        key = member.accept(self) if hasattr(member, "accept") else getattr(member, "value", member)
        return obj[key] if key is not None else None

    def visit_try_expression(self, expr: TryExpressionIR) -> Any:
        try:
            return expr.operand.accept(self)
        except Exception:
            raise

    def visit_match_expression(self, expr: MatchExpressionIR) -> Any:
        scrutinee_value = expr.scrutinee.accept(self)
        matcher = _PatternMatcher(scrutinee_value, self)
        for arm in expr.arms:
            bindings = arm.pattern.accept(matcher)
            if bindings is None:
                continue
            if hasattr(arm.pattern, "guard_expr"):
                with self.env.scope():
                    for var_defid, var_value in bindings.items():
                        if var_defid is not None:
                            self.env.set_value(var_defid, var_value)
                    if not self._to_bool(arm.pattern.guard_expr.accept(self)):
                        continue
            with self.env.scope():
                for var_defid, var_value in bindings.items():
                    if var_defid is not None:
                        self.env.set_value(var_defid, var_value)
                return arm.body.accept(self)
        try:
            raise RuntimeError(f"Match not exhaustive: no pattern matched {scrutinee_value}")
        except RuntimeError as e:
            self._raise_here(e, expr)

    def visit_reduction_expression(self, expr: ReductionExpressionIR) -> Any:
        _reject_non_lowered(type(expr).__name__)

    def evaluate_lowered_reduction(
        self, expr: LoweredReductionIR, parallel_shape: Optional[Tuple[int, ...]] = None
    ) -> Any:
        """Evaluate a lowered reduction, optionally with vectorized path when parallel_shape is set.
        When parallel_shape is None, uses backend._vectorize_parallel_shape if set (e.g. by vectorized clause).
        Fast paths (matmul, conv via einsum) only when parallel_shape is set; stricter conditions avoid LSTM."""
        import os
        if parallel_shape is None:
            parallel_shape = getattr(self, "_vectorize_parallel_shape", None)
        # Recurrence clauses may use partial vectorization but must not use fast_matmul / fast_conv.
        if parallel_shape is not None and not getattr(self, "_einstein_recurrence_clause", False):
            conv_result = _try_conv_im2col_einsum(expr, self)
            if conv_result is not None and isinstance(conv_result, np.ndarray):
                if conv_result.shape == tuple(parallel_shape):
                    setattr(self, "_last_reduction_fast_path", "conv")
                    return conv_result
            matmul_result = _try_matmul_reduction(expr, self)
            if matmul_result is not None and isinstance(matmul_result, np.ndarray):
                if matmul_result.shape == tuple(parallel_shape):
                    setattr(self, "_last_reduction_fast_path", "matmul")
                    return matmul_result
            einsum_result = _try_einsum_reduction(expr, self)
            if einsum_result is not None and isinstance(einsum_result, np.ndarray):
                if einsum_result.shape == tuple(parallel_shape):
                    setattr(self, "_last_reduction_fast_path", "einsum")
                    return einsum_result
        from ..runtime.compute.lowered_execution import execute_reduction_with_loops
        from ..passes.einstein_lowering import _defid_of_var_in_expr
        loc = getattr(expr, "location", None)
        line = int(getattr(loc, "line", 0) or 0)
        profile_reductions = bool(os.environ.get("EINLANG_PROFILE_REDUCTIONS", ""))
        seen = getattr(self, "_reduction_profile_seen", None)
        if seen is None:
            seen = set()
            self._reduction_profile_seen = seen
        def reduction_profile(path: str) -> None:
            if profile_reductions:
                key = (line, path)
                if key not in seen:
                    seen.add(key)
                    print(f"[reduction] {path} L{line}", flush=True)
        def ev(e): return e.accept(self)
        _loop_to_body_defid = {}
        _reduction_defid_names = {}
        for _lp in getattr(expr, "loops", []) or []:
            _v = getattr(_lp, "variable", None)
            if _v is not None:
                _vname = getattr(_v, "name", None)
                _bd = _defid_of_var_in_expr(expr.body, _vname) if _vname else None
                if _bd is not None:
                    _loop_to_body_defid[getattr(_v, "defid", None)] = _bd
                    _reduction_defid_names[_bd] = _vname
                elif getattr(_v, "defid", None):
                    _reduction_defid_names[_v.defid] = _vname
        def _remap_ctx_to_body_defids(ctx):
            out = {}
            for loop_defid, val in (ctx or {}).items():
                if loop_defid is None:
                    continue
                body_defid = _loop_to_body_defid.get(loop_defid)
                if body_defid is not None:
                    out[body_defid] = val
                else:
                    out[loop_defid] = val
            return out
        def body_ev(ctx):
            _ctx = _remap_ctx_to_body_defids(ctx)
            for defid, val in _ctx.items():
                if defid is not None:
                    self.env.set_value(defid, val, name=_reduction_defid_names.get(defid))
            return expr.body.accept(self)
        def guard_ev(ctx):
            if not expr.guards:
                return True
            _ctx = _remap_ctx_to_body_defids(ctx)
            for defid, val in _ctx.items():
                if defid is not None:
                    self.env.set_value(defid, val, name=_reduction_defid_names.get(defid))
            from ..runtime.compute.lowered_execution import check_lowered_guards
            return check_lowered_guards(expr.guards, _ctx, lambda c: self._to_bool(c.accept(self)))
        return execute_reduction_with_loops(
            getattr(expr, "operation", "sum"),
            getattr(expr, "reduction_ranges", {}),
            body_ev,
            ev,
            guard_evaluator=guard_ev if expr.guards else None,
            initial_context={},
            profile_callback=reduction_profile if profile_reductions else None,
            parallel_shape=parallel_shape,
        )

    def visit_lowered_reduction(self, expr: LoweredReductionIR) -> Any:
        return self.evaluate_lowered_reduction(expr, parallel_shape=None)

    def visit_where_expression(self, expr: WhereExpressionIR) -> Any:
        constraints = getattr(expr, "constraints", []) or []
        needs_scope = any(_extract_binding(c) and _extract_binding(c)[0] for c in constraints)
        if needs_scope:
            with self.env.scope():
                for c in constraints:
                    b = _extract_binding(c)
                    if b:
                        var_defid, value_expr = b
                        if var_defid:
                            self.env.set_value(var_defid, value_expr.accept(self))
                for c in constraints:
                    b = _extract_binding(c)
                    if b and b[0]:
                        continue
                    if not self._to_bool(c.accept(self)):
                        return None
                return expr.expr.accept(self)
        for c in constraints:
            b = _extract_binding(c)
            if b and b[0]:
                self.env.set_value(b[0], b[1].accept(self))
        for c in constraints:
            b = _extract_binding(c)
            if b and b[0]:
                continue
            if not self._to_bool(c.accept(self)):
                return None
        return expr.expr.accept(self)

    def visit_pipeline_expression(self, expr: PipelineExpressionIR) -> Any:
        left_value = expr.left.accept(self)
        from .numpy_arrow_pipeline import apply_pipeline_right
        return apply_pipeline_right(expr.right, left_value, expr.location, self)

    def visit_builtin_call(self, expr: BuiltinCallIR) -> Any:
        from ..shared.defid import DefId
        raw = getattr(expr, "defid", None)
        if raw is None:
            raise RuntimeError("Builtin call has no DefId")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            defid = DefId(krate=int(raw[0]), index=int(raw[1]))
        elif hasattr(raw, "krate") and hasattr(raw, "index"):
            defid = DefId(krate=int(raw.krate), index=int(raw.index))
        else:
            defid = raw
        fn = self.env.get_value(defid)
        if fn is None or not callable(fn):
            raise RuntimeError(f"Builtin not found (DefId: {defid})")
        args = [arg.accept(self) for arg in getattr(expr, "args", getattr(expr, "arguments", [])) or []]
        try:
            if fn == builtin_assert:
                if len(args) == 0:
                    raise RuntimeError("assert() called with no arguments")
                return builtin_assert(args[0], args[1] if len(args) > 1 else "Assertion failed")
            return fn(*args)
        except Exception as e:
            self._raise_here(e, expr)

