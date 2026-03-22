"""
Unit tests: Einstein IR element_type propagation
=================================================

The vectorized path in the NumPy backend allocates the output array with a dtype
derived from ``LoweredEinsteinIR.element_type``.  If that field is None or wrong
(e.g. i32 when the body produces floats) the vectorised path silently truncates
results.  The scalar path is unaffected because it accumulates in Python's native
number tower and only writes to the pre-allocated array at the very end.

These tests pin the *compiler* contract:

    EinsteinIR.element_type     (after TypeInferencePass)
        → LoweredEinsteinIR.element_type  (after EinsteinLoweringPass)

so that any regression in the type-inference or lowering chain is caught before
it silently corrupts runtime output.

Patterns tested
---------------
float output (must be f32):
  • sum of products of two float arrays          (weighted moving average)
  • sum of float array elements divided by float (moving average / 3.0)
  • sum of products: float literal array × float array (convolution filter)
  • elementwise product of two float arrays

integer output (must be i32):
  • sum of integer array elements (cumulative sum)
  • sum of products of two integer arrays (cross-correlation with [1,1] pattern)
  • simple elementwise on integer array

The test for the "failing pattern" (weighted_avg) that triggered this work is the
first case in each class.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import (
    BindingIR, EinsteinIR, LoweredEinsteinIR, is_einstein_binding,
    ProgramIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compile(source: str, stop_after: str) -> ProgramIR:
    """Compile source and return the IR after *stop_after* pass, or raise."""
    compiler = CompilerDriver()
    result = compiler.compile(source.strip(), source_file="<test>",
                              stop_after_pass=stop_after)
    assert result.success, (
        f"Compilation failed (stop_after={stop_after}): "
        + (result.tcx.reporter.format_all_errors() if result.tcx else "unknown")
    )
    assert result.ir is not None
    return result.ir


def _collect_bindings(ir: ProgramIR) -> dict:
    """Return {name: binding_node} for every top-level BindingIR in the program."""
    out = {}
    for stmt in (ir.statements or []):
        if isinstance(stmt, BindingIR):
            name = stmt.name
            if name:
                out[name] = stmt
    return out


def _einstein_expr(ir: ProgramIR, name: str) -> EinsteinIR:
    """Return the EinsteinIR expression for binding *name* (after TypeInference)."""
    bindings = _collect_bindings(ir)
    assert name in bindings, f"Binding '{name}' not found; available: {list(bindings)}"
    expr = bindings[name].expr
    assert isinstance(expr, EinsteinIR), (
        f"Expected EinsteinIR for '{name}', got {type(expr).__name__}"
    )
    return expr


def _lowered_einstein(ir: ProgramIR, name: str) -> LoweredEinsteinIR:
    """Return the LoweredEinsteinIR expression for binding *name* (after Lowering)."""
    bindings = _collect_bindings(ir)
    assert name in bindings, f"Binding '{name}' not found; available: {list(bindings)}"
    expr = bindings[name].expr
    assert isinstance(expr, LoweredEinsteinIR), (
        f"Expected LoweredEinsteinIR for '{name}', got {type(expr).__name__}"
    )
    return expr


def _prim_name(type_obj) -> str:
    """Return lowercase primitive name ('f32', 'i32', …) or '' if unresolvable."""
    if type_obj is None:
        return "<None>"
    return type_obj.name.lower()


# ---------------------------------------------------------------------------
# Shared source snippets
# ---------------------------------------------------------------------------

# The direct cause of the regression: sum of products of two float32 arrays.
# weighted_avg[2] = 98*0.5 + 105*0.3 + 103*0.2 = 101.1
_SRC_WEIGHTED_AVG = """
let stock_prices = [100.0, 102.0, 98.0, 105.0, 103.0];
let weights = [0.5, 0.3, 0.2];
let weighted_avg[i in 0..3] = sum[k in 0..3](stock_prices[i+k] * weights[k]);
"""

# Moving average: sum of float array / float literal
_SRC_MOVING_AVG = """
let prices = [10.0, 15.0, 12.0, 18.0, 20.0, 16.0];
let moving_avg[i in 0..2] = sum[k in 0..3](prices[i+2-k]) / 3.0;
"""

# Convolution-style filter: sum of float_array * float_array
_SRC_FILTER = """
let signal = [1.0, 2.0, 3.0, 4.0, 5.0];
let kernel = [0.5, 0.3, 0.2];
let filtered[i in 0..1] = sum[k in 0..3](signal[i+k] * kernel[k]);
"""

# Pure float elementwise (not a reduction, sanity baseline)
_SRC_FLOAT_ELT = """
let A = [1.0, 2.0, 3.0];
let B = [4.0, 5.0, 6.0];
let C[i in 0..3] = A[i] * B[i];
"""

# Integer cumulative sum – must stay i32
_SRC_CUMSUM = """
let data = [1, 2, 3, 4];
let cumsum[i in 0..4] = sum[k in 0..i+1](data[i-k]);
"""

# Integer cross-correlation – must stay i32
_SRC_CORRELATION = """
let sequence = [1, 0, 1, 1, 0, 1];
let pattern = [1, 1];
let correlation[i in 0..3] = sum[k in 0..2](sequence[i+k] * pattern[k]);
"""

# Simple integer elementwise – must stay i32
_SRC_INT_ELT = """
let arr = [1, 2, 3];
let doubled[i in 0..3] = arr[i] * 2;
"""


# ===========================================================================
# Class 1 – EinsteinIR.element_type after TypeInferencePass
# ===========================================================================

class TestEinsteinIRElementTypeAfterTypeInference:
    """EinsteinIR.element_type is set correctly by TypeInferencePass."""

    def _get_element_type(self, source: str, binding_name: str) -> str:
        ir = _compile(source, "TypeInferencePass")
        ein = _einstein_expr(ir, binding_name)
        return _prim_name(ein.element_type)

    # --- float cases (must be f32) ---

    def test_weighted_avg_sum_of_products_of_float_arrays_is_f32(self):
        """sum[k](float_array[…] * float_array[…]) → element_type must be f32."""
        et = self._get_element_type(_SRC_WEIGHTED_AVG, "weighted_avg")
        assert et == "f32", (
            f"weighted_avg: expected element_type='f32', got '{et}'. "
            "TypeInferencePass failed to propagate float type through "
            "sum(float_array * float_array)."
        )

    def test_moving_avg_sum_div_float_literal_is_f32(self):
        """sum(float_array[…]) / 3.0 → element_type must be f32."""
        et = self._get_element_type(_SRC_MOVING_AVG, "moving_avg")
        assert et == "f32", (
            f"moving_avg: expected element_type='f32', got '{et}'."
        )

    def test_filter_sum_of_float_array_products_is_f32(self):
        """sum(float_array * float_array) → element_type must be f32."""
        et = self._get_element_type(_SRC_FILTER, "filtered")
        assert et == "f32", (
            f"filtered: expected element_type='f32', got '{et}'."
        )

    def test_elementwise_float_product_is_f32(self):
        """float_array[i] * float_array[i] → element_type must be f32."""
        et = self._get_element_type(_SRC_FLOAT_ELT, "C")
        assert et == "f32", (
            f"C: expected element_type='f32', got '{et}'."
        )

    # --- integer cases (must be i32) ---

    def test_cumsum_of_int_array_is_i32(self):
        """sum(int_array[…]) → element_type must be i32."""
        et = self._get_element_type(_SRC_CUMSUM, "cumsum")
        assert et == "i32", (
            f"cumsum: expected element_type='i32', got '{et}'."
        )

    def test_correlation_sum_of_int_products_is_i32(self):
        """sum(int_array * int_array) → element_type must be i32."""
        et = self._get_element_type(_SRC_CORRELATION, "correlation")
        assert et == "i32", (
            f"correlation: expected element_type='i32', got '{et}'."
        )

    def test_elementwise_int_product_is_i32(self):
        """int_array[i] * 2 → element_type must be i32."""
        et = self._get_element_type(_SRC_INT_ELT, "doubled")
        assert et == "i32", (
            f"doubled: expected element_type='i32', got '{et}'."
        )


# ===========================================================================
# Class 2 – LoweredEinsteinIR.element_type after EinsteinLoweringPass
# ===========================================================================

class TestLoweredEinsteinIRElementType:
    """LoweredEinsteinIR.element_type carries the correct dtype into the runtime."""

    def _get_element_type(self, source: str, binding_name: str) -> str:
        ir = _compile(source, "EinsteinLoweringPass")
        lowered = _lowered_einstein(ir, binding_name)
        return _prim_name(lowered.element_type)

    # --- float cases ---

    def test_weighted_avg_lowered_element_type_is_f32(self):
        """LoweredEinsteinIR for weighted_avg must carry element_type=f32."""
        et = self._get_element_type(_SRC_WEIGHTED_AVG, "weighted_avg")
        assert et == "f32", (
            f"weighted_avg (lowered): expected element_type='f32', got '{et}'. "
            "EinsteinLoweringPass lost the float type inferred by TypeInferencePass."
        )

    def test_moving_avg_lowered_element_type_is_f32(self):
        et = self._get_element_type(_SRC_MOVING_AVG, "moving_avg")
        assert et == "f32", (
            f"moving_avg (lowered): expected element_type='f32', got '{et}'."
        )

    def test_filter_lowered_element_type_is_f32(self):
        et = self._get_element_type(_SRC_FILTER, "filtered")
        assert et == "f32", (
            f"filtered (lowered): expected element_type='f32', got '{et}'."
        )

    def test_float_elementwise_lowered_element_type_is_f32(self):
        et = self._get_element_type(_SRC_FLOAT_ELT, "C")
        assert et == "f32", (
            f"C (lowered): expected element_type='f32', got '{et}'."
        )

    # --- integer cases ---

    def test_cumsum_lowered_element_type_is_i32(self):
        et = self._get_element_type(_SRC_CUMSUM, "cumsum")
        assert et == "i32", (
            f"cumsum (lowered): expected element_type='i32', got '{et}'."
        )

    def test_correlation_lowered_element_type_is_i32(self):
        et = self._get_element_type(_SRC_CORRELATION, "correlation")
        assert et == "i32", (
            f"correlation (lowered): expected element_type='i32', got '{et}'."
        )

    def test_int_elementwise_lowered_element_type_is_i32(self):
        et = self._get_element_type(_SRC_INT_ELT, "doubled")
        assert et == "i32", (
            f"doubled (lowered): expected element_type='i32', got '{et}'."
        )


# ===========================================================================
# Class 3 – scatter_elements rank-2 axis-1 pattern (test_scatter_ops)
#
# The stdlib scatter_elements (rank 2, axis=1) uses two Einstein declarations:
#   last_l[i,j] = max[l](if indices[i,l]==j { l } else { -1 })  → i32
#   result[i,j] = if last_l[i,j]>=0 { updates[i,last_l[i,j]] } else { data[i,j] } → f32
#
# The `result` declaration must be f32: both conditional branches access float
# arrays.  If TypeInference assigns i32 (because `last_l` is i32), the runtime
# writes integer-truncated values into the output, producing the observed wrong
# third element (1 instead of 3).
# ===========================================================================

_SRC_SCATTER_RANK2_AXIS1 = """
let data = [[1.0, 2.0, 3.0]];
let indices = [[0, 1, 0]];
let updates = [[10.0, 20.0, 30.0]];
let last_l[i in 0..1, j in 0..3] = max[l in 0..3](
    if (indices[i, l] as i32) == j { l } else { -1 }
);
let result[i in 0..1, j in 0..3] = if last_l[i, j] >= 0 {
    updates[i, last_l[i, j]]
} else {
    data[i, j]
};
"""


class TestScatterElementsPatternElementType:
    """IR element_type for the scatter_elements rank-2 axis-1 pattern."""

    def _et_after_lowering(self, binding_name: str) -> str:
        ir = _compile(_SRC_SCATTER_RANK2_AXIS1, "EinsteinLoweringPass")
        return _prim_name(_lowered_einstein(ir, binding_name).element_type)

    def test_last_l_is_i32(self):
        """max of integer values (loop index or -1) → element_type must be i32."""
        et = self._et_after_lowering("last_l")
        assert et == "i32", (
            f"last_l: expected element_type='i32', got '{et}'. "
            "The max-reduction over integer loop index expressions must stay i32."
        )

    def test_result_is_f32(self):
        """Conditional select between two float-array accesses → element_type must be f32."""
        et = self._et_after_lowering("result")
        assert et == "f32", (
            f"result (scatter): expected element_type='f32', got '{et}'. "
            "Both branches of the if/else access float arrays (data, updates); "
            "the output must be f32.  An i32 result truncates 3.0→3 to 3, etc."
        )


# ===========================================================================
# Class 4 – batch_matmul pattern (test_linear_algebra_clustered_accuracy)
#
# The stdlib batch_matmul uses:
#   let result[..batch, i, j] = sum[k](a[..batch, i, k] * b[..batch, k, j])
#
# With concrete ranges (rest patterns cannot be used in unit tests):
#   let result[batch in 0..2, i in 0..2, j in 0..2] =
#       sum[k in 0..2](a_m[batch,i,k] * b_m[batch,k,j])
#
# This is the same sum-of-float-products pattern as weighted_avg but 3-D.
# ===========================================================================

_SRC_BATCH_MATMUL = """
let a_m = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
let b_m = [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]];
let result[batch in 0..2, i in 0..2, j in 0..2] = sum[k in 0..2](a_m[batch, i, k] * b_m[batch, k, j]);
"""

_SRC_LINEAR_LAYER = """
let x = [[1.0, 2.0]];
let weights = [[1.0, 2.0], [3.0, 4.0]];
let bias = [0.5, 1.0];
let output[i in 0..1, j in 0..2] = sum[k in 0..2](x[i, k] * weights[j, k]) + bias[j];
"""


class TestBatchMatmulPatternElementType:
    """IR element_type for the batched matrix-multiply pattern."""

    def test_batch_matmul_type_inference_is_f32(self):
        """sum[k](float3D * float3D) → EinsteinIR.element_type must be f32."""
        ir = _compile(_SRC_BATCH_MATMUL, "TypeInferencePass")
        et = _prim_name(_einstein_expr(ir, "result").element_type)
        assert et == "f32", (
            f"batch_matmul result: TypeInferencePass gave element_type='{et}', want 'f32'."
        )

    def test_batch_matmul_lowered_is_f32(self):
        """LoweredEinsteinIR for 3-D sum-of-products must carry element_type=f32."""
        ir = _compile(_SRC_BATCH_MATMUL, "EinsteinLoweringPass")
        et = _prim_name(_lowered_einstein(ir, "result").element_type)
        assert et == "f32", (
            f"batch_matmul result (lowered): expected 'f32', got '{et}'."
        )

    def test_linear_layer_output_is_f32(self):
        """sum[k](x*W) + bias for float arrays → element_type must be f32."""
        ir = _compile(_SRC_LINEAR_LAYER, "EinsteinLoweringPass")
        et = _prim_name(_lowered_einstein(ir, "output").element_type)
        assert et == "f32", (
            f"linear layer output (lowered): expected 'f32', got '{et}'."
        )


# ===========================================================================
# Class 5 – topk transpose pattern (test_topk_2d_axis0_k1)
#
# When topk is called with axis=0 the stdlib transposes the input so the
# target axis becomes innermost:
#   let X_t[j in 0..N, i in 0..M] = X[i, j]   ← simple transposition
#   let values_work[i in 0..M_work, j in 0..k] = topk_2d_row_values(X_work,…)[j]
#
# The X_t binding must remain f32.  If it is downgraded to i32 the helper
# topk_2d_row_values receives an integer array and comparison / return values
# are corrupted (e.g. column max 6 becomes 1).
# ===========================================================================

_SRC_TOPK_TRANSPOSE = """
let X = [[1.0, 5.0], [4.0, 1.0], [3.0, 6.0]];
let X_t[j in 0..2, i in 0..3] = X[i, j];
"""

_SRC_TOPK_VALUES_SELECT = """
let X = [[1.0, 5.0], [4.0, 1.0], [3.0, 6.0]];
let X_t[j in 0..2, i in 0..3] = X[i, j];
let max_per_row[i in 0..2] = max[j in 0..3](X_t[i, j]);
"""


class TestTopkTransposePatternElementType:
    """IR element_type for the topk transpose and max-select patterns."""

    def test_transpose_type_inference_is_f32(self):
        """let X_t[j,i] = X[i,j] (transpose of float2D) → element_type must be f32."""
        ir = _compile(_SRC_TOPK_TRANSPOSE, "TypeInferencePass")
        et = _prim_name(_einstein_expr(ir, "X_t").element_type)
        assert et == "f32", (
            f"X_t (transpose): TypeInferencePass gave element_type='{et}', want 'f32'. "
            "A transposition of a float array must preserve f32."
        )

    def test_transpose_lowered_is_f32(self):
        ir = _compile(_SRC_TOPK_TRANSPOSE, "EinsteinLoweringPass")
        et = _prim_name(_lowered_einstein(ir, "X_t").element_type)
        assert et == "f32", (
            f"X_t (lowered): expected 'f32', got '{et}'."
        )

    def test_max_over_float_rows_is_f32(self):
        """max[j](float2D[i,j]) → element_type must be f32."""
        ir = _compile(_SRC_TOPK_VALUES_SELECT, "EinsteinLoweringPass")
        et = _prim_name(_lowered_einstein(ir, "max_per_row").element_type)
        assert et == "f32", (
            f"max_per_row: expected 'f32', got '{et}'. "
            "A max-reduction over a float transposed array must be f32."
        )


# ===========================================================================
# Class 6 – trig-ops arithmetic chain (test_trig_ops_clustered_accuracy)
#
# The stdlib trig ops (tanh, sinh, cosh) implement float arithmetic:
#   let exp_x[i,j]     = …      (f32 input)
#   let numerator[i,j] = exp_x[i,j] - exp_neg_x[i,j]   → f32
#   let ratio[i,j]     = numerator[i,j] / denominator[i,j]  → f32
#
# The x input literal [[0.0, 1.5708, 3.14159]] must be f32, and all
# arithmetic Einstein declarations derived from it must also be f32.
# ===========================================================================

_SRC_TRIG_ARITH = """
let x = [[0.0, 1.5708, 3.14159]];
let a = [[1.0, 2.0, 3.0]];
let b = [[4.0, 5.0, 6.0]];
let diff[i in 0..1, j in 0..3] = a[i, j] - b[i, j];
let total[i in 0..1, j in 0..3] = a[i, j] + b[i, j];
let ratio[i in 0..1, j in 0..3] = diff[i, j] / total[i, j];
"""

_SRC_TRIG_INPUT = """
let x = [[0.0, 1.5708, 3.14159]];
let x_copy[i in 0..1, j in 0..3] = x[i, j];
"""


class TestTrigopsFloatPatternElementType:
    """IR element_type for float array arithmetic chains (trig ops substrate)."""

    def test_float_array_literal_x_element_type(self):
        """Input array [[0.0, 1.5708, …]] → LoweredEinsteinIR element_type must be f32."""
        ir = _compile(_SRC_TRIG_INPUT, "EinsteinLoweringPass")
        et = _prim_name(_lowered_einstein(ir, "x_copy").element_type)
        assert et == "f32", (
            f"x_copy (float literal pass-through): expected 'f32', got '{et}'. "
            "An elementwise copy of a float array must preserve f32."
        )

    def test_float_subtraction_is_f32(self):
        """float[i,j] - float[i,j] → element_type must be f32."""
        ir = _compile(_SRC_TRIG_ARITH, "EinsteinLoweringPass")
        et = _prim_name(_lowered_einstein(ir, "diff").element_type)
        assert et == "f32", (
            f"diff: expected 'f32', got '{et}'."
        )

    def test_float_division_is_f32(self):
        """float[i,j] / float[i,j] → element_type must be f32."""
        ir = _compile(_SRC_TRIG_ARITH, "EinsteinLoweringPass")
        et = _prim_name(_lowered_einstein(ir, "ratio").element_type)
        assert et == "f32", (
            f"ratio: expected 'f32', got '{et}'."
        )


# ===========================================================================
# Class 7 – quantize_linear pattern (test_quantize_linear_*)
#
# The stdlib quantize_linear uses a chain of Einstein declarations:
#   scaled[i]    = x[i] / scale          → f32 / f32  → f32
#   floor_val[i] = if scaled[i]>=0 { scaled[i]+0.5 } else { scaled[i]-0.5 }  → f32
#   rounded_i[i] = floor_val[i] as i32   → i32  (intentional cast)
#   rounded[i]   = rounded_i[i] as f32   → f32  (cast back)
#   clamped[i]   = if rounded[i] < -128.0 { -128.0 } else if … → f32
#
# If rounded_i is NOT i32, the integer-truncation step fails (no actual
# truncation happens and the clamp receives wrong inputs).
# If any f32 declaration is incorrectly typed as i32, clamp/shift values
# are integer-truncated and the saturation assertion `actual[1] == -128.0`
# becomes `127.0` (sign flip or wrong clamping).
# ===========================================================================

_SRC_QUANTIZE = """
let x = [1.0, 2.5, -3.0, 0.0, 4.2];
let y_scale = 0.05 as f32;
let scaled[i in 0..5] = x[i] / y_scale;
let floor_val[i in 0..5] = if scaled[i] >= 0.0 { scaled[i] + 0.5 } else { scaled[i] - 0.5 };
let rounded_i[i in 0..5] = floor_val[i] as i32;
let rounded[i in 0..5] = rounded_i[i] as f32;
let shifted[i in 0..5] = rounded[i] + 0.0;
let clamped[i in 0..5] = if shifted[i] < -128.0 { -128.0 } else if shifted[i] > 127.0 { 127.0 } else { shifted[i] };
"""

_SRC_QUANTIZE_SATURATION = """
let x = [100.0, -100.0, 0.5];
let scale = 0.5 as f32;
let scaled[i in 0..3] = x[i] / scale;
let rounded_i[i in 0..3] = scaled[i] as i32;
let rounded[i in 0..3] = rounded_i[i] as f32;
let clamped[i in 0..3] = if rounded[i] < -128.0 { -128.0 } else if rounded[i] > 127.0 { 127.0 } else { rounded[i] };
"""


class TestQuantizeLinearPatternElementType:
    """IR element_type for the quantize_linear step-by-step pipeline."""

    def _et(self, name: str, source: str = _SRC_QUANTIZE) -> str:
        ir = _compile(source, "EinsteinLoweringPass")
        return _prim_name(_lowered_einstein(ir, name).element_type)

    # --- float steps must be f32 ---

    def test_scaled_is_f32(self):
        """x[i] / scale (f32/f32) → element_type must be f32."""
        et = self._et("scaled")
        assert et == "f32", f"scaled: expected 'f32', got '{et}'."

    def test_floor_val_is_f32(self):
        """Conditional ±0.5 on float → element_type must be f32."""
        et = self._et("floor_val")
        assert et == "f32", f"floor_val: expected 'f32', got '{et}'."

    def test_rounded_i_is_i32(self):
        """Explicit `as i32` cast → element_type must be i32 (truncation step)."""
        et = self._et("rounded_i")
        assert et == "i32", (
            f"rounded_i: expected 'i32' (the truncating cast), got '{et}'. "
            "If this is f32 the integer-truncation step is lost and quantisation "
            "produces continuous values instead of discrete integers."
        )

    def test_rounded_is_f32(self):
        """Explicit `as f32` cast (i32→f32) → element_type must be f32."""
        et = self._et("rounded")
        assert et == "f32", f"rounded: expected 'f32' (cast back), got '{et}'."

    def test_clamped_is_f32(self):
        """Conditional clamp between float literals → element_type must be f32."""
        et = self._et("clamped")
        assert et == "f32", (
            f"clamped: expected 'f32', got '{et}'. "
            "A wrong i32 here truncates -128.0 to -128 but saturation assertions "
            "can still pass; however sign-flip bugs arise from integer overflow."
        )

    def test_saturation_clamped_is_f32(self):
        """Saturation pattern (large positive/negative inputs) → clamped must be f32."""
        et = self._et("clamped", _SRC_QUANTIZE_SATURATION)
        assert et == "f32", (
            f"clamped (saturation): expected 'f32', got '{et}'."
        )


# ===========================================================================
# Class 8 – einstein_windowing.ein  (test_execution[einstein_windowing])
#
# This class mirrors the full einstein_windowing.ein example and tests every
# Einstein declaration: IR element_type after EinsteinLoweringPass (and
# TypeInferencePass for selected bindings).
#
# Integer declarations  (data / values / sequence / pattern / matrix / square):
#   cumsum, forward_sum, correlation, cummax, pooled, diagonal, windowed → i32
#
# Float declarations  (prices / signal / kernel / stock_prices / weights):
#   moving_avg, filtered, weighted_avg → f32
#
# weighted_avg is the primary regression guard: it is the declaration whose
# element_type was silently set to i32 in earlier builds, causing
# `weighted_avg[2]` to equal 101 (int) which fails `> 101.0` (strict float).
# ===========================================================================

_WINDOWING_EINFILE = (
    Path(__file__).parent.parent.parent / "examples" / "units" / "einstein_windowing.ein"
)


def _compile_windowing(stop_after: str) -> "ProgramIR":
    """Compile the actual einstein_windowing.ein with its real source_file path."""
    source = _WINDOWING_EINFILE.read_text(encoding="utf-8")
    compiler = CompilerDriver()
    result = compiler.compile(
        source,
        source_file=str(_WINDOWING_EINFILE),
        root_path=_WINDOWING_EINFILE.parent,
        stop_after_pass=stop_after,
    )
    assert result.success, (
        f"Compilation of einstein_windowing.ein failed "
        f"(stop_after={stop_after}): "
        + (result.tcx.reporter.format_all_errors() if result.tcx else "unknown")
    )
    assert result.ir is not None
    return result.ir


class TestEinsteinWindowing:
    """Full IR type coverage for examples/units/einstein_windowing.ein.

    Every Einstein declaration in the file is checked for element_type in the
    lowered IR (and for weighted_avg / cumsum at type-inference time).
    """

    # ------------------------------------------------------------------
    # Shared: compile once per class (class-scoped cache via a tiny helper)
    # ------------------------------------------------------------------

    @staticmethod
    def _ir():
        if not hasattr(TestEinsteinWindowing, "_cached_ir"):
            TestEinsteinWindowing._cached_ir = _compile_windowing("EinsteinLoweringPass")
        return TestEinsteinWindowing._cached_ir

    @staticmethod
    def _et(name: str) -> str:
        return _prim_name(_lowered_einstein(TestEinsteinWindowing._ir(), name).element_type)

    # ------------------------------------------------------------------
    # Integer declarations
    # ------------------------------------------------------------------

    def test_cumsum_is_i32(self):
        """cumsum[i] = sum[k](data[i-k])  data=[1,2,3,4] → i32."""
        assert self._et("cumsum") == "i32", (
            f"cumsum: expected i32, got '{self._et('cumsum')}'."
        )

    def test_forward_sum_is_i32(self):
        """forward_sum[i] = sum[k](data[i+k])  data=[1,2,3,4] → i32."""
        assert self._et("forward_sum") == "i32", (
            f"forward_sum: expected i32, got '{self._et('forward_sum')}'."
        )

    def test_pooled_is_i32(self):
        """pooled[i,j] = max[di,dj](matrix[i+di,j+dj])  matrix integer → i32."""
        assert self._et("pooled") == "i32", (
            f"pooled: expected i32, got '{self._et('pooled')}'."
        )

    def test_diagonal_is_i32(self):
        """diagonal[i] = square[i,i]  square integer → i32."""
        assert self._et("diagonal") == "i32", (
            f"diagonal: expected i32, got '{self._et('diagonal')}'."
        )

    def test_correlation_is_i32(self):
        """correlation[i] = sum[k](sequence[i+k]*pattern[k])  both integer → i32."""
        assert self._et("correlation") == "i32", (
            f"correlation: expected i32, got '{self._et('correlation')}'."
        )

    def test_cummax_is_i32(self):
        """cummax[i] = max[k](values[i-k])  values integer → i32."""
        assert self._et("cummax") == "i32", (
            f"cummax: expected i32, got '{self._et('cummax')}'."
        )

    def test_windowed_is_i32(self):
        """windowed[i] = sum[k](boundary_data[i+k])  boundary_data integer → i32."""
        assert self._et("windowed") == "i32", (
            f"windowed: expected i32, got '{self._et('windowed')}'."
        )

    # ------------------------------------------------------------------
    # Float declarations
    # ------------------------------------------------------------------

    def test_moving_avg_is_f32(self):
        """moving_avg[i] = sum[k](prices[i+2-k]) / 3.0  prices float → f32."""
        assert self._et("moving_avg") == "f32", (
            f"moving_avg: expected f32, got '{self._et('moving_avg')}'."
        )

    def test_filtered_is_f32(self):
        """filtered[i] = sum[k](signal[i+k]*kernel[k])  signal/kernel float → f32."""
        assert self._et("filtered") == "f32", (
            f"filtered: expected f32, got '{self._et('filtered')}'."
        )

    def test_weighted_avg_is_f32(self):
        """weighted_avg[i] = sum[k](stock_prices[i+k]*weights[k])  both float → f32.

        This is the PRIMARY regression guard.  When element_type was i32 the
        runtime allocated an int32 output array; weighted_avg[2] was stored as
        np.int32(101) and the assertion `101 > 101.0` evaluates to False
        (strict inequality with equal integer and float).
        """
        et = self._et("weighted_avg")
        assert et == "f32", (
            f"weighted_avg: expected f32, got '{et}'. "
            "This causes the runtime to allocate int32 output; "
            "weighted_avg[2]=101 (int) fails `> 101.0` (strict float compare)."
        )

    # ------------------------------------------------------------------
    # Type-inference level (EinsteinIR, before lowering)
    # ------------------------------------------------------------------

    def test_weighted_avg_type_inference_is_f32(self):
        """TypeInferencePass must set EinsteinIR.element_type=f32 for weighted_avg."""
        ir = _compile_windowing("TypeInferencePass")
        et = _prim_name(_einstein_expr(ir, "weighted_avg").element_type)
        assert et == "f32", (
            f"weighted_avg after TypeInferencePass: expected f32, got '{et}'."
        )

    def test_cumsum_type_inference_is_i32(self):
        """TypeInferencePass must set EinsteinIR.element_type=i32 for cumsum."""
        ir = _compile_windowing("TypeInferencePass")
        et = _prim_name(_einstein_expr(ir, "cumsum").element_type)
        assert et == "i32", (
            f"cumsum after TypeInferencePass: expected i32, got '{et}'."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
