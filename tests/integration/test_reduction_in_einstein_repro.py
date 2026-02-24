"""
Minimal repro for: Variable not found (defid=...). Name: m.

Pattern: user-written Einstein declaration whose clause value is a reduction
(sum[c, m, n](...)). Same structure as stdlib conv2d, no stdlib dependency.

This user impl passes; the same pattern in stdlib (std::ml::conv -> conv2d)
fails at runtime with Variable not found for reduction var m (test_conv).

Why ml conv fails:
- Entry-point code is resolved in the main name-resolution pass: visit_einstein_declaration
  runs the full reduction path (allocates defids for c,m,n, sets _reduction_loop_var_defids,
  visits body in REDUCTION scope), so body and loop vars share the same defids.
- std::ml::conv2d lives in a loaded module (conv_ops). Its AST is resolved in
  _ensure_module_resolved (same visitor, stmt.accept(self) -> visit_function_definition ->
  body.accept -> visit_block_expression -> stmt.accept -> visit_einstein_declaration).
  So in theory the full path runs there too.
- The likely failure is after specialization: the specialized conv2d is a deep copy of the
  generic IR. unify_local_var_defids_in_program(ir) only walks program.functions and
  program.modules; specialized functions live in tcx.function_ir_map and are not in the
  program, so they may never get _unify_local_var_refs_in_scopes run, or the reduction
  structure (LoweredReductionIR inside LoweredEinsteinClauseIR) may not be traversed so
  loop var defids stay out of sync with the body. At runtime the backend builds context
  from expr.reduction_ranges (loop variable defids); if the body’s "m" has defid 0:3125
  but the loop var "m" has a different defid, the context has no 0:3125 and lookup fails.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


def test_reduction_in_einstein_user_impl(compiler, runtime):
    source = """
let batch = 1;
let c_out = 1;
let x = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
let w = [[[[1.0, 0.0], [0.0, 1.0]]]];
let conv_sum[b in 0..batch, co in 0..c_out, i in 0..2, j in 0..2] =
    sum[c in 0..1, m in 0..2, n in 0..2](x[b, c, i + m, j + n] * w[co, c, m, n]);
let result = conv_sum;
"""
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    out = np.array(result.outputs.get("result", result.value))
    expected = np.array([[[[6.0, 8.0], [12.0, 14.0]]]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_reduction_in_einstein_user_module(compiler, runtime):
    """
    User module (source_overlay) with single conv2d: reduction sum[c,m,n] inside
    Einstein, same pattern as stdlib. Exercises module resolve + lower + call.
    Currently fails with Variable not found (defid=...). Name: m. When fixed, pass.
    """
    my_conv_source = """
pub fn conv2d(X, W, B, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w) {
    let c_in = len(X[0]);
    let kernel_h = len(W[0][0]);
    let kernel_w = len(W[0][0][0]);
    let conv_sum[..batch, co, ih, iw] = sum[c in 0..c_in, m in 0..kernel_h, n in 0..kernel_w](
        X[..batch, c, ih * stride_h - pad_h + m * dilation_h, iw * stride_w - pad_w + n * dilation_w]
        * W[co, c, m, n]
    );
    let output[..batch, co, ih, iw] = conv_sum[..batch, co, ih, iw] + B[co];
    output
}
"""
    main_source = """
use my_conv::conv2d;
let x_conv_2d = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
let w_conv_2d = [[[[1.0, 0.0], [0.0, 1.0]]]];
let b_conv_2d = [0.0];
let result_2d = conv2d(x_conv_2d, w_conv_2d, b_conv_2d, 1, 1, 0, 0, 1, 1);
"""
    source_overlay = {("my_conv",): my_conv_source}
    result = compile_and_execute(
        main_source,
        compiler,
        runtime,
        source_file="main.ein",
        source_overlay=source_overlay,
    )
    assert result.success, f"Execution failed: {result.errors}"

    out = np.array(result.outputs["result_2d"])
    expected = np.array([[[[6.0, 8.0], [12.0, 14.0]]]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_reduction_in_einstein_stdlib_conv(compiler, runtime):
    """
    Fail pattern: use std::ml and call std::ml::conv with 2D inputs. Hits conv2d
    (reduction sum[c,m,n] inside Einstein). Currently fails with Variable not found
    (defid=...). Name: m. When that bug is fixed, this test must pass.
    """
    source = """
use std::ml;
let x_conv_2d = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
let w_conv_2d = [[[[1.0, 0.0], [0.0, 1.0]]]];
let b_conv_2d = [0.0];
let result_2d = std::ml::conv(x_conv_2d, w_conv_2d, b_conv_2d, [1, 1], [0, 0], [1, 1]);
"""
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    out = np.array(result.outputs["result_2d"])
    expected = np.array([[[[6.0, 8.0], [12.0, 14.0]]]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-5)
