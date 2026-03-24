"""
Unit tests for AutodiffPass.

Tests that the autodiff pass runs and expands derivative nodes (@expr, @num/@den)
into plain IR (d_* bindings and references). No diff block; derivatives are in-program.
All tests expect compile and run success; derivative tests assert correct values.

Coverage: pipeline registration, no-@ programs, @expr expansion, quotient @num/@den,
user functions, custom @fn rules, Einstein matmul/conv1d/conv2d/reduction/affine paths,
gradient-step example, numpy two-arg pow and log10/log2 (not in print(@…) goldens),
piecewise clamp/saturate/clamp_min/clamp_max, deg/rad helpers.

Scalar and std::math-style derivative rules (arithmetic, trig, activations, etc.) are
asserted via exact stdout in tests/unit/test_print_at_golden.py and
tests/unit/test_print_at_ml_smoke.py instead of duplicating numeric quotient tests here.

Note: IR expansion catalog (_IR_DUMP_OPS) uses qualified stdlib (e.g. std::math::exp) with repo root_path.
Other tests may use local fn + @fn or python::numpy::* where no defid is required.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import IRNode, ProgramIR
from einlang.passes.autodiff import AutodiffPass, DIFF_PREFIX, USER_DIFF_PREFIX

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _ir_unique_node_count(program) -> int:
    """Count distinct IRNode objects reachable from *program* (shared nodes counted once)."""
    seen_obj: set[int] = set()
    ir_ids: set[int] = set()

    def walk(obj) -> None:
        if obj is None:
            return
        oid = id(obj)
        if oid in seen_obj:
            return
        seen_obj.add(oid)
        if isinstance(obj, IRNode):
            ir_ids.add(oid)
        if isinstance(obj, (list, tuple)):
            for x in obj:
                walk(x)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(k)
                walk(v)
        elif isinstance(obj, IRNode):
            for cls in type(obj).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    walk(getattr(obj, slot, None))

    walk(program)
    return len(ir_ids)


def _assert_allclose(actual, ref, atol=1e-5, rtol=1e-5, msg=""):
    """Compare entire array (or scalar) to NumPy reference; fail with clear diff if not close."""
    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if a.shape != r.shape:
        if a.size != r.size:
            raise AssertionError(
                "%s shape mismatch: actual %s (size %s) vs ref %s (size %s). %s"
                % (msg or "allclose", a.shape, a.size, r.shape, r.size, msg)
            )
        a = a.flat
        r = r.flat
    ok = np.allclose(a, r, atol=atol, rtol=rtol)
    if not ok:
        diff = np.abs(a - r)
        max_diff = np.max(diff)
        raise AssertionError(
            "%s max |actual - ref| = %s (atol=%s). %s"
            % (msg or "allclose", max_diff, atol, msg)
        )


def _scalar_float(outputs, key):
    v = outputs.get(key)
    assert v is not None, "expected output %r, got %s" % (key, list(outputs.keys()))
    return float(v) if hasattr(v, "item") else float(v)


def _compile_run(source, expect_success=True, root_path=None):
    compiler = CompilerDriver()
    if root_path is None:
        root_path = _REPO_ROOT
    result = compiler.compile(source.strip(), source_file="<test>", root_path=root_path)
    if expect_success:
        assert result.success, result.get_errors() or "compile failed"
    from einlang.runtime.runtime import EinlangRuntime
    runtime = EinlangRuntime(backend="numpy")
    exec_result = runtime.execute(result)
    if expect_success:
        assert exec_result.success, getattr(exec_result, "error", None) or exec_result.errors
    return result, exec_result.outputs or {}


class TestAutodiffPass:
    """Test AutodiffPass integration in the compiler pipeline."""

    def test_autodiff_pass_registered(self):
        """AutodiffPass is in the compiler pipeline after RecurrenceOrder and before validation."""
        compiler = CompilerDriver()
        pass_names = [p.__name__ for p in compiler.pass_manager.passes]
        assert "AutodiffPass" in pass_names

    def test_no_differential_ir_skips_diff_block(self):
        """Program without @ has no diff block and empty differential targets."""
        compiler = CompilerDriver()
        source = "let w = 1.0; let loss = w * 2.0;"
        result = compiler.compile(source, source_file="<test>")
        assert result.success, result.get_errors() or "compile failed"
        assert result.tcx is not None
        analysis = result.tcx.get_analysis(AutodiffPass)
        assert analysis["diff_block"] is None
        assert analysis["differential_targets"] == set()
        assert analysis["differential_buffer_by_defid"] == {}

    def test_differential_ir_expanded_into_plain_ir(self):
        """Program with @w expands derivative nodes into plain IR (d_* bindings in diff block)."""
        compiler = CompilerDriver()
        source = """
let w = 1.0;
let loss = w * 2.0;
let dw = @w;
"""
        result = compiler.compile(source.strip(), source_file="<test>")
        assert result.success, result.get_errors() or "compile failed"
        analysis = result.tcx.get_analysis(AutodiffPass)
        diff_block = analysis["diff_block"]
        assert diff_block is not None and len(diff_block) >= 1
        bindings = getattr(result.ir, "bindings", None) or []
        tangent_bindings = [
            b
            for b in bindings
            if (getattr(b, "name", "") or "").startswith(DIFF_PREFIX)
            or (getattr(b, "name", "") or "").startswith(USER_DIFF_PREFIX)
        ]
        assert len(tangent_bindings) >= 1

    def test_quotient_binary_expr(self):
        """@b/@a for b = a*a: compile and run; assert db_da == 2*a == 6 at a=3."""
        compiler = CompilerDriver()
        source = """
let a = 3.0;
let b = a * a;
let db_da = @b / @a;
"""
        result = compiler.compile(source.strip(), source_file="<test>")
        assert result.success, result.get_errors() or "compile failed"
        from einlang.runtime.runtime import EinlangRuntime
        runtime = EinlangRuntime(backend="numpy")
        exec_result = runtime.execute(result)
        assert exec_result.success, getattr(exec_result, "error", None) or exec_result.errors
        outputs = getattr(exec_result, "outputs", {}) or {}
        actual = _scalar_float(outputs, "db_da")
        assert actual == 6.0, "expected db/da = 2*a = 6 at a=3, got %s" % actual

    def test_user_expr_autodiff_compiles_and_runs(self):
        """Differentiate through user fn sq(x)=x*x: @b/@a expands to 2*a; assert db_da == 6 at a=3."""
        compiler = CompilerDriver()
        source = """
fn sq(x) {
    x * x
}
let a = 3.0;
let b = sq(a);
let db_da = @b / @a;
"""
        result = compiler.compile(source.strip(), source_file="<test>")
        assert result.success, result.get_errors() or "compile failed"
        from einlang.runtime.runtime import EinlangRuntime
        runtime = EinlangRuntime(backend="numpy")
        exec_result = runtime.execute(result)
        assert exec_result.success, getattr(exec_result, "error", None) or exec_result.errors
        outputs = getattr(exec_result, "outputs", {}) or {}
        actual = _scalar_float(outputs, "db_da")
        assert actual == 6.0, "expected db/da = 2*a = 6 at a=3 for sq(a)=a^2, got %s" % actual

    def test_autodiff_in_block_scope(self):
        """Autodiff works in a scope (block), not only top-level: @y/@x inside block with let x, let y."""
        compiler = CompilerDriver()
        source = """
let result = {
    let x = 3.0;
    let y = x * x;
    @y / @x
};
"""
        result = compiler.compile(source.strip(), source_file="<test>")
        assert result.success, result.get_errors() or "compile failed"
        from einlang.runtime.runtime import EinlangRuntime
        runtime = EinlangRuntime(backend="numpy")
        exec_result = runtime.execute(result)
        assert exec_result.success, getattr(exec_result, "error", None) or exec_result.errors
        outputs = getattr(exec_result, "outputs", {}) or {}
        actual = _scalar_float(outputs, "result")
        assert actual == 6.0, "expected d(x^2)/dx = 2*x = 6 at x=3 in block scope, got %s" % actual

    def test_quotient_tensor_slice_alias_in_einstein_clause(self):
        """@loss/@w when w = W[i,j] and loss uses sum[a](… W[a,j] …): ∂loss/∂w is nonzero (slice aliases storage)."""
        compiler = CompilerDriver()
        source = """
let W = [[0.0]];
let x = [2.0];
let y0 = 3.0;
let G[i in 0..1] = {
    let logit = sum[a in 0..1](x[a] * W[a, 0]);
    let loss_b = (logit - y0) ** 2.0;
    let w_ij = W[i, 0];
    @loss_b / @w_ij
};
"""
        result = compiler.compile(source.strip(), source_file="<test>")
        assert result.success, result.get_errors() or "compile failed"
        from einlang.runtime.runtime import EinlangRuntime

        runtime = EinlangRuntime(backend="numpy")
        exec_result = runtime.execute(result)
        assert exec_result.success, getattr(exec_result, "error", None) or exec_result.errors
        outputs = getattr(exec_result, "outputs", {}) or {}
        g0 = outputs.get("G")
        assert g0 is not None, "expected G output, got %s" % list(outputs.keys())
        import numpy as np

        arr = np.asarray(g0)
        assert arr.shape == (1,), "expected G shape (1,), got %s" % (arr.shape,)
        assert abs(float(arr[0]) - (-12.0)) < 1e-5, "expected d(loss)/dW[0,0] = 2*(0-3)*2 = -12, got %s" % arr[0]

    def test_einstein_quotient_compiles_and_runs(self):
        """@C/@A when C is Einstein sum: autodiff expands to ∂C/∂A Einstein; compile and run; assert dC_dA shape."""
        compiler = CompilerDriver()
        source = """
let A = [[1.0, 2.0], [3.0, 4.0]];
let B = [[5.0, 6.0], [7.0, 8.0]];
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let dC_dA = @C / @A;
"""
        result = compiler.compile(source.strip(), source_file="<test>")
        assert result.success, result.get_errors() or "compile failed"
        from einlang.runtime.runtime import EinlangRuntime
        runtime = EinlangRuntime(backend="numpy")
        exec_result = runtime.execute(result)
        assert exec_result.success, getattr(exec_result, "error", None) or exec_result.errors
        outputs = getattr(exec_result, "outputs", {}) or {}
        dC_dA = outputs.get("dC_dA")
        assert dC_dA is not None, "expected output dC_dA, got %s" % list(outputs.keys())
        try:
            import numpy as np
            arr = np.asarray(dC_dA)
            # ∂C/∂A for C[i,j]=sum_k A[i,k]*B[k,j] is 4-tensor: dC_dA[i,j,r,s] = B[s,j] when i==r else 0
            assert arr.ndim == 4, "expected dC_dA to be 4D (i,j,r,s), got ndim %s" % arr.ndim
            assert np.isfinite(arr).all(), "dC_dA should be finite, got %s" % arr
            assert arr.shape == (2, 2, 2, 2), "expected shape (2,2,2,2), got %s" % (arr.shape,)
            # NumPy reference: C = A @ B => ∂C[i,j]/∂A[r,s] = B[s,j] if i==r else 0
            A_ref = np.array([[1.0, 2.0], [3.0, 4.0]])
            B_ref = np.array([[5.0, 6.0], [7.0, 8.0]])
            ref = np.zeros((2, 2, 2, 2), dtype=np.float64)
            for i in range(2):
                for j in range(2):
                    for r in range(2):
                        for s in range(2):
                            ref[i, j, r, s] = B_ref[s, j] if i == r else 0.0
            _assert_allclose(arr, ref, msg="dC_dA vs ∂C/∂A for C=A@B")
        except ImportError:
            pass

    # -------------------------------------------------------------------------
    # Tensor / Einstein autodiff: matmul, conv, einsum-style
    # -------------------------------------------------------------------------

    def test_einstein_matmul_dC_dB(self):
        """@C/@B for C[i,j]=sum[k](A[i,k]*B[k,j]): ∂C/∂B has shape (2,2,2,2); ref[i,j,s,t]=A[i,s] if t==j else 0."""
        source = """
let A = [[1.0, 2.0], [3.0, 4.0]];
let B = [[5.0, 6.0], [7.0, 8.0]];
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let dC_dB = @C / @B;
"""
        _, out = _compile_run(source)
        dC_dB = out.get("dC_dB")
        assert dC_dB is not None, "expected dC_dB, got %s" % list(out.keys())
        try:
            import numpy as np
            arr = np.asarray(dC_dB)
            assert arr.ndim == 4 and arr.shape == (2, 2, 2, 2), "dC_dB shape (2,2,2,2), got %s" % (arr.shape,)
            A_ref = np.array([[1.0, 2.0], [3.0, 4.0]])
            B_ref = np.array([[5.0, 6.0], [7.0, 8.0]])
            ref = np.zeros((2, 2, 2, 2), dtype=np.float64)
            for i in range(2):
                for j in range(2):
                    for s in range(2):
                        for t in range(2):
                            ref[i, j, s, t] = A_ref[i, s] if t == j else 0.0
            _assert_allclose(arr, ref, msg="dC_dB vs ∂C/∂B")
        except ImportError:
            pass

    def test_einstein_matmul_both_dC_dA_and_dC_dB(self):
        """Same program: @C/@A and @C/@B; both derivative tensors correct."""
        source = """
let A = [[1.0, 2.0], [3.0, 4.0]];
let B = [[5.0, 6.0], [7.0, 8.0]];
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let dC_dA = @C / @A;
let dC_dB = @C / @B;
"""
        _, out = _compile_run(source)
        try:
            dca = np.asarray(out.get("dC_dA"))
            dcb = np.asarray(out.get("dC_dB"))
            assert dca.shape == (2, 2, 2, 2), "dC_dA full Jacobian layout (Julia ∂C/∂A as 4-tensor), got %s" % (dca.shape,)
            assert dcb.shape == (2, 2, 2, 2), "dC_dB full Jacobian layout, got %s" % (dcb.shape,)
            A_ref = np.array([[1.0, 2.0], [3.0, 4.0]])
            B_ref = np.array([[5.0, 6.0], [7.0, 8.0]])
            ref_dA = np.zeros((2, 2, 2, 2), dtype=np.float64)
            ref_dB = np.zeros((2, 2, 2, 2), dtype=np.float64)
            for i in range(2):
                for j in range(2):
                    for r in range(2):
                        for s in range(2):
                            ref_dA[i, j, r, s] = B_ref[s, j] if i == r else 0.0
                    for s in range(2):
                        for t in range(2):
                            ref_dB[i, j, s, t] = A_ref[i, s] if t == j else 0.0
            _assert_allclose(out.get("dC_dA"), ref_dA, msg="dC_dA both quotients")
            _assert_allclose(out.get("dC_dB"), ref_dB, msg="dC_dB both quotients")
        except ImportError:
            pass

    def test_einstein_row_sum_derivative(self):
        """r[i] = sum[j](M[i,j]); @r/@M cotangent same shape as M (Julia Zygote / ChainRules style), ones."""
        source = """
let M = [[1.0, 2.0], [3.0, 4.0]];
let r[i] = sum[j](M[i, j]);
let dr_dM = @r / @M;
"""
        _, out = _compile_run(source)
        dr_dM = out.get("dr_dM")
        assert dr_dM is not None, "expected dr_dM"
        try:
            import numpy as np
            arr = np.asarray(dr_dM)
            assert arr.shape == (2, 2), "dr_dM shape (2,2), got %s" % (arr.shape,)
            ref = np.ones((2, 2), dtype=np.float64)
            _assert_allclose(arr, ref, msg="dr_dM row-sum pullback")
        except ImportError:
            pass

    def test_einstein_conv_1d_where_clause(self):
        """1D conv with where: out[oh] = sum[kh](in[ih]*w[kh]) where ih = oh + kh; @out/@w."""
        source = """
let in = [1.0, 2.0, 3.0];
let w = [0.5, 0.5];
let out[oh] = sum[kh](in[oh + kh] * w[kh]);
let d_out_dw = @out / @w;
"""
        _, out = _compile_run(source)
        d_out_dw = out.get("d_out_dw")
        assert d_out_dw is not None, "expected d_out_dw"
        try:
            import numpy as np
            arr = np.asarray(d_out_dw)
            assert arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == 2, (
                "d_out_dw shape (2,2), got %s" % (arr.shape,)
            )
            # ∂out[oh]/∂w[kh] = in[oh+kh]: ref[oh, kh] = in[oh+kh]
            in_ref = np.array([1.0, 2.0, 3.0])
            ref = np.array([[in_ref[0], in_ref[1]], [in_ref[1], in_ref[2]]], dtype=np.float64)
            _assert_allclose(arr, ref, msg="d_out_dw vs ∂out/∂w")
        except ImportError:
            pass

    def test_einstein_conv_2d_where_clause(self):
        """2D valid conv: out[oh,ow] = sum[kh,kw](x[oh+kh,ow+kw]*w[kh,kw]); @out/@w is (2,2,2,2)."""
        source = """
let x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
let w = [[0.5, 0.5], [0.5, 0.5]];
let out[oh in 0..2, ow in 0..2] = sum[kh in 0..2, kw in 0..2](x[oh + kh, ow + kw] * w[kh, kw]);
let d_out_dw = @out / @w;
"""
        _, out = _compile_run(source)
        d_out_dw = out.get("d_out_dw")
        assert d_out_dw is not None, "expected d_out_dw"
        try:
            import numpy as np
            arr = np.asarray(d_out_dw)
            assert arr.shape == (2, 2, 2, 2), "d_out_dw shape (2,2,2,2), got %s" % (arr.shape,)
            x_ref = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
            ref = np.zeros((2, 2, 2, 2), dtype=np.float64)
            for oh in range(2):
                for ow in range(2):
                    for kh in range(2):
                        for kw in range(2):
                            ref[oh, ow, kh, kw] = x_ref[oh + kh, ow + kw]
            _assert_allclose(arr, ref, msg="d_out_dw vs ∂out/∂w conv2d")
        except ImportError:
            pass

    def test_einstein_3x3_matmul_derivative(self):
        """Larger matmul: 3x3 @ 3x3, @C/@A shape (3,3,3,3)."""
        source = """
let A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
let B = [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let dC_dA = @C / @A;
"""
        _, out = _compile_run(source)
        dC_dA = out.get("dC_dA")
        assert dC_dA is not None
        try:
            arr = np.asarray(dC_dA)
            assert arr.ndim == 4 and arr.shape == (3, 3, 3, 3)
            A_ref = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            B_ref = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
            ref = np.zeros((3, 3, 3, 3), dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    for r in range(3):
                        for s in range(3):
                            ref[i, j, r, s] = B_ref[s, j] if i == r else 0.0
            _assert_allclose(arr, ref, msg="dC_dA 3x3 matmul")
        except ImportError:
            pass

    def test_einstein_column_sum_derivative(self):
        """c[j] = sum[i](M[i,j]); @c/@M cotangent same shape as M (Julia-style), ones."""
        source = """
let M = [[1.0, 2.0], [3.0, 4.0]];
let c[j] = sum[i](M[i, j]);
let dc_dM = @c / @M;
"""
        _, out = _compile_run(source)
        dc_dM = out.get("dc_dM")
        assert dc_dM is not None
        try:
            import numpy as np
            arr = np.asarray(dc_dM)
            assert arr.shape == (2, 2), "dc_dM shape (2,2), got %s" % (arr.shape,)
            ref = np.ones((2, 2), dtype=np.float64)
            _assert_allclose(arr, ref, msg="dc_dM column-sum pullback")
        except ImportError:
            pass

    def test_softmax_autodiff(self):
        """Differentiate through sum and max reductions (generic reduction + chain rule).
        Exercises d(sum_i body)/d wrt = sum_i d(body)/d wrt and d(max_i body)/d wrt = d(body)/d wrt at argmax.
        Full softmax (with exp from std::math) is supported by the pass but requires use std::math in test."""
        source = """
let x = [[1.0, 2.0, 3.0]];
let max_val[b] = max[j](x[b, j]);
let sums[b] = sum[j](x[b, j]);
let d_max_d_x = @max_val / @x;
let d_sums_d_x = @sums / @x;
"""
        _, out = _compile_run(source)
        assert out.get("d_max_d_x") is not None
        assert out.get("d_sums_d_x") is not None
        try:
            ref_d_max = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
            ref_d_sums = np.ones((1, 3), dtype=np.float64)
            _assert_allclose(out.get("d_max_d_x"), ref_d_max, msg="d_max_d_x")
            _assert_allclose(out.get("d_sums_d_x"), ref_d_sums, msg="d_sums_d_x")
        except ImportError:
            pass

    def test_reduction_autodiff_sum(self):
        """∂(sum_j body)/∂wrt = sum_j ∂(body)/∂wrt. Derivative of sum over index is 1 at each element."""
        source = """
let x = [[1.0, 2.0, 3.0]];
let y[b] = sum[j](x[b, j]);
let dy_dx = @y / @x;
"""
        _, out = _compile_run(source)
        dy_dx = out.get("dy_dx")
        assert dy_dx is not None
        try:
            ref = np.ones((1, 3), dtype=np.float64)
            arr = np.asarray(dy_dx)
            assert arr.shape == (1, 3), "dy_dx shape (1,3), got %s" % (arr.shape,)
            _assert_allclose(arr, ref, msg="dy_dx sum reduction")
        except ImportError:
            pass

    def test_reduction_autodiff_max(self):
        """∂(max_j body)/∂wrt = ∂(body)/∂wrt at argmax. Derivative is 1 at argmax position (per batch)."""
        source = """
let x = [[1.0, 3.0, 2.0]];
let y[b] = max[j](x[b, j]);
let dy_dx = @y / @x;
"""
        _, out = _compile_run(source)
        dy_dx = out.get("dy_dx")
        assert dy_dx is not None
        try:
            arr = np.asarray(dy_dx)
            assert arr.shape == (1, 3), "dy_dx shape (1,3) same as x (Julia pullback), got %s" % (arr.shape,)
            ref = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
            _assert_allclose(dy_dx, ref, msg="dy_dx max reduction")
        except ImportError:
            pass

    def test_reduction_autodiff_min(self):
        """∂(min_j body)/∂wrt = ∂(body)/∂wrt at argmin. Derivative is 1 at argmin position (per batch)."""
        source = """
let x = [[1.0, 3.0, 2.0]];
let y[b] = min[j](x[b, j]);
let dy_dx = @y / @x;
"""
        _, out = _compile_run(source)
        dy_dx = out.get("dy_dx")
        assert dy_dx is not None
        try:
            arr = np.asarray(dy_dx)
            assert arr.shape == (1, 3), "dy_dx shape (1,3) same as x (Julia pullback), got %s" % (arr.shape,)
            ref = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
            _assert_allclose(dy_dx, ref, msg="dy_dx min reduction")
        except ImportError:
            pass

    def test_reduction_autodiff_prod(self):
        """∂(prod_j body)/∂wrt = (prod body) * sum_j (d_body/body). For body=x[b,j]: d(prod)/dx[b,j] = prod_{j'!=j} x[b,j']."""
        source = """
let x = [[1.0, 2.0, 3.0]];
let y[b] = prod[j](x[b, j]);
let dy_dx = @y / @x;
"""
        _, out = _compile_run(source)
        dy_dx = out.get("dy_dx")
        assert dy_dx is not None
        try:
            arr = np.asarray(dy_dx)
            assert arr.shape == (1, 3), "dy_dx shape (1,3) same as x (Julia pullback), got %s" % (arr.shape,)
            ref = np.array([[6.0, 3.0, 2.0]], dtype=np.float64)
            _assert_allclose(dy_dx, ref, msg="dy_dx prod reduction", atol=1e-4)
        except ImportError:
            pass

    def test_max_pool_quotient_shape_and_argmax_scatter(self):
        """`@y/@x` for max_pool should keep x-shape and scatter 1.0 to argmax in each pooled window."""
        source = """
use std::ml;
let x = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]];
let y = std::ml::max_pool(x, [2, 2], [2, 2], [0, 0]);
let dy_dx = @y / @x;
"""
        _, out = _compile_run(source)
        dy_dx = out.get("dy_dx")
        assert dy_dx is not None
        try:
            arr = np.asarray(dy_dx)
            assert arr.shape == (1, 1, 2, 3), (
                "dy_dx shape should match x (1,1,2,3), got %s" % (arr.shape,)
            )
            ref = np.zeros((1, 1, 2, 3), dtype=np.float64)
            # Pool window is x[..., 0:2, 0:2] = [[1,2],[4,5]], argmax at value 5 -> [1,1].
            ref[0, 0, 1, 1] = 1.0
            _assert_allclose(arr, ref, msg="dy_dx max_pool argmax scatter")
        except ImportError:
            pass

    def test_einstein_attention_matmul_chain_no_softmax(self):
        """Single-head attention matmul chain (no softmax): scores = Q@K^T, out = scores@V; @out/@Q.
        MHA uses this plus softmax; the matmul part is differentiable. Cotangent ∂out/∂Q has the same shape as Q (Julia-style)."""
        source = """
let scale = 0.5;
let Q = [[[1.0, 2.0], [3.0, 4.0]]];
let K = [[[1.0, 2.0], [3.0, 4.0]]];
let V = [[[1.0, 0.0], [0.0, 1.0]]];
let scores[b, i, j] = sum[d](Q[b, i, d] * K[b, j, d]) * scale;
let out[b, i, d] = sum[j](scores[b, i, j] * V[b, j, d]);
let d_out_d_Q = @out / @Q;
"""
        _, out = _compile_run(source)
        d_out_d_Q = out.get("d_out_d_Q")
        assert d_out_d_Q is not None
        try:
            arr = np.asarray(d_out_d_Q)
            assert np.isfinite(arr).all(), "d_out_d_Q should be finite"
            assert arr.shape == (1, 2, 2), (
                "d_out_d_Q shape (1,2,2) same as Q (Julia pullback), got %s" % (arr.shape,)
            )
        except ImportError:
            pass

    def test_einstein_two_factor_product(self):
        """y[i] = sum[j](A[i,j]*b[j]); @y/@A and @y/@b; cotangents same shape as A and b (Julia-style)."""
        source = """
let A = [[1.0, 2.0], [3.0, 4.0]];
let b = [5.0, 6.0];
let y[i] = sum[j](A[i, j] * b[j]);
let dy_dA = @y / @A;
let dy_db = @y / @b;
"""
        _, out = _compile_run(source)
        assert out.get("dy_dA") is not None and out.get("dy_db") is not None
        try:
            assert np.asarray(out.get("dy_dA")).shape == (2, 2)
            assert np.asarray(out.get("dy_db")).shape == (2, 2)
            A_ref = np.array([[1.0, 2.0], [3.0, 4.0]])
            b_ref = np.array([5.0, 6.0])
            ref_dy_dA = np.broadcast_to(b_ref, (2, 2)).astype(np.float64)
            ref_dy_db = A_ref.copy()
            _assert_allclose(out.get("dy_dA"), ref_dy_dA, msg="dy_dA")
            _assert_allclose(out.get("dy_db"), ref_dy_db, msg="dy_db")
        except ImportError:
            pass

    def test_einstein_affine_derivatives(self):
        """Affine y[i,j] = sum[k](x[i,k]*W[j,k]) + b[j]; cotangents same shape as x, W, b (Julia-style; AUTODIFF_EINSTEIN_OPS §4)."""
        source = """
let x = [[1.0, 2.0], [3.0, 4.0]];
let W = [[0.5, 0.5], [0.1, 0.2]];
let b = [0.1, 0.2];
let y[i, j] = sum[k](x[i, k] * W[j, k]) + b[j];
let dy_dx = @y / @x;
let dy_dW = @y / @W;
let dy_db = @y / @b;
"""
        _, out = _compile_run(source)
        assert out.get("dy_dx") is not None, "dy_dx"
        assert out.get("dy_dW") is not None, "dy_dW"
        assert out.get("dy_db") is not None, "dy_db"
        try:
            for nm, sh in (("dy_dx", (2, 2)), ("dy_dW", (2, 2)), ("dy_db", (2, 2))):
                a = np.asarray(out.get(nm))
                assert a.shape == sh, "%s shape %s same as primal (Julia pullback), got %s" % (nm, sh, a.shape)
            x_ref = np.array([[1.0, 2.0], [3.0, 4.0]])
            W_ref = np.array([[0.5, 0.5], [0.1, 0.2]])
            ref_dy_dx = np.array([[np.sum(W_ref[0, :]), np.sum(W_ref[1, :])]] * 2, dtype=np.float64)
            _assert_allclose(np.asarray(out.get("dy_dx")), ref_dy_dx, msg="dy_dx")
            ref_dy_dW = np.zeros((2, 2), dtype=np.float64)
            for p in range(2):
                ref_dy_dW[p, :] = np.sum(x_ref[p, :])
            _assert_allclose(np.asarray(out.get("dy_dW")), ref_dy_dW, msg="dy_dW")
            ref_dy_db = np.ones((2, 2), dtype=np.float64)
            _assert_allclose(np.asarray(out.get("dy_db")), ref_dy_db, msg="dy_db")
        except ImportError:
            pass

    def test_einstein_batched_matmul_3d_vs_doc(self):
        """3D batched matmul: C[b,i,j]=sum[k](A[b,i,k]*B[b,k,j]); compare dC_dA, dC_dB to doc (∂C/∂A)_{bijrs}=δ_{ir}B_{bsj}, (∂C/∂B)_{bijrs}=δ_{js}A_{bir}."""
        source = """
let A = [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5], [0.1, 0.2]]];
let B = [[[5.0, 6.0], [7.0, 8.0]], [[1.0, 1.0], [1.0, 1.0]]];
let C[b, i, j] = sum[k](A[b, i, k] * B[b, k, j]);
let dC_dA = @C / @A;
let dC_dB = @C / @B;
"""
        _, out = _compile_run(source)
        dC_dA = np.asarray(out.get("dC_dA"))
        dC_dB = np.asarray(out.get("dC_dB"))
        assert dC_dA is not None and dC_dB is not None
        A_ref = np.array([[[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5], [0.1, 0.2]]])
        B_ref = np.array([[[5.0, 6.0], [7.0, 8.0]], [[1.0, 1.0], [1.0, 1.0]]])
        if dC_dA.ndim == 6:
            ref_dC_dA = np.zeros_like(dC_dA)
            for b in range(ref_dC_dA.shape[0]):
                for i in range(ref_dC_dA.shape[1]):
                    for j in range(ref_dC_dA.shape[2]):
                        for bp in range(ref_dC_dA.shape[3]):
                            for r in range(ref_dC_dA.shape[4]):
                                for s in range(ref_dC_dA.shape[5]):
                                    ref_dC_dA[b, i, j, bp, r, s] = (B_ref[b, s, j] if (i == r and b == bp) else 0.0)
            ref_dC_dB = np.zeros_like(dC_dB)
            for b in range(ref_dC_dB.shape[0]):
                for i in range(ref_dC_dB.shape[1]):
                    for j in range(ref_dC_dB.shape[2]):
                        for bp in range(ref_dC_dB.shape[3]):
                            for r in range(ref_dC_dB.shape[4]):
                                for s in range(ref_dC_dB.shape[5]):
                                    ref_dC_dB[b, i, j, bp, r, s] = (A_ref[b, i, r] if (j == s and b == bp) else 0.0)
            _assert_allclose(dC_dA, ref_dC_dA, msg="dC_dA 3D vs doc")
            _assert_allclose(dC_dB, ref_dC_dB, msg="dC_dB 3D vs doc")
        elif dC_dA.ndim == 5:
            ref_dC_dA = np.zeros_like(dC_dA)
            for b in range(ref_dC_dA.shape[0]):
                for i in range(ref_dC_dA.shape[1]):
                    for j in range(ref_dC_dA.shape[2]):
                        for r in range(ref_dC_dA.shape[3]):
                            for s in range(ref_dC_dA.shape[4]):
                                ref_dC_dA[b, i, j, r, s] = B_ref[b, s, j] if i == r else 0.0
            ref_dC_dB = np.zeros_like(dC_dB)
            for b in range(ref_dC_dB.shape[0]):
                for i in range(ref_dC_dB.shape[1]):
                    for j in range(ref_dC_dB.shape[2]):
                        for r in range(ref_dC_dB.shape[3]):
                            for s in range(ref_dC_dB.shape[4]):
                                ref_dC_dB[b, i, j, r, s] = A_ref[b, i, r] if j == s else 0.0
            _assert_allclose(dC_dA, ref_dC_dA, msg="dC_dA 3D vs doc")
            _assert_allclose(dC_dB, ref_dC_dB, msg="dC_dB 3D vs doc")
        else:
            assert dC_dA.shape == A_ref.shape and dC_dB.shape == B_ref.shape
            ref_dA = np.einsum("bij,bkj->bik", np.ones_like(A_ref), B_ref)
            ref_dB = np.einsum("bij,bir->brj", np.ones_like(B_ref), A_ref)
            _assert_allclose(dC_dA, ref_dA, msg="dC_dA 3D grad shape vs doc")
            _assert_allclose(dC_dB, ref_dB, msg="dC_dB 3D grad shape vs doc")

    def test_einstein_batched_reduction_sum_3d_vs_doc(self):
        """3D batched sum: y[b,i]=sum[j](x[b,i,j]); compare dy_dx to doc ∂y_{bi}/∂x_{bpq}=1 (grad shape of x, ones)."""
        source = """
let x = [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5], [0.1, 0.2]]];
let y[b, i] = sum[j](x[b, i, j]);
let dy_dx = @y / @x;
"""
        _, out = _compile_run(source)
        dy_dx = np.asarray(out.get("dy_dx"))
        assert dy_dx is not None
        x_ref = np.array([[[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5], [0.1, 0.2]]])
        assert dy_dx.shape == (2, 2, 2), "dy_dx shape (2,2,2), got %s vs x %s" % (dy_dx.shape, x_ref.shape)
        ref = np.ones((2, 2, 2), dtype=np.float64)
        _assert_allclose(dy_dx, ref, msg="dy_dx 3D batched sum pullback")

    def test_gradient_descent_autodiff_example(self):
        """One gradient step on ||A*x - b||^2 using @loss/@x0, @loss/@x1; loss decreases, x_next -> (0.5, 0.5)."""
        source = """
let A = [[2.0, 0.0], [0.0, 2.0]];
let b = [1.0, 1.0];
let alpha = 0.2;
let x0 = 0.0;
let x1 = 0.0;
let r0 = A[0, 0] * x0 + A[0, 1] * x1 - b[0];
let r1 = A[1, 0] * x0 + A[1, 1] * x1 - b[1];
let loss = r0 * r0 + r1 * r1;
let g0 = @loss / @x0;
let g1 = @loss / @x1;
let x0_next = x0 - alpha * g0;
let x1_next = x1 - alpha * g1;
let r0_next = A[0, 0] * x0_next + A[0, 1] * x1_next - b[0];
let r1_next = A[1, 0] * x0_next + A[1, 1] * x1_next - b[1];
let loss_next = r0_next * r0_next + r1_next * r1_next;
"""
        _, out = _compile_run(source)
        loss_val = _scalar_float(out, "loss")
        loss_next = _scalar_float(out, "loss_next")
        x0_next = _scalar_float(out, "x0_next")
        x1_next = _scalar_float(out, "x1_next")
        assert loss_next < loss_val, "loss should decrease after one step"
        assert abs(x0_next - 0.8) < 1e-5 and abs(x1_next - 0.8) < 1e-5, "one step with alpha=0.2 from (0,0) gives (0.8, 0.8)"

    def test_mnist_train_autodiff_ops_small(self):
        """Exercise mnist_train autodiff ops on a tiny deterministic setup: sum, *, -, **2, %, indexing alias quotient, and logits/W quotient."""
        source = """
let X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let Y = [[0.5, -1.0], [1.5, 2.0], [-0.5, 0.25]];
let lr = 0.1;

let W_init = [[0.2, -0.3], [0.4, 0.1]];
let logits_v[j in 0..2] = sum[i in 0..2](X[0, i] * W_init[i, j]);
let d_logits_dW = @logits_v / @W_init;

let W[0, i in 0..2, j in 0..2] = W_init[i, j];
let W[step in 1..5, i in 0..2, j in 0..2] = {
    let n = (step - 1) % 3;
    let logit_j = sum[a in 0..2](X[n, a] * W[step - 1, a, j]);
    let loss_b = (logit_j - Y[n, j]) ** 2.0;
    let w_ij = W[step - 1, i, j];
    let g = @loss_b / @w_ij;
    w_ij - lr * g
};
let W1[i in 0..2, j in 0..2] = W[1, i, j];
let W2[i in 0..2, j in 0..2] = W[2, i, j];
let W3[i in 0..2, j in 0..2] = W[3, i, j];
let W4[i in 0..2, j in 0..2] = W[4, i, j];
"""
        _, out = _compile_run(source)
        d_logits_dW = np.asarray(out.get("d_logits_dW"))
        W1 = np.asarray(out.get("W1"))
        W2 = np.asarray(out.get("W2"))
        W3 = np.asarray(out.get("W3"))
        W4 = np.asarray(out.get("W4"))
        assert d_logits_dW.shape == (2, 2), "expected d_logits_dW shape (2,2), got %s" % (d_logits_dW.shape,)
        assert W1.shape == (2, 2), "expected W1 shape (2,2), got %s" % (W1.shape,)
        assert W2.shape == (2, 2), "expected W2 shape (2,2), got %s" % (W2.shape,)
        assert W3.shape == (2, 2), "expected W3 shape (2,2), got %s" % (W3.shape,)
        assert W4.shape == (2, 2), "expected W4 shape (2,2), got %s" % (W4.shape,)
        _assert_allclose(d_logits_dW, np.array([[1.0, 1.0], [2.0, 2.0]]), msg="mnist logits quotient")
        # Validate recurrence with modulo index cycling n = (step-1) % 3.
        X_ref = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        Y_ref = np.array([[0.5, -1.0], [1.5, 2.0], [-0.5, 0.25]], dtype=np.float64)
        lr = 0.1
        W_init_ref = np.array([[0.2, -0.3], [0.4, 0.1]], dtype=np.float64)
        W_ref = [W_init_ref]
        for step in range(1, 5):
            n = (step - 1) % 3
            prev = W_ref[-1]
            cur = np.zeros_like(prev)
            for i in range(2):
                for j in range(2):
                    logit_j = np.sum(X_ref[n, :] * prev[:, j])
                    g = 2.0 * (logit_j - Y_ref[n, j]) * X_ref[n, i]
                    cur[i, j] = prev[i, j] - lr * g
            W_ref.append(cur)
        _assert_allclose(W1, W_ref[1], msg="mnist recurrence step1")
        _assert_allclose(W2, W_ref[2], msg="mnist recurrence step2")
        _assert_allclose(W3, W_ref[3], msg="mnist recurrence step3")
        _assert_allclose(W4, W_ref[4], msg="mnist recurrence step4 modulo-cycle")
        assert np.isfinite(W1).all() and np.isfinite(W2).all() and np.isfinite(W3).all() and np.isfinite(W4).all()

    def test_mnist_main_differentiable_ops_small(self):
        """Cover mnist/main.ein differentiable ops on tiny tensors: logits Einstein sum and quotients wrt X and W."""
        source = """
let X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let W = [[0.2, -0.3], [0.4, 0.1]];

let logits[n in 0..3, j in 0..2] = sum[i in 0..2](X[n, i] * W[i, j]);
let d_logits_dX = @logits / @X;
let d_logits_dW = @logits / @W;
"""
        _, out = _compile_run(source)
        logits = np.asarray(out.get("logits"))
        d_logits_dX = np.asarray(out.get("d_logits_dX"))
        d_logits_dW = np.asarray(out.get("d_logits_dW"))

        assert logits.shape == (3, 2), "expected logits shape (3,2), got %s" % (logits.shape,)
        assert d_logits_dX.shape == (3, 2, 3, 2), "expected d_logits_dX shape (3,2,3,2), got %s" % (d_logits_dX.shape,)
        assert d_logits_dW.shape == (3, 2, 2, 2), "expected d_logits_dW shape (3,2,2,2), got %s" % (d_logits_dW.shape,)

        # Full Jacobians:
        # d_logits_dX[n,j,np,ip] = W[ip,j] if n==np else 0
        # d_logits_dW[n,j,ip,jp] = X[n,ip] if j==jp else 0
        X_ref = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        W_ref = np.array([[0.2, -0.3], [0.4, 0.1]])
        ref_dX = np.zeros((3, 2, 3, 2), dtype=np.float64)
        for n in range(3):
            for j in range(2):
                for np_ in range(3):
                    for ip in range(2):
                        ref_dX[n, j, np_, ip] = W_ref[ip, j] if n == np_ else 0.0
        ref_dW = np.zeros((3, 2, 2, 2), dtype=np.float64)
        for n in range(3):
            for j in range(2):
                for ip in range(2):
                    for jp in range(2):
                        ref_dW[n, j, ip, jp] = X_ref[n, ip] if j == jp else 0.0
        _assert_allclose(d_logits_dX, ref_dX, msg="mnist main logits/X full Jacobian")
        _assert_allclose(d_logits_dW, ref_dW, msg="mnist main logits/W full Jacobian")
        assert np.isfinite(d_logits_dX).all() and np.isfinite(d_logits_dW).all()

    def test_mnist_ops_each_have_y_over_x_quotient(self):
        """Explicit @y/@x checks per MNIST op pattern (sum, *, -, **, %, indexing alias, Einstein logits)."""
        cases = [
            (
                "mul",
                "let x = 3.0; let y = x * 4.0; let dy_dx = @y / @x;",
                4.0,
            ),
            (
                "sub",
                "let x = 3.0; let y = x - 5.0; let dy_dx = @y / @x;",
                1.0,
            ),
            (
                "pow2",
                "let x = 3.0; let y = x ** 2.0; let dy_dx = @y / @x;",
                6.0,
            ),
            (
                "sum_reduce",
                "let x = [1.0, 2.0, 3.0]; let y = sum[i in 0..3](x[i]); let dy_dx = @y / @x;",
                np.array([1.0, 1.0, 1.0], dtype=np.float64),
            ),
            (
                "conv2d_ein_local",
                (
                    "let input = [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]; "
                    "let x = [[0.5,0.5],[0.5,0.5]]; "
                    "let y[oh in 0..2, ow in 0..2] = sum[kh in 0..2, kw in 0..2](input[oh + kh, ow + kw] * x[kh, kw]); "
                    "let dy_dx = @y / @x;"
                ),
                np.array(
                    [
                        [[[1.0, 2.0], [4.0, 5.0]], [[2.0, 3.0], [5.0, 6.0]]],
                        [[[4.0, 5.0], [7.0, 8.0]], [[5.0, 6.0], [8.0, 9.0]]],
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                "relu",
                "let x = [-1.0, 0.0, 2.0]; let y = std::ml::relu(x); let dy_dx = @y / @x;",
                np.array([0.0, 0.0, 2.0], dtype=np.float64),
            ),
            (
                "max_pool",
                (
                    "let x = [[[[1.0, 2.0], [3.0, 4.0]]]]; "
                    "let y = std::ml::max_pool(x, [2,2], [2,2], [0,0]); "
                    "let dy_dx = @y / @x;"
                ),
                np.array([[[[0.0, 0.0], [0.0, 1.0]]]], dtype=np.float64),
            ),
            (
                "flatten_index",
                (
                    "let x = [[[[10.0, 20.0], [30.0, 40.0]]]]; "
                    "let y[k in 0..4] = x[0,0,k / 2, k % 2]; "
                    "let dy_dx = @y / @x;"
                ),
                np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
            ),
            (
                "ein_logits",
                (
                    "let x = [1.0, 2.0]; "
                    "let W = [[0.2, -0.3], [0.4, 0.1]]; "
                    "let y[j in 0..2] = sum[i in 0..2](x[i] * W[i, j]); "
                    "let dy_dx = @y / @x;"
                ),
                np.array([[0.2, 0.4], [-0.3, 0.1]], dtype=np.float64),
            ),
            (
                "index_alias",
                "let W = [[0.2, -0.3], [0.4, 0.1]]; let x = W[1, 0]; let y = x * 3.0; let dy_dx = @y / @x;",
                3.0,
            ),
            (
                "loss_alias_chain",
                (
                    "let X = [[1.0, 2.0]]; "
                    "let Y = [[0.5, -1.0]]; "
                    "let W = [[0.2, -0.3], [0.4, 0.1]]; "
                    "let j = 1; "
                    "let i = 0; "
                    "let logit = sum[a in 0..2](X[0, a] * W[a, j]); "
                    "let y = (logit - Y[0, j]) ** 2.0; "
                    "let x = W[i, j]; "
                    "let dy_dx = @y / @x;"
                ),
                1.8,
            ),
            (
                "mod_index_control",
                (
                    "let coeff = [2.0, 4.0, 6.0]; "
                    "let step = 4; "
                    "let n = (step - 1) % 3; "
                    "let x = 3.0; "
                    "let y = x * coeff[n]; "
                    "let dy_dx = @y / @x;"
                ),
                2.0,
            ),
        ]

        for name, source, expected in cases:
            _, out = _compile_run(source)
            actual = out.get("dy_dx")
            assert actual is not None, "case %s: missing dy_dx output" % name
            if isinstance(expected, np.ndarray):
                arr = np.asarray(actual)
                assert arr.shape == expected.shape, "case %s: shape %s != %s" % (name, arr.shape, expected.shape)
                _assert_allclose(arr, expected, msg="case %s: dy_dx" % name)
            else:
                got = float(actual) if hasattr(actual, "item") else float(actual)
                assert abs(got - float(expected)) < 1e-5, "case %s: dy_dx got %s expected %s" % (name, got, expected)

    def test_user_fn_custom_diff_rule(self):
        """Custom @fn f(x) { 2*@x } gives db_da = 2 for b = f(a) = 2*a."""
        source = """
fn f(x) { x * 2.0 }
@fn f(x) { 2.0 * @x }
let a = 1.0;
let b = f(a);
let db_da = @b / @a;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "db_da") - 2.0) < 1e-6

    def test_single_differential_binding(self):
        """let dw = @w; produces d_w binding and no quotient."""
        source = """
let w = 3.0;
let loss = w * 2.0;
let dw = @w;
"""
        result, out = _compile_run(source)
        analysis = result.tcx.get_analysis(AutodiffPass)
        assert analysis["diff_block"] is not None
        names = {getattr(b, "name", "") for b in (getattr(result.ir, "bindings", None) or [])}
        assert USER_DIFF_PREFIX + "w" in names

    # -------------------------------------------------------------------------
    # Math-like derivatives via user-defined functions (same as stdlib formulas)
    # -------------------------------------------------------------------------

    def test_quotient_exp_like_via_custom_fn(self):
        """Custom @fn for exp-like: f(x)=1+x; @fn gives d/dx = 1. So d(f(a))/da = 1."""
        source = """
fn linear_exp(x) { 1.0 + x }
@fn linear_exp(x) { @x }
let a = 0.5;
let b = linear_exp(a);
let db_da = @b / @a;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "db_da") - 1.0) < 1e-6

    def test_quotient_math_pow_two_arg(self):
        """pow(x,y)=x^y: d/dx = y*x^(y-1), d/dy = x^y*ln(x). At x=2,y=3: pow=8, d/dx=12, d/dy=8*ln(2)≈5.545."""
        source = """
fn pow_xy(x, y) { python::numpy::power(x, y) }
@fn pow_xy(x, y) { (y * (x ** (y - 1.0))) * @x + ((x ** y) * python::numpy::log(x)) * @y }
let x = 2.0;
let y = 3.0;
let z = pow_xy(x, y);
let dz_dx = @z / @x;
let dz_dy = @z / @y;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dz_dx") - 12.0) < 1e-5
        assert abs(_scalar_float(out, "dz_dy") - (8.0 * math.log(2))) < 1e-5

    def test_quotient_math_log10_log2_log1p_expm1(self):
        """log10'(x)=1/(x*ln(10)), log2'(x)=1/(x*ln(2)), log1p'(x)=1/(1+x), expm1'(x)=exp(x)."""
        source = """
fn log10(x) { python::numpy::log10(x) }
fn log2(x) { python::numpy::log2(x) }
fn log1p(x) { python::numpy::log1p(x) }
fn expm1(x) { python::numpy::expm1(x) }
@fn log10(x) { (1.0 / (x * python::numpy::log(10.0))) * @x }
@fn log2(x) { (1.0 / (x * python::numpy::log(2.0))) * @x }
@fn log1p(x) { (1.0 / (1.0 + x)) * @x }
@fn expm1(x) { python::numpy::exp(x) * @x }
let x10 = 10.0;
let x2 = 2.0;
let x0 = 0.0;
let d10 = @log10(x10) / @x10;
let d2 = @log2(x2) / @x2;
let d1p = @log1p(x0) / @x0;
let dem1 = @expm1(x0) / @x0;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d10") - (1.0 / (10.0 * math.log(10)))) < 1e-5
        assert abs(_scalar_float(out, "d2") - (1.0 / (2.0 * math.log(2)))) < 1e-5
        assert abs(_scalar_float(out, "d1p") - 1.0) < 1e-5
        assert abs(_scalar_float(out, "dem1") - 1.0) < 1e-5

    # -------------------------------------------------------------------------
    # Stdlib clamp functions: piecewise derivative (1 inside, 0 outside; subgradient at boundaries)
    # -------------------------------------------------------------------------

    def test_quotient_math_clamp(self):
        """clamp(x,lo,hi): d/dx = 1 if lo<x<hi else 0. At x=2,lo=0,hi=5 => inside d=1; at x=10 => above d=0."""
        source_inside = """
fn clamp(x, lo, hi) { if x < lo { lo } else { if x > hi { hi } else { x } } }
@fn clamp(x, lo, hi) { (if x > lo { if x < hi { 1.0 } else { 0.0 } } else { 0.0 }) * @x }
let lo = 0.0;
let hi = 5.0;
let x = 2.0;
let y = clamp(x, lo, hi);
let d = @y / @x;
"""
        _, out_inside = _compile_run(source_inside)
        assert abs(_scalar_float(out_inside, "d") - 1.0) < 1e-6
        source_above = """
fn clamp(x, lo, hi) { if x < lo { lo } else { if x > hi { hi } else { x } } }
@fn clamp(x, lo, hi) { (if x > lo { if x < hi { 1.0 } else { 0.0 } } else { 0.0 }) * @x }
let lo = 0.0;
let hi = 5.0;
let x = 10.0;
let y = clamp(x, lo, hi);
let d = @y / @x;
"""
        _, out_above = _compile_run(source_above)
        assert abs(_scalar_float(out_above, "d") - 0.0) < 1e-6

    def test_quotient_math_saturate(self):
        """saturate(x)=clamp(x,0,1). d/dx = 1 if 0<x<1 else 0. At x=0.5 => d=1; at x=1.5 => d=0."""
        source_inside = """
fn clamp(x, lo, hi) { if x < lo { lo } else { if x > hi { hi } else { x } } }
@fn clamp(x, lo, hi) { (if x > lo { if x < hi { 1.0 } else { 0.0 } } else { 0.0 }) * @x }
fn saturate(x) { clamp(x, 0.0, 1.0) }
@fn saturate(x) { (if x > 0.0 { if x < 1.0 { 1.0 } else { 0.0 } } else { 0.0 }) * @x }
let x = 0.5;
let y = saturate(x);
let d = @y / @x;
"""
        _, out_inside = _compile_run(source_inside)
        assert abs(_scalar_float(out_inside, "d") - 1.0) < 1e-6
        source_above = """
fn clamp(x, lo, hi) { if x < lo { lo } else { if x > hi { hi } else { x } } }
@fn clamp(x, lo, hi) { (if x > lo { if x < hi { 1.0 } else { 0.0 } } else { 0.0 }) * @x }
fn saturate(x) { clamp(x, 0.0, 1.0) }
@fn saturate(x) { (if x > 0.0 { if x < 1.0 { 1.0 } else { 0.0 } } else { 0.0 }) * @x }
let x = 1.5;
let y = saturate(x);
let d = @y / @x;
"""
        _, out_above = _compile_run(source_above)
        assert abs(_scalar_float(out_above, "d") - 0.0) < 1e-6

    def test_quotient_math_clamp_min(self):
        """clamp_min(x,m)=max(x,m). d/dx = 1 if x > m else 0."""
        source_above = """
fn clamp_min(x, m) { if x < m { m } else { x } }
@fn clamp_min(x, m) { (if x > m { 1.0 } else { 0.0 }) * @x }
let x = 3.0;
let m = 1.0;
let y = clamp_min(x, m);
let d = @y / @x;
"""
        _, out_above = _compile_run(source_above)
        assert abs(_scalar_float(out_above, "d") - 1.0) < 1e-6
        source_below = """
fn clamp_min(x, m) { if x < m { m } else { x } }
@fn clamp_min(x, m) { (if x > m { 1.0 } else { 0.0 }) * @x }
let x = 0.5;
let m = 1.0;
let y = clamp_min(x, m);
let d = @y / @x;
"""
        _, out_below = _compile_run(source_below)
        assert abs(_scalar_float(out_below, "d") - 0.0) < 1e-6

    def test_quotient_math_clamp_max(self):
        """clamp_max(x,m)=min(x,m). d/dx = 1 if x < m else 0."""
        source_below = """
fn clamp_max(x, m) { if x > m { m } else { x } }
@fn clamp_max(x, m) { (if x < m { 1.0 } else { 0.0 }) * @x }
let x = 1.0;
let m = 5.0;
let y = clamp_max(x, m);
let d = @y / @x;
"""
        _, out_below = _compile_run(source_below)
        assert abs(_scalar_float(out_below, "d") - 1.0) < 1e-6
        source_above = """
fn clamp_max(x, m) { if x > m { m } else { x } }
@fn clamp_max(x, m) { (if x < m { 1.0 } else { 0.0 }) * @x }
let x = 10.0;
let m = 5.0;
let y = clamp_max(x, m);
let d = @y / @x;
"""
        _, out_above = _compile_run(source_above)
        assert abs(_scalar_float(out_above, "d") - 0.0) < 1e-6

    def test_quotient_math_deg_to_rad(self):
        """deg_to_rad(d)=d*pi/180 => d/d(d)=pi/180. Local fn + @fn; at d=180 value=pi, @y/@d = pi/180."""
        source = """
fn deg_to_rad(d) { d * 3.14159265359 / 180.0 }
@fn deg_to_rad(d) { (3.14159265359 / 180.0) * @d }
let d = 180.0;
let y = deg_to_rad(d);
let dy_dd = @y / @d;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dy_dd") - (math.pi / 180.0)) < 1e-6

    def test_quotient_math_rad_to_deg(self):
        """rad_to_deg(r)=r*180/pi => d/d(r)=180/pi. Local fn + @fn; at r=pi derivative=180/pi."""
        source = """
fn rad_to_deg(r) { r * 180.0 / 3.14159265359 }
@fn rad_to_deg(r) { (180.0 / 3.14159265359) * @r }
let r = 3.14159265359;
let y = rad_to_deg(r);
let dy_dr = @y / @r;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dy_dr") - (180.0 / math.pi)) < 1e-6


def test_autodiff_ir_expanded_derivative_bindings_are_substantial():
    """Compile a program with multiple @y/@x (matmul, sum, affine); IR graph is large (expanded d_* bindings)."""
    source = """
let A = [[1.0, 2.0], [3.0, 4.0]];
let B = [[5.0, 6.0], [7.0, 8.0]];
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let dC_dA = @C / @A;
let M = [[1.0, 2.0], [3.0, 4.0]];
let r[i] = sum[j](M[i, j]);
let dr_dM = @r / @M;
let x = [[1.0, 2.0], [3.0, 4.0]];
let W = [[0.5, 0.5], [0.1, 0.2]];
let b = [0.1, 0.2];
let y[i, j] = sum[k](x[i, k] * W[j, k]) + b[j];
let dy_dx = @y / @x;
"""
    compiler = CompilerDriver()
    result = compiler.compile(source.strip(), source_file="<test>", root_path=_REPO_ROOT)
    assert result.success, result.get_errors() or "compile failed"
    ad = [b for b in (result.ir.bindings or []) if _is_autodiff_generated_binding(b)]
    assert len(ad) >= 3, "expected multiple autodiff-generated bindings (d*_* names)"
    assert _ir_unique_node_count(result.ir) > 200, "autodiff should expand to a non-trivial IR graph"


_IR_DUMP_OPS = [
    ("elementwise_unary", """
let x = 2.0;
let y = std::math::exp(x);
let dy_dx = @y / @x;
"""),
    ("elementwise_binary", """
let a = 3.0;
let b = 4.0;
let z = a * b;
let dz_da = @z / @a;
let dz_db = @z / @b;
"""),
    ("matmul", """
let A = [[1.0, 2.0], [3.0, 4.0]];
let B = [[5.0, 6.0], [7.0, 8.0]];
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let dC_dA = @C / @A;
let dC_dB = @C / @B;
"""),
    ("affine", """
let x = [[1.0, 2.0], [3.0, 4.0]];
let W = [[0.5, 0.5], [0.1, 0.2]];
let b = [0.1, 0.2];
let y[i, j] = sum[k](x[i, k] * W[j, k]) + b[j];
let dy_dx = @y / @x;
let dy_dW = @y / @W;
let dy_db = @y / @b;
"""),
    ("conv1d", """
let x = [[1.0, 2.0, 3.0, 4.0]];
let w = [0.5, 0.3];
let out[i, c] = sum[k](x[c, i + k] * w[k]) where i + k < 4;
let d_out_dw = @out / @w;
"""),
    ("conv2d", """
let x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
let w = [[0.5, 0.5], [0.5, 0.5]];
let out[oh in 0..2, ow in 0..2] = sum[kh in 0..2, kw in 0..2](x[oh + kh, ow + kw] * w[kh, kw]);
let d_out_dw = @out / @w;
"""),
    ("reduction_sum", """
let M = [[1.0, 2.0], [3.0, 4.0]];
let r[i] = sum[j](M[i, j]);
let dr_dM = @r / @M;
"""),
    ("reduction_max", """
let x = [[1.0, 3.0, 2.0]];
let y[b] = max[j](x[b, j]);
let dy_dx = @y / @x;
"""),
    ("reduction_min", """
let x = [[1.0, 3.0, 2.0]];
let y[b] = min[j](x[b, j]);
let dy_dx = @y / @x;
"""),
    ("reduction_prod", """
let x = [[1.0, 2.0, 3.0]];
let y[b] = prod[j](x[b, j]);
let dy_dx = @y / @x;
"""),
    ("row_sum", """
let M = [[1.0, 2.0], [3.0, 4.0]];
let r[i] = sum[j](M[i, j]);
let dr_dM = @r / @M;
"""),
    ("column_sum", """
let M = [[1.0, 2.0], [3.0, 4.0]];
let c[j] = sum[i](M[i, j]);
let dc_dM = @c / @M;
"""),
    ("two_factor", """
let A = [[1.0, 2.0], [3.0, 4.0]];
let b = [5.0, 6.0];
let y[i] = sum[j](A[i, j] * b[j]);
let dy_dA = @y / @A;
let dy_db = @y / @b;
"""),
    ("attention_matmul_chain", """
let scale = 0.5;
let Q = [[[1.0, 2.0], [3.0, 4.0]]];
let K = [[[1.0, 2.0], [3.0, 4.0]]];
let V = [[[1.0, 0.0], [0.0, 1.0]]];
let scores[b, i, j] = sum[d](Q[b, i, d] * K[b, j, d]) * scale;
let out[b, i, d] = sum[j](scores[b, i, j] * V[b, j, d]);
let d_out_d_Q = @out / @Q;
"""),
    ("batched_matmul", """
let A = [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5], [0.1, 0.2]]];
let B = [[[5.0, 6.0], [7.0, 8.0]], [[1.0, 1.0], [1.0, 1.0]]];
let C[b, i, j] = sum[k](A[b, i, k] * B[b, k, j]);
let dC_dA = @C / @A;
let dC_dB = @C / @B;
"""),
    ("batched_reduction_sum", """
let x = [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5], [0.1, 0.2]]];
let y[b, i] = sum[j](x[b, i, j]);
let dy_dx = @y / @x;
"""),
]


def _is_autodiff_generated_binding(binding):
    try:
        name = (binding.name or "").strip()
        return name.startswith("d") and "_" in name
    except Exception:
        return False


def test_autodiff_ir_dump_all_ops():
    """Each catalog op compiles after autodiff; full-program IR is non-trivial. See docs/AUTODIFF_EINSTEIN_OPS.md."""
    compiler = CompilerDriver()
    for op_name, source in _IR_DUMP_OPS:
        result = compiler.compile(source.strip(), source_file="<test>", root_path=_REPO_ROOT)
        assert result.success, "op %s: %s" % (op_name, result.get_errors())
        assert _ir_unique_node_count(result.ir) > 15, "op %s: expected non-trivial IR" % op_name


def test_autodiff_ir_dump_generated_only():
    """Autodiff-generated bindings (d_* / quotient names) form a non-empty IR subtree per catalog op."""
    compiler = CompilerDriver()
    for op_name, source in _IR_DUMP_OPS:
        result = compiler.compile(source.strip(), source_file="<test>", root_path=_REPO_ROOT)
        assert result.success, "op %s: %s" % (op_name, result.get_errors())
        program = result.ir
        derivative_bindings = [b for b in (program.bindings or []) if _is_autodiff_generated_binding(b)]
        assert len(derivative_bindings) > 0, "op %s: expected at least one autodiff-generated binding (name with _d_)" % op_name
        autodiff_only = ProgramIR(statements=derivative_bindings, source_files=program.source_files, modules=program.modules)
        assert _ir_unique_node_count(autodiff_only) >= 5, "op %s: expected non-trivial derivative-only IR" % op_name


def _expr_contains_node_type(expr, node_type, binding_by_defid=None, visited_defids=None):
    """Return True if expr tree contains a node of type node_type. If binding_by_defid is given, follow IdentifierIR to binding expr (one level per call to avoid infinite loop)."""
    if expr is None:
        return False
    if visited_defids is None:
        visited_defids = set()
    if type(expr).__name__ == node_type:
        return True
    if type(expr).__name__ == "IdentifierIR" and binding_by_defid and getattr(expr, "defid", None) and expr.defid not in visited_defids:
        b = binding_by_defid.get(expr.defid)
        if b and b.expr is not None:
            visited_defids.add(expr.defid)
            if _expr_contains_node_type(b.expr, node_type, binding_by_defid, visited_defids):
                return True
    for attr in ("left", "right", "operand", "expr", "body", "condition", "then_expr", "else_expr", "array", "callee_expr", "arguments", "primal_body", "diff_body"):
        if hasattr(expr, attr):
            val = getattr(expr, attr)
            if val is not None and _expr_contains_node_type(val, node_type, binding_by_defid, visited_defids):
                return True
    for attr in ("clauses", "statements", "items", "loops", "bindings"):
        if hasattr(expr, attr):
            for item in (getattr(expr, attr) or []):
                if _expr_contains_node_type(item, node_type, binding_by_defid, visited_defids):
                    return True
                if hasattr(item, "expr") and _expr_contains_node_type(getattr(item, "expr"), node_type, binding_by_defid, visited_defids):
                    return True
                if hasattr(item, "value") and _expr_contains_node_type(getattr(item, "value"), node_type, binding_by_defid, visited_defids):
                    return True
    return False


_OP_DOC_EXPECTATIONS = [
    ("elementwise_unary", {"dy_dx"}, "scalar"),
    ("elementwise_binary", {"dz_da", "dz_db"}, "scalar"),
    ("matmul", {"dC_dA", "dC_dB"}, "einstein"),
    ("affine", {"dy_dx", "dy_dW", "dy_db"}, "einstein"),
    ("conv1d", {"d_out_dw"}, "einstein"),
    ("conv2d", {"d_out_dw"}, "einstein"),
    ("reduction_sum", {"dr_dM"}, "einstein"),
    ("reduction_max", {"dy_dx"}, "select_at_argmax"),
    ("reduction_min", {"dy_dx"}, "select_at_argmax"),
    ("reduction_prod", {"dy_dx"}, "einstein"),
    ("row_sum", {"dr_dM"}, "einstein"),
    ("column_sum", {"dc_dM"}, "einstein"),
    ("two_factor", {"dy_dA", "dy_db"}, "einstein"),
    ("attention_matmul_chain", {"d_out_d_Q"}, "einstein"),
    ("batched_matmul", {"dC_dA", "dC_dB"}, "einstein"),
    ("batched_reduction_sum", {"dy_dx"}, "einstein"),
]

_OP_DOC_EXPECTED_SHAPES = {
    "elementwise_unary": [("dy_dx", ())],
    "elementwise_binary": [("dz_da", ()), ("dz_db", ())],
    "matmul": [("dC_dA", (2, 2, 2, 2)), ("dC_dB", (2, 2, 2, 2))],
    "affine": [("dy_dx", (2, 2)), ("dy_dW", (2, 2)), ("dy_db", (2, 2))],
    "conv2d": [("d_out_dw", (2, 2, 2, 2))],
    "reduction_sum": [("dr_dM", (2, 2))],
    "reduction_max": [("dy_dx", (1, 3))],
    "reduction_min": [("dy_dx", (1, 3))],
    "reduction_prod": [("dy_dx", (1, 3))],
    "row_sum": [("dr_dM", (2, 2))],
    "column_sum": [("dc_dM", (2, 2))],
    "two_factor": [("dy_dA", (2, 2)), ("dy_db", (2, 2))],
    "attention_matmul_chain": [("d_out_d_Q", (1, 2, 2))],
    "batched_matmul": [("dC_dA", (2, 2, 2, 2, 2, 2)), ("dC_dB", (2, 2, 2, 2, 2, 2))],
    "batched_reduction_sum": [("dy_dx", (2, 2, 2))],
}
_OP_DOC_EXPECTED_SHAPES_SKIP_RUNTIME = frozenset({"conv1d"})


def test_autodiff_dumped_ir_matches_doc():
    """Compare autodiff-generated IR (after compile) to doc: expected binding names and expr structure per AUTODIFF_EINSTEIN_OPS.md."""
    from einlang.ir.nodes import EinsteinIR, LoweredEinsteinIR, SelectAtArgmaxIR, LoweredSelectAtArgmaxIR, BindingIR
    compiler = CompilerDriver()
    for op_name, expected_names, structure in _OP_DOC_EXPECTATIONS:
        source = next(s for n, s in _IR_DUMP_OPS if n == op_name)
        result = compiler.compile(source.strip(), source_file="<test>", root_path=_REPO_ROOT)
        assert result.success, "op %s: %s" % (op_name, result.get_errors())
        program = result.ir
        binding_by_defid = {b.defid: b for b in (program.bindings or []) if getattr(b, "defid", None) is not None}
        derivative_bindings = [b for b in (program.bindings or []) if _is_autodiff_generated_binding(b)]
        names = {b.name for b in derivative_bindings}
        missing = expected_names - names
        assert not missing, "op %s: doc expects bindings %s; missing %s; got %s" % (op_name, expected_names, missing, names)
        for b in derivative_bindings:
            if b.name not in expected_names:
                continue
            if structure == "scalar":
                assert not _expr_contains_node_type(b.expr, "EinsteinIR", binding_by_defid) and not _expr_contains_node_type(b.expr, "LoweredEinsteinIR", binding_by_defid), (
                    "op %s binding %s: doc §1/§2 scalar derivative, but expr has Einstein" % (op_name, b.name))
            elif structure == "einstein":
                assert _expr_contains_node_type(b.expr, "EinsteinIR", binding_by_defid) or _expr_contains_node_type(b.expr, "LoweredEinsteinIR", binding_by_defid), (
                    "op %s binding %s: doc expects Einstein derivative clause; expr has no EinsteinIR/LoweredEinsteinIR" % (op_name, b.name))
            elif structure == "select_at_argmax":
                assert _expr_contains_node_type(b.expr, "SelectAtArgmaxIR", binding_by_defid) or _expr_contains_node_type(b.expr, "LoweredSelectAtArgmaxIR", binding_by_defid), (
                    "op %s binding %s: doc §6 max/min δ at argmax/argmin; expr has no SelectAtArgmaxIR" % (op_name, b.name))
            elif structure == "einstein_or_any":
                pass


def test_autodiff_dumped_ir_shapes_match_doc():
    """Assert runtime output shapes of derivative bindings match doc (AUTODIFF_EINSTEIN_OPS): full Jacobian or grad shape = input shape."""
    for op_name, shape_specs in _OP_DOC_EXPECTED_SHAPES.items():
        if op_name in _OP_DOC_EXPECTED_SHAPES_SKIP_RUNTIME:
            continue
        source = next(s for n, s in _IR_DUMP_OPS if n == op_name)
        _, out = _compile_run(source)
        for binding_name, expected_shape in shape_specs:
            val = out.get(binding_name)
            assert val is not None, "op %s: missing output %s" % (op_name, binding_name)
            arr = np.asarray(val)
            actual = arr.shape
            assert actual == expected_shape, (
                "op %s binding %s: doc expects shape %s, got %s" % (op_name, binding_name, expected_shape, actual))
