"""
Unit tests for AutodiffPass.

Tests that the autodiff pass runs and expands derivative nodes (@expr, @num/@den)
into plain IR (d_* bindings and references). No diff block; derivatives are in-program.
All tests expect compile and run success; derivative tests assert correct values.

Coverage: pipeline registration, no-@ programs, @expr expansion, quotient @num/@den,
scalar math (add/sub/mul/div/pow/mod), unary neg, chain rule through lets, user functions,
custom @fn rules, Einstein ∂C/∂A, multiple quotients, constant derivative zero.
Math-like derivatives via user fns: sqrt (x**0.5), reciprocal (1/x), square (x*x).
PyTorch-style ops: relu, sigmoid, softplus, leaky_relu, elu, reciprocal; all tested in TestPyTorchStyleOps.

Tensor / Einstein autodiff: matmul ∂C/∂A and ∂C/∂B (2x2 and 3x3), both quotients in one
program, row-sum ∂r/∂M and column-sum ∂c/∂M, 1D conv (index expr) ∂out/∂w, matrix-vector
product y[i]=sum[j](A[i,j]*b[j]) with ∂y/∂A and ∂y/∂b. Only sum-of-products Einstein
clauses are differentiated; scalar-from-reduction and elementwise-only Einstein are not.

Note: stdlib module calls (use std::math::sqrt etc.) are not yet supported for
autodiff (callee not in program.bindings); use local user fns or @fn for such math.
"""

import math

from einlang.compiler.driver import CompilerDriver
from einlang.passes.autodiff import AutodiffPass


def _scalar_float(outputs, key):
    v = outputs.get(key)
    assert v is not None, "expected output %r, got %s" % (key, list(outputs.keys()))
    return float(v) if hasattr(v, "item") else float(v)


def _compile_run(source, expect_success=True, root_path=None):
    from pathlib import Path
    compiler = CompilerDriver()
    if root_path is None:
        root_path = Path.cwd()
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
        d_bindings = [b for b in bindings if getattr(b, "name", "").startswith("d_")]
        assert len(d_bindings) >= 1

    def test_quotient_binary_expr(self):
        """@b/@a for b = a*a: compile and run; assert db_da == 2*a == 6 at a=3."""
        compiler = CompilerDriver()
        source = """
let a = 3.0;
let b = a * a;
let db_da = @b / @a;
print(db_da);
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
print(db_da);
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

    def test_einstein_quotient_compiles_and_runs(self):
        """@C/@A when C is Einstein sum: autodiff expands to ∂C/∂A Einstein; compile and run; assert dC_dA shape."""
        compiler = CompilerDriver()
        source = """
let A = [[1.0, 2.0], [3.0, 4.0]];
let B = [[5.0, 6.0], [7.0, 8.0]];
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let dC_dA = @C / @A;
print(dC_dA);
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
            # Strict comparison with reference (enable when backend applies derivative reduction guards)
            max_diff = np.abs(np.asarray(arr, dtype=np.float64) - ref).max()
            assert max_diff < 1e-5, (
                "dC_dA should match NumPy reference (∂C/∂A for C=A@B); max |diff| = %s" % max_diff
            )
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
            max_diff = np.abs(np.asarray(arr, dtype=np.float64) - ref).max()
            assert max_diff < 1e-5, "dC_dB should match ∂C/∂B; max |diff| = %s" % max_diff
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
            import numpy as np
            dA = np.asarray(out.get("dC_dA"))
            dB = np.asarray(out.get("dC_dB"))
            assert dA.shape == (2, 2, 2, 2) and dB.shape == (2, 2, 2, 2)
            A_ref = np.array([[1.0, 2.0], [3.0, 4.0]])
            B_ref = np.array([[5.0, 6.0], [7.0, 8.0]])
            for i in range(2):
                for j in range(2):
                    for r in range(2):
                        for s in range(2):
                            expect_a = B_ref[s, j] if i == r else 0.0
                            assert abs(float(dA[i, j, r, s]) - expect_a) < 1e-5
                    for s in range(2):
                        for t in range(2):
                            expect_b = A_ref[i, s] if t == j else 0.0
                            assert abs(float(dB[i, j, s, t]) - expect_b) < 1e-5
        except ImportError:
            pass

    def test_einstein_row_sum_derivative(self):
        """r[i] = sum[j](M[i,j]); @r/@M is 3-tensor: ∂r[i]/∂M[r,s] = 1 if i==r and s any, else 0."""
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
            assert arr.ndim == 3, "dr_dM should be 3D (i, r, s), got ndim %s" % arr.ndim
            assert arr.shape == (2, 2, 2), "shape (2,2,2), got %s" % (arr.shape,)
            ref = np.zeros((2, 2, 2), dtype=np.float64)
            for i in range(2):
                for r in range(2):
                    for s in range(2):
                        ref[i, r, s] = 1.0 if i == r else 0.0
            max_diff = np.abs(np.asarray(arr, dtype=np.float64) - ref).max()
            assert max_diff < 1e-5, "dr_dM should match ∂r/∂M; max |diff| = %s" % max_diff
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
            # out has shape (2,) for oh=0,1 (oh+kh < 3); wrt w has shape (2,) => derivative shape (2, 2)
            assert arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == 2, (
                "d_out_dw shape (2,2), got %s" % (arr.shape,)
            )
            assert np.isfinite(arr).all()
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
            import numpy as np
            arr = np.asarray(dC_dA)
            assert arr.ndim == 4 and arr.shape == (3, 3, 3, 3)
            assert np.isfinite(arr).all()
        except ImportError:
            pass

    def test_einstein_column_sum_derivative(self):
        """c[j] = sum[i](M[i,j]); @c/@M is 3-tensor (output j, wrt i,j)."""
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
            assert arr.ndim == 3 and arr.shape == (2, 2, 2)
            ref = np.zeros((2, 2, 2), dtype=np.float64)
            for j in range(2):
                for r in range(2):
                    for s in range(2):
                        ref[j, r, s] = 1.0 if s == j else 0.0
            assert np.allclose(arr, ref, atol=1e-5)
        except ImportError:
            pass

    def test_einstein_attention_matmul_chain_no_softmax(self):
        """Single-head attention matmul chain (no softmax): scores = Q@K^T, out = scores@V; @out/@Q.
        MHA uses this plus softmax; the matmul part is differentiable. Asserts compile/run and finite output."""
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
            import numpy as np
            arr = np.asarray(d_out_d_Q)
            assert np.isfinite(arr).all() or arr.size == 1, "d_out_d_Q should be finite"
            # Full Jacobian is 6D (out indices + Q indices); per-quotient run may give tensor or scalar
            assert arr.ndim in (0, 3, 6), "d_out_d_Q ndim 0 (scalar), 3 (out shape), or 6 (full Jacobian)"
        except ImportError:
            pass

    def test_einstein_two_factor_product(self):
        """Single-index sum: y[i] = sum[j](A[i,j]*b[j]); @y/@A and @y/@b."""
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
            import numpy as np
            dy_dA = np.asarray(out.get("dy_dA"))
            dy_db = np.asarray(out.get("dy_db"))
            assert dy_dA.shape == (2, 2, 2), "∂y/∂A shape (i, i, j) = (2,2,2), got %s" % (dy_dA.shape,)
            assert dy_db.shape == (2, 2), "∂y/∂b shape (i, j) = (2,2), got %s" % (dy_db.shape,)
        except ImportError:
            pass

    # -------------------------------------------------------------------------
    # Scalar math: each binary op and unary op
    # -------------------------------------------------------------------------

    def test_quotient_add(self):
        """∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1."""
        source = """
let x = 2.0;
let y = 3.0;
let z = x + y;
let dz_dx = @z / @x;
let dz_dy = @z / @y;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dz_dx") - 1.0) < 1e-6
        assert abs(_scalar_float(out, "dz_dy") - 1.0) < 1e-6

    def test_quotient_sub(self):
        """∂(x-y)/∂x = 1, ∂(x-y)/∂y = -1."""
        source = """
let x = 5.0;
let y = 2.0;
let u = x - y;
let du_dx = @u / @x;
let du_dy = @u / @y;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "du_dx") - 1.0) < 1e-6
        assert abs(_scalar_float(out, "du_dy") - (-1.0)) < 1e-6

    def test_quotient_mul(self):
        """∂(x*y)/∂x = y, ∂(x*y)/∂y = x."""
        source = """
let x = 3.0;
let y = 4.0;
let w = x * y;
let dw_dx = @w / @x;
let dw_dy = @w / @y;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dw_dx") - 4.0) < 1e-6
        assert abs(_scalar_float(out, "dw_dy") - 3.0) < 1e-6

    def test_quotient_div(self):
        """∂(x/y)/∂x = 1/y, ∂(x/y)/∂y = -x/y². At x=3, y=2: 0.5 and -0.75."""
        source = """
let x = 3.0;
let y = 2.0;
let v = x / y;
let dv_dx = @v / @x;
let dv_dy = @v / @y;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dv_dx") - 0.5) < 1e-6
        assert abs(_scalar_float(out, "dv_dy") - (-0.75)) < 1e-6

    def test_quotient_compound_denominator(self):
        """@x/@(x + x**2) = (dx/dx)/(d(x+x²)/dx) = 1/(1+2x). At x=2 => 1/5 = 0.2."""
        source = """
let x = 2.0;
let ratio = @x / @(x + x * x);
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "ratio") - 0.2) < 1e-6

    def test_quotient_pow_literal_exponent(self):
        """∂(x^n)/∂x = n*x^(n-1). x**3 at x=2 => 3*4=12. (POW with literal exponent.)"""
        source = """
let x = 2.0;
let p = x ** 3.0;
let dp_dx = @p / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dp_dx") - 12.0) < 1e-6

    def test_quotient_pow_square(self):
        """∂(x²)/∂x = 2*x. At x=5 => 10."""
        source = """
let x = 5.0;
let q = x * x;
let dq_dx = @q / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dq_dx") - 10.0) < 1e-6

    def test_quotient_unary_neg(self):
        """∂(-x)/∂x = -1."""
        source = """
let x = 7.0;
let n = -x;
let dn_dx = @n / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dn_dx") - (-1.0)) < 1e-6

    def test_chain_rule_through_lets(self):
        """c = b+1, b = a*a => ∂c/∂a = 2*a. At a=3 => 6."""
        source = """
let a = 3.0;
let b = a * a;
let c = b + 1.0;
let dc_da = @c / @a;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dc_da") - 6.0) < 1e-6

    def test_constant_derivative_zero(self):
        """∂k/∂x = 0 when k is a literal (no dependence on x)."""
        source = """
let x = 1.0;
let k = 5.0;
let dk_dx = @k / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dk_dx") - 0.0) < 1e-6

    def test_identifier_wrt_self_one(self):
        """When x = a (identifier), ∂x/∂a = 1. (@x/@a expands using x's defining expr.)"""
        source = """
let a = 42.0;
let x = a;
let dx_da = @x / @a;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dx_da") - 1.0) < 1e-6

    def test_multiple_quotients_same_program(self):
        """Several @num/@den in one program: z=x+y, w=x*y; dz_dx, dw_dx both correct."""
        source = """
let x = 1.0;
let y = 2.0;
let z = x + y;
let w = x * y;
let dz_dx = @z / @x;
let dw_dx = @w / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dz_dx") - 1.0) < 1e-6
        assert abs(_scalar_float(out, "dw_dx") - 2.0) < 1e-6

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
        d_bindings = [b for b in (getattr(result.ir, "bindings", None) or []) if getattr(b, "name", "").startswith("d_")]
        assert any(getattr(b, "name", "") == "d_w" for b in d_bindings)

    # -------------------------------------------------------------------------
    # Math-like derivatives via user-defined functions (same as stdlib formulas)
    # -------------------------------------------------------------------------

    def test_quotient_sqrt_via_user_fn(self):
        """sqrt(x) = x**0.5 => d(sqrt(x))/dx = 1/(2*sqrt(x)). At x=4 => 0.25."""
        source = """
fn sqrt_fn(x) { x ** 0.5 }
let x = 4.0;
let y = sqrt_fn(x);
let dy_dx = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dy_dx") - 0.25) < 1e-6

    def test_quotient_reciprocal_via_user_fn(self):
        """reciprocal(x) = 1/x => d(1/x)/dx = -1/x². At x=2 => -0.25."""
        source = """
fn rec(x) { 1.0 / x }
let x = 2.0;
let y = rec(x);
let dy_dx = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dy_dx") - (-0.25)) < 1e-6

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

    def test_quotient_math_log(self):
        """d/dx log(x)=1/x. At x=1 => 1. @fn body has no python:: so type_info OK."""
        source = """
fn log(x) { python::numpy::log(x) }
@fn log(x) { (1.0 / x) * @x }
let b = 1.0;
let yl = log(b);
let dlog = @yl / @b;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dlog") - 1.0) < 1e-5

    def test_quotient_user_fn_without_at_fn(self):
        """Differentiate through user fn body when no @fn: d/dx (x*x) = 2*x. At x=2: 4."""
        source = """
fn sq(x) { x * x }
let x = 2.0;
let y = sq(x);
let dy_dx = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dy_dx") - 4.0) < 1e-5

    def test_quotient_math_asin_atan(self):
        """d/dx asin(x)=1/sqrt(1-x²), atan(x)=1/(1+x²). At x=0: 1 and 1. @fn body literals+param only."""
        source = """
fn asin(x) { python::numpy::arcsin(x) }
@fn asin(x) { (1.0 / ((1.0 - x * x) ** 0.5)) * @x }
fn atan(x) { python::numpy::arctan(x) }
@fn atan(x) { (1.0 / (1.0 + x * x)) * @x }
let x = 0.0;
let ya = asin(x);
let yt = atan(x);
let dasin = @ya / @x;
let datan = @yt / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dasin") - 1.0) < 1e-5
        assert abs(_scalar_float(out, "datan") - 1.0) < 1e-5

    def test_quotient_math_atan2(self):
        """atan2(y,x): d/dy = x/(x²+y²), d/dx = -y/(x²+y²). At (1,0): 0 and -1."""
        source = """
fn atan2(y, x) { python::numpy::arctan2(y, x) }
@fn atan2(y, x) { (x / (x * x + y * y)) * @y + (-y / (x * x + y * y)) * @x }
let y = 1.0;
let x = 0.0;
let a = atan2(y, x);
let da_dy = @a / @y;
let da_dx = @a / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "da_dy") - 0.0) < 1e-5
        assert abs(_scalar_float(out, "da_dx") - (-1.0)) < 1e-5

    def test_quotient_math_trig_all_stdlib_like(self):
        """All stdlib trig in one program: sin, cos, tan, asin, acos, atan, atan2. At x=0,y=1: ds=1, dc=0, dt=1, dasin=1, dacos=-1, datan=1, atan2 da_dy=0, da_dx=-1."""
        source = """
fn sin(x) { python::numpy::sin(x) }
fn cos(x) { python::numpy::cos(x) }
fn tan(x) { python::numpy::tan(x) }
fn asin(x) { python::numpy::arcsin(x) }
fn acos(x) { python::numpy::arccos(x) }
fn atan(x) { python::numpy::arctan(x) }
fn atan2(y, x) { python::numpy::arctan2(y, x) }
@fn sin(x) { cos(x) * @x }
@fn cos(x) { (-sin(x)) * @x }
@fn tan(x) { (1.0 / (cos(x) * cos(x))) * @x }
@fn asin(x) { (1.0 / ((1.0 - x * x) ** 0.5)) * @x }
@fn acos(x) { (-1.0 / ((1.0 - x * x) ** 0.5)) * @x }
@fn atan(x) { (1.0 / (1.0 + x * x)) * @x }
@fn atan2(y, x) { (x / (x * x + y * y)) * @y + (-y / (x * x + y * y)) * @x }
let x = 0.0;
let y = 1.0;
let ys = sin(x);
let yc = cos(x);
let yt = tan(x);
let ya = asin(x);
let yac = acos(x);
let yat = atan(x);
let a = atan2(y, x);
let ds = @ys / @x;
let dc = @yc / @x;
let dt = @yt / @x;
let dasin = @ya / @x;
let dacos = @yac / @x;
let datan = @yat / @x;
let da_dy = @a / @y;
let da_dx = @a / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "ds") - 1.0) < 1e-5
        assert abs(_scalar_float(out, "dc") - 0.0) < 1e-5
        assert abs(_scalar_float(out, "dt") - 1.0) < 1e-5
        assert abs(_scalar_float(out, "dasin") - 1.0) < 1e-5
        assert abs(_scalar_float(out, "dacos") - (-1.0)) < 1e-5
        assert abs(_scalar_float(out, "datan") - 1.0) < 1e-5
        assert abs(_scalar_float(out, "da_dy") - 0.0) < 1e-5
        assert abs(_scalar_float(out, "da_dx") - (-1.0)) < 1e-5

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

    def test_quotient_math_exp(self):
        """d/dx exp(x)=exp(x). At x=0 => 1."""
        source = """
fn exp(x) { python::numpy::exp(x) }
@fn exp(x) { python::numpy::exp(x) * @x }
let x = 0.0;
let y = exp(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d") - 1.0) < 1e-5

    def test_quotient_math_sin_cos_tan(self):
        """d/dx sin=cos, cos=-sin, tan=1/cos². At x=0: 1, 0, 1."""
        source = """
fn sin(x) { python::numpy::sin(x) }
fn cos(x) { python::numpy::cos(x) }
fn tan(x) { python::numpy::tan(x) }
@fn sin(x) { cos(x) * @x }
@fn cos(x) { (-sin(x)) * @x }
@fn tan(x) { (1.0 / (cos(x) * cos(x))) * @x }
let x = 0.0;
let ys = sin(x);
let yc = cos(x);
let yt = tan(x);
let ds = @ys / @x;
let dc = @yc / @x;
let dt = @yt / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "ds") - 1.0) < 1e-5
        assert abs(_scalar_float(out, "dc") - 0.0) < 1e-5
        assert abs(_scalar_float(out, "dt") - 1.0) < 1e-5

    def test_quotient_math_acos(self):
        """d/dx acos(x)=-1/sqrt(1-x²). At x=0 => -1."""
        source = """
fn acos(x) { python::numpy::arccos(x) }
@fn acos(x) { (-1.0 / ((1.0 - x * x) ** 0.5)) * @x }
let x = 0.0;
let y = acos(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d") - (-1.0)) < 1e-5

    def test_quotient_math_sqrt(self):
        """d/dx sqrt(x)=1/(2*sqrt(x)). At x=1 => 0.5."""
        source = """
fn sqrt(x) { python::numpy::sqrt(x) }
@fn sqrt(x) { (0.5 / python::numpy::sqrt(x)) * @x }
let x = 1.0;
let y = sqrt(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d") - 0.5) < 1e-5

    def test_quotient_math_sinh_cosh_tanh(self):
        """d/dx sinh=cosh, cosh=sinh, tanh=1-tanh². At x=0: 1, 0, 1."""
        source = """
fn sinh(x) { python::numpy::sinh(x) }
fn cosh(x) { python::numpy::cosh(x) }
fn tanh(x) { python::numpy::tanh(x) }
@fn sinh(x) { cosh(x) * @x }
@fn cosh(x) { sinh(x) * @x }
@fn tanh(x) { (1.0 - tanh(x) * tanh(x)) * @x }
let x = 0.0;
let ys = sinh(x);
let yc = cosh(x);
let yt = tanh(x);
let ds = @ys / @x;
let dc = @yc / @x;
let dt = @yt / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "ds") - 1.0) < 1e-5
        assert abs(_scalar_float(out, "dc") - 0.0) < 1e-5
        assert abs(_scalar_float(out, "dt") - 1.0) < 1e-5

    def test_quotient_math_asinh(self):
        """d/dx asinh(x)=1/sqrt(1+x²). At x=0 => 1."""
        source = """
fn asinh(x) { python::numpy::arcsinh(x) }
@fn asinh(x) { (1.0 / ((1.0 + x * x) ** 0.5)) * @x }
let x = 0.0;
let y = asinh(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d") - 1.0) < 1e-5

    def test_quotient_math_acosh(self):
        """d/dx acosh(x)=1/sqrt(x²-1). At x=1.1, derivative ≈ 1/sqrt(0.21)≈2.18."""
        source = """
fn acosh(x) { python::numpy::arccosh(x) }
@fn acosh(x) { (1.0 / ((x * x - 1.0) ** 0.5)) * @x }
let x = 1.1;
let y = acosh(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d") - (1.0 / (0.21**0.5))) < 1e-4

    def test_quotient_math_atanh(self):
        """d/dx atanh(x)=1/(1-x²). At x=0 => 1."""
        source = """
fn atanh(x) { python::numpy::arctanh(x) }
@fn atanh(x) { (1.0 / (1.0 - x * x)) * @x }
let x = 0.0;
let y = atanh(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d") - 1.0) < 1e-5

    def test_quotient_math_erf(self):
        """d/dx erf(x) = (2/sqrt(pi)) * exp(-x²). At x=0, derivative = 2/sqrt(pi) ≈ 1.128."""
        source = """
fn erf(x) { (2.0 / python::numpy::sqrt(python::numpy::pi)) * x }
@fn erf(x) { (2.0 / python::numpy::sqrt(python::numpy::pi)) * python::numpy::exp(0.0 - x * x) * @x }
let x = 0.0;
let y = erf(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        expected = 2.0 / (math.pi ** 0.5)
        assert abs(_scalar_float(out, "d") - expected) < 1e-5

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

    def test_quotient_math_neg(self):
        """d/dx (-x) = -1. Local fn neg(x){ -x }, @fn neg(x){ -1.0 * @x }. At x=3 => @y/@x == -1."""
        source = """
fn neg(x) { -x }
@fn neg(x) { -1.0 * @x }
let x = 3.0;
let y = neg(x);
let dy_dx = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dy_dx") - (-1.0)) < 1e-6

    def test_quotient_math_square(self):
        """stdlib square(x)=x*x, d/dx = 2*x. fn square(x){ x * x }, @fn square(x){ 2.0 * x * @x }. At x=3 => 6."""
        source = """
fn square(x) { x * x }
@fn square(x) { 2.0 * x * @x }
let x = 3.0;
let y = square(x);
let dy_dx = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dy_dx") - 6.0) < 1e-6

    def test_quotient_math_abs(self):
        """d/dx abs(x) = sign(x). fn abs via if; @fn: 1 for x>0, -1 for x<0. At x=2 => 1, at x=-2 => -1."""
        source = """
fn abs(x) { if (x as f32) >= 0.0 { x } else { -x } }
@fn abs(x) { (if (x as f32) > 0.0 { 1.0 } else { -1.0 }) * @x }
let x_pos = 2.0;
let x_neg = -2.0;
let y_pos = abs(x_pos);
let y_neg = abs(x_neg);
let d_pos = @y_pos / @x_pos;
let d_neg = @y_neg / @x_neg;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d_pos") - 1.0) < 1e-6
        assert abs(_scalar_float(out, "d_neg") - (-1.0)) < 1e-6

    def test_quotient_math_sign(self):
        """sign(x): -1 for x<0, 0 for x=0, 1 for x>0. Derivative 0 (subgradient at 0). d/dx sign(x)=0 at x=1 and x=-1."""
        source = """
fn sign(x) { if (x as f32) > 0.0 { 1.0 } else if (x as f32) < 0.0 { -1.0 } else { 0.0 } }
@fn sign(x) { 0.0 * @x }
let x = 1.0;
let y = sign(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d")) < 1e-6
        source_neg = """
fn sign(x) { if (x as f32) > 0.0 { 1.0 } else if (x as f32) < 0.0 { -1.0 } else { 0.0 } }
@fn sign(x) { 0.0 * @x }
let x = -1.0;
let y = sign(x);
let d = @y / @x;
"""
        _, out_neg = _compile_run(source_neg)
        assert abs(_scalar_float(out_neg, "d")) < 1e-6

    def test_quotient_math_ln(self):
        """d/dx ln(x)=1/x. At x=1 => 1."""
        source = """
fn ln(x) { python::numpy::log(x) }
@fn ln(x) { (1.0 / x) * @x }
let x = 1.0;
let y = ln(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d") - 1.0) < 1e-5

    def test_quotient_math_rsqrt(self):
        """rsqrt(x)=1/sqrt(x), d/dx = -1/(2*x^(3/2)). At x=4: rsqrt(4)=0.5, d/dx = -1/16."""
        source = """
fn rsqrt(x) { 1.0 / python::numpy::sqrt(x) }
@fn rsqrt(x) { (-0.5 / (python::numpy::sqrt(x) * x)) * @x }
let x = 4.0;
let y = rsqrt(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d") - (-1 / 16)) < 1e-5

    def test_quotient_math_min(self):
        """min(a,b)=if a<b {a} else {b}; piecewise d/da=1 if a<b else 0, d/db=0 if a<b else 1."""
        source = """
fn min_ab(a, b) { if a < b { a } else { b } }
@fn min_ab(a, b) { (if a < b { 1.0 } else { 0.0 }) * @a + (if a < b { 0.0 } else { 1.0 }) * @b }
let a1 = 1.0;
let b1 = 2.0;
let m1 = min_ab(a1, b1);
let dm1_da1 = @m1 / @a1;
let dm1_db1 = @m1 / @b1;
let a2 = 2.0;
let b2 = 1.0;
let m2 = min_ab(a2, b2);
let dm2_da2 = @m2 / @a2;
let dm2_db2 = @m2 / @b2;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "m1") - 1.0) < 1e-6
        assert abs(_scalar_float(out, "dm1_da1") - 1.0) < 1e-6
        assert abs(_scalar_float(out, "dm1_db1") - 0.0) < 1e-6
        assert abs(_scalar_float(out, "m2") - 1.0) < 1e-6
        assert abs(_scalar_float(out, "dm2_da2") - 0.0) < 1e-6
        assert abs(_scalar_float(out, "dm2_db2") - 1.0) < 1e-6

    def test_quotient_math_max(self):
        """max(a,b)=if a>b {a} else {b}; piecewise d/da=1 if a>b else 0, d/db=0 if a>b else 1."""
        source = """
fn max_ab(a, b) { if a > b { a } else { b } }
@fn max_ab(a, b) { (if a > b { 1.0 } else { 0.0 }) * @a + (if a > b { 0.0 } else { 1.0 }) * @b }
let a1 = 2.0;
let b1 = 1.0;
let m1 = max_ab(a1, b1);
let dm1_da1 = @m1 / @a1;
let dm1_db1 = @m1 / @b1;
let a2 = 1.0;
let b2 = 2.0;
let m2 = max_ab(a2, b2);
let dm2_da2 = @m2 / @a2;
let dm2_db2 = @m2 / @b2;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "m1") - 2.0) < 1e-6
        assert abs(_scalar_float(out, "dm1_da1") - 1.0) < 1e-6
        assert abs(_scalar_float(out, "dm1_db1") - 0.0) < 1e-6
        assert abs(_scalar_float(out, "m2") - 2.0) < 1e-6
        assert abs(_scalar_float(out, "dm2_da2") - 0.0) < 1e-6
        assert abs(_scalar_float(out, "dm2_db2") - 1.0) < 1e-6

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

    def test_quotient_mod(self):
        """remainder a % b: subgradient ∂/∂a = 1, ∂/∂b = 0. At a=7, b=3: 7%3=1, d(1)/da=1."""
        source = """
let a = 7.0;
let b = 3.0;
let r = a % b;
let dr_da = @r / @a;
let dr_db = @r / @b;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "dr_da") - 1.0) < 1e-6
        assert abs(_scalar_float(out, "dr_db") - 0.0) < 1e-6


class TestPyTorchStyleOps:
    """
    Autodiff tests for PyTorch-style ops (not just activations).

    We support and test the same op set PyTorch autograd typically supports:
    - Arithmetic: add, sub, mul, div, neg, pow, mod (remainder; subgradient)
    - Math unary: exp, log/ln, log10, log2, log1p, expm1, sqrt, rsqrt,
      sin, cos, tan, asin, acos, atan, atan2,
      sinh, cosh, tanh, asinh, acosh, atanh, erf,
      abs, sign, square
    - Activations: relu, sigmoid, softplus, leaky_relu, elu (via fn + @fn)
    - Clamp / min-max: min, max, clamp, clamp_min, clamp_max, saturate
    - Other: reciprocal (1/x), deg2rad, rad2deg

    All use local fn + @fn (or IR-level for add/sub/mul/div/pow/neg/mod).
    """

    def test_pytorch_style_relu(self):
        """relu(x)=max(0,x); d/dx = 1 if x>0 else 0. At x=1 => 1, at x=-1 => 0."""
        source = """
fn relu(x) { if (x as f32) > 0.0 { x } else { 0.0 } }
@fn relu(x) { (if (x as f32) > 0.0 { 1.0 } else { 0.0 }) * @x }
let x_pos = 1.0;
let x_neg = -1.0;
let y_pos = relu(x_pos);
let y_neg = relu(x_neg);
let d_pos = @y_pos / @x_pos;
let d_neg = @y_neg / @x_neg;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d_pos") - 1.0) < 1e-6
        assert abs(_scalar_float(out, "d_neg") - 0.0) < 1e-6

    def test_pytorch_style_sigmoid(self):
        """sigmoid(x)=1/(1+e^(-x)); d/dx = sigmoid(x)*(1-sigmoid(x)). At x=0 => 0.25."""
        source = """
fn sigmoid(x) { 1.0 / (1.0 + python::numpy::exp(0.0 - x)) }
@fn sigmoid(x) { (sigmoid(x) * (1.0 - sigmoid(x))) * @x }
let x = 0.0;
let y = sigmoid(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d") - 0.25) < 1e-5

    def test_pytorch_style_softplus(self):
        """softplus(x)=ln(1+e^x); d/dx = sigmoid(x). At x=0 => 0.5."""
        source = """
fn softplus(x) { python::numpy::log(1.0 + python::numpy::exp(x)) }
fn sigmoid(x) { 1.0 / (1.0 + python::numpy::exp(0.0 - x)) }
@fn softplus(x) { sigmoid(x) * @x }
let x = 0.0;
let y = softplus(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d") - 0.5) < 1e-5

    def test_pytorch_style_leaky_relu(self):
        """leaky_relu(x, alpha)=x if x>0 else alpha*x. d/dx = 1 if x>0 else alpha. alpha=0.01."""
        source = """
fn leaky_relu(x, alpha) { if (x as f32) > 0.0 { x } else { alpha * x } }
@fn leaky_relu(x, alpha) { (if (x as f32) > 0.0 { 1.0 } else { alpha }) * @x }
let x_pos = 1.0;
let x_neg = -1.0;
let alpha = 0.01;
let y_pos = leaky_relu(x_pos, alpha);
let y_neg = leaky_relu(x_neg, alpha);
let d_pos = @y_pos / @x_pos;
let d_neg = @y_neg / @x_neg;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d_pos") - 1.0) < 1e-6
        assert abs(_scalar_float(out, "d_neg") - 0.01) < 1e-6

    def test_pytorch_style_elu(self):
        """elu(x, alpha)=x if x>0 else alpha*(e^x-1). d/dx = 1 if x>0 else alpha*e^x. At x=0- => alpha."""
        source = """
fn elu(x, alpha) { if (x as f32) > 0.0 { x } else { alpha * (python::numpy::exp(x) - 1.0) } }
@fn elu(x, alpha) { (if (x as f32) > 0.0 { 1.0 } else { alpha * python::numpy::exp(x) }) * @x }
let x_pos = 1.0;
let x_neg = -1.0;
let alpha = 1.0;
let y_pos = elu(x_pos, alpha);
let y_neg = elu(x_neg, alpha);
let d_pos = @y_pos / @x_pos;
let d_neg = @y_neg / @x_neg;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d_pos") - 1.0) < 1e-6
        assert abs(_scalar_float(out, "d_neg") - (math.e ** (-1.0))) < 1e-5

    def test_pytorch_style_reciprocal(self):
        """reciprocal(x)=1/x; d/dx = -1/x^2. At x=2 => -0.25."""
        source = """
fn reciprocal(x) { 1.0 / x }
@fn reciprocal(x) { (-1.0 / (x * x)) * @x }
let x = 2.0;
let y = reciprocal(x);
let d = @y / @x;
"""
        _, out = _compile_run(source)
        assert abs(_scalar_float(out, "d") - (-0.25)) < 1e-6
