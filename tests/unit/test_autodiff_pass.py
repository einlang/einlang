"""
Unit tests for AutodiffPass.

Tests that the autodiff pass runs and expands derivative nodes (@expr, @num/@den)
into plain IR (d_* bindings and references). No diff block; derivatives are in-program.
All tests expect compile and run success; derivative tests assert correct values.

Coverage: pipeline registration, no-@ programs, @expr expansion, quotient @num/@den,
scalar math (add/sub/mul/div/pow), unary neg, chain rule through lets, user functions,
custom @fn rules, Einstein ∂C/∂A, multiple quotients, constant derivative zero.
Math-like derivatives via user fns: sqrt (x**0.5), reciprocal (1/x), square (x*x).
Note: stdlib module calls (use std::math::sqrt etc.) are not yet supported for
autodiff (callee not in program.bindings); use local user fns or @fn for such math.
"""

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
