"""
Unit tests for AutodiffPass.

Tests that the autodiff pass runs and expands derivative nodes (@expr, @num/@den)
into plain IR (d_* bindings and references). No diff block; derivatives are in-program.
All tests expect compile and run success; derivative tests assert correct values.
"""

from einlang.compiler.driver import CompilerDriver
from einlang.passes.autodiff import AutodiffPass


def _scalar_float(outputs, key):
    v = outputs.get(key)
    assert v is not None, "expected output %r, got %s" % (key, list(outputs.keys()))
    return float(v) if hasattr(v, "item") else float(v)


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
