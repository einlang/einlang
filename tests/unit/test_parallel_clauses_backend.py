"""
Unit tests for backend clause output behavior.
"""

import numpy as np
import pytest

import einlang.backends.numpy_einstein_mixin_clause as numpy_einstein_mixin_clause
import einlang.backends.numpy_expressions_mixin as numpy_expressions_mixin
from einlang.backends.numpy import NumPyBackend
from einlang.shared.defid import DefId

from tests.test_utils import compile_and_execute
from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime


class TestClauseSetOutput:
    """Test _clause_set_output sets value in env."""

    def test_clause_set_output_sets_value(self):
        backend = NumPyBackend()
        fid = DefId(krate=0, index=1)
        backend.env.set_value(fid, None)
        backend._clause_set_output(fid, 42)
        assert backend.env.get_value(fid) == 42


class TestPureRecurrenceTAsOuterLoop:
    """
    Verify pure recurrence dim t is extracted as outer loop (timestep-major).
    Minimal inter-dependent recurrence: one clause is pure t (only loop t),
    the other depends on it at same t. Clause order would give wrong result.
    """

    SOURCE = """
let u[0, 0] = 0.0;
let u[0, 1] = 0.0;
let u[t in 1..11, 0] = u[t - 1, 1];
let u[t in 1..11, 1] = u[t, 0] + 1.0;
u;
"""

    def test_inter_dependent_recurrence_timestep_major(self):
        compiler = CompilerDriver()
        runtime = EinlangRuntime(backend="numpy")
        result = compile_and_execute(
            self.SOURCE.strip(),
            compiler,
            runtime,
            source_file="<pure_rec_t>",
        )
        assert result.success, (result.errors if result.errors else result.error)
        u = np.asarray(result.value if result.value is not None else result.outputs.get("u"))
        assert u is not None and u.ndim == 2 and u.shape[0] == 11 and u.shape[1] == 2

        # Reference: timestep-major order. u[t,0] = u[t-1,1], u[t,1] = u[t,0]+1 => u[t,0]=t-1, u[t,1]=t for t>=1.
        ref = np.zeros((11, 2), dtype=np.float64)
        ref[0, 0], ref[0, 1] = 0.0, 0.0
        for t in range(1, 11):
            ref[t, 0] = ref[t - 1, 1]
            ref[t, 1] = ref[t, 0] + 1.0

        np.testing.assert_allclose(u, ref, rtol=1e-5, atol=1e-5,
                                   err_msg="Pure recurrence t as outer loop (inter-dependent clauses)")


def test_scalar_sumprod_uses_matmul_fast_path_without_parallel_shape(monkeypatch):
    source = """
let A = [[1.0, 2.0], [3.0, 4.0]];
let B = [[5.0, 6.0], [7.0, 8.0]];
let y = sum[i in 0..2, j in 0..2](A[i, j] * B[i, j]);
"""

    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    hits = []
    original_try_matmul = numpy_expressions_mixin._try_matmul_reduction

    def _tracking_try_matmul(expr, backend, plan):
        result = original_try_matmul(expr, backend, plan)
        if result is not None:
            hits.append(
                {
                    "kind": getattr(plan, "kind", None),
                    "parallel_shape": backend._vectorization_parallel_shape(),
                }
            )
        return result

    monkeypatch.setattr(
        numpy_expressions_mixin,
        "_try_matmul_reduction",
        _tracking_try_matmul,
    )

    result = compile_and_execute(
        source.strip(),
        compiler,
        runtime,
        source_file="<scalar_sumprod_fast_path>",
    )

    assert result.success, (result.errors if result.errors else result.error)
    assert float(result.outputs["y"]) == pytest.approx(70.0)
    assert hits, "expected scalar sum-of-products reduction to use the matmul/tensordot fast path"
    assert hits[0]["kind"] == "matmul_sumprod"
    assert hits[0]["parallel_shape"] is None


def test_matmul_fast_path_handles_extra_bias_row_outside_reduction(monkeypatch):
    source = """
let x[n in 0..2, k in 0..4] = (1 + n + k) as f32;
let theta[k in 0..5, j in 0..3] =
    if k == 4 { 0.5 * (1 + j) as f32 } else { 0.1 * (1 + k + j) as f32 };
let logits[n in 0..2, j in 0..3] =
    sum[k in 0..4](x[n, k] * theta[k, j]) + theta[4, j];
"""

    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    hits = []
    original_try_matmul = numpy_expressions_mixin._try_matmul_reduction

    def _tracking_try_matmul(expr, backend, plan):
        result = original_try_matmul(expr, backend, plan)
        if result is not None:
            hits.append(
                {
                    "kind": getattr(plan, "kind", None),
                    "shape": tuple(np.asarray(result).shape),
                }
            )
        return result

    monkeypatch.setattr(
        numpy_expressions_mixin,
        "_try_matmul_reduction",
        _tracking_try_matmul,
    )

    result = compile_and_execute(
        source.strip(),
        compiler,
        runtime,
        source_file="<matmul_bias_row_fast_path>",
    )

    assert result.success, (result.errors if result.errors else result.error)
    np.testing.assert_allclose(
        result.outputs["logits"],
        np.array([[3.5, 5.0, 6.5], [4.5, 6.4, 8.3]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    counts = runtime.get_last_vectorize_counts()
    assert int(counts.get("scalar", 0)) == 0
    assert hits, "expected logits reduction with an extra bias row to use the matmul fast path"
    assert hits[0]["kind"] == "matmul_sumprod"
    assert hits[0]["shape"] == (2, 3)


def test_runtime_uses_compiler_owned_clause_vectorization_plan(monkeypatch):
    source = """
fn row(x, i) { x[i] }
let X = [[1.0, 2.0], [3.0, 4.0]];
let Y[i in 0..2, j in 0..2] = row(X, i)[j];
Y;
"""

    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")

    def _fail(*_args, **_kwargs):
        raise AssertionError("runtime should use compiler-owned clause vectorization plan")

    monkeypatch.setattr(numpy_einstein_mixin_clause, "_body_contains_call_using_loop_var", _fail)
    monkeypatch.setattr(numpy_einstein_mixin_clause, "_loop_defids_in_call_args", _fail)

    result = compile_and_execute(
        source.strip(),
        compiler,
        runtime,
        source_file="<compiler_owned_clause_vectorization>",
    )

    assert result.success, (result.errors if result.errors else result.error)
    np.testing.assert_allclose(
        result.outputs["Y"],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_runtime_uses_compiler_owned_scalar_clause_plan(monkeypatch):
    source = """
fn inc(v) { v }
let x = [1, 2, 3];
let z[i in 0..3] = {
    let t = sum[j in 0..3](x[j]);
    inc(t + x[i])
};
z;
"""

    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")

    def _fail(*_args, **_kwargs):
        raise AssertionError("runtime should use compiler-owned scalar clause plan")

    monkeypatch.setattr(numpy_einstein_mixin_clause, "_expr_contains_nested_lowered_ir", _fail)
    monkeypatch.setattr(numpy_einstein_mixin_clause, "_block_has_direct_nested_lowered_binding", _fail)

    result = compile_and_execute(
        source.strip(),
        compiler,
        runtime,
        source_file="<compiler_owned_scalar_clause_vectorization>",
    )

    assert result.success, (result.errors if result.errors else result.error)
    np.testing.assert_allclose(
        result.outputs["z"],
        np.array([7, 8, 9], dtype=np.int32),
    )
