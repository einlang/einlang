"""
Integration Tests: Precision and Shape Propagation
====================================================

Demonstrates precision inference and propagation end-to-end.
TypeAnalysisPass + CoverageAnalysisPass + runtime execution.

Single shared compile_and_run; test code in chunks of 10-20 lines.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


def compile_and_run(code: str, compiler, runtime):
    """Compile and run Einlang code. Returns object with .outputs dict; also supports out['key']."""
    result = compile_and_execute(code, compiler, runtime)
    if not result.success:
        pytest.fail(f"Execution failed: {result.errors}")

    class Outputs:
        __slots__ = ("outputs",)
        def __init__(self, outputs):
            self.outputs = outputs
        def __getitem__(self, key):
            return self.outputs[key]
    return Outputs(result.outputs)


class TestPrecisionPropagationSimple:
    """Simple precision propagation"""

    def test_scalars_arrays_and_arithmetic(self, compiler, runtime):
        code = """
        let x = 42 as i64;
        let y = 3.14 as f32;
        let arr = [1, 2, 3];
        let a = 10 as i32;
        let b = 20 as i32;
        let c = a + b;
        """
        out = compile_and_run(code.strip(), compiler, runtime)
        assert out["x"] == 42 and isinstance(out["x"], (int, np.integer))
        assert abs(out["y"] - 3.14) < 0.01 and isinstance(out["y"], (float, np.floating))
        assert isinstance(out["arr"], np.ndarray) and list(out["arr"]) == [1, 2, 3]
        assert out["c"] == 30

    def test_einstein_tensors_and_2d(self, compiler, runtime):
        code = """
        let A[0] = 10;
        let A[1] = 20;
        let A[2] = 30;
        let B[0] = 100 as i64;
        let B[1] = 200 as i64;
        let X[0] = 5;
        let X[1] = 10;
        let Y[0] = 2;
        let Y[1] = 3;
        let sum0 = X[0] + Y[0];
        let sum1 = X[1] + Y[1];
        let M[0,0] = 1;
        let M[0,1] = 2;
        let M[1,0] = 3;
        let M[1,1] = 4;
        """
        out = compile_and_run(code.strip(), compiler, runtime)
        assert out["A"].shape == (3,) and list(out["A"]) == [10, 20, 30]
        assert out["B"].dtype == np.int64 and list(out["B"]) == [100, 200]
        assert out["sum0"] == 7 and out["sum1"] == 13
        assert out["M"].shape == (2, 2) and out["M"][1, 1] == 4

    def test_mixed_precision_and_fibonacci_base(self, compiler, runtime):
        code = """
        let P[0] = 10 as i32;
        let P[1] = 20 as i64;
        let Q[0] = 1.5 as f32;
        let Q[1] = 2.5 as f64;
        let fib[0] = 0 as i64;
        let fib[1] = 1 as i64;
        """
        out = compile_and_run(code.strip(), compiler, runtime)
        assert out["P"].dtype == np.int64 and list(out["P"]) == [10, 20]
        assert out["Q"].dtype == np.float64 and np.allclose(out["Q"], [1.5, 2.5])
        assert out["fib"].dtype == np.int64 and list(out["fib"]) == [0, 1]


class TestPrecisionInferenceWorkflow:
    def test_precision_pass_and_runtime_read(self, compiler, runtime):
        code = """
        let x = 42;
        let tensor[i in 0..1] = 100 as i64;
        """
        out = compile_and_run(code.strip(), compiler, runtime)
        assert out["x"] == 42
        t = out["tensor"]
        assert t.dtype in (np.int32, np.int64) and t[0] == 100


class TestPrecisionShapepropagation:
    """Precision and shape propagation across multiple tensors"""

    def test_three_tensor_propagation_and_widening(self, compiler, runtime):
        code = """
        let A[0] = 10 as i64;
        let A[1] = 20 as i64;
        let A[2] = 30 as i64;
        let B[0] = A[0] + 5;
        let B[1] = A[1] + 5;
        let B[2] = A[2] + 5;
        let C[i in 0..3] = A[i] * B[i];
        let X[0] = 1.5 as f32;
        let X[1] = 2.5 as f32;
        let Y[0] = 3.5 as f64;
        let Y[1] = 4.5 as f64;
        let Z[i in 0..2] = X[i] + Y[i];
        """
        out = compile_and_run(code.strip(), compiler, runtime)
        A, B, C = out["A"], out["B"], out["C"]
        assert A.dtype == np.int64 and list(A) == [10, 20, 30]
        assert B.dtype == np.int64 and list(B) == [15, 25, 35]
        assert C.dtype == np.int64 and list(C) == [150, 500, 1050]
        assert out["Z"].dtype == np.float64 and np.allclose(out["Z"], [5.0, 7.0])

    def test_integer_widening_and_multidimensional(self, compiler, runtime):
        code = """
        let M[0] = 100 as i32;
        let M[1] = 200 as i32;
        let M[2] = 300 as i32;
        let N[0] = 1000 as i64;
        let N[1] = 2000 as i64;
        let N[2] = 3000 as i64;
        let P[i in 0..3] = M[i] + N[i];
        let A[0,0] = 1 as i32;
        let A[0,1] = 2 as i32;
        let A[0,2] = 3 as i32;
        let A[1,0] = 4 as i32;
        let A[1,1] = 5 as i32;
        let A[1,2] = 6 as i32;
        let B[i in 0..2, j in 0..3] = A[i,j] * 2;
        let C[i in 0..2, j in 0..3] = A[i,j] + B[i,j];
        """
        out = compile_and_run(code.strip(), compiler, runtime)
        assert out["M"].dtype == np.int32 and out["N"].dtype == np.int64 and out["P"].dtype == np.int64
        assert list(out["P"]) == [1100, 2200, 3300]
        A, B, C = out["A"], out["B"], out["C"]
        assert A.shape == B.shape == C.shape == (2, 3)
        assert np.array_equal(C, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32) * 3)

    def test_recurrence_mixed_promotion_and_defaults(self, compiler, runtime):
        code = """
        let fib[0] = 0 as i64;
        let fib[1] = 1 as i64;
        let fib[i in 2..10] = fib[i-1] + fib[i-2];
        let Q[0] = 10 as i32;
        let Q[1] = 20 as i32;
        let Q[2] = 30 as i64;
        let Q[3] = 40 as i64;
        let D[0] = 42;
        let D[1] = 84;
        let E[i in 0..2] = D[i] * 2;
        let F[0] = 3.14;
        let F[1] = 6.28;
        let G[i in 0..2] = F[i] * 2.0;
        """
        out = compile_and_run(code.strip(), compiler, runtime)
        fib = out["fib"]
        assert fib.dtype == np.int64 and fib.shape == (10,) and list(fib) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        assert out["Q"].dtype == np.int64 and list(out["Q"]) == [10, 20, 30, 40]
        assert out["D"].dtype == np.int32 and out["E"].dtype == np.int32 and list(out["E"]) == [84, 168]
        assert out["F"].dtype == np.float32 and out["G"].dtype == np.float32


class TestPrecisionShapePropagationWithPasses:
    def test_passes_preserve_precision_and_shape(self, compiler, runtime):
        code = """
        let A[0] = 10 as i64;
        let A[1] = 20 as i64;
        let B[i in 0..2] = A[i] * 2;
        let M[0,0] = 1 as f32;
        let M[0,1] = 2 as f32;
        let M[1,0] = 3 as f32;
        let M[1,1] = 4 as f32;
        let N[i in 0..2, j in 0..2] = M[i,j] * 2.0;
        """
        out = compile_and_run(code.strip(), compiler, runtime)
        assert out["A"].dtype == out["B"].dtype == np.int64 and list(out["B"]) == [20, 40]
        assert out["M"].dtype == out["N"].dtype == np.float32
        assert out["M"].shape == out["N"].shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
