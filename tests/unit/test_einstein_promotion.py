"""
Tests for default index ordering in Einstein expressions.

Verifies that:
1. Explicit reductions can omit LHS indices (auto-derived order)
2. Implicit Einstein contractions (repeated index = sum) work
3. Free indices follow first-occurrence left-to-right ordering
4. Existing explicit LHS syntax continues to work
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


def _execute(source, compiler, runtime):
    result = compile_and_execute(source, compiler, runtime, source_file="<test>")
    assert result.success, f"Compilation failed: {result.get_errors()}"
    return result


class TestDefaultIndexOrder:
    """Tests for auto-derived LHS index order in Einstein expressions."""

    # ── Explicit reduction with omitted LHS ──

    def test_explicit_reduction_omitted_lhs_matmul(self, compiler, runtime):
        """let C = sum[k](A[i,k] * B[k,j]) should auto-derive [i, j]."""
        source = """
        let A[i in 0..2, k in 0..3] = i * 10 + k;
        let B[k in 0..3, j in 0..4] = k * 100 + j;
        let C = sum[k](A[i, k] * B[k, j]);
        C;
        """
        result = _execute(source, compiler, runtime)
        C = result.outputs['C']
        assert C.shape == (2, 4), f"expected shape (2, 4), got {C.shape}"

    def test_explicit_reduction_omitted_lhs_outer_product(self, compiler, runtime):
        """let C = sum[k](A[i,k] * B[j,k]) should auto-derive [i, j]."""
        source = """
        let A[i in 0..3, k in 0..2] = i * 10 + k;
        let B[j in 0..4, k in 0..2] = j * 100 + k;
        let C = sum[k](A[i, k] * B[j, k]);
        C;
        """
        result = _execute(source, compiler, runtime)
        C = result.outputs['C']
        assert C.shape == (3, 4), f"expected shape (3, 4), got {C.shape}"

    def test_explicit_reduction_omitted_lhs_vector_reduce(self, compiler, runtime):
        """let C = sum[j](A[i,j]) should auto-derive [i]."""
        source = """
        let A[i in 0..3, j in 0..5] = i * 10 + j;
        let C = sum[j](A[i, j]);
        C;
        """
        result = _execute(source, compiler, runtime)
        C = result.outputs['C']
        assert C.shape == (3,), f"expected shape (3,), got {C.shape}"

    def test_explicit_reduction_scalar_result_unchanged(self, compiler, runtime):
        """let x = sum[i](A[i]) should stay a var_decl (scalar output, no free indices)."""
        source = """
        let A[i in 0..5] = i;
        let x = sum[i](A[i]);
        x;
        """
        result = _execute(source, compiler, runtime)
        assert 'x' in result.outputs
        # Scalar result: sum of 0+1+2+3+4 = 10
        x_val = result.outputs['x']
        if hasattr(x_val, 'shape'):
            assert x_val.shape == (), f"expected scalar, got shape {x_val.shape}"

    def test_explicit_reduction_where_clause(self, compiler, runtime):
        """Reduction with where clause should still work as var_decl (no free indices)."""
        source = """
        let A[i in 0..10] = i;
        let sum_even = sum[i in 0..10](A[i]) where i % 2 == 0;
        sum_even;
        """
        result = _execute(source, compiler, runtime)
        # sum_even = 0+2+4+6+8 = 20
        assert 'sum_even' in result.outputs

    # ── Explicit contraction (default order, no implicit contraction) ──
    # Contraction must use explicit sum[...](...). Indices appearing once
    # are free; indices appearing ≥2 times ARE NOT auto-contracted.

    def test_element_wise_multiplication_free_indices(self, compiler, runtime):
        """let C = A[i,k] * B[k,j] → no implicit contraction, all [i,k,j] are free."""
        source = """
        let A[i in 0..2, k in 0..3] = i * 10 + k;
        let B[k in 0..3, j in 0..4] = k * 100 + j;
        let C = A[i, k] * B[k, j];
        C;
        """
        result = _execute(source, compiler, runtime)
        C = result.outputs['C']
        assert C.shape == (2, 3, 4), f"expected shape (2, 3, 4), got {C.shape}"

    def test_element_wise_multiplication_free_indices_2(self, compiler, runtime):
        """let C = A[i,j] * B[k,i] → all [i,j,k] are free, output [i, j, k]."""
        source = """
        let A[i in 0..2, j in 0..3] = i * 10 + j;
        let B[k in 0..4, i in 0..2] = k * 100 + i;
        let C = A[i, j] * B[k, i];
        C;
        """
        result = _execute(source, compiler, runtime)
        C = result.outputs['C']
        assert C.shape == (2, 3, 4), f"expected shape (2, 3, 4), got {C.shape}"

    # ── First-occurrence order ──

    def test_first_occurrence_order_left_to_right(self, compiler, runtime):
        """let C = sum[i](B[k,i] * A[i,j]) → first free: k (from B), then j (from A)."""
        source = """
        let A[i in 0..2, j in 0..3] = i * 10 + j;
        let B[k in 0..4, i in 0..2] = k * 100 + i;
        let C = sum[i](B[k, i] * A[i, j]);
        C;
        """
        result = _execute(source, compiler, runtime)
        C = result.outputs['C']
        assert C.shape == (4, 3), f"expected shape (4, 3), got {C.shape}"

    # ── Explicit LHS still works ──

    def test_explicit_lhs_still_works(self, compiler, runtime):
        """Existing explicit LHS syntax should be unaffected."""
        source = """
        let A[i in 0..2, k in 0..3] = i * 10 + k;
        let B[k in 0..3, j in 0..4] = k * 100 + j;
        let C[i, j] = sum[k](A[i, k] * B[k, j]);
        C;
        """
        result = _execute(source, compiler, runtime)
        C = result.outputs['C']
        assert C.shape == (2, 4), f"expected shape (2, 4), got {C.shape}"

    def test_explicit_lhs_overrides_default_order(self, compiler, runtime):
        """Explicit LHS [k, j] overrides default [j, k] for sum[i](A[i,j] * B[k,i])."""
        source = """
        let A[i in 0..2, j in 0..3] = i * 10 + j;
        let B[k in 0..4, i in 0..2] = k * 100 + i;
        let C[k, j] = sum[i](A[i, j] * B[k, i]);
        C;
        """
        result = _execute(source, compiler, runtime)
        C = result.outputs['C']
        assert C.shape == (4, 3), f"expected shape (4, 3), got {C.shape}"

    # ── Regular var_decl not affected ──

    def test_regular_array_access_not_promoted(self, compiler, runtime):
        """let x = arr[i] where i is a variable should NOT be promoted."""
        source = """
        let arr = [10, 20, 30];
        let i = 1;
        let x = arr[i];
        x;
        """
        result = _execute(source, compiler, runtime)
        assert 'x' in result.outputs
        x_val = result.outputs['x']
        assert x_val == 20, f"expected 20, got {x_val}"


    # ── Inline Einstein expressions ──

    def test_var_decl_element_wise_addition(self, compiler, runtime):
        """let C = B[i] + C[j] should auto-detect free indices [i, j]."""
        source = """
        let B[i in 0..2] = i * 10;
        let C[j in 0..3] = j * 100;
        let D = B[i] + C[j];
        D;
        """
        result = _execute(source, compiler, runtime)
        D = result.outputs['D']
        assert D.shape == (2, 3), f"expected shape (2, 3), got {D.shape}"

    def test_var_decl_element_wise_expression(self, compiler, runtime):
        """let C = (B[i] + C[j]) ** 2 should auto-detect free indices [i, j]."""
        source = """
        let B[i in 0..2] = i * 10;
        let C[j in 0..3] = j * 100;
        let D = (B[i] + C[j]) ** 2;
        D;
        """
        result = _execute(source, compiler, runtime)
        D = result.outputs['D']
        assert D.shape == (2, 3), f"expected shape (2, 3), got {D.shape}"


    # ── Inline nesting: coordinate function calls ──

    def test_coordinate_func_with_element_wise_body_promotes_free(self, compiler, runtime):
        """let D = id_axis[i](B[i] + C[j]) — coordinate i passes through (in return type), free [i, j] → (2, 3)."""
        source = """
        fn id_axis[j](x: [f32; ..left, j, ..right]) -> [f32; ..left, j, ..right] { x }
        let B[i in 0..2] = i as f32;
        let C[j in 0..3] = j as f32;
        let D = id_axis[i](B[i] + C[j]);
        D;
        """
        result = _execute(source, compiler, runtime)
        D = result.outputs['D']
        assert D.shape == (2, 3), f"expected shape (2, 3), got {D.shape}"

    def test_coordinate_func_with_nested_reduction_body(self, compiler, runtime):
        """let y = max[hidden](sum[k](W[i, k] * X[k, hidden])) — reduction nested in reduction body."""
        source = """
        let W[i in 0..4, k in 0..2] = i as f32 + k as f32;
        let X[k in 0..2, hidden in 0..3] = k as f32 * 10.0 + hidden as f32;
        let y = max[hidden](sum[k](W[i, k] * X[k, hidden]));
        y;
        """
        result = _execute(source, compiler, runtime)
        y = result.outputs['y']
        assert y.shape == (4,), f"expected shape (4,), got {y.shape}"

    def test_element_wise_between_coordinate_func_results(self, compiler, runtime):
        """let D = argmax[n](A[k, n]) + max[n](B[n, j]) — results used in element-wise add."""
        source = """
        let A[k in 0..2, n in 0..3] = (10 * k + n) as f32;
        let B[n in 0..3, j in 0..4] = (10 * n + j) as f32;
        let D = argmax[n](A[k, n]) as f32 + max[n](B[n, j]);
        D;
        """
        result = _execute(source, compiler, runtime)
        D = result.outputs['D']
        assert D.shape == (2, 4), f"expected shape (2, 4), got {D.shape}"

    def test_nested_reductions_with_free_index_promotion(self, compiler, runtime):
        """let y = sum[k](max[n](A[k, n])) + B[i] — k,n consumed, i free → [i]."""
        source = """
        let A[k in 0..2, n in 0..3] = (k * 10 + n) as f32;
        let B[i in 0..4] = i as f32;
        let y = sum[k](max[n](A[k, n])) + B[i];
        y;
        """
        result = _execute(source, compiler, runtime)
        y = result.outputs['y']
        assert y.shape == (4,), f"expected shape (4,), got {y.shape}"

    def test_triple_nested_reduction_with_where(self, compiler, runtime):
        """let z = sum[i](max[j](min[k](A[i, j, k]))) — all contracted → scalar."""
        source = """
        let A[i in 0..2, j in 0..3, k in 0..2] = (i * 100 + j * 10 + k) as f32;
        let z = sum[i](max[j](min[k](A[i, j, k])));
        z;
        """
        result = _execute(source, compiler, runtime)
        assert 'z' in result.outputs

    def test_chained_coordinate_calls_in_expression(self, compiler, runtime):
        """let D = argmax[n](A[k, n]) as f32 * max[n](B[n, j]) — free [k, j] from two results."""
        source = """
        let A[k in 0..2, n in 0..3] = (k * 10 + n) as f32;
        let B[n in 0..3, j in 0..4] = (n * 100 + j) as f32;
        let D = argmax[n](A[k, n]) as f32 * max[n](B[n, j]);
        D;
        """
        result = _execute(source, compiler, runtime)
        D = result.outputs['D']
        assert D.shape == (2, 4), f"expected shape (2, 4), got {D.shape}"


class TestDefaultIndexOrderValues:
    """Verify correctness of computed values with default index order."""

    def test_matmul_values(self, compiler, runtime):
        """Verify matmul values match expected result."""
        source = """
        let A = [[1, 2, 3], [4, 5, 6]];
        let B = [[7, 8], [9, 10], [11, 12]];
        let C = sum[k](A[i, k] * B[k, j]);
        C;
        """
        result = _execute(source, compiler, runtime)
        C = np.asarray(result.outputs['C'])
        # [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        # = [[58, 64], [139, 154]]
        expected = np.array([[58, 64], [139, 154]], dtype=C.dtype)
        np.testing.assert_allclose(C, expected, rtol=1e-5)

    def test_vector_dot_values(self, compiler, runtime):
        """Verify dot product values via explicit reduction."""
        source = """
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let c = sum[i in 0..3](a[i] * b[i]);
        c;
        """
        result = _execute(source, compiler, runtime)
        c_val = result.outputs['c']
        # 1*4 + 2*5 + 3*6 = 32
        assert c_val == 32, f"expected 32, got {c_val}"


class TestCoordinateNestingExample:
    """Executes the coordinate_nesting.ein demo and validates its outputs."""

    def test_nesting_demo_runs_and_produces_correct_shapes(self, compiler, runtime):
        from pathlib import Path
        example_path = Path(__file__).parent.parent.parent / "examples" / "demos" / "coordinate_nesting.ein"
        source = example_path.read_text(encoding="utf-8")
        # The example file has multiple sections each ending with an expression;
        # only the last one (`mixed;`) is the program result.
        result = compile_and_execute(source, compiler, runtime, source_file=str(example_path))
        assert result.success, f"Compilation failed: {result.get_errors()}"
        mixed = result.outputs.get('mixed')
        assert mixed is not None, "expected 'mixed' in outputs"
        assert mixed.shape == (4,), f"expected shape (4,), got {mixed.shape}"
