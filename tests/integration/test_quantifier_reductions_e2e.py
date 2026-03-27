"""
End-to-End Integration Tests for Quantifier Reductions (all[i], any[i])
========================================================================

Tests runtime execution of quantifier reductions matching mathematical notation:
- all[i](expr) → ∀i: expr (universal quantifier)
- any[i](expr) → ∃i: expr (existential quantifier)
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


def _assert_outputs(result, expected):
    assert result.success, f"Execution failed: {result.errors}"
    for name, value in expected.items():
        assert name in result.outputs, f"Missing output {name!r}"
        assert result.outputs[name] == value


class TestQuantifierReductions:
    """Test quantifier reductions (all[i], any[i])"""

    def test_basic_quantifier_truth_table(self, compiler, runtime):
        """Cover simple 1D, ranged, matrix, and complex-expression quantifier cases in one execution."""
        source = """
        let pos = [1, 2, 3, 4, 5];
        let mixed = [1, 2, -3, 4, 5];
        let mostly_negative = [-1, -2, 3, -4, -5];
        let all_negative = [-1, -2, -3, -4, -5];
        let even_values = [2, 4, 6, 8, 10];
        let odd_with_one_even = [1, 3, 5, 6, 7];
        let matrix_all_pos = [[1, 2, 3], [4, 5, 6]];
        let matrix_has_pos = [[-1, -2, -3], [4, -5, -6]];
        let data = [1, 2, 3, 4, 5];
        let other = [6, 7, 8, 9, 10];

        let all_positive = all[i](pos[i] > 0);
        let all_positive_with_negative = all[i](mixed[i] > 0);
        let has_positive = any[i](mostly_negative[i] > 0);
        let has_positive_all_negative = any[i](all_negative[i] > 0);
        let all_positive_ranged = all[i in 0..5](pos[i] > 0);
        let has_positive_ranged = any[i in 0..5](mostly_negative[i] > 0);
        let all_positive_matrix = all[i, j](matrix_all_pos[i, j] > 0);
        let has_positive_matrix = any[i, j](matrix_has_pos[i, j] > 0);
        let all_even_and_positive = all[i](even_values[i] > 0 && even_values[i] % 2 == 0);
        let has_even = any[i](odd_with_one_even[i] % 2 == 0);
        let all_sum_positive = all[j in 0..5](data[j] + other[j] > 0);
        let aligned_all_positive = all[i](data[i] > 0);
        let aligned_has_positive = any[i](data[i] > 0);
        """
        result = compile_and_execute(source, compiler, runtime)
        _assert_outputs(
            result,
            {
                "all_positive": True,
                "all_positive_with_negative": False,
                "has_positive": True,
                "has_positive_all_negative": False,
                "all_positive_ranged": True,
                "has_positive_ranged": True,
                "all_positive_matrix": True,
                "has_positive_matrix": True,
                "all_even_and_positive": True,
                "has_even": True,
                "all_sum_positive": True,
                "aligned_all_positive": True,
                "aligned_has_positive": True,
            },
        )


class TestPracticalMathematicalFormulas:
    """Test quantifiers in practical mathematical formulas"""

    def test_quantifiers_in_practical_formulas(self, compiler, runtime):
        """Cover practical mathematical-formula examples in one compile/execute pass."""
        source = """
        let A = [[1, 0], [0, 1]];  // Identity matrix (orthogonal)
        let SPD = [[2, 1], [1, 2]];
        let bounded_vec = [0.5, -0.3, 0.8, 0.1];
        let sparse = [[0, 0, 0], [0, 5, 0], [0, 0, 0]];
        let monotonic = [1, 2, 3, 4, 5];
        let constraints_x = [1.0, 2.0, 3.0];
        let constraints = [1.0, 2.0, 3.0];
        let strictly_pos = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let zero_matrix = [[1, 2, 0], [4, 5, 6], [7, 8, 9]];
        let p = [0.2, 0.3, 0.5];
        let d = [[0, 1, 2], [1, 0, 1], [2, 1, 0]];

        let is_symmetric = all[i, j](SPD[i, j] == SPD[j, i]);
        let diagonal_positive = all[i](SPD[i, i] > 0);
        let is_positive_definite = is_symmetric && diagonal_positive;
        let bounded = all[i](bounded_vec[i] >= -1.0 && bounded_vec[i] <= 1.0);
        let has_nonzero = any[i, j](sparse[i, j] != 0);
        let is_monotonic = all[i in 0..4](monotonic[i] <= monotonic[i + 1]);
        let dot_product_01 = sum[k](A[k, 0] * A[k, 1]);
        let is_orthogonal = dot_product_01 == 0;
        let feasible = all[i](constraints_x[i] - constraints[i] >= 0.0);
        let strictly_positive = all[i, j](strictly_pos[i, j] > 0);
        let has_zero = any[i, j](zero_matrix[i, j] == 0);
        let non_negative = all[i](p[i] >= 0.0);
        let total = sum[i](p[i]);
        let sums_to_one = total >= 0.999 && total <= 1.001;  // Floating point tolerance
        let is_probability = non_negative && sums_to_one;
        let triangle_inequality = all[i, j, k](d[i, j] <= d[i, k] + d[k, j]);
        """
        result = compile_and_execute(source, compiler, runtime)
        _assert_outputs(
            result,
            {
                "is_symmetric": True,
                "diagonal_positive": True,
                "is_positive_definite": True,
                "bounded": True,
                "has_nonzero": True,
                "is_monotonic": True,
                "is_orthogonal": True,
                "feasible": True,
                "strictly_positive": True,
                "has_zero": True,
                "non_negative": True,
                "sums_to_one": True,
                "is_probability": True,
                "triangle_inequality": True,
            },
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
