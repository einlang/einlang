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


class TestQuantifierReductions:
    """Test quantifier reductions (all[i], any[i])"""
    
    def test_all_positive_simple(self, compiler, runtime):
        """Test all[i] with all positive values"""
        source = """
        let x = [1, 2, 3, 4, 5];
        let all_positive = all[i](x[i] > 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'all_positive' in result.outputs
        assert result.outputs['all_positive'] == True
    
    def test_all_positive_with_negative(self, compiler, runtime):
        """Test all[i] with some negative values"""
        source = """
        let x = [1, 2, -3, 4, 5];
        let all_positive = all[i](x[i] > 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'all_positive' in result.outputs
        assert result.outputs['all_positive'] == False
    
    def test_any_positive_simple(self, compiler, runtime):
        """Test any[i] with at least one positive value"""
        source = """
        let x = [-1, -2, 3, -4, -5];
        let has_positive = any[i](x[i] > 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'has_positive' in result.outputs
        assert result.outputs['has_positive'] == True
    
    def test_any_positive_all_negative(self, compiler, runtime):
        """Test any[i] with all negative values"""
        source = """
        let x = [-1, -2, -3, -4, -5];
        let has_positive = any[i](x[i] > 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'has_positive' in result.outputs
        assert result.outputs['has_positive'] == False
    
    def test_all_with_range(self, compiler, runtime):
        """Test all[i] with explicit range"""
        source = """
        let x = [1, 2, 3, 4, 5];
        let all_positive = all[i in 0..5](x[i] > 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'all_positive' in result.outputs
        assert result.outputs['all_positive'] == True
    
    def test_any_with_range(self, compiler, runtime):
        """Test any[i] with explicit range"""
        source = """
        let x = [-1, -2, 3, -4, -5];
        let has_positive = any[i in 0..5](x[i] > 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'has_positive' in result.outputs
        assert result.outputs['has_positive'] == True
    
    def test_all_2d_matrix(self, compiler, runtime):
        """Test all[i,j] on 2D matrix"""
        source = """
        let matrix = [[1, 2, 3], [4, 5, 6]];
        let all_positive = all[i, j](matrix[i, j] > 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'all_positive' in result.outputs
        assert result.outputs['all_positive'] == True
    
    def test_any_2d_matrix(self, compiler, runtime):
        """Test any[i,j] on 2D matrix"""
        source = """
        let matrix = [[-1, -2, -3], [4, -5, -6]];
        let has_positive = any[i, j](matrix[i, j] > 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'has_positive' in result.outputs
        assert result.outputs['has_positive'] == True
    
    def test_all_with_complex_expression(self, compiler, runtime):
        """Test all[i] with complex boolean expression"""
        source = """
        let x = [2, 4, 6, 8, 10];
        let all_even_and_positive = all[i](x[i] > 0 && x[i] % 2 == 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'all_even_and_positive' in result.outputs
        assert result.outputs['all_even_and_positive'] == True
    
    def test_any_with_complex_expression(self, compiler, runtime):
        """Test any[i] with complex boolean expression"""
        source = """
        let x = [1, 3, 5, 6, 7];
        let has_even = any[i](x[i] % 2 == 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'has_even' in result.outputs
        assert result.outputs['has_even'] == True
    
    def test_all_in_einstein_declaration(self, compiler, runtime):
        """Test all[i] used within variable declaration"""
        # all[j](...) returns a scalar boolean, not an array
        # So we don't need an index variable i
        source = """
        let x = [1, 2, 3, 4, 5];
        let y = [6, 7, 8, 9, 10];
        let all_sum_positive = all[j in 0..5](x[j] + y[j] > 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'all_sum_positive' in result.outputs
        # All sums should be positive, so result should be True (scalar boolean)
        assert result.outputs['all_sum_positive'] == True
    
    def test_mathematical_alignment_example(self, compiler, runtime):
        """Test example from MATHEMATICAL_OPERATORS_ALIGNMENT.md"""
        # Mathematical: ∀i: data_i > 0 and ∃i: data_i > 0
        source = """
        let data = [1, 2, 3, 4, 5];
        let all_positive = all[i](data[i] > 0);
        let has_positive = any[i](data[i] > 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['all_positive'] == True
        assert result.outputs['has_positive'] == True


class TestPracticalMathematicalFormulas:
    """Test quantifiers in practical mathematical formulas"""
    
    def test_matrix_positive_definite_check(self, compiler, runtime):
        """
        Mathematical: Check if matrix is positive definite
        Formula: ∀i,j: A[i,j] = A[j,i] (symmetric) AND ∀i: A[i,i] > 0 (diagonal positive)
        """
        source = """
        let A = [[2, 1], [1, 2]];  // Symmetric positive definite
        let is_symmetric = all[i, j](A[i, j] == A[j, i]);
        let diagonal_positive = all[i](A[i, i] > 0);
        let is_positive_definite = is_symmetric && diagonal_positive;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['is_symmetric'] == True
        assert result.outputs['diagonal_positive'] == True
        assert result.outputs['is_positive_definite'] == True
    
    def test_vector_norm_validation(self, compiler, runtime):
        """
        Mathematical: Check if vector satisfies norm constraint
        Formula: ∀i: |x[i]| ≤ 1 (all elements bounded)
        """
        source = """
        let x = [0.5, -0.3, 0.8, 0.1];
        let bounded = all[i](x[i] >= -1.0 && x[i] <= 1.0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['bounded'] == True
    
    def test_sparse_matrix_check(self, compiler, runtime):
        """
        Mathematical: Check if matrix is sparse (most elements zero)
        Formula: ∃i,j: A[i,j] ≠ 0 (has at least one non-zero)
        """
        source = """
        let A = [[0, 0, 0], [0, 5, 0], [0, 0, 0]];  // Sparse matrix
        let has_nonzero = any[i, j](A[i, j] != 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['has_nonzero'] == True
    
    def test_monotonic_sequence_check(self, compiler, runtime):
        """
        Mathematical: Check if sequence is monotonic increasing
        Formula: ∀i: x[i] ≤ x[i+1] (for all adjacent pairs)
        """
        source = """
        let x = [1, 2, 3, 4, 5];
        let is_monotonic = all[i in 0..4](x[i] <= x[i + 1]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['is_monotonic'] == True
    
    def test_orthogonal_matrix_check(self, compiler, runtime):
        """
        Mathematical: Check if matrix columns are orthogonal
        Formula: ∀i,j where i≠j: dot(col_i, col_j) = 0
        Simplified: Check that off-diagonal dot products are zero
        """
        source = """
        let A = [[1, 0], [0, 1]];  // Identity matrix (orthogonal)
        // For identity matrix, check that A[0,0]*A[0,1] + A[1,0]*A[1,1] = 0
        let dot_product_01 = sum[k](A[k, 0] * A[k, 1]);
        let is_orthogonal = dot_product_01 == 0;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['is_orthogonal'] == True
    
    def test_feasibility_check_optimization(self, compiler, runtime):
        """
        Mathematical: Check if solution satisfies all constraints
        Formula: ∀i: constraint_i(x) ≥ 0 (all constraints satisfied)
        """
        source = """
        let x = [1.0, 2.0, 3.0];
        let constraints = [1.0, 2.0, 3.0];
        let feasible = all[i](x[i] - constraints[i] >= 0.0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['feasible'] == True
    
    def test_strictly_positive_matrix(self, compiler, runtime):
        """
        Mathematical: Check if all matrix elements are strictly positive
        Formula: ∀i,j: A[i,j] > 0
        """
        source = """
        let A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let strictly_positive = all[i, j](A[i, j] > 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['strictly_positive'] == True
    
    def test_has_zero_element(self, compiler, runtime):
        """
        Mathematical: Check if matrix has at least one zero element
        Formula: ∃i,j: A[i,j] = 0
        """
        source = """
        let A = [[1, 2, 0], [4, 5, 6], [7, 8, 9]];
        let has_zero = any[i, j](A[i, j] == 0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['has_zero'] == True
    
    def test_probability_distribution_check(self, compiler, runtime):
        """
        Mathematical: Check if vector is valid probability distribution
        Formula: ∀i: p[i] ≥ 0 AND sum[i](p[i]) = 1
        """
        source = """
        let p = [0.2, 0.3, 0.5];
        let non_negative = all[i](p[i] >= 0.0);
        let total = sum[i](p[i]);
        let sums_to_one = total >= 0.999 && total <= 1.001;  // Floating point tolerance
        let is_probability = non_negative && sums_to_one;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['non_negative'] == True
        assert result.outputs['sums_to_one'] == True
        assert result.outputs['is_probability'] == True
    
    def test_triangle_inequality_check(self, compiler, runtime):
        """
        Mathematical: Check triangle inequality for all pairs
        Formula: ∀i,j,k: d[i,j] ≤ d[i,k] + d[k,j]
        """
        source = """
        let d = [[0, 1, 2], [1, 0, 1], [2, 1, 0]];  // Distance matrix
        let triangle_inequality = all[i, j, k](d[i, j] <= d[i, k] + d[k, j]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        # This should be True for a valid distance matrix
        assert result.outputs['triangle_inequality'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

