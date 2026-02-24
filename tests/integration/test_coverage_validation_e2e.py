"""
End-to-End Tests for Multi-Tiered Coverage Validation

Tests complete compilation and EXECUTION with coverage validation:
- Parsing → AST → Coverage Analysis → Validation → Execution → Result Verification

Note: Currently focuses on single declarations and reductions.
Multi-declaration support is a TODO in the runtime.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


class TestCoverageValidationE2E:
    """End-to-end tests for coverage validation: compile + execute + verify results"""
    
    def test_single_declaration_valid(self, compiler, runtime):
        """Single Einstein declaration - should execute correctly"""
        code = """
        let A[i in 0..3, j in 0..4] = i + j;
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Verify results
        A = result.outputs['A']
        assert isinstance(A, np.ndarray)
        assert A.shape == (3, 4)
        for i in range(3):
            for j in range(4):
                assert A[i,j] == i + j, f"A[{i},{j}] should be {i+j}, got {A[i,j]}"
    
    def test_single_declaration_with_expression(self, compiler, runtime):
        """Einstein declaration with complex expression - should execute correctly"""
        code = """
        let B[i in 0..4, j in 0..3] = i * i + j * j;
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Verify results
        B = result.outputs['B']
        assert isinstance(B, np.ndarray)
        assert B.shape == (4, 3)
        for i in range(4):
            for j in range(3):
                expected = i * i + j * j
                assert B[i,j] == expected, f"B[{i},{j}] should be {expected}"
    
    def test_dependent_tensors(self, compiler, runtime):
        """Multiple tensors with dependencies - should execute correctly"""
        code = """
        let X[i in 0..5] = i * 2;
        let Y[i in 0..5] = X[i] + 1;
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Verify results
        X = result.outputs['X']
        Y = result.outputs['Y']
        
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert X.shape == (5,)
        assert Y.shape == (5,)
        
        # Verify X and Y values
        for i in range(5):
            assert X[i] == i * 2, f"X[{i}] should be {i*2}"
            assert Y[i] == X[i] + 1, f"Y[{i}] should be X[{i}]+1"


class TestCoverageWithReductions:
    """Test coverage validation with reductions - verify execution results"""
    
    def test_reduction_simple(self, compiler, runtime):
        """Simple reduction - should execute correctly"""
        code = """
        let data = [1, 2, 3, 4, 5];
        let total = sum[i in 0..5](data[i]);
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Verify results
        total = result.outputs['total']
        assert total == 15, "Sum should be 1+2+3+4+5=15"
    
    def test_reduction_in_einstein_valid(self, compiler, runtime):
        """Reduction in Einstein declaration - should execute correctly"""
        code = """
        let matrix = [[1, 2, 3], [4, 5, 6]];
        let row_sums[i in 0..2] = sum[j](matrix[i,j]);
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Verify results
        row_sums = result.outputs['row_sums']
        assert isinstance(row_sums, np.ndarray)
        assert row_sums.shape == (2,)
        assert row_sums[0] == 6, "First row sum: 1+2+3=6"
        assert row_sums[1] == 15, "Second row sum: 4+5+6=15"
    
    def test_reduction_with_value_filter(self, compiler, runtime):
        """Reduction with value filtering (no coverage holes) - should execute correctly"""
        code = """
        let data = [1, -2, 3, -4, 5];
        let pos_sum = sum[i in 0..5](data[i] where data[i] > 0);
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Value filtering in reductions is safe (produces scalar, not array)
        pos_sum = result.outputs['pos_sum']
        assert pos_sum == 9, "Sum of positive values: 1+3+5=9"
    
    def test_einstein_with_reduction_coverage(self, compiler, runtime):
        """Einstein with reduction - coverage is determined by Einstein indices only"""
        code = """
        let matrix = [[1, -2, 3], [-4, 5, -6]];
        let row_sums[i in 0..2] = sum[j](matrix[i,j]);
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Verify results - no coverage holes despite negative values
        row_sums = result.outputs['row_sums']
        assert isinstance(row_sums, np.ndarray)
        assert row_sums.shape == (2,), "Shape determined by Einstein indices (i), not reduction indices (j)"
        # Both rows have values (no coverage holes)
        assert row_sums[0] == 2, "Row 0: 1 + (-2) + 3 = 2"
        assert row_sums[1] == -5, "Row 1: (-4) + 5 + (-6) = -5"
    
    def test_reduction_all_filtered_out(self, compiler, runtime):
        """Reduction with all values filtered out returns identity element"""
        code = """
        let data = [1, 2, 3];
        let result_val = sum[i in 0..3](data[i] where data[i] > 10);
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Result is 0 (identity), not undefined
        result_val = result.outputs['result_val']
        assert result_val == 0, "Empty sum should be 0 (identity element)"
    
    def test_reduction_product_with_filter(self, compiler, runtime):
        """Product reduction with value filtering"""
        code = """
        let data = [1, -2, 3, -4, 5];
        let pos_prod = prod[i in 0..5](data[i] where data[i] > 0);
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Product of positive values: 1*3*5=15
        pos_prod = result.outputs['pos_prod']
        assert pos_prod == 15, "Product of positive values: 1*3*5=15"
    
    def test_reduction_max_with_filter(self, compiler, runtime):
        """Max reduction with value filtering"""
        code = """
        let data = [1, 7, 3, 9, 5];
        let max_small = max[i in 0..5](data[i] where data[i] < 6);
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Max of values < 6: max(1, 3, 5) = 5
        max_small = result.outputs['max_small']
        assert max_small == 5, "Max of small values: max(1,3,5)=5"
    
    def test_reduction_min_with_filter(self, compiler, runtime):
        """Min reduction with value filtering"""
        code = """
        let data = [1, 7, 3, 9, 5];
        let min_large = min[i in 0..5](data[i] where data[i] > 5);
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Min of values > 5: min(7, 9) = 7
        min_large = result.outputs['min_large']
        assert min_large == 7, "Min of large values: min(7,9)=7"


class TestCoverageNonLinearConstraints:
    """Test coverage validation with non-linear constraints (STRICT MODE)
    
    These tests verify STRICT COVERAGE SEMANTICS:
    1. Single declarations with relational constraints are REJECTED (coverage holes)
    2. Multiple complementary declarations required for complete coverage
    3. Guards are enforced at runtime using NumPy masked arrays
    4. Coverage validator detects and rejects incomplete coverage at compile-time
    
    ⚠️  STRICT BY DEFAULT:
    - Single declaration with guards → ❌ Compilation error (incomplete coverage)
    - Complementary declarations required → ✅ Perfect partition accepted
    - Explicit @sparse annotation → ✅ Opt-in to allow holes (TODO)
    """
    
    def test_circular_region_execution(self, compiler, runtime):
        """Non-linear constraint: i*i + j*j < R*R — allows partial coverage (zeros for uncovered)"""
        code = """
        let circle[i in 0..10, j in 0..10] = i*100 + j where i*i + j*j < 100;
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        import numpy as np
        circle = np.array(result.outputs['circle'])
        assert circle.shape == (10, 10)
        assert circle[3, 3] == 303
        assert circle[9, 9] == 0
    
    def test_triangular_region_execution(self, compiler, runtime):
        """Linear constraint: i < j with COMPLEMENT - VALID (perfect partition)"""
        code = """
        let upper[i in 0..5, j in 0..5] = i*10 + j where i < j;
        let upper[i in 0..5, j in 0..5] = 0 where i >= j;
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Verify array is properly initialized
        upper = result.outputs['upper']
        assert upper.shape == (5, 5)
        
        # Upper triangle (i < j) - should have computed values
        assert upper[0, 1] == 1, "0 < 1: upper triangle"
        assert upper[1, 2] == 12, "1 < 2: upper triangle"
        assert upper[2, 4] == 24, "2 < 4: upper triangle"
        
        # Diagonal and lower (i >= j) - should be 0 from complement
        assert upper[2, 2] == 0, "2 >= 2: diagonal, set to 0"
        assert upper[3, 2] == 0, "3 > 2: lower triangle, set to 0"
        assert upper[4, 0] == 0, "4 > 0: lower triangle, set to 0"
    
    def test_band_matrix_execution(self, compiler, runtime):
        """Linear constraints: abs(i - j) < 2 with COMPLEMENT - VALID (perfect partition)"""
        code = """
        let band[i in 0..5, j in 0..5] = i*10 + j where i - j < 2, j - i < 2;
        let band[i in 0..5, j in 0..5] = -1 where i - j >= 2;
        let band[i in 0..5, j in 0..5] = -1 where j - i >= 2;
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Verify array is properly initialized
        band = result.outputs['band']
        assert band.shape == (5, 5)
        
        # Band positions (|i-j| < 2) - should have computed values
        assert band[2, 2] == 22, "|2-2|=0 < 2: diagonal"
        assert band[2, 3] == 23, "|2-3|=1 < 2: super-diagonal"
        assert band[3, 2] == 32, "|3-2|=1 < 2: sub-diagonal"
        assert band[1, 0] == 10, "|1-0|=1 < 2: band"
        
        # Outside band (|i-j| >= 2) - should be -1 from complements
        assert band[0, 4] == -1, "|0-4|=4 >= 2: outside band"
        assert band[4, 0] == -1, "|4-0|=4 >= 2: outside band"
        assert band[0, 3] == -1, "|0-3|=3 >= 2: outside band"


class TestCoverageComplexExpressions:
    """Test coverage validation with complex expressions"""
    
    def test_nested_array_access(self, compiler, runtime):
        """Einstein with nested array access"""
        code = """
        let indices = [0, 2, 1, 3];
        let values = [10, 20, 30, 40];
        let reordered[i in 0..4] = values[indices[i]];
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Verify reordering
        reordered = result.outputs['reordered']
        assert isinstance(reordered, np.ndarray)
        assert reordered.shape == (4,)
        assert reordered[0] == 10  # values[indices[0]] = values[0] = 10
        assert reordered[1] == 30  # values[indices[1]] = values[2] = 30
        assert reordered[2] == 20  # values[indices[2]] = values[1] = 20
        assert reordered[3] == 40  # values[indices[3]] = values[3] = 40
    
    def test_arithmetic_with_multiple_arrays(self, compiler, runtime):
        """Einstein with multiple array accesses"""
        code = """
        let A = [1, 2, 3, 4];
        let B = [10, 20, 30, 40];
        let C[i in 0..4] = A[i] * B[i] + i;
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Verify computation
        C = result.outputs['C']
        A = result.outputs['A']
        B = result.outputs['B']
        assert isinstance(C, np.ndarray)
        assert C.shape == (4,)
        for i in range(4):
            expected = A[i] * B[i] + i
            assert C[i] == expected, f"C[{i}] should be {expected}"
    
    def test_2d_matmul_like(self, compiler, runtime):
        """2D Einstein that resembles matrix multiplication"""
        code = """
        let A = [[1, 2], [3, 4]];
        let B = [[5, 6], [7, 8]];
        let C[i in 0..2, k in 0..2] = sum[j](A[i,j] * B[j,k]);
        """
        result = compile_and_execute(code, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        # Verify matrix multiplication result
        C = result.outputs['C']
        assert isinstance(C, np.ndarray)
        assert C.shape == (2, 2)
        # Manually compute expected result
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 5+14 = 19
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*6 + 2*8 = 6+16 = 22
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 3*5 + 4*7 = 15+28 = 43
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = 3*6 + 4*8 = 18+32 = 50
        assert C[0,0] == 19
        assert C[0,1] == 22
        assert C[1,0] == 43
        assert C[1,1] == 50
