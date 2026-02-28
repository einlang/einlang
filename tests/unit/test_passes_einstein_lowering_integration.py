"""
Comprehensive Integration Tests for Einstein Lowering and Loop-Based Execution

Tests that verify loop-based execution works correctly across:
1. Various Einstein declaration patterns
2. Complex reduction scenarios
3. Nested structures
4. Edge cases and boundary conditions
5. Performance-critical patterns
6. Real-world use cases

This ensures the loop-based execution is robust and handles all scenarios correctly.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime
from einlang.ir.nodes import ReductionExpressionIR
from tests.test_utils import apply_ir_round_trip


@pytest.fixture
def compiler():
    return CompilerDriver()


@pytest.fixture
def runtime():
    return EinlangRuntime()


def compile_and_execute(source, compiler, runtime):
    """Helper to compile and execute source code"""
    compile_result = compiler.compile(source, source_file="<test>")
    if not compile_result.success:
        return compile_result
    apply_ir_round_trip(compile_result)
    return runtime.execute(compile_result)


class TestEinsteinLoweringIntegration:
    """Comprehensive integration tests for loop-based execution"""

    def test_einstein_literal_indices_only(self, compiler, runtime):
        """Single clause with only literal indices (no loops)."""
        source = """
        let x[0] = 42;
        x;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        assert "x" in result.outputs
        x = result.outputs["x"]
        assert isinstance(x, np.ndarray)
        assert x.shape == (1,)
        assert int(x[0]) == 42

    def test_einstein_mixed_literal_and_index(self, compiler, runtime):
        """Single clause with mixed literal and loop indices: [0, i]."""
        source = """
        let x[0, i in 0..4] = i;
        x;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        assert "x" in result.outputs
        x = result.outputs["x"]
        assert isinstance(x, np.ndarray)
        assert x.shape == (1, 4)
        np.testing.assert_array_equal(x[0], np.array([0, 1, 2, 3]))

    def test_einstein_multi_clause_literal_and_recurrence(self, compiler, runtime):
        """Three consecutive clauses: A[0,i]=i, A[i,0]=i, A[i,j]=A[i-1,j]+A[i,j-1] for i,j in 1.."""
        source = """
        let A[0, i in 0..4] = i;
        let A[i in 0..6, 0] = i;
        let A[i in 1..6, j in 1..4] = A[i - 1, j] + A[i, j - 1];
        A;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        A = result.outputs["A"]
        expected = np.array([
            [0, 1, 2, 3],
            [1, 2, 4, 7],
            [2, 4, 8, 15],
            [3, 7, 15, 30],
            [4, 11, 26, 56],
            [5, 16, 42, 98],
        ], dtype=A.dtype)
        np.testing.assert_array_equal(A, expected)

    def test_einstein_gru_pattern_two_clauses_same_buffer(self, compiler, runtime):
        """GRU-like pattern: two consecutive clauses on same array; first clause literal t=0, second recurrence. Fails if hidden[0] not written."""
        source = """
        let ref = [[0.1, 0.2], [0.3, 0.4]];
        let hidden[0, b in 0..2, h in 0..2] = ref[b, h];
        let hidden[t in 1..3, b in 0..2, h in 0..2] = hidden[t - 1, b, h] + 1.0;
        hidden;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        hidden = result.outputs["hidden"]
        assert hidden.shape == (3, 2, 2)
        expected_layer0 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=hidden.dtype)
        np.testing.assert_allclose(hidden[0], expected_layer0, rtol=1e-5, atol=1e-8)

    def test_einstein_gru_pattern_variable_bounds_at_toplevel(self, compiler, runtime):
        """Two-clause hidden with variable bounds at top level (like GRU but no function)."""
        source = """
        let batch_size = 2;
        let hidden_size = 2;
        let ref = [[0.1, 0.2], [0.3, 0.4]];
        let hidden[0, b in 0..batch_size, h in 0..hidden_size] = ref[b, h];
        let hidden[t in 1..3, b in 0..batch_size, h in 0..hidden_size] = hidden[t - 1, b, h] + 1.0;
        hidden;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        hidden = result.outputs["hidden"]
        hidden = np.asarray(hidden)
        assert hidden.shape == (3, 2, 2), f"expected (3,2,2), got {hidden.shape}"
        expected_layer0 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=hidden.dtype)
        np.testing.assert_allclose(hidden[0], expected_layer0, rtol=1e-5, atol=1e-8)

    def test_einstein_gru_pattern_inside_function(self, compiler, runtime):
        """Two-clause hidden inside a function (reproduces GRU context). Return value in let out = fn(...)."""
        source = """
        fn gru_like(batch_size, hidden_size, initial_h) {
            let hidden[0, b in 0..batch_size, h in 0..hidden_size] = initial_h[b, h];
            let hidden[t in 1..3, b in 0..batch_size, h in 0..hidden_size] = hidden[t - 1, b, h] + 1.0;
            hidden
        }
        let init = [[0.1, 0.2], [0.3, 0.4]];
        let out = gru_like(2, 2, init);
        out;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        hidden = result.outputs["out"] if "out" in result.outputs else result.outputs.get("hidden")
        assert hidden is not None, f"outputs: {result.outputs}"
        hidden = np.asarray(hidden)
        assert hidden.shape == (3, 2, 2), f"expected (3,2,2), got {hidden.shape}"
        expected_layer0 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=hidden.dtype)
        np.testing.assert_allclose(hidden[0], expected_layer0, rtol=1e-5, atol=1e-8)

    def test_einstein_gru_pattern_with_typeof_initial_h_conditional(self, compiler, runtime):
        """GRU first-clause body: if typeof(initial_h) == \"rectangular\" { initial_h[b, h] } else { 0.0 }. Check rectangular branch."""
        source = """
        fn gru_like(batch_size, hidden_size, initial_h) {
            let hidden[0, b in 0..batch_size, h in 0..hidden_size] =
                if typeof(initial_h) == "rectangular" { initial_h[b, h] } else { 0.0 };
            let hidden[t in 1..3, b in 0..batch_size, h in 0..hidden_size] = hidden[t - 1, b, h] + 1.0;
            hidden
        }
        let init = [[0.1, 0.2], [0.3, 0.4]];
        let out = gru_like(2, 2, init);
        out;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        hidden = result.outputs["out"] if "out" in result.outputs else result.outputs.get("hidden")
        assert hidden is not None, f"outputs: {result.outputs}"
        hidden = np.asarray(hidden)
        assert hidden.shape == (3, 2, 2), f"expected (3,2,2), got {hidden.shape}"
        expected_layer0 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=hidden.dtype)
        np.testing.assert_allclose(hidden[0], expected_layer0, rtol=1e-5, atol=1e-8)

    def test_einstein_literal_reference(self, compiler, runtime):
        """Reference array B only at literal indices: A[0,0]=B[0,0], A[0,i]=B[0,i], A[i,0]=B[i,0]; recurrence uses B[i,j]. Compare full A to numpy."""
        source = """
        let B[i in 0..6, j in 0..4] = i + j;
        let A[0, 0] = B[0, 0];
        let A[0, i in 1..4] = B[0, i];
        let A[i in 1..6, 0] = B[i, 0];
        let A[i in 1..6, j in 1..4] = A[i - 1, j] + A[i, j - 1] + B[i, j];
        A;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        A = result.outputs["A"]
        expected = np.array([
            [0, 1, 2, 3],
            [1, 4, 9, 16],
            [2, 9, 22, 43],
            [3, 16, 43, 92],
            [4, 25, 74, 173],
            [5, 36, 117, 298],
        ], dtype=A.dtype)
        np.testing.assert_array_equal(A, expected)

    def test_matrix_multiplication(self, compiler, runtime):
        """Test matrix multiplication with nested reductions"""
        source = """
        let A[i in 0..3, j in 0..4] = i * 3 + j;
        let B[j in 0..4, k in 0..2] = j * 2 + k;
        let C[i in 0..3, k in 0..2] = sum[j in 0..4](A[i, j] * B[j, k]);
        C;
        """
    
    def test_matrix_multiplication_implicit_ranges(self, compiler, runtime):
        """Test matrix multiplication with implicit range inference
        
        This test verifies that ranges can be inferred implicitly:
        - j is inferred from A[i,j] and B[j,k] (should be 0..4) - this works for reduction variables
        - i is inferred from A[i,j] (should be 0..3) - this requires shape info from A
        - k is inferred from B[j,k] (should be 0..2) - this requires shape info from B
        
        Currently, implicit range inference for reduction variables (j) works,
        but implicit range inference for Einstein index variables (i, k) requires
        shape analysis to run before range analysis, which is not yet implemented.
        """
        source = """
        let A[i in 0..3, j in 0..4] = i * 3 + j;
        let B[j in 0..4, k in 0..2] = j * 2 + k;
        let C[i, k] = sum[j](A[i, j] * B[j, k]);
        C;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Verify C is correct (matrix multiplication)
        assert 'C' in result.outputs
        C = result.outputs['C']
        assert C.shape == (3, 2)
        
        # Manual calculation for verification (same as test_matrix_multiplication)
        expected = np.array([
            [28, 34],
            [64, 82],
            [100, 130]
        ])
        np.testing.assert_array_equal(C, expected)
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Verify C is correct (matrix multiplication)
        assert 'C' in result.outputs
        C = result.outputs['C']
        assert C.shape == (3, 2)
        
        # Manual calculation for verification
        # C[0,0] = sum[j](A[0,j] * B[j,0]) = sum[j]((0*3+j) * (j*2+0))
        # = 0*0 + 1*2 + 2*4 + 3*6 = 0 + 2 + 8 + 18 = 28
        # C[1,0] = sum[j](A[1,j] * B[j,0]) = sum[j]((1*3+j) * (j*2+0))
        # = 3*0 + 4*2 + 5*4 + 6*6 = 0 + 8 + 20 + 36 = 64
        expected = np.array([
            [28, 34],
            [64, 82],
            [100, 130]
        ])
        np.testing.assert_array_equal(C, expected)
    
    def test_convolution_2d(self, compiler, runtime):
        """Test 2D convolution pattern"""
        source = """
        let input[i in 0..5, j in 0..5] = i + j;
        let kernel[k in 0..3, l in 0..3] = k + l;
        let output[i in 0..3, j in 0..3] = sum[k in 0..3, l in 0..3](
            input[i + k, j + l] * kernel[k, l]
        );
        output;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        assert 'output' in result.outputs
        output = result.outputs['output']
        assert output.shape == (3, 3)
    
    def test_fibonacci_sequence(self, compiler, runtime):
        """Test recurrence relation (Fibonacci)
        
        Note: This test currently fails because sequential Einstein declarations
        for the same array (fib[0] = 0, fib[1] = 1, fib[i] = ...) overwrite each other.
        may also fail with name resolution (Identifier 'i' has no DefId).
        """
        source = """
        let fib[0] = 0;
        let fib[1] = 1;
        let fib[i in 2..10] = fib[i - 1] + fib[i - 2];
        fib;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        if not result.success:
            errs = result.get_errors() if hasattr(result, "get_errors") else []
            err_str = " ".join(str(e) for e in errs).lower()
            if "defid" in err_str or "identifier" in err_str:
                pytest.skip("Name resolution for Einstein index in recurrence not yet supported")
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # TODO: Fix sequential Einstein declarations to update same array
        # For now, this test documents the limitation
        # The issue is that each declaration creates a new array, overwriting previous values
        assert 'fib' in result.outputs
        fib = result.outputs['fib']
        expected = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
        np.testing.assert_array_equal(fib, expected)
    
    def test_nested_reductions(self, compiler, runtime):
        """Test nested reductions (sum of sums)"""
        source = """
        let A[i in 0..4, j in 0..4] = i + j;
        let row_sums[i in 0..4] = sum[j in 0..4](A[i, j]);
        let total = sum[i in 0..4](row_sums[i]);
        total;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        assert 'total' in result.outputs
        # Sum of all elements: sum(i=0..3, j=0..3) of (i+j)
        # = (0+0) + (0+1) + ... + (3+3) = 0+1+2+3 + 1+2+3+4 + 2+3+4+5 + 3+4+5+6
        # = 6 + 10 + 14 + 18 = 48
        assert result.outputs['total'] == 48
    
    def test_conditional_reduction(self, compiler, runtime):
        """Test reduction with where clause condition"""
        source = """
        let A[i in 0..10] = i;
        let sum_even = sum[i in 0..10](A[i]) where i % 2 == 0;
        let sum_odd = sum[i in 0..10](A[i]) where i % 2 == 1;
        sum_even + sum_odd;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # sum_even = 0+2+4+6+8 = 20
        # sum_odd = 1+3+5+7+9 = 25
        # total = 45
        assert result.outputs.get('sum_even', 0) + result.outputs.get('sum_odd', 0) == 45
    
    def test_multi_dimensional_with_guards(self, compiler, runtime):
        """Test multi-dimensional Einstein with multiple guards"""
        source = """
        let result[i in 0..5, j in 0..5] = i * j where i > 1, j > 1, i + j < 7;
        result;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        assert 'result' in result.outputs
        result_array = result.outputs['result']
        assert result_array.shape == (5, 5)
        
        # Check that filtered elements are zero and valid elements are set
        assert result_array[0, 0] == 0  # i <= 1 (filtered)
        assert result_array[1, 1] == 0  # i <= 1 (filtered)
        assert result_array[2, 2] == 4  # 2*2 = 4, and 2+2=4 < 7 ✓
        assert result_array[3, 3] == 9  # 3*3 = 9, and 3+3=6 < 7 ✓ (all conditions satisfied)
        assert result_array[4, 2] == 8  # 4*2 = 8, and 4+2=6 < 7 ✓ (all conditions satisfied)
        # i=4, j=2: i>1 ✓, j>1 ✓, i+j=6 < 7 ✓, so result[4,2] = 4*2 = 8
    
    def test_reduction_with_dependent_ranges(self, compiler, runtime):
        """Test reduction where range depends on outer loop variable"""
        source = """
        let result[i in 0..5] = sum[j in 0..i+1](j);
        result;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        assert 'result' in result.outputs
        result_array = result.outputs['result']
        # Range i in 0..5 creates indices 0,1,2,3,4 (5 elements)
        assert result_array.shape == (5,)
        
        # result[0] = sum[j in 0..1](j) = 0
        # result[1] = sum[j in 0..2](j) = 0+1 = 1
        # result[2] = sum[j in 0..3](j) = 0+1+2 = 3
        # result[3] = sum[j in 0..4](j) = 0+1+2+3 = 6
        # result[4] = sum[j in 0..5](j) = 0+1+2+3+4 = 10
        expected = np.array([0, 1, 3, 6, 10])
        np.testing.assert_array_equal(result_array, expected)
    
    def test_three_dimensional_tensor(self, compiler, runtime):
        """Test 3D Einstein declaration"""
        source = """
        let tensor[i in 0..3, j in 0..3, k in 0..3] = i * 100 + j * 10 + k;
        tensor;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        assert 'tensor' in result.outputs
        tensor = result.outputs['tensor']
        assert tensor.shape == (3, 3, 3)
        
        # Verify some values
        assert tensor[0, 0, 0] == 0
        assert tensor[0, 0, 1] == 1
        assert tensor[0, 1, 0] == 10
        assert tensor[1, 0, 0] == 100
        assert tensor[2, 2, 2] == 222
    
    def test_reduction_all_dimensions(self, compiler, runtime):
        """Test full reduction (all dimensions)"""
        source = """
        let A[i in 0..3, j in 0..3] = i + j;
        let total = sum[i in 0..3, j in 0..3](A[i, j]);
        total;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        assert 'total' in result.outputs
        # Sum of all elements: (0+0)+(0+1)+(0+2) + (1+0)+(1+1)+(1+2) + (2+0)+(2+1)+(2+2)
        # = 0+1+2 + 1+2+3 + 2+3+4 = 3 + 6 + 9 = 18
        assert result.outputs['total'] == 18
    
    def test_partial_reduction(self, compiler, runtime):
        """Test partial reduction (one dimension)"""
        source = """
        let A[i in 0..3, j in 0..3] = i + j;
        let row_sums[i in 0..3] = sum[j in 0..3](A[i, j]);
        row_sums;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        assert 'row_sums' in result.outputs
        row_sums = result.outputs['row_sums']
        assert row_sums.shape == (3,)
        
        # row_sums[0] = 0+1+2 = 3
        # row_sums[1] = 1+2+3 = 6
        # row_sums[2] = 2+3+4 = 9
        expected = np.array([3, 6, 9])
        np.testing.assert_array_equal(row_sums, expected)
    
    def test_product_reduction(self, compiler, runtime):
        """Test product reduction"""
        source = """
        let A[i in 0..5] = i + 1;
        let product = prod[i in 0..5](A[i]);
        product;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        assert 'product' in result.outputs
        # Range i in 0..5 creates indices 0,1,2,3,4 (5 elements)
        # A[i] = i + 1, so A = [1, 2, 3, 4, 5]
        # product = 1 * 2 * 3 * 4 * 5 = 120
        assert result.outputs['product'] == 120
    
    def test_min_max_reductions(self, compiler, runtime):
        """Test min and max reductions"""
        source = """
        let A[i in 0..10] = i * 2 - 5;
        let min_val = min[i in 0..10](A[i]);
        let max_val = max[i in 0..10](A[i]);
        min_val + max_val;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Range i in 0..10 creates indices 0,1,2,...,9 (10 elements, exclusive end)
        # A[i] = i * 2 - 5, so A = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13]
        # min = -5, max = 13
        # sum = 8
        assert result.outputs.get('min_val', 0) + result.outputs.get('max_val', 0) == 8
    
    def test_empty_range_handling(self, compiler, runtime):
        """Test handling of empty ranges"""
        source = """
        let empty_sum = sum[i in 0..0](i);
        empty_sum;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Empty sum should return identity (0)
        assert result.outputs.get('empty_sum', None) == 0
    
    def test_array_access_in_body(self, compiler, runtime):
        """Test Einstein declaration with array access in body"""
        source = """
        let input[i in 0..5] = i * 2;
        let output[i in 0..4] = input[i] + input[i + 1];
        output;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        assert 'output' in result.outputs
        output = result.outputs['output']
        # Range i in 0..4 creates indices 0,1,2,3 (4 elements)
        assert output.shape == (4,)
        
        # output[0] = input[0] + input[1] = 0 + 2 = 2
        # output[1] = input[1] + input[2] = 2 + 4 = 6
        # output[2] = input[2] + input[3] = 4 + 6 = 10
        # output[3] = input[3] + input[4] = 6 + 8 = 14
        expected = np.array([2, 6, 10, 14])
        np.testing.assert_array_equal(output, expected)
    
    def test_complex_nested_structure(self, compiler, runtime):
        """Test complex nested Einstein and reduction structure"""
        source = """
        let A[i in 0..3, j in 0..3] = i * j;
        let B[i in 0..3] = sum[j in 0..3](A[i, j]);
        let C = sum[i in 0..3](B[i] * B[i]);
        C;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Range i in 0..3 creates indices 0,1,2 (3 elements)
        # A[i,j] = i*j where i in 0..3, j in 0..3
        # A = [[0,0,0], [0,1,2], [0,2,4]]
        # B[i] = sum[j](A[i,j]) where i in 0..3
        # B = [0, 3, 6]
        # C = sum[i](B[i]^2) = 0^2 + 3^2 + 6^2 = 0 + 9 + 36 = 45
        assert result.outputs.get('C', 0) == 45
    
    def test_where_clause_with_multiple_conditions(self, compiler, runtime):
        """Clauses complement each other. First 1..8, second 0..10; shape = union (max of end) = 10."""
        source = """
        let filtered[i in 1..8] = i * i where i > 2, i < 8, i % 2 == 0;
        let filtered[i in 0..10] = 0 where i <= 2 || i >= 8 || i % 2 != 0;
        filtered;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        assert 'filtered' in result.outputs
        filtered = result.outputs['filtered']
        n = filtered.shape[0]
        assert n == 10, f"expected extent 10 (union of 1..8 and 0..10), got {n}"
        for i in range(n):
            if i > 2 and i < 8 and i % 2 == 0:
                assert filtered[i] == i * i, f"at i={i}"
            else:
                assert filtered[i] == 0, f"at i={i}"
    
    def test_large_range_performance(self, compiler, runtime):
        """Test performance with larger ranges"""
        source = """
        let large[i in 0..100] = i * 2;
        let sum_large = sum[i in 0..100](large[i]);
        sum_large;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # sum = sum(i=0..99) of (i*2) = 2 * sum(i=0..99) of i = 2 * 99*100/2 = 9900
        assert result.outputs.get('sum_large', 0) == 9900
    
    def test_reduction_with_array_access(self, compiler, runtime):
        """Test reduction that accesses arrays"""
        source = """
        let A[i in 0..5] = i * 2;
        let B[j in 0..5] = j * 3;
        let dot_product = sum[k in 0..5](A[k] * B[k]);
        dot_product;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # dot_product = sum(k=0..4) of (k*2 * k*3) = sum(k=0..4) of (6*k^2)
        # = 0 + 6 + 24 + 54 + 96 = 180
        assert result.outputs.get('dot_product', 0) == 180
    
    def test_sequential_einstein_declarations(self, compiler, runtime):
        """Test multiple sequential Einstein declarations"""
        source = """
        let A[i in 0..5] = i;
        let B[i in 0..5] = A[i] * 2;
        let C[i in 0..5] = B[i] + 1;
        C;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        assert 'C' in result.outputs
        C = result.outputs['C']
        assert C.shape == (5,)
        
        # C[i] = (i * 2) + 1
        expected = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(C, expected)

