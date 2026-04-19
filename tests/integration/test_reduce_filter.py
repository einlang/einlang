"""
End-to-end tests for reduction value filtering

Tests value-based filtering in reduction expressions:
- sum[i](data[i]) where i in 0..N, data[i] > 0
- Masked reductions for ML applications
- Complex guard conditions
"""

import pytest
from tests.test_utils import compile_and_execute


class TestBasicValueFiltering:
    """Test basic value filtering in reductions"""
    
    def test_sum_with_positive_filter(self, compiler, runtime):
        """Test sum filtering positive values only"""
        code = """
        let data = [1, -2, 3, -4, 5];
        let pos_sum = sum[i in 0..5](data[i] where data[i] > 0);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['pos_sum'] == 9  # 1 + 3 + 5
    
    def test_sum_with_negative_filter(self, compiler, runtime):
        """Test sum filtering negative values only"""
        code = """
        let data = [1, -2, 3, -4, 5];
        let neg_sum = sum[i in 0..5](data[i] where data[i] < 0);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['neg_sum'] == -6  # -2 + -4
    
    def test_sum_with_range_filter(self, compiler, runtime):
        """Test sum filtering values in a range"""
        code = """
        let data = [1, 5, 3, 8, 2, 9];
        let mid_sum = sum[i in 0..6](data[i] where data[i] >= 3, data[i] <= 6);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['mid_sum'] == 8  # 5 + 3 (values between 3 and 6 inclusive)
    
    def test_count_with_filter(self, compiler, runtime):
        """Test counting with value filter"""
        code = """
        let data = [1, 5, 3, 8, 2, 9];
        let count_large = sum[i in 0..6](1 where data[i] > 5);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['count_large'] == 2  # 8 and 9


class TestMaskedReductions:
    """Test masked reductions (critical for ML)"""
    
    def test_boolean_mask_reduction(self, compiler, runtime):
        """Test reduction with boolean mask"""
        code = """
        let data = [1, 2, 3, 4, 5];
        let mask = [true, false, true, false, true];
        let masked_sum = sum[i in 0..5](data[i] where mask[i]);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['masked_sum'] == 9  # data[0] + data[2] + data[4] = 1 + 3 + 5
    
    def test_attention_mask_per_row(self, compiler, runtime):
        """Test attention masking per row (common in transformers)"""
        code = """
        let scores = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]];
        let attention_mask = [[true, true, false], [true, true, true], [false, true, true]];
        let row0_sum = sum[j in 0..3](scores[0,j] where attention_mask[0,j]);
        let row1_sum = sum[j in 0..3](scores[1,j] where attention_mask[1,j]);
        let row2_sum = sum[j in 0..3](scores[2,j] where attention_mask[2,j]);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert abs(exec_result.outputs['row0_sum'] - 0.9) < 0.001  # 0.8 + 0.1
        assert abs(exec_result.outputs['row1_sum'] - 1.0) < 0.001  # 0.2 + 0.7 + 0.1
        assert abs(exec_result.outputs['row2_sum'] - 0.7) < 0.001  # 0.3 + 0.4


class Test2DValueFiltering:
    """Test value filtering in 2D reductions"""
    
    def test_matrix_conditional_sum(self, compiler, runtime):
        """Test sum of matrix elements meeting condition"""
        code = """
        let matrix = [[1, 2, 3], [4, 5, 6]];
        let large_sum = sum[i in 0..2, j in 0..3](matrix[i,j] where matrix[i,j] > 3);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['large_sum'] == 15  # 4 + 5 + 6
    
    def test_matrix_diagonal_filter(self, compiler, runtime):
        """Test reduction on diagonal with value filter"""
        code = """
        let matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let diag_large = sum[i in 0..3](matrix[i,i] where matrix[i,i] > 3);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['diag_large'] == 14  # 5 + 9


class TestProdWithFiltering:
    """Test product reduction with value filtering"""
    
    def test_prod_with_filter(self, compiler, runtime):
        """Test product of positive values"""
        code = """
        let data = [2, -1, 3, -2, 4];
        let pos_prod = prod[i in 0..5](data[i] where data[i] > 0);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['pos_prod'] == 24  # 2 * 3 * 4


class TestMinMaxWithFiltering:
    """Test min/max reductions with value filtering"""
    
    def test_max_with_filter(self, compiler, runtime):
        """Test max of filtered values"""
        code = """
        let data = [1, 5, 3, 8, 2, 9];
        let max_small = max[i in 0..6](data[i] where data[i] < 6);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['max_small'] == 5  # max of [1, 5, 3, 2]
    
    def test_min_with_filter(self, compiler, runtime):
        """Test min of filtered values"""
        code = """
        let data = [1, 5, 3, 8, 2, 9];
        let min_large = min[i in 0..6](data[i] where data[i] > 5);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['min_large'] == 8  # min of [8, 9]


class TestEdgeCases:
    """Test edge cases for value filtering"""
    
    def test_all_filtered_out_sum(self, compiler, runtime):
        """Test sum when all values are filtered out"""
        code = """
        let data = [1, 2, 3];
        let empty_sum = sum[i in 0..3](data[i] where data[i] > 10);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['empty_sum'] == 0  # Identity element for sum
    
    def test_all_filtered_out_prod(self, compiler, runtime):
        """Test product when all values are filtered out"""
        code = """
        let data = [1, 2, 3];
        let empty_prod = prod[i in 0..3](data[i] where data[i] > 10);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['empty_prod'] == 1  # Identity element for product
    
    def test_none_filtered_out(self, compiler, runtime):
        """Test when no values are filtered out"""
        code = """
        let data = [1, 2, 3, 4, 5];
        let all_sum = sum[i in 0..5](data[i] where data[i] > 0);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['all_sum'] == 15  # All values included


class TestComplexGuards:
    """Test complex guard conditions"""
    
    def test_multiple_guard_conditions(self, compiler, runtime):
        """Test multiple guard conditions (AND logic)"""
        code = """
        let data = [1, 5, 3, 8, 2, 9];
        let filtered = sum[i in 0..6](data[i] where data[i] > 2, data[i] < 9);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['filtered'] == 16  # 5 + 3 + 8
    
    def test_guard_with_expression(self, compiler, runtime):
        """Test guard with complex expression"""
        code = """
        let data = [1, 2, 3, 4, 5, 6];
        let even_sum = sum[i in 0..6](data[i] where data[i] % 2 == 0);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert exec_result.outputs['even_sum'] == 12  # 2 + 4 + 6

