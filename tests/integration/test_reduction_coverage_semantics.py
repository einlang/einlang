"""
Test to verify value filtering in reductions does NOT create coverage holes

Reductions always produce scalars, so value filtering cannot create coverage holes.
This is a key semantic property that distinguishes reductions from comprehensions.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


class TestReductionCoverageSemantics:
    """Verify reductions with value filtering never create coverage holes"""
    
    def test_reduction_with_filter_produces_scalar(self, compiler, runtime):
        """Value filtering in reduction still produces a scalar, not a sparse array"""
        code = """
        let data = [1, -2, 3, -4, 5];
        let result = sum[i in 0..5](data[i] where data[i] > 0);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        
        # Result is a scalar, not an array
        result = exec_result.outputs['result']
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert result == 9
    
    def test_einstein_with_filtered_reduction_full_coverage(self, compiler, runtime):
        """Einstein declaration with filtered reduction has full coverage (no holes)"""
        code = """
        let data = [[1, -2, 3], [-4, 5, -6]];
        let row_sums[i in 0..2] = sum[j](data[i,j] where data[i, j] > 0);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        
        # row_sums is fully defined (no holes) - one value per row
        result = exec_result.outputs['row_sums']
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,), "Shape is determined by Einstein indices, not filters"
        # NOTE: The guard is applied during the reduction, not the Einstein assignment
        # So all rows are assigned (no coverage holes), but each row's sum is filtered
        # This is the key property: reduction guards don't create coverage holes!
        # We just verify the shape is fully defined
        assert len(result) == 2, "Both rows have values (no holes)"
    
    def test_all_filtered_out_still_no_hole(self, compiler, runtime):
        """Even when all values filtered out, result is identity element (not a hole)"""
        code = """
        let data = [1, 2, 3];
        let result = sum[i in 0..3](data[i] where data[i] > 10);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        
        # Result is 0 (identity), not undefined/hole
        assert exec_result.outputs['result'] == 0
    
    def test_contrast_with_comprehension_coverage(self, compiler, runtime):
        """
        Document the contrast: comprehensions CAN create sparse arrays,
        but reductions with filtering still produce regular arrays
        """
        code = """
        let data = [1, -2, 3, -4, 5];
        
        # Comprehension: creates jagged array (irregular)
        let sparse = [data[i] | i in 0..5, data[i] > 0];
        
        # Reduction: produces scalar (always "covered")
        let dense_sum = sum[i in 0..5](data[i] where data[i] > 0);
        
        # Einstein with reduction: produces regular array (full coverage)
        let per_elem[i in 0..5] = sum[j](data[j] where data[j] > i);
        """
        exec_result = compile_and_execute(code, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        
        # sparse is jagged (length 3)
        assert len(exec_result.outputs['sparse']) == 3
        
        # dense_sum is scalar
        assert isinstance(exec_result.outputs['dense_sum'], (int, float, np.integer, np.floating))
        
        # per_elem is regular array (length 5) - full coverage
        assert exec_result.outputs['per_elem'].shape == (5,)

