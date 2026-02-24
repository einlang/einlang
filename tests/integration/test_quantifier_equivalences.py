"""
Test quantifier equivalences: all[i] and any[i] vs sum/max/min/prod
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


class TestQuantifierEquivalences:
    """Test that quantifiers can be expressed using other reductions"""
    
    def test_all_using_min(self, compiler, runtime):
        """Test that all[i] can be expressed using min[i]"""
        source_all = """
        let x = [1, 2, 3, 4, 5];
        let result = all[i](x[i] > 0);
        """
        source_min = """
        let x = [1, 2, 3, 4, 5];
        let result = min[i]((x[i] > 0) as i32) == 1;
        """
        exec_all = compile_and_execute(source_all, compiler, runtime)
        exec_min = compile_and_execute(source_min, compiler, runtime)
        assert exec_all.success, f"Execution failed: {exec_all.errors}"
        assert exec_min.success, f"Execution failed: {exec_min.errors}"
        assert exec_all.outputs['result'] == exec_min.outputs['result']
        assert exec_all.outputs['result'] == True
    
    def test_all_using_prod(self, compiler, runtime):
        """Test that all[i] can be expressed using prod[i]"""
        source_all = """
        let x = [1, 2, 3, 4, 5];
        let result = all[i](x[i] > 0);
        """
        source_prod = """
        let x = [1, 2, 3, 4, 5];
        let result = prod[i]((x[i] > 0) as i32) == 1;
        """
        exec_all = compile_and_execute(source_all, compiler, runtime)
        exec_prod = compile_and_execute(source_prod, compiler, runtime)
        assert exec_all.success, f"Execution failed: {exec_all.errors}"
        assert exec_prod.success, f"Execution failed: {exec_prod.errors}"
        assert exec_all.outputs['result'] == exec_prod.outputs['result']
        assert exec_all.outputs['result'] == True
    
    def test_all_using_sum(self, compiler, runtime):
        """Test that all[i] can be expressed using sum[i]"""
        source_all = """
        let x = [1, 2, 3, 4, 5];
        let result = all[i](x[i] > 0);
        """
        source_sum = """
        let x = [1, 2, 3, 4, 5];
        let result = sum[i]((x[i] > 0) as i32) == len(x);
        """
        exec_all = compile_and_execute(source_all, compiler, runtime)
        exec_sum = compile_and_execute(source_sum, compiler, runtime)
        assert exec_all.success, f"Execution failed: {exec_all.errors}"
        assert exec_sum.success, f"Execution failed: {exec_sum.errors}"
        assert exec_all.outputs['result'] == exec_sum.outputs['result']
        assert exec_all.outputs['result'] == True
    
    def test_any_using_max(self, compiler, runtime):
        """Test that any[i] can be expressed using max[i]"""
        source_any = """
        let x = [-1, -2, 3, -4, -5];
        let result = any[i](x[i] > 0);
        """
        source_max = """
        let x = [-1, -2, 3, -4, -5];
        let result = max[i]((x[i] > 0) as i32) == 1;
        """
        exec_any = compile_and_execute(source_any, compiler, runtime)
        exec_max = compile_and_execute(source_max, compiler, runtime)
        assert exec_any.success, f"Execution failed: {exec_any.errors}"
        assert exec_max.success, f"Execution failed: {exec_max.errors}"
        assert exec_any.outputs['result'] == exec_max.outputs['result']
        assert exec_any.outputs['result'] == True
    
    def test_any_using_sum(self, compiler, runtime):
        """Test that any[i] can be expressed using sum[i]"""
        source_any = """
        let x = [-1, -2, 3, -4, -5];
        let result = any[i](x[i] > 0);
        """
        source_sum = """
        let x = [-1, -2, 3, -4, -5];
        let result = sum[i]((x[i] > 0) as i32) > 0;
        """
        exec_any = compile_and_execute(source_any, compiler, runtime)
        exec_sum = compile_and_execute(source_sum, compiler, runtime)
        assert exec_any.success, f"Execution failed: {exec_any.errors}"
        assert exec_sum.success, f"Execution failed: {exec_sum.errors}"
        assert exec_any.outputs['result'] == exec_sum.outputs['result']
        assert exec_any.outputs['result'] == True
    
    def test_all_false_case(self, compiler, runtime):
        """Test all[i] with some false values using different methods"""
        source_all = """
        let x = [1, 2, -3, 4, 5];
        let result = all[i](x[i] > 0);
        """
        source_min = """
        let x = [1, 2, -3, 4, 5];
        let result = min[i]((x[i] > 0) as i32) == 1;
        """
        exec_all = compile_and_execute(source_all, compiler, runtime)
        exec_min = compile_and_execute(source_min, compiler, runtime)
        assert exec_all.success, f"Execution failed: {exec_all.errors}"
        assert exec_min.success, f"Execution failed: {exec_min.errors}"
        assert exec_all.outputs['result'] == exec_min.outputs['result']
        assert exec_all.outputs['result'] == False
    
    def test_any_false_case(self, compiler, runtime):
        """Test any[i] with all false values using different methods"""
        source_any = """
        let x = [-1, -2, -3, -4, -5];
        let result = any[i](x[i] > 0);
        """
        source_max = """
        let x = [-1, -2, -3, -4, -5];
        let result = max[i]((x[i] > 0) as i32) == 1;
        """
        exec_any = compile_and_execute(source_any, compiler, runtime)
        exec_max = compile_and_execute(source_max, compiler, runtime)
        assert exec_any.success, f"Execution failed: {exec_any.errors}"
        assert exec_max.success, f"Execution failed: {exec_max.errors}"
        assert exec_any.outputs['result'] == exec_max.outputs['result']
        assert exec_any.outputs['result'] == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

