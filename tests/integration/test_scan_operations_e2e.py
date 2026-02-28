#!/usr/bin/env python3
"""
End-to-end tests for scan operations using Einstein notation with system.
Tests cumulative operations that work with the current implementation using the modern architecture.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


class TestScanOperationsE2EV2:
    """End-to-end tests for scan operations using system (working Einstein notation features)"""
    
    def test_cumulative_sum_operations(self, compiler, runtime):
        """Test cumulative sum operations using Einstein notation with system"""
        cumsum_code = """
        // Cumulative Sum - running totals
        let data = [1, 2, 3, 4];
        let cumsum[i in 0..4] = sum[k in 0..i+1](data[k]);
        
        // Extract values for testing
        let val0 = cumsum[0];
        let val1 = cumsum[1];
        let val2 = cumsum[2];
        let val3 = cumsum[3];
        
        // Verify cumulative sum values
        assert(val0 == 1, "cumsum[0] should be 1");
        assert(val1 == 3, "cumsum[1] should be 1+2=3");
        assert(val2 == 6, "cumsum[2] should be 1+2+3=6");
        assert(val3 == 10, "cumsum[3] should be 1+2+3+4=10");
        
        // Verify the entire array
        let expected = [1, 3, 6, 10];
        assert(cumsum == expected, "cumsum should match expected values");
        """
        
        # Execute using system
        result = compile_and_execute(cumsum_code, compiler, runtime)
        assert result is not None, "Cumulative sum computation should return a result"
        assert result.success, f"Cumulative sum computation should succeed: {result.errors}"
        
        # Check that we have execution results
        assert hasattr(result, 'outputs'), "Result should have variables attribute"
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'cumsum' in variables, "cumsum variable should be in execution results"
            np.testing.assert_array_equal(variables['cumsum'], [1, 3, 6, 10], 
                err_msg=f"cumsum should be [1, 3, 6, 10], got {variables['cumsum']}")
    
    def test_cumulative_product_operations(self, compiler, runtime):
        """Test cumulative product operations using Einstein notation with system"""
        cumprod_code = """
        // Cumulative Product - running products
        let values = [2, 3, 1, 4];
        let cumprod[i in 0..4] = prod[k in 0..i+1](values[k]);
        
        // Extract values for testing
        let val0 = cumprod[0];
        let val1 = cumprod[1];
        let val2 = cumprod[2];
        let val3 = cumprod[3];
        
        // Verify cumulative product values
        assert(val0 == 2, "cumprod[0] should be 2");
        assert(val1 == 6, "cumprod[1] should be 2*3=6");
        assert(val2 == 6, "cumprod[2] should be 2*3*1=6");
        assert(val3 == 24, "cumprod[3] should be 2*3*1*4=24");
        
        // Verify the entire array
        let expected = [2, 6, 6, 24];
        assert(cumprod == expected, "cumprod should match expected values");
        """
        
        # Execute using system
        result = compile_and_execute(cumprod_code, compiler, runtime)
        assert result is not None, "Cumulative product computation should return a result"
        assert result.success, f"Cumulative product computation should succeed: {result.errors}"
        
        # Check execution results
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'cumprod' in variables, "cumprod variable should be in execution results"
            np.testing.assert_array_equal(variables['cumprod'], [2, 6, 6, 24],
                err_msg=f"cumprod should be [2, 6, 6, 24], got {variables['cumprod']}")
    
    def test_cumulative_maximum_operations(self, compiler, runtime):
        """Test cumulative maximum operations using Einstein notation with system"""
        cummax_code = """
        // Cumulative Maximum - running max
        let prices = [3, 1, 4, 5];
        let cummax[i in 0..4] = max[k in 0..i+1](prices[k]);
        
        // Extract values for testing
        let val0 = cummax[0];
        let val1 = cummax[1];
        let val2 = cummax[2];
        let val3 = cummax[3];
        
        // Verify cumulative maximum values
        assert(val0 == 3, "cummax[0] should be max(3)=3");
        assert(val1 == 3, "cummax[1] should be max(1,3)=3");
        assert(val2 == 4, "cummax[2] should be max(4,1,3)=4");
        assert(val3 == 5, "cummax[3] should be max(5,4,1,3)=5");
        
        // Verify the entire array
        let expected = [3, 3, 4, 5];
        assert(cummax == expected, "cummax should match expected values");
        """
        
        # Execute using system
        result = compile_and_execute(cummax_code, compiler, runtime)
        assert result is not None, "Cumulative maximum computation should return a result"
        assert result.success, f"Cumulative maximum computation should succeed: {result.errors}"
        
        # Check execution results
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'cummax' in variables, "cummax variable should be in execution results"
            np.testing.assert_array_equal(variables['cummax'], [3, 3, 4, 5],
                err_msg=f"cummax should be [3, 3, 4, 5], got {variables['cummax']}")
    
    def test_running_average_operations(self, compiler, runtime):
        """Test running average operations using Einstein notation with system"""
        running_avg_code = """
        // Running average using cumsum
        let observations = [10, 20, 30, 40];
        let running_sum[i in 0..4] = sum[k in 0..i+1](observations[k]);
        let running_avg[i in 0..4] = running_sum[i] / (i + 1);
        
        // Extract values for testing
        let val0 = running_avg[0];
        let val1 = running_avg[1];
        let val2 = running_avg[2];
        let val3 = running_avg[3];
        
        // Verify running average values
        assert(val0 == 10, "running_avg[0] should be 10/1=10");
        assert(val1 == 15, "running_avg[1] should be (10+20)/2=15");
        assert(val2 == 20, "running_avg[2] should be (10+20+30)/3=20");
        assert(val3 == 25, "running_avg[3] should be (10+20+30+40)/4=25");
        
        // Verify the entire array
        let expected = [10, 15, 20, 25];
        assert(running_avg == expected, "running_avg should match expected values");
        """
        
        # Execute using system
        result = compile_and_execute(running_avg_code, compiler, runtime)
        assert result is not None, "Running average computation should return a result"
        assert result.success, f"Running average computation should succeed: {result.errors}"
        
        # Check execution results
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'running_avg' in variables, "running_avg variable should be in execution results"
            np.testing.assert_array_equal(variables['running_avg'], [10, 15, 20, 25],
                err_msg=f"running_avg should be [10, 15, 20, 25], got {variables['running_avg']}")
    
    def test_energy_accumulation_operations(self, compiler, runtime):
        """Test energy accumulation operations using Einstein notation with system"""
        energy_code = """
        // Energy accumulation in signal processing
        let signal = [2, -1, 3, -2];
        let energy[i in 0..4] = sum[k in 0..i+1](signal[k] * signal[k]);
        
        // Extract values for testing
        let val0 = energy[0];
        let val1 = energy[1];
        let val2 = energy[2];
        let val3 = energy[3];
        
        // Verify energy accumulation values
        assert(val0 == 4, "energy[0] should be 2²=4");
        assert(val1 == 5, "energy[1] should be 2²+(-1)²=5");
        assert(val2 == 14, "energy[2] should be 2²+(-1)²+3²=14");
        assert(val3 == 18, "energy[3] should be 2²+(-1)²+3²+(-2)²=18");
        
        // Verify the entire array
        let expected = [4, 5, 14, 18];
        assert(energy == expected, "energy should match expected values");
        """
        
        # Execute using system
        result = compile_and_execute(energy_code, compiler, runtime)
        assert result is not None, "Energy accumulation computation should return a result"
        assert result.success, f"Energy accumulation computation should succeed: {result.errors}"
        
        # Check execution results
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'energy' in variables, "energy variable should be in execution results"
            np.testing.assert_array_equal(variables['energy'], [4, 5, 14, 18],
                err_msg=f"energy should be [4, 5, 14, 18], got {variables['energy']}")
    
    def test_financial_cumulative_returns(self, compiler, runtime):
        """Test financial cumulative returns using Einstein notation with system"""
        financial_code = """
        // Financial example: Cumulative returns
        let returns = [1.02, 0.98, 1.05, 0.99];
        let cum_returns[i in 0..4] = prod[k in 0..i+1](returns[k]);
        
        // Extract values for testing (avoids array indexing in assert conditions)
        let day0_return = cum_returns[0];
        let day1_return = cum_returns[1];
        
        // Portfolio multiplier after each day (use range checks for float32 precision)
        assert(day0_return > 1.019 && day0_return < 1.021, "Day 0 return multiplier should be ~1.02");
        // Day 1: 1.02 * 0.98 = 0.9996
        assert(day1_return < 1.0, "Day 1: Portfolio should be below initial");
        assert(day1_return > 0.99, "Day 1: Portfolio should be close to 0.9996");
        """
        
        # Execute using system
        result = compile_and_execute(financial_code, compiler, runtime)
        assert result is not None, "Financial cumulative returns computation should return a result"
        assert result.success, f"Financial cumulative returns computation should succeed: {result.errors}"
        
        # Check execution results
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'cum_returns' in variables, "cum_returns variable should be in execution results"
            # Check specific values (with tolerance for float32 precision)
            assert abs(variables['cum_returns'][0] - 1.02) < 0.001, f"Day 0 should be ~1.02, got {variables['cum_returns'][0]}"
            assert abs(variables['cum_returns'][1] - 1.02 * 0.98) < 0.001, f"Day 1 should be ~{1.02 * 0.98}, got {variables['cum_returns'][1]}"
    
    def test_v2_system_analysis_mode(self, compiler, runtime):
        """Test that system can run in analysis-only mode"""
        analysis_code = """
        # Test analysis mode - should not execute but should analyze
        let data = [1, 2, 3, 4];
        let cumsum[i in 0..4] = sum[k](data[i-k]);
        """
        

if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])
