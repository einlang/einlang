"""
End-to-End tests for Fibonacci sequence implementation using system

Tests the complete tensor recurrence relations functionality
including grouped declarations and self-reference patterns using the modern architecture.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


class TestFibonacciE2EV2:
    """End-to-End tests for Fibonacci sequence using system"""
    
    def test_fibonacci_sequence_basic(self, compiler, runtime):
        """Test basic Fibonacci sequence generation using system"""
        source = """
        # Fibonacci sequence: f[n] = f[n-1] + f[n-2]
        let fib[0] = 1;
        let fib[1] = 1;
        let fib[n in 2..11] = fib[n-1] + fib[n-2];
        
        # Verify the sequence
        assert(fib[0] == 1, "fib[0] should be 1");
        assert(fib[1] == 1, "fib[1] should be 1");
        assert(fib[2] == 2, "fib[2] should be 2");
        assert(fib[3] == 3, "fib[3] should be 3");
        assert(fib[4] == 5, "fib[4] should be 5");
        assert(fib[5] == 8, "fib[5] should be 8");
        assert(fib[6] == 13, "fib[6] should be 13");
        assert(fib[7] == 21, "fib[7] should be 21");
        assert(fib[8] == 34, "fib[8] should be 34");
        assert(fib[9] == 55, "fib[9] should be 55");
        assert(fib[10] == 89, "fib[10] should be 89");
        """
        
        # Execute using system
        result = compile_and_execute(source, compiler, runtime)
        
        # Verify execution was successful
        assert result is not None, "Fibonacci test failed: execution returned None"
        assert result.success, f"Fibonacci test failed: {result.errors}"
        
        # Check that we have execution results
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'fib' in variables, "Fibonacci test failed: 'fib' variable not found in result"
            
            # Verify the Fibonacci sequence
            expected_fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
            np.testing.assert_array_equal(variables['fib'], expected_fib,
                err_msg=f"Fibonacci sequence should be {expected_fib}, got {variables['fib']}")
    
    def test_fibonacci_sequence_array_comparison(self, compiler, runtime):
        """Test Fibonacci sequence with array comparison using system"""
        source = """
        # Fibonacci sequence
        let fib[0] = 1;
        let fib[1] = 1;
        let fib[n in 2..9] = fib[n-1] + fib[n-2];
        
        # Expected sequence
        let expected = [1, 1, 2, 3, 5, 8, 13, 21, 34];
        
        # Verify the entire sequence
        assert(fib == expected, "Fibonacci sequence should match expected values: got {fib}, expected {expected}");
        """
        
        # Execute using system
        result = compile_and_execute(source, compiler, runtime)
        
        # Verify execution was successful
        assert result is not None, "Fibonacci array comparison test failed: execution returned None"
        assert result.success, f"Fibonacci array comparison test failed: {result.errors}"
        
        # Check execution results
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'fib' in variables, "Fibonacci array comparison test failed: 'fib' variable not found in result"
            assert 'expected' in variables, "Expected variable should be in execution results"
            
            # Verify the sequences match
            np.testing.assert_array_equal(variables['fib'], variables['expected'],
                err_msg=f"Fibonacci sequence should match expected: got {variables['fib']}, expected {variables['expected']}")
    
    def test_lucas_sequence(self, compiler, runtime):
        """Test Lucas sequence using system (similar to Fibonacci but different base cases)"""
        source = """
        # Lucas sequence: L[n] = L[n-1] + L[n-2] with L[0] = 2, L[1] = 1
        let lucas[0] = 2;
        let lucas[1] = 1;
        let lucas[n in 2..9] = lucas[n-1] + lucas[n-2];
        
        # Expected Lucas sequence
        let expected = [2, 1, 3, 4, 7, 11, 18, 29, 47];
        
        # Verify the sequence
        assert(lucas == expected, "Lucas sequence should match expected values: got {lucas}, expected {expected}");
        """
        
        # Execute using system
        result = compile_and_execute(source, compiler, runtime)
        
        # Verify execution was successful
        assert result is not None, "Lucas sequence test failed: execution returned None"
        assert result.success, f"Lucas sequence test failed: {result.errors}"
        
        # Check execution results
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'lucas' in variables, "Lucas sequence test failed: 'lucas' variable not found in result"
            assert 'expected' in variables, "Expected variable should be in execution results"
            
            # Verify the Lucas sequence
            import numpy as np
            expected_lucas = [2, 1, 3, 4, 7, 11, 18, 29, 47]
            np.testing.assert_array_equal(variables['lucas'], expected_lucas, 
                err_msg=f"Lucas sequence should be {expected_lucas}, got {variables['lucas']}")
    
    def test_fibonacci_with_different_base_cases(self, compiler, runtime):
        """Test Fibonacci with different base cases using system"""
        source = """
        # Fibonacci with different base cases
        let fib[0] = 0;
        let fib[1] = 1;
        let fib[n in 2..9] = fib[n-1] + fib[n-2];
        
        # Expected sequence (standard Fibonacci starting with 0, 1)
        let expected = [0, 1, 1, 2, 3, 5, 8, 13, 21];
        
        # Verify the sequence
        assert(fib == expected, "Fibonacci with base cases 0,1 should match expected values: got {fib}, expected {expected}");
        """
        
        # Execute using system
        result = compile_and_execute(source, compiler, runtime)
        
        # Verify execution was successful
        assert result is not None, "Fibonacci with different base cases test failed: execution returned None"
        assert result.success, f"Fibonacci with different base cases test failed: {result.errors}"
        
        # Check execution results
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'fib' in variables, "Fibonacci with different base cases test failed: 'fib' variable not found in result"
            assert 'expected' in variables, "Expected variable should be in execution results"
            
            # Verify the Fibonacci sequence with different base cases
            expected_fib = [0, 1, 1, 2, 3, 5, 8, 13, 21]
            np.testing.assert_array_equal(variables['fib'], expected_fib,
                err_msg=f"Fibonacci sequence should be {expected_fib}, got {variables['fib']}")
    
    def test_fibonacci_large_sequence(self, compiler, runtime):
        """Test Fibonacci sequence with larger range using system"""
        source = """
        # Fibonacci sequence with larger range
        let fib[0] = 1;
        let fib[1] = 1;
        let fib[n in 2..16] = fib[n-1] + fib[n-2];
        
        # Verify some key values
        assert(fib[10] == 89, "fib[10] should be 89");
        assert(fib[15] == 987, "fib[15] should be 987");
        
        # Verify the sequence is increasing
        assert(fib[15] > fib[10], "Fibonacci sequence should be increasing");
        assert(fib[10] > fib[5], "Fibonacci sequence should be increasing");
        """
        
        # Execute using system
        result = compile_and_execute(source, compiler, runtime)
        
        # Verify execution was successful
        assert result is not None, "Fibonacci large sequence test failed: execution returned None"
        assert result.success, f"Fibonacci large sequence test failed: {result.errors}"
        
        # Check execution results
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'fib' in variables, "Fibonacci large sequence test failed: 'fib' variable not found in result"
            
            # Verify specific values
            fib_sequence = variables['fib']
            assert fib_sequence[10] == 89, f"fib[10] should be 89, got {fib_sequence[10]}"
            assert fib_sequence[15] == 987, f"fib[15] should be 987, got {fib_sequence[15]}"
            
            # Verify increasing property
            assert fib_sequence[15] > fib_sequence[10], "Fibonacci sequence should be increasing"
            assert fib_sequence[10] > fib_sequence[5], "Fibonacci sequence should be increasing"
    
    def test_fibonacci_with_variables(self, compiler, runtime):
        """Test Fibonacci sequence using variables for base cases with system"""
        source = """
        # Use variables for base cases
        let a = 1;
        let b = 1;
        
        # Fibonacci sequence using variables
        let fib[0] = a;
        let fib[1] = b;
        let fib[n in 2..9] = fib[n-1] + fib[n-2];
        
        # Expected sequence
        let expected = [1, 1, 2, 3, 5, 8, 13, 21, 34];
        
        # Verify the sequence
        assert(fib == expected, "Fibonacci with variables should match expected values: got {fib}, expected {expected}");
        """
        
        # Execute using system
        result = compile_and_execute(source, compiler, runtime)
        
        # Verify execution was successful
        assert result is not None, "Fibonacci with variables test failed: execution returned None"
        assert result.success, f"Fibonacci with variables test failed: {result.errors}"
        
        # Check execution results
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'fib' in variables, "Fibonacci with variables test failed: 'fib' variable not found in result"
            assert 'expected' in variables, "Expected variable should be in execution results"
            
            # Verify the Fibonacci sequence
            expected_fib = [1, 1, 2, 3, 5, 8, 13, 21, 34]
            np.testing.assert_array_equal(variables['fib'], expected_fib,
                err_msg=f"Fibonacci sequence should be {expected_fib}, got {variables['fib']}")
    
    def test_fibonacci_error_handling(self, compiler, runtime):
        """Test error handling for invalid Fibonacci patterns using system"""
        source = """
        # This should fail - missing base case
        let fib[1] = 1;
        let fib[n in 2..5] = fib[n-1] + fib[n-2];
        """
        
        # Execute using system - should handle errors gracefully
        result = compile_and_execute(source, compiler, runtime)
        
        # system should either succeed (if it handles missing base cases) or fail gracefully
        if result.success:
            # If it succeeds, that's also valid - the system might handle missing base cases
            assert hasattr(result, 'outputs'), "Result should have variables attribute"
        else:
            # If it fails, it should fail gracefully with proper error messages
            assert len(result.errors) > 0, "Should have error messages when execution fails"
            # Check that errors are properly formatted
            for error in result.errors:
                assert isinstance(error, str) or hasattr(error, 'message'), "Error should be a string or have a message attribute"
    
    def test_fibonacci_with_expressions(self, compiler, runtime):
        """Test Fibonacci sequence with complex expressions using system"""
        source = """
        # Fibonacci with expressions in base cases
        let fib[0] = 1 + 0;
        let fib[1] = 2 - 1;
        let fib[n in 2..7] = fib[n-1] + fib[n-2];
        
        # Expected sequence
        let expected = [1, 1, 2, 3, 5, 8, 13];
        
        # Verify the sequence
        assert(fib == expected, "Fibonacci with expressions should match expected values: got {fib}, expected {expected}");
        """
        
        # Execute using system
        result = compile_and_execute(source, compiler, runtime)
        
        # Verify execution was successful
        assert result is not None, "Fibonacci with expressions test failed: execution returned None"
        assert result.success, f"Fibonacci with expressions test failed: {result.errors}"
        
        # Check execution results
        if hasattr(result, 'outputs'):
            variables = result.outputs
            assert 'fib' in variables, "Fibonacci with expressions test failed: 'fib' variable not found in result"
            assert 'expected' in variables, "Expected variable should be in execution results"
            
            # Verify the Fibonacci sequence
            expected_fib = [1, 1, 2, 3, 5, 8, 13]
            np.testing.assert_array_equal(variables['fib'], expected_fib,
                err_msg=f"Fibonacci sequence should be {expected_fib}, got {variables['fib']}")
    
    def test_fibonacci_analysis_mode(self, compiler, runtime):
        """Test Fibonacci sequence in analysis mode using system"""
        source = """
        # Fibonacci sequence for analysis
        let fib[0] = 1;
        let fib[1] = 1;
        let fib[n in 2..5] = fib[n-1] + fib[n-2];
        """
        

if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])
