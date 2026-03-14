"""
Tests for Einstein Notation Analyzer using EinlangCompiler
"""
import pytest
from tests.test_utils import compile_and_execute


class TestEinsteinNotationAnalyzer:
    """Tests for Einstein Notation Analyzer using EinlangCompiler"""
    
    def test_extract_components_matrix_multiply(self, compiler):
        """Test matrix multiplication component extraction using system"""
        # Test through compilation system (NEW SYNTAX: explicit reduction)
        source = """
        let B = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let C = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let A[i in 0..5, j in 0..3] = sum[k](B[i,k] * C[k,j]);
        """
        
        # Uses compile() API
        result = compiler.compile(source, "<test>")
        assert result is not None
        errs = result.get_errors()
        assert result.success, f"Compilation should succeed: {errs}"
        # Uses .ir attribute
        assert result.ir is not None, "Result should have IR"
    
    def test_validate_reductions(self, compiler):
        """Test reduction validation using system"""
        # Test through compilation system
        source = """
        let B = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]];
        let A[i in 0..5] = sum[j](B[i,j]);
        """
        
        result = compiler.compile(source, "<test>")
        assert result is not None
        errs = result.get_errors()
        assert result.success, f"Compilation should succeed: {errs}"
        assert result.ir is not None, "Result should have IR"

    def test_has_einstein_notation_in_expression(self, compiler):
        """Test Einstein notation detection in expressions using system"""
        # Test through compilation system (NEW SYNTAX: explicit reduction)
        source = """
        let B = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let C = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let A[i in 0..5, j in 0..3] = sum[k](B[i,k] * C[k,j]);
        """
        
        result = compiler.compile(source, "<test>")
        assert result is not None
        errs = result.get_errors()
        assert result.success, f"Compilation should succeed: {errs}"
        assert result.ir is not None, "Result should have IR"

    def test_einstein_notation_analyzer_with_execution(self, compiler, runtime):
        """Test Einstein notation analyzer with execution using system"""
        # Test execution through system
        source = """
        let A[i in 0..2, j in 0..2] = 1.0;
        """
        
        # Execute through system using runtime
        result = compile_and_execute(source, compiler, runtime)
        assert result is not None
        assert result.success, f"Execution should succeed: {result.get_errors()}"
        
        # Check execution results
        assert result.outputs is not None, "Execution result should have outputs attribute"
        
        # Verify tensor was created
        if result.outputs:
            assert 'A' in result.outputs, "Variable A should be created"
            tensor_A = result.outputs['A']
            assert len(tensor_A) >= 2, "Tensor A should have at least 2 rows"
            assert len(tensor_A[0]) >= 2, "Tensor A should have at least 2 columns"
    
    def test_einstein_notation_analyzer_error_handling(self, compiler):
        """Test Einstein notation analyzer error handling using system"""
        # Test with invalid syntax
        invalid_source = """
        let A[i in 0..5, j in 0..3] = 1.0  # New syntax: domain definitions inline (missing semicolon)
        """
        
        result = compiler.compile(invalid_source, "<test>")
        assert result is not None
        if not result.success:
            errs = result.get_errors()
            assert len(errs) > 0, "Should have error messages for invalid syntax"
    

if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])
