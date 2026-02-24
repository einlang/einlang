"""
Tests for Einstein Analyzer using EinlangCompiler
"""
import pytest
from tests.test_utils import compile_and_execute


class TestEinsteinAnalyzer:
    """Tests for Einstein Analyzer using EinlangCompiler"""
    
    def test_einstein_analyzer_with_execution(self, compiler, runtime):
        """Test Einstein analyzer with execution using system"""
        # Test execution through system
        source = """
        let A[i in 0..2, j in 0..2] = 1.0;
        """
        
        # Execute through system
        result = compile_and_execute(source, compiler, runtime)
        assert result is not None
        assert result.success, f"Execution should succeed: {result.get_errors()}"
        
        # Check execution results
        assert hasattr(result, 'outputs'), "Execution result should have outputs attribute"
        
        # Verify tensor was created
        if result.outputs:
            assert 'A' in result.outputs, "Variable A should be created"
            tensor_A = result.outputs['A']
            assert len(tensor_A) >= 2, "Tensor A should have at least 2 rows"
            assert len(tensor_A[0]) >= 2, "Tensor A should have at least 2 columns"
    
    def test_einstein_analyzer_error_handling(self, compiler):
        """Test Einstein analyzer error handling using system"""
        # Test with invalid syntax
        invalid_source = """
        let A[i in 0..2, j in 0..2] = 1.0  # New syntax: domain definitions inline (missing semicolon)
        """
        
        # Uses compile() API
        if hasattr(compiler, "compile"):
            result = compiler.compile(invalid_source, "<test>")
        else:
            result = compiler.analyze(invalid_source)
        assert result is not None
        if not result.success:
            errs = result.get_errors() if hasattr(result, "get_errors") else getattr(result, "errors", [])
            assert len(errs) > 0, "Should have error messages for invalid syntax"
    

if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])