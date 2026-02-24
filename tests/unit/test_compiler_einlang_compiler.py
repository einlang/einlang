"""
Unit tests for EinlangCompiler System

Tests basic compiler functionality without heavy mocking.
"""

import pytest
from tests.test_utils import apply_ir_round_trip


class TestEinlangCompiler:
    """Test EinlangCompiler basic functionality"""
    
    def test_execution_compilation(self, compiler, runtime):
        """Test execution compilation with real runtime"""
        source_code = "let x = 42;"
        
        compile_result = compiler.compile(source_code, "<test>")
        assert compile_result.success
        apply_ir_round_trip(compile_result)
        exec_result = runtime.execute(compile_result, inputs={})
        assert exec_result.success
    
    def test_execution_with_inputs(self, compiler, runtime):
        """Test execution with input variables"""
        source_code = "let x = 5; let y = x + 10;"
        
        compile_result = compiler.compile(source_code, "<test>")
        assert compile_result.success
        apply_ir_round_trip(compile_result)
        exec_result = runtime.execute(compile_result, inputs={})
        assert exec_result.success
        assert exec_result.outputs["y"] == 15
    

if __name__ == "__main__":
    pytest.main([__file__])
