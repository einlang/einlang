"""
Integration tests for deserialization Compiler System

Tests the complete compiler system integration with real execution.
No mocking of internal pass manager - tests actual behavior.
"""

import pytest
from tests.test_utils import compile_and_execute


class TestEinlangCompilerIntegration:
    """Test EinlangCompiler with system integration"""
    
    def test_compiler_initialization(self, compiler):
        """Test compiler initialization with components"""
        assert compiler.parser is not None
        # Compiler is stateless - passes are instantiated per compilation
        # Uses pass_manager.passes instead of _pass_classes
        assert compiler.pass_manager is not None
        assert compiler.pass_manager.passes is not None
        assert isinstance(compiler.pass_manager.passes, list)
        assert len(compiler.pass_manager.passes) > 0
        
        # Check that pass classes are properly defined
        for PassClass in compiler.pass_manager.passes:
            # All passes should be classes (not instances)
            assert isinstance(PassClass, type)
    
    def test_compiler_has_pipeline(self, compiler):
        """Test that compiler has default pipeline"""
        # Uses pass_manager.passes instead of _pass_classes
        pass_classes = compiler.pass_manager.passes
        
        assert pass_classes is not None
        assert isinstance(pass_classes, list)
        assert len(pass_classes) > 0
        
        # Check for key passes by checking class names
        # Note: NameResolutionPass runs before pass manager, so check for IR passes
        pass_names = [p.__name__ for p in pass_classes]
        assert any('RangeAnalysis' in name for name in pass_names), f"Expected RangeAnalysis pass, got: {pass_names}"
        assert any('ShapeAnalysis' in name or 'Shape' in name for name in pass_names), f"Expected ShapeAnalysis pass, got: {pass_names}"
        assert any('TypeInference' in name or 'Type' in name for name in pass_names), f"Expected TypeInference pass, got: {pass_names}"
    
    def test_compiler_analysis_mode(self, compiler):
        """Test compiler analysis mode with system"""
        source_code = "let x = 42;"
        
        # Uses compile() instead of analyze()
        result = compiler.compile(source_code)
        
        # Test behavior, not types - does compilation work?
        assert result.success
        assert result.ir is not None
    
    def test_compiler_execution_mode(self, compiler, runtime):
        """Test execution mode with system using Runtime"""
        source_code = "let x = 42;"
        
        result = compile_and_execute(source_code, compiler, runtime)
        
        assert hasattr(result, 'value')
        assert result.success
    
    def test_compiler_with_einstein_notation(self, compiler, runtime):
        """Test compiler with Einstein notation using system"""
        source_code = "let data = [1, 2, 3]; let fib[i] = data[i] * 2;"
        
        result = compile_and_execute(source_code, compiler, runtime)
        
        assert result.success
        assert result.outputs is not None
        assert 'fib' in result.outputs or 'data' in result.outputs


class TestSystemEndToEnd:
    """End-to-end tests for deserialization system"""
    
    def test_mixed_program_compilation(self, compiler, runtime):
        """Test mixed program compilation with system"""
        source_code = "let x = 42; let data = [1, 2, 3]; let fib[i] = data[i] + x;"
        
        execution_result = compile_and_execute(source_code, compiler, runtime)
        
        assert execution_result.success
        assert execution_result.outputs is not None
        assert 'x' in execution_result.outputs
        assert execution_result.outputs['x'] == 42


if __name__ == "__main__":
    pytest.main([__file__])
