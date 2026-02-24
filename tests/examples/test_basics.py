#!/usr/bin/env python3
"""
Parametrized basics tests for Einlang using pytest parametrize with system.
Automatically generates test cases for all .ein files in the examples/basics/ directory using the modern architecture.
"""

import pytest
from pathlib import Path
from tests.test_utils import compile_and_execute


def get_all_basics_files():
    """Get all basics tutorial files for parameterized testing"""
    project_root = Path(__file__).parent.parent.parent
    basics_dir = project_root / "examples" / "basics"
    if basics_dir.exists():
        return sorted(basics_dir.glob("*.ein"))
    return []


class TestBasics:
    """
    Tests for basics tutorial files using system.
    
    Industry Best Practice: Tests use proper separation
    - Compiler for parsing and analysis
    - Runtime for execution
    """
    
    
    @pytest.mark.parametrize("basics_file", get_all_basics_files(), ids=lambda f: f.stem)
    def test_execution(self, compiler, runtime, basics_file):
        """
        Test that each basics tutorial can execute successfully using system.
        Tests using IR execution path.
        
        Industry Best Practice: Uses Runtime (not Compiler) for execution
        """
        assert basics_file.exists(), f"{basics_file.name} should exist"
        
        with open(basics_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            # Use Runtime for execution (proper separation from compilation)
            result = compile_and_execute(content, compiler, runtime)
            
            # system returns structured results
            assert result is not None, f"Execution should return a result for {basics_file.name}"
            assert hasattr(result, 'success'), f"Result should have success attribute for {basics_file.name}"
            
            # Always check result.success unless it is a negative test
            if result.success:
                print(f"✅ {basics_file.name} executed successfully with system (IR mode)")
                
                # Check that we have execution results
                if hasattr(result, 'outputs'):
                    variables = result.outputs
                    print(f"   Variables: {list(variables.keys())}")
            else:
                # If execution failed, check if it's an expected failure
                errors = result.get_errors() if hasattr(result, 'get_errors') else result.errors if hasattr(result, 'errors') else []
                error_msg = str(errors) if errors else "Unknown error"
                print(f"❌ Execution failed for {basics_file.name} (IR mode): {error_msg[:100]}...")
                
                # For basics tutorials, execution failures should fail the test
                # unless it's a known limitation
                pytest.fail(f"Basics tutorial execution failed with system (IR mode): {error_msg}")
                
        except Exception as e:
            # Handle unexpected exceptions
            print(f"❌ Unexpected error for {basics_file.name} (IR mode): {str(e)[:100]}...")
            pytest.fail(f"Basics tutorial execution failed with system (IR mode): {str(e)}")
    

if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])
