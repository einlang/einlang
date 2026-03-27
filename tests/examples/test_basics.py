#!/usr/bin/env python3
"""
Parametrized basics tests for Einlang using pytest parametrize with system.
Automatically generates test cases for all .ein files in the examples/basics/ directory using the modern architecture.
"""

import pytest
from tests.test_utils import compile_and_execute, load_example_sources


_BASICS_SOURCES = load_example_sources("examples/basics")


class TestBasics:
    """
    Tests for basics tutorial files using system.
    
    Industry Best Practice: Tests use proper separation
    - Compiler for parsing and analysis
    - Runtime for execution
    """
    
    
    @pytest.mark.parametrize("basics_source", _BASICS_SOURCES, ids=lambda source: source.name)
    def test_execution(self, compiler, runtime, basics_source):
        """Test that each basics tutorial executes successfully without extra console noise."""
        assert basics_source.path.exists(), f"{basics_source.path.name} should exist"

        result = compile_and_execute(basics_source.content, compiler, runtime)
        assert result is not None, f"Execution should return a result for {basics_source.path.name}"
        assert result.success is not None, f"Result should have success attribute for {basics_source.path.name}"
        assert result.success, (
            f"Basics tutorial execution failed with system (IR mode): "
            f"{result.get_errors() if hasattr(result, 'get_errors') else result.errors}"
        )
    

if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])
