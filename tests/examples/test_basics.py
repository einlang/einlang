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
        """Test that each basics tutorial executes successfully without extra console noise."""
        assert basics_file.exists(), f"{basics_file.name} should exist"

        with open(basics_file, 'r', encoding='utf-8') as f:
            content = f.read()

        result = compile_and_execute(content, compiler, runtime)
        assert result is not None, f"Execution should return a result for {basics_file.name}"
        assert result.success is not None, f"Result should have success attribute for {basics_file.name}"
        assert result.success, (
            f"Basics tutorial execution failed with system (IR mode): "
            f"{result.get_errors() if hasattr(result, 'get_errors') else result.errors}"
        )
    

if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])
