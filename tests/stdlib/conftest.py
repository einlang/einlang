"""
Pytest configuration and fixtures for stdlib tests.

Provides shared fixtures for compiler and runtime instances.
"""

import pytest
from ..test_utils import compile_and_execute as util_compile_and_execute


@pytest.fixture
def compile_and_execute(compiler, runtime):
    """
    Convenience fixture that provides a pre-configured compile_and_execute function.
    
    Usage:
        def test_something(compile_and_execute):
            result = compile_and_execute("let x = 5; x")
            assert result.success
    """
    def _execute(source_code):
        return util_compile_and_execute(source_code, compiler, runtime)
    return _execute

