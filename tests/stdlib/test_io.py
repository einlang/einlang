#!/usr/bin/env python3
"""
Tests for std::io module functionality with string source examples.
Tests execute and check results to ensure complete IO stdlib coverage.
"""

import pytest
from ..test_utils import compile_and_execute
from einlang.shared.errors import EinlangSourceError


class TestIOModule:
    """Complete std::io coverage with execution validation"""
    
    def _test_and_execute(self, source: str, compiler, runtime, expected_result=None) -> dict:
        """Execute source and return result for checking"""
        execution_result = compile_and_execute(source, compiler, runtime)
        assert execution_result.success, f"Execution failed: {execution_result.errors}"
        
        result = execution_result.value
        if expected_result is not None:
            assert result == expected_result
        return result
    
    def test_print_operations(self, compiler, runtime):
        """Test print operations - concatenated for speed"""
        source = '''
        use std::io; print("Hello, World!");
        use std::io; let io0 = 42; print("Value:", io0);
        use std::io; let io1 = 1; let io2 = 2; print(io1, io2);
        use std::io; print();
        '''
        self._test_and_execute(source, compiler, runtime)
    
    def test_print_statements_builtin(self, compiler, runtime):
        """Test built-in print - concatenated for speed"""
        source = '''
        print("hello");
        let io3 = 42; print("value:", io3);
        print();
        let io4 = 1; let io5 = 2; let io6 = 3; print(io4, io5, io6);
        '''
        self._test_and_execute(source, compiler, runtime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
