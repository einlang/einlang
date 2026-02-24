"""
Tests for function existence checking in ModulePass.

Verifies that ModulePass now catches undefined functions at compile-time
instead of letting them fail at runtime.
"""

import pytest
from pathlib import Path


class TestFunctionExistenceValidation:
    """Test that ModulePass validates function existence"""
    
    def test_valid_function_call_passes(self, compiler, runtime):
        """Verify that valid function calls pass validation"""
        code = """
        use std::math::sin;
        let x = sin(1.0);
        """
        
        result = compiler.compile(code, source_file="test.ein")
        assert result.success, f"Compilation failed: {result.get_errors()}"
    
    def test_undefined_function_caught_at_compile_time(self, compiler, runtime):
        """Verify undefined function is caught at compile-time"""
        code = """
        use std::math;
        let x = math::nonexistent_function(1.0);
        """
        
        result = compiler.compile(code, source_file="test.ein")
        
        # Should fail compilation
        assert not result.success, "Expected compilation to fail for undefined function"
        
        # Should have error about undefined function (error about undefined function)
        error_messages = [str(err) for err in result.get_errors()]
        error_str = " ".join(error_messages).lower()
        assert "not found" in error_str or "resolve" in error_str or "nonexistent" in error_str, \
            f"Expected undefined function error, got: {error_messages}"
    
    def test_private_function_not_accessible(self, compiler, runtime):
        """Verify private functions are not accessible (uses source_overlay, no file I/O)"""
        main_source = """
use crate::my_module;
let x = my_module::private_func();
"""
        my_module_source = """
fn private_func() { 42 }
pub fn public_func() { 100 }
"""
        source_overlay = {("my_module",): my_module_source}
        result = compiler.compile(
            main_source,
            source_file="main.ein",
            root_path=Path.cwd(),
            source_overlay=source_overlay,
        )
        
        # Should fail due to private function access
        assert not result.success, "Expected compilation to fail for private function access"
        assert len(result.get_errors()) > 0
        error_messages = [str(err) for err in result.get_errors()]
        assert any("private" in msg.lower() for msg in error_messages), \
            f"Expected 'private' error, got: {error_messages}"
    
    def test_builtin_function_still_works(self, compiler, runtime):
        """Verify builtin functions still work without imports"""
        code = """
        let arr = [1, 2, 3, 4, 5];
        let total = sum(arr);
        """
        
        result = compiler.compile(code, source_file="test.ein")
        
        # Should compile successfully
        assert result.success, f"Compilation failed: {result.get_errors()}"
    
    def test_multiple_undefined_functions_all_caught(self, compiler, runtime):
        """Verify multiple undefined functions are all caught"""
        code = """
        use std::math;
        let x = math::func1(1.0);
        let y = math::func2(2.0);
        let z = math::func3(3.0);
        """
        
        result = compiler.compile(code, source_file="test.ein")
        
        # Should fail compilation
        assert not result.success, "Expected compilation to fail"
        
        # Should have at least one error mentioning an undefined function (compiler may stop after first)
        error_messages = result.get_errors()
        error_str = " ".join(str(e) for e in error_messages)
        assert any(name in error_str for name in ("func1", "func2", "func3")), \
            f"Expected error mentioning at least one of func1/func2/func3, got: {error_str[:500]}"

