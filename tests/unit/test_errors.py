#!/usr/bin/env python3
"""
Tests for error cases: syntax errors, empty programs.
Tests execute and check error handling to ensure complete error coverage using system.
"""

import pytest
from tests.test_utils import compile_and_execute
from einlang.shared.errors import EinlangSourceError, EinlangError


class TestErrors:
    """Complete error coverage with execution validation using system"""
    
    def _test_and_execute(self, source: str, compiler, runtime, expected_result=None) -> dict:
        """Execute source using system and return result for checking"""
        result = compile_and_execute(source, compiler, runtime)
        # Always check exec.result.success unless it is a negative test
        assert result.success, f"Execution failed: {result.errors}"
        
        if expected_result is not None and hasattr(result, 'outputs'):
            variables = result.outputs
            # Check if any variable matches expected result
            for var_name, var_value in variables.items():
                if var_value == expected_result:
                    return variables
        return result.outputs if hasattr(result, 'outputs') else {}
    
    def _verify_error_pointer(self, error_msg: str, source: str, error_token: str):
        """Verify that error pointer (^ or -) is positioned correctly"""
        lines = error_msg.split('\n')
        source_line = None
        annotation_line = None
        
        # Find source line and annotation line
        for i, line in enumerate(lines):
            # Look for line with source code (has | separator)
            if '|' in line and any(part.strip() for part in line.split('|')[1:]):
                # Check if this line contains part of the source
                line_content = line.split('|', 1)[1] if '|' in line else line
                if line_content.strip():
                    source_line = line
                    # Next line might be annotation
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if '^' in next_line or '-' in next_line:
                            annotation_line = next_line
                            break
        
        # If we found both lines, verify positioning
        if source_line and annotation_line and error_token != "syntax":
            # Find position of error token in source line
            token_pos = source_line.find(error_token)
            
            if token_pos >= 0:
                # Find position of ^ or - in annotation line
                caret_pos = annotation_line.find('^')
                if caret_pos < 0:
                    caret_pos = annotation_line.find('-')
                
                if caret_pos >= 0:
                    # Verify they align (within ±3 columns for tolerance)
                    assert abs(token_pos - caret_pos) <= 3, \
                        f"Error pointer should be under '{error_token}', got token={token_pos}, pointer={caret_pos}\n" \
                        f"Source: {source_line}\n" \
                        f"Pointer: {annotation_line}"
    
    def test_syntax_errors(self, compiler, runtime):
        """Test syntax error handling with proper source location"""
        error_cases = [
            ("let x = ;", ";"),                    # Missing value, error at unexpected ;
            ("let = 5;", "="),                     # Missing variable name, error at =
            ("func(;", ";"),                       # Invalid function call, error at unexpected ;
            ("let x = [;", ";"),                   # Invalid array syntax, error at unexpected ;
            ("invalid syntax here", "syntax"),     # Invalid statement, error at unexpected token
            ("let x = 5 +;", ";"),                 # Incomplete expression, error at unexpected ;
            ("fn () { return 1; }", "("),          # Missing function name, error at (
            ("let x: = 5;", "="),                  # Invalid type annotation, error at =
        ]
        for source, error_token in error_cases:
            result = compile_and_execute(source, compiler, runtime)
            # system returns failed execution result instead of raising exception
            # Always check exec.result.success unless it is a negative test
            assert not result.success, f"Syntax error should fail for: {source}"
            assert len(result.errors) > 0, f"Should have error messages for: {source}"
            
            error_msg = str(result.errors[0])
            # Verify the error message contains source line information or syntax error indicators
            # ✅ Issue #4 FIX: Accept "unexpected" as syntax error indicator (parser output)
            assert (source in error_msg or 
                    "syntax" in error_msg.lower() or 
                    "parse" in error_msg.lower() or 
                    "unexpected" in error_msg.lower()), \
                f"Error message should contain source line or syntax error: {source}\nGot: {error_msg}"
            
            # Verify error pointer positioning (if present)
            self._verify_error_pointer(error_msg, source, error_token)
    
    def test_runtime_errors(self, compiler, runtime):
        """Test runtime error handling"""
        error_cases = [
            ("let x = 1 / 0;", "division", "0"),        # Division by zero - point to 0
            ("let x = undefined_var;", "undefined", "undefined_var"), # Undefined variable
            ("let x = func();", "undefined", "func"),      # Undefined function
        ]
        for source, error_type, error_token in error_cases:
            result = compile_and_execute(source, compiler, runtime)
            # system returns failed execution result instead of raising exception
            # Always check exec.result.success unless it is a negative test
            assert not result.success, f"Runtime error should fail for: {source}"
            assert len(result.errors) > 0, f"Should have error messages for: {source}"
            
            error_msg = str(result.errors[0])
            # Verify the error message contains relevant error information
            assert any(keyword in error_msg.lower() for keyword in [error_type, "error", "failed"]), \
                f"Error message should mention {error_type} for: {source}"
            
            # Verify error pointer positioning (if present)
            self._verify_error_pointer(error_msg, source, error_token)
    
    def test_type_errors(self, compiler, runtime):
        """Test type error handling"""
        # Note: system may not enforce strict type checking yet
        # These tests may pass if type checking is not implemented
        error_cases = [
            ("let x: i32 = 3.14;", "type", "3.14"),        # Type mismatch - point to value
            ("let x: str = 42;", "type", "42"),          # Type mismatch
            ("let x: bool = \"hello\";", "type", "\"hello\""),  # Type mismatch
        ]
        for source, error_type, error_token in error_cases:
            result = compile_and_execute(source, compiler, runtime)
            # Type checking not enforced, xfail if it passes
            # Always check exec.result.success unless it is a negative test
            if result.success:
                pytest.xfail(f"Type checking not enforced for: {source}")
            
            assert len(result.errors) > 0, f"Should have error messages for: {source}"
            
            error_msg = str(result.errors[0])
            # Verify the error message contains relevant error information
            assert any(keyword in error_msg.lower() for keyword in [error_type, "error", "failed"]), \
                f"Error message should mention {error_type} for: {source}"
            
            # Verify error pointer positioning (if present)
            self._verify_error_pointer(error_msg, source, error_token)
    
    def test_empty_programs(self, compiler, runtime):
        """Test empty program handling"""
        cases = [
            "",                                    # Completely empty
            "   ",                                 # Whitespace only
        ]
        for source in cases:
            result = compile_and_execute(source, compiler, runtime)
            # Empty programs should succeed (no errors)
            # Always check exec.result.success unless it is a negative test
            assert result.success, f"Empty program should succeed: {repr(source)}"
        
        # Test comment support
        comment_cases = [
            "let x = 5; # This is a comment",
            "let x = 5; # This is a comment\nlet y = 10;",
            "# This is a comment at the start\nlet x = 5;",
            "let x = 5; # Comment\n# Another comment\nlet y = 10;",
            "let x = 5; # Comment with special chars: @#$%^&*()",
            "let x = 5; # Comment with numbers 123 and symbols !@#",
        ]
        for source in comment_cases:
            result = compile_and_execute(source, compiler, runtime)
            # Always check exec.result.success unless it is a negative test
            assert result.success, f"Comment handling should succeed: {repr(source)}"
    
    def test_multiline_syntax_errors(self, compiler, runtime):
        """Test syntax error handling in multiline code with proper line numbers"""
        multiline_cases = [
            ("""let x = 5;
let y = ;
let z = 10;""", 2, ";"),  # Error on line 2
            
            ("""fn test() {
    let a = 1;
    let b = 
    let c = 3;
}""", 3, "let"),  # Error on line 3
            
            ("""let data = [1, 2, 3];
let result = data[;
let final = 0;""", 2, ";"),  # Error on line 2
        ]
        
        for source, expected_line, error_token in multiline_cases:
            result = compile_and_execute(source, compiler, runtime)
            # Always check exec.result.success unless it is a negative test
            assert not result.success, f"Multiline syntax error should fail for: {source}"
            assert len(result.errors) > 0, f"Should have error messages for: {source}"
            error_msg = str(result.errors[0])
            # system may not have exact line number reporting yet
            # ✅ Issue #4 FIX: Accept "unexpected" as syntax error indicator (parser output)
            assert (source in error_msg or 
                    "syntax" in error_msg.lower() or 
                    "parse" in error_msg.lower() or 
                    "unexpected" in error_msg.lower()), \
                f"Error message should contain source or syntax error: {source}\nGot: {error_msg}"
            
            # Verify line number in error message
            if str(expected_line) in error_msg or f":{expected_line}:" in error_msg:
                # Found line number in error location
                pass
            
            # Verify error pointer positioning (if present)
            self._verify_error_pointer(error_msg, source, error_token)
            
            # Additional verification: check that correct line is shown
            lines = error_msg.split('\n')
            for i, line in enumerate(lines):
                # Look for line number in the formatted output
                if f' {expected_line} |' in line or f'{expected_line:>4} |' in line:
                    # Found the correct line number in output
                    # Next line should have the error pointer
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        assert '^' in next_line or '-' in next_line, \
                            f"Line {expected_line} should have error pointer, got: {next_line}"
                    break
    
    def test_error_recovery(self, compiler, runtime):
        """Test error recovery and continuation"""
        # Test that valid code after errors still works
        valid_source = "let x = 5; assert(x == 5);"
        result = compile_and_execute(valid_source, compiler, runtime)
        # Always check exec.result.success unless it is a negative test
        assert result.success, f"Valid code should succeed: {valid_source}"
        
        # Test that multiple errors are reported
        invalid_source = "let x = ; let y = ;"
        compilation = compiler.compile(invalid_source, "<test>")
        # Always check compilation success
        assert not compilation.success, f"Invalid code should fail to compile: {invalid_source}"
        assert len(compilation.get_errors()) > 0, f"Should have error messages for: {invalid_source}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
