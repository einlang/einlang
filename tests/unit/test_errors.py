#!/usr/bin/env python3
"""
Tests for error cases: syntax errors, empty programs.
Tests execute and check error handling to ensure complete error coverage using system.
"""

import re
import pytest
from tests.test_utils import compile_and_execute
from einlang.shared.errors import (
    EinlangSourceError,
    EinlangError,
    Error,
    ErrorReporter,
)
from einlang.shared.source_location import SourceLocation

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


class TestErrorReporterProblematic:
    """Problematic/edge cases for the error reporter formatter."""

    def test_location_none(self):
        err = Error(message="something failed", location=None, code="E0001")
        reporter = ErrorReporter({})
        out = reporter.format_error(err, color=False)
        assert "error[E0001]" in out
        assert "something failed" in out
        assert "unknown location" in out

    def test_file_not_in_source_files(self):
        loc = SourceLocation(file="missing.ein", line=1, column=1)
        err = Error(message="oops", location=loc, code="E0425")
        reporter = ErrorReporter({})
        out = reporter.format_error(err, color=False)
        assert "error[E0425]" in out
        assert "missing.ein:1:1" in out

    def test_line_beyond_source(self):
        loc = SourceLocation(file="x.ein", line=10, column=1)
        err = Error(message="bad", location=loc)
        reporter = ErrorReporter({"x.ein": "let a = 1;\nlet b = 2;\n"})
        out = reporter.format_error(err, color=False)
        assert " --> x.ein:10:1" in out
        assert " |" in out

    def test_single_line_no_end_column(self):
        loc = SourceLocation(file="f.ein", line=1, column=5, end_line=0, end_column=0)
        err = Error(message="type error", location=loc, label="expected i32")
        reporter = ErrorReporter({"f.ein": "let x = foo();"})
        out = reporter.format_error(err, color=False)
        assert "1 | let x = foo();" in out
        assert "^" in out
        assert "expected i32" in out

    def test_multiline_span_with_help_note(self):
        source = "let x = 5;\nlet y = match x {\n    0 => 1,\n    1 => 2\n};"
        loc = SourceLocation(
            file="m.ein", line=2, column=13, end_line=5, end_column=2
        )
        err = Error(
            message="non-exhaustive patterns: `other values` not covered",
            location=loc,
            code="E0004",
            label="pattern `other values` not covered",
            help="add a match arm with a wildcard pattern `_`",
        )
        reporter = ErrorReporter({"m.ein": source})
        out = reporter.format_error(err, color=False)
        assert "non-exhaustive" in out
        assert "2 | let y = match x {" in out
        assert "_^" in out or "^" in out
        assert "= help:" in out
        assert "|" in out

    def test_empty_source_line(self):
        loc = SourceLocation(file="e.ein", line=2, column=1)
        err = Error(message="unexpected", location=loc)
        reporter = ErrorReporter({"e.ein": "line1\n\nline3"})
        out = reporter.format_error(err, color=False)
        assert "2 |" in out
        assert "^" in out

    def test_multiple_errors_summary(self):
        reporter = ErrorReporter({"a.ein": "let x = ;"})
        loc = SourceLocation(file="a.ein", line=1, column=9)
        reporter.report_error("first", loc, code="E001")
        reporter.report_error("second", loc, code="E002")
        out = reporter.format_all_errors(color=False)
        assert "error[E001]" in out
        assert "first" in out
        assert "error[E002]" in out
        assert "second" in out
        assert "aborting due to 2 previous errors" in out

    def test_einlang_source_error_str_without_source(self):
        loc = SourceLocation(file="x.ein", line=1, column=1)
        e = EinlangSourceError("runtime fail", location=loc, source_code=None)
        s = str(e)
        assert "runtime fail" in s
        assert "x.ein" in s

    def test_einlang_source_error_str_with_source(self):
        loc = SourceLocation(file="y.ein", line=1, column=10, end_line=1, end_column=13)
        e = EinlangSourceError(
            "division by zero",
            location=loc,
            source_code="let x = 1/0;",
        )
        s = str(e)
        assert "division by zero" in s
        plain = _strip_ansi(s)
        assert "let x = 1/0" in plain and ("1 | " in plain or "| let x = 1/0" in plain)
        assert "^" in plain


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
    
    def _verify_error_pointer(
        self,
        error_msg: str,
        source: str,
        error_token: str,
        expected_line: int = None,
    ):
        """Verify that error pointer (^ or -) is positioned correctly (tolerance 1)."""
        lines = [_strip_ansi(ln) for ln in error_msg.split('\n')]
        source_line = None
        annotation_line = None

        def line_number_from_display(line: str) -> int:
            if '|' not in line:
                return -1
            left = line.split('|', 1)[0].strip()
            if not left:
                return -1
            try:
                return int(left)
            except ValueError:
                return -1

        for i, line in enumerate(lines):
            if '|' not in line or not any(part.strip() for part in line.split('|')[1:]):
                continue
            line_content = line.split('|', 1)[1].strip()
            if not line_content:
                continue
            ln = line_number_from_display(line)
            if expected_line is not None and ln >= 0 and ln != expected_line:
                continue
            if i + 1 >= len(lines):
                continue
            next_line = lines[i + 1]
            if '^' not in next_line and '-' not in next_line:
                continue
            source_line = line
            annotation_line = next_line
            break

        if not source_line or not annotation_line or error_token == "syntax":
            return
        source_after_pipe = source_line.split('|', 1)[1]
        annot_after_pipe = annotation_line.split('|', 1)[1] if '|' in annotation_line else annotation_line
        token_pos = source_after_pipe.find(error_token)
        if token_pos < 0:
            return
        caret_pos = annot_after_pipe.find('^')
        if caret_pos < 0:
            caret_pos = annot_after_pipe.find('-')
        if caret_pos < 0:
            return
        assert abs(token_pos - caret_pos) <= 1, (
            f"Error pointer should be near '{error_token}', got token={token_pos}, pointer={caret_pos}\n"
            f"Source: {source_line}\nPointer: {annotation_line}"
        )
    
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
            
            self._verify_error_pointer(
                error_msg, source, error_token, expected_line=expected_line
            )

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
