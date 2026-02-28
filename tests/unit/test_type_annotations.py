"""
Tests for type annotation compatibility and inference.

Tests the fixes for:
- Nested array literal flattening ([[1,2,3], [4,5,6]] → [i32; 2, 3])
- Dynamic rank support ([f32; *])
- Wildcard dimension support ([i32; ?, ?])
- Type object consistency (no string conversions)
"""

import pytest
from tests.test_utils import compile_and_execute, apply_ir_round_trip


class TestTypeAnnotations:
    """Test type annotation compatibility"""
    
    def test_nested_array_literal_flattening(self, compiler, runtime):
        """Test that nested array literals are correctly inferred as multi-dimensional arrays"""
        cases = [
            # 2D arrays
            ("let matrix: [i32; 2, 3] = [[1, 2, 3], [4, 5, 6]]; assert(matrix[0,0] == 1);", True),
            ("let matrix: [i32; 3, 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; assert(matrix[1,1] == 5);", True),
            
            # 3D arrays
            ("let tensor: [i32; 2, 2, 2] = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]; assert(tensor[0,0,0] == 1);", True),
            
            # Mismatch: wrong shape (strict checking)
            ("let matrix: [i32; 2, 2] = [[1, 2, 3], [4, 5, 6]];", False),
            ("let matrix: [i32; 3, 2] = [[1, 2, 3], [4, 5, 6]];", False),
        ]
        
        for source, should_succeed in cases:
            result = compile_and_execute(source, compiler, runtime, source_file='test.ein')
            if should_succeed:
                assert result.success, f"Expected success but got errors: {result.errors}"
            else:
                assert not result.success, f"Expected failure but compilation succeeded for: {source}"
    
    def test_dynamic_rank_arrays(self, compiler, runtime):
        """Test that [T; *] accepts arrays of any shape"""
        cases = [
            # 1D array
            "let arr: [f32; *] = [1.0, 2.0, 3.0]; assert(len(arr) == 3);",
            
            # 2D array
            "let matrix: [f32; *] = [[1.0, 2.0], [3.0, 4.0]]; assert(matrix[0,0] == 1.0);",
            
            # 3D array
            "let tensor: [i32; *] = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]; assert(tensor[1,0,0] == 5);",
        ]
        
        for source in cases:
            result = compile_and_execute(source, compiler, runtime, source_file='test.ein')
            assert result.success, f"Execution failed: {result.errors}"
    
    def test_wildcard_dimensions(self, compiler, runtime):
        """Test that [T; ?, ?] accepts any matching rank array"""
        cases = [
            # Wildcard accepts any size
            ("let matrix: [i32; ?, ?] = [[1, 2, 3], [4, 5, 6]]; assert(matrix[0,0] == 1);", True),
            ("let matrix: [i32; ?, ?] = [[1, 2], [3, 4], [5, 6]]; assert(matrix[1,1] == 4);", True),
            
            # Single wildcard
            ("let arr: [f32; ?] = [1.0, 2.0, 3.0]; assert(len(arr) == 3);", True),
            ("let arr: [f32; ?] = [1.0, 2.0]; assert(len(arr) == 2);", True),
            
            # Mixed concrete and wildcard
            ("let matrix: [i32; 2, ?] = [[1, 2, 3], [4, 5, 6]]; assert(matrix[1,2] == 6);", True),
            
            # Mismatch: wrong rank (strict checking)
            ("let matrix: [i32; ?, ?] = [1, 2, 3];", False),
            ("let arr: [i32; ?] = [[1, 2], [3, 4]];", False),
            
            # Mismatch: concrete dimension doesn't match
            ("let matrix: [i32; 2, ?] = [[1, 2], [3, 4], [5, 6]];", False),
        ]
        
        for source, should_succeed in cases:
            context = compiler.compile(source, source_file='test.ein')
            if should_succeed:
                assert context.success, f"Expected success but got errors: {context.get_errors()}"
                apply_ir_round_trip(context)
                result = runtime.execute(context)
                assert result.success, f"Execution failed: {result.errors}"
            else:
                assert not context.success, f"Expected failure but compilation succeeded for: {source}"
    
    def test_i64_literal_coercion_at_binding_only(self, compiler):
        """Without :i64, using a value in i64 context fails; literal at binding coerces."""
        ok = "let x: i64 = 42;"
        result = compiler.compile(ok, "<test>")
        assert result.success, f"let x: i64 = 42; should succeed (literal coercion): {result.get_errors()}"
        fail = "let n = 42; let x: i64 = n;"
        result_fail = compiler.compile(fail, "<test>")
        assert not result_fail.success, "let n = 42; let x: i64 = n; should fail (no variable coercion to i64)"
        errors = " ".join(str(e) for e in result_fail.get_errors()).lower()
        assert "mismatch" in errors or "expected" in errors or "i64" in errors or "e0308" in errors

    def test_i32_power_overflow_fails(self, compiler):
        """let x = 2**32; should fail (result does not fit in i32)."""
        result = compiler.compile("let x = 2**32;", "<test>")
        assert not result.success, "let x = 2**32; should fail (i32 overflow)"
        errors = " ".join(str(e) for e in result.get_errors()).lower()
        assert "overflow" in errors or "does not fit" in errors or "e002" in errors

    def test_default_precision_literal_overflow_fails(self, compiler):
        """Default precision is fixed (i32); literal that does not fit i32 fails without annotation."""
        result = compiler.compile("let x = 4294967296;", "<test>")
        assert not result.success, "let x = 4294967296; should fail (default i32, no value-based promotion)"
        errors = " ".join(str(e) for e in result.get_errors()).lower()
        assert "overflow" in errors or "does not fit" in errors or "e002" in errors

    def test_i64_power_overflow_succeeds(self, compiler, runtime):
        """let x: i64 = 2**32; should pass and x equals 2**32 (no huge literal; default precision is i32)."""
        source = "let x: i64 = 2**32; let y: i64 = 2**32; assert(x == y);"
        result = compile_and_execute(source, compiler, runtime, source_file="<test>")
        assert result.success, f"let x: i64 = 2**32; should succeed: {result.errors}"

    def test_default_precision_pass_and_overflow(self, compiler):
        """Default precision: i32 for int, f32 for float. Pass when value fits; overflow when it does not."""
        cases = [
            ("let x = 42;", True),
            ("let x = 2**32;", False),
            ("let x = 4294967296;", False),
            ("let x = 3.14;", True),
            ("let x = 1e50;", False),
        ]
        for source, should_pass in cases:
            result = compiler.compile(source, "<test>")
            if should_pass:
                assert result.success, f"Expected pass: {source!r} -> {result.get_errors()}"
            else:
                assert not result.success, f"Expected overflow fail: {source!r}"
                err = " ".join(str(e) for e in result.get_errors()).lower()
                assert "overflow" in err or "does not fit" in err or "e002" in err

    def test_explicit_i32_pass_and_overflow(self, compiler):
        """Explicit i32: pass when value fits, overflow when it does not."""
        cases = [
            ("let x: i32 = 42;", True),
            ("let x: i32 = 2**32;", False),
            ("let x: i32 = 4294967296;", False),
        ]
        for source, should_pass in cases:
            result = compiler.compile(source, "<test>")
            if should_pass:
                assert result.success, f"Expected pass: {source!r} -> {result.get_errors()}"
            else:
                assert not result.success, f"Expected overflow: {source!r}"
                err = " ".join(str(e) for e in result.get_errors()).lower()
                assert "overflow" in err or "does not fit" in err or "e002" in err

    def test_explicit_i64_pass_and_overflow(self, compiler):
        """Explicit i64: pass when value fits, overflow when it does not."""
        cases = [
            ("let x: i64 = 42;", True),
            ("let x: i64 = 2**32;", True),
            ("let x: i64 = 9223372036854775808;", False),
        ]
        for source, should_pass in cases:
            result = compiler.compile(source, "<test>")
            if should_pass:
                assert result.success, f"Expected pass: {source!r} -> {result.get_errors()}"
            else:
                assert not result.success, f"Expected overflow: {source!r}"
                err = " ".join(str(e) for e in result.get_errors()).lower()
                assert "overflow" in err or "does not fit" in err or "e002" in err

    def test_explicit_f32_pass_and_overflow(self, compiler):
        """Explicit f32: pass when value fits, overflow when it does not."""
        cases = [
            ("let x: f32 = 3.14;", True),
            ("let x: f32 = 1e50;", False),
        ]
        for source, should_pass in cases:
            result = compiler.compile(source, "<test>")
            if should_pass:
                assert result.success, f"Expected pass: {source!r} -> {result.get_errors()}"
            else:
                assert not result.success, f"Expected overflow: {source!r}"
                err = " ".join(str(e) for e in result.get_errors()).lower()
                assert "overflow" in err or "does not fit" in err or "e002" in err

    def test_explicit_f64_pass_and_overflow(self, compiler):
        """Explicit f64: pass when value fits, overflow when it does not."""
        cases = [
            ("let x: f64 = 3.14;", True),
            ("let x: f64 = 1e200;", True),
            ("let x: f64 = 1e400;", False),
        ]
        for source, should_pass in cases:
            result = compiler.compile(source, "<test>")
            if should_pass:
                assert result.success, f"Expected pass: {source!r} -> {result.get_errors()}"
            else:
                assert not result.success, f"Expected overflow: {source!r}"
                err = " ".join(str(e) for e in result.get_errors()).lower()
                assert "overflow" in err or "does not fit" in err or "e002" in err

    def test_cast_with_type_objects(self, compiler, runtime):
        """Test that cast expressions work with proper type enum objects (not strings)"""
        cases = [
            # Simple casts
            "let x = sum[i]([1.0, 2.0][i]) as f32; let y = x / 2.0; assert(y == 1.5);",
            "let x = sum[i]([1, 2, 3][i]) as i32; let y = x + 1; assert(y == 7);",
            
            # Cast in expressions
            "let data = [1.0, 2.0, 3.0]; let total = sum[i](data[i]) as f32; assert(total == 6.0);",
        ]
        
        for source in cases:
            result = compile_and_execute(source, compiler, runtime, source_file='test.ein')
            assert result.success, f"Execution failed: {result.errors}"
    
    def test_shape_dimension_normalization(self, compiler, runtime):
        """Test that shape dimensions are normalized to integers where possible"""
        cases = [
            # Literal integer dimensions
            "let matrix: [i32; 3, 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; assert(matrix[2,2] == 9);",
            
            # Type annotation matches inferred shape
            "let arr: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0]; assert(len(arr) == 5);",
        ]
        
        for source in cases:
            context = compiler.compile(source, source_file='test.ein')
            assert context.success, f"Failed to compile: {context.get_errors()}"
            apply_ir_round_trip(context)
            result = runtime.execute(context)
            assert result.success, f"Execution failed: {result.errors}"
    
    def test_type_widening_same_category(self, compiler, runtime):
        """Test Rust semantics: NO implicit conversions, ALL require explicit cast"""
        cases = [
            # Rust: Integer widening requires explicit cast
            ("let x: i32 = 42; let y: i64 = x;", False),  # ERROR in Rust
            ("let x: i32 = 42; let y: i64 = x as i64; assert(y == 42);", True),  # OK with cast
            
            # Rust: Float widening requires explicit cast
            ("let x: f32 = 3.14; let y: f64 = x;", False),  # ERROR in Rust
            ("let x: f32 = 3.14; let y: f64 = x as f64;", True),  # OK with cast
            
            # Rust: Array element type must match exactly
            ("let arr1: [i32; 3] = [1, 2, 3]; let arr2: [i64; 3] = arr1;", False),  # ERROR
            
            # Rust: Cross-category requires explicit cast
            ("let x: i32 = 42; let y: f64 = x;", False),  # ERROR
            ("let x: i32 = 42; let y: f64 = x as f64; assert(y == 42.0);", True),  # OK with cast
            
            # Rust: Narrowing requires explicit cast
            ("let x: i64 = 42; let y: i32 = x;", False),  # ERROR
            ("let x: i64 = 42; let y: i32 = x as i32; assert(y == 42);", True),  # OK with cast
            
            # Rust: Float narrowing requires explicit cast
            ("let x: f64 = 3.14; let y: f32 = x;", False),  # ERROR
            ("let x: f64 = 3.14; let y: f32 = x as f32;", True),  # OK with cast
        ]
        
        for source, should_succeed in cases:
            result = compile_and_execute(source, compiler, runtime, source_file='test.ein')
            if should_succeed:
                assert result.success, f"Expected success but got errors: {result.errors} for: {source}"
            else:
                assert not result.success, f"Expected failure but compilation succeeded for: {source}"

