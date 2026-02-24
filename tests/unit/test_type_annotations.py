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

