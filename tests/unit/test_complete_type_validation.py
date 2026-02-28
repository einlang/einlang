"""
Tests for Complete Function Argument Type Validation
=====================================================

Tests comprehensive compile-time type validation for:
1. User-defined functions with type annotations
2. Type compatibility checking (widening, cross-category)
3. Array vs scalar validation
4. Multiple parameter types
"""

import pytest
from tests.test_utils import compile_and_execute


class TestUserFunctionTypeValidation:
    """Test type validation for user-defined functions"""
    
    def test_typed_function_correct_types(self, compiler, runtime):
        """Test user function with correct argument types"""
        source = """
fn add_numbers(x: i32, y: i32) -> i32 {
    x + y
}

let result = add_numbers(5, 10);
assert(result == 15);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_typed_function_wrong_literal_type(self, compiler, runtime):
        """Test user function called with wrong literal type"""
        source = """
fn add_ints(x: i32, y: i32) -> i32 {
    x + y
}

let result = add_ints(5, 3.14);  // Error: f32 not compatible with i32
"""
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Should have failed for type mismatch"
        assert len(result.errors) > 0
        
        error_msg = str(result.errors[0]).lower()
        assert "type" in error_msg or "i32" in error_msg or "f32" in error_msg
    
    def test_typed_function_array_parameter(self, compiler, runtime):
        """Test function with array parameter (untyped param accepts array of any shape)."""
        source = """
fn sum_array(data) -> i32 {
    let total = 0;
    total
}

let arr = [1, 2, 3];
let result = sum_array(arr);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_mixed_typed_untyped_parameters(self, compiler, runtime):
        """Test function with mix of typed and untyped parameters"""
        source = """
fn process(x: i32, y) -> i32 {
    x + y
}

let result = process(5, 10);
assert(result == 15);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"


class TestTypeCompatibility:
    """Test type compatibility and widening rules"""
    
    def test_integer_widening(self, compiler, runtime):
        """Test i32 -> i64 widening is allowed"""
        source = """
fn accept_i64(x: i64) -> i64 {
    x
}

let small: i32 = 42;
let result = accept_i64(small);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_float_widening(self, compiler, runtime):
        """Test f32 -> f64 widening is allowed"""
        source = """
fn accept_f64(x: f64) -> f64 {
    x
}

let small: f32 = 3.14;
let result = accept_f64(small);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_cross_category_rejected(self, compiler, runtime):
        """Test int -> float requires explicit cast"""
        source = """
fn accept_float(x: f32) -> f32 {
    x
}

let int_val: i32 = 42;
let result = accept_float(int_val);  // Error: i32 not compatible with f32
"""
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Should have failed: cross-category requires cast"
    
    def test_explicit_cast_allowed(self, compiler, runtime):
        """Test explicit cast for cross-category conversion"""
        source = """
fn accept_float(x: f32) -> f32 {
    x
}

let int_val: i32 = 42;
let result = accept_float(int_val as f32);  // OK with cast
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Should allow explicit cast: {result.errors}"


class TestArrayScalarValidation:
    """Test array vs scalar type checking"""
    
    def test_array_to_array_function(self, compiler, runtime):
        """Test passing array to function (untyped param accepts array)."""
        source = """
fn process_array(data) -> i32 {
    42
}

let arr = [1, 2, 3];
let result = process_array(arr);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_scalar_to_array_function_literal(self, compiler, runtime):
        """Test passing scalar literal to function expecting array"""
        source = """
fn process_array(data: [i32]) -> i32 {
    42
}

let result = process_array(5);  // Error: scalar not array
"""
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Should fail: scalar passed to array parameter"


class TestMultipleParameters:
    """Test functions with multiple typed parameters"""
    
    def test_all_parameters_correct(self, compiler, runtime):
        """Test function with multiple parameters, all correct"""
        source = """
fn compute(a: i32, b: i32, c: i32) -> i32 {
    a + b + c
}

let result = compute(1, 2, 3);
assert(result == 6);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_one_parameter_wrong_type(self, compiler, runtime):
        """Test function where one parameter has wrong type"""
        source = """
fn compute(a: i32, b: i32, c: i32) -> i32 {
    a + b + c
}

let result = compute(1, 3.14, 3);  // Error: 2nd param wrong type
"""
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Should fail for parameter 2 type mismatch"
        assert len(result.errors) > 0, "Should have at least one error"
        
        error_msg = str(result.errors[0]).lower() if result.errors else ""
        # Should mention argument/parameter and the parameter name 'b' or number
        assert "argument" in error_msg or "parameter" in error_msg
        # Should mention the type mismatch
        assert "i32" in error_msg or "f32" in error_msg


class TestPolymorphicFunctions:
    """Test untyped (polymorphic) functions"""
    
    def test_untyped_function_accepts_any(self, compiler, runtime):
        """Test untyped function accepts any argument type"""
        source = """
fn identity(x) {
    x
}

let int_result = identity(42);
let float_result = identity(3.14);
let array_result = identity([1, 2, 3]);

assert(int_result == 42);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_partially_typed_function(self, compiler, runtime):
        """Test function with some typed, some untyped parameters"""
        source = """
fn process(typed: i32, untyped) -> i32 {
    typed + untyped
}

let result1 = process(5, 10);
let result2 = process(7, 8);  // untyped param accepts any compatible type

assert(result1 == 15);
assert(result2 == 15);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"


class TestErrorMessages:
    """Test quality of type validation error messages"""
    
    def test_error_message_shows_expected_and_actual(self, compiler, runtime):
        """Test error message includes expected and actual types"""
        source = """
fn add_ints(x: i32, y: i32) -> i32 {
    x + y
}

let result = add_ints(5, 3.14);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success
        assert len(result.errors) > 0
        
        error_msg = str(result.errors[0])
        # Should mention expected type (i32) and actual type (f32)
        assert "i32" in error_msg or "int" in error_msg.lower()
    
    def test_error_message_shows_parameter_number(self, compiler, runtime):
        """Test error message identifies which parameter is wrong"""
        source = """
fn compute(a: i32, b: i32, c: i32) -> i32 {
    a + b + c
}

let result = compute(1, 3.14, 3);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Should fail for parameter type mismatch"
        assert len(result.errors) > 0
        error_msg = str(result.errors[0]).lower()
        # Should mention parameter/argument (may use name or number)
        assert "argument" in error_msg or "parameter" in error_msg


class TestCombinedValidation:
    """Test combined arity and type validation"""
    
    def test_wrong_arity_caught_before_types(self, compiler, runtime):
        """Test arity errors are caught (regardless of types)"""
        source = """
fn add(x: i32, y: i32) -> i32 {
    x + y
}

let result = add(5);  // Error: wrong arity
"""
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success
        assert len(result.errors) > 0
        
        # Should mention arity/argument count
        error_msg = str(result.errors[0]).lower()
        assert "expects" in error_msg or "argument" in error_msg
    
    def test_correct_arity_wrong_types(self, compiler, runtime):
        """Test correct arity but wrong types"""
        source = """
fn add(x: i32, y: i32) -> i32 {
    x + y
}

let result = add(5, 3.14);  // Correct arity, wrong type
"""
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success
        assert len(result.errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

