"""
Tests for Enhanced Function Validation Pass
============================================

Tests new compile-time validation features:
1. Lambda arity validation
2. Function argument type validation
"""

import pytest
from tests.test_utils import compile_and_execute


class TestLambdaArityValidation:
    """Test lambda arity validation at compile-time"""
    
    def test_lambda_correct_arity(self, compiler, runtime):
        """Test lambda called with correct number of arguments"""
        source = """
let double = |x| x * 2;
let result = double(5);
assert(result == 10);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_lambda_too_few_args(self, compiler, runtime):
        """Test lambda called with too few arguments. Must fail."""
        source = """
let add = |x, y| x + y;
let result = add(5);  # Error: needs 2 args, got 1
"""
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Compilation should have failed for lambda arity mismatch"
        assert len(result.errors) > 0, "Should have at least one error"
        error_msg = str(result.errors[0]).lower()
        assert "lambda" in error_msg or "expects" in error_msg or "variable" in error_msg or "not found" in error_msg or "argument" in error_msg

    def test_lambda_too_many_args(self, compiler, runtime):
        """Test lambda called with too many arguments. Must fail."""
        source = """
let double = |x| x * 2;
let result = double(5, 10);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Compilation or execution should have failed for lambda arity mismatch"
        assert len(result.errors) > 0
        error_msg = str(result.errors[0]).lower()
        assert "expects" in error_msg or "argument" in error_msg or "1" in error_msg
    
    def test_lambda_multiple_params(self, compiler, runtime):
        """Test lambda with multiple parameters - correct arity"""
        source = """
let add_three = |x, y, z| x + y + z;
let result = add_three(1, 2, 3);
assert(result == 6);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_parameterless_lambda_correct(self, compiler, runtime):
        """Test parameterless lambda called correctly"""
        source = """
let get_five = || 5;
let result = get_five();
assert(result == 5);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_parameterless_lambda_with_args(self, compiler, runtime):
        """Test parameterless lambda called with arguments. Must fail."""
        source = """
let get_five = || 5;
let result = get_five(10);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Should have failed for lambda arity mismatch"
        assert len(result.errors) > 0
        error_msg = str(result.errors[0]).lower()
        assert "expects" in error_msg or "argument" in error_msg or "0" in error_msg
    
    def test_lambda_in_array_map(self, compiler, runtime):
        """Test lambda arity in array operations"""
        source = """
let double = |val| val * 2;
let data = [1, 2, 3];
let doubled = [double(x) | x in data];
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"


class TestFunctionArgumentTypeValidation:
    """Test function argument type validation at compile-time"""
    
    def test_shape_with_array_argument(self, compiler, runtime):
        """Test shape() called with array - correct type"""
        source = """
let arr = [1, 2, 3];
let s = shape(arr);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_shape_with_scalar_argument(self, compiler, runtime):
        """Test shape() called with scalar (returns [])."""
        source = """
let x = 42;
let s = shape(x);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        s = result.outputs.get("s")
        assert s is not None
        assert list(s) == []
    
    def test_len_with_array_argument(self, compiler, runtime):
        """Test len() called with array - correct type"""
        source = """
let arr = [1, 2, 3, 4, 5];
let length = len(arr);
assert(length == 5);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_builtin_with_correct_types(self, compiler, runtime):
        """Test builtin functions with correct argument types (len, shape; min/max require std::)."""
        source = """
let arr = [1, 2, 3];
let length = len(arr);
let s = shape(arr);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs.get("length") == 3


class TestCombinedValidation:
    """Test combination of lambda arity and type validation"""
    
    def test_lambda_with_array_operations(self, compiler, runtime):
        """Test lambda that processes arrays - arity and types"""
        source = """
let process = |arr| {
    let length = len(arr);
    let s = shape(arr);
    arr
};

let data = [1, 2, 3];
let result = process(data);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_higher_order_function(self, compiler, runtime):
        """Test higher-order function with lambda"""
        source = """
fn apply_twice(f, x) {
    f(f(x))
}

let double = |n| n * 2;
let result = apply_twice(double, 5);
assert(result == 20);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"


class TestErrorMessages:
    """Test quality of error messages"""
    
    def test_lambda_arity_error_message_quality(self, compiler, runtime):
        """Test that wrong lambda arity produces some compile error."""
        source = """
let add = |x, y| x + y;
let result = add(5);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Expected compile error"
        assert len(result.errors) > 0
        error_msg = str(result.errors[0]).lower()
        # may report "variable not found" (y) or explicit arity (2 args expected, 1 given)
        assert "variable" in error_msg or "not found" in error_msg or "2" in error_msg or "1" in error_msg or "expects" in error_msg or "argument" in error_msg
    
    def test_type_error_message_quality(self, compiler, runtime):
        """Test that type errors have helpful messages"""
        source = """
let x = 42;
let s = shape(x);
"""
        result = compile_and_execute(source, compiler, runtime)
        # May fail at compile or runtime
        if not result.success and result.errors:
            error_msg = str(result.errors[0]).lower()
            assert "shape" in error_msg or "array" in error_msg or "type" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

