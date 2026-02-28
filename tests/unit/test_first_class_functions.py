"""
Tests for lambda and function passed as argument (first-class functions).
"""

import pytest
from tests.test_utils import compile_and_execute


class TestLambdaPassedAsArg:
    """Lambda passed as argument and invoked."""

    def test_lambda_passed_and_called(self, compiler, runtime):
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

    def test_inline_lambda_passed_as_arg(self, compiler, runtime):
        source = """
fn apply_twice(f, x) { f(f(x)) }
let result = apply_twice(|n| n * 2, 5);
assert(result == 20);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"

    def test_lambda_variable_passed_as_arg(self, compiler, runtime):
        source = """
fn call_with(f, x) { f(x) }
let inc = |n| n + 1;
let r = call_with(inc, 10);
assert(r == 11);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"


class TestNamedFnPassedAsArg:
    """Named function (fn) passed as argument and invoked."""

    def test_named_fn_passed_and_called(self, compiler, runtime):
        source = """
fn add(a, b) { a + b }
fn apply_binary(f, x, y) { f(x, y) }
let result = apply_binary(add, 3, 5);
assert(result == 8);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"

    def test_named_fn_passed_multiple_times(self, compiler, runtime):
        source = """
fn mul(a, b) { a * b }
fn apply_binary(f, x, y) { f(x, y) }
let r1 = apply_binary(mul, 2, 3);
let r2 = apply_binary(mul, 4, 5);
assert(r1 == 6);
assert(r2 == 20);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"

    def test_different_named_fns_passed(self, compiler, runtime):
        source = """
fn add(a, b) { a + b }
fn sub(a, b) { a - b }
fn apply_binary(f, x, y) { f(x, y) }
assert(apply_binary(add, 10, 3) == 13);
assert(apply_binary(sub, 10, 3) == 7);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"


class TestFirstClassComposition:
    """First-class functions: store in variable, pass, call."""

    def test_fn_in_var_then_passed(self, compiler, runtime):
        source = """
fn id(x) { x }
fn apply_once(f, x) { f(x) }
let f = id;
let r = apply_once(f, 42);
assert(r == 42);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"

    def test_two_callables_passed_and_called(self, compiler, runtime):
        source = """
fn apply_pair(f, g, x) { f(x) + g(x) }
let inc = |n| n + 1;
let double = |n| n * 2;
let r = apply_pair(inc, double, 10);
assert(r == 31);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"

    @pytest.mark.skip(reason="Comprehension inside first-class call: Non-lowered IR at runtime (E0007)")
    def test_lambda_in_comprehension_via_first_class(self, compiler, runtime):
        source = """
fn map_arr(f, arr) {
    [f(x) | x in arr]
}
let double = |x| x * 2;
let data = [1, 2, 3];
let out = map_arr(double, data);
assert(out[0] == 2);
assert(out[1] == 4);
assert(out[2] == 6);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"

    def test_chain_first_class_calls(self, compiler, runtime):
        source = """
fn apply(f, x) { f(x) }
fn inc(n) { n + 1 }
let r = apply(inc, apply(inc, 0));
assert(r == 2);
"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
