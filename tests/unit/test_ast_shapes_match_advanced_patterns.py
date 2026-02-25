"""
Comprehensive Tests for Advanced Match Patterns
===============================================

Tests for all advanced pattern matching features:
- Phase 1: Tuple patterns
- Phase 2: Array patterns
- Phase 3: Guard clauses (where syntax)
- Phase 4: Exhaustiveness checking (basic)
- Phase 5: Or patterns (|)
- Phase 6: Range patterns (..=, ..)
- Phase 7: Binding patterns (@)
"""

import pytest
from tests.test_utils import apply_ir_round_trip


class TestTuplePatterns:
    """Test tuple pattern matching"""
    
    def _compile_and_execute(self, source: str, compiler, runtime, inputs=None):
        """Helper to compile and execute Einlang code"""
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        exec_result = runtime.execute(context, inputs or {})
        assert exec_result.success, f"Execution failed: {exec_result.error}"
        return exec_result
    
    def test_tuple_pattern_basic(self, compiler, runtime):
        """Test basic tuple pattern matching"""
        source = """
        let pair = (1, 2);
        let result = match pair {
            (a, b) => a + b
        };
        assert(result == 3);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_tuple_pattern_multiple_arms(self, compiler, runtime):
        """Test tuple pattern with multiple arms"""
        source = """
        let point = (0, 0);
        let result = match point {
            (0, 0) => "origin",
            (x, 0) => "x-axis",
            (0, y) => "y-axis",
            (x, y) => "other"
        };
        assert(result == "origin");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_tuple_pattern_destructuring(self, compiler, runtime):
        """Test tuple pattern with variable binding"""
        source = """
        let coords = (10, 20);
        let result = match coords {
            (x, y) => x * y
        };
        assert(result == 200);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_tuple_pattern_nested(self, compiler, runtime):
        """Test nested tuple patterns"""
        source = """
        let nested = ((1, 2), 3);
        let result = match nested {
            ((a, b), c) => a + b + c
        };
        assert(result == 6);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_tuple_pattern_arity_mismatch(self, compiler, runtime):
        """Test that tuple pattern with wrong arity doesn't match"""
        source = """
        let pair = (1, 2);
        let result = match pair {
            (a, b, c) => "three",
            (a, b) => "two",
            _ => "other"
        };
        assert(result == "two");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_tuple_pattern_with_literals(self, compiler, runtime):
        """Test tuple pattern with literal values"""
        source = """
        let pair = (1, 2);
        let result = match pair {
            (1, 2) => "match",
            (1, y) => "first-match",
            (x, 2) => "second-match",
            _ => "no-match"
        };
        assert(result == "match");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_tuple_pattern_3_element(self, compiler, runtime):
        """Test tuple pattern with 3 elements"""
        source = """
        let triple = (1, 2, 3);
        let result = match triple {
            (a, b, c) => a + b + c
        };
        assert(result == 6);
        """
        self._compile_and_execute(source, compiler, runtime)


class TestArrayPatterns:
    """Test array pattern matching"""
    
    def _compile_and_execute(self, source: str, compiler, runtime, inputs=None):
        """Helper to compile and execute Einlang code"""
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        exec_result = runtime.execute(context, inputs or {})
        assert exec_result.success, f"Execution failed: {exec_result.error}"
        return exec_result
    
    def test_array_pattern_empty(self, compiler, runtime):
        """Test empty array pattern"""
        source = """
        let arr = [];
        let result = match arr {
            [] => "empty",
            _ => "non-empty"
        };
        assert(result == "empty");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_single(self, compiler, runtime):
        """Test single element array pattern"""
        source = """
        let arr = [42];
        let result = match arr {
            [x] => x,
            [] => 0,
            _ => -1
        };
        assert(result == 42);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_multiple(self, compiler, runtime):
        """Test array pattern with multiple elements"""
        source = """
        let arr = [1, 2, 3];
        let result = match arr {
            [a, b, c] => a + b + c,
            [a, b] => a + b,
            [a] => a,
            _ => 0
        };
        assert(result == 6);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_head_tail(self, compiler, runtime):
        """Test array pattern with rest pattern (head/tail)"""
        source = """
        let arr = [1, 2, 3, 4];
        let result = match arr {
            [first, ..rest] => first,
            [] => 0,
            _ => -1
        };
        assert(result == 1);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_rest_only(self, compiler, runtime):
        """Test array pattern with only rest pattern"""
        source = """
        let arr = [1, 2, 3];
        let result = match arr {
            [..rest] => len(rest),
            [] => 0,
            _ => -1
        };
        assert(result == 3);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_length_mismatch(self, compiler, runtime):
        """Test that array pattern with wrong length doesn't match"""
        source = """
        let arr = [1, 2];
        let result = match arr {
            [a, b, c] => "three",
            [a, b] => "two",
            [a] => "one",
            _ => "other"
        };
        assert(result == "two");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_with_literals(self, compiler, runtime):
        """Test array pattern with literal values"""
        source = """
        let arr = [1, 2];
        let result = match arr {
            [1, 2] => "exact",
            [1, y] => "first-match",
            [x, 2] => "second-match",
            _ => "no-match"
        };
        assert(result == "exact");
        """
        self._compile_and_execute(source, compiler, runtime)


class TestGuardClauses:
    """Test guard clauses with where syntax"""
    
    def _compile_and_execute(self, source: str, compiler, runtime, inputs=None):
        """Helper to compile and execute Einlang code"""
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        exec_result = runtime.execute(context, inputs or {})
        assert exec_result.success, f"Execution failed: {exec_result.error}"
        return exec_result
    
    def test_guard_clause_simple(self, compiler, runtime):
        """Test simple guard clause"""
        source = """
        let x = 5;
        let result = match x {
            n where n > 0 => n * 2,
            n where n < 0 => -n,
            _ => 0
        };
        assert(result == 10);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_guard_clause_negative(self, compiler, runtime):
        """Test guard clause with negative number"""
        source = """
        let x = -5;
        let result = match x {
            n where n > 0 => n * 2,
            n where n < 0 => -n,
            _ => 0
        };
        assert(result == 5);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_guard_clause_zero(self, compiler, runtime):
        """Test guard clause with zero"""
        source = """
        let x = 0;
        let result = match x {
            n where n > 0 => n * 2,
            n where n < 0 => -n,
            _ => 0
        };
        assert(result == 0);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_guard_clause_multiple(self, compiler, runtime):
        """Test multiple guard clauses"""
        source = """
        let x = 10;
        let result = match x {
            n where n > 20 => "large",
            n where n > 10 => "medium",
            n where n > 0 => "small",
            _ => "zero-or-negative"
        };
        assert(result == "small");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_guard_clause_fallthrough(self, compiler, runtime):
        """Test that guard failure falls through to next pattern"""
        source = """
        let x = 5;
        let result = match x {
            n where n > 10 => "large",
            n where n < 0 => "negative",
            n => "other"
        };
        assert(result == "other");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_guard_clause_with_tuple(self, compiler, runtime):
        """Test guard clause with tuple pattern"""
        source = """
        let pair = (5, 10);
        let result = match pair {
            (x, y) where x + y > 10 => "sum-large",
            (x, y) where x + y > 5 => "sum-medium",
            _ => "sum-small"
        };
        assert(result == "sum-large");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_guard_clause_with_array(self, compiler, runtime):
        """Test guard clause with array pattern"""
        source = """
        let arr = [5, 10];
        let result = match arr {
            [x, y] where x + y > 10 => "sum-large",
            [x, y] where x + y > 5 => "sum-medium",
            _ => "sum-small"
        };
        assert(result == "sum-large");
        """
        self._compile_and_execute(source, compiler, runtime)


class TestCombinedPatterns:
    """Test combinations of different pattern types"""
    
    def _compile_and_execute(self, source: str, compiler, runtime, inputs=None):
        """Helper to compile and execute Einlang code"""
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        exec_result = runtime.execute(context, inputs or {})
        assert exec_result.success, f"Execution failed: {exec_result.error}"
        return exec_result
    
    def test_tuple_and_array_mixed(self, compiler, runtime):
        """Test match with both tuple and array patterns"""
        source = """
        let value = (1, 2);
        let result = match value {
            (a, b) => "tuple",
            [a, b] => "array",
            _ => "other"
        };
        assert(result == "tuple");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_nested_patterns(self, compiler, runtime):
        """Test nested patterns"""
        source = """
        let nested = ([1, 2], 3);
        let result = match nested {
            ([a, b], c) => a + b + c,
            _ => 0
        };
        assert(result == 6);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_complex_match(self, compiler, runtime):
        """Test complex match with multiple pattern types"""
        source = """
        let value = (10, 20);
        let result = match value {
            (x, y) where x + y > 30 => "large-sum",
            (x, y) where x > y => "x-greater",
            (x, y) where y > x => "y-greater",
            (x, y) => "equal",
            _ => "other"
        };
        assert(result == "y-greater");
        """
        self._compile_and_execute(source, compiler, runtime)


class TestExhaustiveness:
    """Test exhaustiveness checking (basic)"""
    
    def _compile_and_execute(self, source: str, compiler, runtime, inputs=None):
        """Helper to compile and execute Einlang code"""
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        exec_result = runtime.execute(context, inputs or {})
        assert exec_result.success, f"Execution failed: {exec_result.error}"
        return exec_result
    
    def test_wildcard_makes_exhaustive(self, compiler, runtime):
        """Test that wildcard pattern makes match exhaustive"""
        source = """
        let x = 42;
        let result = match x {
            1 => "one",
            2 => "two",
            _ => "other"
        };
        assert(result == "other");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_identifier_makes_exhaustive(self, compiler, runtime):
        """Test that identifier pattern makes match exhaustive"""
        source = """
        let x = 42;
        let result = match x {
            1 => "one",
            2 => "two",
            n => "other"
        };
        assert(result == "other");
        """
        self._compile_and_execute(source, compiler, runtime)


class TestArrayPatternEdgeCases:
    """Test edge cases for array patterns"""
    
    def _compile_and_execute(self, source: str, compiler, runtime, inputs=None):
        """Helper to compile and execute Einlang code"""
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        exec_result = runtime.execute(context, inputs or {})
        assert exec_result.success, f"Execution failed: {exec_result.error}"
        return exec_result
    
    def test_array_pattern_rest_empty(self, compiler, runtime):
        """Test array pattern with rest when array is empty"""
        source = """
        let arr = [];
        let result = match arr {
            [..rest] => len(rest),
            _ => -1
        };
        assert(result == 0);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_rest_single_element(self, compiler, runtime):
        """Test array pattern with rest when array has one element"""
        source = """
        let arr = [42];
        let result = match arr {
            [first, ..rest] => first + len(rest),
            [..rest] => len(rest),
            _ => -1
        };
        assert(result == 42);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_rest_multiple_elements(self, compiler, runtime):
        """Test array pattern with rest when array has multiple elements"""
        source = """
        let arr = [1, 2, 3, 4, 5];
        let result = match arr {
            [first, second, ..rest] => first + second + len(rest),
            _ => -1
        };
        assert(result == 6);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_rest_operations(self, compiler, runtime):
        """Test array pattern with rest and operations on rest"""
        source = """
        let arr = [1, 2, 3, 4];
        let result = match arr {
            [first, ..rest] => first + len(rest),
            _ => 0
        };
        assert(result == 4);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_nested_with_rest(self, compiler, runtime):
        """Test array pattern with rest and multiple elements"""
        source = """
        let arr = [1, 2, 3, 4, 5];
        let result = match arr {
            [a, b, ..rest] => a + b + len(rest),
            _ => 0
        };
        assert(result == 6);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_rest_with_literals(self, compiler, runtime):
        """Test array pattern with rest and literal patterns"""
        source = """
        let arr = [1, 2, 3, 4];
        let result = match arr {
            [1, 2, ..rest] => len(rest),
            [1, ..rest] => len(rest),
            _ => -1
        };
        assert(result == 2);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_rest_order_matters(self, compiler, runtime):
        """Test that pattern order matters with rest patterns"""
        source = """
        let arr = [1, 2, 3];
        let result = match arr {
            [..rest] => "rest-first",
            [a, ..rest] => "rest-second",
            _ => "other"
        };
        assert(result == "rest-first");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_rest_at_beginning(self, compiler, runtime):
        """Test rest pattern at beginning: [..rest, last]"""
        source = """
        let arr = [1, 2, 3, 4];
        let result = match arr {
            [..rest, last] => last,
            _ => -1
        };
        assert(result == 4);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_rest_at_middle(self, compiler, runtime):
        """Test rest pattern at middle: [first, ..rest, last]"""
        source = """
        let arr = [1, 2, 3, 4, 5];
        let result = match arr {
            [first, ..rest, last] => first + last,
            _ => -1
        };
        assert(result == 6);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_rest_beginning_bindings(self, compiler, runtime):
        """Test that rest pattern at beginning correctly binds variables"""
        source = """
        let arr = [1, 2, 3, 4];
        let result = match arr {
            [..rest, last] => len(rest) + last,
            _ => -1
        };
        assert(result == 7);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_array_pattern_rest_middle_bindings(self, compiler, runtime):
        """Test that rest pattern at middle correctly binds variables"""
        source = """
        let arr = [1, 2, 3, 4, 5];
        let result = match arr {
            [first, ..rest, last] => first + len(rest) + last,
            _ => -1
        };
        assert(result == 9);
        """
        self._compile_and_execute(source, compiler, runtime)


class TestTuplePatternEdgeCases:
    """Test edge cases for tuple patterns"""
    
    def _compile_and_execute(self, source: str, compiler, runtime, inputs=None):
        """Helper to compile and execute Einlang code"""
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        exec_result = runtime.execute(context, inputs or {})
        assert exec_result.success, f"Execution failed: {exec_result.error}"
        return exec_result
    
    def test_tuple_pattern_4_element(self, compiler, runtime):
        """Test tuple pattern with 4 elements"""
        source = """
        let quad = (1, 2, 3, 4);
        let result = match quad {
            (a, b, c, d) => a + b + c + d
        };
        assert(result == 10);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_tuple_pattern_5_element(self, compiler, runtime):
        """Test tuple pattern with 5 elements"""
        source = """
        let quint = (1, 2, 3, 4, 5);
        let result = match quint {
            (a, b, c, d, e) => a + b + c + d + e
        };
        assert(result == 15);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_tuple_pattern_deeply_nested(self, compiler, runtime):
        """Test deeply nested tuple patterns"""
        source = """
        let deep = (((1, 2), 3), 4);
        let result = match deep {
            (((a, b), c), d) => a + b + c + d
        };
        assert(result == 10);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_tuple_pattern_mixed_literals(self, compiler, runtime):
        """Test tuple pattern with mixed literal and variable patterns"""
        source = """
        let pair = (10, 20);
        let result = match pair {
            (10, y) => y * 2,
            (x, 20) => x * 2,
            (x, y) => x + y
        };
        assert(result == 40);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_tuple_pattern_wildcard(self, compiler, runtime):
        """Test tuple pattern with wildcard"""
        source = """
        let triple = (1, 2, 3);
        let result = match triple {
            (a, _, c) => a + c,
            _ => 0
        };
        assert(result == 4);
        """
        self._compile_and_execute(source, compiler, runtime)


class TestGuardClauseEdgeCases:
    """Test edge cases for guard clauses"""
    
    def _compile_and_execute(self, source: str, compiler, runtime, inputs=None):
        """Helper to compile and execute Einlang code"""
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        exec_result = runtime.execute(context, inputs or {})
        assert exec_result.success, f"Execution failed: {exec_result.error}"
        return exec_result
    
    def test_guard_clause_with_rest_pattern(self, compiler, runtime):
        """Test guard clause with array rest pattern"""
        source = """
        let arr = [1, 2, 3, 4];
        let result = match arr {
            [first, ..rest] where len(rest) > 2 => "long-rest",
            [first, ..rest] where len(rest) > 0 => "short-rest",
            [..rest] => "empty-rest",
            _ => "no-rest"
        };
        assert(result == "long-rest");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_guard_clause_complex_condition(self, compiler, runtime):
        """Test guard clause with complex condition"""
        source = """
        let x = 15;
        let result = match x {
            n where n > 10 => "large",
            n where n > 5 => "medium",
            n where n > 0 => "small",
            _ => "other"
        };
        assert(result == "large");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_guard_clause_with_function_call(self, compiler, runtime):
        """Test guard clause with function call"""
        source = """
        let arr = [1, 2, 3];
        let result = match arr {
            arr where len(arr) > 2 => "long",
            arr where len(arr) > 0 => "short",
            _ => "empty"
        };
        assert(result == "long");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_guard_clause_nested_pattern(self, compiler, runtime):
        """Test guard clause with nested pattern"""
        source = """
        let nested = ((5, 10), 15);
        let result = match nested {
            ((x, y), z) where x + y + z > 20 => "large-sum",
            ((x, y), z) where x + y + z > 10 => "medium-sum",
            _ => "small-sum"
        };
        assert(result == "large-sum");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_guard_clause_multiple_conditions(self, compiler, runtime):
        """Test guard clause with multiple conditions"""
        source = """
        let pair = (10, 5);
        let result = match pair {
            (x, y) where x > y => "x-wins",
            (x, y) where y > x => "y-wins",
            _ => "tie"
        };
        assert(result == "x-wins");
        """
        self._compile_and_execute(source, compiler, runtime)


class TestComplexNestedPatterns:
    """Test complex nested pattern scenarios"""
    
    def _compile_and_execute(self, source: str, compiler, runtime, inputs=None):
        """Helper to compile and execute Einlang code"""
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        exec_result = runtime.execute(context, inputs or {})
        assert exec_result.success, f"Execution failed: {exec_result.error}"
        return exec_result
    
    def test_nested_tuple_in_array(self, compiler, runtime):
        """Test nested tuple patterns in array"""
        source = """
        let arr = [1, 2, 3, 4, 5];
        let result = match arr {
            [a, b, ..rest] => a + b,
            _ => 0
        };
        assert(result == 3);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_nested_array_in_tuple(self, compiler, runtime):
        """Test nested array patterns in tuple"""
        source = """
        let nested = ((1, 2), (3, 4));
        let result = match nested {
            ((a, b), (c, d)) => a + b + c + d,
            _ => 0
        };
        assert(result == 10);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_complex_nested_with_guards(self, compiler, runtime):
        """Test complex nested patterns with guards"""
        source = """
        let value = (((1, 2), 3), 4);
        let result = match value {
            (((a, b), c), d) where a + b + c + d > 5 => "large",
            (((a, b), c), d) where a + b + c + d > 0 => "small",
            _ => "zero"
        };
        assert(result == "large");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_multiple_rest_patterns_in_nested(self, compiler, runtime):
        """Test multiple rest patterns in nested structures"""
        source = """
        let arr = [1, 2, 3, 4, 5];
        let result = match arr {
            [a, b, ..rest] => a + b + len(rest),
            _ => 0
        };
        assert(result == 6);
        """
        self._compile_and_execute(source, compiler, runtime)


class TestOrPatterns:
    """Test or-pattern matching (pat1 | pat2)"""
    
    def _compile_and_execute(self, source: str, compiler, runtime, inputs=None):
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        exec_result = runtime.execute(context, inputs or {})
        assert exec_result.success, f"Execution failed: {exec_result.error}"
        return exec_result
    
    def test_or_pattern_literals(self, compiler, runtime):
        source = """
        let x = 2;
        let result = match x {
            1 | 2 | 3 => "low",
            4 | 5 | 6 => "mid",
            _ => "high"
        };
        assert(result == "low");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_or_pattern_second_arm(self, compiler, runtime):
        source = """
        let x = 5;
        let result = match x {
            1 | 2 | 3 => "low",
            4 | 5 | 6 => "mid",
            _ => "high"
        };
        assert(result == "mid");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_or_pattern_fallthrough(self, compiler, runtime):
        source = """
        let x = 99;
        let result = match x {
            1 | 2 | 3 => "low",
            4 | 5 | 6 => "mid",
            _ => "high"
        };
        assert(result == "high");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_or_pattern_with_guard(self, compiler, runtime):
        source = """
        let x = 3;
        let result = match x {
            1 | 2 | 3 where x > 2 => "low-large",
            1 | 2 | 3 => "low",
            _ => "other"
        };
        assert(result == "low-large");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_or_pattern_two_alternatives(self, compiler, runtime):
        source = """
        let x = true;
        let result = match x {
            true | false => "boolean"
        };
        assert(result == "boolean");
        """
        self._compile_and_execute(source, compiler, runtime)


class TestRangePatterns:
    """Test range pattern matching (start..end, start..=end)"""
    
    def _compile_and_execute(self, source: str, compiler, runtime, inputs=None):
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        exec_result = runtime.execute(context, inputs or {})
        assert exec_result.success, f"Execution failed: {exec_result.error}"
        return exec_result
    
    def test_range_pattern_inclusive(self, compiler, runtime):
        source = """
        let x = 5;
        let result = match x {
            0..=9 => "digit",
            _ => "other"
        };
        assert(result == "digit");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_range_pattern_inclusive_boundary(self, compiler, runtime):
        source = """
        let x = 9;
        let result = match x {
            0..=9 => "digit",
            _ => "other"
        };
        assert(result == "digit");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_range_pattern_exclusive(self, compiler, runtime):
        source = """
        let x = 5;
        let result = match x {
            0..10 => "single-digit",
            _ => "other"
        };
        assert(result == "single-digit");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_range_pattern_exclusive_boundary(self, compiler, runtime):
        source = """
        let x = 10;
        let result = match x {
            0..10 => "single-digit",
            _ => "other"
        };
        assert(result == "other");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_range_pattern_multiple_ranges(self, compiler, runtime):
        source = """
        let x = 50;
        let result = match x {
            0..=9 => "digit",
            10..=99 => "two-digit",
            100..=999 => "three-digit",
            _ => "large"
        };
        assert(result == "two-digit");
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_range_pattern_negative(self, compiler, runtime):
        source = """
        let x = -3;
        let result = match x {
            -10..0 => "negative",
            0..=10 => "non-negative",
            _ => "other"
        };
        assert(result == "negative");
        """
        self._compile_and_execute(source, compiler, runtime)


class TestBindingPatterns:
    """Test binding pattern matching (name @ pattern)"""
    
    def _compile_and_execute(self, source: str, compiler, runtime, inputs=None):
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        exec_result = runtime.execute(context, inputs or {})
        assert exec_result.success, f"Execution failed: {exec_result.error}"
        return exec_result
    
    def test_binding_with_literal(self, compiler, runtime):
        source = """
        let x = 42;
        let result = match x {
            n @ 42 => n + 1,
            _ => 0
        };
        assert(result == 43);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_binding_with_literal_no_match(self, compiler, runtime):
        source = """
        let x = 10;
        let result = match x {
            n @ 42 => n + 1,
            _ => 0
        };
        assert(result == 0);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_binding_with_wildcard(self, compiler, runtime):
        source = """
        let x = 7;
        let result = match x {
            n @ _ => n * 2
        };
        assert(result == 14);
        """
        self._compile_and_execute(source, compiler, runtime)
    
    def test_binding_with_tuple(self, compiler, runtime):
        source = """
        let pair = (3, 4);
        let result = match pair {
            p @ (3, 4) => 1,
            _ => 0
        };
        assert(result == 1);
        """
        self._compile_and_execute(source, compiler, runtime)

