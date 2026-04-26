"""
End-to-end tests for lexical scope and redeclaration rules.

Same-scope redeclaration is rejected. Inner scopes may introduce local names
that hide outer names, and user-defined functions can still hide builtins.
"""

import pytest
from tests.test_utils import compile_and_execute



class TestLexicalScope:
    """Test current lexical scope semantics."""
    
    def test_lambda_redefinition_of_function_in_same_scope_errors(self, compiler, runtime):
        """A let binding cannot redefine a function in the same scope."""
        source = """
        // Define a function
        fn double(x: i32) -> i32 {
            x * 2
        }
        
        // Same-scope lambda redefinition is rejected.
        let double = |x| x * 3;

        let result[0] = double(5);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Expected same-scope lambda/function redefinition to fail"
        assert "redefinition" in " ".join(result.errors)
        
    
    def test_function_shadows_builtin(self, compiler, runtime):
        """Test that user-defined functions shadow builtins."""
        source = """
        // Define a function that shadows builtin 'min'
        fn min(a: i32, b: i32) -> i32 {
            a + b  // Not actually min, just adds
        }
        
        // Should call our function (a + b), not builtin min
        let result[0] = min(5, 3);
        
        assert(result[0] == 8, "Function should shadow builtin: 5 + 3 = 8");
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Test failed: {result.errors}"
        
    
    def test_nested_scope_shadowing(self, compiler, runtime):
        """Test that inner scopes shadow outer scopes."""
        source = """
        // Outer function
        fn compute(x: i32) -> i32 {
            x * 2
        }
        
        // Outer usage - calls outer compute
        let outer_result[0] = compute(5);
        
        // Inner scope with shadowing function
        fn wrapper() -> i32 {
            // Inner function shadows outer
            fn compute(x: i32) -> i32 {
                x * 3
            }
            
            compute(5)  // Should use inner compute
        }
        
        let inner_result[0] = wrapper();
        
        assert(outer_result[0] == 10, "Outer scope: 5 * 2 = 10");
        assert(inner_result[0] == 15, "Inner scope shadows: 5 * 3 = 15");
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Test failed: {result.errors}"
        
    
    def test_lambda_shadows_builtin(self, compiler, runtime):
        """Test that lambda variables can shadow builtins."""
        source = """
        // Lambda shadows builtin 'max'
        let max = |a, b| a + b;  // Not actually max, just adds
        
        // Should call lambda (a + b), not builtin max
        let result[0] = max(5, 3);
        
        assert(result[0] == 8, "Lambda should shadow builtin: 5 + 3 = 8");
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Test failed: {result.errors}"
        
    
    def test_lambda_redefinition_after_builtin_shadowing_errors(self, compiler, runtime):
        """A local function can hide a builtin, but a later same-scope let cannot replace it."""
        source = """
        // Layer 1: Builtin 'min' exists
        
        // Layer 2: Function shadows builtin
        fn min(a: i32, b: i32) -> i32 {
            a * 10  // Returns a * 10
        }
        
        let layer2_result[0] = min(5, 3);
        
        // Layer 3: Same-scope lambda redefinition is rejected.
        let min = |a, b| a * 100;  // Returns a * 100
        
        let layer3_result[0] = min(5, 3);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Expected same-scope lambda/function redefinition to fail"
        assert "redefinition" in " ".join(result.errors)
        
    
    def test_shadowing_with_same_name_different_scopes(self, compiler, runtime):
        """Test that same name can have different meanings in different scopes."""
        source = """
        // Global function
        fn compute(x: i32) -> i32 {
            x * 2
        }
        
        // Function A uses global compute
        fn funcA() -> i32 {
            compute(5)  // Uses global: 5 * 2 = 10
        }
        
        // Function B shadows compute with lambda
        fn funcB() -> i32 {
            let compute = |x| x * 3;  // Shadows global
            compute(5)  // Uses lambda: 5 * 3 = 15
        }
        
        // Function C shadows compute with function
        fn funcC() -> i32 {
            fn compute(x: i32) -> i32 {
                x * 4  // Shadows global
            }
            compute(5)  // Uses inner function: 5 * 4 = 20
        }
        
        let resultA[0] = funcA();
        let resultB[0] = funcB();
        let resultC[0] = funcC();
        
        assert(resultA[0] == 10, "funcA uses global: 5 * 2 = 10");
        assert(resultB[0] == 15, "funcB uses lambda: 5 * 3 = 15");
        assert(resultC[0] == 20, "funcC uses inner function: 5 * 4 = 20");
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Test failed: {result.errors}"
        
    
    def test_let_cannot_replace_hoisted_function_in_same_scope(self, compiler, runtime):
        """
        Test that function definitions are hoisted and cannot be replaced by a
        same-scope let binding.
        """
        source = """
        // Function shadows builtin everywhere in this scope (hoisting)
        fn min(a: i32, b: i32) -> i32 {
            a * 2  // Not actual min
        }
        
        let test1[0] = min(3, 5);  // Uses function (already hoisted): 3 * 2 = 6
        
        // Same-scope lambda redefinition is rejected.
        let min = |a, b| a * 3;  // Not actual min
        
        let test2[0] = min(3, 5);  // Uses lambda: 3 * 3 = 9
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Expected same-scope lambda/function redefinition to fail"
        assert "redefinition" in " ".join(result.errors)
        
    
    def test_builtins_cannot_shadow_locals(self, compiler, runtime):
        """
        Test that builtins are in outermost scope and cannot shadow locals.
        
        In the current model:
        - Builtins/prelude are in the outermost scope
        - Local definitions are in inner scopes
        - Inner scopes win over outer scopes
        - Builtins can BE shadowed, but cannot SHADOW others
        """
        source = """
        // Define a local function 'min'
        fn min(a: i32, b: i32) -> i32 {
            a + b  // Not actual min, just adds
        }
        
        // Local function should be used, NOT builtin
        let result[0] = min(5, 3);
        
        // Even if we nest deeper, local function still shadows builtin
        fn nested() -> i32 {
            min(10, 20)  // Uses outer function, not builtin
        }
        
        let nested_result[0] = nested();
        
        assert(result[0] == 8, "Local function wins over builtin: 5 + 3 = 8");
        assert(nested_result[0] == 30, "Nested scope uses outer function: 10 + 20 = 30");
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Test failed: {result.errors}"
        
    
    def test_scope_chain_with_builtins(self, compiler, runtime):
        """
        Test the complete scope chain: local → outer → builtin.
        
        Demonstrates resolution order:
        1. Function definitions are hoisted within their scope
        2. Nested scopes see outer scope functions
        3. Builtins are fallback (outermost scope)
        """
        source = """
        // Function max is hoisted - shadows builtin everywhere in main scope
        fn max(a: i32, b: i32) -> i32 {
            a * 2
        }
        
        let use_function[0] = max(5, 3);  // Function hoisted, shadows builtin
        
        // Nested scope sees outer function
        fn nested() -> i32 {
            max(7, 2)  // Uses outer function
        }
        
        let use_outer[0] = nested();
        
        // Nested scope can shadow with lambda
        fn double_nested() -> i32 {
            let max = |a, b| a * 3;  // Lambda shadows outer function
            max(7, 2)
        }
        
        let use_lambda[0] = double_nested();
        
        // Test builtin in nested scope where no local definition
        fn use_builtin_scope() -> i32 {
            min(8, 4)  // No local min → uses builtin
        }
        
        let use_builtin[0] = use_builtin_scope();
        
        assert(use_function[0] == 10, "Function shadows builtin: 5 * 2 = 10");
        assert(use_outer[0] == 14, "Nested sees outer function: 7 * 2 = 14");
        assert(use_lambda[0] == 21, "Lambda shadows outer function: 7 * 3 = 21");
        assert(use_builtin[0] == 4, "No local definition: use builtin min(8,4) = 4");
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Test failed: {result.errors}"
        
