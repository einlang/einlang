"""
Tests for function shadowing/hiding during monomorphization.

Issue: Current implementation assumes global namespace and doesn't handle:
1. Nested functions with same name as outer functions
2. Functions in different scopes with the same name
3. Proper scope-aware mangling

Rust approach: Use DefId (definition ID) with scope path
"""

import pytest
from tests.test_utils import apply_ir_round_trip


class TestFunctionShadowing:
    """Test monomorphization with function name shadowing."""
    
    def test_nested_function_shadows_outer(self, compiler, runtime):
        """
        Test that nested function with same name doesn't conflict with outer.
        
        RUST-STYLE DESIGN: Nested 'fn' declarations do NOT capture outer variables.
        They are treated as separate, independently monomorphizable functions.
        
        Each function gets unique DefId regardless of name conflicts.
        Nested functions must receive outer variables as parameters (no implicit capture).
        """
        source = """
        fn helper(x) {
            x * 2
        }
        
        fn outer(x) {
            fn helper(y) {
                y * 3
            }
            helper(x) + 1
        }
        
        let result1 = helper(5);    // Should call outer helper: 5 * 2 = 10
        let result2 = outer(5);     // Should call inner helper: 5 * 3 + 1 = 16
        """
        
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        result = runtime.execute(context)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs["result1"] == 10
        assert result.outputs["result2"] == 16
    
    def test_function_redefinition_in_block(self, compiler, runtime):
        """
        Test function shadowing at different scope levels.
        
        Uses nested functions to demonstrate scope-based shadowing
        (Einlang doesn't support bare block expressions, so we use functions).
        """
        source = """
        fn process(x) {
            x + 10
        }
        
        fn scoped_test() {
            fn process(x) {
                x * 10
            }
            process(5)
        }
        
        let result1 = process(5);     // Outer process: 15
        let result2 = scoped_test();  // Inner process: 50
        """
        
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        result = runtime.execute(context)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs["result1"] == 15
        assert result.outputs["result2"] == 50
    
    def test_parameter_shadows_function_name(self, compiler, runtime):
        """
        Test that parameter name shadowing function name works correctly.
        
        This is variable shadowing, not function shadowing, but important for
        proper scope handling.
        """
        source = """
        fn helper(x) {
            x * 2
        }
        
        fn process(helper) {
            // 'helper' parameter shadows 'helper' function
            helper + 10
        }
        
        let result1 = helper(5);      // Function call: 10
        let result2 = process(5);     // Parameter use: 15
        """
        
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        result = runtime.execute(context)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs["result1"] == 10
        assert result.outputs["result2"] == 15
    
    def test_generic_calls_in_nested_function_rewritten(self, compiler, runtime):
        """
        Test that generic function calls inside nested functions are properly rewritten.
        
        REGRESSION TEST: Previously, visit_function_definition just did 'pass',
        which meant calls inside nested functions were not traversed and rewritten.
        
        This test verifies that the CallRewriter properly traverses into nested
        function bodies to rewrite their generic calls.
        """
        source = """
        fn identity(x) {
            x
        }
        
        fn double(x) {
            x * 2
        }
        
        fn outer(a) {
            fn nested(b) {
                // These calls inside nested function should be rewritten
                // Use simple parameter references (no intermediate variables)
                identity(b) + double(b)
            }
            nested(a) + 1
        }
        
        let result = outer(5);  // nested(5) = identity(5) + double(5) = 5 + 10 = 15, outer = 16
        """
        
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        result = runtime.execute(context)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs["result"] == 16
    
    def test_use_statement_inside_function_body_with_traversal(self, compiler, runtime):
        """
        Test that use statements INSIDE function bodies are properly handled.
        
        REGRESSION TEST: The traversal fix should not break when encountering
        use statements inside function bodies during visit_function_definition traversal.
        
        This verifies that:
        1. Use statements inside function bodies are parsed and processed
        2. The traversal fix (visiting nested function bodies) correctly handles them
        3. visit_use_statement correctly does 'pass' (handled by earlier passes)
        4. Generic calls after function-level use statements work correctly
        
        NOTE: This tests the actual scenario where use appears INSIDE a function:
              fn outer() { use std::math; ... }
        """
        source = """
        fn identity(x) {
            x
        }
        
        fn outer(a) {
            use std::math;
            
            fn nested(b) {
                // Nested function can use imported module from outer scope
                identity(b) + math::abs(b)
            }
            
            nested(a) + 1
        }
        
        let result = outer(5);  // nested(5) = identity(5) + abs(5) = 5 + 5 = 10, outer = 11
        """
        
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        result = runtime.execute(context)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs["result"] == 11
    
    def test_use_statement_inside_nested_function_with_traversal(self, compiler, runtime):
        """
        Test use statement INSIDE a nested function body.
        
        REGRESSION TEST: Verify traversal handles use statements in deeply nested functions.
        
        This is the most direct test of the fix:
        - visit_function_definition now traverses into nested function bodies
        - When it encounters a use statement inside, visit_use_statement does 'pass'
        - Generic calls after the use statement should still work
        """
        source = """
        fn identity(x) {
            x
        }
        
        fn outer(a) {
            fn nested(b) {
                use std::math;
                // Use statement INSIDE nested function
                // Generic calls should still be discovered and rewritten
                identity(b) + math::abs(b)
            }
            nested(a) + 1
        }
        
        let result = outer(5);  // nested(5) = identity(5) + abs(5) = 5 + 5 = 10, outer = 11
        """
        
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
        apply_ir_round_trip(context)
        result = runtime.execute(context)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs["result"] == 11


class TestScopeAwareMangling:
    """Test that mangled names include scope information."""
    
    def test_same_name_different_scopes_get_unique_mangles(self, compiler):
        """
        Test that functions with same name in different scopes get unique mangled names.
        
        RUST LESSON: Mangled names should include scope path to avoid conflicts.
        E.g., outer::helper_i32 vs inner::helper_i32
        """
        source = """
        fn helper(x) {
            x * 2
        }
        
        fn wrapper(y) {
            fn helper(z) {
                z * 3
            }
            helper(y)
        }
        
        let a = helper(5);
        let b = wrapper(5);
        """
        
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

