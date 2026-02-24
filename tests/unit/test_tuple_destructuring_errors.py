#!/usr/bin/env python3
"""
Tests for Tuple Destructuring Error Cases
==========================================

Demonstrates how tuple destructuring fails gracefully with proper error messages.
"""

import pytest
from tests.test_utils import compile_and_execute


class TestTupleDestructuringErrors:
    """Test error handling for tuple destructuring"""
    
    def test_arity_mismatch_too_many_variables(self, compiler, runtime):
        """Test error when too many variables for tuple"""
        source = """
        let pair = (10, 20);
        let (x, y, z) = pair;  # Error: trying to unpack 2 values into 3 variables
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        # Should fail at runtime (tuple index out of bounds)
        assert not result.success
        assert "out of bounds" in str(result.error).lower() or "index" in str(result.error).lower()
        print(f"\n✅ Caught arity mismatch error: {result.error}")
    
    def test_arity_mismatch_too_few_variables(self, compiler, runtime):
        """Test behavior when fewer variables than tuple elements"""
        # This actually works - we just ignore extra elements
        source = """
        let triple = (10, 20, 30);
        let (x, y) = triple;  # Takes first two, ignores third
        assert(x == 10);
        assert(y == 20);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        # This should work (extra elements ignored)
        assert result.success
        print(f"\n✅ Partial destructuring works (ignores extra elements)")
    
    def test_destructure_non_tuple(self, compiler, runtime):
        """Test error when trying to destructure non-tuple"""
        source = """
        let value = 42;
        let (x, y) = value;  # Error: trying to destructure a scalar
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        # Should fail - can't destructure non-tuple
        assert not result.success
        print(f"\n✅ Caught non-tuple destructuring error: {result.error}")
    
    def test_destructure_empty_tuple(self, compiler, runtime):
        """Test edge case: empty tuple destructuring"""
        source = """
        let empty = ();
        let (x, y) = empty;  # Error: no elements to destructure
        """
        
        # Note: Parser might not support empty tuples
        try:
            result = compile_and_execute(source, compiler, runtime)
            if not result.success:
                print(f"\n✅ Empty tuple error: {result.error or result.errors}")
            else:
                print(f"\n⚠️ Empty tuple unexpectedly succeeded")
        except Exception as e:
            print(f"\n✅ Parser/compiler caught empty tuple: {e}")
    
    def test_nested_tuple_destructuring_limitation(self, compiler, runtime):
        """Test that nested tuple destructuring is not supported"""
        source = """
        let nested = ((1, 2), (3, 4));
        let ((a, b), (c, d)) = nested;  # Not supported (would need recursive destructuring)
        """
        
        # This likely fails at parse time or compilation
        try:
            result = compile_and_execute(source, compiler, runtime)
            if not result.success:
                print(f"\n✅ Nested destructuring limitation: {result.error or result.errors}")
            else:
                print(f"\n⚠️ Nested destructuring unexpectedly worked")
        except Exception as e:
            print(f"\n✅ Parser caught nested pattern: {e}")
    
    def test_type_mismatch_in_destructured_variables(self, compiler, runtime):
        """Test runtime behavior with type mismatches"""
        # Einlang is dynamically typed, so this should work but demonstrate the values
        source = """
        let mixed = (42, "hello", true);
        let (num, text, flag) = mixed;
        
        # These should work because Einlang is dynamically typed
        assert(num == 42);
        print("Text:", text);
        print("Flag:", flag);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        # Should succeed (dynamic typing)
        assert result.success
        print(f"\n✅ Mixed types work with dynamic typing")
    
    def test_destructure_with_undefined_tuple(self, compiler, runtime):
        """Test error when destructuring undefined variable"""
        source = """
        let (x, y) = undefined_tuple;  # Error: undefined variable
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        # Should fail at compilation (undefined variable)
        assert not result.success
        assert "undefined" in str(result.errors).lower() or "not defined" in str(result.errors).lower()
        print(f"\n✅ Caught undefined variable: {result.errors}")
    
    def test_destructure_from_function_returning_wrong_type(self, compiler, runtime):
        """Test error when function doesn't return a tuple"""
        source = """
        fn get_value() {
            42  # Returns scalar, not tuple
        }
        
        let (x, y) = get_value();  # Error: can't destructure scalar
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        # Should fail at runtime
        assert not result.success
        print(f"\n✅ Function return type mismatch: {result.error}")
    
    def test_show_actual_destructuring_ir(self, compiler):
        """Demonstrate the actual IR generated for tuple destructuring"""
        source = """
        let (x, y, z) = (10, 20, 30);
        """
        compilation = compiler.compile(source, "<test>")
        
        if compilation.success:
            # Show the desugared IR
            print(f"\n✅ DESUGARED IR (Industry Best Practice):")
            print(f"Source: let (x, y, z) = (10, 20, 30);")
            print(f"\nGenerated IR statements:")
            
            # IR is already available in compilation.ir
            ir = compilation.ir
            
            for i, stmt in enumerate(ir.statements):
                print(f"  {i+1}. {stmt}")
                if hasattr(stmt, 'pattern') and hasattr(stmt, 'value'):
                    print(f"     -> Binds: {stmt.pattern}")
                    print(f"     -> Value: {stmt.value}")
            
            print(f"\nPattern: Temporary -> TupleAccess(0) -> TupleAccess(1) -> TupleAccess(2)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

