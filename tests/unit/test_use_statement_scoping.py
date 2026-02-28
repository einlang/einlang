"""
Test Use Statement Scoping

Tests the position-aware scoping semantics of use statements following Issue #3:
- Use statements only affect code AFTER them in the same scope
- Nested scopes inherit from outer scopes (lexical scoping)
- Function-local imports don't leak to global scope

Uses source_overlay (in-memory modules) to avoid I/O on the critical path.
"""

import pytest
from tests.test_utils import compile_and_execute

# In-memory module sources (avoid file I/O)
MY_MATH_ADD = "pub fn add(a, b) { a + b }"
MY_MATH_ADD_MUL = "pub fn add(a, b) { a + b }\npub fn multiply(a, b) { a * b }"


class TestUseStatementScoping:
    """Test use statement scoping behavior"""

    def test_use_statement_position_matters(self, compiler, runtime):
        """
        Test that use statements only affect code after them.
        
        Before the use statement, the function should not be in scope.
        After the use statement, the function should be in scope.
        """
        source_code = """
mod my_math;
use my_math::add;

let result = add(5, 3);
assert(result == 8);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): MY_MATH_ADD},
        )
        assert exec_result.success, f"Execution failed: {exec_result.errors}"

    def test_use_before_statement_fails(self, compiler, runtime):
        """
        Test that using a function before the use statement fails.
        
        This should produce an error because the function is not in scope yet.
        
        NOTE: This test documents the INTENDED behavior for Issue #3 (position-aware scoping).
        Skipped for deserialization: imports are resolved globally at compile time.
        """
        source_code = """
mod my_math;

let result = add(5, 3);  // Error: add not in scope yet

use my_math::add;
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): MY_MATH_ADD},
        )
        assert not exec_result.success, "Should fail: function used before import"

    def test_function_local_imports(self, compiler, runtime):
        """
        Test that function-local imports are scoped to the function.
        
        Imports inside a function should not leak to global scope.
        """
        source_code = """
mod my_math;

fn test_function() {
    use my_math::add;
    let local_result = add(5, 3);
    local_result
}

let func_result = test_function();
assert(func_result == 8);

// This should fail: add is not in global scope
// let global_result = add(10, 20);  // Would error if uncommented
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): MY_MATH_ADD_MUL},
        )
        assert exec_result.success, f"Execution failed: {exec_result.errors}"

    def test_nested_scopes_inherit_imports(self, compiler, runtime):
        """
        Test that nested scopes inherit imports from outer scopes.
        
        A function should be able to use imports from the global scope.
        """
        source_code = """
mod my_math;
use my_math::add;

fn use_inherited_import() {
    // add should be available from outer scope
    let result = add(10, 20);
    result
}

let func_result = use_inherited_import();
assert(func_result == 30);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): MY_MATH_ADD_MUL},
        )
        assert exec_result.success, f"Execution failed: {exec_result.errors}"

    def test_multiple_use_statements_same_scope(self, compiler, runtime):
        """
        Test multiple use statements in the same scope.
        
        Each use statement should be available after its declaration.
        """
        my_math_three = "pub fn add(a, b) { a + b }\npub fn multiply(a, b) { a * b }\npub fn subtract(a, b) { a - b }"
        source_code = """
mod my_math;

use my_math::add;
let result1 = add(5, 3);
assert(result1 == 8);

use my_math::multiply;
let result2 = multiply(4, 6);
assert(result2 == 24);

use my_math::subtract;
let result3 = subtract(10, 3);
assert(result3 == 7);

// All three should still be available
let result4 = add(result2, result3);
assert(result4 == 31);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): my_math_three},
        )
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
    
    def test_function_local_import_shadows_global(self, compiler, runtime):
        """
        Test that function-local imports can shadow global imports.
        
        A function should be able to import the same name locally,
        shadowing the global import within the function scope.
        
        NOTE: This test documents the INTENDED behavior for Issue #3 (scope shadowing).
        Currently XFAIL because imports with the same name conflict in the global namespace.
        """
        source_overlay = {("math1",): "pub fn compute(x) { x * 2 }", ("math2",): "pub fn compute(x) { x * 3 }"}
        source_code = """
mod math1;
mod math2;

use math1::compute;

let global_result = compute(5);
print("Test 1 - global_result:", global_result, "expected: 10");
assert(global_result == 10);

fn test_shadowing() {
    use math2::compute;
    let local_result = compute(5);
    print("Test 2 - local_result:", local_result, "expected: 15");
    local_result
}

let func_result = test_shadowing();
print("Test 3 - func_result:", func_result, "expected: 15");
assert(func_result == 15);

// Global compute should still work
let global_result2 = compute(7);
print("Test 4 - global_result2:", global_result2, "expected: 14");
assert(global_result2 == 14);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay=source_overlay,
        )
        assert exec_result.success, f"Execution failed: {exec_result.errors}"

    def test_wildcard_import_scoping(self, compiler, runtime):
        """
        Test that wildcard imports respect scoping rules.
        
        Wildcard imports should only affect code after them in the same scope.
        """
        source_code = """
mod my_math;

use my_math::*;

// All public functions should be available
let result1 = add(5, 3);
let result2 = multiply(4, 6);

assert(result1 == 8);
assert(result2 == 24);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): MY_MATH_ADD_MUL},
        )
        assert exec_result.success, f"Execution failed: {exec_result.errors}"

    def test_use_statement_in_nested_functions(self, compiler, runtime):
        """
        Test that outer scope use statements work with nested function calls.
        
        NOTE: This test verifies that imports at outer scope work for nested
        function definitions. Einlang's current `use` statement system doesn't
        yet support imports INSIDE nested functions (architectural limitation).
        
        RUST-STYLE: Outer scope imports should be visible in nested fn declarations
        when they're part of the compilation unit.
        """
        source_code = """
mod my_math;
use my_math::add;
use my_math::multiply;

fn outer_function(x) -> i32 {
    fn inner_function(y) -> i32 {
        // Can call module functions from outer scope imports
        add(y, 1)
    }
    
    // Outer can use multiply
    multiply(inner_function(x), 2)
}

// Test: outer_function(5) = multiply(add(5, 1), 2) = multiply(6, 2) = 12
assert(outer_function(5) == 12);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): MY_MATH_ADD_MUL},
        )
        assert exec_result.success, f"Execution failed: {exec_result.errors}"

    def test_sequential_use_statements(self, compiler, runtime):
        """
        Test that use statements are processed sequentially.
        
        Each use statement should be in effect immediately after it's declared.
        """
        my_math_steps = "pub fn step1(x) { x + 1 }\npub fn step2(x) { x * 2 }\npub fn step3(x) { x - 3 }"
        source_code = """
mod my_math;

use my_math::step1;
let a = step1(5);
assert(a == 6);

use my_math::step2;
let b = step2(a);
assert(b == 12);

use my_math::step3;
let c = step3(b);
assert(c == 9);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): my_math_steps},
        )
        assert exec_result.success, f"Execution failed: {exec_result.errors}"


class TestNegativeScoping:
    """Negative tests - verify that scoping violations are caught"""

    def test_function_local_import_not_available_globally(self, compiler, runtime):
        """
        NEGATIVE TEST: Function-local imports should not leak to global scope.
        
        This should fail when trying to use a function-local import globally.
        
        NOTE: This test documents the INTENDED behavior for Issue #3 (scope isolation).
        Skipped for deserialization: all imports are added to the global namespace.
        """
        source_code = """
mod my_math;

fn test_function() {
    use my_math::add;
    let local_result = add(5, 3);
    local_result
}

// This should fail: add is not in global scope
let global_result = add(10, 20);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): MY_MATH_ADD},
        )
        assert not exec_result.success, "Should fail: function-local import used in global scope"

    def test_use_before_wildcard_import_fails(self, compiler, runtime):
        """
        NEGATIVE TEST: Functions should not be available before wildcard import.
        
        Even with wildcard imports, scoping rules apply.
        
        NOTE: This test documents the INTENDED behavior for Issue #3 (position-aware scoping).
        Skipped for deserialization: wildcard imports are resolved globally at compile time.
        """
        source_code = """
mod my_math;

let result = add(5, 3);  // Error: add not in scope yet

use my_math::*;
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): MY_MATH_ADD},
        )
        assert not exec_result.success, "Should fail: function used before wildcard import"
    
    def test_module_alias_not_available_after_function_scope(self, compiler, runtime):
        """
        NEGATIVE TEST: Module aliases defined in functions don't leak out.
        
        Trying to use a function-local module alias globally should fail.
        """
        source_code = """
mod my_math;

fn test_function() {
    use my_math as m;
    let local_result = m::add(5, 3);
    local_result
}

let func_result = test_function();

// This should fail: m is not in global scope
let global_result = m::add(10, 20);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): MY_MATH_ADD},
        )
        assert not exec_result.success, "Should fail: module alias used outside its scope"

    def test_private_function_not_importable(self, compiler, runtime):
        """
        NEGATIVE TEST: Private functions should not be importable.
        
        Only public functions should be accessible via use statements.
        """
        my_math_private = """
pub fn add(a, b) { a + b }

fn private_helper(x) { x * 2 }  // Private function
"""
        source_code = """
mod my_math;

use my_math::private_helper;  // Should fail: private function

let result = private_helper(5);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): my_math_private},
        )
        assert not exec_result.success, "Should fail: cannot import private function"

    def test_nonexistent_function_import_fails(self, compiler, runtime):
        """
        NEGATIVE TEST: Importing a non-existent function should fail.
        """
        source_code = """
mod my_math;

use my_math::nonexistent_function;  // Should fail: function doesn't exist

let result = nonexistent_function(5);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): MY_MATH_ADD},
        )
        assert not exec_result.success, "Should fail: importing non-existent function"

    def test_using_function_from_nonexistent_module_fails(self, compiler, runtime):
        """
        NEGATIVE TEST: Using a module that doesn't exist should fail.
        
        Note: We DON'T require 'mod' declarations (Rust-style!), but the module
        must exist in the filesystem for ModulePass to find it.
        """
        source_code = """
use nonexistent_module::add;  // Should fail: module doesn't exist

let result = add(5, 3);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
        )
        assert not exec_result.success, "Should fail: module doesn't exist"

    def test_wildcard_import_does_not_import_private_functions(self, compiler, runtime):
        """
        NEGATIVE TEST: Wildcard import should only import public functions.
        
        Private functions should not be accessible even with wildcard import.
        """
        my_math_private = """
pub fn add(a, b) { a + b }

fn private_helper(x) { x * 2 }  // Private function
"""
        source_code = """
mod my_math;

use my_math::*;

let result1 = add(5, 3);  // Should work
let result2 = private_helper(5);  // Should fail: private function
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): my_math_private},
        )
        assert not exec_result.success, "Should fail: wildcard import does not include private functions"


class TestModuleAliasScoping:
    """Test module alias scoping with 'use module as alias'"""

    def test_module_alias_scoping(self, compiler, runtime):
        """
        Test that module aliases respect scoping rules.
        
        Module aliases should only be available after the use statement.
        """
        source_code = """
mod my_math;

use my_math as m;

let result = m::add(5, 3);
assert(result == 8);
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): MY_MATH_ADD},
        )
        assert exec_result.success, f"Execution failed: {exec_result.errors}"

    def test_function_local_module_alias(self, compiler, runtime):
        """
        Test that function-local module aliases don't leak to global scope.
        """
        source_code = """
mod my_math;

fn test_function() {
    use my_math as m;
    let local_result = m::add(5, 3);
    local_result
}

let func_result = test_function();
assert(func_result == 8);

// m should not be available in global scope
// let global_result = m::add(10, 20);  // Would error if uncommented
"""
        exec_result = compile_and_execute(
            source_code=source_code,
            compiler=compiler,
            runtime=runtime,
            source_file="<test>",
            source_overlay={("my_math",): MY_MATH_ADD},
        )
        assert exec_result.success, f"Execution failed: {exec_result.errors}"

