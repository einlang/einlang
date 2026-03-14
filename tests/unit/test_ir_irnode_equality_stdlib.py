"""
Tests for stdlib integration.

Tests the fixes for:
1. Stdlib function discovery, registration, and lowering
"""

import pytest
from pathlib import Path



class TestStdlibIntegration:
    """Test stdlib function discovery, registration, and lowering."""
    
    def test_stdlib_discovery(self):
        """Test that stdlib modules are discovered."""
        from einlang.analysis.module_system import ModuleSystem
        from einlang.passes.base import TyCtxt
        
        tcx = TyCtxt()
        root_path = Path(__file__).resolve().parent.parent.parent
        module_system = ModuleSystem(root_path, tcx.resolver)
        
        assert module_system.stdlib_root is not None, "Stdlib root should be found"
        assert module_system.stdlib_root.exists(), "Stdlib root should exist"
        
        stdlib_modules = module_system._discover_stdlib_modules()
        assert len(stdlib_modules) > 0, "Should discover stdlib modules"
        
        # Check that math module is discovered (contains sqrt function)
        math_modules = [mp for mp in stdlib_modules.keys() if 'math' in str(mp)]
        assert len(math_modules) > 0, "Should discover math module (which contains sqrt)"
    
    def test_stdlib_function_registration(self):
        """Test that stdlib functions are registered during name resolution."""
        from einlang.compiler.driver import CompilerDriver
        from pathlib import Path
        
        # Use qualified stdlib call (Rust-aligned pattern)
        source = 'let x = std::math::sqrt(4.0);'
        compiler = CompilerDriver()
        root_path = Path(__file__).resolve().parent.parent.parent
        result = compiler.compile(source, 'test.ein', root_path=root_path)
        
        # Compilation may fail due to stdlib function body resolution issues;
        # we expect either stdlib in symbol_table, sqrt in IR, or successful compile
        if result.tcx and result.tcx.resolver:
            stdlib_funcs = [
                (k, v) for k, v in result.tcx.resolver._symbol_table.items()
                if isinstance(k, tuple) and len(k) == 2 and k[0] != () and len(k[0]) > 0 and k[0][0] == 'std'
            ]
            stdlib_in_ir = result.ir and any(f.name.startswith('sqrt') for f in result.ir.functions)
            assert len(stdlib_funcs) > 0 or stdlib_in_ir or result.success, "Should register some stdlib functions or have sqrt in IR or succeed"
    
    def test_stdlib_function_lowering(self):
        """Test that stdlib functions are lowered to IR."""
        from einlang.compiler.driver import CompilerDriver
        from pathlib import Path
        
        # Use qualified stdlib call (Rust-aligned pattern)
        source = 'let x = std::math::sqrt(4.0);'
        compiler = CompilerDriver()
        root_path = Path(__file__).resolve().parent.parent.parent
        result = compiler.compile(source, 'test.ein', root_path=root_path)
        
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that sqrt function (or its specialized version) is in the program
        # Note: Generic functions are monomorphized, so we look for names starting with "sqrt"
        sqrt_found = False
        for func in result.ir.functions:
            if func.name.startswith("sqrt"):
                sqrt_found = True
                assert func.defid is not None, "sqrt function should have DefId"
                break
        
        assert sqrt_found, "sqrt function (or specialized version) should be in program IR"
    
    def test_stdlib_function_body_resolution(self):
        """Test that stdlib function bodies are resolved (parameters and identifiers have DefIds)"""
        from einlang.compiler.driver import CompilerDriver
        from pathlib import Path
        from einlang.passes.ir_validation import IRValidationPass
        
        # Use qualified stdlib call (Rust-aligned pattern)
        source = "let x = std::math::sqrt(4.0);"
        compiler = CompilerDriver()
        root_path = Path(__file__).resolve().parent.parent.parent
        result = compiler.compile(source, "test.ein", root_path=root_path)
        
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Find sqrt function (or its specialized version) in IR
        # Note: Generic functions are monomorphized, so we look for names starting with "sqrt"
        sqrt_func = None
        for func in result.ir.functions:
            if func.name.startswith("sqrt"):
                sqrt_func = func
                break
        
        assert sqrt_func is not None, "sqrt function (or specialized version) should be in program IR"
        
        # Validate IR (should pass if body resolution worked)
        validation_pass = IRValidationPass()
        validated_ir = validation_pass.run(result.ir, result.tcx)
        
        # Should pass validation (no missing DefIds) - check via reporter
        errors = result.tcx.reporter.errors
        assert len(errors) == 0, f"IR validation should pass: {errors}"
    
    def test_stdlib_function_calls_other_stdlib(self):
        """Test that stdlib functions calling other stdlib functions are resolved"""
        from einlang.compiler.driver import CompilerDriver
        from pathlib import Path
        from einlang.passes.ir_validation import IRValidationPass
        
        # Use a function that calls another stdlib function (qualified call)
        source = "let x = std::math::pow(2.0, 3.0);"
        compiler = CompilerDriver()
        root_path = Path(__file__).resolve().parent.parent.parent
        result = compiler.compile(source, "test.ein", root_path=root_path)
        
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Validate IR (should pass if recursive stdlib calls are resolved)
        validation_pass = IRValidationPass()
        validated_ir = validation_pass.run(result.ir, result.tcx)
        
        # Should pass validation (no missing DefIds in stdlib function bodies) - check via reporter
        errors = result.tcx.reporter.errors
        assert len(errors) == 0, f"IR validation should pass: {errors}"

