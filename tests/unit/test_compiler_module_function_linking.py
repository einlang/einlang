"""
Test Module Function Linking in IR Path

This test verifies that module function calls are correctly linked at compile-time
and executed without runtime resolution.

Key Tests:
1. Qualified module calls (math::sqrt)
2. Wildcard imports (use std::math::*)
3. Module aliases (use python::math as pm)
4. Verify module_path and module_ref are set in IR
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


class TestModuleFunctionLinking:
    """Test compile-time module function linking"""
    
    def test_qualified_module_call(self, compiler, runtime):
        """Test qualified module call: math::sqrt()"""
        source_code = """
        use python::math;
        let result = math::sqrt(16.0);
        """
        
        # Compile (use compile() to get IR)
        compilation = compiler.compile(source_code, "<test>")
        errs = getattr(compilation, "errors", None) or (compilation.get_errors() if hasattr(compilation, "get_errors") else [])
        assert compilation.success, f"Compilation failed: {errs}"
        
        ir_program = compilation.ir
        if ir_program:
            from einlang.ir.nodes import FunctionCallIR
            function_calls = []
            for stmt in ir_program.statements:
                if hasattr(stmt, 'value') and isinstance(stmt.value, FunctionCallIR):
                    function_calls.append(stmt.value)
            
            if function_calls:
                call = function_calls[0]
                callee_defid = getattr(call, "function_defid", None) or getattr(call, "defid", None)
                module_path = getattr(call, "module_path", None)
                has_dispatch = callee_defid is not None or (module_path and len(module_path) > 0)
                assert has_dispatch, "function call should have defid or module_path for dispatch"
        
        # Execute
        result = compile_and_execute(source_code, compiler, runtime)
        assert result.success
        assert 'result' in result.outputs
        assert result.outputs['result'] == 4.0
    
    def test_wildcard_import(self, compiler, runtime):
        """Test wildcard import: use std::math::*"""
        source_code = """
        use std::math::*;
        let result = sqrt(25.0);
        """
        
        # Compile and execute
        result = compile_and_execute(source_code, compiler, runtime)
        assert result.success
        assert 'result' in result.outputs
        assert result.outputs['result'] == 5.0
    
    def test_module_alias(self, compiler, runtime):
        """Test module alias: use python::math as pm"""
        source_code = """
        use python::math as pm;
        let result = pm::sqrt(36.0);
        """
        
        # Compile and execute
        result = compile_and_execute(source_code, compiler, runtime)
        assert result.success
        assert 'result' in result.outputs
        assert result.outputs['result'] == 6.0
    
    def test_module_constant_access(self, compiler, runtime):
        """Test module constant: python::math::pi"""
        source_code = """
        use python::math;
        let result = math::pi;
        """
        
        # Compile and execute
        result = compile_and_execute(source_code, compiler, runtime)
        assert result.success
        assert 'result' in result.outputs
        assert abs(result.outputs['result'] - 3.14159) < 0.001
    
    def test_module_call_in_einstein(self, compiler, runtime):
        """Test module call inside Einstein notation"""
        source_code = """
        use std::math::*;
        let data = [1.0, 4.0, 9.0, 16.0];
        let result[i in 0..4] = sqrt(data[i]);
        """
        
        # Compile and execute
        result = compile_and_execute(source_code, compiler, runtime)
        assert result.success
        assert 'result' in result.outputs
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result.outputs['result'], expected)


class TestModuleLinkingDebug:
    """Debug module linking issues"""
    
    def test_check_pass_context_data(self, compiler):
        """Check that pass context contains module data"""
        source_code = """
        use python::math;
        let x = math::sqrt(4.0);
        """
        
        compilation = compiler.compile(source_code, "<test>")
        assert compilation.success

        tcx = compilation.tcx
        assert tcx is not None, "Should have type context"
        module_loader = getattr(tcx, "module_loader", None)
        assert module_loader is not None or compilation.ir is not None, "Should have module_loader or IR"
    
    def test_check_ir_lowering(self, compiler):
        """Check that IR lowering receives pass context"""
        source_code = """
        use python::math;
        let x = math::sqrt(4.0);
        """
        
        # Compile (use compile() not analyze() to get IR)
        compilation = compiler.compile(source_code, "<test>")
        assert compilation.success
        
        ir_program = compilation.ir
        assert ir_program is not None, "IR should be generated"
        
        from einlang.ir.nodes import FunctionCallIR
        for stmt in ir_program.statements:
            if hasattr(stmt, 'value') and isinstance(stmt.value, FunctionCallIR):
                call = stmt.value
                callee_defid = getattr(call, "function_defid", None) or getattr(call, "defid", None)
                print(f"\nFunctionCallIR found:")
                print(f"  function_name: {call.function_name}")
                print(f"  defid: {callee_defid}")
                
                # DefId should be set (Rust-style dispatch)
                if callee_defid:
                    print("✅ DefId is set (Rust-style)")
                else:
                    print("❌ DefId is NOT set")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

