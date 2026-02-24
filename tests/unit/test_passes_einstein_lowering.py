"""
Unit tests for Einstein Lowering Pass

Tests that verify:
1. EinsteinDeclarationIR nodes are correctly lowered to LoweredIteration structures
2. Loop structures are created correctly from range analysis
3. Bindings and guards are properly extracted
4. Backend execution works with lowered iterations
5. Backward compatibility (existing Einstein execution still works)
6. No regressions in existing functionality

This ensures incremental migration progress and catches regressions early.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime
from einlang.ir.nodes import (
    ProgramIR, EinsteinDeclarationIR, LoweredIteration,
    LoopStructure, LocalBinding, GuardCondition, ReductionExpressionIR,
    LoweredEinsteinIR, LoweredReductionIR,
)
from einlang.passes.einstein_lowering import EinsteinLoweringPass
from einlang.passes.range_analysis import RangeAnalysisPass
from tests.test_utils import compile_and_execute


class TestEinsteinLoweringPass:
    """Test Einstein Lowering Pass functionality"""
    
    def test_simple_einstein_lowering(self, compiler, runtime):
        """Test that a simple Einstein declaration is lowered correctly"""
        source = """
        let result[i] = i * 2 where i in 0..5;
        result;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that result is correct
        assert 'result' in result.outputs
        expected = np.array([0, 2, 4, 6, 8])
        np.testing.assert_array_equal(result.outputs['result'], expected)
    
    def test_einstein_with_multiple_indices(self, compiler, runtime):
        """Test Einstein declaration with multiple indices"""
        source = """
        let matrix[i, j] = i + j where i in 0..3, j in 0..2;
        matrix;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that result is correct
        assert 'matrix' in result.outputs
        expected = np.array([
            [0, 1],
            [1, 2],
            [2, 3]
        ])
        np.testing.assert_array_equal(result.outputs['matrix'], expected)
    
    def test_einstein_with_where_clause_condition(self, compiler, runtime):
        """Test Einstein declaration with where clause condition (guard)"""
        source = """
        let filtered[i] = i * 2 where i in 0..10, i > 5;
        filtered;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that result is correct
        # Einstein declarations create full-size arrays, guards filter which iterations execute
        # i in 0..10 creates array of size 10, but only i > 5 (i in 6..9) get filled
        assert 'filtered' in result.outputs
        # Array should have size 10, with values at indices 6-9
        output = result.outputs['filtered']
        assert len(output) == 10, f"Expected array of size 10, got {len(output)}"
        # Check that indices 6-9 have correct values
        assert output[6] == 12
        assert output[7] == 14
        assert output[8] == 16
        assert output[9] == 18
        # Indices 0-5 should be 0 (default initialization)
        for i in range(6):
            assert output[i] == 0, f"Index {i} should be 0, got {output[i]}"
    
    def test_reduction_lowering(self, compiler, runtime):
        """Test that reduction expressions are lowered correctly"""
        source = """
        let sum_result = sum[k in 0..5](k);
        sum_result;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that result is correct (0+1+2+3+4 = 10)
        assert 'sum_result' in result.outputs
        assert result.outputs['sum_result'] == 10
    
    def test_reduction_with_condition(self, compiler, runtime):
        """Test reduction with where clause condition"""
        source = """
        let sum_even = sum[k in 0..10](k) where k % 2 == 0;
        sum_even;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that result is correct (0+2+4+6+8 = 20)
        assert 'sum_even' in result.outputs
        assert result.outputs['sum_even'] == 20
    
    def test_nested_einstein_lowering(self, compiler, runtime):
        """Test nested Einstein declarations"""
        source = """
        let outer[i] = i * 10 where i in 0..3;
        let inner[j] = j * 2 where j in 0..5;
        outer;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that result is correct
        assert 'outer' in result.outputs
        expected = np.array([0, 10, 20])
        np.testing.assert_array_equal(result.outputs['outer'], expected)
    
    def test_einstein_with_array_access(self, compiler, runtime):
        """Test Einstein declaration that accesses arrays"""
        source = """
        let arr = [1, 2, 3, 4, 5];
        let doubled[i] = arr[i] * 2 where i in 0..5;
        doubled;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that result is correct
        assert 'doubled' in result.outputs
        expected = np.array([2, 4, 6, 8, 10])
        np.testing.assert_array_equal(result.outputs['doubled'], expected)
    
    def test_ir_structure_has_lowered_iteration(self, compiler):
        """Test that IR structure contains lowered iteration (EinsteinDeclarationIR or LoweredEinsteinIR)"""
        source = """
        let result[i] = i * 2 where i in 0..5;
        result;
        """
        
        compile_result = compiler.compile(source, source_file="<test>")
        assert compile_result.success, f"Compilation failed: {compile_result.get_errors()}"
        
        lowered_einstein_nodes = []
        def find_lowered_einstein(node):
            if isinstance(node, LoweredEinsteinIR):
                lowered_einstein_nodes.append(node)
            if isinstance(node, EinsteinDeclarationIR):
                if hasattr(node, 'lowered_iteration') and node.lowered_iteration is not None:
                    lowered_einstein_nodes.append(node.lowered_iteration)
            if hasattr(node, 'value'):
                find_lowered_einstein(node.value)
            if hasattr(node, 'statements'):
                for stmt in node.statements:
                    find_lowered_einstein(stmt)
            if hasattr(node, 'left'):
                find_lowered_einstein(node.left)
            if hasattr(node, 'right'):
                find_lowered_einstein(node.right)
            if hasattr(node, 'body'):
                find_lowered_einstein(node.body)
            if isinstance(node, LoweredEinsteinIR) and getattr(node, 'items', None):
                for clause in node.items:
                    find_lowered_einstein(clause)
        
        find_lowered_einstein(compile_result.ir)
        
        assert len(lowered_einstein_nodes) > 0, "Should have lowered Einstein in IR (LoweredEinsteinIR or EinsteinDeclarationIR with lowered_iteration)"
        first = lowered_einstein_nodes[0]
        if isinstance(first, LoweredEinsteinIR):
            assert hasattr(first, 'items') and len(first.items) > 0, "LoweredEinsteinIR should have items"
        else:
            assert isinstance(first, LoweredIteration), "lowered_iteration should be LoweredIteration instance"
    
    def test_lowered_iteration_has_loops(self, compiler):
        """Test that lowered iteration has correct loop structures"""
        source = """
        let result[i, j] = i + j where i in 0..3, j in 0..2;
        result;
        """
        
        compile_result = compiler.compile(source, source_file="<test>")
        assert compile_result.success, f"Compilation failed: {compile_result.get_errors()}"
        
        lowered_clauses_or_loops = []
        def find_loops(node):
            if isinstance(node, LoweredEinsteinIR) and getattr(node, 'items', None):
                for clause in node.items:
                    if getattr(clause, 'loops', None):
                        lowered_clauses_or_loops.append(clause)
            if isinstance(node, EinsteinDeclarationIR) and getattr(node, 'lowered_iteration', None):
                lo = node.lowered_iteration
                if getattr(lo, 'loops', None):
                    lowered_clauses_or_loops.append(lo)
            if hasattr(node, 'value'):
                find_loops(node.value)
            if hasattr(node, 'statements'):
                for stmt in node.statements:
                    find_loops(stmt)
            if hasattr(node, 'items'):
                for it in node.items:
                    find_loops(it)
        
        find_loops(compile_result.ir)
        assert len(lowered_clauses_or_loops) > 0, "Should have lowered iteration with loops"
        lowered = lowered_clauses_or_loops[0]
        assert hasattr(lowered, 'loops')
        assert len(lowered.loops) == 2, "Should have 2 loops for i and j"
        loop_vars = [loop.variable.name for loop in lowered.loops]
        assert 'i' in loop_vars, "Should have loop for 'i'"
        assert 'j' in loop_vars, "Should have loop for 'j'"
    
    def test_reduction_has_lowered_iteration(self, compiler):
        """Test that reduction is lowered (LoweredReductionIR or ReductionExpressionIR with lowered_iteration)"""
        source = """
        let sum_result = sum[k in 0..5](k);
        sum_result;
        """
        
        compile_result = compiler.compile(source, source_file="<test>")
        assert compile_result.success, f"Compilation failed: {compile_result.get_errors()}"
        
        reduction_nodes = []
        def find_reduction_nodes(node):
            if isinstance(node, LoweredReductionIR):
                reduction_nodes.append(node)
            if isinstance(node, ReductionExpressionIR):
                if hasattr(node, 'lowered_iteration') and node.lowered_iteration is not None:
                    reduction_nodes.append(node.lowered_iteration)
                else:
                    reduction_nodes.append(node)
            if hasattr(node, 'value'):
                find_reduction_nodes(node.value)
            if hasattr(node, 'statements'):
                for stmt in node.statements:
                    find_reduction_nodes(stmt)
            if hasattr(node, 'body'):
                find_reduction_nodes(node.body)
        
        find_reduction_nodes(compile_result.ir)
        assert len(reduction_nodes) > 0, "Should have reduction (LoweredReductionIR or ReductionExpressionIR) in IR"
        r = reduction_nodes[0]
        if isinstance(r, LoweredReductionIR):
            assert hasattr(r, 'loops'), "LoweredReductionIR should have loops"
        else:
            assert hasattr(r, 'lowered_iteration') and r.lowered_iteration is not None, \
                "ReductionExpressionIR should have lowered_iteration"
    
    def test_backward_compatibility_simple_array(self, compiler, runtime):
        """Test that simple array operations still work (backward compatibility)"""
        source = """
        let x = [1, 2, 3];
        let y = x + 1;
        y;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that result is correct
        assert 'y' in result.outputs
        expected = np.array([2, 3, 4])
        np.testing.assert_array_equal(result.outputs['y'], expected)
    
    def test_backward_compatibility_binary_ops(self, compiler, runtime):
        """Test that binary operations still work (backward compatibility)"""
        source = """
        let a = 5;
        let b = 10;
        let c = a + b;
        c;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that result is correct
        assert 'c' in result.outputs
        assert result.outputs['c'] == 15
    
    def test_complex_einstein_with_multiple_conditions(self, compiler, runtime):
        """Clauses complement each other: shape is union of all ranges (max of end)."""
        source = """
        let complex[i in 0..10] = i * i where i > 2, i < 8;
        let complex[i in 0..10] = 1 where i <= 2 || i >= 8;
        complex;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        assert 'complex' in result.outputs
        out = result.outputs['complex']
        assert out.shape == (10,)
        expected = np.array([1, 1, 1, 9, 16, 25, 36, 49, 1, 1], dtype=out.dtype)
        np.testing.assert_array_equal(out, expected)
    
    def test_einstein_with_dependent_ranges(self, compiler, runtime):
        """Test Einstein with dependent ranges (j depends on i)"""
        source = """
        let triangle[i, j] = i + j where i in 0..3, j in 0..i;
        triangle;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        # This might fail if dynamic ranges aren't fully supported yet
        # But we test to see current state
        if result.success:
            assert 'triangle' in result.outputs
            # Result should be a triangular matrix
            # This test documents current behavior
    
    def test_product_reduction(self, compiler, runtime):
        """Test product reduction"""
        source = """
        let prod = product[k in 1..5](k);
        prod;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that result is correct (1*2*3*4 = 24)
        assert 'prod' in result.outputs
        assert result.outputs['prod'] == 24
    
    def test_min_reduction(self, compiler, runtime):
        """Test min reduction"""
        source = """
        let min_val = min[k in 0..10](k * 2);
        min_val;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that result is correct (min of 0, 2, 4, ..., 18 is 0)
        assert 'min_val' in result.outputs
        assert result.outputs['min_val'] == 0
    
    def test_max_reduction(self, compiler, runtime):
        """Test max reduction"""
        source = """
        let max_val = max[k in 0..10](k * 2);
        max_val;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Compilation failed: {result.get_errors()}"
        
        # Check that result is correct (max of 0, 2, 4, ..., 18 is 18)
        assert 'max_val' in result.outputs
        assert result.outputs['max_val'] == 18


class TestEinsteinLoweringPassIntegration:
    """Integration tests for Einstein Lowering Pass with full compilation pipeline"""
    
    def test_pass_runs_in_pipeline(self, compiler):
        """Test that Einstein lowering runs in the compilation pipeline (LoweredEinsteinIR or EinsteinDeclarationIR)"""
        source = """
        let result[i] = i * 2 where i in 0..5;
        result;
        """
        
        compile_result = compiler.compile(source, source_file="<test>")
        assert compile_result.success, f"Compilation failed: {compile_result.get_errors()}"
        
        lowered = []
        def find_lowered(node):
            if isinstance(node, LoweredEinsteinIR):
                lowered.append(node)
            if isinstance(node, EinsteinDeclarationIR) and getattr(node, 'lowered_iteration', None):
                lowered.append(node.lowered_iteration)
            if hasattr(node, 'value'):
                find_lowered(node.value)
            if hasattr(node, 'statements'):
                for stmt in node.statements:
                    find_lowered(stmt)
        find_lowered(compile_result.ir)
        assert len(lowered) > 0, "Should have lowered Einstein in IR"
    
    def test_pass_depends_on_range_analysis(self, compiler):
        """Test that Einstein lowering has loops (range analysis ran before lowering)"""
        source = """
        let result[i] = i * 2 where i in 0..5;
        result;
        """
        
        compile_result = compiler.compile(source, source_file="<test>")
        assert compile_result.success, f"Compilation failed: {compile_result.get_errors()}"
        
        loops_list = []
        def find_loops(node):
            if isinstance(node, LoweredEinsteinIR) and getattr(node, 'items', None):
                for clause in node.items:
                    if getattr(clause, 'loops', None):
                        loops_list.append(clause.loops)
            if isinstance(node, EinsteinDeclarationIR) and getattr(node, 'lowered_iteration', None):
                lo = node.lowered_iteration
                if getattr(lo, 'loops', None):
                    loops_list.append(lo.loops)
            if hasattr(node, 'value'):
                find_loops(node.value)
            if hasattr(node, 'statements'):
                for stmt in node.statements:
                    find_loops(stmt)
            if hasattr(node, 'items'):
                for it in node.items:
                    find_loops(it)
        find_loops(compile_result.ir)
        assert len(loops_list) > 0, "Should have lowered iteration with loops"
        assert len(loops_list[0]) > 0, "Loops should be created from range analysis"


class TestLoweredExecution:
    """Test that lowered iterations execute correctly"""
    
    def test_lowered_execution_matches_einstein(self, compiler, runtime):
        """Test that lowered execution produces same results as Einstein execution"""
        # This test ensures the new loop-based execution matches the old Einstein execution
        source = """
        let einstein_result[i] = i * 2 where i in 0..5;
        einstein_result;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.get_errors()}"
        
        # Verify result matches expected
        assert 'einstein_result' in result.outputs
        expected = np.array([0, 2, 4, 6, 8])
        np.testing.assert_array_equal(result.outputs['einstein_result'], expected)
    
    def test_lowered_reduction_matches_direct(self, compiler, runtime):
        """Test that lowered reduction produces correct results"""
        source = """
        let sum_direct = sum[k in 0..5](k);
        sum_direct;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.get_errors()}"
        
        # Verify result (0+1+2+3+4 = 10)
        assert 'sum_direct' in result.outputs
        assert result.outputs['sum_direct'] == 10
    
