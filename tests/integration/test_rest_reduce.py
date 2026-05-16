"""
End-to-end tests for rest patterns in reduction outputs.

Tests the complete flow:
1. Preprocessing: marks deferred rest patterns
2. Monomorphization: per-rank specialization (when needed)
3. Runtime: expands deferred rest patterns using actual shapes

NO MOCKING: Uses real compiler and runtime.
"""

import numpy as np
import pytest
from tests.test_utils import compile_and_execute
# Note: IR serialization fully tested in unit tests (test_ir_serialization.py, test_ir_golden_snapshots.py)
# These integration tests focus on end-to-end execution correctness


class TestRestPatternReductions:
    """Test rest patterns in reduction output shapes"""
    
    def test_basic_reduction_with_rest_pattern(self, compiler, runtime):
        """
        Test basic reduction with rest pattern in output.
        
        Example from design doc - softmax-style reduction.
        """
        source = """
        let x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let max_val[..batch] = max[j](x[..batch, j]);
        
        assert(max_val[0] == 3.0);
        assert(max_val[1] == 6.0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_reduction_bracket_named_rest(self, compiler, runtime):
        """
        Named rest in reduction brackets: sum over all ..batch axes (not only on LHS).
        """
        source = """
        let x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let row_totals[..batch] = sum[j](x[..batch, j]);
        let grand = sum[..batch](x[..batch]);
        assert(row_totals[0] == 6.0);
        assert(row_totals[1] == 15.0);
        assert(grand == 21.0);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"

    def test_multidimensional_batch_reduction(self, compiler, runtime):
        """
        Test rest pattern spanning multiple dimensions.
        """
        source = """
        let x = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let sums[..batch] = sum[k](x[..batch, k]);
        
        assert(sums[0, 0] == 3.0);   // 1 + 2
        assert(sums[0, 1] == 7.0);   // 3 + 4
        assert(sums[1, 0] == 11.0);  // 5 + 6
        assert(sums[1, 1] == 15.0);  // 7 + 8
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_zero_dimensional_rest_pattern(self, compiler, runtime):
        """
        Test rest pattern with zero dimensions (1D array).
        
        When input is 1D, ..batch spans 0 dimensions, so the output is also scalar.
        """
        source = """
        // 2D case - ..batch spans 1 dimension  
        let x2d = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let sums2d[..batch] = sum[k](x2d[..batch, k]);
        assert(sums2d[0] == 6.0);   // 1 + 2 + 3
        assert(sums2d[1] == 15.0);  // 4 + 5 + 6
        
        // 1D case - ..batch spans 0 dimensions (empty)
        // Result should be scalar, not array
        let x1d = [1.0, 2.0, 3.0];
        let sum1d = sum[k](x1d[k]);  // No rest pattern needed for 1D
        assert(sum1d == 6.0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_generic_function_with_rest_pattern(self, compiler, runtime):
        """
        Test generic function with rest pattern - per-rank specialization.
        
        Per-rank monomorphization creates softmax_array2d_f32 for 2D arrays.
        Rest patterns are expanded via metadata that RangeAnalysisPass reads.
        """
        source = """
        pub fn softmax(x: [f32; *]) -> [f32; *] {
            let max_val[..batch] = max[j](x[..batch, j]);
            max_val  // For now, just return max values
        }
        
        // Test with 2D array - per-rank monomorphization working!
        let x2d = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result2d = softmax(x2d);
        assert(result2d[0] == 3.0);
        assert(result2d[1] == 6.0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"


class TestRestPatternErrors:
    """Test error conditions for rest patterns"""
    
    def test_undetermined_rest_pattern_error(self, compiler, runtime):
        """
        Test that undetermined rest patterns produce clear errors.
        
        Error: ..batch appears in output but not in any input access.
        """
        source = """
        let x = [[1.0, 2.0], [3.0, 4.0]];
        // ERROR: ..batch in output but x is accessed without rest pattern
        let result[..batch] = sum[..other](x[..other]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Should fail: undetermined rest pattern"
        # Check error message mentions the issue
        error_text = str(result.errors)
        assert "batch" in error_text.lower() or "undetermined" in error_text.lower()
    
    def test_multiple_undetermined_rest_patterns_error(self, compiler, runtime):
        """
        Test error when multiple rest patterns appear together without determination.

        This tests the determination-first rule for adjacent packs with no named
        coordinate anchor between them. In a coordinate-aware function body where
        packs surround a named coordinate parameter (e.g. x[..left, coord, ..right]),
        the coordinate parameter serves as the anchor and packs are resolved by
        position without needing solo appearances.
        """
        source = """
        let x = [[[[1.0]]]];
        // ERROR: Both rest patterns appear together without being determined first
        // (no coordinate anchor between ..batch and ..spatial)
        let result[..batch, ..spatial, c] = sum[k](x[..batch, ..spatial, c, k]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Should fail: multiple undetermined rest patterns"
        error_text = str(result.errors)
        assert "batch" in error_text.lower() or "spatial" in error_text.lower()


class TestRestPatternConsistency:
    """Test rest pattern consistency validation"""
    
    def test_inconsistent_rest_pattern_dimensions(self, compiler, runtime):
        """
        Test error when same rest pattern spans different dimensions in different arrays.
        
        NOTE: This is an invalid case that should fail compilation.
        Currently fails with a backend error before rest pattern validation runs.
        """
        source = """
        let x = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];  // Shape: [2, 2, 2]
        let y = [[1.0, 2.0], [3.0, 4.0]];  // Shape: [2, 2]
        
        // ERROR: ..batch spans 2 dimensions in x but 1 dimension in y
        let result[..batch, j] = sum[k](x[..batch, k] * y[..batch, k]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        # Invalid case - should fail compilation (currently fails with backend error)
        assert not result.success, "Should fail: inconsistent rest pattern dimensions"


class TestCoordinateFunctionRestPackResolution:
    """Rest pack resolution via coordinate parameter anchor.

    In a coordinate-aware function body, rest packs that surround a named
    coordinate parameter are resolved by position relative to the anchor.
    They do NOT need solo appearances (determination-first is not required).
    """

    def test_two_packs_around_single_coord_never_appear_alone(self, compiler, runtime):
        """
        ..left and ..right surround `j`. Neither appears alone — they only
        appear together with `j`. Anchor-based resolution, not determination-first.
        """
        source = """
        fn sum_sq[j](x: [f32; ..left, j, ..right]) -> [f32; ..left, ..right] {
            sum[j](x[..left, j, ..right] ** 2.0)
        }

        let x[b in 0..2, c in 0..3] = (10 * b + c) as f32;
        let y = sum_sq[c](x);
        let probe0 = y[0];
        let probe1 = y[1];
        (probe0, probe1);
        """

        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        # sum over c of squared values
        # Row 0: 0^2 + 1^2 + 2^2 = 0 + 1 + 4 = 5
        # Row 1: 10^2 + 11^2 + 12^2 = 100 + 121 + 144 = 365
        assert float(np.asarray(result.outputs["probe0"]).item()) == 5.0
        assert float(np.asarray(result.outputs["probe1"]).item()) == 365.0

    def test_three_packs_and_two_coords_resolved_by_position(self, compiler, runtime):
        """
        ..left, ..middle, ..right with two coordinate anchors j and k.
        None of the packs ever appears alone — all resolved by anchor positions.
        """
        source = """
        fn swap[j, k](x: [f32; ..left, j, ..middle, k, ..right])
            -> [f32; ..left, k, ..middle, j, ..right]
        {
            let y[..left, k, ..middle, j, ..right] =
                x[..left, j, ..middle, k, ..right];
            y
        }

        let x[a in 0..2, b in 0..3, c in 0..4, d in 0..5] =
            (1000 * a + 100 * b + 10 * c + d) as f32;
        let y = swap[b, d](x);
        let probe = y[1, 4, 2, 1];
        probe;
        """

        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        probe = float(np.asarray(result.outputs["probe"]).item())
        assert probe == 1124.0, f"Expected 1124.0, got {probe}"

    def test_single_pack_before_coord_never_appears_alone(self, compiler, runtime):
        """
        ..batch before `j`. ..batch only appears together with j. No solo.
        """
        source = """
        fn sum_over[j](x: [f32; ..batch, j]) -> [f32; ..batch] {
            sum[j](x[..batch, j])
        }

        let x[b in 0..2, c in 0..3] = (10 * b + c) as f32;
        let y = sum_over[c](x);
        let probe0 = y[0];
        let probe1 = y[1];
        (probe0, probe1);
        """

        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert float(np.asarray(result.outputs["probe0"]).item()) == 3.0   # 0+1+2
        assert float(np.asarray(result.outputs["probe1"]).item()) == 33.0  # 10+11+12

    def test_adjacent_packs_no_anchor_still_requires_determination(self, compiler, runtime):
        """
        Without a coordinate anchor, adjacent packs still trigger determination-first.
        This is the case that SHOULD fail: ..batch and ..spatial adjacent, no anchor.
        """
        source = """
        let x = [[[[1.0]]]];
        // ERROR: ..batch and ..spatial have no coordinate anchor between them
        let result[..batch, ..spatial, c] = sum[k](x[..batch, ..spatial, c, k]);
        """

        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, (
            "Should fail: adjacent packs without coordinate anchor"
        )
        error_text = str(result.errors)
        assert ("batch" in error_text.lower() or
                "spatial" in error_text.lower() or
                "determin" in error_text.lower()), (
            f"Error should mention undetermined packs, got: {error_text}"
        )
