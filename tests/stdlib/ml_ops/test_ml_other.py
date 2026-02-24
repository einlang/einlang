#!/usr/bin/env python3
"""
Comprehensive accuracy tests for std::ml operations against ONNX/NumPy reference implementations.
Tests all operations added to ml.ein for correctness.
"""

import pytest
import numpy as np
try:
    import scipy.special
except ImportError:
    scipy = None
from tests.test_utils import compile_and_execute, assert_float_close
# ===========================================================================
# Activation Function Tests
# ===========================================================================



# ===========================================================================
# Other Operation Tests
# Clustered tests for efficiency - all other ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_repeat_interleave_accuracy(compiler, runtime):
    """Test repeat_interleave operation"""
    source = """
    use std::ml;
    let x = [[1.0, 2.0, 3.0]];
    let result = std::ml::ml_ex::repeat_interleave(x, 2);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    expected = np.repeat(x, 2, axis=1)  # [[1, 1, 2, 2, 3, 3]]
    actual = np.array(result.outputs['result'])

    np.testing.assert_allclose(actual, expected, rtol=1e-6)
