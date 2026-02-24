"""
Tests for ONNX Attention Operations - one compile/execute for speed.
"""

import numpy as np
import pytest
from tests.test_utils import compile_and_execute


def test_attention_all_variants(compiler, runtime):
    """Test attention_dummy and multi_head_attention_simple (1, 2, 4 heads) in one run"""
    source = """use std::ml;
    let q_d = [[[1.0, 2.0], [3.0, 4.0]]];
    let k_d = [[[1.0, 2.0], [3.0, 4.0]]];
    let v_d = [[[1.0, 2.0], [3.0, 4.0]]];
    let scale_d = 0.7071067811865476;
    let out_dummy = std::ml::attention_dummy(q_d, k_d, v_d, scale_d);
    let q_2h = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]];
    let k_2h = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]];
    let v_2h = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]];
    let out_2h = std::ml::multi_head_attention_simple(q_2h, k_2h, v_2h, 2, scale_d);
    let q_1h = [[[1.0, 2.0], [3.0, 4.0]]];
    let k_1h = [[[1.0, 2.0], [3.0, 4.0]]];
    let v_1h = [[[1.0, 2.0], [3.0, 4.0]]];
    let out_1h = std::ml::multi_head_attention_simple(q_1h, k_1h, v_1h, 1, scale_d);
    let q_4h = [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]];
    let k_4h = [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]];
    let v_4h = [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]];
    let out_4h = std::ml::multi_head_attention_simple(q_4h, k_4h, v_4h, 4, 0.5);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    out_dummy = np.array(result.outputs['out_dummy'])
    assert out_dummy.shape == (1, 2, 2), f"out_dummy shape should be (1, 2, 2), got {out_dummy.shape}"
    assert np.all(np.isfinite(out_dummy)) and np.all(out_dummy >= -10.0) and np.all(out_dummy <= 10.0)

    out_2h = np.array(result.outputs['out_2h'])
    assert out_2h.shape == (1, 2, 4), f"out_2h shape should be (1, 2, 4), got {out_2h.shape}"
    assert np.all(np.isfinite(out_2h)) and np.all(out_2h >= -10.0) and np.all(out_2h <= 10.0)

    out_1h = np.array(result.outputs['out_1h'])
    assert out_1h.shape == (1, 2, 2), f"out_1h shape should be (1, 2, 2), got {out_1h.shape}"
    assert np.all(np.isfinite(out_1h))

    out_4h = np.array(result.outputs['out_4h'])
    assert out_4h.shape == (1, 1, 8), f"out_4h shape should be (1, 1, 8), got {out_4h.shape}"
    assert np.all(np.isfinite(out_4h))
