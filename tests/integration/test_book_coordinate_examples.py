import numpy as np

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime
from tests.test_utils import compile_and_execute


def _run(source: str):
    result = compile_and_execute(
        source,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )
    assert result.success, result.errors if result.errors else result.error
    return result.outputs


def test_chapter_6_stable_softmax_keeps_output_and_scan_roles_separate():
    outputs = _run(
        """
use std::math::exp;

let x = [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]];

let output[b in 0..2, j in 0..3] =
    exp(x[b, j] - max[q](x[b, q]))
    / sum[k](exp(x[b, k] - max[q](x[b, q])));

let row_sum[b in 0..2] = sum[j](output[b, j]);
"""
    )

    expected = np.exp(np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32))
    expected = expected / expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(outputs["output"], expected, rtol=1e-6)
    np.testing.assert_allclose(outputs["row_sum"], np.array([1.0, 1.0], dtype=np.float32), rtol=1e-6)


def test_chapter_16_dense_top1_route_and_hard_onehot():
    outputs = _run(
        """
use std::math::exp;

let batch = 2;
let seq_len = 2;
let num_experts = 3;

let gate_score = [
    [[1.0, 3.0, 2.0], [4.0, 1.0, 0.0]],
    [[0.5, 2.5, 1.5], [1.0, 1.0, 5.0]]
];

let gate_max[b in 0..batch, t in 0..seq_len] =
    max[e in 0..num_experts](gate_score[b, t, e]);

let gate_exp[b in 0..batch, t in 0..seq_len, e in 0..num_experts] =
    exp(gate_score[b, t, e] - gate_max[b, t]);

let gate_sum[b in 0..batch, t in 0..seq_len] =
    sum[e in 0..num_experts](gate_exp[b, t, e]);

let gate_prob[b in 0..batch, t in 0..seq_len, e in 0..num_experts] =
    gate_exp[b, t, e] / gate_sum[b, t];

let route[b in 0..batch, t in 0..seq_len] =
    argmax[e](gate_prob[b, t, e]);

let hard_onehot[b in 0..batch, t in 0..seq_len, e in 0..num_experts] =
    if e == route[b, t] { 1.0 } else { 0.0 };
"""
    )

    np.testing.assert_array_equal(outputs["route"], np.array([[1, 0], [1, 2]], dtype=np.int32))
    np.testing.assert_array_equal(
        outputs["hard_onehot"],
        np.array(
            [
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_chapter_16_top1_capacity_router_tracks_slots_and_keep_mask():
    outputs = _run(
        """
fn route_top1_with_capacity(gate_prob, capacity: i32) {
    let batch = len(gate_prob);
    let seq_len = len(gate_prob[0]);

    let route[b in 0..batch, t in 0..seq_len] =
        argmax[e](gate_prob[b, t, e]);

    let slot[b in 0..batch, t in 0..seq_len] =
        sum[bb in 0..b, tt in 0..seq_len](
            if route[bb, tt] == route[b, t] { 1 } else { 0 }
        )
        +
        sum[tt in 0..t](
            if route[b, tt] == route[b, t] { 1 } else { 0 }
        );

    let keep[b in 0..batch, t in 0..seq_len] =
        slot[b, t] < capacity;

    (route, slot, keep)
}

let gate_prob = [
    [[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]],
    [[0.6, 0.4], [0.2, 0.8], [0.9, 0.1]]
];

let (route, slot, keep) = route_top1_with_capacity(gate_prob, 2);
"""
    )

    np.testing.assert_array_equal(outputs["route"], np.array([[0, 1, 0], [0, 1, 0]], dtype=np.int32))
    np.testing.assert_array_equal(outputs["slot"], np.array([[0, 0, 1], [2, 1, 3]], dtype=np.int32))
    np.testing.assert_array_equal(
        outputs["keep"],
        np.array([[True, True, True], [False, True, False]]),
    )
