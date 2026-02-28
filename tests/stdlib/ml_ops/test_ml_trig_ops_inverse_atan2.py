#!/usr/bin/env python3
"""
Inverse trig and atan2 all-ranks tests. Split from test_ml_trig_ops.
"""

import numpy as np
from tests.test_utils import compile_and_execute


def test_inverse_trig_all_ranks(compiler, runtime):
    """Test inverse trigonometric functions (asin, acos, atan, asinh, acosh, atanh) across all ranks"""

    source = """use std::ml;
    // Test 0D (scalar)
    let x_scalar = 0.5;
    let asin_0d = std::ml::asin(x_scalar);
    let acos_0d = std::ml::acos(x_scalar);
    let atan_0d = std::ml::atan(x_scalar);
    let asinh_0d = std::ml::asinh(x_scalar);
    let acosh_0d = std::ml::acosh(1.5);
    let atanh_0d = std::ml::atanh(x_scalar);

    // Test 1D
    let x_1d = [0.0, 0.5, -0.5, 0.707];
    let asin_1d = std::ml::asin(x_1d);
    let acos_1d = std::ml::acos(x_1d);
    let atan_1d = std::ml::atan(x_1d);
    let asinh_1d = std::ml::asinh(x_1d);
    let acosh_1d = std::ml::acosh([1.0, 1.5, 2.0, 3.0]);
    let atanh_1d = std::ml::atanh([0.0, 0.5, -0.5, 0.707]);

    // Test 2D
    let x_2d = [[0.0, 0.5], [-0.5, 0.707]];
    let asin_2d = std::ml::asin(x_2d);
    let acos_2d = std::ml::acos(x_2d);
    let atan_2d = std::ml::atan(x_2d);
    let asinh_2d = std::ml::asinh(x_2d);
    let acosh_2d = std::ml::acosh([[1.0, 1.5], [2.0, 3.0]]);
    let atanh_2d = std::ml::atanh([[0.0, 0.5], [-0.5, 0.707]]);

    // Test 3D
    let x_3d = [[[0.0, 0.5], [-0.5, 0.707]], [[0.3, -0.3], [0.8, -0.8]]];
    let asin_3d = std::ml::asin(x_3d);
    let acos_3d = std::ml::acos(x_3d);
    let atan_3d = std::ml::atan(x_3d);
    let asinh_3d = std::ml::asinh(x_3d);
    let acosh_3d = std::ml::acosh([[[1.0, 1.5], [2.0, 3.0]], [[1.2, 2.5], [1.8, 4.0]]]);
    let atanh_3d = std::ml::atanh([[[0.0, 0.5], [-0.5, 0.707]], [[0.3, -0.3], [0.8, -0.8]]]);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Test failed: {result.errors}"

    # Verify 0D
    x_scalar = 0.5
    np.testing.assert_allclose(np.array(result.outputs['asin_0d']), np.arcsin(x_scalar), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['acos_0d']), np.arccos(x_scalar), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['atan_0d']), np.arctan(x_scalar), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['asinh_0d']), np.arcsinh(x_scalar), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['acosh_0d']), np.arccosh(1.5), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['atanh_0d']), np.arctanh(x_scalar), rtol=1e-5)

    # Verify 1D
    x_1d = np.array([0.0, 0.5, -0.5, 0.707], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['asin_1d']), np.arcsin(x_1d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['acos_1d']), np.arccos(x_1d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['atan_1d']), np.arctan(x_1d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['asinh_1d']), np.arcsinh(x_1d), rtol=1e-5)
    acosh_input = np.array([1.0, 1.5, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['acosh_1d']), np.arccosh(acosh_input), rtol=1e-5)
    atanh_input = np.array([0.0, 0.5, -0.5, 0.707], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['atanh_1d']), np.arctanh(atanh_input), rtol=1e-5)

    # Verify 2D
    x_2d = np.array([[0.0, 0.5], [-0.5, 0.707]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['asin_2d']), np.arcsin(x_2d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['acos_2d']), np.arccos(x_2d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['atan_2d']), np.arctan(x_2d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['asinh_2d']), np.arcsinh(x_2d), rtol=1e-5)
    acosh_2d = np.array([[1.0, 1.5], [2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['acosh_2d']), np.arccosh(acosh_2d), rtol=1e-5)
    atanh_2d = np.array([[0.0, 0.5], [-0.5, 0.707]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['atanh_2d']), np.arctanh(atanh_2d), rtol=1e-5)

    # Verify 3D
    x_3d = np.array([[[0.0, 0.5], [-0.5, 0.707]], [[0.3, -0.3], [0.8, -0.8]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['asin_3d']), np.arcsin(x_3d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['acos_3d']), np.arccos(x_3d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['atan_3d']), np.arctan(x_3d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['asinh_3d']), np.arcsinh(x_3d), rtol=1e-5)
    acosh_3d = np.array([[[1.0, 1.5], [2.0, 3.0]], [[1.2, 2.5], [1.8, 4.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['acosh_3d']), np.arccosh(acosh_3d), rtol=1e-5)
    atanh_3d = np.array([[[0.0, 0.5], [-0.5, 0.707]], [[0.3, -0.3], [0.8, -0.8]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['atanh_3d']), np.arctanh(atanh_3d), rtol=1e-5)


def test_atan2_all_ranks(compiler, runtime):
    """Test atan2 operation across all supported ranks"""

    source = """use std::ml;
    // Test 0D (scalar)
    let y_scalar = 1.0;
    let x_scalar = 1.0;
    let atan2_0d = std::ml::atan2(y_scalar, x_scalar);

    // Test 1D
    let y_1d = [1.0, 0.0, -1.0, 1.0];
    let x_1d = [1.0, 1.0, 1.0, 0.0];
    let atan2_1d = std::ml::atan2(y_1d, x_1d);

    // Test 2D
    let y_2d = [[1.0, 0.0], [-1.0, 1.0]];
    let x_2d = [[1.0, 1.0], [1.0, 0.0]];
    let atan2_2d = std::ml::atan2(y_2d, x_2d);

    // Test 3D
    let y_3d = [[[1.0, 0.0], [-1.0, 1.0]], [[0.0, 1.0], [1.0, -1.0]]];
    let x_3d = [[[1.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]]];
    let atan2_3d = std::ml::atan2(y_3d, x_3d);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Test failed: {result.errors}"

    # Verify 0D
    np.testing.assert_allclose(np.array(result.outputs['atan2_0d']), np.arctan2(1.0, 1.0), rtol=1e-5)

    # Verify 1D
    y_1d = np.array([1.0, 0.0, -1.0, 1.0], dtype=np.float32)
    x_1d = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['atan2_1d']), np.arctan2(y_1d, x_1d), rtol=1e-5)

    # Verify 2D
    y_2d = np.array([[1.0, 0.0], [-1.0, 1.0]], dtype=np.float32)
    x_2d = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['atan2_2d']), np.arctan2(y_2d, x_2d), rtol=1e-5)

    # Verify 3D
    y_3d = np.array([[[1.0, 0.0], [-1.0, 1.0]], [[0.0, 1.0], [1.0, -1.0]]], dtype=np.float32)
    x_3d = np.array([[[1.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['atan2_3d']), np.arctan2(y_3d, x_3d), rtol=1e-5)
