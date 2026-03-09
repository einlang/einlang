"""
Tests for std::numerics (ode, optim, dp). General-purpose APIs; no hardcoded sizes.
Julia-style: n_steps, n_states, n_iters are parameters.
"""

import numpy as np
import pytest

from ..test_utils import compile_and_execute


class TestNumericsOde:
    """std::numerics::ode - euler_decay with n_steps parameter."""

    def test_euler_decay_step(self, compiler, runtime):
        source = """
        use std::numerics::ode;
        let u1 = ode::euler_decay_step(1.0, 0.05, 0.1);
        let u2 = ode::euler_decay_step(u1, 0.05, 0.1);
        print(u2);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        u2 = result.outputs.get("u2")
        assert u2 is not None
        assert abs(float(u2) - 0.990025) < 1e-5

    def test_euler_decay_variable_steps(self, compiler, runtime):
        source = """
        use std::numerics::ode;
        let u = ode::euler_decay(1.0, 0.05, 0.1, 50);
        print(u);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        u = np.asarray(result.outputs["u"])
        assert u.shape == (51,), f"expected (51,), got {u.shape}"
        assert abs(u[0] - 1.0) < 1e-6
        ref_50 = 1.0 * np.exp(-0.05 * 50 * 0.1)
        assert abs(u[50] - ref_50) < 5e-3  # Euler vs analytical

    def test_euler_decay_different_n_steps(self, compiler, runtime):
        source = """
        use std::numerics::ode;
        let u = ode::euler_decay(1.0, 0.05, 0.1, 10);
        print(u);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        u = np.asarray(result.outputs["u"])
        assert u.shape == (11,), f"expected (11,), got {u.shape}"

    def test_midpoint_decay_step(self, compiler, runtime):
        source = """
        use std::numerics::ode;
        let u1 = ode::midpoint_decay_step(1.0, 0.05, 0.1);
        print(u1);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        u1 = float(result.outputs["u1"])
        # midpoint: u_mid = 1*(1 - 0.5*0.05*0.1)=0.9975, u1 = 1 + 0.1*(-0.05*0.9975) = 0.9950125
        assert abs(u1 - 0.9950125) < 1e-5

    def test_midpoint_decay_trajectory(self, compiler, runtime):
        source = """
        use std::numerics::ode;
        let u = ode::midpoint_decay(1.0, 0.05, 0.1, 10);
        print(u);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        u = np.asarray(result.outputs["u"])
        assert u.shape == (11,)
        assert abs(u[0] - 1.0) < 1e-6

    def test_rk4_decay_step(self, compiler, runtime):
        source = """
        use std::numerics::ode;
        let u1 = ode::rk4_decay_step(1.0, 0.05, 0.1);
        print(u1);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        u1 = float(result.outputs["u1"])
        # RK4 should be very close to exact exp(-0.05*0.1)
        assert abs(u1 - np.exp(-0.005)) < 1e-8

    def test_rk4_decay_trajectory(self, compiler, runtime):
        source = """
        use std::numerics::ode;
        let u = ode::rk4_decay(1.0, 0.05, 0.1, 5);
        print(u);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        u = np.asarray(result.outputs["u"])
        assert u.shape == (6,)

    def test_euler_linear_step(self, compiler, runtime):
        source = """
        use std::numerics::ode;
        let u = [1.0, 0.5];
        let A = [[-0.05, 0.0], [0.0, -0.1]];
        let u1 = ode::euler_linear_step(u, A, 0.1, 2);
        print(u1);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        u1 = np.asarray(result.outputs["u1"])
        assert u1.shape == (2,)
        # u_next[i] = u[i] + dt*(A*u)[i]: [1 - 0.005, 0.5 - 0.005] = [0.995, 0.495]
        assert abs(u1[0] - 0.995) < 1e-5
        assert abs(u1[1] - 0.495) < 1e-5

    def test_euler_linear_trajectory(self, compiler, runtime):
        source = """
        use std::numerics::ode;
        let u0 = [1.0, 0.5];
        let A = [[-0.05, 0.0], [0.0, -0.1]];
        let u = ode::euler_linear(u0, A, 0.1, 5, 2);
        print(u);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        u = np.asarray(result.outputs["u"])
        assert u.shape == (6, 2)
        assert np.allclose(u[0], [1.0, 0.5])


class TestNumericsOptim:
    """std::numerics::optim - gradient_descent_2d with n_steps parameter."""

    def test_gradient_descent_2d_variable_steps(self, compiler, runtime):
        source = """
        use std::numerics::optim;
        let A = [[2.0, 0.0], [0.0, 2.0]];
        let b = [1.0, 1.0];
        let x_traj = optim::gradient_descent_2d(A, b, 0.25, 30);
        print(x_traj);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        x = np.asarray(result.outputs["x_traj"])
        assert x.shape == (31, 2), f"expected (31, 2), got {x.shape}"
        assert np.allclose(x[0], [0.0, 0.0])
        assert np.allclose(x[1], [0.25, 0.25])
        assert np.allclose(x[2], [0.375, 0.375])

    def test_quadratic_value_and_gradient_2d(self, compiler, runtime):
        source = """
        use std::numerics::optim;
        let A = [[2.0, 0.0], [0.0, 2.0]];
        let b = [1.0, 1.0];
        let x = [0.5, 0.5];
        let val = optim::quadratic_value_2d(x, A, b);
        let grad = optim::quadratic_gradient_2d(x, A, b);
        print(val);
        print(grad);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        val = float(result.outputs["val"])
        grad = np.asarray(result.outputs["grad"])
        # 0.5*x'Ax - b'x at [0.5,0.5]: 0.5*2*(0.25+0.25) - (0.5+0.5) = 0.5 - 1 = -0.5
        assert abs(val - (-0.5)) < 1e-6
        assert np.allclose(grad, [0.0, 0.0])  # at optimum

    def test_gradient_descent_2d_from(self, compiler, runtime):
        source = """
        use std::numerics::optim;
        let A = [[2.0, 0.0], [0.0, 2.0]];
        let b = [1.0, 1.0];
        let x0 = [1.0, 1.0];
        let x_traj = optim::gradient_descent_2d_from(x0, A, b, 0.25, 3);
        print(x_traj);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        x = np.asarray(result.outputs["x_traj"])
        assert x.shape == (4, 2)
        assert np.allclose(x[0], [1.0, 1.0])

    def test_quadratic_residual_norm_2d(self, compiler, runtime):
        source = """
        use std::numerics::optim;
        let A = [[2.0, 0.0], [0.0, 2.0]];
        let b = [1.0, 1.0];
        let x_opt = [0.5, 0.5];
        let r = optim::quadratic_residual_norm_2d(x_opt, A, b);
        print(r);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        r = float(result.outputs["r"])
        assert abs(r) < 1e-10

    def test_least_squares_slope_intercept(self, compiler, runtime):
        source = """
        use std::numerics::optim;
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.5, 3.7, 4.9, 6.1, 7.3];
        let coef = optim::least_squares_slope_intercept(x, y, 5);
        print(coef);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        coef = np.asarray(result.outputs["coef"])
        assert coef.shape == (2,)
        # y ≈ 1.2*x + 1.3
        assert abs(coef[0] - 1.2) < 0.02
        assert abs(coef[1] - 1.3) < 0.02


class TestNumericsDp:
    """std::numerics::dp - value_iteration with n_states, n_iters parameters."""

    def test_value_iteration_3_states_50_iters(self, compiler, runtime):
        source = """
        use std::numerics::dp;
        let r = [0.0, 1.0, 2.0];
        let P = [[0.5, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.7]];
        let V = dp::value_iteration(r, P, 0.95, 3, 50);
        print(V);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        V = np.asarray(result.outputs["V"])
        assert V.shape == (51, 3), f"expected (51, 3), got {V.shape}"
        assert np.allclose(V[0], [0.0, 0.0, 0.0])
        assert np.allclose(V[1], [0.0, 1.0, 2.0])

    def test_expected_value_at_state(self, compiler, runtime):
        source = """
        use std::numerics::dp;
        let V = [1.0, 2.0, 3.0];
        let P = [[0.5, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.7]];
        let ev0 = dp::expected_value_at_state(V, P, 0, 3);
        let ev1 = dp::expected_value_at_state(V, P, 1, 3);
        print(ev0);
        print(ev1);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        ev0 = float(result.outputs["ev0"])  # P[:,0]'*V = 0.5*1+0.3*2+0.2*3 = 0.5+0.6+0.6 = 1.7
        ev1 = float(result.outputs["ev1"])  # 0.2*1+0.5*2+0.3*3 = 0.2+1+0.9 = 2.1
        assert abs(ev0 - 1.7) < 1e-6
        assert abs(ev1 - 2.1) < 1e-6

    def test_discounted_return(self, compiler, runtime):
        source = """
        use std::numerics::dp;
        let rewards = [1.0, 2.0, 3.0, 4.0];
        let total = dp::discounted_return(rewards, 0.9, 4);
        print(total);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        total = float(result.outputs["total"])
        # 1 + 0.9*2 + 0.81*3 + 0.729*4 = 1 + 1.8 + 2.43 + 2.916 = 8.166
        expected = 1.0 + 0.9*2 + 0.9**2*3 + 0.9**3*4
        assert abs(total - expected) < 1e-5

    def test_bellman_residual(self, compiler, runtime):
        source = """
        use std::numerics::dp;
        let V = [1.0, 2.0, 3.0];
        let r = [0.0, 1.0, 2.0];
        let P = [[0.5, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.7]];
        let res = dp::bellman_residual(V, r, P, 0.95, 3);
        print(res);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, result.errors
        res = float(result.outputs["res"])
        assert res >= 0.0
        # At fixed point residual would be 0; V above is not fixed point
        assert res < 10.0
