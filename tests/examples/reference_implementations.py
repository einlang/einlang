"""
Independent reference implementations for simulation accuracy tests.

Pure NumPy; no Einlang. Same equations, parameters, and discretization as the
corresponding .ein examples. Compared in-test (no stored files).
"""

import numpy as np


def _laplacian_2d(u: np.ndarray) -> np.ndarray:
    """5-point Laplacian on interior; shape same as u (boundary not updated)."""
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4.0 * u[1:-1, 1:-1]
    return lap


def wave_2d_reference() -> np.ndarray:
    """2D wave equation, leapfrog. Same as examples/wave_2d/main.ein."""
    r = 0.5
    cx, cy = 20, 20
    sigma2 = 50.0
    nx, ny = 40, 40
    nsteps = 200
    i = np.arange(nx, dtype=np.float64)
    j = np.arange(ny, dtype=np.float64)
    I, J = np.meshgrid(i, j)
    h = np.zeros((nsteps, ny, nx), dtype=np.float64)
    h[0] = 10.0 * np.exp(-((I - cx) ** 2 + (J - cy) ** 2) / sigma2)
    h[1] = h[0]
    for t in range(2, nsteps):
        lap = _laplacian_2d(h[t - 1])
        h[t, 1:-1, 1:-1] = 2.0 * h[t - 1, 1:-1, 1:-1] - h[t - 2, 1:-1, 1:-1] + r * lap[1:-1, 1:-1]
    return h


def heat_minimal_reference() -> np.ndarray:
    """2D heat equation, explicit Euler. Same as test inline heat_minimal (25 steps, 11x11)."""
    r = 0.2
    cx, cy = 5, 5
    R2 = 4.0
    ny, nx = 11, 11
    nsteps = 25
    i = np.arange(nx, dtype=np.float64)
    j = np.arange(ny, dtype=np.float64)
    I, J = np.meshgrid(i, j)
    d2 = (I - cx) ** 2 + (J - cy) ** 2
    u = np.zeros((nsteps, ny, nx), dtype=np.float64)
    u[0] = np.where(d2 <= R2, 10.0 * (1.0 - d2 / R2), 0.0)
    for t in range(1, nsteps):
        u[t] = u[t - 1].copy()
        lap = _laplacian_2d(u[t - 1])
        u[t, 1:-1, 1:-1] = u[t - 1, 1:-1, 1:-1] + r * lap[1:-1, 1:-1]
    return u


def reaction_diffusion_reference() -> np.ndarray:
    """Gray-Scott. Same as examples/reaction_diffusion/main.ein."""
    Du, Dv = 0.1, 0.05
    f, k = 0.055, 0.062
    dt = 1.0
    n = 128
    nsteps = 500
    state = np.zeros((nsteps, 2, n, n), dtype=np.float64)
    state[0, 0] = 1.0
    state[0, 1, 60:69, 60:69] = 0.25
    for t in range(1, nsteps):
        state[t] = state[t - 1].copy()
        U, V = state[t - 1, 0], state[t - 1, 1]
        lap_u = _laplacian_2d(U)
        lap_v = _laplacian_2d(V)
        uv2 = U * V * V
        state[t, 0, 1:-1, 1:-1] = U[1:-1, 1:-1] + dt * (Du * lap_u[1:-1, 1:-1] - uv2[1:-1, 1:-1] + f * (1.0 - U[1:-1, 1:-1]))
        state[t, 1, 1:-1, 1:-1] = V[1:-1, 1:-1] + dt * (Dv * lap_v[1:-1, 1:-1] + uv2[1:-1, 1:-1] - (f + k) * V[1:-1, 1:-1])
    return state


def lorenz_reference() -> np.ndarray:
    """Lorenz system, Euler. Same as examples/lorenz/main.ein (t in 1..2000 => 2000 points)."""
    sigma, rho, beta = 10.0, 28.0, 2.666666666666667
    dt = 0.01
    x0, y0, z0 = 1.0, 1.0, 1.0
    nsteps = 2000  # t=0 + t in 1..2000 (exclusive end => 2000 total)
    u = np.zeros((nsteps, 3), dtype=np.float64)
    u[0] = [x0, y0, z0]
    for t in range(1, nsteps):
        x, y, z = u[t - 1, 0], u[t - 1, 1], u[t - 1, 2]
        u[t, 0] = x + dt * (sigma * (y - x))
        u[t, 1] = y + dt * (x * (rho - z) - y)
        u[t, 2] = z + dt * (x * y - beta * z)
    return u


def lotka_volterra_reference() -> np.ndarray:
    """Lotka-Volterra, Euler. Same as examples/lotka_volterra/main.ein (t in 1..500 => 500 total)."""
    a, b, c, d = 1.0, 0.5, 1.0, 0.5
    dt = 0.05
    u0, v0 = 2.0, 1.0
    nsteps = 500
    state = np.zeros((nsteps, 2), dtype=np.float64)
    state[0] = [u0, v0]
    for t in range(1, nsteps):
        u, v = state[t - 1, 0], state[t - 1, 1]
        state[t, 0] = u + dt * (a * u - b * u * v)
        state[t, 1] = v + dt * (-c * v + d * u * v)
    return state


def heat_1d_reference() -> np.ndarray:
    """1D heat, Dirichlet BC, explicit Euler. Same as examples/heat_1d/main.ein (t in 1..200 => 200 total)."""
    r = 0.2
    nx = 41
    nsteps = 200
    u = np.zeros((nsteps, nx), dtype=np.float64)
    u[0, 0] = 0.0
    u[0, 40] = 0.0
    u[0, 1:40] = 0.0
    u[0, 20] = 10.0
    for t in range(1, nsteps):
        u[t, 0] = 0.0
        u[t, 40] = 0.0
        u[t, 1:40] = u[t - 1, 1:40] + r * (
            u[t - 1, 0:39] - 2.0 * u[t - 1, 1:40] + u[t - 1, 2:41]
        )
    return u  # shape (200, 41)


def linear_ode_reference() -> np.ndarray:
    """Linear ODE du/dt = A*u, Euler. Same as examples/linear_ode/main.ein (t in 1..500 => 500 total)."""
    dt = 0.01
    A = np.array([[-1.0, 1.0], [0.5, -1.5]], dtype=np.float64)
    u0 = np.array([1.0, 0.0], dtype=np.float64)
    nsteps = 500
    u = np.zeros((nsteps, 2), dtype=np.float64)
    u[0] = u0
    for t in range(1, nsteps):
        u[t] = u[t - 1] + dt * (A @ u[t - 1])
    return u


def brusselator_reference() -> np.ndarray:
    """Brusselator PDE. Same as examples/brusselator/main.ein (t in 1..300 => 300 total)."""
    A, B = 1.0, 2.0
    alpha = 0.02
    dt = 0.2
    n = 64
    nsteps = 300
    state = np.zeros((nsteps, 2, n, n), dtype=np.float64)
    state[0, 0] = 1.0
    state[0, 1] = 0.0
    state[0, 1, 30:35, 30:35] = 0.5
    for t in range(1, nsteps):
        state[t] = state[t - 1].copy()
        U, V = state[t - 1, 0], state[t - 1, 1]
        lap_u = _laplacian_2d(U)
        lap_v = _laplacian_2d(V)
        u2v = U * U * V
        state[t, 0, 1:-1, 1:-1] = (
            U[1:-1, 1:-1]
            + dt
            * (
                B
                + u2v[1:-1, 1:-1]
                - (A + 1.0) * U[1:-1, 1:-1]
                + alpha * lap_u[1:-1, 1:-1]
            )
        )
        state[t, 1, 1:-1, 1:-1] = (
            V[1:-1, 1:-1]
            + dt
            * (
                A * U[1:-1, 1:-1]
                - u2v[1:-1, 1:-1]
                + alpha * lap_v[1:-1, 1:-1]
            )
        )
    return state
