"""
Independent reference implementations for simulation accuracy tests.

Pure NumPy; no Einlang. Same equations, parameters, and discretization as the
corresponding .ein examples. Compared in-test (no stored files).

Every function here is used in test_simulation_accuracy.py (ALL_ACCURACY_EXAMPLES).
When adding a new simulation example, add a reference here and register it there.
"""

import numpy as np


def _laplacian_2d(u: np.ndarray) -> np.ndarray:
    """5-point Laplacian on interior; shape same as u (boundary not updated)."""
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4.0 * u[1:-1, 1:-1]
    return lap


def decay_reference() -> np.ndarray:
    """Exponential decay ODE. Same as examples/ode/decay.ein (50 steps, u[0]=u0, u[t]=u[t-1]*(1-k*dt))."""
    u0, k, dt = 1.0, 0.05, 0.1
    n = 50
    return np.array([u0 * np.exp(-k * (i * dt)) for i in range(n)], dtype=np.float64)


def euler_decay_reference() -> np.ndarray:
    """Euler decay trajectory. Same as numerics::diffeq euler_decay (51 points, u[0]=u0, u[t]=u[t-1]*(1-k*dt))."""
    u0, k, dt = 1.0, 0.05, 0.1
    n = 51
    return np.array([u0 * np.exp(-k * (i * dt)) for i in range(n)], dtype=np.float64)


def gradient_descent_2d_reference() -> np.ndarray:
    """Gradient descent 2D trajectory. Same as numerics::optim gradient_descent_2d (31 steps, A=2I, b=[1,1], alpha=0.25)."""
    A = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    b = np.array([1.0, 1.0], dtype=np.float64)
    alpha = 0.25
    nsteps = 31
    x = np.zeros((nsteps, 2), dtype=np.float64)
    for k in range(1, nsteps):
        r = A @ x[k - 1] - b
        x[k] = x[k - 1] - alpha * r
    return x


def value_iteration_quantecon_reference() -> np.ndarray:
    """Value iteration trajectory. Same as numerics::quantecon value_iteration (51 iters, 3 states, r,P,beta=0.95)."""
    beta = 0.95
    r = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    P = np.array([[0.5, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.7]], dtype=np.float64)
    nsteps = 51
    nstates = 3
    V = np.zeros((nsteps, nstates), dtype=np.float64)
    for k in range(1, nsteps):
        V[k] = r + beta * (P.T @ V[k - 1])
    return V


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


def lorenz_reference() -> np.ndarray:
    """Lorenz system, Euler. Same as examples/ode/lorenz.ein (t in 1..2000 => 2000 points)."""
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


def pendulum_reference() -> np.ndarray:
    """Simple pendulum Euler. Same as examples/ode/pendulum.ein (200 steps)."""
    g, L = 9.81, 1.0
    dt = 0.05
    theta0, omega0 = 0.2, 0.0
    nsteps = 200
    state = np.zeros((nsteps, 2), dtype=np.float64)
    state[0] = [theta0, omega0]
    for t in range(1, nsteps):
        state[t, 0] = state[t - 1, 0] + dt * state[t - 1, 1]
        state[t, 1] = state[t - 1, 1] - dt * (g / L) * np.sin(state[t - 1, 0])
    return state


def van_der_pol_reference() -> np.ndarray:
    """Van der Pol oscillator Euler, μ=1. Same as examples/ode/van_der_pol.ein (200 steps)."""
    mu, dt = 1.0, 0.05
    x0, y0 = 1.0, 0.0
    nsteps = 200
    state = np.zeros((nsteps, 2), dtype=np.float64)
    state[0] = [x0, y0]
    for t in range(1, nsteps):
        x, y = state[t - 1, 0], state[t - 1, 1]
        state[t, 0] = x + dt * y
        state[t, 1] = y + dt * mu * ((1.0 - x * x) * y - x)
    return state


def sir_reference() -> np.ndarray:
    """SIR model Euler. Same as examples/ode/sir.ein (100 steps)."""
    beta, gamma = 0.001, 0.1
    dt = 0.2
    S0, I0, R0 = 999.0, 1.0, 0.0
    nsteps = 100
    state = np.zeros((nsteps, 3), dtype=np.float64)
    state[0] = [S0, I0, R0]
    for t in range(1, nsteps):
        S, I, R = state[t - 1, 0], state[t - 1, 1], state[t - 1, 2]
        state[t, 0] = S - dt * beta * S * I
        state[t, 1] = I + dt * (beta * S * I - gamma * I)
        state[t, 2] = R + dt * gamma * I
    return state


def harmonic_reference() -> np.ndarray:
    """Harmonic oscillator Euler. Same as examples/ode/harmonic.ein (200 steps)."""
    omega, dt = 1.0, 0.05
    x0, v0 = 1.0, 0.0
    nsteps = 200
    state = np.zeros((nsteps, 2), dtype=np.float64)
    state[0] = [x0, v0]
    for t in range(1, nsteps):
        state[t, 0] = state[t - 1, 0] + dt * state[t - 1, 1]
        state[t, 1] = state[t - 1, 1] - dt * (omega ** 2) * state[t - 1, 0]
    return state


def logistic_reference() -> np.ndarray:
    """Logistic map. Same as examples/recurrence/logistic.ein (50 steps)."""
    r = 3.7
    n = 50
    x = np.zeros(n, dtype=np.float64)
    x[0] = 0.5
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x


def gradient_descent_reference() -> np.ndarray:
    """Gradient descent for quadratic. Same as examples/optimization/gradient_descent.ein (30 iters, 2D)."""
    A = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    b = np.array([1.0, 1.0], dtype=np.float64)
    alpha = 0.25
    nsteps = 30
    x = np.zeros((nsteps, 2), dtype=np.float64)
    for k in range(1, nsteps):
        r = A @ x[k - 1] - b
        x[k] = x[k - 1] - alpha * r
    return x


def power_iteration_reference() -> np.ndarray:
    """Power iteration for dominant eigenvector. Same as examples/recurrence/power_iteration.ein (20 iters, 2D)."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    nsteps = 20
    v = np.zeros((nsteps, 2), dtype=np.float64)
    v[0] = [1.0, 0.0]
    for k in range(1, nsteps):
        Av = A @ v[k - 1]
        n = np.linalg.norm(Av)
        v[k] = Av / n
    return v


def markov_stationary_reference() -> np.ndarray:
    """Stationary distribution by power iteration ψ = ψ P. Same as examples/recurrence/markov_stationary.ein (50 iters, 3 states)."""
    P = np.array([[0.9, 0.1, 0.0], [0.2, 0.6, 0.2], [0.0, 0.1, 0.9]], dtype=np.float64)
    nsteps, n = 50, 3
    psi = np.zeros((nsteps, n), dtype=np.float64)
    psi[0] = 1.0 / 3.0
    for k in range(1, nsteps):
        psi[k] = psi[k - 1] @ P
    return psi


def lotka_volterra_reference() -> np.ndarray:
    """Lotka-Volterra, Euler. Same as examples/ode/lotka_volterra.ein (t in 1..500 => 500 total)."""
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
    """1D heat, Dirichlet BC, explicit Euler. Same as examples/pde_1d/heat_1d.ein (t in 1..200 => 200 total)."""
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
    """Linear ODE du/dt = A*u, Euler. Same as examples/ode/linear.ein (t in 1..500 => 500 total)."""
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


def value_iteration_reference() -> np.ndarray:
    """Bellman value iteration. Same as examples/value_iteration/main.ein (50 iters, 3 states)."""
    beta = 0.95
    r = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    P = np.array([[0.5, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.7]], dtype=np.float64)
    nsteps = 50
    nstates = 3
    V = np.zeros((nsteps, nstates), dtype=np.float64)
    for k in range(1, nsteps):
        V[k] = r + beta * (P.T @ V[k - 1])
    return V


def fibonacci_reference() -> np.ndarray:
    """Fibonacci. Same as examples/recurrence/fibonacci.ein (n in 2..25 => 25 elements)."""
    n = 25
    fib = np.zeros(n, dtype=np.int64)
    fib[0], fib[1] = 0, 1
    for i in range(2, n):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib


def advection_1d_reference() -> np.ndarray:
    """1D advection, upwind, periodic. Same as examples/pde_1d/advection_1d.ein (80 steps, 40 points)."""
    r = 0.5
    nx = 40
    nsteps = 80
    u = np.zeros((nsteps, nx), dtype=np.float64)
    u[0, 10] = 1.0
    for t in range(1, nsteps):
        u[t, 1:] = u[t - 1, 1:] - r * (u[t - 1, 1:] - u[t - 1, :-1])
        u[t, 0] = u[t - 1, 0] - r * (u[t - 1, 0] - u[t - 1, -1])
    return u


def random_walk_reference() -> np.ndarray:
    """1D random walk with fixed steps. Same as examples/recurrence/random_walk.ein (21 points)."""
    steps = np.array([1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1], dtype=np.float64)
    x = np.zeros(21, dtype=np.float64)
    x[0] = 0
    for t in range(1, 21):
        x[t] = x[t - 1] + steps[t - 1]
    return x


def savings_reference() -> np.ndarray:
    """Savings / compound interest. Same as examples/finance/savings.ein (61 months)."""
    initial = 1000.0
    r = 0.05 / 12.0
    deposit = 100.0
    nsteps = 61
    b = np.zeros(nsteps, dtype=np.float64)
    b[0] = initial
    for t in range(1, nsteps):
        b[t] = b[t - 1] * (1.0 + r) + deposit
    return b


def projected_gradient_reference() -> np.ndarray:
    """Projected gradient descent (box [0,1]^2). Same as examples/optimization/projected_gradient.ein (20 iters, 2D)."""
    c = np.array([1.5, 0.3], dtype=np.float64)
    alpha = 0.5
    nsteps = 20
    x = np.zeros((nsteps, 2), dtype=np.float64)
    for k in range(1, nsteps):
        u = x[k - 1] - alpha * (x[k - 1] - c)
        x[k] = np.clip(u, 0.0, 1.0)
    return x


def rosenbrock_reference() -> np.ndarray:
    """Rosenbrock gradient descent. Same as examples/optimization/rosenbrock.ein (2001 iters, 2D)."""
    a, b = 1.0, 100.0
    alpha = 0.001
    nsteps = 2001
    x = np.zeros((nsteps, 2), dtype=np.float64)
    for k in range(1, nsteps):
        x1, x2 = x[k - 1, 0], x[k - 1, 1]
        g1 = -2.0 * (a - x1) - 4.0 * b * x1 * (x2 - x1 * x1)
        g2 = 2.0 * b * (x2 - x1 * x1)
        x[k, 0] = x1 - alpha * g1
        x[k, 1] = x2 - alpha * g2
    return x


def exponential_smoothing_reference() -> np.ndarray:
    """Simple exponential smoothing. Same as examples/time_series/exponential_smoothing.ein (30 points)."""
    y = np.array(
        [2.0, 3.0, 2.5, 4.0, 3.5, 5.0, 4.2, 5.5, 5.0, 6.0, 5.5, 7.0, 6.2, 7.5, 7.0, 8.0, 7.5, 9.0, 8.2, 9.5, 9.0, 10.0, 9.5, 11.0, 10.2, 11.5, 11.0, 12.0, 11.5, 13.0],
        dtype=np.float64,
    )
    alpha = 0.3
    n = len(y)
    s = np.zeros(n, dtype=np.float64)
    s[0] = y[0]
    for t in range(1, n):
        s[t] = alpha * y[t] + (1.0 - alpha) * s[t - 1]
    return s


def mccall_reference() -> np.ndarray:
    """McCall job search: value function iteration. Same as examples/job_search/mccall.ein (100 iters, 11 wages)."""
    n = 10  # 11 wage points indices 0..10
    c = 25.0
    beta = 0.99
    w = np.array([10.0 + i * 5.0 for i in range(11)], dtype=np.float64)
    p = np.ones(11, dtype=np.float64) / 11.0
    nsteps = 100
    V = np.zeros((nsteps, 11), dtype=np.float64)
    V[0] = w / (1.0 - beta)
    for k in range(1, nsteps):
        cv = c + beta * np.dot(V[k - 1], p)
        V[k] = np.maximum(w / (1.0 - beta), cv)
    return V


def fitzhugh_nagumo_reference() -> np.ndarray:
    """FitzHugh-Nagumo 2D ODE, Euler. Same as examples/ode/fitzhugh_nagumo.ein (3000 steps)."""
    a, b, c = 0.7, 0.8, 10.0
    I_ext = 0.5
    dt = 0.01
    v0, u0 = 0.0, 0.0
    nsteps = 3000
    state = np.zeros((nsteps, 2), dtype=np.float64)
    state[0] = [v0, u0]
    for t in range(1, nsteps):
        v, u = state[t - 1, 0], state[t - 1, 1]
        state[t, 0] = v + dt * c * (v - v**3 / 3.0 - u + I_ext)
        state[t, 1] = u + dt * (v - b * u + a)
    return state


def lorenz96_reference() -> np.ndarray:
    """Lorenz 96 chaotic ODE, Euler, periodic indices. Same as examples/ode/lorenz96.ein (500 steps, N=5)."""
    N = 5
    F = 8.0
    dt = 0.01
    nsteps = 500
    X = np.zeros((nsteps, N), dtype=np.float64)
    X[0] = F
    X[0, 0] = F + 0.01
    for t in range(1, nsteps):
        for i in range(N):
            ip1 = (i + 1) % N
            im1 = (i + 4) % N
            im2 = (i + 3) % N
            X[t, i] = (
                X[t - 1, i]
                + dt * ((X[t - 1, ip1] - X[t - 1, im2]) * X[t - 1, im1] - X[t - 1, i] + F)
            )
    return X
