#!/usr/bin/env python3
"""
Real NumPy application: Discrete-time Kalman filter for constant-velocity (CV) motion model.

Use case: Track position and velocity from noisy position-only measurements (e.g. radar,
GPS, or vision). The filter predicts state (position, velocity), then corrects with each
measurement. This is a standard application in navigation, target tracking, and sensor
fusion.

Model:
  State x = [position, velocity]'
  Dynamics: x_{k+1} = F x_k + w_k,  F = [[1, dt], [0, 1]],  w ~ N(0, Q)
  Measurement: z_k = H x_k + v_k,   H = [1, 0],  v ~ N(0, R)

Equations (predict then update per time step):
  Predict:  x_pred = F @ x,   P_pred = F @ P @ F.T + Q
  Update:   y = z - H @ x_pred (innovation)
            S = H @ P_pred @ H.T + R  (innovation covariance, scalar for 1D measurement)
            K = P_pred @ H.T / S     (Kalman gain)
            x = x_pred + K * y
            P = (I - K @ H) @ P_pred

Reference: FilterPy, Kalman and Bayesian Filters in Python (rlabbe), wikipedia.
This file is the canonical NumPy implementation; the Einlang version in main.ein
reproduces the same algorithm and outputs for comparison.
"""

import numpy as np
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# Configuration: constant-velocity model parameters
# -----------------------------------------------------------------------------
DT = 0.1          # Time step (s)
Q_POS = 0.01      # Process noise variance for position (m^2)
Q_VEL = 0.1       # Process noise variance for velocity (m^2/s^2)
R_MEAS = 1.0      # Measurement noise variance (m^2)
N_STEPS = 100     # Number of time steps

# Initial state: position (m), velocity (m/s)
X0 = np.array([0.0, 1.0], dtype=np.float64)
# Initial covariance: uncertain position and velocity
P0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)


def build_F(dt: float) -> np.ndarray:
    """State transition matrix for constant-velocity model."""
    return np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)


def build_Q(dt: float, q_pos: float, q_vel: float) -> np.ndarray:
    """Process noise covariance (discrete white noise acceleration model)."""
    # Q from continuous white noise: integral of G Q_c G' dt, G = [dt^2/2, dt]'
    q = np.array([
        [dt**4 / 4 * q_pos + dt**2 * q_vel, dt**3 / 2 * q_vel],
        [dt**3 / 2 * q_vel, dt**2 * q_vel]
    ], dtype=np.float64)
    return q


def predict(
    x: np.ndarray,
    P: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict state and covariance one step ahead."""
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def update(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    z: float,
    H: np.ndarray,
    R: float
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Update state and covariance with measurement z.
    Returns (x, P, innovation, innovation_covariance).
    """
    H = np.asarray(H).reshape(1, -1)
    y = z - (H @ x_pred).item()
    S = (H @ P_pred @ H.T).item() + R
    K = (P_pred @ H.T) / S
    x = x_pred + (K * y).ravel()
    I = np.eye(P_pred.shape[0], dtype=np.float64)
    P = (I - K @ H) @ P_pred
    return x, P, y, S


def run_filter(
    measurements: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    R: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Kalman filter over all measurements.
    Returns (x_traj, P_traj, innovations, S_traj).
    x_traj: (n_steps+1, 2), P_traj: (n_steps+1, 2, 2), innovations/S: (n_steps,).
    """
    n = len(measurements)
    x_traj = np.zeros((n + 1, 2), dtype=np.float64)
    P_traj = np.zeros((n + 1, 2, 2), dtype=np.float64)
    innovations = np.zeros(n, dtype=np.float64)
    S_traj = np.zeros(n, dtype=np.float64)

    x_traj[0] = x0
    P_traj[0] = P0

    for k in range(n):
        x_pred, P_pred = predict(x_traj[k], P_traj[k], F, Q)
        x_upd, P_upd, y, S = update(x_pred, P_pred, measurements[k], H, R)
        x_traj[k + 1] = x_upd
        P_traj[k + 1] = P_upd
        innovations[k] = y
        S_traj[k] = S

    return x_traj, P_traj, innovations, S_traj


def generate_synthetic_data(
    n_steps: int,
    x0: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    R: float,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate true state trajectory and noisy measurements.
    Returns (true_states, measurements).
    """
    rng = np.random.default_rng(seed)
    true_states = np.zeros((n_steps + 1, 2), dtype=np.float64)
    true_states[0] = x0
    for k in range(n_steps):
        w = rng.multivariate_normal(np.zeros(2), Q)
        true_states[k + 1] = F @ true_states[k] + w
    H_ = np.asarray(H).reshape(1, -1)
    z = (true_states[1:] @ H_.T).ravel() + np.sqrt(R) * rng.standard_normal(n_steps)
    return true_states, z


def rmse_states(estimated: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-state RMSE over time (excluding initial). Shape (2,)."""
    diff = estimated[1:] - true[1:]
    return np.sqrt(np.mean(diff**2, axis=0))


def innovation_log_likelihood(innovations: np.ndarray, S_traj: np.ndarray) -> float:
    """Sum of log pdf of innovations under N(0, S_traj)."""
    return float(np.sum(-0.5 * (innovations**2 / S_traj + np.log(2 * np.pi * S_traj))))


def run_scenario(
    name: str,
    measurements: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    R: float,
    true_states: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run filter for one scenario and optionally compute RMSE vs true states."""
    x_traj, P_traj, innovations, S_traj = run_filter(
        measurements, x0, P0, F, Q, H, R
    )
    if true_states is not None:
        err = rmse_states(x_traj, true_states)
        ll = innovation_log_likelihood(innovations, S_traj)
        print(f"  {name}: position RMSE = {err[0]:.4f}, velocity RMSE = {err[1]:.4f}, log-lik = {ll:.2f}")
    return x_traj, P_traj, innovations, S_traj


def main() -> None:
    """Run the full application: generate data, filter, and output key arrays."""
    F = build_F(DT)
    Q = build_Q(DT, Q_POS, Q_VEL)
    H = np.array([1.0, 0.0], dtype=np.float64)

    true_states, measurements = generate_synthetic_data(
        N_STEPS, X0, F, Q, H, R_MEAS
    )

    print("Scenario 1: default noise (R=1.0)")
    x_traj, P_traj, innovations, S_traj = run_scenario(
        "default",
        measurements, X0, P0, F, Q, H, R_MEAS,
        true_states=true_states,
    )

    # Second scenario: higher measurement noise (worse observations)
    R_high = 4.0
    print("Scenario 2: high measurement noise (R=4.0)")
    x_traj_2, P_traj_2, innovations_2, S_traj_2 = run_scenario(
        "high_R",
        measurements, X0, P0, F, Q, H, R_high,
        true_states=true_states,
    )

    # Outputs for comparison with Einlang (same data, single scenario)
    np.save("kalman_x_traj.npy", x_traj)
    np.save("kalman_P_traj.npy", P_traj)
    np.save("kalman_innovations.npy", innovations)
    np.save("kalman_measurements.npy", measurements)

    print("Kalman filter (NumPy) done. Outputs: x_traj, P_traj, innovations.")
    print("Final state:", x_traj[-1])
    print("Final P diag:", np.diag(P_traj[-1]))


if __name__ == "__main__":
    main()
