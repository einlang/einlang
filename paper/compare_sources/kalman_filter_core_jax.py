import jax.numpy as jnp

dt = 0.1
q_pos = 0.01
q_vel = 0.1
r_meas = 1.0
n_steps = 100

F = jnp.array([[1.0, dt], [0.0, 1.0]], dtype=jnp.float32)
Q = jnp.array(
    [
        [dt**4 / 4 * q_pos + dt**2 * q_vel, dt**3 / 2 * q_vel],
        [dt**3 / 2 * q_vel, dt**2 * q_vel],
    ],
    dtype=jnp.float32,
)
H = jnp.array([1.0, 0.0], dtype=jnp.float32)
z = jnp.arange(n_steps, dtype=jnp.float32) * dt + 0.5 * (jnp.arange(n_steps, dtype=jnp.float32) * 0.01 - 0.5)

xs = [jnp.array([0.0, 1.0], dtype=jnp.float32)]
Ps = [jnp.eye(2, dtype=jnp.float32)]
I = jnp.eye(2, dtype=jnp.float32)

for t in range(n_steps):
    x_pred = F @ xs[-1]
    P_pred = F @ Ps[-1] @ F.T + Q
    y = z[t] - H @ x_pred
    S = H @ P_pred @ H + r_meas
    K = (P_pred @ H) / S
    xs.append(x_pred + K * y)
    Ps.append((I - jnp.outer(K, H)) @ P_pred)

x = jnp.stack(xs)
P = jnp.stack(Ps)
x_final = x[n_steps]
P_final_diag = jnp.diag(P[n_steps])

print(x_final)
print(P_final_diag)
