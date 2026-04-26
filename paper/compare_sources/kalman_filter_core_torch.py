import torch

dt = 0.1
q_pos = 0.01
q_vel = 0.1
r_meas = 1.0
n_steps = 100

F = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float32)
Q = torch.tensor(
    [
        [dt**4 / 4 * q_pos + dt**2 * q_vel, dt**3 / 2 * q_vel],
        [dt**3 / 2 * q_vel, dt**2 * q_vel],
    ],
    dtype=torch.float32,
)
H = torch.tensor([1.0, 0.0], dtype=torch.float32)
z = torch.arange(n_steps, dtype=torch.float32) * dt + 0.5 * (torch.arange(n_steps, dtype=torch.float32) * 0.01 - 0.5)

x = torch.zeros(n_steps + 1, 2, dtype=torch.float32)
P = torch.zeros(n_steps + 1, 2, 2, dtype=torch.float32)
x[0] = torch.tensor([0.0, 1.0], dtype=torch.float32)
P[0] = torch.eye(2, dtype=torch.float32)
I = torch.eye(2, dtype=torch.float32)

for t in range(1, n_steps + 1):
    x_pred = F @ x[t - 1]
    P_pred = F @ P[t - 1] @ F.T + Q
    y = z[t - 1] - H @ x_pred
    S = H @ P_pred @ H + r_meas
    K = (P_pred @ H) / S
    x[t] = x_pred + K * y
    P[t] = (I - torch.outer(K, H)) @ P_pred

x_final = x[n_steps]
P_final_diag = torch.diagonal(P[n_steps])

print(x_final)
print(P_final_diag)
