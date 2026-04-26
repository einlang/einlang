import jax
import jax.numpy as jnp

samples = 10
features = 8
classes = 3
epochs = 5
lr = 0.2

n = jnp.arange(samples, dtype=jnp.float32)[:, None]
d = jnp.arange(features, dtype=jnp.float32)[None, :]
x = jnp.mod(n * 5 + d * 2 + 1, 11.0) / 10.0

d_teacher = jnp.arange(features, dtype=jnp.float32)[:, None]
c_teacher = jnp.arange(classes, dtype=jnp.float32)[None, :]
teacher = 0.05 * (1.0 + jnp.mod(d_teacher * 3 + c_teacher * 7, 9.0))
teacher_b = 0.1 * (1.0 + jnp.arange(classes, dtype=jnp.float32))
teacher_logits = x @ teacher + teacher_b
truth = jnp.argmax(teacher_logits, axis=1)
y = jax.nn.one_hot(truth, classes, dtype=jnp.float32)

row = jnp.arange(features + 1, dtype=jnp.float32)[:, None]
col = jnp.arange(classes, dtype=jnp.float32)[None, :]
theta0 = jnp.where(
    row == features,
    0.01 * (1.0 + jnp.mod(col * 2, 5.0)),
    0.02 * (1.0 + jnp.mod(row * 7 + col * 3, 13.0)),
)

def logits(theta):
    return x @ theta[:-1] + theta[-1]

def loss(theta):
    return jnp.mean((logits(theta) - y) ** 2)

theta = theta0
for step in range(1, epochs + 1):
    loss_before = loss(theta)
    d_theta = jax.grad(loss)(theta)
    theta = theta - lr * d_theta
    correct_after = jnp.sum(jnp.argmax(logits(theta), axis=1) == truth)
    print(step)
    print(float(loss_before))
    print(int(correct_after))

final_correct = jnp.sum(jnp.argmax(logits(theta), axis=1) == truth)

print(int(final_correct))
