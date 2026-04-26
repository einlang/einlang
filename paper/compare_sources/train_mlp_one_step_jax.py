import jax
import jax.numpy as jnp

batch_size = 16
eval_size = 16
hidden_size = 12
lr = 0.1

n_train = jnp.arange(batch_size, dtype=jnp.float32)[:, None]
d_train = jnp.arange(64, dtype=jnp.float32)[None, :]
train_images = jnp.mod(n_train * 11 + d_train * 3, 17.0) / 16.0
train_truth = jnp.mod(jnp.arange(batch_size) * 3, 10).astype(jnp.int32)
train_labels = jax.nn.one_hot(train_truth, 10, dtype=jnp.float32)

n_eval = jnp.arange(eval_size, dtype=jnp.float32)[:, None]
d_eval = jnp.arange(64, dtype=jnp.float32)[None, :]
eval_images = jnp.mod((n_eval + 5) * 7 + d_eval * 5, 19.0) / 18.0
eval_truth = jnp.mod(jnp.arange(eval_size) * 3 + 1, 10).astype(jnp.int32)

row1 = jnp.arange(65, dtype=jnp.float32)[:, None]
col1 = jnp.arange(hidden_size, dtype=jnp.float32)[None, :]
w1_0 = jnp.where(
    row1 == 64,
    0.02 * (1.0 + jnp.mod(col1 * 5, 7.0)),
    5e-3 * (1.0 + jnp.mod(row1 * 7 + col1 * 3, 17.0)),
)

row2 = jnp.arange(13, dtype=jnp.float32)[:, None]
col2 = jnp.arange(10, dtype=jnp.float32)[None, :]
w2_0 = jnp.where(
    row2 == 12,
    0.01 * (1.0 + jnp.mod(col2 * 2, 5.0)),
    5e-3 * (1.0 + jnp.mod(row2 * 5 + col2 * 7, 19.0)),
)

def forward(images, w1, w2):
    hidden = jax.nn.relu(images @ w1[:-1] + w1[-1])
    return hidden @ w2[:-1] + w2[-1]

def batch_loss(w1, w2):
    return jnp.mean((forward(train_images, w1, w2) - train_labels) ** 2)

batch_logits0 = forward(train_images, w1_0, w2_0)
batch_loss0 = batch_loss(w1_0, w2_0)
eval_logits0 = forward(eval_images, w1_0, w2_0)
eval_correct0 = jnp.sum(jnp.argmax(eval_logits0, axis=1) == eval_truth)

d_w1_0, d_w2_0 = jax.grad(batch_loss, argnums=(0, 1))(w1_0, w2_0)
w1_1 = w1_0 - lr * d_w1_0
w2_1 = w2_0 - lr * d_w2_0

batch_logits1 = forward(train_images, w1_1, w2_1)
batch_loss1 = batch_loss(w1_1, w2_1)
eval_logits1 = forward(eval_images, w1_1, w2_1)
eval_correct1 = jnp.sum(jnp.argmax(eval_logits1, axis=1) == eval_truth)

print(float(batch_loss0))
print(float(batch_loss1))
print(int(eval_correct0))
print(int(eval_correct1))
