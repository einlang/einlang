import torch

batch_size = 16
eval_size = 16
hidden_size = 12
lr = 0.1

n_train = torch.arange(batch_size, dtype=torch.float32).unsqueeze(1)
d_train = torch.arange(64, dtype=torch.float32).unsqueeze(0)
train_images = ((n_train * 11 + d_train * 3).remainder(17)) / 16.0
train_truth = (torch.arange(batch_size) * 3).remainder(10)
train_labels = torch.nn.functional.one_hot(train_truth, num_classes=10).to(torch.float32)

n_eval = torch.arange(eval_size, dtype=torch.float32).unsqueeze(1)
d_eval = torch.arange(64, dtype=torch.float32).unsqueeze(0)
eval_images = (((n_eval + 5) * 7 + d_eval * 5).remainder(19)) / 18.0
eval_truth = (torch.arange(eval_size) * 3 + 1).remainder(10)

row1 = torch.arange(65, dtype=torch.float32).unsqueeze(1)
col1 = torch.arange(hidden_size, dtype=torch.float32).unsqueeze(0)
w1_0 = torch.where(
    row1 == 64,
    0.02 * (1.0 + (col1 * 5).remainder(7)),
    5e-3 * (1.0 + (row1 * 7 + col1 * 3).remainder(17)),
).clone().requires_grad_(True)

row2 = torch.arange(13, dtype=torch.float32).unsqueeze(1)
col2 = torch.arange(10, dtype=torch.float32).unsqueeze(0)
w2_0 = torch.where(
    row2 == 12,
    0.01 * (1.0 + (col2 * 2).remainder(5)),
    5e-3 * (1.0 + (row2 * 5 + col2 * 7).remainder(19)),
).clone().requires_grad_(True)

hidden0 = torch.relu(train_images @ w1_0[:-1] + w1_0[-1])
batch_logits0 = hidden0 @ w2_0[:-1] + w2_0[-1]
batch_loss0 = ((batch_logits0 - train_labels) ** 2).mean()
eval_hidden0 = torch.relu(eval_images @ w1_0[:-1] + w1_0[-1])
eval_logits0 = eval_hidden0 @ w2_0[:-1] + w2_0[-1]
eval_correct0 = (eval_logits0.argmax(dim=1) == eval_truth).sum()

d_w1_0, d_w2_0 = torch.autograd.grad(batch_loss0, (w1_0, w2_0))
w1_1 = w1_0 - lr * d_w1_0
w2_1 = w2_0 - lr * d_w2_0

hidden1 = torch.relu(train_images @ w1_1[:-1] + w1_1[-1])
batch_logits1 = hidden1 @ w2_1[:-1] + w2_1[-1]
batch_loss1 = ((batch_logits1 - train_labels) ** 2).mean()
eval_hidden1 = torch.relu(eval_images @ w1_1[:-1] + w1_1[-1])
eval_logits1 = eval_hidden1 @ w2_1[:-1] + w2_1[-1]
eval_correct1 = (eval_logits1.argmax(dim=1) == eval_truth).sum()

print(batch_loss0.item())
print(batch_loss1.item())
print(int(eval_correct0))
print(int(eval_correct1))
