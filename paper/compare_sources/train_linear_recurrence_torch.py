import torch

samples = 10
features = 8
classes = 3
epochs = 5
lr = 0.2

n = torch.arange(samples, dtype=torch.float32).unsqueeze(1)
d = torch.arange(features, dtype=torch.float32).unsqueeze(0)
x = ((n * 5 + d * 2 + 1).remainder(11)) / 10.0

d_teacher = torch.arange(features, dtype=torch.float32).unsqueeze(1)
c_teacher = torch.arange(classes, dtype=torch.float32).unsqueeze(0)
teacher = 0.05 * (1.0 + (d_teacher * 3 + c_teacher * 7).remainder(9))
teacher_b = 0.1 * (1.0 + torch.arange(classes, dtype=torch.float32))
teacher_logits = x @ teacher + teacher_b
truth = teacher_logits.argmax(dim=1)
y = torch.nn.functional.one_hot(truth, num_classes=classes).to(torch.float32)

row = torch.arange(features + 1, dtype=torch.float32).unsqueeze(1)
col = torch.arange(classes, dtype=torch.float32).unsqueeze(0)
theta = torch.where(
    row == features,
    0.01 * (1.0 + (col * 2).remainder(5)),
    0.02 * (1.0 + (row * 7 + col * 3).remainder(13)),
)

for step in range(1, epochs + 1):
    theta = theta.clone().requires_grad_(True)
    logits_before = x @ theta[:-1] + theta[-1]
    loss_before = ((logits_before - y) ** 2).mean()
    (d_theta,) = torch.autograd.grad(loss_before, (theta,))
    theta = theta - lr * d_theta
    logits_after = x @ theta[:-1] + theta[-1]
    correct_after = (logits_after.argmax(dim=1) == truth).sum()
    print(step)
    print(loss_before.item())
    print(int(correct_after))

final_logits = x @ theta[:-1] + theta[-1]
final_correct = (final_logits.argmax(dim=1) == truth).sum()

print(int(final_correct))
