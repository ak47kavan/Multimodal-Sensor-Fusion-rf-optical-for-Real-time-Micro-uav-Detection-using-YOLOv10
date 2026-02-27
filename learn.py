import torch

w = torch.tensor(10.0, requires_grad=True)
x = torch.tensor(1.0)
target = torch.tensor(5.0)

y = w * x
loss = (y - target) ** 2

loss.backward()
print(w.grad)

lr = 0.1
w = w - lr * w.grad