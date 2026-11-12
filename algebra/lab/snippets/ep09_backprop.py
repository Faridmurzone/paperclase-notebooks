import torch
torch.manual_seed(0)
x = torch.randn(3); t = torch.tensor([1.0])
W1 = torch.randn(4, 3, requires_grad=True); b1 = torch.zeros(4, requires_grad=True)
W2 = torch.randn(1, 4, requires_grad=True); b2 = torch.zeros(1, requires_grad=True)
# >>>
z1 = W1 @ x + b1                  # forward: 3 → 4 → 1
h = torch.relu(z1)
y = W2 @ h + b2
L = 0.5 * ((y - t) ** 2).sum()

d_y = y - t                       # backward a mano
d_W2 = torch.outer(d_y, h)        # δ hᵀ
d_h = W2.T @ d_y                  # W2ᵀ δ
d_z1 = d_h * (z1 > 0)             # la ReLU deja pasar o corta
d_W1 = torch.outer(d_z1, x)       # δ xᵀ

L.backward()                      # autograd, para comparar
print("dW2 iguales:", torch.allclose(d_W2, W2.grad))
print("dW1 iguales:", torch.allclose(d_W1, W1.grad))
print("ReLU activas:", (z1 > 0).int().tolist())
print("formas:", tuple(d_W1.shape), tuple(d_W2.shape))
# <<<
