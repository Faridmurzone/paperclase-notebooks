import torch
from torch import nn
# >>>
x = torch.tensor([1.0, 2.0, 3.0, 6.0])     # un token, d = 4

mu = x.mean()                               # promedio
var = ((x - mu) ** 2).mean()                # varianza
a_mano = (x - mu) / torch.sqrt(var + 1e-5)
print("μ =", mu.item(), "  σ² =", var.item())
print("a mano   :", a_mano.numpy().round(3))

ln = nn.LayerNorm(4)                        # γ = 1, β = 0 al inicio
print("LayerNorm:", ln(x).detach().numpy().round(3))
print("γ:", ln.weight.data.numpy(), " β:", ln.bias.data.numpy())
print("media =", round(a_mano.mean().item(), 3),
      " desvío =", round(a_mano.std(unbiased=False).item(), 3))
# <<<
