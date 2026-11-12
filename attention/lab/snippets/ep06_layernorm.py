import torch
import torch.nn as nn
torch.manual_seed(0)
# >>>
x = torch.tensor([2., 4., 6., 8.])            # un token, d = 4
mu = x.mean()
var = ((x - mu) ** 2).mean()                  # varianza sin corregir
x_hat = (x - mu) / torch.sqrt(var + 1e-5)
print("media", mu.item(), "· varianza", var.item())
print("normalizado", [round(v, 2) for v in x_hat.tolist()])

ln = nn.LayerNorm(4)                          # gamma = 1, beta = 0 al inicio
print("igual a nn.LayerNorm:", torch.allclose(ln(x), x_hat, atol=1e-6))

X = 3 + 10 * torch.randn(5, 512)              # 5 tokens desordenados
Y = nn.LayerNorm(512)(X)
print("media por token", [round(v, 3) + 0.0       # +0.0: sin -0.0
                         for v in Y.mean(-1).tolist()])
print("desvío por token", [round(v, 3) for v in Y.std(-1, unbiased=False).tolist()])
# <<<
