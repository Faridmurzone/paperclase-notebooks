import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _juguete import MoE, rutear
torch.manual_seed(0)
moe = MoE(d=32, h=32, E=8, k=2)
x = torch.randn(1024, 32)
# >>>
def moe_loop(x, moe):
    pesos, idx, _ = rutear(moe.router(x), moe.k)    # [T, k] y [T, k]
    y = torch.zeros_like(x)
    llamadas = 0
    for t in range(x.shape[0]):                      # cada token
        for j in range(moe.k):                       # cada elegido
            e = idx[t, j].item()
            y[t] += pesos[t, j] * moe.expertos[e](x[t])
            llamadas += 1
    return y, llamadas

with torch.no_grad():
    y, llamadas = moe_loop(x, moe)
print("tokens", x.shape[0], "| llamadas", llamadas)
# <<<
