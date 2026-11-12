import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _juguete import MoE, rutear
torch.manual_seed(0)
moe = MoE(d=32, h=32, E=8, k=2)
x = torch.randn(1024, 32)


def moe_loop(x, moe):                           # el de ep03_loop
    pesos, idx, _ = rutear(moe.router(x), moe.k)
    y = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(moe.k):
            y[t] += pesos[t, j] * moe.expertos[idx[t, j].item()](x[t])
    return y, None
# >>>
def moe_agrupado(x, moe):
    pesos, idx, _ = rutear(moe.router(x), moe.k)
    tok = torch.arange(x.shape[0]).repeat_interleave(moe.k)
    exp, w = idx.reshape(-1), pesos.reshape(-1)
    orden = exp.argsort(stable=True)
    tok, exp, w = tok[orden], exp[orden], w[orden]       # dispatch
    cuenta = torch.bincount(exp, minlength=moe.E).tolist()
    y = torch.zeros_like(x)
    for e, t, g in zip(range(moe.E), tok.split(cuenta), w.split(cuenta)):
        salida = moe.expertos[e](x[t])                    # 1 llamada
        y.index_add_(0, t, g[:, None] * salida)           # combine
    return y

with torch.no_grad():
    y1, _ = moe_loop(x, moe)                              # la referencia
    y2 = moe_agrupado(x, moe)
print(f"error máximo: {(y1 - y2).abs().max().item():.1e}")
# <<<
