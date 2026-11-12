import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import MoE, Red, tarea, entrenar
import torch
import torch.nn.functional as F
X, Y, _ = tarea(8000, semilla=0)
Xt, Yt, _ = tarea(2000, semilla=1)
configs = {"grueso": dict(h=64, E=8, k=2), "fino": dict(h=16, E=32, k=8),
           "fino+comp": dict(h=16, E=31, k=7, n_compartidos=1)}
# >>>
def parecido(moe, h):              # coseno medio entre expertos
    s = torch.stack([e(h).flatten() for e in moe.expertos])
    s = F.normalize(s, dim=-1) @ F.normalize(s, dim=-1).T
    E = len(moe.expertos)
    return ((s.sum() - E) / (E * (E - 1))).item()   # sin la diagonal

for nombre, c in configs.items():
    torch.manual_seed(0)
    moe = MoE(32, **c); red = Red(16, 32, 8, moe)
    entrenar(red, X, Y, pasos=800, alfa=0.01)
    with torch.no_grad():
        mse = F.mse_loss(red(Xt), Yt).item()
        h = red.norma(red.entrada(Xt[:512]))   # lo que entra a la capa MoE
        print(f"{nombre:9} MSE {mse:.3f} · parecido {parecido(moe, h):.3f}")
# <<<
