import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import MoE, Red, tarea
import torch
X, Y, _ = tarea(256, semilla=0)
torch.manual_seed(0)
moe = MoE(32, 64, E=8, k=2)
red = Red(16, 32, 8, moe)
# >>>
u = 0.01                                     # velocidad de ajuste del sesgo
with torch.no_grad():
    red(X)                                   # un batch: 256 tokens, top-2
c = moe.ultimo["carga"].float()              # asignaciones por experto
error = c.mean() - c                         # > 0: le faltan tokens
moe.sesgo += u * torch.sign(error)           # b_i ← b_i + u·sign(c̄ − c_i)
print("carga  :", " ".join(str(x) for x in c.int().tolist()))
print("media  :", c.mean().item())
print("signo  :", " ".join("+" if e > 0 else "−" for e in error))
print("sesgo/u:", " ".join(f"{b / u:+.0f}" for b in moe.sesgo.tolist()))
print("¿gradiente?", moe.sesgo.requires_grad)
# <<<
