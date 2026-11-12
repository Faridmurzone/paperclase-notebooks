import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import MoE, Denso, Red, tarea, entrenar
import torch
import torch.nn.functional as F
X, Y, _ = tarea(8000, semilla=0)              # tarea sintética: 8 grupos
Xt, Yt, _ = tarea(2000, semilla=1)
torch.manual_seed(0)
d, h = 32, 64
base = Red(16, d, 8, Denso(d, h))
entrenar(base, X, Y, pasos=800)               # el "modelo denso ya entrenado"
def mse(red):
    with torch.no_grad():
        return F.mse_loss(red(Xt), Yt).item()
# >>>
print(f"denso de partida      : {mse(base):.3f}")
densa = copy.deepcopy(base)                   # opción 1: seguir con el denso
entrenar(densa, X, Y, pasos=800, semilla=1)

up = copy.deepcopy(base)                      # opción 2: upcycling
moe = MoE(d, h, E=8, k=2)
for e in moe.expertos:
    e.load_state_dict(base.capa.state_dict())
torch.nn.init.normal_(moe.router.weight, std=0.02)
up.capa = moe
print(f"upcycled, paso 0      : {mse(up):.3f}")   # = denso
entrenar(up, X, Y, pasos=800, semilla=1, alfa=0.01)
print(f"+800 pasos, denso     : {mse(densa):.3f}")
print(f"+800 pasos, upcycled  : {mse(up):.3f}")
# <<<
