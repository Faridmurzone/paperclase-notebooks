import os
import sys
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import torch
from _juguete import MoE, Red, tarea
torch.set_printoptions(precision=2)
Xt, Yt, _ = tarea(2048, semilla=1, grupos=1)


def cargar(alfa):
    red = Red(16, 32, 8, MoE(32, 32, E=8, k=2))
    red.load_state_dict(torch.load(f"{AQUI}/../datos/ep04_alfa{alfa}.pt"))
    return red
# >>>
pct = lambda v: " ".join(f"{round(100 * x)}" for x in v.tolist())
for alfa in ("0.0", "0.1"):              # entrenados en preparar_ep04
    red = cargar(alfa)
    with torch.no_grad():
        red(Xt)
    u = red.capa.ultimo
    f = torch.bincount(u["idx"].reshape(-1), minlength=8) / u["idx"].numel()
    P = u["probs"].mean(0)               # prob. media por experto
    L = 8 * (f * P).sum()
    print(f"α={alfa} L_aux={L:.2f} | f% {pct(f)} | P% {pct(P)}")
# <<<
