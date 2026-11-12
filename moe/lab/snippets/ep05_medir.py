import os
import sys
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import torch
import torch.nn.functional as F
from _juguete import MoE, Red, tarea
Xt, Yt, _ = tarea(2048, semilla=1, grupos=1)      # los datos de test del ep. 4


def cargar(alfa):
    red = Red(16, 32, 8, MoE(32, 32, E=8, k=2))
    red.load_state_dict(torch.load(f"{AQUI}/../datos/ep04_alfa{alfa}.pt"))
    return red
# >>>
pct = lambda v: " ".join(f"{100 * x:4.1f}" for x in v)
for alfa in ("0.0", "0.1"):                  # sin y con balanceo (ep. 4)
    red, desc, sin, mse = cargar(alfa), [], [], []
    for factor in (None, 1.0, 1.25, 2.0):         # None: sin límite
        red.capa.factor_capacidad = factor
        with torch.no_grad():
            mse.append(F.mse_loss(red(Xt), Yt).item())
        desc.append(red.capa.ultimo["descartadas"] / 4096)
        sin.append(red.capa.ultimo["tokens_sin_experto"] / 2048)
    print(f"α={alfa} % descart. {pct(desc)} | % sin experto {pct(sin)}")
    print(f"      MSE {' '.join(f'{m:.4f}' for m in mse)}")
# <<<
