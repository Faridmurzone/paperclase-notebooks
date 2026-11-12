import os
import sys
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import math
import torch
from _juguete import MoE, Red, tarea
Xt, Yt, _ = tarea(2048, semilla=1, grupos=1)
# >>>
for alfa in ("0.0", "0.1"):
    red = Red(16, 32, 8, MoE(32, 32, E=8, k=2))
    red.load_state_dict(torch.load(f"{AQUI}/../datos/ep04_alfa{alfa}.pt"))
    with torch.no_grad():
        red(Xt)
    carga = red.capa.ultimo["carga"]               # asignaciones por experto
    C = math.ceil(2.0 * 2048 * 2 / 8)              # factor 2,0 → 1024
    usados = carga.clamp(max=C).sum().item()       # lugares con un token
    print(f"α={alfa}: carga máx {carga.max().item():>4} | lugares usados "
          f"{usados}/{8 * C} = {usados / (8 * C):.0%}")
# <<<
