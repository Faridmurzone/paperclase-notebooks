import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _juguete import MoE, Red
torch.manual_seed(0)
# >>>
red = Red(16, 32, 8, MoE(32, 32, E=8, k=2, factor_capacidad=0.25))
x = torch.randn(64, 16)
salidas = {}
red.capa.register_forward_hook(lambda m, i, o: salidas.update(moe=o))
with torch.no_grad():
    red(x)
u = red.capa.ultimo
sin = salidas["moe"].abs().sum(-1) == 0           # filas en cero
print("capacidad:", red.capa.capacidad(64), "por experto")
print("tokens sin experto:", u["tokens_sin_experto"], "de 64")
print("salida de la capa en cero para esos tokens:", int(sin.sum()))
# h = h + capa(norma(h)): esos tokens siguen con h intacto
# <<<
