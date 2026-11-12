import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import Experto, parametros
import torch
torch.manual_seed(0)
d, h, m = 16, 64, 4
x = torch.randn(10, d)
# >>>
grande = Experto(d, h)                        # un experto: d → 64 → d
chicos = [Experto(d, h // m) for _ in range(m)]   # 4 expertos: d → 16 → d
with torch.no_grad():
    for j, c in enumerate(chicos):
        filas = slice(j * h // m, (j + 1) * h // m)   # un cuarto de la capa
        c.arriba.weight.copy_(grande.arriba.weight[filas])
        c.arriba.bias.copy_(grande.arriba.bias[filas])
        c.abajo.weight.copy_(grande.abajo.weight[:, filas])
        c.abajo.bias.copy_(grande.abajo.bias if j == 0 else 0 * c.abajo.bias)
    suma = sum(c(x) for c in chicos)
print(f"máx dif = {(grande(x) - suma).abs().max():.1e}")
print("parámetros:", parametros(grande), "vs",
      sum(parametros(c) for c in chicos))
# <<<
