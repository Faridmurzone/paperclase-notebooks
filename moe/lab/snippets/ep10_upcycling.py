import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import MoE, Denso, parametros
import torch
torch.manual_seed(0)
d, h = 16, 64
denso = Denso(d, h)                            # el MLP de un modelo ya entrenado
x = torch.randn(32, d)
# >>>
moe = MoE(d, h, E=8, k=2)                      # 8 expertos, 2 por token
for experto in moe.expertos:
    experto.load_state_dict(denso.state_dict())   # copia exacta del denso
torch.nn.init.normal_(moe.router.weight, std=0.02)  # router nuevo, al azar

with torch.no_grad():
    dif = (moe(x) - denso(x)).abs().max().item()
print(f"máx |MoE(x) − denso(x)| = {dif:.1e}")
print("parámetros denso:", parametros(denso))
print("parámetros MoE  :", parametros(moe), "· activos por token:",
      2 * parametros(denso) + moe.router.weight.numel())
# <<<
