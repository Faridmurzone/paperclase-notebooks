import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _juguete import MoE
torch.manual_seed(0)
# >>>
moe = MoE(d=32, h=32, E=8, k=8)                  # k = E: todos activos
with torch.no_grad():
    moe.router.weight.zero_()                    # logits iguales
    x = torch.randn(5, 32)
    y = moe(x)
    promedio = sum(e(x) for e in moe.expertos) / 8
print("pesos de un token:", moe.ultimo["pesos"][0].tolist()[:4], "...")
print(f"diferencia con el promedio: {(y - promedio).abs().max().item():.1e}")
# <<<
