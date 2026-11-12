import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _juguete import MoE
torch.manual_seed(0)
# >>>
moe = MoE(d=8, h=8, E=4, k=2)
x = torch.randn(1, 8)                            # un token
moe(x).pow(2).sum().backward()                   # una pérdida cualquiera
print("elegidos:", moe.ultimo["idx"][0].tolist())
for i, e in enumerate(moe.expertos):
    g_exp = e.arriba.weight.grad
    n_exp = 0.0 if g_exp is None else g_exp.norm().item()
    n_rou = moe.router.weight.grad[i].norm().item()  # fila i de W_rᵀ
    print(f"E{i}: pesos {n_exp:.3f} | router {n_rou:.3f}")
# <<<
