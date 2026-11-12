import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _juguete import MoE, parametros
torch.manual_seed(0)
# >>>
moe = MoE(d=32, h=32, E=8, k=2)          # 8 expertos, 2 por token
p_exp = parametros(moe.expertos[0])      # un experto: 32 → 32 → 32
p_router = parametros(moe.router)        # W_r: 32 × 8
print("un experto:", p_exp, "| router:", p_router)
print("totales:   ", p_router + 8 * p_exp)
print("activos:   ", p_router + 2 * p_exp, "por token")
x = torch.randn(4, 32)                   # 4 tokens
y = moe(x)
for t in range(4):
    e = moe.ultimo["idx"][t].tolist()
    g = moe.ultimo["pesos"][t].tolist()
    print(f"t{t} → {e}  g {g[0]:.2f} {g[1]:.2f}")
# <<<
