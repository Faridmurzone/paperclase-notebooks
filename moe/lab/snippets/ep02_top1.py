import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _juguete import MoE
x = torch.randn(16, 8, generator=torch.Generator().manual_seed(1))
# >>>
for renorm in (True, False):
    torch.manual_seed(0)
    moe = MoE(d=8, h=8, E=4, k=1, renormalizar=renorm)
    moe(x).pow(2).sum().backward()
    pesos = moe.ultimo["pesos"][:3, 0].tolist()
    g = moe.router.weight.grad.norm().item()
    print(f"renormalizar={renorm!s:5} pesos {[round(p, 2) for p in pesos]}"
          f" grad router {g:.3f}")
# <<<
