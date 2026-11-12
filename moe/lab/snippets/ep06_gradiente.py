import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import MoE
import torch
torch.manual_seed(0)
x = torch.randn(64, 16)
# >>>
for renormalizar in [True, False]:
    torch.manual_seed(1)
    moe = MoE(d=16, h=32, E=8, k=1, renormalizar=renormalizar)   # top-1
    moe(x).pow(2).mean().backward()            # una pérdida cualquiera
    g = moe.ultimo["pesos"][:3, 0].tolist()
    print(f"renormalizar={renormalizar!s:5}  pesos {[round(v, 3) for v in g]}"
          f"  |∇W_r| = {moe.router.weight.grad.norm():.4f}")
# <<<
