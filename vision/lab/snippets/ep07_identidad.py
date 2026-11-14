import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _resnet import Bloque
torch.manual_seed(0)
# >>>
plano = Bloque(16, 16, 1, residual=False, bn=True).eval()
resid = Bloque(16, 16, 1, residual=True, bn=True).eval()
for b in (plano, resid):
    torch.nn.init.zeros_(b.bn2.weight)   # γ = 0 en la última BN: F(x) = 0
x = torch.relu(torch.randn(4, 16, 8, 8))     # entrada ≥ 0, como tras una ReLU

with torch.no_grad():
    print("plano:    |y| =", round(plano(x).norm().item(), 2))
    print("residual: |y − x| =", round((resid(x) - x).norm().item(), 2),
          "  |x| =", round(x.norm().item(), 2))
# <<<
