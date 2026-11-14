import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from _redes import LeNet, contar_parametros

torch.manual_seed(0)
# >>>
red = LeNet()                               # LeNet-5 para 3 canales
x = torch.randn(1, 3, 32, 32)
for nombre, capa in [("conv1", red.conv1), ("conv2", red.conv2)]:
    x = F.relu(capa(x))
    print(f"{nombre} → {tuple(x.shape[1:])}")
    x = F.max_pool2d(x, 2)
    print(f"pool  → {tuple(x.shape[1:])}")
print(f"flat  → {x[0].numel()} → 120 → 84 → 10 · {contar_parametros(red):,}")
# <<<
