import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from _cifar import ruta_lab

# >>>
x = torch.arange(1., 17.).view(1, 1, 4, 4)   # imagen 4 × 4 con 1..16
cols = F.unfold(x, kernel_size=2)             # parches de 2 × 2
print("cols:", tuple(cols.shape))             # (batch, 4 valores, 9 parches)
print(cols[0].long())
# <<<
json.dump({"x": x[0, 0].int().tolist(), "cols": cols[0].int().tolist()}, open(ruta_lab("datos", "figs", "ep03_unfold.json"), "w"))
