import math
import torch
from torch import nn
torch.manual_seed(0)
# >>>
capa = nn.Linear(64, 64)                     # init por defecto de PyTorch
cota = 1 / math.sqrt(64)
print("defecto: máx |w| =", round(capa.weight.abs().max().item(), 4))
print("         1/√64   =", cota)
print("         std(w)  =", round(capa.weight.std().item(), 4))
print("         cota/√3 =", round(cota / math.sqrt(3), 4))

nn.init.kaiming_normal_(capa.weight, nonlinearity="relu")
print("Kaiming: std(w)  =", round(capa.weight.std().item(), 4))
print("         √(2/64) =", round(math.sqrt(2 / 64), 4))
# <<<
