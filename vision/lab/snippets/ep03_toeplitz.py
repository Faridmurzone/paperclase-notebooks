import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from _cifar import ruta_lab

torch.set_printoptions(precision=0, sci_mode=False)
# >>>
x = torch.tensor([1., 2., 0., 3., 1., 2.])   # señal de 6 valores
w = torch.tensor([1., 0., -1.])              # kernel de 3
T = torch.zeros(4, 6)                        # 4 posiciones × 6 valores
for i in range(4):
    T[i, i:i + 3] = w                        # el mismo w, corrido
print(T)
print("T @ x  :", (T @ x).tolist())
print("conv1d :", F.conv1d(x[None, None], w[None, None])[0, 0].tolist())
# <<<
json.dump({"T": T.int().tolist()}, open(ruta_lab("datos", "figs", "ep03_toeplitz.json"), "w"))
