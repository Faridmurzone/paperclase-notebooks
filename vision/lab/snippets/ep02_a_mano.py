import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _cifar import ruta_lab

# >>>
x = torch.tensor([[1., 1., 1., 8., 8.]] * 5)   # 5 filas: oscuro | claro
w = torch.tensor([[-1., 0., 1.]] * 3)          # kernel: derecha − izquierda
k = 3
salida = x.shape[0] - k + 1                # 5 − 3 + 1 = 3
y = torch.zeros(salida, salida)
for i in range(salida):
    for j in range(salida):
        ventana = x[i:i + k, j:j + k]      # parche de 3 × 3
        y[i, j] = (ventana * w).sum()      # suma de productos
print(y)
# <<<
prod = (x[0:3, 1:4] * w).flatten().tolist()
json.dump({"x": x.tolist(), "y": y.tolist(), "prod": prod}, open(ruta_lab("datos", "figs", "ep02_a_mano.json"), "w"))
