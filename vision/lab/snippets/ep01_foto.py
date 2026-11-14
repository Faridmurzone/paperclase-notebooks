import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from _cifar import ruta_lab

FOTOS = ruta_lab("datos", "fotos.npz")
# >>>
foto = np.load(FOTOS)["flor"]                 # la foto de ejemplo de sklearn
print(type(foto).__name__, foto.shape, foto.dtype)
print("píxel (210, 250):", foto[210, 250])    # rojo, verde, azul
x = torch.from_numpy(foto)                    # (H, W, C), enteros 0..255
x = x.permute(2, 0, 1).float() / 255          # (C, H, W), reales 0..1
print("tensor:", tuple(x.shape), x.dtype)
print("números:", x.numel())
# <<<
chica = F.avg_pool2d(x[None], 32)[0]          # 3 × 13 × 20, solo para la figura
lado = [[round(float(v), 2) for c in range(3) for v in list(chica[c, i]) + ([0.0] if c < 2 else [])] for i in range(chica.shape[1])]
json.dump({"canales": lado, "forma": list(chica.shape)}, open(ruta_lab("datos", "figs", "ep01_foto.json"), "w"))
