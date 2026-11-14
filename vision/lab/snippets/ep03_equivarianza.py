import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from _cifar import ruta_lab

foto = torch.from_numpy(np.load(ruta_lab("datos", "fotos.npz"))["flor"]).float() / 255
gris = (foto @ torch.tensor([0.299, 0.587, 0.114]))[None, None]
chica = F.avg_pool2d(gris, 16)                        # la flor en 26 × 40
torch.manual_seed(0)
# >>>
w = torch.randn(1, 1, 3, 3)                           # un kernel cualquiera
conv = lambda img: F.conv2d(img, w, padding=1)
mover = lambda img: torch.roll(img, shifts=6, dims=3) # 6 píxeles a la derecha

a = conv(mover(chica))                                # mover y después filtrar
b = mover(conv(chica))                                # filtrar y después mover
dif = (a - b).abs()[0, 0]
print("forma:", tuple(dif.shape))
print(f"diferencia en el interior   : {dif[:, 8:-1].max():.1e}")
print(f"diferencia en todo          : {dif.max():.2f}")
cols = torch.nonzero(dif.amax(0) > 1e-5).flatten().tolist()
print("columnas con diferencia:", cols)
# <<<
json.dump({"dif": [[round(float(v), 3) for v in f] for f in dif]}, open(ruta_lab("datos", "figs", "ep03_equivarianza.json"), "w"))
