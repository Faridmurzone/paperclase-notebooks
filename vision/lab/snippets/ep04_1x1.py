import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from _cifar import ruta_lab

foto = torch.from_numpy(np.load(ruta_lab("datos", "fotos.npz"))["flor"]).float() / 255
# >>>
x = foto.permute(2, 0, 1)[None]                 # (1, 3, H, W)
w = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
gris = F.conv2d(x, w)                           # conv 1×1: 3 → 1 canal
print("gris:", tuple(gris.shape))
print("igual a foto @ pesos:",
      torch.allclose(gris[0, 0], foto @ w.flatten()))
r, g, b = x[0, :, 210, 250].tolist()            # un píxel de la flor
print(f"R {r:.2f} G {g:.2f} B {b:.2f} → gris {gris[0, 0, 210, 250]:.2f}")
w8 = torch.randn(8, 3, 1, 1)                    # 3 → 8 canales
y = torch.einsum("oc,nchw->nohw", w8[:, :, 0, 0], x)
print("conv 1×1 = matmul:", torch.allclose(F.conv2d(x, w8), y, atol=1e-5))
# <<<
json.dump({"rgb": [round(v, 2) for v in (r, g, b)], "gris": round(float(gris[0, 0, 210, 250]), 2)},
          open(ruta_lab("datos", "figs", "ep04_1x1.json"), "w"))
