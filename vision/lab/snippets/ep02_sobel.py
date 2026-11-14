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
foto = torch.from_numpy(np.load(FOTOS)["flor"]).float() / 255
gris = foto @ torch.tensor([0.299, 0.587, 0.114])  # (H, W)
sobel_x = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]])
sobel_y = sobel_x.T                                  # bordes horizontales
x = gris[None, None]       # (batch, canal, alto, ancho)
gx = F.conv2d(x, sobel_x[None, None])
gy = F.conv2d(x, sobel_y[None, None])
borde = (gx**2 + gy**2).sqrt()
print("foto ", tuple(foto.shape), " gris", tuple(gris.shape))
print("borde", tuple(borde.shape[-2:]))
print(f"píxeles con borde fuerte (> 1): {(borde > 1).float().mean():.1%}")
# <<<
# Figuras (no se muestran): la misma cuenta sobre la foto achicada 16 veces (promedio de bloques de
# 16 × 16 píxeles), para que en pantalla se vea cada píxel. padding=1 para conservar el tamaño 26 × 40.
chica = F.avg_pool2d(x, 16)
gxc = F.conv2d(chica, sobel_x[None, None], padding=1)[0, 0]
gyc = F.conv2d(chica, sobel_y[None, None], padding=1)[0, 0]
bc = (gxc**2 + gyc**2).sqrt()
r = lambda t: [[round(float(v), 3) for v in fila] for fila in t]
# ventana real de 3 × 3 sobre el borde izquierdo de la flor (en la foto chica), para la figura de productos
fila_c, col_c = 13, int(gxc[13].argmax())
ventana = chica[0, 0, fila_c - 1:fila_c + 2, col_c - 1:col_c + 2]
json.dump({"gris": r(chica[0, 0]), "gx": r(gxc), "borde": r(bc),
           "ventana": r(ventana), "prod": [round(float(v), 3) for v in (ventana * sobel_x).flatten()],
           "gx_centro": round(float(gxc[fila_c, col_c]), 3), "pos": [fila_c, col_c]},
          open(ruta_lab("datos", "figs", "ep02_sobel.json"), "w"))
