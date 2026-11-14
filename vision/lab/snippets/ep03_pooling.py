import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from _cifar import ruta_lab

foto = torch.from_numpy(np.load(ruta_lab("datos", "fotos.npz"))["flor"]).float() / 255
gris = (foto @ torch.tensor([0.299, 0.587, 0.114]))[None, None]
sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None]
# >>>
mapa = F.relu(F.conv2d(gris, sobel_x))        # bordes de la flor
movido = torch.roll(mapa, 1, dims=3)          # 1 píxel a la derecha
coseno = lambda a, b: F.cosine_similarity(a.flatten(), b.flatten(), 0)

print(f"sin pooling     : {coseno(mapa, movido):.2f}")
for k in (2, 4, 8):
    a, b = F.max_pool2d(mapa, k), F.max_pool2d(movido, k)
    print(f"max pooling {k}×{k} : {coseno(a, b):.2f}")
# <<<
