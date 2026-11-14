import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from _cifar import ruta_lab

torch.manual_seed(0)
# >>>
conv = lambda cin, cout: [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU()]
red = nn.Sequential(*conv(3, 16), *conv(16, 16), nn.MaxPool2d(2),
                    *conv(16, 32), *conv(32, 32), nn.MaxPool2d(2),
                    *conv(32, 32), *conv(32, 32))          # pesos al azar
G = torch.zeros(64, 64)
for _ in range(20):                                 # 20 imágenes al azar
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    y = red(x)                                      # (1, 32, 16, 16)
    y[0, :, 8, 8].sum().backward()                  # una posición del centro
    G += x.grad[0].abs().sum(0)
filas, cols = torch.nonzero(G).T
print("zona con gradiente ≠ 0:",
      f"{filas.max() - filas.min() + 1}×{cols.max() - cols.min() + 1}")
f0, c0 = filas.min(), cols.min()                     # esquina de la zona
for lado in (8, 16, 32):
    m = (32 - lado) // 2
    c = G[f0 + m:f0 + m + lado, c0 + m:c0 + m + lado]
    print(f"centro {lado:2d}×{lado:<2d}: {c.sum() / G.sum():.0%} del gradiente")
# <<<
f0, c0 = int(f0), int(c0)
rec = G[f0:f0 + 32, c0:c0 + 32]
json.dump({"G": [[round(float(v), 3) for v in f] for f in rec / rec.max()], "origen": [f0, c0]},
          open(ruta_lab("datos", "figs", "ep04_gradiente.json"), "w"))
