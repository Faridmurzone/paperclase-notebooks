import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from _cifar import cargar, normalizar
from _redes import CNNChica, evaluar

torch.manual_seed(0)
X, y = cargar("train", n=10_000)
Xt, yt = cargar("test", n=2_000)
X, Xt = normalizar(X), normalizar(Xt)
# >>>
red = CNNChica()
opt = torch.optim.Adam(red.parameters(), lr=1e-3)
for paso, i in enumerate(range(0, len(X), 128)):    # 1 epoch de 10.000
    xb, yb = X[i:i + 128], y[i:i + 128]
    perdida = F.cross_entropy(red(xb), yb)       # logits vs. etiqueta
    opt.zero_grad()
    perdida.backward()                           # gradientes
    opt.step()                                   # actualizar pesos
    if paso in (0, 40, 78):
        print(f"paso {paso:2d}: pérdida {perdida.item():.2f}")
print(f"test tras 1 epoch chica: {evaluar(red, Xt, yt):.1%}")
# <<<
