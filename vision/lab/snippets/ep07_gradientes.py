import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cifar import cargar, normalizar
from _resnet import RedCifar
X, y = cargar("train", n=256)
X = normalizar(X)
# >>>
def norma_grad_por_capa(red):
    F.cross_entropy(red(X), y).backward()          # un batch de 256
    convs = [m for m in red.modules() if isinstance(m, torch.nn.Conv2d)
             and m.kernel_size == (3, 3)]
    return [c.weight.grad.norm().item() for c in convs]

perfil = {}
for nombre, res, bn in [("plana-56 sin BN", False, False),
                        ("plana-56 con BN", False, True),
                        ("residual-56 con BN", True, True)]:
    torch.manual_seed(0)
    g = norma_grad_por_capa(RedCifar(n=9, residual=res, bn=bn))
    print(f"{nombre:19} capa 1: {g[0]:.1e}  capa 55: {g[-1]:.1e}")
    perfil[nombre] = g
# <<<
import json
json.dump(perfil, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep07_grad.json"), "w"))
