import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _resnet import RedCifar, contar
# >>>
print("capas   plana     residual   diferencia")
for n in [1, 3, 9]:                          # 6n + 2 capas con pesos
    plana = RedCifar(n=n, residual=False)
    resid = RedCifar(n=n, residual=True)
    p, r = contar(plana), contar(resid)
    print(f"{6 * n + 2:5} {p:9,} {r:10,} {r - p:+10,}")

bloques = resid.bloques                      # los 27 bloques de la de 56
proy = sum(not isinstance(b.atajo, torch.nn.Identity) for b in bloques)
print("bloques:", len(bloques), "| con proyección 1×1:", proy)
# <<<
