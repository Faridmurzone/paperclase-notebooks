import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cifar import CLASES, cargar, normalizar, ruta_lab

# >>>
X, y = cargar("train")                  # los pickles originales
Xt, yt = cargar("test")
print("train:", tuple(X.shape), X.dtype, "· test:", len(Xt))
print("por clase:", y.bincount().tolist()[:4], "...")
print("clases:", ", ".join(CLASES[:5]))
print("       ", ", ".join(CLASES[5:]))
x = normalizar(X)                       # por canal
print(f"normalizado: media {x.mean():.2f}, desvío {x.std():.2f}")
# <<<
ej = []
for k, i in enumerate([int((y == c).nonzero()[0]) for c in (3, 7)]):  # gato, auto, caballo
    g = (X[i].float() / 255 * __import__("torch").tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(0)
    ej.append(g)
filas = [[round(float(v), 2) for k, g in enumerate(ej) for v in list(g[r]) + ([0.0, 0.0] if k < 1 else [])] for r in range(32)]
json.dump({"ejemplos": filas, "clases": ["gato", "caballo"]}, open(ruta_lab("datos", "figs", "ep01_cifar.json"), "w"))
