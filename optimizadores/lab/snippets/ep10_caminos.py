import json, os, sys
import numpy as np
import torch
AQUI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, AQUI)
from ep10_lion import Lion
DATOS = os.path.join(AQUI, "..", "datos")

# >>>
def f(p): return 0.1 * p[0] ** 2 + 3 * p[1] ** 2          # el valle del episodio uno

def camino(Clase, lr, pasos=100):
    p = torch.tensor([-4.5, 1.5], requires_grad=True)
    opt = Clase([p], lr=lr)
    puntos = [p.tolist()]
    for _ in range(pasos):
        opt.zero_grad(); f(p).backward(); opt.step()
        puntos.append(p.tolist())
    return np.array(puntos)
cam = {"AdamW": camino(torch.optim.AdamW, 0.1), "Lion": camino(Lion, 0.05)}
for k, c in cam.items():
    dx = np.abs(np.diff(c, axis=0))                        # largo de cada paso
    print(f"{k:5}  final ({c[-1, 0]:5.2f}, {c[-1, 1]:5.2f})"
          f"  pasos en x: {dx[0, 0]:.3f} → {dx[-1, 0]:.3f}")
# <<<
json.dump({k: np.round(v, 4).tolist() for k, v in cam.items()},
          open(os.path.join(DATOS, "ep10_caminos.json"), "w"))
