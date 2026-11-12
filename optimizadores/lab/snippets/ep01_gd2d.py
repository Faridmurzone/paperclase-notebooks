import json, os
import numpy as np
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

# >>>
def g_cuenco(p): return 2 * p                          # f = x² + y²
def g_valle(p):  return np.array([0.2, 6.0]) * p       # f = 0,1x² + 3y²

def descenso(grad, p0, lr, pasos=20):
    p = np.array(p0, dtype=float); camino = [p.copy()]
    for _ in range(pasos):
        p = p - lr * grad(p)                           # la misma regla, en 2D
        camino.append(p.copy())
    return np.array(camino)
caminos = {}
for nombre, g, p0, lr in [("cuenco", g_cuenco, [-2.5, 1.5], 0.1),
                          ("cuenco", g_cuenco, [-2.5, -1.5], 0.9),
                          ("valle", g_valle, [-4.5, 1.5], 0.05),
                          ("valle", g_valle, [-4.5, 1.5], 0.3),
                          ("valle", g_valle, [-4.5, 1.5], 0.34)]:
    c = descenso(g, p0, lr)
    caminos[f"{nombre} {lr}"] = c
    print(f"{nombre:6} lr {lr:4}: ({c[-1][0]:6.3f}, {c[-1][1]:6.3f})")
# <<<
json.dump({k: np.round(v, 4).tolist() for k, v in caminos.items()},
          open(os.path.join(DATOS, "ep01_caminos.json"), "w"))
