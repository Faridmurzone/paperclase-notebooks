import json, os
import numpy as np
caminos = {}
# >>>
def perdida(p):                       # el valle: 0,1·x² + 3·y²
    return 0.1 * p[0]**2 + 3 * p[1]**2

def grad(p):                          # (0,2·x, 6·y)
    return np.array([0.2 * p[0], 6 * p[1]])

for eta in [0.1, 0.3, 0.34]:          # eta: learning rate
    p = np.array([-4.5, 1.5])
    camino = [p]
    for paso in range(30):
        p = p - eta * grad(p)         # descenso por gradiente
        camino.append(p)
    print(f"eta={eta}: p=({p[0]:+.3f}, {p[1]:+.3f})  pérdida={perdida(p):.4f}")
    caminos[eta] = camino
# <<<
json.dump({str(k): [[round(float(a), 4), round(float(b), 4)] for a, b in v] for k, v in caminos.items()},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep10_gd.json"), "w"))
