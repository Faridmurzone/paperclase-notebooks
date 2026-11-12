import json, os
import numpy as np
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
rng = np.random.default_rng(0)
C = rng.normal(0, 1, size=(1000, 2)); C = C - C.mean(0)
def grad(p, idx): return 2 * (p - C[idx]).mean(0)

# >>>
def sgd(B, lr=0.1, pasos=40, semilla=1):
    rng = np.random.default_rng(semilla)
    p = np.array([-2.5, 1.5]); camino = [p.copy()]
    for _ in range(pasos):
        idx = rng.choice(1000, B, replace=False)   # un minibatch al azar
        p = p - lr * grad(p, idx)                  # el paso de siempre
        camino.append(p.copy())
    return np.array(camino)

caminos = {B: sgd(B) for B in [1000, 10, 1]}
for B, c in caminos.items():
    lejos = np.linalg.norm(c[-10:], axis=1).mean()
    print(f"B = {B:4}: últimos 10 pasos, a {lejos:.3f} del mínimo")
# <<<
json.dump({str(B): np.round(c, 4).tolist() for B, c in caminos.items()},
          open(os.path.join(DATOS, "ep02_caminos.json"), "w"))
