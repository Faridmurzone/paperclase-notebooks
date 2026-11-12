import json, os
import numpy as np
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

# >>>
def grad(p):                                   # (1-x)² + 100(y-x²)²
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x * x),
                     200 * (y - x * x)])

def correr(mu, nesterov=False, lr=0.001, pasos=500):
    p, v = np.array([-1.2, 1.0]), np.zeros(2); camino = [p]
    for _ in range(pasos):
        g = grad(p + mu * v) if nesterov else grad(p)
        v = mu * v - lr * g; p = p + v; camino.append(p)
    return np.array(camino)

for nombre, c in [("descenso", correr(0.0)), ("momentum", correr(0.9)),
                  ("Nesterov", correr(0.9, nesterov=True))]:
    print(f"{nombre:9}: a {np.linalg.norm(c[-1] - [1, 1]):.3f} de (1, 1)")
# <<<
caminos = {n: correr(*a)[::10].round(4).tolist() for n, a in
           [("descenso", (0.0,)), ("momentum", (0.9,)), ("Nesterov", (0.9, True))]}
json.dump(caminos, open(os.path.join(DATOS, "ep03_rosenbrock.json"), "w"))
