import json, os
import numpy as np
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

# >>>
def grad(p):                                   # Rosenbrock
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)])

def correr(regla, lr, pasos=500, b1=0.9, b2=0.999):
    p, m, v = np.array([-1.2, 1.0]), 0.0, 0.0; camino = [p]
    for t in range(1, pasos + 1):
        g = grad(p)
        if regla == "momentum":
            m = 0.9 * m + g; p = p - lr * m
        else:                                   # RMSProp (b1 = 0) o Adam
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g ** 2
            mh, vh = (m / (1 - b1 ** t), v / (1 - b2 ** t)) if regla == "Adam" else (g, v)
            p = p - lr * mh / (np.sqrt(vh) + 1e-8)
        camino.append(p)
    return np.array(camino)

caminos = {"momentum": correr("momentum", 0.001), "RMSProp": correr("RMSProp", 0.02, b2=0.9),
           "Adam": correr("Adam", 0.1)}
for nombre, c in caminos.items():
    print(f"{nombre:8}: a {np.linalg.norm(c[-1] - [1, 1]):.3f} de (1, 1)")
# <<<
json.dump({k: c[::10].round(4).tolist() for k, c in caminos.items()},
          open(os.path.join(DATOS, "ep05_rosenbrock.json"), "w"))
