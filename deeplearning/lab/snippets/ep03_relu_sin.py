import json, os
import numpy as np
AQUI = os.path.dirname(os.path.abspath(__file__))
# >>>
x = np.linspace(-3, 3, 200)
y = np.sin(x)                                   # la curva a imitar

def aproximar(n):                               # n neuronas ReLU
    codos = np.linspace(-3, 3, n)                # dónde dobla cada una
    H = np.maximum(0, x[:, None] - codos)        # hidden layer: (200, n)
    H = np.hstack([H, np.ones((200, 1))])        # + sesgo
    a = np.linalg.lstsq(H, y, rcond=None)[0]     # pesos de salida
    return H @ a

for n in (2, 4, 16):
    err = np.abs(aproximar(n) - y).max()
    print(f"{n:2} neuronas ReLU: error máximo {err:.3f}")
# <<<
pts = {n: [[round(float(a), 3), round(float(b), 3)] for a, b in zip(x[::5], aproximar(n)[::5])] for n in (2, 4, 16)}
pts["sin"] = [[round(float(a), 3), round(float(b), 3)] for a, b in zip(x[::5], y[::5])]
json.dump(pts, open(os.path.join(AQUI, "..", "datos", "ep03_relu_sin.json"), "w"))
