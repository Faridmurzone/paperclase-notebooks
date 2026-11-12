import json, os
import numpy as np
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

# >>>
g, b1, b2 = 0.5, 0.9, 0.999          # gradiente constante, betas de Adam
m = v = 0.0
for t in range(1, 1001):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    if t in (1, 10, 100, 1000):
        crudo = m / np.sqrt(v)
        corregido = (m / (1 - b1 ** t)) / np.sqrt(v / (1 - b2 ** t))
        print(f"t = {t:4}: m = {m:.3f}  v = {v:.5f}  "
              f"paso crudo {crudo:5.2f}  corregido {corregido:.2f}")
# <<<
m = v = 0.0; crudo = []
for t in range(1, 101):
    m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g ** 2
    crudo.append(round(float(m / np.sqrt(v)), 4))
json.dump(crudo, open(os.path.join(DATOS, "ep05_sesgo.json"), "w"))
