import json, os
import numpy as np
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

# >>>
def pasos(regla, n=100, g=0.5):         # pendiente constante: g = 0,5
    s, v, largos = 0.0, 0.0, []
    for t in range(n):
        s = s + g ** 2                  # AdaGrad: suma
        v = 0.9 * v + 0.1 * g ** 2      # RMSProp: promedio móvil
        largos.append(g / np.sqrt(s) if regla == "AdaGrad" else g / np.sqrt(v))
    return np.array(largos)             # largo de cada paso, con lr = 1

for regla in ["AdaGrad", "RMSProp"]:
    l = pasos(regla)
    print(f"{regla:8}: pasos 1, 2, 10, 100 →",
          " ".join(f"{l[i]:.2f}" for i in (0, 1, 9, 99)))
# <<<
json.dump({r: pasos(r).round(4).tolist() for r in ["AdaGrad", "RMSProp"]},
          open(os.path.join(DATOS, "ep04_comparar.json"), "w"))
