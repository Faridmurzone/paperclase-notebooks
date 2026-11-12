import json, os
import numpy as np
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
def grad(p): return np.array([0.2, 6.0]) * p

# >>>
l1, ln = 0.2, 6.0                         # curvaturas del valle
k = ln / l1                               # condition number: 30
lr_gd = 2 / (l1 + ln)                     # el mejor lr para descenso
lr_m = (2 / (np.sqrt(l1) + np.sqrt(ln))) ** 2      # Goh, 2017
mu_m = ((np.sqrt(ln) - np.sqrt(l1)) / (np.sqrt(ln) + np.sqrt(l1))) ** 2
print(f"descenso: lr {lr_gd:.3f}  ritmo {(k - 1) / (k + 1):.3f}")
print(f"momentum: lr {lr_m:.3f}  mu {mu_m:.3f}  ritmo "
      f"{(np.sqrt(k) - 1) / (np.sqrt(k) + 1):.3f}")

def pasos_hasta(lr, mu, tol=1e-3):
    p, v, t = np.array([-4.5, 1.5]), 0.0, 0
    while np.linalg.norm(p) > tol:
        v = mu * v + grad(p); p = p - lr * v; t += 1
    return t
print("pasos hasta 0,001:", pasos_hasta(lr_gd, 0.0), "contra", pasos_hasta(lr_m, mu_m))
# <<<
def camino(lr, mu, n=40):
    p, v, c = np.array([-4.5, 1.5]), 0.0, [[-4.5, 1.5]]
    for _ in range(n):
        v = mu * v + grad(p); p = p - lr * v; c.append(p.round(4).tolist())
    return c
json.dump({"gd": camino(0.3, 0.0), "gd_mejor": camino(lr_gd, 0.0), "goh": camino(lr_m, mu_m)},
          open(os.path.join(DATOS, "ep03_goh.json"), "w"))
