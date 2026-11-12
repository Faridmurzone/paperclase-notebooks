import json, os
import numpy as np
import torch
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

# >>>
def f(p):                                       # dos mínimos: (x² − 1)² + 0,5y² + 0,2x
    return (p[0] ** 2 - 1) ** 2 + 0.5 * p[1] ** 2 + 0.2 * p[0]

def camino(Clase, wd, pasos):
    p = torch.tensor([1.8, 1.5], requires_grad=True)
    opt = Clase([p], lr=0.05, weight_decay=wd)
    puntos = [p.tolist()]
    for _ in range(pasos):
        opt.zero_grad(); f(p).backward(); opt.step()
        puntos.append(p.tolist())
    est = opt.state[p]                          # √v̂ de x: la escala que recuerda Adam
    rv = (est["exp_avg_sq"][0] / (1 - 0.999 ** est["step"])).sqrt().item()
    return np.array(puntos), rv
cam = {"Adam": camino(torch.optim.Adam, 0.0, 200),
       "Adam + L2": camino(torch.optim.Adam, 0.5, 200),
       "AdamW": camino(torch.optim.AdamW, 0.5, 200)}
for k, (c, rv) in cam.items():
    print(f"{k:9} paso 200: x = {c[-1, 0]:.3f}")
c, rv800 = camino(torch.optim.AdamW, 0.5, 800)
print(f"AdamW     paso 800: x = {c[-1, 0]:.3f}")
print(f"AdamW √v̂ de x: paso 200 {cam['AdamW'][1]:.2f}, paso 800 {rv800:.2f}")
# <<<
json.dump({k: np.round(v[0], 4).tolist() for k, v in cam.items()},
          open(os.path.join(DATOS, "ep07_caminos.json"), "w"))
