import json, os
import numpy as np
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

# >>>
p, lr = np.array([-4.5, 1.5]), 0.3        # el valle del episodio uno
grads = []
for t in range(6):
    g = np.array([0.2, 6.0]) * p          # gradiente: (0,2x, 6y)
    grads.append(g)
    print(f"paso {t}: g = ({g[0]:6.3f}, {g[1]:6.3f})")
    p = p - lr * g                        # descenso por gradiente
# <<<
json.dump(np.round(grads, 4).tolist(), open(os.path.join(DATOS, "ep03_grads.json"), "w"))
