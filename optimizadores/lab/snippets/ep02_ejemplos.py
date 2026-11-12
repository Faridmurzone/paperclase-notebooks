import json, os
import numpy as np
np.set_printoptions(precision=3, suppress=True)
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

# >>>
rng = np.random.default_rng(0)
C = rng.normal(0, 1, size=(1000, 2))   # 1000 ejemplos: cada uno, un punto
C = C - C.mean(0)                      # centrados: el promedio es (0, 0)

def grad(p, idx):                      # pérdida de i: |p - C[i]|²
    return 2 * (p - C[idx]).mean(0)    # gradiente promedio de esos i

p = np.array([-2.5, 1.5])
print("todos (N=1000):   ", grad(p, np.arange(1000)))
for _ in range(2):
    print("minibatch de 10:  ", grad(p, rng.choice(1000, 10, replace=False)))
for _ in range(2):
    print("un solo ejemplo:  ", grad(p, rng.choice(1000, 1)))
muchos = [grad(p, rng.choice(1000, 10, replace=False)) for _ in range(5000)]
print("promedio de 5000: ", np.mean(muchos, 0))
# <<<
rng = np.random.default_rng(0); rng.normal(0, 1, size=(1000, 2))
vs = [grad(p, np.arange(1000))] + [grad(p, rng.choice(1000, 10, replace=False)) for _ in range(2)] \
     + [grad(p, rng.choice(1000, 1)) for _ in range(2)]
json.dump([np.round(v, 3).tolist() for v in vs], open(os.path.join(DATOS, "ep02_vectores.json"), "w"))
