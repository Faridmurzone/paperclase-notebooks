import numpy as np
from itertools import combinations
rng = np.random.default_rng(0)
K, PROMPTS, RUIDO = 4, 2000, 0.8
calidad = rng.normal(0, 1, (PROMPTS, K))            # calidad "real" de cada respuesta (sintética)
def etiquetador(q):                                 # ordena con su propio ruido
    return np.argsort(-(q + rng.normal(0, RUIDO, len(q))))
# >>>
def pares(orden):                    # ranking → ¿i antes que j? para cada par
    pos = {y: k for k, y in enumerate(orden)}
    return [pos[i] < pos[j] for i, j in combinations(range(K), 2)]

acuerdos = []
for q in calidad:                    # dos personas ordenan las mismas K
    a, b = pares(etiquetador(q)), pares(etiquetador(q))
    iguales = [x == y for x, y in zip(a, b)]
    acuerdos.append(np.mean(iguales))
    if len(acuerdos) == 1:
        print("pares:", len(iguales), "| coinciden:", [int(v) for v in iguales])
print(f"acuerdo promedio: {np.mean(acuerdos):.1%}")
# <<<
