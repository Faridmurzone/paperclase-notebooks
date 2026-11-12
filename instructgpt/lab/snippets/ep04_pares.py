from itertools import combinations
from math import comb
# Un ranking sintético de K = 4 respuestas, con un empate (puesto 2 compartido).
# >>>
respuestas = ["A", "B", "C", "D"]
puesto = {"A": 2, "B": 1, "C": 3, "D": 2}     # 1 = la mejor; A y D empatan

pares = []
for i, j in combinations(respuestas, 2):       # los C(K,2) pares posibles
    if puesto[i] == puesto[j]:
        continue                                # los empates se descartan
    w, l = (i, j) if puesto[i] < puesto[j] else (j, i)
    pares.append((w, l))                        # (ganadora, perdedora)

print("K =", len(respuestas), "→ pares posibles:", comb(len(respuestas), 2))
print("sin empates:", len(pares))
print(" ".join(f"{w}≻{l}" for w, l in pares))
# <<<
import json, os
idx = {r: k for k, r in enumerate(respuestas)}
M = [[0] * 4 for _ in range(4)]
for w, l in pares:
    M[idx[w]][idx[l]] = 1
json.dump({"M": M}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep04_pares.json"), "w"))
