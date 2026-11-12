import math
from collections import Counter


def ngramas(palabras, n):
    return Counter(tuple(palabras[i:i + n]) for i in range(len(palabras) - n + 1))


# >>>
ref = "the cat is on the mat".split()       # ejemplo 2 del paper de BLEU
cand = "the the the the the the the".split()
c, r = ngramas(cand, 1), ngramas(ref, 1)
print("precisión común:", sum(k for g, k in c.items() if g in r), "/", len(cand))
print("modified (recortada):", sum(min(k, r[g]) for g, k in c.items()),
      "/", len(cand))

ref = "el gato negro duerme sobre la alfombra roja de la casa".split()
for cand in ["el gato negro duerme", "el gato negro duerme en la alfombra"]:
    c = len(cand.split())
    print(f"{c:2} palabras de {len(ref)}: BP = e^(1 − r/c) ="
          f" {math.exp(1 - len(ref) / c):.3f}")
# <<<
