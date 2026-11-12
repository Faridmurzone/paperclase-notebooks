import math
from collections import Counter
# >>>
def ngramas(palabras, n):
    return Counter(tuple(palabras[i:i + n]) for i in range(len(palabras) - n + 1))

def precision(cand, ref, n):                  # modified n-gram precision
    c, r = ngramas(cand, n), ngramas(ref, n)
    recortado = sum(min(k, r[g]) for g, k in c.items())   # Count_clip
    return recortado, sum(c.values())

def bleu(cand, ref, N=4):
    cand, ref = cand.split(), ref.split()
    p = [a / b for a, b in (precision(cand, ref, n) for n in range(1, N + 1))]
    bp = 1 if len(cand) > len(ref) else math.exp(1 - len(ref) / len(cand))
    return bp * math.exp(sum(math.log(x) / N for x in p)), p, bp

ref = "el gato negro duerme sobre la alfombra roja de la casa"
cand = "el gato negro duerme en la alfombra roja de la casa"
b, p, bp = bleu(cand, ref)
print("p1..p4:", [f"{x:.3f}" for x in p], f"· BP {bp:.2f}")
print(f"BLEU = {b:.3f} → {100 * b:.1f} en la escala del paper")
# <<<
