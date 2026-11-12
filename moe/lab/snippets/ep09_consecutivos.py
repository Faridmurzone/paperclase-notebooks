import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _granite import cargar, TEXTOS, capturar_ruteo
import torch
tok, model = cargar()
elegidos, hs = capturar_ruteo(model)
for tipo, textos in TEXTOS.items():
    for t in textos:
        model(**tok(t, return_tensors="pt"))
# >>>
def repeticion(textos):
    misma, comunes = [], []
    for idx in textos:                       # [T, 8] de un texto
        a, b = idx[:-1], idx[1:]             # token i y token i+1
        misma += (a[:, 0] == b[:, 0]).tolist()   # ¿misma 1.ª elección?
        comunes += [len(set(x.tolist()) & set(y.tolist()))
                    for x, y in zip(a, b)]
    return sum(misma) / len(misma), sum(comunes) / len(comunes)

print(f"azar: misma 1.ª {1 / 32:.1%} · comunes {8 * 8 / 32:.1f}")
for c in [0, 12, 23]:
    r1, comun = repeticion(elegidos[c])
    print(f"capa {c:2d}: misma 1.ª {r1:.1%} · comunes {comun:.1f}")
# <<<
