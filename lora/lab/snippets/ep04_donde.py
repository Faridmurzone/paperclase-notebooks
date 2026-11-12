import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import *
import json
# >>>
# capas 0, 2 y 4 de la red de juguete: 64→256, 256→256, 256→10
pruebas = [("primera, r=8", (0,), 8),                    # ~2.500 entrenables
           ("del medio, r=5", (2,), 5),
           ("última, r=10", (4,), 10),
           ("las tres, r=2", (0, 2, 4), 2)]
res = {}
for nombre, capas, r in pruebas:
    n, acc = con_lora(capas, r)                          # congela, agrega, entrena
    res[nombre] = acc
    print(f"{nombre:15} {n:>5,} entrenables → {acc:.1%}")
# <<<
json.dump(res, open(os.path.join(DATOS, "ep04_donde.json"), "w"))
