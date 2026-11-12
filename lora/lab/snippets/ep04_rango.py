import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import *
import json
# >>>
res = {}
for r in [1, 2, 4, 8, 16, 32]:
    n, acc = con_lora((0, 2, 4), r)                      # LoRA en las tres capas
    res[r] = acc
    print(f"r={r:2}: {n:>6,} entrenables → {acc:.1%}")
# <<<
json.dump(res, open(os.path.join(DATOS, "ep04_rango.json"), "w"))
