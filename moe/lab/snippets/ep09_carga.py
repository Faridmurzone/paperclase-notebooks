import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _granite import cargar, TEXTOS, capturar_ruteo
import torch
tok, model = cargar()
elegidos, hs = capturar_ruteo(model)       # hook en el router de las 24 capas
for tipo, textos in TEXTOS.items():
    for t in textos:
        model(**tok(t, return_tensors="pt"))
# >>>
T = torch.cat(elegidos[0]).shape[0]          # tokens de todos los textos
print("tokens:", T, "· uniforme: 8/32 =", f"{8 / 32:.0%}")
for c in [0, 6, 12, 18, 23]:
    idx = torch.cat(elegidos[c])             # [T, 8]
    carga = torch.bincount(idx.reshape(-1), minlength=32) / T
    print(f"capa {c:2d}: máx {carga.max():.0%} (exp. {carga.argmax()})"
          f" · mín {carga.min():.0%} (exp. {carga.argmin()})")
# <<<
