import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _granite import cargar, TEXTOS, capturar_ruteo
import torch
tok, model = cargar()
elegidos, hs = capturar_ruteo(model)
for tipo, textos in TEXTOS.items():
    for t in textos:
        model(**tok(t, return_tensors="pt"))
idx = torch.cat(elegidos[12])                  # capa 12: [1055, 8]
g = torch.Generator().manual_seed(0)
# >>>
E, k = 32, 8
mb_exp = 3 * 512 * 1024 * 2 / 1e6            # MB de un experto en bf16
print("  n  dist  azar    MoE  d=act  d=tot")   # MB por token y capa
for n in [1, 4, 16, 64, 256]:
    distintos = []
    for _ in range(200):                       # 200 batches de n tokens
        fila = torch.randperm(len(idx), generator=g)[:n]
        distintos.append(idx[fila].unique().numel())
    d = sum(distintos) / len(distintos)
    azar = E * (1 - (1 - k / E) ** n)
    moe, act, tot = d * mb_exp / n, k * mb_exp / n, E * mb_exp / n
    print(f"{n:3d} {d:5.1f} {azar:5.1f} {moe:6.2f} {act:6.2f} {tot:6.2f}")
# <<<
