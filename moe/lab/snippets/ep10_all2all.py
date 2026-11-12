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
# >>>
D = 4                                          # 4 placas, 8 expertos cada una
T = len(idx)
origen = torch.arange(T) * D // T              # data parallel: tokens repartidos
destino = idx // 8                             # experto e vive en la placa e // 8
trafico = torch.zeros(D, D, dtype=torch.long)
for o in range(D):
    a_donde = destino[origen == o].reshape(-1)
    trafico[o] = torch.bincount(a_donde, minlength=D)
print("de \\ a ", " ".join(f"{d:5d}" for d in range(D)))
for o in range(D):
    print(f"placa {o}", " ".join(f"{x:5d}" for x in trafico[o].tolist()))
local = trafico.diag().sum() / trafico.sum()
print(f"se queda: {local:.0%} · viaja: {1 - local:.0%}")
mb = T * 8 * 1024 * 2 / 1e6                    # T·k vectores de d=1024 en bf16
print(f"ida {mb * (1 - local):.1f} de {mb:.1f} MB, igual de vuelta")
# <<<
