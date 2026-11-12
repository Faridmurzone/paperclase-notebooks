import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import *
torch.manual_seed(0)
import json
c = torch.load(os.path.join(DATOS, "ep02_completo.pt"))
# >>>
dW = c["dW"][1]                                        # capa del medio: 256 × 256
S = torch.linalg.svdvals(dW)                           # 256 valores singulares
energia = S**2 / (S**2).sum()                          # fracción de cada uno

azar = torch.randn(256, 256)                           # misma forma, sin estructura
Sa = torch.linalg.svdvals(azar)
energia_azar = Sa**2 / (Sa**2).sum()

print("sigma 1..6:", [round(s, 2) for s in S[:6].tolist()])
for k in [1, 4, 8, 32]:
    e, ea = energia[:k].sum(), energia_azar[:k].sum()
    print(f"top {k:2}: ΔW {e:.0%}   al azar {ea:.0%}")
# <<<
ks = list(range(0, 65, 2))
json.dump(dict(sigma=[round(s, 4) for s in S[:16].tolist()],
               energia=[round(e, 4) for e in energia[:16].tolist()],
               ks=ks, acum=[round(energia[:k].sum().item(), 4) for k in ks],
               acum_azar=[round(energia_azar[:k].sum().item(), 4) for k in ks]),
          open(os.path.join(DATOS, "ep02_espectro.json"), "w"))
