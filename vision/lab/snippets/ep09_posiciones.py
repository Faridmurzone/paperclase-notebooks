import os, sys
import torch
import torch.nn.functional as F
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _resnet import cargar_red, ViTChica
torch.set_grad_enabled(False)
vit, _ = cargar_red(f"{AQUI}/../datos/ep09_vit_aug.pt")
torch.manual_seed(0)
al_azar = ViTChica()
# >>>
def parecido(red, fila, col):
    E = F.normalize(red.pos[0, 1:], dim=-1)     # 64 posiciones × 128
    k = fila * 8 + col                          # el patch (fila, col)
    return (E @ E[k]).view(8, 8)                # coseno contra los 64

s = parecido(vit, 3, 4)
print("vecino de al lado:", round(s[3, 5].item(), 2),
      " lejos (fila 7, col 0):", round(s[7, 0].item(), 2))
s0 = parecido(al_azar, 3, 4)
print("sin entrenar, vecino:", round(s0[3, 5].item(), 2),
      " lejos:", round(s0[7, 0].item(), 2))
# <<<
import json
ruta = os.path.join(AQUI, "..", "datos", "ep09_fig.json")
fig = json.load(open(ruta)) if os.path.exists(ruta) else {}
fig["pos_mapa"] = [[round(v, 2) for v in fila] for fila in s.tolist()]
json.dump(fig, open(ruta, "w"))
