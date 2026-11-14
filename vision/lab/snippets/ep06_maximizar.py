import os, sys
import torch
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _redes import cargar_modelo
from _figs import norm, mosaico, guardar
cnn, _ = cargar_modelo(f"{AQUI}/../datos/cnn_ep05.pt")
for p in cnn.parameters():
    p.requires_grad_(False)
torch.manual_seed(0)
# >>>
def maximizar(canal, pasos=200, lr=0.05):
    x = (torch.randn(1, 3, 32, 32) * 0.1).requires_grad_(True)  # ruido
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(pasos):
        act = cnn.features[:8](x)[0, canal]   # salida de conv3 + ReLU
        perdida = -act.mean() + 0.01 * x.pow(2).sum()   # subir, sin explotar
        opt.zero_grad(); perdida.backward(); opt.step()
    return x.detach()[0], act.mean().item()

for canal in [0, 1, 2]:
    img, a = maximizar(canal)
    print(f"canal {canal}: activación media {a:.2f}")
# <<<
from _figs import gris
torch.manual_seed(0)
paneles = []
for canal in [0, 1, 2]:
    img, _ = maximizar(canal)
    paneles.append(norm(gris(img)))
guardar("ep06_max.json", {"paneles": [p.tolist() for p in paneles]})
