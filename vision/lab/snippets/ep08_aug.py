import os, sys, json
import torch
import torch.nn.functional as F
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _cifar import cargar
X, y = cargar("test", n=200)
i = int((y == 1).nonzero()[0])                     # el primer auto del test
img = X[i:i + 1].float() / 255                     # (1, 3, 32, 32), sin normalizar para mirarla
torch.manual_seed(1)
# >>>
def augmentar(x, pad=4):
    espejo = bool(torch.rand(1) < 0.5)
    if espejo:
        x = x.flip(3)                              # espejo horizontal
    xp = F.pad(x, (pad, pad, pad, pad))            # 40×40, borde de ceros
    dy, dx = torch.randint(0, 2 * pad + 1, (2,)).tolist()
    return xp[:, :, dy:dy + 32, dx:dx + 32], espejo, (dy - pad, dx - pad)

versiones = []
for k in range(3):
    v, espejo, (dy, dx) = augmentar(img)
    versiones.append(v)
    print(f"versión {k + 1}: {tuple(v.shape)}  espejo={espejo}"
          f"  corrida ({dy:+d}, {dx:+d})")
# <<<
gris = lambda t: (0.299 * t[0, 0] + 0.587 * t[0, 1] + 0.114 * t[0, 2])
paneles = [gris(img), gris(versiones[0]), gris(versiones[2])]
filas = [sum(([round(float(p[r, c]), 2) for c in range(32)] + ([0.0] if j < len(paneles) - 1 else [])
              for j, p in enumerate(paneles)), []) for r in range(32)]
json.dump({"filas": filas}, open(os.path.join(AQUI, "..", "datos", "ep08_aug_fig.json"), "w"))
