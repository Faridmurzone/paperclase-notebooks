import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _cifar import cargar, normalizar, permutacion, permutar, ruta_lab
from _redes import cargar_modelo, evaluar

Xt, yt = cargar("test")
mlp, _ = cargar_modelo(ruta_lab("datos", "mlp_ep01.pt"))
# >>>
perm = permutacion(semilla=0)              # un orden fijo de los 1.024 píxeles
Xp = normalizar(permutar(Xt, perm))        # misma mezcla para todas las fotos
mlp_p, _ = cargar_modelo(ruta_lab("datos", "mlp_perm_ep01.pt"))
print(f"entrenado con píxeles mezclados: {evaluar(mlp_p, Xp, yt):.1%}")

idx = torch.cat([c * 1024 + perm for c in range(3)])   # en el vector de 3.072
with torch.no_grad():
    mlp.capa1.weight[:] = mlp.capa1.weight[:, idx]   # reordenar columnas
print(f"MLP original, columnas reordenadas: {evaluar(mlp, Xp, yt):.1%}")
# <<<
torch.manual_seed(0)
g = lambda img: (img.float() / 255 * torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(0)
i = int((yt == 3).nonzero()[0])
a, b = g(Xt[i]), g(permutar(Xt[i:i + 1], perm)[0])
filas = [[round(float(v), 2) for v in list(a[r]) + [0.0, 0.0] + list(b[r])] for r in range(32)]
json.dump({"par": filas}, open(ruta_lab("datos", "figs", "ep01_permutar.json"), "w"))
