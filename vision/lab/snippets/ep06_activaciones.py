import os, sys
import torch
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _redes import cargar_modelo
from _cifar import cargar, normalizar, CLASES
from _figs import norm, gris, mosaico, guardar
torch.set_grad_enabled(False)
cnn, _ = cargar_modelo(f"{AQUI}/../datos/cnn_ep05.pt")
X, y = cargar("test", n=100)
i = 21                                             # un avión negro sobre fondo blanco
x = normalizar(X[i:i + 1])
# >>>
mapas = {}
def guardar_salida(nombre):
    def hook(mod, ent, sal):
        mapas[nombre] = sal[0]                # (canales, alto, ancho)
    return hook

for k, nombre in [(1, "relu1"), (4, "relu2"), (7, "relu3")]:
    cnn.features[k].register_forward_hook(guardar_salida(nombre))
logits = cnn(x)
print("predice:", CLASES[logits.argmax().item()], "| es:", CLASES[y[i].item()])
for nombre, m in mapas.items():
    activos = (m > 0).float().mean().item()
    print(f"{nombre}: {tuple(m.shape)}  activo: {activos:.0%}")
# <<<
m1 = mapas["relu1"]
top = m1.mean(dim=(1, 2)).argsort(descending=True)[:3]
paneles = [gris(X[i].float() / 255)] + [norm(m1[c]) for c in top]
guardar("ep06_activ.json", {"paneles": [p.tolist() for p in paneles], "canales": top.tolist(),
                            "relu2": [norm(mapas["relu2"][c]).tolist() for c in mapas["relu2"].mean(dim=(1, 2)).argsort(descending=True)[:3]]})
