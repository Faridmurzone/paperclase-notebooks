import os, sys
import torch
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _redes import cargar_modelo
from _cifar import cargar, normalizar, CLASES
from _figs import guardar, gris
torch.set_grad_enabled(False)
cnn, _ = cargar_modelo(f"{AQUI}/../datos/cnn_ep05.pt")
X, y = cargar("test", n=100)
i = 21                                             # el mismo avión
avion = gris(X[i].float() / 255) < 0.5             # máscara: píxeles oscuros = el avión
# >>>
x, k = normalizar(X[i:i + 1]), y[i].item()
p0 = cnn(x).softmax(-1)[0, k].item()
caida = torch.zeros(7, 7)
tapa_avion = torch.zeros(7, 7, dtype=torch.bool)
for a in range(7):
    for b in range(7):                        # parche de 8×8, paso 4
        tapada = x.clone()
        tapada[:, :, 4*a:4*a + 8, 4*b:4*b + 8] = 0     # 0 = gris medio
        caida[a, b] = p0 - cnn(tapada).softmax(-1)[0, k].item()
        tapa_avion[a, b] = avion[4*a:4*a + 8, 4*b:4*b + 8].any()
print(f"p({CLASES[k]}) sin tapar: {p0:.2f}")
print(f"tapando parte del avión: caída media {caida[tapa_avion].mean():.2f}")
print(f"tapando solo fondo:      caída media {caida[~tapa_avion].mean():.2f}")
print(f"la mayor caída: {caida.max():.2f}, en",
      "el avión" if tapa_avion.flatten()[caida.argmax()] else "el fondo")
# <<<
guardar("ep06_ocl.json", {"caida": [[round(v, 2) for v in fila] for fila in caida.tolist()]})
