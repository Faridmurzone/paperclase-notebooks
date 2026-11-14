import os, sys
import torch
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _redes import cargar_modelo
from _figs import norm, mosaico, guardar
torch.set_grad_enabled(False)
# >>>
cnn, info = cargar_modelo(f"{AQUI}/../datos/cnn_ep05.pt")   # la del ep05
W = cnn.conv1.weight                         # (32, 3, 5, 5)
print("conv1:", tuple(W.shape), " precisión test:", f"{info['precision_test']:.1%}")

gris_ = W.mean(dim=1, keepdim=True)          # lo común a R, G y B: luz
color = W - gris_                            # lo que cambia entre canales
r = color.flatten(1).norm(dim=1) / (gris_.flatten(1).norm(dim=1) * 3 ** 0.5)
print("filtros más de color que de luz:", (r > 1).sum().item(), "de 32")
# <<<
orden = r.argsort()                          # primero los de luz
lum = gris_[:, 0]
guardar("ep06_filtros.json", {"filas": mosaico([lum[i] / lum.abs().max() for i in orden], 8),
                              "orden": orden.tolist()})
