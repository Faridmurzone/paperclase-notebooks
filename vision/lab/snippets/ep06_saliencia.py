import os, sys
import torch
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _redes import cargar_modelo, CNNChica
from _cifar import cargar, normalizar
from _figs import norm, gris, mosaico, guardar
cnn, _ = cargar_modelo(f"{AQUI}/../datos/cnn_ep05.pt")
X, y = cargar("test", n=100)
torch.manual_seed(0)
al_azar = CNNChica().eval()                        # misma arquitectura, sin entrenar
i = 21                                             # un avión negro sobre fondo blanco


def corr(a, b):                                    # correlación de Pearson entre dos mapas
    a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


def bordes(img):                                   # |diferencia con el vecino|: un detector de bordes sin red
    g = gris(img.float() / 255)
    return torch.nn.functional.pad((g[1:, :-1] - g[:-1, :-1]).abs() + (g[:-1, 1:] - g[:-1, :-1]).abs(), (0, 1, 0, 1))
# >>>
def saliencia(red, x, clase):
    x = x.clone().requires_grad_(True)
    red(x)[0, clase].backward()               # ∂ logit / ∂ píxel
    return x.grad[0].abs().amax(dim=0)        # (32, 32): máx. sobre RGB

parecido = []
for j in range(100):                          # 100 imágenes de test
    x, k = normalizar(X[j:j + 1]), y[j].item()
    parecido.append(corr(saliencia(cnn, x, k), saliencia(al_azar, x, k)))
print("entrenada vs. al azar (100 imágenes):", round(sum(parecido) / 100, 2))

x, k = normalizar(X[i:i + 1]), y[i].item()  # el avión negro
avion = gris(X[i].float() / 255) < 0.5       # sus píxeles oscuros
print(f"el avión ocupa {avion.float().mean():.0%} de la imagen")
for nombre, red in [("entrenada", cnn), ("al azar", al_azar)]:
    s = saliencia(red, x, k)
    print(f"saliencia {nombre} sobre el avión: {s[avion].sum() / s.sum():.0%}")
# <<<
guardar("ep06_sal.json", {"paneles": [gris(X[i].float() / 255).tolist(),
                                      norm(saliencia(cnn, x, k)).tolist(), norm(saliencia(al_azar, x, k)).tolist()]})
