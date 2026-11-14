import os, sys
import torch
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _cifar import cargar, normalizar
from _resnet import cargar_red
torch.set_grad_enabled(False)
X, y = cargar("test", n=2000)
X = normalizar(X)
red, _ = cargar_red(f"{AQUI}/../datos/ep07_res20.pt")   # la ResNet-20 del ep07
# >>>
def precision(red, modo_train, batch):
    red.train(modo_train)                    # cambia qué estadística usa la BN
    pred = torch.cat([red(X[i:i + batch]).argmax(1)
                      for i in range(0, len(X), batch)])
    return (pred == y).float().mean().item()

print("eval,  batch 1:   ", f"{precision(red, False, 1):.1%}")
print("train, batch 500: ", f"{precision(red, True, 500):.1%}")
print("train, batch 4:   ", f"{precision(red, True, 4):.1%}")
print("train, batch 1:   ", f"{precision(red, True, 1):.1%}")
# <<<
