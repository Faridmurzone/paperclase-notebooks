import os, sys
import torch
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _cifar import cargar, normalizar
from _resnet import cargar_red
from _redes import evaluar
torch.set_grad_enabled(False)
Xtr, ytr = cargar("train", n=2000); Xte, yte = cargar("test", n=2000)
Xtr, Xte = normalizar(Xtr), normalizar(Xte)
# >>>
print("red          capas  error train  error test")
for nombre in ["plana8", "plana20", "plana56", "res8", "res20", "res56"]:
    red, info = cargar_red(f"{AQUI}/../datos/ep07_{nombre}.pt")
    e_tr = 1 - evaluar(red, Xtr, ytr)        # 2.000 imágenes de train
    e_te = 1 - evaluar(red, Xte, yte)        # 2.000 de test
    print(f"{nombre:12} {info['capas']:5}  {e_tr:10.1%}  {e_te:9.1%}")
# <<<
