import os, sys
import torch
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _cifar import cargar, normalizar
from _resnet import cargar_red, contar
from _redes import evaluar
torch.set_grad_enabled(False)
Xte, yte = cargar("test", n=2000)
Xte = normalizar(Xte)
# >>>
modelos = [("ResNet-20", "ep07_res20"), ("ViT", "ep09_vit"),
           ("ResNet-20 + aug", "ep08_res20_aug"), ("ViT + aug", "ep09_vit_aug")]
for nombre, archivo in modelos:
    red, info = cargar_red(f"{AQUI}/../datos/{archivo}.pt")
    acc = evaluar(red, Xte, yte)             # 2.000 de test
    print(f"{nombre:16} {contar(red):>8,} parám.  test {acc:.1%}")
# <<<
