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
for nombre, ruta in [("sin augmentation", "ep07_res20.pt"),
                     ("con augmentation", "ep08_res20_aug.pt")]:
    red, _ = cargar_red(f"{AQUI}/../datos/{ruta}")   # ResNet-20, 30 epochs
    tr = evaluar(red, Xtr, ytr)              # 2.000 imágenes que vio
    te = evaluar(red, Xte, yte)              # 2.000 que nunca vio
    print(f"{nombre}: train {tr:.1%}  test {te:.1%}  brecha {tr - te:.1%}")
# <<<
