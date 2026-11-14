"""Entrena UNA vez el ViT chico del ep09 (sin y con augmentation) y guarda pesos + historia en lab/datos/.
Se compara con la ResNet-20 del ep07 (sin augmentation) y del ep08 (con augmentation): mismos datos y mismas epochs.
Uso: .venv/bin/python preparar_ep09.py [epochs] [nombres,separados]"""
import os, sys, json, time
import torch
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "snippets"))
from _cifar import cargar, normalizar
from _resnet import ViTChica, guardar_red, contar
from _entreno import entrenar, guardar_json

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SOLO = sys.argv[2].split(",") if len(sys.argv) > 2 else None
dev = "mps" if torch.backends.mps.is_available() else "cpu"
Xtr, ytr = cargar("train"); Xte, yte = cargar("test")
Xtr, Xte = normalizar(Xtr), normalizar(Xte)
ruta_hist = os.path.join(AQUI, "datos", "ep09_hist.json")
hist = json.load(open(ruta_hist)) if os.path.exists(ruta_hist) else {}
for nombre, aug in [("vit", False), ("vit_aug", True)]:
    if SOLO and nombre not in SOLO:
        continue
    torch.manual_seed(0)
    m = ViTChica()
    t0 = time.time()
    h = entrenar(m, Xtr, ytr, Xte, yte, EPOCHS, dev, lr=1e-3, opt="adam", wd=0.05, aug=aug, nombre=nombre)
    hist[nombre] = dict(m.kwargs, aug=aug, parametros=contar(m), epochs=EPOCHS, hist=h, segundos=round(time.time() - t0))
    guardar_red(m, os.path.join(AQUI, "datos", f"ep09_{nombre}.pt"), **{k: v for k, v in hist[nombre].items() if k != "hist"})
    guardar_json(hist, ruta_hist)
print("listo")
