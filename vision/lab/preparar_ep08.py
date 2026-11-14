"""Entrena UNA vez las redes del ep08 y guarda pesos + historia por epoch en lab/datos/.
  BatchNorm: la red plana de 8 capas (la del ep07) sin BN, con learning rate 0,02 y 0,1, y con BN a 0,02
  (con BN a 0,1 es ep07_plana8). Augmentation: la ResNet-20 del ep07 (sin augmentation) contra la misma con flip + recorte.
Uso: .venv/bin/python preparar_ep08.py [epochs] [nombres,separados]"""
import os, sys, json, time
import torch
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "snippets"))
from _cifar import cargar, normalizar
from _resnet import RedCifar, guardar_red, contar
from _entreno import entrenar, guardar_json

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SOLO = sys.argv[2].split(",") if len(sys.argv) > 2 else None
dev = "mps" if torch.backends.mps.is_available() else "cpu"
Xtr, ytr = cargar("train"); Xte, yte = cargar("test")
Xtr, Xte = normalizar(Xtr), normalizar(Xte)
ruta_hist = os.path.join(AQUI, "datos", "ep08_hist.json")
hist = json.load(open(ruta_hist)) if os.path.exists(ruta_hist) else {}
CONF = [("sinbn_lr002", dict(n=1, residual=False, bn=False), 0.02, False),
        ("sinbn_lr01", dict(n=1, residual=False, bn=False), 0.1, False),
        ("conbn_lr002", dict(n=1, residual=False, bn=True), 0.02, False),
        ("res20_aug", dict(n=3, residual=True, bn=True), 0.1, True)]
for nombre, kw, lr, aug in CONF:
    if SOLO and nombre not in SOLO:
        continue
    torch.manual_seed(0)
    m = RedCifar(**kw)
    t0 = time.time()
    h = entrenar(m, Xtr, ytr, Xte, yte, EPOCHS, dev, lr=lr, opt="sgd", wd=1e-4, aug=aug, nombre=nombre)
    hist[nombre] = dict(kw, lr=lr, aug=aug, parametros=contar(m), epochs=EPOCHS, hist=h, segundos=round(time.time() - t0))
    guardar_red(m, os.path.join(AQUI, "datos", f"ep08_{nombre}.pt"), **{k: v for k, v in hist[nombre].items() if k != "hist"})
    guardar_json(hist, ruta_hist)
print("listo")
