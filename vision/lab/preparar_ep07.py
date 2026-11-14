"""Entrena UNA vez las redes del ep07 (plana vs. residual, 8 vs. 20 capas, todas con BatchNorm como en el paper)
y guarda pesos + historia por epoch en lab/datos/. Los snippets ep07_* solo cargan y evalúan.
Uso: .venv/bin/python preparar_ep07.py [epochs]"""
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
ruta_hist = os.path.join(AQUI, "datos", "ep07_hist.json")
hist = json.load(open(ruta_hist)) if os.path.exists(ruta_hist) else {}
for nombre, n, res in [("plana8", 1, False), ("plana20", 3, False), ("res8", 1, True), ("res20", 3, True), ("plana56", 9, False), ("res56", 9, True)]:
    if SOLO and nombre not in SOLO:
        continue
    torch.manual_seed(0)
    m = RedCifar(n=n, residual=res, bn=True)
    t0 = time.time()
    h = entrenar(m, Xtr, ytr, Xte, yte, EPOCHS, dev, lr=0.1, opt="sgd", wd=1e-4, nombre=nombre)
    hist[nombre] = dict(capas=6 * n + 2, residual=res, parametros=contar(m), epochs=EPOCHS, hist=h, segundos=round(time.time() - t0))
    guardar_red(m, os.path.join(AQUI, "datos", f"ep07_{nombre}.pt"), **{k: v for k, v in hist[nombre].items() if k != "hist"})
    guardar_json(hist, ruta_hist)
print("listo")
