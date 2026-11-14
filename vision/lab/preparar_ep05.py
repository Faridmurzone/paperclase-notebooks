"""Entrena UNA vez las redes del ep05 y guarda pesos (los snippets solo cargan y evalúan).

- lenet_ep05.pt:     LeNet (de _redes), píxeles tal cual.
- cnn_ep05.pt:       CNNChica (de _redes): la que usa también el ep06. 3→32→64→64 canales, fc 1024→10.
- cnn_perm_ep05.pt:  CNNChica entrenada con los píxeles permutados (la misma permutación que el MLP del ep01).
Adam lr 1e-3, batch 128, EPOCHS epochs, sin augmentation ni BatchNorm, seed 0. Guarda la historia de test por epoch.
Uso: .venv/bin/python preparar_ep05.py   No volver a correr: cambia los números narrados en ep05 y ep06.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "snippets"))
import torch
from _cifar import cargar, normalizar, permutacion, permutar, ruta_lab, MEDIA, DESVIO
from _redes import CNNChica, LeNet, entrenar, evaluar, guardar_modelo

EPOCHS, LR, BATCH = 12, 1e-3, 128
dev = "mps" if torch.backends.mps.is_available() else "cpu"
Xtr, ytr = cargar("train")
Xte, yte = cargar("test")
perm = permutacion(0)

for nombre, clase, transf in (("lenet_ep05", LeNet, lambda X: X), ("cnn_ep05", CNNChica, lambda X: X),
                              ("cnn_perm_ep05", CNNChica, lambda X: permutar(X, perm))):
    t0 = time.time()
    xtr, xte = normalizar(transf(Xtr)), normalizar(transf(Xte))
    torch.manual_seed(0)
    m = clase()
    hist = []
    def medir(ep, perdida):
        acc = evaluar(m, xte, yte)
        m.train()
        hist.append({"epoch": ep, "perdida": round(perdida, 4), "test": round(acc, 4)})
        print(f"  {nombre} epoch {ep}: pérdida {perdida:.4f} test {acc:.4f}", flush=True)
    entrenar(m, xtr, ytr, EPOCHS, lr=LR, batch=BATCH, device=dev, semilla=0, log=lambda s: None, al_final_de_epoch=medir)
    m.cpu().eval()
    acc_te, acc_tr = evaluar(m, xte, yte), evaluar(m, xtr, ytr)
    guardar_modelo(m, ruta_lab("datos", nombre + ".pt"), precision_test=acc_te, precision_train=acc_tr,
                   epochs=EPOCHS, lr=LR, batch=BATCH, historia=hist, media=MEDIA, desvio=DESVIO,
                   permutado=(nombre == "cnn_perm_ep05"))
    print(f"{nombre}: test {acc_te:.4f} train {acc_tr:.4f} ({time.time() - t0:.0f}s)", flush=True)
