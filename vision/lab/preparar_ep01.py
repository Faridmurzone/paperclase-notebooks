"""Entrena UNA vez los dos MLP del ep01 y guarda pesos (los snippets solo cargan y evalúan).

- mlp_ep01.pt:       MLP sobre los píxeles de CIFAR-10 tal cual.
- mlp_perm_ep01.pt:  el mismo MLP, entrenado con los píxeles permutados (permutacion(semilla=0)).
Mismo seed, mismos hiperparámetros. Guarda también la precisión de test por epoch (para la curva).
Uso: .venv/bin/python preparar_ep01.py   (MPS si está disponible). No volver a correr: cambia los números narrados.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "snippets"))
import torch
from _cifar import cargar, normalizar, permutacion, permutar, ruta_lab, MEDIA, DESVIO
from _redes import MLP, entrenar, evaluar, guardar_modelo

EPOCHS, LR, BATCH = 10, 1e-3, 128
dev = "mps" if torch.backends.mps.is_available() else "cpu"
Xtr, ytr = cargar("train")
Xte, yte = cargar("test")
perm = permutacion(0)

for nombre, transf in (("mlp_ep01", lambda X: X), ("mlp_perm_ep01", lambda X: permutar(X, perm))):
    t0 = time.time()
    xtr, xte = normalizar(transf(Xtr)), normalizar(transf(Xte))
    torch.manual_seed(0)
    m = MLP()
    hist = []
    def medir(ep, perdida):
        acc = evaluar(m, xte, yte)
        m.train()
        hist.append({"epoch": ep, "perdida": round(perdida, 4), "test": round(acc, 4)})
        print(f"  {nombre} epoch {ep}: test {acc:.4f}")
    entrenar(m, xtr, ytr, EPOCHS, lr=LR, batch=BATCH, device=dev, semilla=0, log=lambda s: None, al_final_de_epoch=medir)
    m.cpu().eval()
    acc_te, acc_tr = evaluar(m, xte, yte), evaluar(m, xtr, ytr)
    guardar_modelo(m, ruta_lab("datos", nombre + ".pt"), precision_test=acc_te, precision_train=acc_tr,
                   epochs=EPOCHS, lr=LR, batch=BATCH, historia=hist, media=MEDIA, desvio=DESVIO,
                   permutado=(nombre == "mlp_perm_ep01"))
    print(f"{nombre}: test {acc_te:.4f} train {acc_tr:.4f} ({time.time() - t0:.0f}s)")
