"""Entrena una vez los modelos del episodio 3 (MoE vs. densos) y guarda resultados en lab/datos/ep03_resultados.json.
Tarea sintética de _juguete.tarea (8 regiones, cada una con su función). Uso: .venv/bin/python preparar_ep03.py"""
import json
import os
import sys
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "snippets"))
import torch
import torch.nn.functional as F
from _juguete import MoE, Denso, Red, tarea, entrenar, parametros

X, Y, _ = tarea(8192, semilla=0)
Xt, Yt, _ = tarea(2048, semilla=1)
modelos = {"moe_8x32_top2": lambda: MoE(32, 32, E=8, k=2),
           "denso_64": lambda: Denso(32, 64),       # mismos parámetros activos que el MoE (2 expertos de 32)
           "denso_256": lambda: Denso(32, 256)}     # mismos parámetros totales que los 8 expertos
res = {}
for nombre, fab in modelos.items():
    tests, curvas = [], []
    for semilla in range(3):
        torch.manual_seed(semilla)
        red = Red(16, 32, 8, fab())
        h = entrenar(red, X, Y, pasos=1500, semilla=semilla)
        with torch.no_grad():
            tests.append(F.mse_loss(red(Xt), Yt).item())
        curvas.append(h["perdida"][::25])
        if semilla == 0:
            torch.save(red.state_dict(), os.path.join(AQUI, "datos", f"ep03_{nombre}.pt"))
    capa = fab()
    activos = parametros(capa) if not isinstance(capa, MoE) else parametros(capa.router) + capa.k * parametros(capa.expertos[0])
    res[nombre] = dict(test=tests, curva=curvas[0], params_capa=parametros(capa), activos_capa=activos)
    print(nombre, [round(t, 3) for t in tests], parametros(capa), activos)
json.dump(res, open(os.path.join(AQUI, "datos", "ep03_resultados.json"), "w"), indent=1)
