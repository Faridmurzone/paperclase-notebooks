"""Entrena una vez los MoE del episodio 4 (colapso del router y pérdida de balanceo) y guarda en lab/datos/:
ep04_resultados.json (carga por paso, pérdida de test por semilla) y ep04_alfa{α}.pt (pesos, semilla 0; los usa el ep. 5).
Tarea: _juguete.tarea con grupos=1 (una sola función, sin regiones). Uso: .venv/bin/python preparar_ep04.py"""
import json
import os
import sys
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "snippets"))
import torch
import torch.nn.functional as F
from _juguete import MoE, Red, tarea, entrenar

X, Y, _ = tarea(8192, semilla=0, grupos=1)
Xt, Yt, _ = tarea(2048, semilla=1, grupos=1)
res = {}
for alfa in (0.0, 0.01, 0.1):
    r = dict(test=[], carga_final=[], carga_pasos=None)
    for semilla in range(3):
        torch.manual_seed(semilla)
        red = Red(16, 32, 8, MoE(32, 32, E=8, k=2))
        h = entrenar(red, X, Y, pasos=1500, alfa=alfa, semilla=semilla)
        with torch.no_grad():
            r["test"].append(F.mse_loss(red(Xt), Yt).item())
            r["carga_final"].append(red.capa.ultimo["carga"].tolist())
        if semilla == 0:
            r["carga_pasos"] = h["carga"][::10]
            torch.save(red.state_dict(), os.path.join(AQUI, "datos", f"ep04_alfa{alfa}.pt"))
    res[str(alfa)] = r
    print(alfa, [round(t, 4) for t in r["test"]], r["carga_final"])
json.dump(res, open(os.path.join(AQUI, "datos", "ep04_resultados.json"), "w"))
