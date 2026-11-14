import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _cifar import CLASES, cargar, normalizar, ruta_lab
from _redes import cargar_modelo, predecir

Xt, yt = cargar("test")
# >>>
red, _ = cargar_modelo(ruta_lab("datos", "cnn_ep05.pt"))
pred = predecir(red, normalizar(Xt))
M = torch.zeros(10, 10, dtype=torch.long)    # fila: verdad; col: predicción
M.index_put_((yt, pred), torch.ones_like(yt), accumulate=True)
acierto = M.diag() / M.sum(1)
for c in acierto.argsort(descending=True)[[0, 1, -2, -1]].tolist():
    print(f"{CLASES[c]:8s} {acierto[c]:.0%}")
gato, perro = CLASES.index("gato"), CLASES.index("perro")
print(f"gatos que dijo perro: {M[gato, perro]} de 1000")
print(f"perros que dijo gato: {M[perro, gato]} de 1000")
# <<<
with torch.no_grad():
    z = red(normalizar(Xt[:1]))[0]
json.dump({"M": M.tolist(), "acierto": [round(float(a), 3) for a in acierto], "logits0": [round(float(v), 2) for v in z],
           "probs0": [round(float(v), 3) for v in z.softmax(0)], "clase0": CLASES[int(yt[0])]}, open(ruta_lab("datos", "figs", "ep05_confusion.json"), "w"))
