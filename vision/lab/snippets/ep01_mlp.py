import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cifar import cargar, normalizar, ruta_lab
from _redes import cargar_modelo, contar_parametros, evaluar

Xt, yt = cargar("test")
# >>>
mlp, info = cargar_modelo(ruta_lab("datos", "mlp_ep01.pt"))   # ya entrenado
print([n for n, _ in mlp.named_children()])   # capa1, capa2, salida
print(f"parámetros: {contar_parametros(mlp):,}")
print(f"capa1 por neurona: {mlp.capa1.weight.shape[1]:,} pesos")
print(f"precisión en test: {evaluar(mlp, normalizar(Xt), yt):.1%}"
      f" (train {info['precision_train']:.1%}; al azar 10%)")
# <<<
import json
import torch
from _cifar import CLASES
with torch.no_grad():
    z = mlp(normalizar(Xt[:1]))[0]
json.dump({"clase": CLASES[int(yt[0])], "logits": [round(float(v), 2) for v in z], "probs": [round(float(v), 3) for v in z.softmax(0)]},
          open(ruta_lab("datos", "figs", "ep01_mlp.json"), "w"))
