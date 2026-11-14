import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cifar import cargar, normalizar, ruta_lab
from _redes import cargar_modelo, contar_parametros, evaluar

Xt, yt = cargar("test")
# >>>
Xt = normalizar(Xt)
for archivo in ("mlp_ep01", "lenet_ep05", "cnn_ep05"):
    red, info = cargar_modelo(ruta_lab("datos", archivo + ".pt"))
    print(f"{archivo:10s} {contar_parametros(red):>9,} parámetros"
          f"  test {evaluar(red, Xt, yt):.1%}"
          f"  train {info['precision_train']:.1%}")
# <<<
hist = {a: cargar_modelo(ruta_lab("datos", a + ".pt"))[1]["historia"] for a in ("mlp_ep01", "lenet_ep05", "cnn_ep05")}
json.dump(hist, open(ruta_lab("datos", "figs", "ep05_historia.json"), "w"))
