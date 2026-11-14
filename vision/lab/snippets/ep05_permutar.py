import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cifar import cargar, normalizar, permutacion, permutar, ruta_lab
from _redes import cargar_modelo, evaluar

Xt, yt = cargar("test")
# >>>
perm = permutacion(semilla=0)          # la misma mezcla del episodio 1
normal, mezclado = normalizar(Xt), normalizar(permutar(Xt, perm))
for nombre, a, b in [("MLP", "mlp_ep01", "mlp_perm_ep01"),
                     ("CNN", "cnn_ep05", "cnn_perm_ep05")]:
    r_a, _ = cargar_modelo(ruta_lab("datos", a + ".pt"))
    r_b, _ = cargar_modelo(ruta_lab("datos", b + ".pt"))
    print(f"{nombre}: píxeles en orden {evaluar(r_a, normal, yt):.1%}"
          f" · mezclados {evaluar(r_b, mezclado, yt):.1%}")
# <<<
