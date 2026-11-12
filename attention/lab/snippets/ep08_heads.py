import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _ordenar import Ordenador, entrenar, evaluar   # ordenar 10 dígitos, receta del ep. 7
# >>>
# fila (A): d_model fijo en 32, cambia cuántas heads lo parten
for h in (1, 4, 8):
    torch.manual_seed(0)
    m = entrenar(Ordenador(d=32, h=h), pasos=400)
    r = evaluar(m)
    print(f"h = {h} (d_k = {32 // h:2}): secuencias bien {r['secuencia']:.1%}"
          f" · dígitos {r['digito']:.1%}")
# <<<
