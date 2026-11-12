import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _ordenar import Ordenador, entrenar, evaluar   # ordenar 10 dígitos, receta del ep. 7
# >>>
# fila (D): P_drop, con secuencias nuevas en cada paso
for p in (0.0, 0.1, 0.3):
    torch.manual_seed(0)
    m = entrenar(Ordenador(p=p), pasos=400)
    r = evaluar(m)
    print(f"P_drop = {p}: secuencias bien {r['secuencia']:.1%}"
          f" · dígitos {r['digito']:.1%}")
# <<<
