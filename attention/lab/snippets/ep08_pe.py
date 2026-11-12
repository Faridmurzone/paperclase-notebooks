import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _ordenar import Ordenador, entrenar, evaluar   # ordenar 10 dígitos, receta del ep. 7
# >>>
# fila (E): senos fijos, una tabla aprendida, o nada
for pe_tipo in ("seno", "aprendido", "ninguno"):
    torch.manual_seed(0)
    m = entrenar(Ordenador(pe_tipo=pe_tipo), pasos=400)
    r = evaluar(m)
    print(f"{pe_tipo:9}: secuencias bien {r['secuencia']:.1%}"
          f" · dígitos {r['digito']:.1%}")
# <<<
