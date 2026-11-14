import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cifar import cargar

X, _ = cargar("test", n=1)
# >>>
x = X[0]                              # una imagen: (3, 32, 32)
v = x.flatten()                       # un vector de 3 · 32 · 32
print("forma:", tuple(x.shape), "→", tuple(v.shape))
c, i, j = 1, 5, 7                     # canal verde, fila 5, columna 7
k = c * 32 * 32 + i * 32 + j          # dónde cae en el vector
print(f"x[{c}, {i}, {j}] = {x[c, i, j]}   v[{k}] = {v[k]}")
# <<<
