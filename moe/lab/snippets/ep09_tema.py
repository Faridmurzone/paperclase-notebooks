import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _granite import cargar, TEXTOS, capturar_ruteo
import torch
tok, model = cargar()
elegidos, hs = capturar_ruteo(model)
def rutas(textos):
    elegidos.clear()
    for t in textos:
        model(**tok(t, return_tensors="pt"))
    return {c: torch.cat(v) for c, v in elegidos.items()}
# >>>
def reparto(idx):                         # distribución de las asignaciones
    d = torch.bincount(idx.reshape(-1), minlength=32).float()
    return d / d.sum()
def tv(a, b):                             # total variation: 0 igual, 1 disjuntas
    return 0.5 * (reparto(a) - reparto(b)).abs().sum().item()

coc = TEXTOS["cocina"]
A, B = rutas(coc[:3]), rutas(coc[3:])      # dos mitades del mismo tema
C, D = rutas(coc), rutas(TEXTOS["astronomia"])
P, N = rutas(TEXTOS["codigo"]), rutas(TEXTOS["numeros"])
print("capa mitad astro código núm.")
for c in [0, 12, 23]:
    print(f"{c:4d} {tv(A[c], B[c]):5.2f} {tv(C[c], D[c]):5.2f}"
          f" {tv(C[c], P[c]):6.2f} {tv(C[c], N[c]):5.2f}")
# <<<
