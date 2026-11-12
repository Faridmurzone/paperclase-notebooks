import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import *
torch.manual_seed(0)
import json
from torch.func import functional_call
# >>>
red = cargar_base()
nombres = [n for n, _ in red.named_parameters()]
formas = [p.shape for p in red.parameters()]
tamanos = [p.numel() for p in red.parameters()]
theta0 = torch.cat([p.detach().flatten() for p in red.parameters()])
D = theta0.numel()                                     # todos los parámetros

def red_en_subespacio(d):
    torch.manual_seed(d)
    P = torch.randn(D, d) / D**0.5                     # proyección al azar, fija
    z = torch.zeros(d, requires_grad=True)             # lo único que se entrena
    def f(X):
        theta = theta0 + P @ z                         # θ = θ0 + P·z
        trozos = [t.view(s) for t, s in zip(theta.split(tamanos), formas)]
        return functional_call(red, dict(zip(nombres, trozos)), (X,))
    return f, z

print("D =", f"{D:,}")
for d in [10, 50, 200, 1000]:
    f, z = red_en_subespacio(d)
    entrenar(f, espejo(Xtr), ytr, pasos=300, lr=0.05, params=[z])
    print(f"d = {d:4}: {precision(f, espejo(Xte), yte):.1%}")
# <<<
res = {}
for d in [10, 25, 50, 100, 200, 500, 1000]:
    f, z = red_en_subespacio(d)
    entrenar(f, espejo(Xtr), ytr, pasos=300, lr=0.05, params=[z])
    res[d] = precision(f, espejo(Xte), yte)
json.dump(res, open(os.path.join(DATOS, "ep02_intrinseca.json"), "w"))
