import json, os, sys
import torch
AQUI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, AQUI)
from ep10_lion import Lion
DATOS = os.path.join(AQUI, "..", "datos")

# >>>
opciones = [("SGD", torch.optim.SGD, dict(lr=0.1)),
            ("SGD + momentum", torch.optim.SGD, dict(lr=0.1, momentum=0.9)),
            ("AdamW", torch.optim.AdamW, {}),
            ("Lion", Lion, {}),
            ("Muon", torch.optim.Muon, {}),
            ("Adafactor", torch.optim.Adafactor, {})]
res = {}
for nombre, Clase, kw in opciones:
    W = torch.nn.Parameter(torch.randn(1000, 1000))   # un millón de pesos
    opt = Clase([W], **kw)
    W.grad = torch.randn(1000, 1000); opt.step()      # el estado nace aquí
    n = sum(v.numel() for v in opt.state[W].values()
            if torch.is_tensor(v) and v.dim() > 0)    # tensores guardados
    res[nombre] = n / W.numel()
    print(f"{nombre:15} {n:9,} números  →  {n / W.numel():.3f} por peso")
# <<<
json.dump(res, open(os.path.join(DATOS, "ep10_memoria.json"), "w"))
