import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from _ordenar import Ordenador, datos, evaluar, forma   # tarea: ordenar 10 dígitos
# >>>
def receta(modelo, eps_ls, pasos=800, pico=2e-3, warmup=100):
    opt = torch.optim.Adam(modelo.parameters(), lr=pico,
                           betas=(0.9, 0.98), eps=1e-9)   # sección 5.3
    sch = LambdaLR(opt, lambda s: forma(s + 1, warmup))  # ecuación (3)
    modelo.train()                                       # dropout prendido
    for _ in range(pasos):
        x, y, y_in = datos(64)
        loss = F.cross_entropy(modelo(x, y_in).flatten(0, 1), y.flatten(),
                               label_smoothing=eps_ls)    # sección 5.4
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return modelo

for eps_ls in (0.1, 0.0):
    torch.manual_seed(0)
    r = evaluar(receta(Ordenador(p=0.1), eps_ls))      # P_drop = 0,1
    print(f"ε_ls {eps_ls}: secuencias {r['secuencia']:.1%} · confianza "
          f"{r['confianza']:.2f} · perplejidad {r['perplejidad']:.3f}")
# <<<
