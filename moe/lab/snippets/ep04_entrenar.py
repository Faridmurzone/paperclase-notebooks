import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from _juguete import MoE, Red, tarea
X, Y, _ = tarea(8192, semilla=0, grupos=1)        # una sola función, sin regiones
# >>>
for alfa in (0.0, 0.1):
    torch.manual_seed(0)
    red = Red(16, 32, 8, MoE(32, 32, E=8, k=2))
    opt = torch.optim.Adam(red.parameters(), lr=3e-3)
    for paso in range(400):
        i = torch.randint(0, len(X), (256,))
        mse = F.mse_loss(red(X[i]), Y[i])
        aux = red.capa.ultimo["l_aux"]           # E · Σ f_i · P_i
        loss = mse + alfa * aux
        opt.zero_grad(); loss.backward(); opt.step()
    carga = red.capa.ultimo["carga"].tolist()     # de 512 asignaciones
    print(f"α={alfa}: carga {carga}")
# <<<
