import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import MoE, Red, tarea
import torch
import torch.nn.functional as F
X, Y, _ = tarea(8000, semilla=0)
Xt, Yt, _ = tarea(2000, semilla=1)
def maxvio(c):
    c = c.float(); return ((c.max() - c.mean()) / c.mean()).item()
def entrenar(alfa=0.0, u=0.0, pasos=800):
    torch.manual_seed(0)
    moe = MoE(32, 64, E=8, k=2); red = Red(16, 32, 8, moe)
    opt = torch.optim.Adam(red.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0); mv = []
    for _ in range(pasos):
        i = torch.randint(0, len(X), (256,), generator=g)
        loss = F.mse_loss(red(X[i]), Y[i]) + alfa * moe.ultimo["l_aux"]
        opt.zero_grad(); loss.backward(); opt.step()
        c = moe.ultimo["carga"].float(); mv.append(maxvio(c))
        moe.sesgo += u * torch.sign(c.mean() - c)    # 0 si u = 0
    with torch.no_grad():
        mse = F.mse_loss(red(Xt), Yt).item()
    return mse, maxvio(moe.ultimo["carga"]), mv

# >>>
for u in [0.001, 0.01, 0.1]:              # velocidad del termostato
    mse, mv_test, mv = entrenar(u=u)
    primeros = sum(mv[:100]) / 100            # MaxVio medio, pasos 1–100
    ultimos = sum(mv[-100:]) / 100            # MaxVio medio, pasos 701–800
    print(f"u={u:<5} MSE {mse:.3f} · MaxVio {primeros:.2f} → {ultimos:.2f}")
# <<<
