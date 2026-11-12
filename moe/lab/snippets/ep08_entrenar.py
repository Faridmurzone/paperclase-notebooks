import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import MoE, Red, tarea
import torch
import torch.nn.functional as F
X, Y, _ = tarea(8000, semilla=0)
Xt, Yt, _ = tarea(2000, semilla=1)
def maxvio(c):
    c = c.float(); return ((c.max() - c.mean()) / c.mean()).item()
historia = {}
# >>>
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

for nombre, kw in {"nada": {}, "L_aux": dict(alfa=0.01),
                   "sesgo": dict(u=0.01)}.items():
    mse, mv_test, mv = entrenar(**kw); historia[nombre] = mv
    print(f"{nombre:6} MSE {mse:.3f} · MaxVio test {mv_test:.2f}")
# <<<
promedios = {k: [round(sum(v[j:j + 50]) / 50, 3) for j in range(0, len(v), 50)] for k, v in historia.items()}
json.dump(promedios, open(os.path.join(os.path.dirname(__file__), "..", "datos", "ep08_maxvio.json"), "w"), indent=1)
