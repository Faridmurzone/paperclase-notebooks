import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import MoE, Red, tarea
import torch
import torch.nn.functional as F
X, Y, _ = tarea(8000, semilla=0)
Xt, Yt, _ = tarea(2000, semilla=1)
# >>>
def entrenar_z(c_z, pasos=800):
    torch.manual_seed(0)
    moe = MoE(d=32, h=64, E=8, k=2); red = Red(16, 32, 8, moe)
    opt = torch.optim.Adam(red.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(pasos):
        i = torch.randint(0, len(X), (256,), generator=g)
        mse = F.mse_loss(red(X[i]), Y[i])
        z = (torch.logsumexp(moe.ultimo["logits"], -1) ** 2).mean()
        loss = mse + 0.01 * moe.ultimo["l_aux"] + c_z * z
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mse = F.mse_loss(red(Xt), Yt).item(); lg = moe.ultimo["logits"]
    print(f"c_z={c_z}: MSE {mse:.3f} z {(lg.logsumexp(-1) ** 2).mean():.2f}"
          f" máx {lg.max(-1).values.mean():.2f}")
for c_z in [0.0, 0.001]:
    entrenar_z(c_z)
# <<<
