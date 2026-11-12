import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import MoE, Red, tarea, entrenar
import torch
import torch.nn.functional as F
X, Y, _ = tarea(8000, semilla=0)
Xt, Yt, _ = tarea(2000, semilla=1)
# >>>
for k, renorm in [(2, True), (1, False)]:      # top-2 clásico y Switch
    torch.manual_seed(0)
    moe = MoE(d=32, h=64, E=8, k=k, renormalizar=renorm)
    red = Red(16, 32, 8, moe)
    entrenar(red, X, Y, pasos=800, alfa=0.01)  # con L_aux de Switch
    with torch.no_grad():
        mse = F.mse_loss(red(Xt), Yt).item()
        red(Xt[:256])                           # un batch de 256 tokens
    n = moe.ultimo['carga'].sum().item()
    print(f"top-{k}: MSE {mse:.3f}, {n} asignaciones")
# <<<
