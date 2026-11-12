import math
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# >>>
eta_max, eta_min, T = 1e-3, 1e-5, 900
def coseno(t):                                # la fórmula, a mano
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / T))

opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=eta_max)
sched = CosineAnnealingLR(opt, T_max=T, eta_min=eta_min)
dif = 0.0
for t in range(T + 1):
    dif = max(dif, abs(sched.get_last_lr()[0] - coseno(t)))
    opt.step(); sched.step()
print(f"t = 0: {coseno(0):.6f}  t = 450: {coseno(450):.6f}  t = 900: {coseno(900):.6f}")
print(f"máx |torch − fórmula| = {dif:.1e}")
# <<<
