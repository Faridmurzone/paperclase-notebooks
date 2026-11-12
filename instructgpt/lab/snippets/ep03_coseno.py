import torch
# >>>
w = torch.nn.Parameter(torch.zeros(1))
lr, pasos = 9.65e-6, 400                       # el lr del paper (1,3B y 6B)
opt = torch.optim.Adam([w], lr=lr, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=pasos, eta_min=0.1 * lr)        # baja hasta el 10 %, sin warmup

lrs = []
for paso in range(pasos + 1):
    lrs.append(sched.get_last_lr()[0])
    opt.step(); sched.step()
for t in [0, 100, 200, 300, 400]:
    print(f"paso {t:3d}  lr = {lrs[t]:.3e}  ({lrs[t] / lr:.0%})")
# <<<
import json, os
json.dump([[t, round(lrs[t] / lr, 4)] for t in range(0, pasos + 1, 10)],
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep03_coseno.json"), "w"))
