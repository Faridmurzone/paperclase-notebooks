import json, os
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

# >>>
w = torch.zeros(10, requires_grad=True)
opt = torch.optim.AdamW([w], lr=1e-3)
warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=100)
coseno = CosineAnnealingLR(opt, T_max=900, eta_min=1e-5)
sched = SequentialLR(opt, [warmup, coseno], milestones=[100])
lrs = []
for paso in range(1000):
    lrs.append(sched.get_last_lr()[0])        # el lr que usa este paso
    w.grad = torch.randn(10); opt.step()
    sched.step()                              # después de opt.step()
for t in [0, 50, 100, 550, 999]:
    print(f"paso {t:3}: lr = {lrs[t]:.6f}")
print(f"suma: warmup {sum(lrs[:100]):.4f}, coseno {sum(lrs[100:]):.4f}")
# <<<
json.dump([round(x, 7) for x in lrs], open(os.path.join(DATOS, "ep08_lrs.json"), "w"))
