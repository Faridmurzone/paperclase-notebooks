import json, os
import torch
from transformers import get_cosine_schedule_with_warmup
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
_w = torch.zeros(1, requires_grad=True); _o = torch.optim.AdamW([_w], lr=1e-3)   # mismo schedule que ep08_schedule
_s = SequentialLR(_o, [LinearLR(_o, start_factor=0.01, end_factor=1.0, total_iters=100),
                       CosineAnnealingLR(_o, T_max=900, eta_min=1e-5)], milestones=[100])
nuestro = []
for _ in range(1000):
    nuestro.append(_s.get_last_lr()[0]); _w.grad = torch.zeros(1); _o.step(); _s.step()

# >>>
w = torch.zeros(10, requires_grad=True)
opt = torch.optim.AdamW([w], lr=1e-3)
sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=100,
                                        num_training_steps=1000)
hf = []
for paso in range(1000):
    hf.append(sched.get_last_lr()[0])
    w.grad = torch.randn(10); opt.step(); sched.step()
print("paso   transformers   SequentialLR")
for t in [0, 1, 50, 100, 550, 999]:
    print(f"{t:4}   {hf[t]:.6f}       {nuestro[t]:.6f}")
# <<<
json.dump([round(x, 7) for x in hf], open(os.path.join(DATOS, "ep08_hf.json"), "w"))
