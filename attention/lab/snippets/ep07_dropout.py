import torch
import torch.nn as nn
torch.manual_seed(0)
# >>>
drop = nn.Dropout(p=0.1)                     # P_drop del paper
x = torch.ones(8)

drop.train()
print("train:", [round(v, 2) for v in drop(x).tolist()])
grande = drop(torch.ones(100_000))
print(f"apagados {(grande == 0).float().mean():.1%} · "
      f"promedio {grande.mean():.3f}")

drop.eval()
print("eval: ", drop(x).tolist())
# <<<
