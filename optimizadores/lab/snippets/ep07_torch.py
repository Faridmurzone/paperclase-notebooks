import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ep07_mano import AdamWMano
torch.manual_seed(0)
X, Y = torch.randn(64, 5), torch.randn(64, 3)

# >>>
red_a = torch.nn.Linear(5, 3)
red_b = torch.nn.Linear(5, 3)
red_b.load_state_dict(red_a.state_dict())                 # mismos pesos
cfg = dict(lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
mano = AdamWMano(red_a.parameters(), **cfg)
ref = torch.optim.AdamW(red_b.parameters(), **cfg)
for paso in range(200):
    for red, opt in [(red_a, mano), (red_b, ref)]:
        for p in red.parameters(): p.grad = None
        ((red(X) - Y) ** 2).mean().backward()
        opt.step()
dif = max((a - b).abs().max().item()
          for a, b in zip(red_a.parameters(), red_b.parameters()))
print("pérdida a mano:", round(((red_a(X) - Y) ** 2).mean().item(), 5))
print("pérdida torch :", round(((red_b(X) - Y) ** 2).mean().item(), 5))
print(f"máx |diferencia| en los pesos: {dif:.1e}")
print("defaults de torch:", {k: torch.optim.AdamW([torch.zeros(1)]).defaults[k]
                              for k in ("lr", "eps", "weight_decay")})
# <<<
