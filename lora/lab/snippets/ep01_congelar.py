import torch
torch.manual_seed(0)
# >>>
base = torch.nn.Linear(2304, 2304)          # hace de W: se congela
extra = torch.nn.Linear(2304, 8)            # hace de la parte entrenable
base.requires_grad_(False)                  # sin gradiente

opt = torch.optim.Adam(extra.parameters(), lr=1e-3)
x = torch.randn(4, 2304)
(base(x).sum() + extra(x).sum()).backward()
opt.step()

for nombre, capa in [("base", base), ("extra", extra)]:
    w = capa.weight
    g = None if w.grad is None else tuple(w.grad.shape)
    n = sum(p.numel() for p in capa.parameters())
    print(f"{nombre:5} grad={g}  Adam={w in opt.state}  n={n:,}")
# <<<
