import torch
torch.manual_seed(0)

# >>>
modelo = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8),
                             torch.nn.Linear(8, 2))
con = [p for p in modelo.parameters() if p.dim() >= 2]    # matrices
sin = [p for p in modelo.parameters() if p.dim() < 2]     # sesgos y LayerNorm
opt = torch.optim.AdamW([{"params": con, "weight_decay": 0.1},
                         {"params": sin, "weight_decay": 0.0}], lr=0.01)
antes = [p.detach().norm().item() for p in modelo.parameters()]
for _ in range(100):
    for p in modelo.parameters(): p.grad = torch.zeros_like(p)
    opt.step()                                            # solo actúa el decay
for (n, p), a in zip(modelo.named_parameters(), antes):
    queda = f"{p.norm().item() / a:.4f}" if a else "era 0"
    print(f"{n:9} {tuple(p.shape)!s:7} queda {queda}")
# <<<
