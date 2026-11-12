import torch
torch.manual_seed(0)
# >>>
w = torch.nn.Parameter(torch.randn(1000))
(100 * w).pow(2).sum().backward()          # gradientes enormes
antes = torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
print(f"norma antes: {antes.item():,.0f}   después: {w.grad.norm().item():.3f}")

a = torch.nn.Parameter(torch.ones(3))      # AdamW: decay desacoplado
opt_w = torch.optim.AdamW([a], lr=0.1, weight_decay=0.1)
b = torch.nn.Parameter(torch.ones(3))      # Adam con weight_decay (L2)
opt_l2 = torch.optim.Adam([b], lr=0.1, weight_decay=0.1)

a.grad, b.grad = torch.zeros(3), torch.zeros(3)   # gradiente cero
opt_w.step(); opt_l2.step()
print("AdamW:", [round(x, 3) for x in a.tolist()])
print("Adam + L2:", [round(x, 3) for x in b.tolist()])
# <<<
