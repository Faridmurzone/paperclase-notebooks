import torch
d, n = 512, 100
pos = torch.arange(n).unsqueeze(1)
omega = 1 / 10000 ** (torch.arange(0, d, 2) / d)
PE = torch.zeros(n, d)
PE[:, 0::2], PE[:, 1::2] = torch.sin(pos * omega), torch.cos(pos * omega)
# >>>
print("norma de cada PE(pos):", PE.norm(dim=1)[:3].numpy().round(2), "…")
for k in [1, 5, 20]:
    puntos = [PE[p] @ PE[p + k] for p in (0, 10, 50)]
    print(f"k={k:2}: PE(p)·PE(p+k) con p=0, 10, 50 →",
          [round(x.item(), 2) for x in puntos])
# <<<
