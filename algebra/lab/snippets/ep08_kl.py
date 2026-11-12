import torch
import torch.nn.functional as F
torch.manual_seed(0)
# >>>
def KL(p, q):
    return (p * torch.log(p / q)).sum().item()

p = torch.tensor([0.50, 0.25, 0.15, 0.10])
q = torch.tensor([0.40, 0.30, 0.20, 0.10])
print("términos p log(p/q):", [round(x, 3) for x in (p * torch.log(p / q)).tolist()])
print(f"KL(p||q) = {KL(p, q):.4f}   KL(q||p) = {KL(q, p):.4f}")
print(f"KL(p||p) = {KL(p, p):.4f}")
print(f"F.kl_div: {F.kl_div(torch.log(q), p, reduction='sum').item():.4f}")

y = torch.multinomial(p, 100000, replacement=True)   # muestras de p
k = torch.log(p[y] / q[y])                            # log p/q por muestra
print("k de 6 muestras:", [round(x, 3) for x in k[:6].tolist()])
print(f"promedio de k: {k.mean().item():.4f}")
# <<<
