import torch
import torch.nn.functional as F
# >>>
def CE(p, q):                              # entropía cruzada H(p, q)
    return -(p * torch.log(q)).sum().item()

p = torch.tensor([0.50, 0.25, 0.15, 0.10])   # los datos
q1 = torch.tensor([0.25, 0.25, 0.25, 0.25])  # un modelo uniforme
q2 = torch.tensor([0.40, 0.30, 0.20, 0.10])  # un modelo mejor
print(f"H(p)     = {CE(p, p):.3f}")
print(f"H(p, q1) = {CE(p, q1):.3f}")
print(f"H(p, q2) = {CE(p, q2):.3f}")

uno = torch.tensor([0.0, 1.0, 0.0, 0.0])     # one-hot: vino el token 1
logits = torch.log(q2)
print(f"one-hot: {CE(uno, q2):.3f} = -log 0,30")
print(f"F.cross_entropy: {F.cross_entropy(logits[None], torch.tensor([1])).item():.3f}")
# <<<
