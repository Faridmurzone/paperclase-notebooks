import torch
torch.manual_seed(0)
def r3(t): return [round(x, 3) for x in t.tolist()]
# >>>
pi_ref = torch.tensor([0.40, 0.30, 0.15, 0.10, 0.05])  # Alineación ep. 5
r = torch.tensor([0.0, 1.0, 2.0, 0.5, 3.0])
beta = 1.0

pi = pi_ref * torch.exp(r / beta)          # π_ref · e^(r/β)
Z = pi.sum()
pi = pi / Z                                # normalizar: Σ π = 1
print("π* =", r3(pi))

# derivada del Lagrangiano respecto de cada π_y (sin el -λ)
d = r - beta * (torch.log(pi / pi_ref) + 1)
print("r - β(log π*/π_ref + 1) =", r3(d))
print("β(log Z - 1) =", round((beta * (Z.log() - 1)).item(), 3))
print("softmax(log π_ref + r/β) =", r3(torch.softmax(pi_ref.log() + r / beta, 0)))
# <<<
